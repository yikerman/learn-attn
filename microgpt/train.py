"""MicroGPT training script (single-GPU and multi-GPU via DDP).

Usage:
    # Single GPU
    uv run python -m microgpt.train

    # Multi-GPU (8 GPUs on one node)
    uv run torchrun --nproc_per_node=8 -m microgpt.train

    # Multi-node (see vast.ai guide in Chapter 6)
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$RANK \
        --master_addr=$MASTER --master_port=29500 -m microgpt.train
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .config import GPTConfig
from .model import MicroGPT
from .tokenizer import BPETokenizer
from .dataloader import create_dataloader
from .distributed import ddp_init, ddp_cleanup, is_master, print0


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

# Model
MODEL_CONFIG = GPTConfig()

# Optimization
MAX_STEPS = 20_000             # total training steps
WARMUP_STEPS = 200             # linear LR warmup
MAX_LR = 6e-4                  # peak learning rate
MIN_LR = 6e-5                  # final learning rate (10% of peak)
WEIGHT_DECAY = 0.1             # AdamW weight decay
BETA1, BETA2 = 0.9, 0.95      # AdamW betas
GRAD_CLIP = 1.0                # max gradient norm

# Batch sizing
DEVICE_BATCH_SIZE = 16         # micro-batch per GPU per accumulation step
TOTAL_BATCH_TOKENS = 524_288   # ~0.5M tokens per optimizer step
# grad_accum_steps is computed from these at runtime

# Evaluation
EVAL_INTERVAL = 500            # steps between evaluations
EVAL_STEPS = 20                # batches per evaluation

# Checkpointing
CHECKPOINT_DIR = "microgpt_checkpoints"
SAVE_INTERVAL = 2000           # steps between checkpoint saves

# Tokenizer
TOKENIZER_DIR = "tokenizer"


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int) -> float:
    """Cosine learning rate schedule with linear warmup.

    1. Linear warmup from 0 to MAX_LR over WARMUP_STEPS.
    2. Cosine decay from MAX_LR to MIN_LR over remaining steps.
    """
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return MIN_LR
    # Cosine decay
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Optimizer setup
# ---------------------------------------------------------------------------

def configure_optimizer(model: nn.Module) -> torch.optim.AdamW:
    """Configure AdamW with separate parameter groups for decay/no-decay.

    Weight decay is applied to all 2D+ parameters (weight matrices).
    Biases, norms, and embedding weights are not decayed.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        groups,
        lr=MAX_LR,
        betas=(BETA1, BETA2),
        fused=True,  # faster fused CUDA kernel
    )
    return optimizer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, val_loader, device: str, steps: int) -> float:
    """Compute average validation loss over a fixed number of batches."""
    model.eval()
    total_loss = 0.0
    for i, (inputs, targets) in enumerate(val_loader):
        if i >= steps:
            break
        _, loss = model(inputs, targets)
        total_loss += loss.item()
    model.train()
    return total_loss / min(steps, i + 1)


# ---------------------------------------------------------------------------
# MFU estimation
# ---------------------------------------------------------------------------

def get_peak_flops(device: str = "cuda") -> float:
    """Look up bf16 peak FLOPS for the current GPU.

    Returns float('inf') for unknown GPUs so MFU shows as 0%.
    """
    if not torch.cuda.is_available():
        return float("inf")
    name = torch.cuda.get_device_name(device).lower()
    # (substrings to match, peak bf16 TFLOPS)
    table = [
        (["h100"], 989e12),
        (["h200"], 989e12),
        (["a100"], 312e12),
        (["l40s"], 362e12),
        (["4090"], 165e12),
        (["3090"], 71e12),
        (["3080"], 47e12),
    ]
    for patterns, flops in table:
        if all(p in name for p in patterns):
            return flops
    return float("inf")


def estimate_mfu(
    model: MicroGPT,
    tokens_per_second: float,
    device: str = "cuda",
) -> float:
    """Estimate Model FLOPs Utilization (MFU).

    MFU = (actual FLOPs/sec) / (theoretical peak FLOPs/sec).
    Useful for gauging how efficiently the GPUs are being used.
    """
    # Approximate FLOPs per token (forward + backward ≈ 6 * params)
    N = model.count_parameters()
    flops_per_token = 6 * N
    actual_flops = tokens_per_second * flops_per_token
    peak_flops = get_peak_flops(device)
    return actual_flops / peak_flops


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train() -> None:
    # ---- DDP setup --------------------------------------------------------
    rank, local_rank, world_size = ddp_init()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    is_ddp = world_size > 1

    # Use TF32 for float32 matmuls (faster on Ampere+ GPUs, no accuracy loss
    # for our use case since activations are bf16 anyway)
    torch.set_float32_matmul_precision("high")

    # ---- Gradient accumulation calculation --------------------------------
    tokens_per_step = DEVICE_BATCH_SIZE * MODEL_CONFIG.sequence_len
    grad_accum_steps = TOTAL_BATCH_TOKENS // (tokens_per_step * world_size)
    grad_accum_steps = max(1, grad_accum_steps)
    effective_batch_tokens = (
        DEVICE_BATCH_SIZE * MODEL_CONFIG.sequence_len * world_size * grad_accum_steps
    )
    print0(f"DDP: {world_size} GPUs, grad_accum={grad_accum_steps}, "
           f"effective batch = {effective_batch_tokens:,} tokens/step")

    # ---- Tokenizer --------------------------------------------------------
    if not os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer.pkl")):
        print0(f"No tokenizer found at {TOKENIZER_DIR}/. "
               f"Train one first: python -m microgpt.tokenizer --input <file> --output {TOKENIZER_DIR}")
        ddp_cleanup()
        return
    tokenizer = BPETokenizer.load(TOKENIZER_DIR)
    print0(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}")

    # ---- Model ------------------------------------------------------------
    model = MicroGPT(MODEL_CONFIG).to(device)
    print0(f"Model: {model.count_parameters():,} parameters")

    # Compile for speed (PyTorch 2.0+)
    model = torch.compile(model)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if is_ddp else model

    # ---- Optimizer --------------------------------------------------------
    optimizer = configure_optimizer(raw_model)

    # ---- Data loaders -----------------------------------------------------
    train_loader = create_dataloader(
        tokenizer, DEVICE_BATCH_SIZE, MODEL_CONFIG.sequence_len,
        split="train", rank=rank, world_size=world_size, device=device,
    )
    val_loader = create_dataloader(
        tokenizer, DEVICE_BATCH_SIZE, MODEL_CONFIG.sequence_len,
        split="val", rank=rank, world_size=world_size, device=device,
    )

    # ---- Training ---------------------------------------------------------
    print0(f"Starting training for {MAX_STEPS} steps")
    model.train()
    best_val_loss = float("inf")
    t0 = time.time()
    tokens_processed = 0

    pbar = tqdm(range(MAX_STEPS), desc="Training", unit="step",
                disable=not is_master(rank))
    for step in pbar:
        # Set learning rate
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation loop
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(grad_accum_steps):
            inputs, targets = next(train_loader)
            # Only sync gradients on the last micro-step
            if is_ddp:
                ctx = model.no_sync() if micro_step < grad_accum_steps - 1 else nullcontext()
            else:
                ctx = nullcontext()

            with ctx:
                with torch.amp.autocast(device_type, dtype=torch.bfloat16):
                    _, loss = model(inputs, targets)
                    loss = loss / grad_accum_steps
                loss.backward()
            accum_loss += loss.item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), GRAD_CLIP)

        # Optimizer step
        optimizer.step()

        tokens_processed += effective_batch_tokens

        # ---- Logging & evaluation -----------------------------------------
        dt = time.time() - t0
        tok_per_sec = tokens_processed / dt if dt > 0 else 0
        pbar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{lr:.2e}",
                         tok_s=f"{tok_per_sec:,.0f}")

        if step % 50 == 0 or step == MAX_STEPS - 1:
            mfu = estimate_mfu(raw_model, tok_per_sec, device)
            pbar.write(
                f"step {step:>6d}/{MAX_STEPS} | "
                f"loss {accum_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"tok/s {tok_per_sec:,.0f} | "
                f"mfu {mfu:.1%}"
            )

        if (step + 1) % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            val_loss = evaluate(raw_model, val_loader, device, EVAL_STEPS)
            pbar.write(f"  val_loss: {val_loss:.4f}")

            if val_loss < best_val_loss and is_master(rank):
                best_val_loss = val_loss
                save_checkpoint(raw_model, optimizer, step, val_loss, "best.pt")

        if (step + 1) % SAVE_INTERVAL == 0 and is_master(rank):
            save_checkpoint(raw_model, optimizer, step, accum_loss, f"step_{step+1}.pt")

    # Save final checkpoint
    if is_master(rank):
        save_checkpoint(raw_model, optimizer, MAX_STEPS - 1, accum_loss, "final.pt")

    print0(f"Training complete. Total tokens: {tokens_processed:,}")
    ddp_cleanup()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    filename: str,
) -> None:
    """Save model + optimizer state to checkpoint."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "loss": loss,
        "config": model.config,
    }, path)
    print0(f"  Saved checkpoint: {path}")


# contextlib.nullcontext is Python 3.10+
from contextlib import nullcontext


if __name__ == "__main__":
    train()
