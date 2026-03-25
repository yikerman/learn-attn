"""Training script for BabyGPT on TinyShakespeare."""

import argparse
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .config import GPTConfig
from .dataset import get_dataloaders
from .model import BabyGPT

# ---- Hyperparameters ----
BATCH_SIZE = 64
MAX_ITERS = 5000
EVAL_INTERVAL = 250
EVAL_ITERS = 200

LEARNING_RATE = 1e-3
MIN_LR = 1e-4
WARMUP_ITERS = 100
LR_DECAY_ITERS = 5000

WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.99
GRAD_CLIP = 1.0

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def get_lr(step: int) -> float:
    """Cosine annealing with linear warmup.

    - Warmup: linear ramp from 0 to LEARNING_RATE over WARMUP_ITERS steps.
    - Cosine decay: smoothly decrease from LEARNING_RATE to MIN_LR.

    Math: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
    where progress goes from 0 to 1 over the decay phase.
    """
    # Linear warmup
    if step < WARMUP_ITERS:
        return LEARNING_RATE * step / WARMUP_ITERS
    # After decay period, stay at min
    if step > LR_DECAY_ITERS:
        return MIN_LR
    # Cosine decay
    progress = (step - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


def configure_optimizer(model: BabyGPT) -> torch.optim.AdamW:
    """Set up AdamW with separate parameter groups for weight decay.

    Weight matrices get decayed; biases, layer norm params, and embeddings don't.
    This is standard practice -- weight decay on biases/norms doesn't help and
    can hurt.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't decay 1D parameters (biases, layer norm weights)
        if param.dim() < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(BETA1, BETA2))


@torch.no_grad()
def estimate_loss(
    model: BabyGPT,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    eval_iters: int = EVAL_ITERS,
) -> dict[str, float]:
    """Estimate train and val loss by averaging over eval_iters batches."""
    model.eval()
    results = {}
    for split_name, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []
        loader_iter = iter(loader)
        for _ in range(eval_iters):
            try:
                inputs, targets = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                inputs, targets = next(loader_iter)
            inputs, targets = inputs.to(device), targets.to(device)
            _, loss = model(inputs, targets)
            losses.append(loss.item())
        results[split_name] = sum(losses) / len(losses)
    model.train()
    return results


def train(args: argparse.Namespace) -> None:
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Seed
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)

    # Data
    config = GPTConfig()
    train_loader, val_loader, tokenizer = get_dataloaders(
        config, batch_size=BATCH_SIZE
    )
    # Update vocab_size from actual data
    config.vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {config.vocab_size}")

    # Model
    model = BabyGPT(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = configure_optimizer(model)

    # Mixed precision
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.float16  # RTX 3080 supports bf16 too, but fp16 is safe default

    # Training loop
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    train_iter = iter(train_loader)
    start_time = time.time()

    pbar = tqdm(range(args.max_iters), desc="Training", unit="step")
    for step in pbar:
        # Set learning rate for this step
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch (cycle through loader)
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward + backward with mixed precision
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            _, loss = model(inputs, targets)

        scaler.scale(loss).backward()

        # Gradient clipping (unscale first for correct norm computation)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Update progress bar with current loss
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        # Logging and evaluation
        if step % EVAL_INTERVAL == 0 or step == args.max_iters - 1:
            losses = estimate_loss(model, train_loader, val_loader, device)
            elapsed = time.time() - start_time
            tokens_per_sec = (step + 1) * BATCH_SIZE * config.context_size / elapsed
            pbar.write(
                f"step {step:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"lr {lr:.2e} | "
                f"tok/s {tokens_per_sec:.0f}"
            )

            # Save best checkpoint
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint = {
                    "model": model.state_dict(),
                    "config": config,
                    "step": step,
                    "val_loss": best_val_loss,
                    "vocab": tokenizer.char_to_index,
                }
                torch.save(checkpoint, CHECKPOINT_DIR / "best.pt")
                pbar.write(f"  -> saved checkpoint (val_loss={best_val_loss:.4f})")

    # Save final checkpoint
    checkpoint = {
        "model": model.state_dict(),
        "config": config,
        "step": args.max_iters,
        "val_loss": best_val_loss,
        "vocab": tokenizer.char_to_index,
    }
    torch.save(checkpoint, CHECKPOINT_DIR / "final.pt")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {CHECKPOINT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BabyGPT on TinyShakespeare")
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
