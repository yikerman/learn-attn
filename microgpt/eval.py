"""MicroGPT evaluation: validation loss and bits-per-byte.

Bits per byte (BPB) is a tokenizer-independent metric that measures how
well the model compresses text.  Lower is better.

BPB = (sum of per-token cross-entropy losses, weighted by token byte lengths)
      / (total bytes in the text)

Usage:
    uv run python -m microgpt.eval \
        --checkpoint microgpt_checkpoints/best.pt \
        --tokenizer tokenizer \
        --num-batches 50
"""

import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import GPTConfig
from .model import MicroGPT
from .tokenizer import BPETokenizer
from .dataloader import create_dataloader


@torch.no_grad()
def compute_val_loss(
    model: MicroGPT,
    val_loader,
    num_batches: int = 50,
) -> float:
    """Compute average cross-entropy loss on validation data."""
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (inputs, targets) in enumerate(tqdm(val_loader, desc="Val loss",
                                                total=num_batches, unit="batch")):
        if i >= num_batches:
            break
        _, loss = model(inputs, targets)
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


@torch.no_grad()
def compute_bpb(
    model: MicroGPT,
    val_loader,
    tokenizer: BPETokenizer,
    num_batches: int = 50,
) -> float:
    """Compute bits per byte (BPB) on validation data.

    BPB is tokenizer-independent: it measures compression in terms of
    raw bytes rather than tokens.  This allows fair comparison between
    models with different tokenizers.

    BPB = mean_cross_entropy / ln(2) * (tokens / bytes)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0

    for i, (inputs, targets) in enumerate(tqdm(val_loader, desc="BPB",
                                                total=num_batches, unit="batch")):
        if i >= num_batches:
            break

        logits, _ = model(inputs)
        # Per-token cross-entropy
        B, T, V = logits.shape
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            reduction="none",
        ).view(B, T)

        # Count bytes per token
        for b in range(B):
            for t in range(T):
                token_id = targets[b, t].item()
                try:
                    token_bytes = tokenizer.token_to_bytes(token_id)
                    n_bytes = len(token_bytes)
                except Exception:
                    n_bytes = 1  # fallback for special tokens

                total_loss += loss_per_token[b, t].item()
                total_bytes += n_bytes
                total_tokens += 1

    # BPB = total_nats / total_bytes / ln(2)
    import math
    bpb = total_loss / max(total_bytes, 1) / math.log(2)
    return bpb


def perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity."""
    import math
    return math.exp(loss)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MicroGPT")
    parser.add_argument("--checkpoint", type=str, default="microgpt_checkpoints/best.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer")
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = BPETokenizer.load(args.tokenizer)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = MicroGPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    print(f"Model: {model.count_parameters():,} parameters")

    # Create val dataloader
    val_loader = create_dataloader(
        tokenizer, args.batch_size, config.sequence_len,
        split="val", device=device,
    )

    # Compute metrics
    val_loss = compute_val_loss(model, val_loader, args.num_batches)
    ppl = perplexity(val_loss)
    print(f"Val loss: {val_loss:.4f}")
    print(f"Perplexity: {ppl:.2f}")

    # Recreate val loader (it was consumed)
    val_loader = create_dataloader(
        tokenizer, args.batch_size, config.sequence_len,
        split="val", device=device,
    )
    bpb = compute_bpb(model, val_loader, tokenizer, args.num_batches)
    print(f"Bits per byte: {bpb:.4f}")
