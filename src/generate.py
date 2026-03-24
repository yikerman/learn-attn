"""Generate text from a trained BabyGPT checkpoint."""

import argparse
from pathlib import Path

import torch

from src.config import GPTConfig
from src.model import BabyGPT
from src.tokenizer import CharTokenizer

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[BabyGPT, CharTokenizer]:
    """Load a model and tokenizer from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    model = BabyGPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # Reconstruct tokenizer from saved vocab
    vocab = checkpoint["vocab"]
    # Create a dummy tokenizer and set its mappings
    tok = CharTokenizer.__new__(CharTokenizer)
    tok.char_to_idx = vocab
    tok.idx_to_char = {i: ch for ch, i in vocab.items()}

    print(f"Loaded model from {checkpoint_path}")
    print(f"  step={checkpoint.get('step', '?')}, val_loss={checkpoint.get('val_loss', '?'):.4f}")

    return model, tok


def generate(
    model: BabyGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 500,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate text given a prompt string."""
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    output = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)

    return tokenizer.decode(output[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text with BabyGPT")
    parser.add_argument("--prompt", type=str, default="\n", help="Starting text")
    parser.add_argument("--max-tokens", type=int, default=500, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_DIR / "best.pt"),
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(Path(args.checkpoint), device)

    text = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
