"""TinyShakespeare dataset: download, tokenize, and create DataLoaders."""

import os
from pathlib import Path

import requests
import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer import CharTokenizer
from src.config import GPTConfig

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "tinyshakespeare.txt"


def download_data() -> str:
    """Download TinyShakespeare if not already present. Return the text."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        print(f"Downloading TinyShakespeare to {DATA_FILE} ...")
        resp = requests.get(DATA_URL, timeout=30)
        resp.raise_for_status()
        DATA_FILE.write_text(resp.text)
        print(f"Downloaded {len(resp.text):,} characters.")
    return DATA_FILE.read_text()


class ShakespeareDataset(Dataset):
    """Yields (x, y) pairs where x = chunk[:-1] and y = chunk[1:].

    Each sample is a contiguous slice of block_size+1 characters from the
    encoded text. The input x is the first block_size tokens; the target y
    is the same window shifted right by one position. Every position in x
    provides one next-token prediction training example.
    """

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]  # (block_size,)
        y = chunk[1:]   # (block_size,)
        return x, y


def get_dataloaders(
    config: GPTConfig,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, CharTokenizer]:
    """Download data, build tokenizer, return (train_loader, val_loader, tokenizer)."""
    text = download_data()
    tokenizer = CharTokenizer(text)

    # Encode entire text as a single long tensor
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # 90/10 train/val split
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    train_dataset = ShakespeareDataset(train_data, config.block_size)
    val_dataset = ShakespeareDataset(val_data, config.block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    text = download_data()
    tok = CharTokenizer(text)
    print(f"Characters: {len(text):,}")
    print(f"Vocab size: {tok.vocab_size}")
    print(f"Vocab: {''.join(sorted(tok.char_to_idx.keys()))!r}")

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    print(f"Train tokens: {split:,}")
    print(f"Val tokens:   {len(data) - split:,}")
