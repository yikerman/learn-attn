"""TinyShakespeare dataset: tokenize and create DataLoaders."""

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import CharTokenizer
from .config import GPTConfig

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"


class ShakespeareDataset(Dataset):
    """Yields (input, target) pairs where input = chunk[:-1] and target = chunk[1:].

    Each sample is a contiguous slice of context_size+1 characters from the
    encoded text. The input is the first context_size tokens; the target
    is the same window shifted right by one position. Every position in the
    input provides one next-token prediction training example.
    """

    def __init__(self, data: torch.Tensor, context_size: int) -> None:
        self.data = data
        self.context_size = context_size

    def __len__(self) -> int:
        return len(self.data) - self.context_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[index : index + self.context_size + 1]
        inputs = chunk[:-1]   # (context_size,)
        targets = chunk[1:]   # (context_size,)
        return inputs, targets


def get_dataloaders(
    config: GPTConfig,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, CharTokenizer]:
    """Read data, build tokenizer, return (train_loader, val_loader, tokenizer)."""
    text = DATA_FILE.read_text()
    tokenizer = CharTokenizer(text)

    # Encode entire text as a single long tensor
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # 90/10 train/val split
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    train_dataset = ShakespeareDataset(train_data, config.context_size)
    val_dataset = ShakespeareDataset(val_data, config.context_size)

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
    text = DATA_FILE.read_text()
    tokenizer = CharTokenizer(text)
    print(f"Characters: {len(text):,}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Vocab: {''.join(sorted(tokenizer.char_to_index.keys()))!r}")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    print(f"Train tokens: {split:,}")
    print(f"Val tokens:   {len(data) - split:,}")
