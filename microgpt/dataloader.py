"""Distributed dataloader with BOS-aligned best-fit packing.

This dataloader:
1. Reads documents from parquet shards (via dataset.py)
2. Tokenizes them with BOS prepended
3. Packs documents into fixed-length rows using best-fit bin packing
4. Distributes shards across DDP ranks
5. Pre-allocates pinned CPU buffers for efficient GPU transfer

Every row starts with a BOS token, so the model always sees the beginning
of at least one document.  Documents that don't fit are cropped (the cropped
portion is discarded, not carried over).  This wastes ~35% of tokens but
ensures every token can attend back to its document's BOS.
"""

import torch
import pyarrow.parquet as pq

from .dataset import list_parquet_files


# ---------------------------------------------------------------------------
# Document iterator (with DDP sharding)
# ---------------------------------------------------------------------------

def _document_batches(
    split: str,
    rank: int = 0,
    world_size: int = 1,
    batch_size: int = 128,
):
    """Infinite iterator over batches of document strings from parquet files.

    DDP sharding: each rank reads every world_size-th row group, so ranks
    see disjoint data without any communication.

    Yields: (text_batch, shard_idx)
    """
    paths = list_parquet_files()
    assert paths, "No parquet files found. Run: python -m microgpt.dataset"
    # Last shard is validation, rest are training
    if split == "train":
        paths = paths[:-1]
    else:
        paths = paths[-1:]

    while True:  # infinite iteration (multi-epoch)
        for shard_idx, filepath in enumerate(paths):
            pf = pq.ParquetFile(filepath)
            # Each rank reads every world_size-th row group
            rg_idx = rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                texts = rg.column("text").to_pylist()
                for i in range(0, len(texts), batch_size):
                    yield texts[i : i + batch_size], shard_idx
                rg_idx += world_size


# ---------------------------------------------------------------------------
# Best-fit packing
# ---------------------------------------------------------------------------

def _pack_row(
    doc_buffer: list[list[int]],
    row: torch.Tensor,
    row_capacity: int,
) -> None:
    """Pack documents from doc_buffer into a single row using best-fit.

    Algorithm:
    1. Find the largest document that fits entirely in the remaining space.
    2. Append it and repeat.
    3. When no document fits, crop the shortest one to fill exactly.

    This achieves 100% utilization (no padding) at the cost of ~35% cropped
    tokens.
    """
    pos = 0
    while pos < row_capacity:
        remaining = row_capacity - pos

        # Find largest doc that fits entirely
        best_idx = -1
        best_len = 0
        for i, doc in enumerate(doc_buffer):
            doc_len = len(doc)
            if doc_len <= remaining and doc_len > best_len:
                best_idx = i
                best_len = doc_len

        if best_idx >= 0:
            doc = doc_buffer.pop(best_idx)
            row[pos : pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
            pos += len(doc)
        else:
            # No doc fits — crop shortest to fill remaining space exactly
            shortest_idx = min(
                range(len(doc_buffer)), key=lambda i: len(doc_buffer[i])
            )
            doc = doc_buffer.pop(shortest_idx)
            row[pos : pos + remaining] = torch.tensor(
                doc[:remaining], dtype=torch.long
            )
            pos += remaining


# ---------------------------------------------------------------------------
# Main dataloader
# ---------------------------------------------------------------------------

def create_dataloader(
    tokenizer,
    batch_size: int,
    sequence_len: int,
    split: str = "train",
    rank: int = 0,
    world_size: int = 1,
    device: str = "cuda",
    buffer_size: int = 1000,
):
    """Create an infinite dataloader that yields (inputs, targets) batches.

    Args:
        tokenizer: BPETokenizer instance (needs .encode_with_bos and .bos_id)
        batch_size: number of rows per batch (per GPU)
        sequence_len: context length (T)
        split: "train" or "val"
        rank: DDP rank (0 for single-GPU)
        world_size: DDP world size (1 for single-GPU)
        device: target device ("cuda" or "cpu")
        buffer_size: min documents in buffer before packing

    Yields:
        (inputs, targets): both shape (batch_size, sequence_len), on device
    """
    row_capacity = sequence_len + 1  # +1 because targets are shifted by 1
    batches = _document_batches(split, rank, world_size)
    doc_buffer: list[list[int]] = []

    # Pre-allocate buffers for efficient transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty(
        (batch_size, row_capacity), dtype=torch.long
    )
    cpu_buffer = torch.empty(
        2 * batch_size * sequence_len, dtype=torch.long, pin_memory=use_cuda
    )
    gpu_buffer = torch.empty(
        2 * batch_size * sequence_len, dtype=torch.long, device=device
    )
    # Views into the flat buffers
    cpu_inputs = cpu_buffer[: batch_size * sequence_len].view(
        batch_size, sequence_len
    )
    cpu_targets = cpu_buffer[batch_size * sequence_len :].view(
        batch_size, sequence_len
    )
    inputs = gpu_buffer[: batch_size * sequence_len].view(
        batch_size, sequence_len
    )
    targets = gpu_buffer[batch_size * sequence_len :].view(
        batch_size, sequence_len
    )

    while True:
        # Build a full batch
        for row_idx in range(batch_size):
            # Refill buffer if needed
            while len(doc_buffer) < buffer_size:
                text_batch, _ = next(batches)
                for text in text_batch:
                    doc_ids = tokenizer.encode_with_bos(text)
                    doc_buffer.append(doc_ids)

            _pack_row(doc_buffer, row_buffer[row_idx], row_capacity)

        # Split into inputs and targets (shifted by 1)
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        # Single host-to-device transfer
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets
