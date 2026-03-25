"""Distributed training utilities for DDP.

This module provides helpers for setting up and tearing down PyTorch
Distributed Data Parallel (DDP) training with NCCL backend.

Usage:
    rank, local_rank, world_size = ddp_init()
    # ... training loop ...
    ddp_cleanup()
"""

import os

import torch
import torch.distributed as dist


def ddp_init() -> tuple[int, int, int]:
    """Initialize DDP and return (rank, local_rank, world_size).

    Reads RANK, LOCAL_RANK, WORLD_SIZE from environment (set by torchrun).
    If these are not set, assumes single-GPU mode.
    """
    if "RANK" not in os.environ:
        # Single-GPU mode
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def ddp_cleanup() -> None:
    """Destroy the DDP process group if it exists."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_master(rank: int) -> bool:
    """Return True if this is the master process (rank 0)."""
    return rank == 0


def print0(*args, **kwargs) -> None:
    """Print only on rank 0 (or when not using DDP)."""
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        print(*args, **kwargs)
