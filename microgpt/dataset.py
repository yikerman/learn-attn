"""Dataset management: download and list FineWeb-Edu parquet shards.

FineWeb-Edu is an open, high-quality web text dataset filtered for
educational content.  It is distributed as parquet files on HuggingFace.
Each file contains a 'text' column with document strings.

Usage:
    python -m microgpt.dataset                  # download 10 shards (~5 GB)
    python -m microgpt.dataset --num-shards 50  # download more for full training
"""

import os
import glob

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "microgpt")

# FineWeb-Edu-score-2 sample: ~10B tokens total, split into 100 parquet files
# Each shard is ~50-100 MB, containing ~100M tokens
BASE_URL = (
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2"
    "/resolve/main/data/CC-MAIN-2024-10"
)
# Shard filenames: train-00000-of-00099.parquet ... train-00098-of-00099.parquet
NUM_SHARDS_AVAILABLE = 99


def shard_filename(idx: int) -> str:
    return f"train-{idx:05d}-of-{NUM_SHARDS_AVAILABLE:05d}.parquet"


def shard_path(idx: int) -> str:
    return os.path.join(DATA_DIR, shard_filename(idx))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_shard(idx: int) -> None:
    """Download a single parquet shard if not already present."""
    path = shard_path(idx)
    if os.path.exists(path):
        return
    url = f"{BASE_URL}/{shard_filename(idx)}"
    print(f"  Downloading shard {idx}: {url}")
    os.makedirs(DATA_DIR, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    # Write to a temp file first, then rename (atomic on POSIX)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
    os.rename(tmp_path, path)


def download_shards(num_shards: int = 10) -> None:
    """Download parquet shards for training."""
    num_shards = min(num_shards, NUM_SHARDS_AVAILABLE)
    print(f"Downloading {num_shards} FineWeb-Edu shards to {DATA_DIR}")
    for idx in range(num_shards):
        download_shard(idx)
    print(f"Done. {num_shards} shards in {DATA_DIR}")


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_parquet_files() -> list[str]:
    """Return sorted list of all downloaded parquet shard paths."""
    pattern = os.path.join(DATA_DIR, "train-*.parquet")
    paths = sorted(glob.glob(pattern))
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description="Download FineWeb-Edu dataset shards")
    parser.add_argument("--num-shards", type=int, default=10, help="Number of shards to download")
    args = parser.parse_args()

    download_shards(args.num_shards)

    # Print stats for the first shard
    paths = list_parquet_files()
    if paths:
        pf = pq.ParquetFile(paths[0])
        print(f"\nFirst shard: {paths[0]}")
        print(f"  Row groups: {pf.num_row_groups}")
        print(f"  Total rows: {pf.metadata.num_rows:,}")
        rg = pf.read_row_group(0)
        sample = rg.column("text")[0].as_py()
        print(f"  Sample text (first 200 chars): {sample[:200]}")
