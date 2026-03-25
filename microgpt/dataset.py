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
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "microgpt")

# FineWeb-Edu sample-10BT: ~10B tokens total, split into 14 parquet files
# Each shard is ~2 GB on disk, containing ~700M tokens
BASE_URL = (
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu"
    "/resolve/main/sample/10BT"
)
# Shard filenames: 000_00000.parquet ... 013_00000.parquet
NUM_SHARDS_AVAILABLE = 14


def shard_filename(idx: int) -> str:
    return f"{idx:03d}_00000.parquet"


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
    total_size = int(response.headers.get("content-length", 0))
    # Write to a temp file first, then rename (atomic on POSIX)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True,
                  desc=f"  Shard {idx}", leave=False) as pbar:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))
    os.rename(tmp_path, path)


def download_shards(num_shards: int = 10) -> None:
    """Download parquet shards for training."""
    num_shards = min(num_shards, NUM_SHARDS_AVAILABLE)
    print(f"Downloading {num_shards} FineWeb-Edu shards to {DATA_DIR}")
    for idx in tqdm(range(num_shards), desc="Shards", unit="shard"):
        download_shard(idx)
    print(f"Done. {num_shards} shards in {DATA_DIR}")


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_parquet_files() -> list[str]:
    """Return sorted list of all downloaded parquet shard paths."""
    pattern = os.path.join(DATA_DIR, "*.parquet")
    paths = sorted(p for p in glob.glob(pattern) if not p.endswith(".tmp"))
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
