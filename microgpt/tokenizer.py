"""BPE tokenizer: train from scratch, encode/decode via tiktoken.

This module implements byte-level Byte Pair Encoding (BPE) as used by GPT-2
and later models.  The training algorithm is pure Python so every step is
visible.  Once trained, the merge table is handed to tiktoken for fast
encoding at inference time.

Typical usage:
    # Train on a corpus
    tokenizer = BPETokenizer.train(corpus_text, vocab_size=32768)
    tokenizer.save("tokenizer_dir")

    # Later: load and use
    tokenizer = BPETokenizer.load("tokenizer_dir")
    ids = tokenizer.encode("Hello, world!")
    text = tokenizer.decode(ids)
"""

import os
import pickle
import regex  # 're' doesn't support \p{L} Unicode categories; use 'regex'

import tiktoken
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Pre-tokenization split pattern (GPT-4 style)
# ---------------------------------------------------------------------------
# Before BPE sees the text, we split it into coarse chunks so that merges
# never cross word or number boundaries.  Each group in the alternation
# captures a specific kind of token:
#
#   'contractions    — 's, 't, 'd, 'm, 'll, 've, 're (case-insensitive)
#   letter words     — optional leading non-letter/non-digit, then letters
#   numbers          — 1-2 digit groups (keeps numbers short)
#   punctuation runs — non-letter/non-digit/non-space followed by newlines
#   lone newlines    — whitespace-only lines
#   trailing space   — spaces not followed by non-space (eaten separately)
#   other spaces     — remaining whitespace
#
# The \p{L} and \p{N} classes are Unicode-aware (handled by the 'regex'
# library, not stdlib 're').

SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"
    r"|[^\r\n\p{L}\p{N}]?+\p{L}+"
    r"|\p{N}{1,2}"
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*"
    r"|\s*[\r\n]"
    r"|\s+(?!\S)"
    r"|\s+"
)

# Special tokens prepended/appended to documents during training.
SPECIAL_TOKENS = ["<|bos|>"]


# ---------------------------------------------------------------------------
# BPE training (pure Python)
# ---------------------------------------------------------------------------

def _get_pair_counts(
    chunk_ids_list: list[list[int]],
) -> dict[tuple[int, int], int]:
    """Count every adjacent pair across all chunks."""
    counts: dict[tuple[int, int], int] = {}
    for ids in chunk_ids_list:
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge_pair(
    ids: list[int],
    pair: tuple[int, int],
    new_id: int,
) -> list[int]:
    """Replace every occurrence of *pair* in *ids* with *new_id*."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _merge_chunk_inplace(
    ids: list[int],
    pair: tuple[int, int],
    new_id: int,
) -> list[int]:
    """Replace pair in ids, returning new list. Optimized: skips chunks
    that can't possibly contain the pair (short-circuit on length)."""
    if len(ids) < 2:
        return ids
    out: list[int] = []
    i = 0
    a, b = pair
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(
    text: str,
    vocab_size: int,
    *,
    verbose: bool = True,
) -> list[tuple[tuple[int, int], int]]:
    """Train a byte-level BPE tokenizer on *text*.

    Returns a list of merges: ``[(pair, new_id), ...]`` in priority order.
    The base vocabulary is the 256 raw byte values (0-255), so the first
    merge produces token 256, the second 257, and so on.

    Args:
        text: Training corpus (plain string).
        vocab_size: Target vocabulary size including the 256 byte tokens.
        verbose: Print progress every 1000 merges.
    """
    import time

    assert vocab_size >= 256, "vocab_size must be >= 256 (the byte alphabet)"
    num_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
    assert num_merges >= 0, "vocab_size too small to fit byte alphabet + special tokens"

    # Step 1: Pre-tokenize the text into chunks using the split pattern.
    # This prevents merges from crossing word/number boundaries.
    pat = regex.compile(SPLIT_PATTERN)
    chunks = pat.findall(text)
    if verbose:
        print(f"  Pre-tokenized into {len(chunks):,} chunks")

    # Step 2: Convert each chunk to a list of raw byte values.
    chunk_ids_list: list[list[int]] = [
        list(chunk.encode("utf-8")) for chunk in chunks
    ]

    # Step 3: Iteratively merge the most frequent adjacent pair.
    merges: list[tuple[tuple[int, int], int]] = []
    t0 = time.time()
    pbar = tqdm(range(num_merges), desc="  BPE merges", unit="merge",
                disable=not verbose)
    for i in pbar:
        pair_counts = _get_pair_counts(chunk_ids_list)
        if not pair_counts:
            break  # nothing left to merge
        best_pair = max(pair_counts, key=pair_counts.get)
        new_id = 256 + i
        # Apply the merge to every chunk
        chunk_ids_list = [
            _merge_chunk_inplace(ids, best_pair, new_id) for ids in chunk_ids_list
        ]
        merges.append((best_pair, new_id))

        if (i + 1) % 100 == 0 or i == num_merges - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            pbar.set_postfix(rate=f"{rate:.0f}/s")

    if verbose:
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s — "
              f"{len(merges)} merges (vocab_size={256 + len(merges) + len(SPECIAL_TOKENS)})")

    return merges


# ---------------------------------------------------------------------------
# Tokenizer class (wraps tiktoken for fast encode/decode)
# ---------------------------------------------------------------------------

class BPETokenizer:
    """Byte-level BPE tokenizer backed by tiktoken.

    The merge table is trained with :func:`train_bpe` (pure Python) and then
    handed to tiktoken, which provides a fast Rust-based encoder.
    """

    def __init__(self, enc: tiktoken.Encoding) -> None:
        self._enc = enc
        self.bos_id: int = enc.encode_single_token("<|bos|>")

    # ---- Factory methods ---------------------------------------------------

    @classmethod
    def train(cls, text: str, vocab_size: int = 32768, *, verbose: bool = True) -> "BPETokenizer":
        """Train a new BPE tokenizer on *text*."""
        merges = train_bpe(text, vocab_size, verbose=verbose)

        # Build the mergeable_ranks dict that tiktoken expects:
        # bytes -> merge rank (lower = higher priority).
        # The first 256 entries are single-byte tokens.
        mergeable_ranks: dict[bytes, int] = {}
        for i in range(256):
            mergeable_ranks[bytes([i])] = i

        # Each merge combines two existing byte-strings into a new one.
        # We need to be able to reconstruct the byte-string for any token.
        token_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in merges:
            new_bytes = token_bytes[a] + token_bytes[b]
            token_bytes[new_id] = new_bytes
            mergeable_ranks[new_bytes] = new_id

        # Special tokens are assigned IDs after the merges.
        num_regular = len(mergeable_ranks)
        special_tokens = {
            name: num_regular + i for i, name in enumerate(SPECIAL_TOKENS)
        }

        enc = tiktoken.Encoding(
            name="microgpt",
            pat_str=SPLIT_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc)

    @classmethod
    def load(cls, directory: str) -> "BPETokenizer":
        """Load a previously saved tokenizer from *directory*."""
        path = os.path.join(directory, "tokenizer.pkl")
        with open(path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    @classmethod
    def from_pretrained(cls, name: str = "gpt2") -> "BPETokenizer":
        """Load a pretrained tiktoken encoding (e.g. ``"gpt2"``)."""
        enc = tiktoken.get_encoding(name)
        # Pretrained encoders (like GPT-2) use <|endoftext|> as BOS,
        # not <|bos|>, so we skip __init__ and set fields manually.
        obj = cls.__new__(cls)
        obj._enc = enc
        obj.bos_id = enc.encode_single_token("<|endoftext|>")
        return obj

    # ---- Core API ----------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        """Encode *text* into a list of token IDs (no special tokens)."""
        return self._enc.encode_ordinary(text)

    def encode_with_bos(self, text: str) -> list[int]:
        """Encode *text* with a leading BOS token."""
        return [self.bos_id] + self._enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to a string."""
        return self._enc.decode(ids)

    # ---- Persistence -------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save the tiktoken encoding to *directory*."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "tokenizer.pkl")
        with open(path, "wb") as f:
            pickle.dump(self._enc, f)
        print(f"Saved tokenizer to {path}")

    # ---- Inspection --------------------------------------------------------

    def token_to_bytes(self, token_id: int) -> bytes:
        """Return the raw bytes for a single token."""
        return self._enc.decode_single_token_bytes(token_id)

    def compression_ratio(self, text: str) -> float:
        """Tokens per byte: lower is better compression."""
        ids = self.encode(text)
        return len(ids) / len(text.encode("utf-8"))


# ---------------------------------------------------------------------------
# CLI: train a tokenizer on a text file
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("--input", type=str, required=True, help="Training text file")
    parser.add_argument("--vocab-size", type=int, default=32768, help="Target vocab size")
    parser.add_argument("--output", type=str, default="tokenizer", help="Output directory")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        text = f.read()

    print(f"Training BPE tokenizer on {len(text):,} characters (vocab_size={args.vocab_size})")
    tokenizer = BPETokenizer.train(text, vocab_size=args.vocab_size)
    tokenizer.save(args.output)

    # Quick sanity check
    sample = text[:200]
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)
    assert decoded == sample, f"Round-trip failed!\n{sample!r}\n{decoded!r}"
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Compression: {tokenizer.compression_ratio(text):.3f} tokens/byte")
    print(f"Sample: {sample[:80]!r} -> {len(ids)} tokens")
    print("Round-trip: OK")
