"""MicroGPT model configuration."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """GPT-2 124M configuration.

    This is the smallest GPT-2 variant from Radford et al. (2019).
    Compared to BabyGPT (6 layers, 6 heads, 384 dim, 256 context, 65 vocab):
    every axis is scaled up, and character-level tokenization is replaced by BPE.
    """
    sequence_len: int = 1024     # context window (was 256 in BabyGPT)
    vocab_size: int = 32768      # BPE vocabulary (was 65)
    n_layer: int = 12            # transformer blocks (was 6)
    n_head: int = 12             # query heads (was 6)
    n_kv_head: int = 12          # key/value heads (= n_head for MHA, < for GQA)
    n_embd: int = 768            # embedding dimension (was 384)
    dropout: float = 0.0         # no dropout at scale (was 0.2)
    bias: bool = False           # no bias in Linear/LayerNorm (same as BabyGPT)
