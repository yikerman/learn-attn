from dataclasses import dataclass


@dataclass
class GPTConfig:
    context_size: int = 256     # max sequence length (context window)
    vocab_size: int = 65        # number of unique tokens (characters in TinyShakespeare)
    n_layer: int = 6            # number of transformer blocks
    n_head: int = 6             # number of attention heads
    n_embd: int = 384           # embedding dimension (d_model)
    dropout: float = 0.2        # dropout rate (higher because dataset is small)
    bias: bool = False          # use bias in Linear layers and LayerNorm
