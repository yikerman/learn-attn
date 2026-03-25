"""MicroGPT: a GPT-2 124M decoder-only transformer language model.

This version incorporates modern improvements over the canonical GPT-2:
- RMSNorm instead of LayerNorm
- Rotary Position Embeddings (RoPE) instead of learned positional embeddings
- Grouped-Query Attention (GQA) with QK normalization
- Flash Attention via PyTorch SDPA
- relu-squared activation in FFN instead of GELU
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .attention import CausalSelfAttention, precompute_rope_frequencies


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (Zhang & Sennrich, 2019).

    RMSNorm(x) = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)

    Compared to LayerNorm, RMSNorm skips the mean-centering step.
    This makes it simpler (fewer ops, no bias parameter) and empirically
    works just as well.  Used by LLaMA, Gemma, and most modern LLMs.
    """

    def __init__(self, n_embd: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class FeedForward(nn.Module):
    """Position-wise FFN with relu-squared activation (So et al., 2021).

    FFN(x) = W2 * relu(W1 * x)^2

    relu-squared is computationally simpler than GELU and produces sparse
    activations (many exact zeros plus a few strongly positive values).
    Empirically competitive with GELU at GPT-2 scale and above.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()  # relu^2
        x = self.c_proj(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with modern components.

    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            bias=config.bias,
        )
        self.ln_2 = RMSNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.ffn(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MicroGPT(nn.Module):
    """Decoder-only transformer language model (GPT-2 scale, modern arch).

    Key differences from the canonical GPT-2 (Chapter 2):
    - RMSNorm instead of LayerNorm
    - RoPE instead of learned positional embeddings (no wpe)
    - GQA support (n_kv_head can be < n_head)
    - QK normalization inside attention
    - relu-squared activation instead of GELU
    - Flash Attention via SDPA
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Pad vocab to nearest multiple of 64 for tensor-core efficiency
        padded_vocab = ((config.vocab_size + 63) // 64) * 64

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(padded_vocab, config.n_embd),
            blocks=nn.ModuleList(
                [TransformerBlock(config) for _ in range(config.n_layer)]
            ),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)

        # No positional embedding table --- RoPE handles positions
        # Precompute RoPE cos/sin tables
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope_frequencies(
            head_dim, config.sequence_len, device=None,
        )
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # RMSNorm after token embedding (stabilizes early training)
        self.embed_norm = RMSNorm(config.n_embd)

        # Initialize weights
        self.apply(self._init_weights)
        # Scaled init for residual projections
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = tokens.shape
        assert T <= self.config.sequence_len

        # Token embeddings only --- no positional embeddings (RoPE handles this)
        x = self.transformer.wte(tokens)
        x = self.embed_norm(x)

        # Pass cos/sin through the blocks for RoPE
        cos = self.rope_cos.to(x.dtype)
        sin = self.rope_sin.to(x.dtype)

        for block in self.transformer.blocks:
            x = block(x, cos, sin)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        # Crop logits to actual vocab_size (remove padding tokens)
        logits = logits[..., :self.config.vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation (no KV cache --- Chapter 7 adds that)."""
        for _ in range(max_new_tokens):
            tokens_cropped = (
                tokens if tokens.size(1) <= self.config.sequence_len
                else tokens[:, -self.config.sequence_len:]
            )
            logits, _ = self(tokens_cropped)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens

    def count_parameters(self) -> int:
        """Trainable parameters (excludes RoPE buffers)."""
        return sum(p.numel() for p in self.parameters())
