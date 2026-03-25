"""Modern attention components: RoPE, GQA, QK-norm, Flash Attention.

This module replaces the canonical GPT-2 attention (Chapter 2) with
techniques that have become standard since 2019.  Each is a self-contained
improvement with a clear paper trail.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE) — Su et al. 2021
# ---------------------------------------------------------------------------
# Instead of adding a learned position vector to each token, RoPE encodes
# relative position by *rotating* query and key vectors in 2D subspaces.
#
# Given a head-dim d, we pair up the dimensions into d/2 pairs.
# Each pair (x_{2i}, x_{2i+1}) is treated as a 2D vector and rotated
# by an angle θ_i * position, where θ_i = 1 / 10000^{2i/d}.
#
# The rotation matrix for pair i at position m is:
#
#   [ cos(m·θ_i)  -sin(m·θ_i) ]   [ x_{2i}   ]
#   [ sin(m·θ_i)   cos(m·θ_i) ] × [ x_{2i+1} ]
#
# This means: attention between positions m and n depends only on the
# *difference* (m - n), giving the model relative position awareness
# without any learned parameters.


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE.

    Returns (cos, sin) each of shape (max_seq_len, 1, 1, head_dim).
    The singleton dims broadcast over batch and heads.
    """
    assert head_dim % 2 == 0
    # θ_i = 1 / 10000^{2i/d} for i = 0, 1, ..., d/2 - 1
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # positions: 0, 1, ..., max_seq_len - 1
    positions = torch.arange(max_seq_len, device=device).float()
    # outer product: (max_seq_len, head_dim/2)
    angles = torch.outer(positions, freqs)
    # Duplicate to full head_dim so broadcasting with x works:
    # each angle applies to a *pair* of dimensions, but our implementation
    # splits the head into two halves rather than interleaving pairs.
    # Shape: (1, max_seq_len, 1, d/2) — broadcasts over (B, T, H, D/2)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(2)  # (1, max_seq_len, 1, d/2)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(2)
    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE rotation to x.

    x: (B, T, H, D)  where D = head_dim
    cos, sin: (T, 1, 1, D/2)  — precomputed tables
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # Rotation: [x1, x2] -> [x1·cos + x2·sin, -x1·sin + x2·cos]
    # Wait — the standard rotation is [x1·cos - x2·sin, x1·sin + x2·cos].
    # The nanochat convention (which we follow) swaps the sign pattern.
    # Both are valid — they just rotate in opposite directions.
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


# ---------------------------------------------------------------------------
# Causal Self-Attention with GQA, RoPE, QK-norm, and Flash Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Modern causal self-attention.

    Improvements over the canonical GPT-2 attention:
    - RoPE instead of learned positional embeddings
    - Grouped-Query Attention (GQA): n_kv_head <= n_head
    - QK normalization for training stability
    - PyTorch SDPA (Flash Attention / memory-efficient backend)
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % n_kv_head == 0

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.n_kv_groups = n_head // n_kv_head  # queries per KV head

        # Q projects to full n_head * head_dim; K and V project to n_kv_head * head_dim
        self.W_Q = nn.Linear(n_embd, n_head * self.head_dim, bias=bias)
        self.W_K = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=bias)
        self.W_V = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V
        # Q: (B, T, n_head, head_dim)
        # K, V: (B, T, n_kv_head, head_dim)
        q = self.W_Q(x).view(B, T, self.n_head, self.head_dim)
        k = self.W_K(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.W_V(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply RoPE to queries and keys
        # cos/sin are (1, max_seq_len, 1, D/2); slice to (1, T, 1, D/2)
        q = apply_rotary_emb(q, cos[:, :T], sin[:, :T])
        k = apply_rotary_emb(k, cos[:, :T], sin[:, :T])

        # QK normalization: normalize Q and K after RoPE
        # Prevents attention logits from growing with depth.
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))

        # Transpose to (B, H, T, D) for SDPA
        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        v = v.transpose(1, 2)  # (B, n_kv_head, T, head_dim)

        # GQA: expand K and V heads to match Q heads
        # Each group of (n_head / n_kv_head) query heads shares one KV head.
        if self.n_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_kv_groups, -1, -1)
            k = k.reshape(B, self.n_head, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_kv_groups, -1, -1)
            v = v.reshape(B, self.n_head, T, self.head_dim)

        # Flash Attention via SDPA (automatically selects the best backend)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reassemble heads: (B, n_head, T, head_dim) -> (B, T, n_embd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)
