"""BabyGPT: a decoder-only transformer language model.

Every class here maps directly to a concept from the "Attention Is All You Need"
paper, adapted for the decoder-only (GPT) variant. See learn/ documents for the
full derivations.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import GPTConfig


class LayerNorm(nn.Module):
    """Layer normalization with optional bias.

    Normalizes across the last dimension (features) for each position
    independently. Learnable affine parameters gamma (weight) and beta (bias)
    rescale and shift the output.

    Math: LayerNorm(x) = gamma * (x - mean) / (std + eps) + beta
    """

    def __init__(self, n_embd: int, bias: bool = False, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (autoregressive) mask.

    Implements the core attention formula:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Multiple heads run in parallel by reshaping the projected Q, K, V tensors
    so that the batch and head dimensions are both treated as batch dimensions.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Combined projection for Q, K, V (more efficient than three separate ones)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Project to Q, K, V and split into heads
        # (B, T, 3*C) -> 3 x (B, T, C) -> 3 x (B, n_head, T, head_dim)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        # PyTorch's F.scaled_dot_product_attention handles the mask, scaling,
        # softmax, and dropout in a single fused kernel (flash attention).
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # Reassemble heads: (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + dropout
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    A two-layer MLP applied independently to each position:
        FFN(x) = W2 * GELU(W1 * x) + b2

    The hidden dimension is 4x the embedding dimension, giving the network
    more capacity to process each position's representation.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """A single transformer block: pre-norm attention + pre-norm FFN.

    Data flow:
        x -> LayerNorm -> CausalSelfAttention -> + residual
          -> LayerNorm -> FeedForward          -> + residual

    Pre-norm means we normalize *before* the sublayer, not after (the original
    paper used post-norm). Pre-norm is more stable to train and is the default
    in GPT-2, GPT-3, and most modern transformers.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class BabyGPT(nn.Module):
    """Decoder-only transformer language model (GPT architecture).

    Components:
        1. Token embedding: maps token IDs to vectors of size n_embd
        2. Position embedding: learned embedding for each position 0..block_size-1
        3. Transformer blocks: N stacked TransformerBlock layers
        4. Final LayerNorm
        5. Language model head: projects back to vocab_size logits

    Weight tying: the token embedding and lm_head share the same weight matrix.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),          # token embedding
            wpe=nn.Embedding(config.block_size, config.n_embd),          # position embedding
            drop=nn.Dropout(config.dropout),
            blocks=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),             # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: embedding and output projection share weights.
        # This reduces parameters and acts as a form of regularization — the
        # model is encouraged to produce embeddings that are directly useful
        # for predicting the next token.
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Scaled initialization for residual projections (GPT-2 convention):
        # scale down by 1/sqrt(2*n_layer) to keep variance from growing with depth
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
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
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target token indices, or None for inference

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar cross-entropy loss if targets provided, else None
        """
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Position indices: 0, 1, 2, ..., T-1
        pos = torch.arange(T, device=idx.device)

        # Token embeddings + position embeddings
        tok_emb = self.transformer.wte(idx)     # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)     # (T, n_embd) — broadcast over batch
        x = self.transformer.drop(tok_emb + pos_emb)

        # Pass through all transformer blocks
        for block in self.transformer.blocks:
            x = block(x)

        # Final layer norm + project to vocabulary
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy expects (N, C) and (N,), so flatten
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive token generation.

        Args:
            idx: (B, T) conditioning token indices
            max_new_tokens: number of tokens to generate
            temperature: >1 = more random, <1 = more deterministic
            top_k: if set, only sample from the top k most probable tokens

        Returns:
            (B, T + max_new_tokens) tensor of token indices
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if the sequence has grown too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            # Take logits at the last position
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            if top_k is not None:
                # Zero out everything below the top-k values
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def count_parameters(self) -> int:
        """Return total number of trainable parameters (excluding tied weights)."""
        # lm_head.weight is tied to wte.weight, so subtract it
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= self.transformer.wpe.weight.numel()  # don't subtract — this is unique
        # Actually, just count unique parameters
        n_params = sum(p.numel() for p in self.parameters())
        # wte and lm_head share weight, so we've double-counted
        n_params -= self.lm_head.weight.numel()
        return n_params
