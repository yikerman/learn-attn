"""BabyGPT: a decoder-only transformer language model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class LayerNorm(nn.Module):
    """Hand-written LayerNorm with optional bias.

    LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
    Normalizes across the last dimension (features) per position.
    gamma (weight) and beta (bias) are learnable affine parameters.
    """

    def __init__(self, config: GPTConfig, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))    # gamma
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None  # beta
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        out = self.weight * x_normalized
        if self.bias is not None:
            out = out + self.bias
        return out


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal mask.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    Runs h heads in parallel by reshaping into (batch, h, seq_len, d_k).
    Causal mask sets future positions to -inf before softmax.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Each projection is (n_embd, n_embd), i.e. (384, 384).
        # Conceptually this is h separate (n_embd, head_dim) projections
        # stacked together: 6 heads x 64 dims = 384.
        self.W_Q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.W_K = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.W_V = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head       # h = 6
        self.n_embd = config.n_embd       # 384
        self.head_dim = config.n_embd // config.n_head  # d_k = 384/6 = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, embed_dim = x.shape

        # Project then split into heads:
        # (batch, seq_len, 384) -> view as (batch, seq_len, 6, 64) -> (batch, 6, seq_len, 64)
        # Each head gets its own 64-dim slice of Q, K, V.
        query = self.W_Q(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        key = self.W_K(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        value = self.W_V(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Attention per head: (batch, 6, seq_len, 64) @ (batch, 6, 64, seq_len) -> (batch, 6, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        attended = torch.matmul(weights, value)  # (batch, 6, seq_len, 64)

        # Reassemble heads: (batch, 6, seq_len, 64) -> (batch, seq_len, 6, 64) -> (batch, seq_len, 384)
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        return self.resid_dropout(self.c_proj(attended))


class FeedForward(nn.Module):
    """Position-wise FFN with GELU.

    FFN(x) = W2 * GELU(W1 * x)
    Hidden dim is 4x the embedding dim (expand then contract).
    """

    def __init__(self, config: GPTConfig) -> None:
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
    """Pre-norm transformer block.

    x = x + Attention(LayerNorm(x))   -- pre-norm + residual
    x = x + FFN(LayerNorm(x))         -- pre-norm + residual
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class BabyGPT(nn.Module):
    """Decoder-only transformer language model.

    Forward: token_emb + pos_emb -> N x TransformerBlock -> LayerNorm -> lm_head
    Loss: cross-entropy over vocab at every position (next-token prediction).
    Weight tying: lm_head shares weights with token embedding.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.context_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            blocks=nn.ModuleList(
                [TransformerBlock(config) for _ in range(config.n_layer)]
            ),
            ln_f=LayerNorm(config),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 convention)
        for param_name, param in self.named_parameters():
            if param_name.endswith("c_proj.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

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
        batch, seq_len = tokens.shape
        assert seq_len <= self.config.context_size
        positions = torch.arange(seq_len, device=tokens.device)

        token_emb = self.transformer.wte(tokens)       # (batch, seq_len, n_embd)
        pos_emb = self.transformer.wpe(positions)       # (seq_len, n_embd)
        x = self.transformer.drop(token_emb + pos_emb)

        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                        # (batch, seq_len, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
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
        """Autoregressive token generation.

        Loop: forward pass -> logits / temperature -> top-k filter
              -> softmax -> multinomial sample -> append token.
        Crops to context_size when sequence exceeds context window.
        """
        for _ in range(max_new_tokens):
            tokens_cropped = (
                tokens if tokens.size(1) <= self.config.context_size
                else tokens[:, -self.config.context_size:]
            )

            logits, _ = self(tokens_cropped)
            logits = logits[:, -1, :] / temperature  # (batch, vocab_size)

            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens

    def count_parameters(self) -> int:
        """Trainable parameters (excluding tied duplicates)."""
        total = sum(p.numel() for p in self.parameters())
        total -= self.lm_head.weight.numel()  # tied with wte
        return total
