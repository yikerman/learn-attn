"""MicroGPT text generation with KV cache.

The KV cache stores previously computed keys and values so that each
new token only requires computing Q for the new position (plus looking
up cached K and V for all previous positions).  This reduces per-token
cost from O(T) to O(1) in the attention computation.

Usage:
    uv run python -m microgpt.generate \
        --prompt "The meaning of life" \
        --temperature 0.8 \
        --top-k 200 \
        --max-tokens 256
"""

import argparse
import os

import torch
import torch.nn.functional as F

from .config import GPTConfig
from .model import MicroGPT, RMSNorm
from .attention import CausalSelfAttention, apply_rotary_emb, precompute_rope_frequencies
from .tokenizer import BPETokenizer


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

class KVCache:
    """Pre-allocated key/value cache for autoregressive generation.

    For each layer, stores keys and values of all previously processed
    tokens.  Shape per layer: (batch, max_seq_len, n_kv_head, head_dim).
    """

    def __init__(
        self,
        n_layers: int,
        max_seq_len: int,
        n_kv_head: int,
        head_dim: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> None:
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.pos = 0  # next write position

        # Pre-allocate: (n_layers, 2, batch, max_seq_len, n_kv_head, head_dim)
        # The '2' dimension is for K and V
        self.cache = torch.zeros(
            n_layers, 2, batch_size, max_seq_len, n_kv_head, head_dim,
            dtype=dtype, device=device,
        )

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new K, V and return the full cached K, V.

        k, v: (B, T_new, n_kv_head, head_dim) — new tokens
        Returns: (B, T_total, n_kv_head, head_dim) — all cached tokens
        """
        T_new = k.size(1)
        self.cache[layer_idx, 0, :, self.pos : self.pos + T_new] = k
        self.cache[layer_idx, 1, :, self.pos : self.pos + T_new] = v
        # Return everything up to and including the new tokens
        end = self.pos + T_new
        return (
            self.cache[layer_idx, 0, :, :end],
            self.cache[layer_idx, 1, :, :end],
        )

    def advance(self, n: int) -> None:
        """Advance the write position by n tokens."""
        self.pos += n


# ---------------------------------------------------------------------------
# Generation with KV cache
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: MicroGPT,
    tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Generate tokens autoregressively with KV cache.

    Two phases:
    1. Prefill: process the entire prompt at once, filling the cache.
    2. Decode: generate one token at a time, using cached K/V.

    Args:
        model: MicroGPT model (not compiled, not DDP-wrapped)
        tokens: (1, T_prompt) — prompt token IDs
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature (< 1 = sharper, > 1 = flatter)
        top_k: keep top-k tokens before sampling
        top_p: nucleus sampling threshold (keep smallest set with cumprob >= p)

    Returns:
        (1, T_prompt + max_new_tokens) — full sequence including prompt
    """
    model.eval()
    cfg = model.config
    device = tokens.device
    B = tokens.size(0)

    # Initialize KV cache
    head_dim = cfg.n_embd // cfg.n_head
    kv_cache = KVCache(
        n_layers=cfg.n_layer,
        max_seq_len=cfg.sequence_len,
        n_kv_head=cfg.n_kv_head,
        head_dim=head_dim,
        batch_size=B,
        dtype=torch.bfloat16,
        device=device,
    )

    # RoPE tables
    cos = model.rope_cos.to(torch.bfloat16)
    sin = model.rope_sin.to(torch.bfloat16)

    # Get the raw model (unwrap compile if needed)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model

    def forward_with_cache(tok: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Forward pass using KV cache. tok: (B, T_new)."""
        T_new = tok.size(1)
        x = raw.transformer.wte(tok)
        x = raw.embed_norm(x)
        x = x.to(torch.bfloat16)

        # RoPE for the positions we are processing
        cos_slice = cos[:, start_pos : start_pos + T_new]
        sin_slice = sin[:, start_pos : start_pos + T_new]

        for layer_idx, block in enumerate(raw.transformer.blocks):
            # --- Attention with cache ---
            h = block.ln_1(x)
            attn = block.attn

            q = attn.W_Q(h).view(B, T_new, attn.n_head, attn.head_dim)
            k = attn.W_K(h).view(B, T_new, attn.n_kv_head, attn.head_dim)
            v = attn.W_V(h).view(B, T_new, attn.n_kv_head, attn.head_dim)

            q = apply_rotary_emb(q, cos_slice, sin_slice)
            k = apply_rotary_emb(k, cos_slice, sin_slice)
            q = F.rms_norm(q, (attn.head_dim,))
            k = F.rms_norm(k, (attn.head_dim,))

            # Update cache and get full K, V history
            k_full, v_full = kv_cache.update(layer_idx, k, v)

            # GQA expansion
            q = q.transpose(1, 2)  # (B, n_head, T_new, head_dim)
            k_full = k_full.transpose(1, 2)  # (B, n_kv_head, T_total, head_dim)
            v_full = v_full.transpose(1, 2)

            if attn.n_kv_groups > 1:
                k_full = k_full.unsqueeze(2).expand(-1, -1, attn.n_kv_groups, -1, -1)
                k_full = k_full.reshape(B, attn.n_head, -1, attn.head_dim)
                v_full = v_full.unsqueeze(2).expand(-1, -1, attn.n_kv_groups, -1, -1)
                v_full = v_full.reshape(B, attn.n_head, -1, attn.head_dim)

            out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=(T_new > 1))
            out = out.transpose(1, 2).contiguous().view(B, T_new, -1)
            out = attn.c_proj(out)

            x = x + out

            # --- FFN ---
            x = x + block.ffn(block.ln_2(x))

        x = raw.transformer.ln_f(x)
        logits = raw.lm_head(x[:, -1:, :])  # only last position
        logits = logits[..., :raw.config.vocab_size]  # crop padding
        return logits.float()  # back to fp32 for sampling

    # Phase 1: Prefill — process entire prompt
    logits = forward_with_cache(tokens, start_pos=0)
    kv_cache.advance(tokens.size(1))

    # Phase 2: Decode — generate one token at a time
    all_tokens = [tokens]
    for _ in range(max_new_tokens):
        logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k is not None:
            topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < topk_vals[:, [-1]]] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above threshold
            remove_mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[remove_mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        all_tokens.append(next_token)

        # Forward pass for next token
        logits = forward_with_cache(next_token, start_pos=kv_cache.pos)
        kv_cache.advance(1)

    return torch.cat(all_tokens, dim=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with MicroGPT")
    parser.add_argument("--checkpoint", type=str, default="microgpt_checkpoints/best.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer")
    parser.add_argument("--prompt", type=str, default="The")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-p", type=float, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = BPETokenizer.load(args.tokenizer)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = MicroGPT(config).to(device)
    model.load_state_dict(checkpoint["model"])

    # Encode prompt
    prompt_ids = tokenizer.encode_with_bos(args.prompt)
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Generate
    output = generate(
        model, tokens,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # Decode and print
    text = tokenizer.decode(output[0].tolist())
    print(text)
