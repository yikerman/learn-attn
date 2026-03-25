# GPT from Scratch — Educational Tutorials

## Project Overview
Educational project with two tutorials, each building a GPT language model from scratch using plain PyTorch:

1. **BabyGPT** (basic) — A minimal 10.7M-param character-level GPT trained on TinyShakespeare.
   Based on "Attention Is All You Need" (Vaswani et al., 2017) and Karpathy's nanoGPT.

2. **MicroGPT** (advanced) — A full GPT-2 124M reproduction with BPE tokenization,
   modern architecture improvements (RoPE, RMSNorm, GQA, Flash Attention), distributed
   multi-GPU training, and evaluation on standard benchmarks.
   Based on "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
   and Karpathy's nanochat.

## Project Structure
- `docs/basic/` — BabyGPT LaTeX tutorial (6 chapters → `babygpt-tutorial.pdf`)
- `docs/advanced/` — MicroGPT LaTeX tutorial (8 chapters → `microgpt-tutorial.pdf`)
- `babygpt/` — BabyGPT Python package (model, tokenizer, dataset, training, generation)
- `microgpt/` — MicroGPT Python package (model, tokenizer, dataset, distributed training, inference, eval)
- `data/` — datasets (TinyShakespeare for BabyGPT; FineWeb-Edu shards for MicroGPT)
- `CLAUDE.md` — this file

## Tech Stack
- Python 3.14, managed by uv
- PyTorch (with CUDA, targeting modern NVIDIA GPUs)
- tiktoken (BPE tokenizer encoding for MicroGPT)
- pyarrow (Parquet data loading for MicroGPT)
- No other ML frameworks — plain PyTorch only

## Key Commands
```bash
./build.sh                                                                # rebuild both PDFs
uv sync                                                                   # install deps

# BabyGPT
uv run python -m babygpt.dataset                                          # download TinyShakespeare + print stats
uv run python -m babygpt.train                                            # train the model (~14 min on RTX 3080)
uv run python -m babygpt.generate --prompt "ROMEO:" --temperature 0.8     # generate text

# MicroGPT
uv run python -m microgpt.dataset                                         # download FineWeb-Edu shards
uv run torchrun --nproc_per_node=8 -m microgpt.train                      # train on 8 GPUs
uv run python -m microgpt.generate --prompt "The meaning of" --temperature 0.8
uv run python -m microgpt.eval                                            # run evaluation (BPB, benchmarks)
```

## Model Config (BabyGPT)
- 6 layers, 6 heads, 384 embedding dim, 256 context window
- ~10.7M parameters, character-level tokenization (vocab_size=65)
- Dropout 0.2, no bias in linear layers

## Model Config (MicroGPT)
- 12 layers, 12 heads, 768 embedding dim, 1024 context window
- ~135M parameters, BPE tokenization (vocab_size=32768)
- No dropout (large dataset), no bias in linear layers

## Training Config (BabyGPT)
- Batch size 64, 5000 iterations, AdamW (lr=1e-3, wd=0.1)
- Cosine LR schedule with 100-step warmup
- Mixed precision (float16), gradient clipping at 1.0
- Expected val loss ~1.47, training time ~14 min on RTX 3080

## Training Config (MicroGPT)
- Multi-GPU via DDP (torchrun), gradient accumulation for large effective batch sizes
- AdamW optimizer, cosine LR schedule with warmup and warmdown
- Mixed precision (bfloat16 activations, fp32 master weights)
- torch.compile for fused operations
- Checkpointing and resume support
- Target: ~10B tokens on FineWeb-Edu, trainable on 8×H100 via vast.ai

## Architecture Decisions (BabyGPT)
- **Pre-norm** (LayerNorm before sublayer) — matches GPT-2 and modern practice
- **GELU** activation in FFN
- **Learned positional embeddings** (not sinusoidal)
- **Weight tying** between token embedding and output head
- **bias=False** in linear layers — modern convention (GPT-J, LLaMA)
- **Character-level** tokenization — zero external deps, fully visible pipeline

## Architecture Decisions (MicroGPT)
- **RMSNorm** instead of LayerNorm — simpler, no centering step (Zhang & Sennrich, 2019)
- **Rotary Position Embeddings (RoPE)** — relative position via rotation (Su et al., 2021)
- **Grouped-Query Attention (GQA)** — fewer KV heads for efficient inference (Ainslie et al., 2023)
- **QK Normalization** — prevents attention score explosion in deep models
- **Flash Attention** via PyTorch SDPA — memory-efficient exact attention (Dao et al., 2022)
- **relu² activation** in FFN — simpler, competitive with GELU (So et al., 2021)
- **No weight tying** — separate embedding and output head (follows GPT-2 paper)
- **BPE tokenization** via tiktoken — subword vocabulary, handles arbitrary UTF-8 text

## Writing Style for Tutorial Documents
- Audience: undergrad math (basic multivar calc, linear algebra), some ML/DL exposure
- No silly metaphors. Expand math and engineering concepts clearly.
- Pattern: math formula → plain English explanation → PyTorch code implementing it
- **Docs are self-contained**: a reader should be able to build the entire model from scratch
  by following the chapters alone. The Python packages (`babygpt/`, `microgpt/`) are reference
  implementations, not the primary source. Every code listing tagged with `\filetag` must be
  identical to the corresponding code in the source files. When editing code, always update
  both the source files and the matching doc listings.

## Teaching Assistant Mode
When the user asks questions while reading tutorial documents:
1. Pull ./the-annotated-transformer.txt into context. (It is scraped from the Annotated Transformer website and contains all the text and math formulas.)
2. Answer the question in conversation.
3. Record the answer as a **Tip** section inside the relevant chapter file, matching the document's existing writing style and tone.
4. Place the tip near the content that prompted the question so future readers benefit.
5. Make sure documents' code are sync with the source packages.

## Document Structure

### BabyGPT Tutorial (docs/basic/)
- `babygpt-tutorial.tex` — main file (title page, TOC, includes chapters)
- `preamble.tex` — shared LaTeX preamble (also used by advanced tutorial)
- `ch-01-attention.tex` — scaled dot-product attention, multi-head, masking
- `ch-02-transformer-block.tex` — residual connections, layer norm, FFN, block assembly
- `ch-03-full-transformer.tex` — encoder-decoder architecture from original paper
- `ch-04-from-transformer-to-gpt.tex` — decoder-only GPT, what changes and why
- `ch-05-training.tex` — data pipeline, optimizer, LR schedule, training loop
- `ch-06-generation.tex` — autoregressive generation, temperature, top-k

### MicroGPT Tutorial (docs/advanced/)
- `microgpt-tutorial.tex` — main file (title page, TOC, includes chapters)
- `preamble-advanced.tex` — imports shared preamble + advanced-specific commands
- `ch-01-bpe-tokenization.tex` — byte-level BPE, regex pre-tokenization, tiktoken
- `ch-02-scaling-up.tex` — GPT-2 configurations, parameter count, canonical model
- `ch-03-modern-attention.tex` — RoPE, RMSNorm, GQA, QK norm, Flash Attention, relu²
- `ch-04-data-pipeline.tex` — FineWeb-Edu, Parquet, BOS-aligned packing, distributed loading
- `ch-05-distributed-training.tex` — DDP, torchrun, gradient accumulation, mixed precision
- `ch-06-training-at-scale.tex` — optimizer, LR schedule, monitoring, checkpointing, vast.ai
- `ch-07-efficient-inference.tex` — KV cache, prefill/decode, top-p sampling
- `ch-08-evaluation.tex` — perplexity, bits-per-byte, HellaSwag, scaling laws

## Dependencies
- torch (CUDA wheels)
- numpy
- tiktoken (MicroGPT BPE encoding)
- pyarrow (MicroGPT Parquet data loading)
- requests (MicroGPT dataset downloading)

## Common Issues
- If `uv sync` fails on torch, check that the PyTorch index URL matches your CUDA version
- If training OOMs, reduce batch_size or gradient accumulation steps
- BabyGPT checkpoints saved to `checkpoints/`
- MicroGPT checkpoints saved to `microgpt_checkpoints/`
