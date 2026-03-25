# Baby GPT from Scratch

## Project Overview
Educational project: build a GPT language model from scratch using plain PyTorch.
Based on "Attention Is All You Need" (Vaswani et al., 2017) and Karpathy's nanoGPT.
Character-level language model trained on TinyShakespeare (~1.1M chars, 65 unique chars).

## Project Structure
- `learn/` — LaTeX tutorial (6 chapters assembled into one PDF via `babygpt-tutorial.tex`)
- `babygpt/` — Python package with model, tokenizer, dataset, training, generation
- `data/` — TinyShakespeare dataset
- `CLAUDE.md` — this file

## Tech Stack
- Python 3.14, managed by uv
- PyTorch (with CUDA, targeting modern NVIDIA GPUs)
- No other ML frameworks — plain PyTorch only

## Key Commands
```bash
./build.sh                                                                # rebuild PDF
uv sync                                                                   # install deps
uv run python -m babygpt.dataset                                          # download TinyShakespeare + print stats
uv run python -m babygpt.train                                            # train the model (~14 min on RTX 3080)
uv run python -m babygpt.generate --prompt "ROMEO:" --temperature 0.8     # generate text
```

## Model Config (BabyGPT)
- 6 layers, 6 heads, 384 embedding dim, 256 context window
- ~10.7M parameters, character-level tokenization (vocab_size=65)
- Dropout 0.2, no bias in linear layers

## Training Config
- Batch size 64, 5000 iterations, AdamW (lr=1e-3, wd=0.1)
- Cosine LR schedule with 100-step warmup
- Mixed precision (float16), gradient clipping at 1.0
- Expected val loss ~1.47, training time ~14 min on RTX 3080

## Architecture Decisions
- **Pre-norm** (LayerNorm before sublayer) — matches GPT-2 and modern practice
- **GELU** activation in FFN
- **Learned positional embeddings** (not sinusoidal)
- **Weight tying** between token embedding and output head
- **bias=False** in linear layers — modern convention (GPT-J, LLaMA)
- **Character-level** tokenization — zero external deps, fully visible pipeline

## Writing Style for learn/ Documents
- Audience: undergrad math (basic multivar calc, linear algebra), some ML/DL exposure
- No silly metaphors. Expand math and engineering concepts clearly.
- Pattern: math formula → plain English explanation → PyTorch code implementing it
- **Docs are self-contained**: a reader should be able to build the entire GPT from scratch by following the 6 chapters alone — `babygpt/` is the reference implementation, not the primary source. Every code listing tagged with `\filetag` must be identical to the corresponding code in `babygpt/`. When editing code, always update both the source files and the matching doc listings.

## Teaching Assistant Mode
When the user asks questions while reading `learn/` documents:
1. Pull ./the-annotated-transformer.txt into context. (It is scraped from the Annotated Transformer website and contains all the text and math formulas.)
2. Answer the question in conversation.
3. Record the answer as a **Tip** section inside the relevant `learn/` chapter file, matching the document's existing writing style and tone.
4. Place the tip near the content that prompted the question so future readers benefit.
5. Make sure documents' code are sync with `babygpt/`.

## Document Structure
The tutorial is a single PDF compiled from separate chapter files:
- `learn/babygpt-tutorial.tex` — main file (title page, TOC, includes chapters)
- `learn/preamble.tex` — shared LaTeX preamble
- `learn/ch-01-attention.tex` — scaled dot-product attention, multi-head, masking
- `learn/ch-02-transformer-block.tex` — residual connections, layer norm, FFN, block assembly
- `learn/ch-03-full-transformer.tex` — encoder-decoder architecture from original paper
- `learn/ch-04-from-transformer-to-gpt.tex` — decoder-only GPT, what changes and why
- `learn/ch-05-training.tex` — data pipeline, optimizer, LR schedule, training loop
- `learn/ch-06-generation.tex` — autoregressive generation, temperature, top-k

## Dependencies
- torch (CUDA wheels)
- numpy

## Common Issues
- If `uv sync` fails on torch, check that the PyTorch index URL matches your CUDA version
- If training OOMs, reduce batch_size in babygpt/train.py
- Model checkpoints saved to `checkpoints/` directory
