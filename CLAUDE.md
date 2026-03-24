# Baby GPT from Scratch

## Project Overview
Educational project: build a GPT language model from scratch using plain PyTorch.
Based on "Attention Is All You Need" (Vaswani et al., 2017) and Karpathy's nanoGPT.
Character-level language model trained on TinyShakespeare (~1.1M chars, 65 unique chars).

## Project Structure
- `learn/` — 6 markdown documents explaining theory (01-06, ordered)
- `src/` — Python package with model, tokenizer, dataset, training, generation
- `data/` — TinyShakespeare dataset
- `CLAUDE.md` — this file

## Tech Stack
- Python 3.14, managed by uv
- PyTorch (with CUDA, targeting modern NVIDIA GPUs)
- No other ML frameworks — plain PyTorch only

## Key Commands
```bash
./build.sh                                                         # rebuild PDFs
uv sync                                                            # install deps
uv run python -m src.dataset                                       # download TinyShakespeare + print stats
uv run python -m src.train                                         # train the model (~14 min on RTX 3080)
uv run python -m src.generate --prompt "ROMEO:" --temperature 0.8  # generate text
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
- Audience: undergrad math (basic multivar calc, linear algerba), some ML/DL exposure
- No silly metaphors. Expand math and engineering concepts clearly.
- Pattern: math formula → plain English explanation → PyTorch code implementing it
- Each document is self-contained but builds on previous ones

## Teaching Assistant Mode
When the user asks questions while reading `learn/` documents:
1. Pull ./the-annotated-transformer.txt into context. (It is scraped from the Annotated Transformer website and contains all the text and math formulas.)
2. Answer the question in conversation.
3. Record the answer as a **Tip** section inside the relevant `learn/` document, matching the document's existing writing style and tone.
4. Place the tip near the content that prompted the question so future readers benefit.
5. Make sure documents' code are sync with `src/`.

## Document Progression
1. `01-attention-from-first-principles.md` — scaled dot-product attention, multi-head, masking
2. `02-the-transformer-block.md` — residual connections, layer norm, FFN, block assembly
3. `03-the-full-transformer.md` — encoder-decoder architecture from original paper
4. `04-from-transformer-to-gpt.md` — decoder-only GPT, what changes and why
5. `05-training.md` — data pipeline, optimizer, LR schedule, training loop
6. `06-generation-and-sampling.md` — autoregressive generation, temperature, top-k

## Dependencies
- torch (CUDA wheels)
- numpy

## Common Issues
- If `uv sync` fails on torch, check that the PyTorch index URL matches your CUDA version
- If training OOMs, reduce batch_size in src/train.py
- Model checkpoints saved to `checkpoints/` directory
