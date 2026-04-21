# GPT From Scratch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A minimal GPT implementation built from first principles—no high-level abstractions. Understand transformer architecture by implementing every component yourself.

---

## Overview

| | |
|---|---|
| **Goal** | Implement GPT architecture from scratch: tokenizer → attention → transformer blocks → training → generation |
| **Stack** | PyTorch, Python 3.8+ |
| **Output** | Autoregressive text generation with learnable next-token prediction |

---

## Features

- **Tokenizer** — Char-level and word-level tokenization
- **Attention** — Scaled dot-product attention, multi-head attention
- **Transformer** — Pre-norm blocks, FFN, positional embeddings
- **KV cache (inference)** — Per-layer key/value caching for autoregressive decoding; optional in `generate.py` (`use_kv_cache`). Training path unchanged (`use_cache=False` by default).
- **Positional encoding (modular)** — `model/position/` holds a beginner-friendly layout for comparing **learned absolute** positions (GPT-style: added to token embeddings) with **RoPE** (rotary embeddings on query/key inside attention). The default model still uses learned embeddings in `gpt.py`; RoPE helpers live in `rope.py` and will be wired through attention in a later step.
- **Training** — CrossEntropy loss, AdamW optimizer, next-token prediction
- **Generation** — Autoregressive sampling with configurable context window; cached or full-recompute paths

---

## KV cache (inference)

### Problem

Without a cache, each new token requires a forward pass over the **entire** prefix. Attention recomputes key/value projections for all past tokens even though those activations are fixed for prior positions.

### Approach

- **`MultiHeadAttention`** accepts optional `past_kv=(K, V)` and returns `present_kv` after concatenating new keys/values along the sequence dimension. Causal masking uses **global** key/query positions so cached prefixes remain valid.
- **`TransformerBlock`** threads `past_kv` and `position_offset` through attention only (FFN unchanged).
- **`GPTModel`** accepts `past_kv_list` (one `(K, V)` per layer), `position_offset` (global index of `x[:, 0]`), and `use_cache`. When `use_cache=True`, returns `(logits, presents)`.

### API (summary)

| Symbol | Role |
|--------|------|
| `past_kv` / `present_kv` | Tuple `(K, V)` per attention layer; shapes `(B, num_heads, T, head_dim)` |
| `past_kv_list` | List of length `num_layers`, one entry per block |
| `position_offset` | Global start index for positions in this forward (e.g. `0` for full prompt; `L` when forwarding only the token at index `L`) |
| `use_cache` | `False` in training; `True` when decoding with cache |

### Constraints

- Positional embeddings are indexed by `position_offset + local_index`. Ensure **`position_offset + T ≤ max_len`** (constructor `max_len`, often equal to `block_size` in scripts).
- The cached path in **`generate.py`** is used when `use_kv_cache` is on **and** sequence length `≤ block_size`. If the sequence grows past `block_size`, generation falls back to a sliding window **without** cache (same behavior as before for long contexts).

### Correctness

`tests/test_kv_cache.py` asserts that **last-step logits** from a **full forward** on `prompt ∥ next_token` match **prefill + one cached step** (numerical parity).

```bash
python -m pytest tests/test_kv_cache.py -v
```

### Inference benchmark (example)

`tests/test_kv_cache_metrics.py` compares the same workload with KV cache on vs off (512 new tokens, `block_size=4096`, CPU in this run). Your numbers will vary by hardware; the screenshot below is one representative run.

![KV cache vs no-cache benchmark terminal output](docs/images/kv-cache-benchmark.png)

| | KV cache ON | KV cache OFF |
|---|-------------|--------------|
| **Total time** | 535 ms | 1920 ms |
| **Per new token** | ~1.05 ms | ~3.75 ms |
| **Speedup (no cache / cache)** | **~3.6×** | — |

Reproduce the report:

```bash
pytest tests/test_kv_cache_metrics.py::test_kv_cache_timing_metrics_report -v -s
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/Syedafzal059/gpt-from-scratch.git
cd gpt-from-scratch

# Environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

# Train (default: 200 epochs on sample text)
python train.py

# Generate (KV cache on by default; see generate.py)
python generate.py
```

---

## Git workflow (contributors)

Conventional commits and short-lived feature branches keep history readable.

| Convention | Example |
|------------|---------|
| Branch | `feat/kv-cache`, `fix/tokenizer-edge-case` |
| Commit | `feat(kv-cache): add cached decode path` |
| Scope | Lowercase; hyphenate multi-word scopes |

**Suggested commands for a KV-cache change:**

```bash
git checkout main
git pull origin main
git checkout -b feat/kv-cache
# ... implement ...
git add model/ generate.py tests/test_kv_cache.py README.md
git commit -m "feat(kv-cache): add inference KV cache with parity test"
git push -u origin feat/kv-cache
```

Open a PR into `main` with a short summary and test evidence (e.g. pytest output).

---

## Project Structure

```
gpt-from-scratch/
├── model/
│   ├── attention.py          # Scaled dot-product attention
│   ├── embeddings.py         # Token + positional embedding helpers
│   ├── multi_head_attention.py
│   ├── transformer_block.py  # Pre-norm block + FFN
│   ├── gpt.py                # Full model (token + learned position)
│   └── position/             # Modular positional encoding (learned vs RoPE)
│       ├── __init__.py
│       ├── learned.py        # Learned absolute positions (to align with gpt.py)
│       ├── rope.py           # RotaryEmbedding: angles, sin/cos (RoPE building blocks)
│       └── types.py          # Shared enums / small types (optional)
├── tests/
│   ├── test_kv_cache.py          # KV cache vs full-forward logit parity
│   ├── test_kv_cache_metrics.py  # Timing / speedup; same output with fixed seed
│   └── test_position_rope.py # Tests for position / RoPE (expand as you wire it)
├── utils/tokenizer/
│   ├── char_tokenizer.py
│   └── word_tokenizer.py
├── dataset.py                # Next-token prediction dataset
├── train.py
├── generate.py
└── configs/                  # Model configs (YAML)
```

---

## Usage

### Training

```bash
python train.py
```

Default hyperparameters (editable in `train.py`):

| Param | Value |
|-------|-------|
| `block_size` | 32 |
| `embed_dim` | 64 |
| `num_heads` | 4 |
| `num_layers` | 2 |
| `learning_rate` | 3e-4 |
| `epochs` | 200 |

Loss typically converges from ~3.0 → ~0.05 on the included sample corpus.

### Generation

```bash
python generate.py
```

Uses `CharTokenizer` and the hyperparameters in the `__main__` block. **`use_kv_cache`** (default `True`) enables prefill + single-token forwards when `len(sequence) ≤ block_size`; otherwise the loop uses the last `block_size` tokens without cache. Set `use_kv_cache=False` in code to force full recomputation each step (slower, useful for debugging).

Output quality depends on training data and model size—small demos look noisy; that is expected.

---

## Positional encoding (two ideas, two places)

For teaching, it helps to separate **where** position enters the model:

| Style | Where it applies | Idea |
|-------|------------------|------|
| **Learned absolute** (current default) | After token embedding in `gpt.py` | Each position index gets a learned vector, added to the token vector. |
| **RoPE** (scaffolding in `model/position/rope.py`) | On **Q** and **K** inside attention | No position vector on the main stream; rotations encode position in the attention scores. |

Do not use both at full strength on the same run without meaning to—that would double-count position. The `position/` package is meant to grow into a simple switch (e.g. enum + small factory) so beginners can flip modes in one place.

---

## Architecture

```
Input tokens → Token Embedding → + Positional Embedding (learned, default)
       → [Transformer Block × N] → LayerNorm → Linear(vocab_size)
       → Logits
```

With RoPE fully wired (future), the first line becomes **token embedding only**, and position is applied inside each attention layer to Q and K.

**Model Flow**

```
Input (B, T) → Embedding (B, T, C) → Transformer Blocks → LayerNorm → Linear Head → Logits (B, T, V)
```

| Phase | Components |
|-------|-------------|
| **1** | Tokenizer (char/word), embeddings, scaled dot-product attention, multi-head attention |
| **2** | Transformer block (pre-norm), FFN, stacked blocks, token + positional embeddings, output head |
| **3** | Dataset pipeline, training loop (CrossEntropy + AdamW), autoregressive generation |

---

## Development

```bash
# Full test suite
pytest tests/ -v

# KV cache parity only
pytest tests/test_kv_cache.py -v

# Timing + speedup report (print metrics: pytest -s; uses 512 new tokens, ~seconds on CPU)
pytest tests/test_kv_cache_metrics.py -v -s

# Skip the slow timing benchmark; still runs the same-text KV vs no-cache test
pytest tests/test_kv_cache_metrics.py -v -m "not slow"
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- numpy, pyyaml, tqdm

CUDA optional for GPU acceleration.

---

## License

MIT
