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
- **Positional encoding (modular)** — `model/position/` holds a beginner-friendly layout for comparing **learned absolute** positions (GPT-style: added to token embeddings) with **RoPE** (rotary embeddings on query/key inside attention). The default model still uses learned embeddings in `gpt.py`; RoPE helpers live in `rope.py` and will be wired through attention in a later step.
- **Training** — CrossEntropy loss, AdamW optimizer, next-token prediction
- **Generation** — Autoregressive sampling with configurable context window

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

# Generate
python generate.py
```

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

Prompts the model with `"h"` and generates 50 tokens via autoregressive sampling. Output is noisy on small datasets—expected behavior.

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
# Run tests
pytest tests/ -v
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
