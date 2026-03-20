# GPT-from-Scratch

A minimal PyTorch implementation of GPT-style components: tokenizers, embeddings, positional encoding, and self-attention.

## Implemented Features

- **Tokenizers**: Character-level (`CharTokenizer`) and word-level (`WordTokenizer`) with `<UNK>` support
- **Token embeddings**: Learned token representations via `nn.Embedding`
- **Positional embeddings**: Learned positional encoding
- **Self-attention**: Single-head scaled dot-product attention (no causal mask yet)

## Project Structure

```
gpt-from-scratch/
├── configs/           # Model configs (gpt2_tiny, gpt2_small)
├── data/             # Sample text for smoke tests
├── model/            # Model components
│   ├── attention.py  # SelfAttention
│   └── embeddings.py # TokenEmbedding, PositionalEmbedding
├── tests/            # Pytest tests
├── utils/
│   └── tokenizer/    # CharTokenizer, WordTokenizer
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run Tests

```bash
python -m pytest tests/ -v
```

## Usage Example

```python
from utils import CharTokenizer, WordTokenizer
from model import TokenEmbedding, PositionalEmbedding, SelfAttention
import torch

# Tokenizer
text = "Hello world"
tok = CharTokenizer(text)
ids = tok.encode("hello")
decoded = tok.decode(ids)  # "hello"

# Embeddings + attention
x = torch.tensor([ids])
token_emb = TokenEmbedding(tok.vocab_size, 64)
pos_emb = PositionalEmbedding(128, 64)
attn = SelfAttention(64)
h = token_emb(x) + pos_emb(x)
out = attn(h)
```
