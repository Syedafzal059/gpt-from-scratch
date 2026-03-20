"""Tests for TokenEmbedding and PositionalEmbedding."""

import torch

from model.embeddings import PositionalEmbedding, TokenEmbedding


def test_token_embedding_shape():
    """TokenEmbedding outputs (B, T, embedding_dim)."""
    vocab_size, embedding_dim = 20, 8
    batch_size, seq_len = 2, 5
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    token_emb = TokenEmbedding(vocab_size, embedding_dim)
    out = token_emb(x)
    assert out.shape == (batch_size, seq_len, embedding_dim)


def test_positional_embedding_shape():
    """PositionalEmbedding outputs (B, T, embedding_dim)."""
    max_len, embedding_dim = 10, 8
    batch_size, seq_len = 2, 5
    x = torch.randint(0, 10, (batch_size, seq_len))
    pos_emb = PositionalEmbedding(max_len=max_len, embedding_dim=embedding_dim)
    out = pos_emb(x)
    assert out.shape == (batch_size, seq_len, embedding_dim)


def test_combined_embeddings_shape():
    """Token + positional embeddings can be combined (same shape)."""
    vocab_size, embedding_dim, max_len = 20, 8, 10
    batch_size, seq_len = 2, 5
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    token_emb = TokenEmbedding(vocab_size, embedding_dim)
    pos_emb = PositionalEmbedding(max_len=max_len, embedding_dim=embedding_dim)
    combined = token_emb(x) + pos_emb(x)
    assert combined.shape == (batch_size, seq_len, embedding_dim)
