"""Tests for SelfAttention module."""

import torch

from model.attention import SelfAttention


def test_self_attention_preserves_shape():
    """SelfAttention output shape matches input (B, T, C)."""
    batch_size, seq_len, embedding_dim = 2, 5, 8
    x = torch.randn(batch_size, seq_len, embedding_dim)
    model = SelfAttention(embedding_dim=embedding_dim)
    out = model(x)
    assert out.shape == (batch_size, seq_len, embedding_dim)


def test_self_attention_deterministic_with_eval():
    """SelfAttention produces same output for same input in eval mode."""
    torch.manual_seed(42)
    x = torch.randn(1, 4, 16)
    model = SelfAttention(embedding_dim=16)
    model.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1, out2)
