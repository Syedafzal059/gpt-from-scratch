"""Tests for CharTokenizer and WordTokenizer."""

from utils.tokenizer import CharTokenizer, WordTokenizer


def test_char_tokenizer_encode_decode_roundtrip():
    """CharTokenizer encode/decode roundtrip returns original string."""
    text = "hello world"
    tokenizer = CharTokenizer(text)
    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(encoded)
    assert decoded == "hello"


def test_char_tokenizer_vocab_size():
    """CharTokenizer vocab size equals number of unique chars."""
    text = "hello"
    tokenizer = CharTokenizer(text)
    assert tokenizer.vocab_size == len(set(text))


def test_word_tokenizer_encode_decode_roundtrip():
    """WordTokenizer encode/decode roundtrip preserves words."""
    text = "Hello Afzal This is GPT"
    tokenizer = WordTokenizer(text)
    encoded = tokenizer.encode("Hello GPT")
    decoded = tokenizer.decode(encoded)
    assert decoded == "Hello GPT"


def test_word_tokenizer_unk_handles_unknown():
    """WordTokenizer maps unknown words to <UNK> index."""
    text = "Hello world"
    tokenizer = WordTokenizer(text)
    encoded = tokenizer.encode("Hello xyz unknown")
    assert all(isinstance(t, int) for t in encoded)
    decoded = tokenizer.decode(encoded)
    assert "<UNK>" in decoded
