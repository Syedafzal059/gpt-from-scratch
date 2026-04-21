"""
Compare KV-cache vs no-cache generation: correctness (same output under fixed seed)
and quantified timing (wall time, ms/token, speedup). Optional CUDA memory.
"""

import time

import torch

import pytest

from generate import generate
from model.gpt import GPTModel
from utils.tokenizer import CharTokenizer

# Long benchmark: sequence length must stay ≤ block_size or generate() drops KV cache.
# Larger max_new_tokens → no-cache path does many full forwards over growing T → speedup shows up.
BENCH_BLOCK_SIZE = 4096
BENCH_MAX_NEW_TOKENS = 512


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_generate(model, tokenizer, *, device, block_size, use_kv_cache, max_new_tokens, start_text, seed):
    torch.manual_seed(seed)
    _sync_cuda()
    t0 = time.perf_counter()
    with torch.no_grad():
        text = generate(
            model,
            tokenizer,
            start_text=start_text,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            device=device,
            use_kv_cache=use_kv_cache,
            verbose=False,
        )
    _sync_cuda()
    elapsed = time.perf_counter() - t0
    return text, elapsed


@pytest.fixture(scope="module")
def bench_setup():
    """Small model + tokenizer; block_size large enough for long KV-cache runs."""
    corpus = "hello Afzal this is a simple dataset for training a gpt model from scratch"
    tokenizer = CharTokenizer(corpus)
    block_size = BENCH_BLOCK_SIZE
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPTModel(
        tokenizer.vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        block_size,
    ).to(device)
    model.eval()
    return model, tokenizer, device, block_size


def test_cached_vs_uncached_same_text_under_fixed_seed(bench_setup):
    """With the same RNG seed, sampling should match if logits match step by step."""
    model, tokenizer, device, block_size = bench_setup
    start_text = "Afz"
    max_new_tokens = 8
    seed = 42

    torch.manual_seed(seed)
    out_cached = generate(
        model,
        tokenizer,
        start_text=start_text,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        device=device,
        use_kv_cache=True,
        verbose=False,
    )
    torch.manual_seed(seed)
    out_no_cache = generate(
        model,
        tokenizer,
        start_text=start_text,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        device=device,
        use_kv_cache=False,
        verbose=False,
    )
    assert out_cached == out_no_cache, "Decoded text should match when logits path is equivalent."


@pytest.mark.slow
def test_kv_cache_timing_metrics_report(bench_setup):
    """
    Quantified comparison: wall time, ms per new token, speedup (no_cache / cache).
    Uses many new tokens so the no-cache path (full forward each step) pays O(T) per step.
    Prints a short report (use `pytest -s` to see in terminal).

    Skip with: pytest -m "not slow"
    """
    model, tokenizer, device, block_size = bench_setup
    start_text = "hello"
    max_new_tokens = BENCH_MAX_NEW_TOKENS
    seed = 123

    # Warmup (especially CUDA)
    _time_generate(
        model,
        tokenizer,
        device=device,
        block_size=block_size,
        use_kv_cache=True,
        max_new_tokens=min(16, max_new_tokens),
        start_text=start_text,
        seed=seed,
    )
    _sync_cuda()

    _, t_cached = _time_generate(
        model,
        tokenizer,
        device=device,
        block_size=block_size,
        use_kv_cache=True,
        max_new_tokens=max_new_tokens,
        start_text=start_text,
        seed=seed,
    )
    _sync_cuda()

    _, t_no_cache = _time_generate(
        model,
        tokenizer,
        device=device,
        block_size=block_size,
        use_kv_cache=False,
        max_new_tokens=max_new_tokens,
        start_text=start_text,
        seed=seed,
    )

    ms_per_tok_cached = (t_cached / max_new_tokens) * 1000.0
    ms_per_tok_no = (t_no_cache / max_new_tokens) * 1000.0
    speedup = t_no_cache / t_cached if t_cached > 0 else float("inf")

    cuda_peak_mb = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed)
        _sync_cuda()
        with torch.no_grad():
            generate(
                model,
                tokenizer,
                start_text=start_text,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                device=device,
                use_kv_cache=True,
                verbose=False,
            )
        _sync_cuda()
        cuda_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

    report_lines = [
        "",
        "=== KV cache vs no-cache (same workload) ===",
        f"device:              {device}",
        f"block_size:          {block_size}  (must exceed prompt+new for KV path)",
        f"max_new_tokens:      {max_new_tokens}",
        f"KV cache ON:         {t_cached*1000:.2f} ms total, {ms_per_tok_cached:.3f} ms/token",
        f"KV cache OFF:        {t_no_cache*1000:.2f} ms total, {ms_per_tok_no:.3f} ms/token",
        f"speedup (no/KV):     {speedup:.2f}x  (higher means cache is faster)",
    ]
    if cuda_peak_mb is not None:
        report_lines.append(f"CUDA peak alloc (cached run): {cuda_peak_mb:.2f} MB")
    report_lines.append("==========================================")
    print("\n".join(report_lines))

    # Do not assert speedup: tiny models on CPU can show ~1x or noise.
    assert t_cached >= 0 and t_no_cache >= 0

