"""KV cache: cached decode step must match full forward on the same prefix + token."""

import torch

from model.gpt import GPTModel


def test_kv_cache_last_logits_match_full_forward():
    """Prefill + one cached step equals one full forward on prompt + next token."""
    torch.manual_seed(0)
    vocab_size = 50
    embed_dim = 32
    num_heads = 4
    num_layers = 2
    max_len = 64

    model = GPTModel(vocab_size, embed_dim, num_heads, num_layers, max_len)
    model.eval()

    prompt_len = 6
    x_prompt = torch.randint(0, vocab_size, (1, prompt_len))
    new_token = torch.randint(0, vocab_size, (1, 1))
    x_full = torch.cat([x_prompt, new_token], dim=1)

    with torch.no_grad():
        logits_full = model(x_full, use_cache=False)
        logits_expected = logits_full[:, -1, :]

        _, past = model(
            x_prompt,
            past_kv_list=None,
            position_offset=0,
            use_cache=True,
        )
        logits_cached, _ = model(
            new_token,
            past_kv_list=past,
            position_offset=prompt_len,
            use_cache=True,
        )
        logits_actual = logits_cached[:, -1, :]

    torch.testing.assert_close(logits_actual, logits_expected, rtol=1e-5, atol=1e-6)
