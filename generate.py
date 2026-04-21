import torch
from model.gpt import GPTModel
from utils.tokenizer import CharTokenizer


def generate(
    model,
    tokenizer,
    start_text,
    max_new_tokens=50,
    block_size=32,
    device="cpu",
    use_kv_cache=True,
    verbose=False,
):
    model.eval()

    # Step 1: Convert text → tokens
    tokens = tokenizer.encode(start_text)
    # Example: "Afz" → [3, 5, 7]
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    # shape: (1, T)

    past = None  # KV cache
    for step in range(max_new_tokens):
        if use_kv_cache and x.shape[1] <= block_size:
            # First step → no cache yet
            if past is None:
                # Send full input
                logits, past = model(
                    x,
                    past_kv_list=None,
                    position_offset=0,
                    use_cache=True,
                )
            else:
                # Next steps → only send last token
                logits, past = model(
                    x[:, -1:],         #only new token
                    past_kv_list=past, # reuse old kv
                    position_offset=x.shape[1] - 1,
                    use_cache=True,
                )
        else:
            context = x[:, -block_size:]  # Sliding window: only last block_size tokens
            logits = model(context)  # (1, T, vocab_size)
            past = None

        #Take last token prediction
        logits = logits[:, -1, :] #(1, vocab_size)

        #Convert to probabilities
        probs = torch.softmax(logits, dim=-1) #(1, vocab_size)

        #Sample next token
        next_token = torch.multinomial(probs, num_samples=1) #(1, 1)

        #Append to input
        x = torch.cat([x, next_token], dim=1) #(1, T+1)

        if verbose:
            print(f"Step {step}: sequence =", x.tolist())

    # Decode output
    output_tokens = x[0].tolist()
    return tokenizer.decode(output_tokens)





if __name__ == "__main__":
    # Config must match train.py
    text = "hello Afzal this is a simple dataset for training a gpt model from "
    block_size = 32
    embed_dim = 64
    num_heads = 4
    num_layers = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CharTokenizer(text)
    model = GPTModel(tokenizer.vocab_size, embed_dim, num_heads, num_layers, block_size)
    model.to(device)

    output = generate(
        model,
        tokenizer,
        start_text="Afz",
        max_new_tokens=5,
        block_size=block_size,
        device=device,
        verbose=True,
    )

    print(output)