import torch
import torch.nn as nn
from model.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_len, embed_dim)

        # ModuleList: Converts normal Python list → PyTorch-aware list
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)


    def forward(self, x, past_kv_list=None, position_offset=0, use_cache=False):
        B, T = x.shape
        
        # Token Embedding 
        tok_emb = self.token_embedding(x)

        # Positional Embedding
        pos = torch.arange(T, device=x.device) + position_offset
        pos_emb = self.positional_embedding(pos)

        x = tok_emb + pos_emb #(B, T, C)

        # Transformer block 
        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_i = past_kv_list[i] if past_kv_list is not None else None
            x, present_kv = block(x, past_kv=past_i, position_offset=position_offset)
            if use_cache:
                presents.append(present_kv)
        #Final norms
        x = self.ln_f(x)
        logits = self.head(x)

        if use_cache:
            return logits, presents
        
        #Output Logits

        return logits



if __name__ == "__main__":
    vocab_size = 100
    embed_dim = 32
    num_heads = 4
    num_layers = 2
    max_len = 50

    model = GPTModel(vocab_size, embed_dim, num_heads, num_layers, max_len)

    x = torch.randint(0, vocab_size, (2, 10))  # (B, T)

    logits = model(x)

    print("Input:", x.shape)
    print("Output:", logits.shape) # torch.Size([2, 10, 100])