import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)


    def forward(self, x):
        return self.embedding(x)






class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super().__init__()
        self.positional_embedding = nn.Embedding(max_len, embedding_dim)

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return self.positional_embedding(positions)

