import torch 
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)