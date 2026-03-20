import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        B, T, C = x.shape #batch, time, channles

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        #attention Score
        score = Q @ K.transpose(-2, -1)
        #scale
        score = score/ (C**0.5)
        #softmax
        wights = F.softmax(score, dim=-1)

        #weighted sum 
        out = wights @ V #(B,T,C)
        return out

