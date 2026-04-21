import torch 
import torch.nn as nn
from model.multi_head_attention import MultiHeadAttention
from model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn  = FeedForward(embed_dim)
        self.ln1  = nn.LayerNorm(embed_dim)
        self.ln2  = nn.LayerNorm(embed_dim) 


    def forward(self, x, past_kv = None, position_offset=0):

        attn_out, present_kv = self.attn(
            self.ln1(x),
            past_kv = past_kv,
            position_offset=position_offset,
        )
        # Attention =Residual
        x = x+ attn_out
        # FFN + Residual
        x = x+ self.ffn(self.ln2(x))
        return x, present_kv


if __name__ == "__main__":
    B, T, C = 2, 5, 8
    num_heads = 2

    x = torch.randn(B, T, C)

    block = TransformerBlock(C, num_heads)
    out, kv = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("K shape:", kv[0].shape, "V shape:", kv[1].shape)