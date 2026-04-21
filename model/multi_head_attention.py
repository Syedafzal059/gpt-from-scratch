### Multi Head Ateention Intution 
# Each head learns something different:
# Head 1 → grammar
# Head 2 → long dependencies
# Head 3 → subject-object relations

# 👉 More heads = richer understanding
# 🧩 Core idea

# Instead of:
# (B, T, C)
# You split into:
# (B, T, num_heads, head_dim)
# Where:
# C = num_heads × head_dim
import torch 
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        
        assert embedding_dim % num_heads == 0

        self.num_head = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, past_kv=None, position_offset=0):
        """
        x: (B, T_new, C)
        past_kv: None or (K_past, V_past), each (B, num_heads, T_past, head_dim)
        position_offset: global index of x[:, 0, :] in the full sequence.
        Returns:
            out: (B, T_new, C)
            present_kv: (K, V) each (B, num_heads, T_past + T_new, head_dim)
        """
        B, T_new, C = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)        

        # reshape for multi head
        Q = Q.view(B, T_new, self.num_head, self.head_dim).transpose(1,2)
        K = K.view(B, T_new, self.num_head, self.head_dim).transpose(1,2)
        V = V.view(B, T_new, self.num_head, self.head_dim).transpose(1,2)


        #kv cache 
        if past_kv is not None:
            K_past, V_past = past_kv
            K = torch.cat([K_past, K], dim=2)
            V = torch.cat([V_past, V], dim=2)
        T_past = 0 if past_kv is None else past_kv[0].shape[2] 
        T_total = T_past + T_new





        # In PyTorch, the @ operator (matrix multiplication) ALWAYS works on the "LAST TWO" dimensions.
        # Q  = (B, num_heads, T, head_dim)
        # K  = (B, num_heads, T, head_dim)
        # K.transpose(-2, -1) → (B, num_heads, head_dim, T)
        # (T × head_dim) @ (head_dim × T) → (T × T)
        scores = Q @ K.transpose(-2,-1)
        scores = scores /(self.head_dim**0.5)

        device = x.device
        key_global = torch.empty(T_total, device=device, dtype=torch.long)
        if T_past > 0:
            key_global[:T_past]=torch.arange(T_past, device=device)
        if T_new >0:
            key_global[T_past:] = position_offset + torch.arange(T_new, device=device)
        query_global = position_offset + torch.arange(T_new, device=device)

        causal_mask = key_global.unsqueeze(0) <= query_global.unsqueeze(-1)
        scores = scores.masked_fill(~causal_mask, float("-inf"))



        weights = F.softmax(scores, dim=-1)
        out = weights @ V #(B, num_head, T, head_dim)
        #merge heads
        #transpose = rearranging view
        out = out.transpose(1,2) #(B, num_head, T, head_dim) -> (B, T, num_head, head_dim)
        # Before:
        #   [[ [1,2], [7,8] ]]  → (num_heads=2, head_dim=2)
        # After reshape:
        #   [1,2,7,8] → (num_heads*head_dim=4)
        out = out.reshape(B, T_new, C) #(B, T, num_head, head_dim) -> (B, T, C)
        present_kv = (K,V)
        return self.out(out), present_kv  #output = W × input + b, now W are learnable parameters


if __name__ =="__main__":
    x = torch.randn(2, 5, 8)

    model = MultiHeadAttention(embedding_dim=8, num_heads=2)
    out, _ = model(x)

    print(out.shape)  # should be (2, 5, 8)
