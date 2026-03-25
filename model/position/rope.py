import torch 
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        #torch.arange(0,dim, 2).float() -> [0, 2, 4, ..., dim-2][x1,x2,x3,x4]->(x1, x2), (x3, x4), ... = (2i)
        #si​=1/10000^(2i/dim)
        
        self.inv_freq = 1.0/(base**(torch.arange(0,dim, 2).float()/dim))
        #different speed for different dimensions

    def get_angels(self, seq_len, device):
        #output(θ)=position×speed(θp,i​=p⋅si​)
        
        #create positions
        #torch.arange(start, end, step)
        positions = torch.arange(seq_len, device=device).float()
        
        #compute angles
        #multiplication + summation using index notation
        angles = torch.einsum("i,j->ij", positions, self.inv_freq)
        return angles
    
    def get_sin_cos(self, seq_len, device):
        angles = self.get_angels(seq_len, device)
        
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        #Each pair = one rotation
        #repeat_interleave = "apply same rotation to both components"
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        cos = torch.repeat_interleave(cos, 2, dim=-1)

        return sin, cos