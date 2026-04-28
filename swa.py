import math
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8) * self.weight

def apply_rope(x, negate=False):
    B, H, T, C = x.shape
    x_rope = x[..., -16:]
    x_pass = x[..., :-16]
    
    t = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    if negate:
        t = -t
        
    k = torch.arange(8, device=x.device, dtype=x.dtype)
    theta = 10000 ** (-2 * k / 16)
    theta = theta.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    freqs = t * theta
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    x1 = x_rope[..., 0::2]
    x2 = x_rope[..., 1::2]
    
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    
    rx = torch.stack([rx1, rx2], dim=-1).flatten(-2)
    return torch.cat([x_pass, rx], dim=-1)

class PureSlidingWindowAttention(nn.Module):
    def __init__(self, d=256, c=64, n_h=4):
        super().__init__()
        self.d = d
        self.c = c
        self.n_h = n_h
        self.n_win = 16
        
        self.W_Q = nn.Linear(d, n_h * c, bias=False)
        self.W_K = nn.Linear(d, c, bias=False)
        self.W_V = nn.Linear(d, c, bias=False)
        
        self.rmsnorm_q = nn.ModuleList([RMSNorm(c) for _ in range(n_h)])
        self.rmsnorm_k = RMSNorm(c)
        self.rmsnorm_v = RMSNorm(c)
        
        self.proj_g0 = nn.Linear(2 * c, 64)
        self.proj_g1 = nn.Linear(2 * c, 64)
        self.proj_out = nn.Linear(128, d)

    def forward(self, H):
        B, T, d_dim = H.shape
        
        Q = self.W_Q(H).view(B, T, self.n_h, self.c).permute(0, 2, 1, 3)
        K = self.W_K(H).view(B, T, 1, self.c).permute(0, 2, 1, 3)
        V = self.W_V(H).view(B, T, 1, self.c).permute(0, 2, 1, 3)
        
        for h in range(self.n_h):
            Q[:, h, :, :] = self.rmsnorm_q[h](Q[:, h, :, :])
        K = self.rmsnorm_k(K)
        V = self.rmsnorm_v(V)
        
        Q = apply_rope(Q, negate=False)
        K = apply_rope(K, negate=False)
        V = apply_rope(V, negate=False)
        
        S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.c)
        
        mask = torch.full((T, T), -float('inf'), device=H.device)
        for t in range(T):
            start = max(0, t - self.n_win + 1)
            mask[t, start : t + 1] = 0.0
            
        S = S + mask.unsqueeze(0).unsqueeze(0)
        P = torch.softmax(S, dim=-1)
        O = torch.matmul(P, V)
        
        O = apply_rope(O, negate=True)
        
        O_g0 = O[:, :2, :, :].permute(0, 2, 1, 3).flatten(-2)
        O_g1 = O[:, 2:, :, :].permute(0, 2, 1, 3).flatten(-2)
        
        p0 = self.proj_g0(O_g0)
        p1 = self.proj_g1(O_g1)
        
        p_all = torch.cat([p0, p1], dim=-1)
        return self.proj_out(p_all)
