import math
import torch
import torch.nn as nn
from token_compressor import TokenCompressor
from swa import RMSNorm, apply_rope

class HeavilyCompressedAttention(nn.Module):
    def __init__(self, d=256, c=64, n_h=4):
        super().__init__()
        self.d = d
        self.c = c
        self.n_h = n_h
        self.n_win = 16
        
        self.compressor = TokenCompressor(d, c, mode='non-overlapped')
        self.W_Q = nn.Linear(d, n_h * c, bias=False)
        
        self.rmsnorm_q = nn.ModuleList([RMSNorm(c) for _ in range(n_h)])
        self.rmsnorm_k = RMSNorm(c)
        self.rmsnorm_v = RMSNorm(c)
        
        self.proj_g0 = nn.Linear(2 * c, 64)
        self.proj_g1 = nn.Linear(2 * c, 64)
        self.proj_out = nn.Linear(128, d)

    def forward(self, H):
        B, T, d_dim = H.shape
        
        Q = self.W_Q(H).view(B, T, self.n_h, self.c).permute(0, 2, 1, 3)
        
        H_comp = self.compressor(H)
        H_sliding = H[:, -self.n_win:, :] @ self.compressor.W_KV
        
        KV = torch.cat([H_comp, H_sliding], dim=1).unsqueeze(1)
        
        for h in range(self.n_h):
            Q[:, h, :, :] = self.rmsnorm_q[h](Q[:, h, :, :])
        K = self.rmsnorm_k(KV)
        V = self.rmsnorm_v(KV)
        
        Q = apply_rope(Q, negate=False)
        K = apply_rope(K, negate=False)
        V = apply_rope(V, negate=False)
        
        S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.c)
        P = torch.softmax(S, dim=-1)
        O = torch.matmul(P, V)
        
        O = apply_rope(O, negate=True)
        
        O_g0 = O[:, :2, :, :].permute(0, 2, 1, 3).flatten(-2)
        O_g1 = O[:, 2:, :, :].permute(0, 2, 1, 3).flatten(-2)
        
        p0 = self.proj_g0(O_g0)
        p1 = self.proj_g1(O_g1)
        
        p_all = torch.cat([p0, p1], dim=-1)
        return self.proj_out(p_all)
