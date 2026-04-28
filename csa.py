import math
import torch
import torch.nn as nn
from token_compressor import TokenCompressor
from swa import RMSNorm, apply_rope

class CompressedSparseAttention(nn.Module):
    def __init__(self, d=256, c=64, n_h=4):
        super().__init__()
        self.d = d
        self.c = c
        self.n_h = n_h
        self.n_win = 16
        
        self.compressor = TokenCompressor(d, c, mode='overlapped')
        self.indexer_compressor = TokenCompressor(d, 32, mode='overlapped')
        
        self.W_DQ = nn.Linear(d, 64, bias=False)
        self.W_IUQ = nn.Linear(64, 4 * 32, bias=False)
        self.W_w = nn.Linear(64, 4, bias=False)
        
        self.W_Q = nn.Linear(d, n_h * c, bias=False)
        self.W_KV = nn.Linear(d, c, bias=False)
        
        self.rmsnorm_q = nn.ModuleList([RMSNorm(c) for _ in range(n_h)])
        self.rmsnorm_k = RMSNorm(c)
        self.rmsnorm_v = RMSNorm(c)
        
        self.proj_g0 = nn.Linear(2 * c, 64)
        self.proj_g1 = nn.Linear(2 * c, 64)
        self.proj_out = nn.Linear(128, d)

    def forward(self, H):
        B, T, d_dim = H.shape
        
        H_dc = self.W_DQ(H)
        Q_I = self.W_IUQ(H_dc).view(B, T, 4, 32)
        W_I = self.W_w(H_dc).unsqueeze(-1)
        
        K_I = self.indexer_compressor(H)
        
        dot = torch.einsum('bthc,bsc->bths', Q_I, K_I)
        relu_dot = torch.relu(dot)
        I_scores = W_I * relu_dot
        I_ts = I_scores.sum(dim=2)
        
        mask_causal_I = torch.full((T, T // 4), -float('inf'), device=H.device)
        for t in range(T):
            for s in range(T // 4):
                if 4 * s <= t:
                    mask_causal_I[t, s] = 0.0
                    
        I_ts_masked = I_ts + mask_causal_I.unsqueeze(0)
        
        k = min(8, T // 4)
        if k > 0:
            topk_vals, topk_idx = torch.topk(I_ts_masked, k, dim=-1)
            M_comp = torch.full_like(I_ts, -float('inf'))
            M_comp.scatter_(-1, topk_idx, 0.0)
        else:
            M_comp = torch.full_like(I_ts, -float('inf'))
            
        KV_comp = self.compressor(H)
        KV_sliding = self.W_KV(H)
        
        KV = torch.cat([KV_comp, KV_sliding], dim=1).unsqueeze(1)
        
        Q = self.W_Q(H).view(B, T, self.n_h, self.c).permute(0, 2, 1, 3)
        
        for h in range(self.n_h):
            Q[:, h, :, :] = self.rmsnorm_q[h](Q[:, h, :, :])
        K = self.rmsnorm_k(KV)
        V = self.rmsnorm_v(KV)
        
        Q = apply_rope(Q, negate=False)
        K = apply_rope(K, negate=False)
        V = apply_rope(V, negate=False)
        
        S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.c)
        
        S_comp = S[:, :, :, :T//4] + M_comp.unsqueeze(1)
        
        M_sliding = torch.full((T, T), -float('inf'), device=H.device)
        for t in range(T):
            start = max(0, t - self.n_win + 1)
            M_sliding[t, start : t + 1] = 0.0
            
        S_sliding = S[:, :, :, T//4:] + M_sliding.unsqueeze(0).unsqueeze(0)
        
        S_masked = torch.cat([S_comp, S_sliding], dim=-1)
        P = torch.softmax(S_masked, dim=-1)
        O = torch.matmul(P, V)
        
        O = apply_rope(O, negate=True)
        
        O_g0 = O[:, :2, :, :].permute(0, 2, 1, 3).flatten(-2)
        O_g1 = O[:, 2:, :, :].permute(0, 2, 1, 3).flatten(-2)
        
        p0 = self.proj_g0(O_g0)
        p1 = self.proj_g1(O_g1)
        
        p_all = torch.cat([p0, p1], dim=-1)
        return self.proj_out(p_all)
