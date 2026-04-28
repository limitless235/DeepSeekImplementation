import torch
import torch.nn as nn

class TokenCompressor(nn.Module):
    def __init__(self, d=256, c=64, mode='overlapped'):
        super().__init__()
        if mode == 'csa':
            mode = 'overlapped'
        elif mode == 'hca':
            mode = 'non-overlapped'
        self.mode = mode
        self.d = d
        self.c = c
        if mode == 'overlapped':
            self.W_a_KV = nn.Parameter(torch.randn(d, c))
            self.W_b_KV = nn.Parameter(torch.randn(d, c))
            self.W_a_Z = nn.Parameter(torch.randn(d, c))
            self.W_b_Z = nn.Parameter(torch.randn(d, c))
            self.B_a = nn.Parameter(torch.randn(4, c))
            self.B_b = nn.Parameter(torch.randn(4, c))
        elif mode == 'non-overlapped':
            self.W_KV = nn.Parameter(torch.randn(d, c))
            self.W_Z = nn.Parameter(torch.randn(d, c))
            self.B = nn.Parameter(torch.randn(16, c))

    def forward(self, H):
        B_sz, T, d_dim = H.shape
        if self.mode == 'overlapped':
            m = 4
            C_a = H @ self.W_a_KV
            C_b = H @ self.W_b_KV
            Z_a = H @ self.W_a_Z
            Z_b = H @ self.W_b_Z
            
            C_a_blocks = C_a.view(B_sz, T // m, m, self.c)
            Z_a_blocks = Z_a.view(B_sz, T // m, m, self.c) + self.B_a.unsqueeze(0).unsqueeze(0)
            
            C_b_blocks = torch.zeros_like(C_a_blocks)
            if T // m > 1:
                C_b_blocks[:, 1:, :, :] = C_b[:, :-m, :].reshape(B_sz, T // m - 1, m, self.c)
                
            Z_b_blocks = torch.full_like(Z_a_blocks, -float('inf'))
            if T // m > 1:
                Z_b_reshaped = Z_b[:, :-m, :].reshape(B_sz, T // m - 1, m, self.c)
                Z_b_blocks[:, 1:, :, :] = Z_b_reshaped + self.B_b.unsqueeze(0).unsqueeze(0)
                
            logits = torch.cat([Z_a_blocks, Z_b_blocks], dim=2)
            probs = torch.softmax(logits, dim=2)
            
            vals = torch.cat([C_a_blocks, C_b_blocks], dim=2)
            comp = (probs * vals).sum(dim=2)
            return comp
            
        elif self.mode == 'non-overlapped':
            m_prime = 16
            C = H @ self.W_KV
            Z = H @ self.W_Z
            
            C_blocks = C.view(B_sz, T // m_prime, m_prime, self.c)
            Z_blocks = Z.view(B_sz, T // m_prime, m_prime, self.c) + self.B.unsqueeze(0).unsqueeze(0)
            
            probs = torch.softmax(Z_blocks, dim=2)
            comp = (probs * C_blocks).sum(dim=2)
            return comp
