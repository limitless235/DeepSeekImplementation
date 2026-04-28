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
            
            compressed_blocks = []
            for i in range(T // m):
                Z_a_slice = Z_a[:, m*i : m*(i+1), :] + self.B_a
                if i == 0:
                    Z_b_slice = torch.full((B_sz, m, self.c), -float('inf'), device=H.device, dtype=H.dtype)
                    C_b_slice = torch.zeros((B_sz, m, self.c), device=H.device, dtype=H.dtype)
                else:
                    Z_b_slice = Z_b[:, m*(i-1) : m*i, :] + self.B_b
                    C_b_slice = C_b[:, m*(i-1) : m*i, :]
                
                C_a_slice = C_a[:, m*i : m*(i+1), :]
                
                logits = torch.cat([Z_a_slice, Z_b_slice], dim=1)
                probs = torch.softmax(logits, dim=1)
                
                vals = torch.cat([C_a_slice, C_b_slice], dim=1)
                comp_i = (probs * vals).sum(dim=1)
                compressed_blocks.append(comp_i)
                
            return torch.stack(compressed_blocks, dim=1)
            
        elif self.mode == 'non-overlapped':
            m_prime = 16
            C = H @ self.W_KV
            Z = H @ self.W_Z
            
            compressed_blocks = []
            for j in range(T // m_prime):
                Z_slice = Z[:, m_prime*j : m_prime*(j+1), :] + self.B
                probs = torch.softmax(Z_slice, dim=1)
                C_slice = C[:, m_prime*j : m_prime*(j+1), :]
                comp_j = (probs * C_slice).sum(dim=1)
                compressed_blocks.append(comp_j)
                
            return torch.stack(compressed_blocks, dim=1)
