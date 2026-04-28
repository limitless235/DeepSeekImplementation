import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8) * self.weight

class ManifoldConstrainedHyperConnections(nn.Module):
    def __init__(self, d=256, n_hc=4):
        super().__init__()
        self.d = d
        self.n_hc = n_hc
        self.rmsnorm = RMSNorm(n_hc * d)
        
        self.proj_A = nn.Linear(n_hc * d, n_hc)
        self.proj_B = nn.Linear(n_hc * d, n_hc * n_hc)
        self.proj_C = nn.Linear(n_hc * d, n_hc)
        
        self.alpha_pre = nn.Parameter(torch.tensor(1e-3))
        self.alpha_res = nn.Parameter(torch.tensor(1e-3))
        self.alpha_post = nn.Parameter(torch.tensor(1e-3))

    def forward(self, X_l, F_l):
        B, n_hc, T, d = X_l.shape
        
        X_flat = X_l.permute(0, 2, 1, 3).reshape(B, T, n_hc * d)
        X_hat = self.rmsnorm(X_flat)
        
        A_tilde = self.proj_A(X_hat) * self.alpha_pre
        B_tilde = self.proj_B(X_hat) * self.alpha_res
        C_tilde = self.proj_C(X_hat) * self.alpha_post
        
        B_tilde = B_tilde.view(B, T, n_hc, n_hc)
        
        A_l = torch.sigmoid(A_tilde)
        C_l = 2 * torch.sigmoid(C_tilde)
        
        P = torch.exp(B_tilde)
        for _ in range(20):
            P = P / (P.sum(dim=-1, keepdim=True) + 1e-8)
            P = P / (P.sum(dim=-2, keepdim=True) + 1e-8)
        B_l = P
        
        X_l_permuted = X_l.permute(0, 2, 1, 3)
        X_in = (A_l.unsqueeze(-1) * X_l_permuted).sum(dim=2)
        
        F_out = F_l(X_in)
        
        mix = torch.matmul(B_l, X_l_permuted)
        add_term = C_l.unsqueeze(-1) * F_out.unsqueeze(2)
        
        X_out_permuted = mix + add_term
        X_out = X_out_permuted.permute(0, 2, 1, 3)
        
        return X_out
