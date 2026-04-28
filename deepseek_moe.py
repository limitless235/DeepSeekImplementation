import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d)
        )
    def forward(self, x):
        return self.net(x)

class DeepSeekMoE(nn.Module):
    def __init__(self, d=256, layer_idx=0):
        super().__init__()
        self.d = d
        self.layer_idx = layer_idx
        self.N_experts = 8
        self.k = 2
        hidden_dim = (4 * d) // self.N_experts
        
        self.shared_expert = Expert(d, hidden_dim)
        self.routed_experts = nn.ModuleList([Expert(d, hidden_dim) for _ in range(self.N_experts)])
        
        if self.layer_idx in [3, 4, 5]:
            self.routing_W = nn.Linear(d, self.N_experts, bias=False)
            self.register_buffer("expert_bias", torch.zeros(self.N_experts))
            
        self.balance_loss = 0.0

    def forward(self, X, token_ids=None):
        B, T, d = X.shape
        out = self.shared_expert(X)
        self.balance_loss = 0.0
        
        if self.layer_idx in [0, 1, 2]:
            idx1 = token_ids % self.N_experts
            idx2 = (token_ids + 1) % self.N_experts
            
            routed_out = torch.zeros_like(X)
            
            for i, expert in enumerate(self.routed_experts):
                mask1 = (idx1 == i)
                mask2 = (idx2 == i)
                mask = mask1 | mask2
                if mask.any():
                    expert_out = expert(X[mask])
                    routed_out[mask] += 0.5 * expert_out
            
            out = out + routed_out
            
        else:
            logits = self.routing_W(X) + self.expert_bias
            activation = torch.sqrt(F.softplus(logits))
            
            top_vals, top_idx = torch.topk(activation, self.k, dim=-1)
            
            routed_out = torch.zeros_like(X)
            expert_counts = torch.zeros(self.N_experts, device=X.device)
            
            for i, expert in enumerate(self.routed_experts):
                mask = (top_idx == i).any(dim=-1)
                if mask.any():
                    mask_i0 = (top_idx[..., 0] == i)
                    mask_i1 = (top_idx[..., 1] == i)
                    
                    scores_0 = top_vals[..., 0][mask_i0]
                    scores_1 = top_vals[..., 1][mask_i1]
                    
                    if mask_i0.any():
                        expert_out_0 = expert(X[mask_i0])
                        routed_out[mask_i0] += scores_0.unsqueeze(-1) * expert_out_0
                    if mask_i1.any():
                        expert_out_1 = expert(X[mask_i1])
                        routed_out[mask_i1] += scores_1.unsqueeze(-1) * expert_out_1
                        
                    expert_counts[i] = mask.sum().float()
                    
            out = out + routed_out
            
            f_i = expert_counts / (B * T * self.k + 1e-8)
            P_i = torch.softmax(logits, dim=-1).mean(dim=(0, 1))
            self.balance_loss = 0.0001 * self.N_experts * torch.sum(f_i * P_i)
            
            self.last_expert_counts = expert_counts
            
        return out

    def update_biases(self):
        if self.layer_idx in [3, 4, 5] and hasattr(self, 'last_expert_counts'):
            avg_count = self.last_expert_counts.mean()
            for i in range(self.N_experts):
                if self.last_expert_counts[i] > avg_count:
                    self.expert_bias[i] -= 0.001
                elif self.last_expert_counts[i] < avg_count:
                    self.expert_bias[i] += 0.001
