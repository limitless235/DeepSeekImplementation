import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_block import TransformerBlock
from mhc import RMSNorm

class LightweightBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d*4), nn.SiLU(), nn.Linear(d*4, d))
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
    
    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class DeepSeekV4Toy(nn.Module):
    def __init__(self, vocab_size=512, d=256, n_hc=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d)
        
        self.blocks = nn.ModuleList([TransformerBlock(i, d=d) for i in range(6)])
        
        self.rmsnorm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        
        self.mtp_proj = nn.Linear(2 * d, d)
        self.mtp_block = LightweightBlock(d)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        
        X = self.embedding(input_ids)
        X_stream = X.unsqueeze(1).expand(-1, 4, -1, -1).clone()
        
        for block in self.blocks:
            X_stream = block(X_stream, input_ids)
            
        X_collapsed = X_stream.mean(dim=1)
        X_norm = self.rmsnorm(X_collapsed)
        base_logits = self.lm_head(X_norm)
        
        if targets is not None:
            loss_base = F.cross_entropy(base_logits[:, :-1].reshape(-1, 512), targets[:, 1:].reshape(-1))
            
            collapsed_t = X_collapsed[:, :-1, :]
            target_emb_t1 = self.embedding(targets[:, 1:])
            mtp_in = torch.cat([collapsed_t, target_emb_t1], dim=-1)
            
            mtp_h = self.mtp_proj(mtp_in)
            mtp_h = self.mtp_block(mtp_h)
            
            mtp_logits = self.lm_head(self.rmsnorm(mtp_h))
            
            loss_mtp = F.cross_entropy(mtp_logits[:, :-1].reshape(-1, 512), targets[:, 2:].reshape(-1))
            
            balance_loss = 0.0
            for block in self.blocks:
                balance_loss += block.moe.balance_loss
                
            total_loss = loss_base + 0.3 * loss_mtp + balance_loss
            return base_logits, total_loss
            
        return base_logits
