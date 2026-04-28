import torch
import torch.nn as nn
from mhc import ManifoldConstrainedHyperConnections
from attention_layer import AttentionDispatcher
from deepseek_moe import DeepSeekMoE

class TransformerBlock(nn.Module):
    def __init__(self, layer_idx, d=256, c=64, n_h=4):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.attn_mhc = ManifoldConstrainedHyperConnections(d=d)
        self.moe_mhc = ManifoldConstrainedHyperConnections(d=d)
        
        self.attn_dispatcher = AttentionDispatcher(layer_idx, d, c, n_h)
        self.moe = DeepSeekMoE(d, layer_idx)

    def forward(self, X_stream, token_ids):
        X_stream = self.attn_mhc(X_stream, self.attn_dispatcher)
        X_stream = self.moe_mhc(X_stream, lambda x: self.moe(x, token_ids))
        return X_stream
