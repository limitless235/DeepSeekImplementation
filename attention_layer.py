import torch
import torch.nn as nn
from swa import PureSlidingWindowAttention
from csa import CompressedSparseAttention
from hca import HeavilyCompressedAttention

class AttentionDispatcher(nn.Module):
    def __init__(self, layer_idx, d=256, c=64, n_h=4):
        super().__init__()
        self.layer_idx = layer_idx
        
        if layer_idx in [0, 1]:
            self.attn = PureSlidingWindowAttention(d, c, n_h)
        elif layer_idx in [2, 4]:
            self.attn = CompressedSparseAttention(d, c, n_h)
        elif layer_idx in [3, 5]:
            self.attn = HeavilyCompressedAttention(d, c, n_h)
        else:
            self.attn = PureSlidingWindowAttention(d, c, n_h)

    def forward(self, X):
        return self.attn(X)
