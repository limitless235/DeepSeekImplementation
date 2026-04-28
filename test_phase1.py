import torch
from muon import newton_schulz_10

def run_test():
    M = torch.randn(128, 128)
    M_ortho = newton_schulz_10(M)
    
    S = torch.linalg.svdvals(M_ortho)
    
    max_val = S.max().item()
    min_val = S.min().item()
    
    print("Max singular value:", max_val)
    print("Min singular value:", min_val)
    
    assert torch.allclose(S, torch.ones_like(S), atol=1e-2)
    print("Phase 1 Passed: Singular values are stabilized at 1.0")

if __name__ == "__main__":
    run_test()