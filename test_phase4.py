import torch
import torch.nn as nn
from mhc import ManifoldConstrainedHyperConnections
from deepseek_moe import DeepSeekMoE

def run_test():
    batch_size = 2
    n_hc = 4
    seq_len = 64
    d = 256
    
    X_stream = torch.randn(batch_size, n_hc, seq_len, d)
    dummy_inner_layer = nn.Linear(d, d)

    mhc = ManifoldConstrainedHyperConnections()
    mhc_out = mhc(X_stream, dummy_inner_layer)
    assert mhc_out.shape == (batch_size, n_hc, seq_len, d)
    print("mHC stream expansion and folding verified.")

    token_ids = torch.randint(0, 512, (batch_size, seq_len))
    
    moe_hash = DeepSeekMoE(layer_idx=0)
    single_stream_input = X_stream[:, 0, :, :]
    moe_hash_out = moe_hash(single_stream_input, token_ids)
    assert moe_hash_out.shape == (batch_size, seq_len, d)
    print("MoE Hash routing execution verified.")

    moe_learned = DeepSeekMoE(layer_idx=3)
    moe_learned_out = moe_learned(single_stream_input, token_ids)
    assert moe_learned_out.shape == (batch_size, seq_len, d)
    
    moe_learned.update_biases()
    print("MoE Learned routing and bias update verified.")
    print("Phase 4 Passed")

if __name__ == "__main__":
    run_test()