import torch
from csa import CompressedSparseAttention
from attention_layer import AttentionDispatcher

def run_test():
    batch_size = 2
    seq_len = 64
    d = 256
    H = torch.randn(batch_size, seq_len, d)

    csa = CompressedSparseAttention()
    csa_out = csa(H)
    assert csa_out.shape == (batch_size, seq_len, d)
    print("CSA module output shape verified.")

    dispatcher_swa = AttentionDispatcher(layer_idx=0)
    assert dispatcher_swa(H).shape == (batch_size, seq_len, d)
    print("Dispatcher routed SWA successfully.")

    dispatcher_csa = AttentionDispatcher(layer_idx=2)
    assert dispatcher_csa(H).shape == (batch_size, seq_len, d)
    print("Dispatcher routed CSA successfully.")

    dispatcher_hca = AttentionDispatcher(layer_idx=3)
    assert dispatcher_hca(H).shape == (batch_size, seq_len, d)
    print("Dispatcher routed HCA successfully.")
    
    print("Phase 3 Passed")

if __name__ == "__main__":
    run_test()