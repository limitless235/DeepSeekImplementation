import torch
from token_compressor import TokenCompressor
from swa import PureSlidingWindowAttention
from hca import HeavilyCompressedAttention

def run_test():
    batch_size = 2
    seq_len = 64
    d = 256
    H = torch.randn(batch_size, seq_len, d)

    tc_csa = TokenCompressor(mode='csa')
    csa_out = tc_csa(H)
    assert csa_out.shape == (batch_size, seq_len // 4, 64)
    print("CSA Compressor output shape verified.")

    tc_hca = TokenCompressor(mode='hca')
    hca_out = tc_hca(H)
    assert hca_out.shape == (batch_size, seq_len // 16, 64)
    print("HCA Compressor output shape verified.")

    swa = PureSlidingWindowAttention()
    swa_out = swa(H)
    assert swa_out.shape == (batch_size, seq_len, d)
    print("SWA output shape verified.")

    hca = HeavilyCompressedAttention()
    hca_out_final = hca(H)
    assert hca_out_final.shape == (batch_size, seq_len, d)
    print("HCA full module output shape verified.")
    print("Phase 2 Passed")

if __name__ == "__main__":
    run_test()