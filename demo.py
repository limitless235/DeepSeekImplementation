import torch
from model import DeepSeekV4Toy
from muon import Muon

def run_demo():
    torch.manual_seed(42)
    model = DeepSeekV4Toy()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameter Count: {total_params}")
    
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 512, (batch_size, seq_len))
    targets = torch.randint(0, 512, (batch_size, seq_len))
    
    base_logits, total_loss = model(input_ids, targets)
    print(f"Forward Pass Output Shape: {base_logits.shape}")
    
    optimizer = Muon(model.named_parameters(), lr=1e-3)
    
    print("\nStarting Training...")
    for step in range(1, 11):
        optimizer.zero_grad()
        logits, loss = model(input_ids, targets)
        loss.backward()
        optimizer.step()
        print(f"Step {step} | Loss: {loss.item():.4f}")
        
    print("\nEvaluating Muon Orthogonalization...")
    for group in optimizer.param_groups:
        if group.get('use_muon', False):
            sample_weight = group['params'][0]
            S = torch.linalg.svdvals(sample_weight)
            print(f"Sample Weight Shape: {sample_weight.shape}")
            print(f"Max Singular Value: {S.max().item():.4f}")
            print(f"Min Singular Value: {S.min().item():.4f}")
            print(f"Mean Singular Value: {S.mean().item():.4f}")
            break

if __name__ == "__main__":
    run_demo()
