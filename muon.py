import math
import torch
from torch.optim.optimizer import Optimizer

def newton_schulz_10(M):
    M = M / (torch.linalg.norm(M) + 1e-12)
    n, m = M.shape
    transposed = False
    if n > m:
        X = M.T
        transposed = True
    else:
        X = M

    for i in range(1, 11):
        if i <= 8:
            a, b, c = 3.445, -4.7750, 2.0315
        else:
            a, b, c = 2.0, -1.5, 0.5
        
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        
    if transposed:
        return X.T
    return X

class Muon(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, betas=betas, eps=eps)
        
        muon_params = []
        adamw_params = []
        
        for item in params:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                name, p = item
                name_lower = name.lower()
                if p.ndim >= 2 and not any(k in name_lower for k in ["embed", "head", "norm", "bias", "gating"]):
                    muon_params.append(p)
                else:
                    adamw_params.append(p)
            else:
                p = item if not isinstance(item, dict) else item.get('params', item)
                if isinstance(p, list):
                    for sub_p in p:
                        if sub_p.ndim >= 2:
                            muon_params.append(sub_p)
                        else:
                            adamw_params.append(sub_p)
                elif isinstance(p, torch.Tensor):
                    if p.ndim >= 2:
                        muon_params.append(p)
                    else:
                        adamw_params.append(p)
        
        param_groups = [
            {'params': muon_params, 'use_muon': True},
            {'params': adamw_params, 'use_muon': False}
        ]
        
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon = group.get('use_muon', False)
            lr = group['lr']
            weight_decay = group['weight_decay']

            if not use_muon:
                beta1, beta2 = group['betas']
                eps = group['eps']
                
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    grad = p.grad
                    state = self.state[p]
                    
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        
                    state['step'] += 1
                    
                    if weight_decay != 0:
                        p.mul_(1.0 - lr * weight_decay)
                        
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    
                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    
                    p.addcdiv_(exp_avg, denom, value=-step_size)
            
            else:
                momentum = group['momentum']
                
                for p in group['params']:
                    if p.grad is None:
                        continue
                        
                    grad = p.grad
                    state = self.state[p]
                    
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                        
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    
                    O_prime_t = buf.mul(momentum).add(grad)
                    
                    O_prime_t.div_(torch.linalg.norm(O_prime_t) + 1e-12)
                    
                    M_10 = newton_schulz_10(O_prime_t)
                    
                    n, m = p.shape
                    gamma = 0.18
                    
                    O_t = M_10 * gamma * max(n, m)**0.5
                    
                    if weight_decay != 0:
                        p.mul_(1.0 - lr * weight_decay)
                        
                    p.add_(O_t, alpha=-lr)

        return loss
