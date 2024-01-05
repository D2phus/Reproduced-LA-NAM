import numpy as np 
import torch

def Ld_norm(X: torch.Tensor, d: int=1)->torch.Tensor:
        """
        L1/L2 norm of X. 
        X: (batch_size, in_features)
        """
        N = len(X) # batch_size
        if d == 2:
            ld = torch.mean(torch.square(X), 1).sum() / N
        elif d == 1: 
            ld = torch.mean(torch.abs(X), 1).sum() / N
        return ld
    
def estimate_hsic(X: torch.Tensor): 
    """X: (batch_size, in_features)"""
    in_features = X.shape[1]
    
    rbf_X = [rbf(X[:, i]) for i in range(in_features)]
    
    pairwise_h = list()
    for i in range(in_features): 
        for j in range(i, in_features): 
            pairwise_h.append(hsic(rbf_X[i], rbf_X[j]))
            
    return sum(pairwise_h) / (in_features*(in_features-1)/2)
    

def hsic(kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor: 
    """biased HSIC.
    https://github.com/clovaai/rebias/blob/master/criterions/hsic.py
    kx: kernel x
    ky: kernel y"""
    # kxh = kx - kx.mean(0, keepdim=True)
    # kyh = ky - ky.mean(0, keepdim=True)
    
    # N = kx.shape[0]
    
    # return torch.trace(kxh @ kyh / (N-1) ** 2)
    tx = kx - torch.diag(kx)
    ty = ky - torch.diag(ky)
    
    N = kx.shape[0]
    
    h = torch.trace(tx @ ty) 
    + (torch.sum(tx) * torch.sum(ty) / (N - 1) / (N - 2)) 
    - (2 * torch.sum(tx, 0).dot(torch.sum(ty, 0)) / (N - 2))
    
    return h / (N * (N - 3))
    # kxy = torch.matmul(kx, ky)
    # N = kxy.shape[0]
    # h = torch.trace(kxy) / N**2 + kx.mean() * ky.mean() - 2 * kxy.mean() / N
    
    # return h * N**2 / (N - 1)**2

def unbias_hsic(kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor: 
    """unbias HSIC. 
    https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=c0c0d7b1e4da2a61f62a1d1e0df85ea4e3a932f3"""
    tx = kx - torch.diag(kx)
    ty = ky - torch.diag(ky)
    
    N = kx.shape[0]
    
    h = torch.trace(tx @ ty) 
    + (torch.sum(tx) * torch.sum(ty) / (N - 1) / (N - 2)) 
    - (2 * torch.sum(tx, 0).dot(torch.sum(ty, 0)) / (N - 2))
    
    return h / (N * (N - 3))
    
def rbf(x: torch.Tensor, l: float=1) -> torch.Tensor:
    """RBF kernel with var = 1
    x: (batch_size, 1)"""
    
    if x.ndim == 1: 
        x = x.unsqueeze(1)
        
    kx = x - torch.squeeze(x)
    kx = torch.exp(-kx**2/(2*(l**2)))
    return kx 