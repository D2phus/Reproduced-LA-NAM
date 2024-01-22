import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.kernel_approximation import Nystroem

import copy

from typing import List, Tuple, Dict


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
    
    
def hsic(X: torch.Tensor): 
    """measure of (all modes of) dependence with HSIC: 
    
    \sum_{i=1\cdots d} \sum_{j=i+1\cdots d} HSIC(f_i(X_i), f_j(X_j)). 
    """
    if X.ndim == 1: 
        raise ValueError('X should be of shape (batch_size, in_features)')
        
    in_features = X.shape[1]
    if in_features == 1:
        raise ValueError('There should be at least one variables.')
    
    pairwise_h = list()
    for i in range(in_features): 
        for j in range(i+1, in_features): 
            pairwise_h.append(biased_hsic_expectation_form(X[:, i], X[:, j]))
            
    return sum(pairwise_h) / (in_features*(in_features-1)/2)
    
    
def rbf(X: torch.tensor, scale: float=1) -> torch.tensor: 
    """RBF kernel. 
    Args: 
    ---
    X of shape (batch_size, 1)
    scale: length_scale. 
    """
    if X.ndim == 1: 
        # dimension check -> (batch_size, 1)
        X = X.unsqueeze(1)

    KX = torch.exp(-torch.square(X - X.T)/(2*scale**2))
    
    return KX

    
def biased_hsic_expectation_form(X: torch.tensor, Y: torch.tensor) -> torch.tensor:
    """biased HSIC between two random variables X and Y, expectation/sampling form."""
    kX = rbf(X)
    kY = rbf(Y)
    
    N = kX.shape[0] # the number of samples 
    
    hsic = torch.trace(kX@kY) / N**2 # 1/N**2 k(xi, xj)k(yi, yj), for i = 1 to n, j = 1 to n
    hsic += kX.mean() * kY.mean()
    # hsic -= 2* torch.mean(kX@kY) / N 
    hsic -= 2 * kX.mean(dim=1)@kY.mean(dim=1) / N # equivalent to above but more intuitive  
    
    return hsic


def biased_hsic_matrix_form(X: torch.tensor, Y: torch.tensor) -> torch.tensor: 
    """biased HSIC, matrix form."""
    kX = rbf(X)
    kY = rbf(Y)
    
    N = kX.shape[0] 
    
    centering = torch.eye(N) - 1/N # the centering matrix
    
    hsic = torch.trace(kX@centering@kY@centering) / N**2
    
    return hsic

    
def nystrom_approximation(X: torch.tensor, M: int=500): 
    N = len(X)
    X_tilde = copy.deepcopy(X[torch.randperm(N)[:M], :]) # a subsample of size M
    
    kX_tilde = rbf(X_tilde)
    
    
def biased_hsic_matrix_nystrom_approximation(X: torch.tensor, Y: torch.tensor, scale: float=1) -> torch.tensor: 
    """biased HSIC using nystrom approximation for covariance matrix K. 
    nystrom approximation describes the covariance matrix in terms of a smaller number of points, M \ll N. 
    disadvantage: no guarantee on convergence. """
    N = len(X)
    
    
    
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





def concurvity(X: torch.Tensor, eps: float=1e-12) -> torch.Tensor: 
    """measure of (linear) correlation with Pearson correlation coefficient.  
    
    \sum_{i=1\cdots d} \sum_{j=i+1\cdots d} p(f_i(X_i), f_j(X_j)). 
    Args: 
    ----
    X of shape (batch_size, num_variables)
        feature batch.
    eps: float
        when computing the Pearson correlation coefficient, a small value is added to the standard deviation to avoid divide zero issue. 
    """
    in_features = X.shape[1]
    cov = pairwise_correlation(X, eps)
    
    R = torch.triu(cov.abs(), diagonal=1).sum()
    R /= (in_features*(in_features-1)/2)
    return R


def pairwise_correlation(X: torch.Tensor, eps: float=1e-12) -> torch.Tensor: 
    '''normalized covariance matrix for a batch of features X.
    Args:
    ---- 
    X of shape (batch_size, num_variables)
        feature batch.
    eps: float
        a small value is added to the standard deviation to avoid divide zero issue. 
        
    Returns: 
    -----
    correlation matrix of shape (num_variables, num_variablesa)
    '''
    batch_size, in_features = X.shape 
    if batch_size < 2 or in_features < 2: 
        # no sufficient samples or features for correlation matrix  
        return torch.zeros_like(X)
    
    std = X.std(dim=0)
    std = std*std.reshape(-1, 1)
    cov = torch.cov(X.T) # (unnormalized) covariance matrix. NOTE torch.cov and torch.coeff needs data of shape (num_variables, batch_size)
    cov /= (std + eps) # correlation matrix
    cov = torch.where(std==0.0, 0.0, cov)   
    
    return cov
