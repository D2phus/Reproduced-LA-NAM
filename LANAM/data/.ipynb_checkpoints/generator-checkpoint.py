"""Synthetic 1-dimensional example generator"""
import torch 

import math

from typing import Sequence 
from typing import Callable

def pow_shape(x):
    return 8*torch.pow((x-0.5), 2)

def exp_shape1(x):
    return 0.1 * torch.exp(-8*x+4)

def exp_shape2(x):
    return 5*torch.exp(-2*torch.pow((2*x-1), 2))

def linear_shape(x, slope=0.3, bias=-0.7):
    return slope*x + bias

def zero_shape(x):
    return torch.zeros_like(x)

def task():
    """
    Returns:
    gen_funcs: list of length in_features
    """
    gen_funcs = [pow_shape, exp_shape1, exp_shape2, zero_shape]
    gen_funcs_name = [lambda func=x: func.__name__ for x in gen_funcs]
    return gen_funcs, gen_funcs_name

def sparse_task():
    """
    Returns:
    gen_funcs: list of length in_features
    """
    gen_funcs = [pow_shape, exp_shape1, exp_shape2, zero_shape, zero_shape, zero_shape, zero_shape, zero_shape]
    gen_funcs_name = [lambda func=x: func.__name__ for x in gen_funcs]
    return gen_funcs, gen_funcs_name

class SingleFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.fnn = y.unsqueeze()
        
    def __getitem__(self, index): 
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)