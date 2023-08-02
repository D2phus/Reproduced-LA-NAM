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

def get_linear_examples(num_samples=200, 
                        slope=0.1,
                        bias=0.2,
                        sigma=0.2, 
                        batch_size=64):
    """
    generate regression dataset from function ax + b with observation centered Gaussian noise ~ N(0, sigma^2)
    
    Args:
    num_samples: the number of training samples
    slope, bias: the slope and bias for the linear function
    sigma: the standard deviation of observation noise
    batch_size
    
    Returns: 
    X_train of shape (num_samples): training samples uniformly sampled on interval [0, 8]
    y_train of shape (num_samples): training labels with noise
    
    train_loader: data loader for the training data
    X_test of shape (500): test samples which are evenly spaced between -5 and 13
    """
    X_train = (torch.rand(num_samples) * 8).unsqueeze(-1)
    y_train = slope*X_train + bias + torch.randn_like(X_train) * sigma
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size
    )
    X_test = torch.linspace(-5, 13, 500).unsqueeze(-1)
    return X_train, y_train, train_loader, X_test



def get_sin_examples(num_samples=200, 
                    sigma=0.2,
                    batch_size=64):
    """
    generate regression dataset from function \sin(X) with observation centered Gaussian noise ~ N(0, sigma^2)
    
    Args:
    num_samples: the number of training samples
    sigma: the standard deviation of observation noise
    batch_size
    
    Returns: 
    X_train of shape (num_samples): training samples uniformly sampled on interval [0, 8]
    y_train of shape (num_samples): training labels with noise
    train_loader: data loader for the training data
    X_test of shape (500): test samples which are evenly spaced between -5 and 13
    """
    X_train = (torch.rand(num_samples) * 8).unsqueeze(-1)
    y_train = torch.sin(X_train) + torch.randn_like(X_train) * sigma
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size
    )
    X_test = torch.linspace(-0.5, 8.5, 200).unsqueeze(-1)
    return X_train, y_train, train_loader, X_test
