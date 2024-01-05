import torch
import pytest 
from sklearn.gaussian_process.kernels import RBF
from LANAM.utils.hsic import *


def test_rbf():
    x = torch.randn(100, 1)
    kernel = RBF()
    
    k1 = rbf(x)
    k2 = kernel(x)

    torch.testing.assert_close(k1, torch.Tensor(k2))