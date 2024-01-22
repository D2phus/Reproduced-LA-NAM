import os
import sys
path = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(path, os.pardir)))# add `LANAM` to system paths

import torch
import pytest 
from sklearn.gaussian_process.kernels import RBF
from LANAM.utils.regularizer import *


@pytest.mark.parametrize('x', [(torch.randn(10))])
def test_rbf(x):
    kernel = RBF()
    
    k1 = rbf(x)
    k2 = kernel(x.unsqueeze(1))

    torch.testing.assert_close(k1, torch.Tensor(k2))
    
    
@pytest.mark.parametrize('x,y', [(torch.randn(1000), torch.randn(1000))])
def test_biased_hsic(x, y):
    """verify if the biased HSIC given by the expectation form and matrix form are closed"""
    H_E = biased_hsic_expectation_form(x, y)

    H_M = biased_hsic_matrix_form(x, y)

    torch.testing.assert_close(H_E, H_M)