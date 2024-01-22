import os
import sys
path = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(path, os.pardir)))# add `LANAM` to system paths

import torch
import pytest 
from LANAM.trainer import *

import math

testdata = [
    (torch.Tensor([[4, 4, 3], 
                   [5, 4, 1]]),
    torch.Tensor([[1, 4, 5], 
                   [1, 2, 7]]), 
    torch.Tensor([12.5000,  2.0000, 20.0000]).sqrt()), 
    (torch.Tensor([[4, 4, 3], 
                   [5, 4, 1]]),
    torch.Tensor([[4, 4, 3], 
                   [5, 4, 1]]),
    torch.zeros(3)), 
]


@pytest.mark.parametrize('preds,targets,rmsed_truth', testdata)
def test_rmse_d(preds, targets, rmsed_truth):
    rmsed_pred = rmse_d(preds, targets)
    
    torch.testing.assert_close(rmsed_pred, rmsed_truth)