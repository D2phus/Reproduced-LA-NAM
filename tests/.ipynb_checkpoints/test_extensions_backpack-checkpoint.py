import torch
import pytest

from backpack import backpack, extend
from backpack.extensions import BatchGrad
from backpack.extensions.firstorder.base import FirstOrderModuleExtension

import os
import sys
path = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(path, os.pardir)))# add `LANAM` to system paths

from LANAM.models.activation.exu import ExU
from LANAM.extensions.backpack.firstorder.batchgrad import BatchGradExU

@pytest.fixture
def input_size():
    return 4

@pytest.fixture
def output_size():
    return 2

@pytest.fixture
def cls_Xy(input_size, output_size): 
    """classificaiton data"""
    X = torch.randn(10, input_size)
    y = torch.randint(0, output_size, (10,))
    return X, y

@pytest.fixture
def reg_Xy(input_size, output_size):
    """regression data"""
    X = torch.randn(10, input_size)
    y = torch.randn(10, output_size)
    return X, y

@pytest.fixture
def cls_lossfunc():
    reduction = ["mean", "sum"][1]
    return torch.nn.CrossEntropyLoss(reduction=reduction)

@pytest.fixture
def reg_lossfunc():
    reduction = ["mean", "sum"][1]
    return torch.nn.MSELoss(reduction=reduction)

def test_exu_batch_grad_reg(reg_Xy, input_size, output_size, reg_lossfunc): 
    """TOFIX"""
    X, y = reg_Xy
    batch_axis = 0
    
    # register module-computation mapping
    extension = BatchGrad()
    extension.set_module_extension(ExU, BatchGradExU())
    exu = ExU(input_size, output_size)
    
    # using autograd
    grad_batch_autograd = []
    for input_n, target_n in zip(
        X.split(1, dim=batch_axis), y.split(1, dim=batch_axis)
    ):
        loss_n = reg_lossfunc(exu(input_n), target_n)
        grad_n = torch.autograd.grad(loss_n, [exu.bias])[0]
        grad_batch_autograd.append(grad_n)
    grad_batch_autograd = torch.stack(grad_batch_autograd)
    
    # using backpack
    exu = extend(exu)
    reg_lossfunc = extend(reg_lossfunc)
    loss = reg_lossfunc(exu(X), y)
    with backpack(extension):
        loss.backward()
    grad_batch_backpack = exu.bias.grad_batch
    
    torch.allclose(grad_batch_autograd, grad_batch_backpack)

def test_exu_batch_grad_cls(cls_Xy, input_size, output_size, cls_lossfunc): 
    X, y = cls_Xy
    batch_axis = 0
    
    # register module-computation mapping
    extension = BatchGrad()
    extension.set_module_extension(ExU, BatchGradExU())
    exu = ExU(input_size, output_size)
    
    # using autograd
    grad_batch_autograd = []
    for input_n, target_n in zip(
        X.split(1, dim=batch_axis), y.split(1, dim=batch_axis)
    ):
        loss_n = cls_lossfunc(exu(input_n), target_n)
        grad_n = torch.autograd.grad(loss_n, [exu.bias])[0]
        grad_batch_autograd.append(grad_n)
    grad_batch_autograd = torch.stack(grad_batch_autograd)
    
    # using backpack
    exu = extend(exu)
    cls_lossfunc = extend(cls_lossfunc)
    loss = cls_lossfunc(exu(X), y)
    with backpack(extension):
        loss.backward()
    grad_batch_backpack = exu.bias.grad_batch
    
    torch.allclose(grad_batch_autograd, grad_batch_backpack)