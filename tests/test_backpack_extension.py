import torch
import pytest

from backpack import backpack, extend
from backpack.extensions import BatchGrad
from backpack.extensions.firstorder.base import FirstOrderModuleExtension

import os
import sys
sys.path.append(os.getcwd()) # add `LANAM` to system paths

from LANAM.models.activation.exu import ExU
from LANAM.models.extended_laplace.curvature.extensions import BatchGradExU


@pytest.fixture
def input_size():
    return 4

@pytest.fixture
def output_size():
    return 2

@pytest.fixture
def cls_Xy(input_size, output_size): 
    X = torch.randn(10, input_size)
    y = torch.randint(0, output_size, (10,))
    return X, y

@pytest.fixture
def reg_Xy(input_size, output_size):
    X = torch.randn(batch_size, input_size)
    y = torch.randn(batch_size, output_size)
    return X, y

@pytest.fixture
def lossfunc():
    reduction = ["mean", "sum"][1]
    return torch.nn.CrossEntropyLoss(reduction=reduction)

def test_exu_batch_grad_cls(cls_Xy, input_size, output_size, lossfunc): 
    X, y = cls_Xy
    batch_axis = 0
    
    # register module-computation mapping
    extension = BatchGrad()
    extension.set_module_extension(ExU, BatchGradExU())
    exu = ExU(input_size, output_size)
    
    # using autograd
    grad_batch_autograd = []
    for input_n, target_n in zip(
        inputs.split(1, dim=batch_axis), targets.split(1, dim=batch_axis)
    ):
        loss_n = lossfunc(exu(input_n), target_n)
        grad_n = torch.autograd.grad(loss_n, [exu.bias])[0]
        grad_batch_autograd.append(grad_n)
    grad_batch_autograd = torch.stack(grad_batch_autograd)
    
    # using backpack
    exu = extend(exu)
    lossfunc = extend(lossfunc)
    loss = lossfunc(exu(inputs), targets)
    with backpack(extension):
        loss.backward()
    grad_batch_backpack = exu.bias.grad_batch
    
    assert grad_batch_autograd == grad_batch_backpack