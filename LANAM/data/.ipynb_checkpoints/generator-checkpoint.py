"""Synthetic 1-dimensional example generator"""
import torch 

import math

from typing import Sequence 
from typing import Callable


class Generator:
    """Generator wrapping some synthetic functions."""
    def pow_shape(self, x):
        return 8*torch.pow((x-0.5), 2)

    def exp_shape1(self, x):
        return 0.1 * torch.exp(-8*x+4)

    def exp_shape2(self, x):
        return 5*torch.exp(-2*torch.pow((2*x-1), 2))

    def linear_shape(self, x, slope=0.3, bias=-0.7):
        return slope*x + bias

    def sin_shape(self, x):
        return torch.sin(x)

    def zero_shape(self, x):
        return torch.zeros_like(x)

    def identity_shape(self, x):
        return x

    
grt = Generator()
def synthetic_example(generator=grt):
    return [generator.pow_shape, generator.exp_shape1, generator.exp_shape2, generator.zero_shape]

def sparse_example(generator=grt): 
    return [generator.pow_shape, generator.exp_shape1, generator.exp_shape2, generator.zero_shape, generator.zero_shape, generator.zero_shape, generator.zero_shape, generator.zero_shape]

