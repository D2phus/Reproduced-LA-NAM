import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from typing import List 

from backpack import backpack, extend
from backpack.extensions import BatchGrad
from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase

from .derivatives.exu import ExUDerivatives


# make deterministic 
torch.manual_seed(0) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BatchGradExU(BatchGradBase):
    def __init__(self):
        super().__init__(derivatives=ExUDerivatives(), params=["weights", "bias"])
        

class ExU(torch.nn.Module):
    """Custom ExU that supports BackPack's first-order extensions.
    References: 
    -------------
    https://github.com/AmrMKayid/nam/blob/main/nam/models/activation/exu.py
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(ExU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))
        self.n = 1
        self.reset_parameters()

    def reset_parameters(self) -> None:
        ## Page(4): initializing the weights using a normal distribution
        ##          N(x; 0:5) with x 2 [3; 4] works well in practice.
        torch.nn.init.trunc_normal_(self.weights, mean=4.0, std=0.5)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(
        self,
        inputs: torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        self.n = n
        output = (inputs - self.bias).matmul(torch.exp(self.weights))

        # ReLU activations capped at n (ReLU-n)
        output = F.relu(output)
        output = torch.clamp(output, 0, n)

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'