import torch 

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
        super().__init__(derivatives=ExUDerivatives(), params=["weight", "bias"])
        
