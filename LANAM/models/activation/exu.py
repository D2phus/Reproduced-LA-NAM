import torch
import torch.nn as nn
import torch.nn.functional as F

class ExU(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        ) -> None: # type-check
            """
            an exp-centered(ExU) hidden unit, which uses an activation function f (e.g. ReLU-n) to compute h(x) = ReLU_n[exp(w)*(x-b)], where w and b are the weight and bias parameters, and ReLU_n is ReLU capped at n.
            Args:
            in_features: scalar, size of each input sample
            out_feature: scalar, size of each output sample
            """
            super(ExU, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weights, self.bias = self.initialize_parameters()  
            
    def ReLU_n(self, n, x):
        """
        ReLU capped at n
        """
        x = F.relu(x)
        return torch.clamp(x, 0, n)
    
    def initialize_parameters(self)->None:
        """
        Initializing the parameters of the ExU unit, specifically:
        - for weights: normal distribution N(x, 0,5²) with x in [3, 4]
        - for bias: N(0, 0.5²)
        
        note that: 
        1. A variance of 0.5 is introduced in the paper; while a std of 0.5 is used when implementing.
        2. The source code uses truncated normal, but this method has default values for the bound!  
        """     
               
        weights_mean = torch.ones(self.in_features, self.out_features)*4.0
        weights = nn.Parameter(torch.normal(mean=weights_mean, std=0.5))
        
        bias_mean = torch.zeros(self.in_features)
        bias = nn.Parameter(torch.normal(mean=bias_mean, std=0.5))
        
        return weights, bias
             
    def forward(self, 
               inputs: torch.Tensor, 
               n: int = 1
               ) -> torch.Tensor:
        """
        Args:
        inputs of shape (batch_size, in_features)
        
        """
        # note: matrix product!
        # print("exp:", exp_cofficient)
        output = (inputs-self.bias).matmul(torch.exp(self.weights))
        # print("output:", output)
        # ReLU-n
        # relu_n = ReLU_n(n)
        output = self.ReLU_n(n, output)
        # print("relu: ", output)
        return output
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'
    