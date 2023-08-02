import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearReLU(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        ) -> None: # type-check
            """
            Standard linear ReLU hidden unit.
            in_features: scalar, the size of input sample
            out_feature: scalar, the size of output sample
            """
            super(LinearReLU, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weights = nn.Parameter(torch.Tensor(in_features, out_features))
            self.bias = nn.Parameter(torch.Tensor(in_features)) # note; a bias for each feature
            
            self.initialize_parameters()  
            
            # TODO: how about bias? - bias ~ N(0, 0.5^2)
    def initialize_parameters(self):
        """
        Initializing the parameters. 
        - weights: regular, xavier uniform(why uniform?)
        - bias: N(0, 0.5²)
        """
        nn.init.xavier_uniform_(self.weights)
        #torch.nn.init.trunc_normal_(self.bias, std=0.5) # note 
        
        bias_mean = torch.zeros(self.in_features)
        self.bias = nn.Parameter(torch.normal(mean=bias_mean, std=0.5))
        
          
            
    def forward(self, 
               inputs: torch.Tensor, 
               ) -> torch.Tensor:
        """
        Args:
        inputs of shape (batch_size, in_features)
        """
        output = torch.matmul((inputs-self.bias), self.weights)
        output = F.relu(output)
        return output
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'