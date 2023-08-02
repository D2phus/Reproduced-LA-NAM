from .activation import ExU
from .activation import LinearReLU
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureNN(nn.Module):
    def __init__(
        self, 
        config, 
        name: str, 
        in_features: int,
        num_units: int, 
        feature_index: int, 
        ) -> None: # type-check
            """
            There is a DNN sub net for each feature. The architectore for the feature net is selected amongst 
            1. single hidden layer with standard ReLU units and 3 regular hidden layers,
            2. single hidden layer with ExU units and 3 regular hidden layers. 
            Additionally, dropout layers are added to the end of each hidden layer.
            
            Args:
            in_features: scalar, size of each input sample 
            num_units: scalar, number of ExU/LinearReLU hidden units in the single hidden layer 
            feature_index: indicate which feature is learn in this subnet
            """
            super(FeatureNN, self).__init__()
            
            self.config = config
            self.in_features = in_features
            self.num_units = num_units
            self.feature_index = feature_index
            
            # self.dropout = nn.Dropout(p=self.config.dropout)
            hidden_sizes = [self.num_units] + self.config.hidden_sizes
             
            layers = []
            # The first layer is consisted of ExU units
            if self.config.activation == "exu":
                layers.append(ExU(in_features=self.in_features, out_features=self.num_units))
            else:
                layers.append(LinearReLU(in_features=self.in_features, out_features=self.num_units))
            layers.append(nn.Dropout(p=self.config.dropout))
            
            # followed by several standard relu layers
            for in_f, out_f in zip(hidden_sizes[:], hidden_sizes[1:]):
                layers.append(LinearReLU(in_features=in_f, out_features=out_f))
                layers.append(nn.Dropout(p=self.config.dropout))
                
            layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=1)) # last linear layer; out_features=1
            layers.append(nn.Dropout(p=self.config.dropout))
            
            # self.model = nn.ModuleList(layers) 
            self.model = nn.Sequential(*layers)
            
    def forward(self, inputs) -> torch.Tensor:
        """
        Args: 
        inputs of shape (batch_size): a batch of inputs 
        Return of shape (batch_size, out_features) = (batch_size, 1): a batch of outputs 
        
        """
        outputs = inputs.unsqueeze(1) # TODO: of shape (batch_size, 1)?
        return self.model(outputs)
        # for layer in self.model:
          #   outputs = self.dropout(layer(outputs)) # Dropout to regularzie ExUs in each feature net.
        # return outputs 