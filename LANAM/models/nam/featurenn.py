"""DNN-based sub net for each input feature."""
from LANAM.models.activation import ExU
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNN(nn.Module):
    def __init__(
        self, 
        config, 
        name: str, 
        in_features: int,
        feature_index: int, 
        ) -> None: # type-check
            """
            There is a DNN-based sub net for each feature. The first hidden layer is selected amongst:
            1. standard ReLU units
            2. ExU units
            Additionally, dropout layers are added to the end of each hidden layer.
            
            Args:
            in_features: scalar, size of each input sample; default value = 1
            feature_index: indicate which feature is learn in this subnet
            """
            super(FeatureNN, self).__init__()
            if config.activation_cls not in ['relu', 'exu', 'gelu', 'elu', 'leakyrelu']:
                raise ValueError('Activation unit should be `gelu`, `relu`, `exu`, `elu`, or `leakyrelu`.')
                
            self.name = name
            self.config = config
            self.in_features = in_features
            self.feature_index = feature_index
            # self.dropout = nn.Dropout(p=self.config.dropout)
            self.activation_cls = config.activation_cls
            self.hidden_sizes = config.hidden_sizes
            self.model = self.setup_model()
            
       
    def setup_model(self): 
        layers = list()
        hidden_sizes = [self.in_features] + self.hidden_sizes
        if self.activation_cls == "exu":
            for index, (in_f, out_f) in enumerate(zip(hidden_sizes[:], hidden_sizes[1:])):
                if index == 0: 
                    layers.append(ExU(in_f, out_f))
                else:
                    layers.append(nn.Linear(in_f, out_f,  bias=True))
                    layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=self.config.dropout))
        else: 
            if self.activation_cls == 'gelu': 
                activation_cls = nn.GELU
            elif self.activation_cls == 'relu':
                activation_cls = nn.ReLU
            elif self.activation_cls == 'elu': 
                activation_cls = nn.ELU
            elif self.activation_cls == 'leakyrelu':
                activation_cls = nn.LeakyReLU
            for in_f, out_f in zip(hidden_sizes[:], hidden_sizes[1:]):
                    layers.append(nn.Linear(in_f, out_f,  bias=True))
                    layers.append(activation_cls())
                    layers.append(nn.Dropout(p=self.config.dropout))
            
        layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=1)) # output layer; out_features=1
        layers.append(nn.Dropout(p=self.config.dropout))
            
        # gausian layer for uncertainty estimation 
        # layers.append(GaussianLayer(in_features=1))
            
        return nn.Sequential(*layers)
            
            
    def forward(self, inputs) -> torch.Tensor:
        """
        Args: 
        inputs of shape (batch_size): a batch of inputs 
        Return of shape (batch_size, out_features) = (batch_size, 1): a batch of outputs 
        
        """
        outputs = inputs.unsqueeze(1) # TODO: of shape (batch_size, 1)?
        outputs = self.model(outputs)
        return outputs
        # mean, variance = self.model(outputs)
        # return mean, variance
