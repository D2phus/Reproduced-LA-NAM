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
        ) -> None: 
            """DNN for each feature. 
            """
            super(FeatureNN, self).__init__()
            if config.activation_cls not in ['relu', 'exu', 'gelu', 'elu', 'leakyrelu']:
                raise ValueError('Activation unit type should be `gelu`, `relu`, `exu`, `elu`, or `leakyrelu`.')
                
            self.name = name
            self.config = config
            self.in_features = in_features
            self.feature_index = feature_index
            self.activation_cls = config.activation_cls
            self.hidden_sizes = [config.num_units for _ in range(config.num_layers)]
            # self.hidden_sizes = config.hidden_sizes
            self.model = self.setup_model()
            
       
    def setup_model(self): 
        layers = list()
        hidden_sizes = [self.in_features] + self.hidden_sizes
        
        if self.activation_cls == "exu":
            # [ExU, Linear + ReLU...]
            for index, (in_f, out_f) in enumerate(zip(hidden_sizes[:], hidden_sizes[1:])):
                if index == 0: 
                    layers.append(ExU(in_f, out_f))
                else:
                    layers.append(nn.Linear(in_f, out_f,  bias=True))
                    layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=self.config.dropout))
        else: 
            # [Linear + activation_cls]
            cls_dict = {'gelu': nn.GELU, 'relu': nn.ReLU, 'elu': nn.ELU, 'leakyrelu': nn.LeakyReLU}
            activation_cls = cls_dict[self.activation_cls]
                
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
        if inputs.ndim == 1: 
            inputs = inputs.unsqueeze(1) # (batch_size, 1)

        outputs = self.model(inputs)
        return outputs
        # mean, variance = self.model(outputs)
        # return mean, variance
