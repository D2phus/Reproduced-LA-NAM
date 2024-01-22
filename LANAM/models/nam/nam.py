"""Neural additive model"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from typing import Sequence
from typing import List

from .featurenn import FeatureNN

class NAM(nn.Module):
    def __init__(
        self, 
        config, 
        name: str, 
        in_features: int 
        ) -> None: 
            """The vanilla neural additive model class. 
            """
            super(NAM, self).__init__()
            self.config = config
            
            self.in_features = in_features
            self.feature_dropout = nn.Dropout(p=self.config.feature_dropout)
            
            self.feature_nns = nn.ModuleList([FeatureNN(config, name=f"FeatureNN_{feature_index}", in_features=1, feature_index=feature_index) for feature_index in range(in_features)])
            
            self.bias = nn.Parameter(data=torch.zeros(1)) # global bias
            
    def features_output(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """pass features to associated subnets. 
        Returns
        ----
        Sequence of torch.Tensor(batch_size, 1) 
            contribution of each feature (net). 
        """
        return [self.feature_nns[feature_index](inputs[:, feature_index]) for feature_index in range(self.in_features)] # feature of shape (1, batch_size)
            
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
        -----
        inputs (batch_size, in_features)
        
        Returns:
        -----
        outputs (batch_size)
        dropout_outputs(batch_size, in_features): feature (net) contributions. """
        nn_outputs = self.features_output(inputs) 
        cat_outputs = torch.cat(nn_outputs, dim=-1) 
        dropout_outputs = self.feature_dropout(cat_outputs) 
        
        outputs = dropout_outputs.sum(dim=-1) 
        return outputs + self.bias, dropout_outputs