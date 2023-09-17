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
        ) -> None: # type-check
            """
            The neural additive model learns a linear combination of nerual networks each of which attends to a single input feature. The outputs of subnets are added up, with a scalar bias, and passed through a link function for prediction. 
            Args:
            in_features: size of each input sample 
            """
            super(NAM, self).__init__()
            self.config = config
            # a feature NN for each feature
            self.in_features = in_features
            self.feature_dropout = nn.Dropout(p=self.config.feature_dropout)
            
            # each feature subnet attends to a single input feature
            self.feature_nns = nn.ModuleList([
                FeatureNN(config, 
                          name=f"FeatureNN_{feature_index}", 
                          in_features=1, 
                          feature_index=feature_index) # note the in_features shape 
                for feature_index in range(in_features)
            ])
            self.bias = nn.Parameter(data=torch.zeros(1)) # bias of shape (1), initialized with zero
            
    def features_output(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """
        Return list [torch.Tensor of shape (batch_size, 1)]: the outputs of feature neural nets
        """
        return [self.feature_nns[feature_index](inputs[:, feature_index]) for feature_index in range(self.in_features)] # feature of shape (1, batch_size)
            
    def forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
        inputs of shape (batch_size, in_features): input samples, 
        
        Returns: 
        nam mean of shape (batch_size): add up the means of feature nets and bias
        nam variance of shape (batch_size): add up the variance of feature nets
        
        fnn mean of shape (batch_size, in_features): mean of each feature net
        fnn var of shape (batch_size, in_features): variance of each feature net 
        """
        nn_outputs = self.features_output(inputs) # list [Tensor(batch_size,  1)]
        cat_outputs = torch.cat(nn_outputs, dim=-1) # of shape (batch_size, in_features)
        
        dropout_outputs = self.feature_dropout(cat_outputs) # feature dropout
        outputs = dropout_outputs.sum(dim=-1) # sum along the features => of shape (batch_size)
        # print(f"nam.forward, nn_outputs: {nn_outputs}, cat_outputs: {cat_outputs}, dropout_outputs: {dropout_outputs}, outputs: {outputs + self.bias}")
        return outputs + self.bias, dropout_outputs
#         fnn_out_list = self.features_output(inputs) # list [Tensor(batch_size,  2)]
#         stacked_out = torch.permute(torch.stack(fnn_out_list), (1, 0, -1))
#         mean = stacked_out[:, :, 0] # (batch_size, in_features)
#         var = stacked_out[:, :, 1]

#         dropout_mean =  self.feature_dropout(mean)
#         dropout_var =  self.feature_dropout(var)

#         additive_mean = dropout_mean.sum(dim=-1)
#         additive_var = dropout_var.sum(dim=-1)
#         # print(f"nam.forward, nn_outputs: {nn_outputs}, cat_outputs: {cat_outputs}, dropout_outputs: {dropout_outputs}, outputs: {outputs + self.bias}")

#         return additive_mean + self.bias, additive_var, dropout_mean, dropout_var # note that outputs + bias => broadcast
