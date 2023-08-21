from copy import deepcopy

import torch 
import torch.nn as nn
from LANAM.models import LaNAM

    
class OutputFilter(nn.Module):
    """Customize a single-output model consisted of selected feature nets from trained LA-NAM. Enable post-hoc Laplace.
    
    Args:
    ------
    model: LaNAM
        trained LA-NAM consisted of independent feature nets.
    subset_of_features: list
       indices of feature nets to be considered. 
    
    Attrs:
    ------
    model: nn.ModuleList
        list of deepcopied feature nets to be considered.
    """
    def __init__(self, model, subset_of_features=None): 
        super(OutputFilter, self).__init__()
        
        self.subset_of_features = subset_of_features
        self.model = self._feature_net_subset(model)
    
    
    def _feature_net_subset(self, model: nn.Module): 
        """deepcopy feature nets to be considered. 
        When training LA-NAM, some attributes that require grads, e.g. lossfunc, sigma_noise, prior_precision, are modified and turn into non-leaf nodes. `copy.deepcopy` does not support to copy non-leaf nodes. To solve this, attributes' gradients are cleared up before copy.
        
        Args: 
        -----
        model: LaNAM
        """
        if self.subset_of_features is None: 
            self.subset_of_features = [*range(model.in_features)] 
        else: 
            self.subset_of_features.sort()
            
        #print(f'subset of features: {self.subset_of_features}')
        # clear up gradients for deepcopy
        model.prior_precision = model.prior_precision.detach()
        model.sigma_noise = model.sigma_noise.detach()
        cfnns = nn.ModuleList()
        for idx in self.subset_of_features:
            model.feature_nns[idx]._la = None # reinit laplace of the feature net for deepcopy
            cfnns.append(deepcopy(model.feature_nns[idx]))
        
        return cfnns
    
    def forward(self, inputs):
        """
        Returns:
        -------
        out: Tensor(batch_size, )
            additive output of feature nets to be considered.
        """
        fnn = list()
        for idx, feature_net_indice in enumerate(self.subset_of_features):
            fnn.append(self.model[idx](inputs[:, feature_net_indice]))
         
        out = torch.stack(fnn, dim=-1).sum(dim=-1) # sum along the features => of shape (batch_size)
        return out