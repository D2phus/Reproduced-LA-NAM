"""penalized loss for training"""
import torch
import torch.nn as nn
import torch.nn.functional as F



def mse_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
)-> torch.Tensor:
    """
    Mean squared error loss for regression 
    Args:
    logits of shape (batch_size): the predictions
    targets of shape (batch_size): the targets
    """
    return F.mse_loss(logits.view(-1), targets.view(-1))

def bce_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
)-> torch.Tensor:
    """
    Binary cross entropy loss for classification
    
    Args:
    logits of shape (batch_size)
    targets of shape (batch_size), binary classification
    """
    # note that we use bce instead of ce
    # return F.cross_entropy(logits, targets)
    # view is not necessary
    return F.binary_cross_entropy_with_logits(logits.view(-1), targets.view(-1)) 

def penalized_loss(
    config, 
    nam_out: torch.Tensor, 
    fnn_out: torch.Tensor, 
    model: nn.Module, 
    targets: torch.Tensor, 
)-> torch.Tensor:
    """
    Compute penalized loss of NAM
    
    Args:
    nam_out of shape (batch_size): model output 
    fnn_out of shape (batch_size, in_features): output of each feature nn
    model: the model that we use
    targets of shape (batch_size): targets of each sample 
    """
    def fnn_loss(
        fnn_out: torch.Tensor
    )->torch.Tensor:
        """
        Penalizes the L2 norm of the prediction of each feautre net
        
        note output penalty is set to zero when we use DNN as baseline
        Args: 
        fnn_out of shape (batch_size, in_features): output of each featrue nn
        """
        num_fnn = len(fnn_out) # number of feature nets
        return torch.mean(torch.square(fnn_out), 1).sum() / num_fnn
        
    def weight_decay(
        model: nn.Module 
    )->torch.Tensor:
        """
        Penalizes the L2 norm of weights in each *feature net*
        
        """
        num_networks = len(model.feature_nns)
        l2_losses = [(p**2).sum() for p in model.parameters()]
        return sum(l2_losses) / num_networks
        
    output_regularization = config.output_regularization
    l2_regularization = config.l2_regularization
    
    loss = 0.0
    # task dependent function 
    # TODO: not going through a logistic function?
    if config.regression: 
        loss += mse_loss(nam_out, targets)
    else:
        loss += bce_loss(nam_out, targets)
        
    if output_regularization > 0:
        loss += output_regularization * fnn_loss(fnn_out) # output penalty
        
    if l2_regularization > 0:
        loss += l2_regularization * weight_decay(model) # weight decay 
        
    return loss
    # l = F.cross_entropy(out, targets)
    # F.mse_loss(outputs, targets)

