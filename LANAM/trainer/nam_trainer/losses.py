"""penalized loss for training"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from LANAM.utils.regularizer import *


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
    out: torch.Tensor, 
    fnn_out: torch.Tensor, 
    model: nn.Module, 
    target: torch.Tensor, 
    conc_reg: bool=True,
)-> torch.Tensor:
    """penalized loss of NAM
    
    Args:
    out of shape (batch_size): NAM's prediction. 
    fnn_out of shape (batch_size, in_features): feature (net) contributions. 
    model: NAM
    target of shape (batch_size): target of each sample 
    conc_reg: whether to apply concurvity_regularization. For stability reasons, concurvity regularization is added only after 5% of the total optimization steps.
    """
    def fnn_loss(
        fnn_out: torch.Tensor, 
        d: int=2
    )->torch.Tensor:
        """Penalizes the Ld norm of the output of each feature net
        """
        num_fnn = len(fnn_out) # batch_size
        if d == 2:
            losses = torch.mean(torch.square(fnn_out), 1).sum() / num_fnn
        elif d == 1: 
            losses = torch.mean(torch.abs(fnn_out), 1).sum() / num_fnn
        return losses
        
    def weight_decay(
        model: nn.Module, 
        d: int=2
    )->torch.Tensor:
        """Penalizes the d-norm of weights in each feature net
        """
        num_networks = len(model.feature_nns)
        if d == 2: 
            losses = [(p**2).sum() for p in model.parameters()] # l2
        elif d == 1: 
            losses = [p.abs().sum() for p in model.parameters()] # l1 
        return sum(losses) / num_networks
        
    loss = 0.0
    loss += mse_loss(out, target) if config.likelihood == 'regression' else bce_loss(out, target)
        
    # if config.output_regularization > 0: # equivalent to the L1 regularization below
    #     loss += config.output_regularization * fnn_loss(fnn_out, d=1) # output penalty
        
    if config.l2_regularization > 0:
        loss += config.l2_regularization * weight_decay(model)
    
    # concurvity regularization
    if conc_reg: 
        if config.concurvity_regularization > 0: # correlation 
            loss += config.concurvity_regularization * concurvity(fnn_out)
        if config.l1_regularization > 0: # L1 
            loss += config.l1_regularization *  Ld_norm(fnn_out, d=1)
        if config.hsic_regularization > 0: # HSIC
            loss += config.hsic_regularization * hsic(fnn_out)
        
    return loss