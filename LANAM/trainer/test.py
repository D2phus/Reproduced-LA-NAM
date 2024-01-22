import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

"""testing"""
def test(likelihood, 
         model: nn.Module, 
         dataloader: DataLoader) -> float:
    """test generalized additive model with given dataloader
    average mse loss is used for regression task, and average accuracy is used for classification task.
    Args:
    ------
    model: nn.Module 
        generalized additive model giving overall and individual prediction 
        
    Returns:
    ------
    loss: float 
        average loss
    """
    device = next(model.parameters()).device
    
    criterion = lambda f, y: (f-y).square().sum() if likelihood=='regression' else lambda f, y: torch.sum(torch.argmax(f, dim=-1) == y) 
    loss = 0.0
    
    for  X, y in dataloader: 
        X, y = X.to(device), y.to(device)
        f_mu, f_mu_fnn= model(X) 
        f, y = f_mu.detach().flatten(), y.flatten() # NOTE the shape
        step_loss = criterion(f, y)
        loss += step_loss 
    
    loss /= len(dataloader.dataset)
    return loss


        