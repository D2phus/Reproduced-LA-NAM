"""metrics for evaluation"""
import torch 

def rmse(
    logits: torch.Tensor, 
    targets: torch.Tensor
)->torch.Tensor:
    """
    Root mean-squared error for regression 
    Args:
    """
    criterion = nn.MSELoss()
    loss = torch.sqrt(criterion(logits.view(-1), targets.view(-1)))
    return loss

def mse(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
)-> torch.Tensor:
    """
    Mean squared error loss for regression 
    Args:
    logits of shape (batch_size): the predictions
    targets of shape (batch_size): the targets
    """
    return (((logits.view(-1) - targets.view(-1)).pow(2)).sum() / targets.numel()).item()

def mae(
    logits: torch.Tensor, 
    targets: torch.Tensor
)->torch.Tensor:
    """
    Mean absolute error 
    Args: 
    
    """
    return (((logits.view(-1) - targets.view(-1)).abs()).sum() / targets.numel()).item()

def accuracy(
    logits: torch.Tensor, 
    targets: torch.Tensor
)-> torch.Tensor:
    """
    Accuracy for classification
    Args: 
    """
    return (((targets.view(-1) > 0) == (logits.view(-1) > 0.5)).sum() / targets.numel()).item()

