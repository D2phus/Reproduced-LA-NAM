"""Evaluation metrics."""
import torch 
import math
    
    
def rmse_d(preds: torch.Tensor, 
           targets: torch.Tensor) -> torch.Tensor:
    """Root mean-squared error for each dimension. 
    RMSE_d = \sqrt{1/N \Sigma (f^{(d)}(X_i^{(d)}) - \hat{f}(X_i^{(d)})})^2}
    Args: 
    ----
    preds of shape (batch_size, in_features)
    targets of shape (batch_size, in_features)
    Returns: 
    ----
    loss of shape (in_features)
    """
    if targets is None:
        return torch.tensor(0)
    
    loss = (preds - targets).pow(2).sum(dim=0) / targets.size(0) # (in_features)
    return torch.sqrt(loss)

    
def rmse(
    logits: torch.Tensor, 
    targets: torch.Tensor
)->torch.Tensor:
    """Root mean-squared error for regression 
    Args:
    """
    loss = (((logits.view(-1) - targets.view(-1)).pow(2)).sum() / targets.numel()).item() # view(-1) = flatten(0), numel: the numbe of elements in `targets`
    return math.sqrt(loss)


def mse(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
)-> torch.Tensor:
    """Mean squared error loss for regression 
    Args:
    logits of shape (batch_size): the predictions
    targets of shape (batch_size): the targets
    """
    return (((logits.view(-1) - targets.view(-1)).pow(2)).sum() / targets.numel()).item()


def mae(
    logits: torch.Tensor, 
    targets: torch.Tensor
)->torch.Tensor:
    """Mean absolute error for regression
    Args: 
    
    """
    return (((logits.view(-1) - targets.view(-1)).abs()).sum() / targets.numel()).item()


def accuracy(
    logits: torch.Tensor, 
    targets: torch.Tensor
)-> torch.Tensor:
    """Accuracy for classification
    Args: 
    """
    return (((targets.view(-1) > 0) == (logits.view(-1) > 0.5)).sum() / targets.numel()).item()

