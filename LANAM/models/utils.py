import torch 

def normal_samples(mean, var, n_samples):
    """
    Sampling from a batch of normal distributions either parameterized
    by a diagonal or full covariance.

    Args: 
    ----------
    mean of shape (batch_size, out_features)
    var : (co)variance of the Normal distribution of shape (batch_size, out_features, out_features) or (batch_size, out_features)
    
    Returns: 
    ----------
    samples of shape (n_samples, batch_size, out_features)
    """
    assert mean.ndim == 2, 'Mean should be 2-dimensional.'
    randn_samples = torch.randn((*mean.shape, n_samples), dtype=mean.dtype) # of shape (batch_size, out_features, n_samples)
    
    if mean.shape == var.shape:
        # diagonal covariance
        scaled_samples = var.sqrt().unsqueeze(-1) * randn_samples
        return (mean.unsqueeze(-1) + scaled_samples).permute((2, 0, 1))
    elif mean.shape == var.shape[:2] and var.shape[-1] == mean.shape[1]:
        # full covariance
        scale = torch.linalg.cholesky(var)
        scaled_samples = torch.matmul(scale, randn_samples)  
        return (mean.unsqueeze(-1) + scaled_samples).permute((2, 0, 1))
    else:
        raise ValueError('Invalid input shapes.')