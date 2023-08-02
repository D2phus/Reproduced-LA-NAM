import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt 
import numpy as np

from typing import Sequence

import math
    
def reg_predictive_plot(X, y, fnn, f_mu, pred_var, f_mu_fnn, pred_var_fnn, predictive_samples=None, plot_additive=False): 
    """
    Visualize the predictive posterior of each feature neural net.
    Note that samples should be ordered for correct visualization.
    Args:
    fnn of shape (batch_size, in_features)
    f_mu: of shape (batch_size)
    pred_var of shape (batch_size, 1)
    f_mu_fnn of shape (batch_size, in_features)
    pred_var_fnn of shape (batch_size, in_features, 1)
    predictive_samples of shape ( n_samples, batch_size, in_features)
    """
    in_features = f_mu_fnn.shape[1]
    if predictive_samples is not None:
        n_samples = predictive_samples.shape[0]
        f_samples = predictive_samples.sum(dim=-1)# of shape (n_samples, batch_size)
        # E_samples = predictive_samples.mean(dim=1).unsqueeze(1) # of shape (n_samples, 1, in_features)
        Ef_samples_fnn = torch.stack([f_mu_fnn]*n_samples, dim=0) # (n_samples, batch_size, in_features)
        residual = torch.stack([f_samples - torch.cat([Ef_samples_fnn[:, :, 0:index], Ef_samples_fnn[:, :, index+1:]], dim=-1).sum(dim=-1) for index in range(in_features)], dim=-1) #of shape (n_samples, batch_size, in_features)
        residual = (residual - residual.mean(dim=1).unsqueeze(1)).detach().numpy()
    
    # re-center the features before visualization
    Ef_mu_fnn = f_mu_fnn.mean(dim=0).reshape(1, -1) # of shape (1, in_features)
    f_mu_fnn = f_mu_fnn - Ef_mu_fnn 
    fnn = fnn - fnn.mean(dim=0).reshape(1, -1)
    
    f_mu, pred_var = f_mu.flatten().detach().numpy(), pred_var.flatten().detach().numpy() # of shpe (batch_size)
    
    
    f_mu_fnn, pred_var_fnn = f_mu_fnn.flatten(1).detach().numpy(), pred_var_fnn.flatten(1).detach().numpy() # of shape (batch_size, in_features)
    std = np.sqrt(pred_var)
    std_fnn = np.sqrt(pred_var_fnn)
    
    cols = 3 
    rows = math.ceil(in_features / cols)
    fig, axs = plt.subplots(rows, cols)
    axs = axs.ravel() # 
    fig.tight_layout()
    for index in range(in_features): 
        axs[index].plot(X[:, index], fnn[:, index], '--', label="targeted", color="gray")
        axs[index].plot(X[:, index], f_mu_fnn[:, index], '-', label="prediction", color="royalblue")
        
        if predictive_samples is not None:
            axs[index].plot(torch.stack([X[:, index]]*n_samples, dim=0), residual[:, :, index], 'o', color='lightgray', label='residuals', alpha=0.2)
        axs[index].fill_between(X[:, index], f_mu_fnn[:, index]-2*std_fnn[:, index], f_mu_fnn[:, index]+2*std_fnn[:, index], alpha=0.2)
    
    if plot_additive: 
        fig, axs = plt.subplots()
        axs.plot(X[:, 0], y, '--', label="targeted", color="gray")
        axs.plot(X[:, 0], f_mu, '-', label="prediction", color="royalblue")
        axs.fill_between(X[:, 0].flatten(), f_mu-2*std, f_mu+2*std, alpha=0.2)
        
        
def plot_training(num_epochs: int, 
                  losses_train: Sequence, 
                  metricses_train: Sequence, 
                  losses_val: Sequence, 
                  metricses_val: Sequence):
    """
    Plot the training & validation loss and metrics
    """
    print(f"The minimum validation loss: {min(losses_val)}")
    print(f"The minimum validation metrics: {min(metricses_val)}")
    x = np.arange(num_epochs)
    fig = plt.figure()
    plt.plot(x, losses_train, '-', label="train loss")
    plt.plot(x, metricses_train, '-', label="train metrics")
    plt.plot(x, losses_val, '-', label="validation loss")
    plt.plot(x, metricses_val, '-', label="validation metrics")
    plt.legend()