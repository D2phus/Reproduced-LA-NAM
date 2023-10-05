import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt 
import plotly.express as px

import numpy as np

import math

from typing import List

 
def plot_recovered_functions(X, y, feature_out, f_mu_fnn, f_var_fnn, X_train=None, feature_out_train=None): 
    """plot recovery of known additive structure.
    Args: 
    X: (num_samples, in_features)
        input features.
    y: (num_samples, 1)
        additive truth.
    feature_out: (num_samples, in_features)
        individual truth.
    f_mu_fnn: (num_samples, in_features)
        predictive mean for each feature function. 
    f_var_fnn: (num_samples, in_features, 1)
        predictive variance for each feature function.
    X_train: (num_samples, in_features)
        input features of training data.
    feature_out_train: (num_samples, in_features)
        individual truth of training data.
    """
    def sort_by_indices(x: torch.Tensor, indices: List):
        """sort x by given indices of the same shape."""
        d1, d2 = x.size()
        ret = torch.stack([x[indices[:, idx], idx] for idx in range(x.shape[1])] ,dim=1)
        return ret
    
    in_features = X.shape[1]
    f_var_fnn = f_var_fnn.flatten(1)
    
    # for visiualization, we sort input and outputs
    X, indices = torch.sort(X, dim=0) # sort input features along each dimension
    # sort known function values, prediction mean, and variance according to permutation indices of X
    feature_out = sort_by_indices(feature_out, indices)
    f_mu_fnn = sort_by_indices(f_mu_fnn, indices)
    f_var_fnn = sort_by_indices(f_var_fnn, indices)
    
    # feature-wise residuals are the generated data points with mean contribution of the other feature networks subtracted
    # make no sense when there is an observation noise and using output variance
    #samples = y.repeat(1, in_features)
    #residual = torch.stack([torch.cat([f_mu_fnn[:, :idx], f_mu_fnn[:, idx+1:]], dim=1).sum(dim=1) for idx in range(in_features)]).transpose(1, 0) # (num_samples, 1)
    #samples -= residual
    # recenter
    #samples -= samples.mean(dim=0) # (batch_size, in_features)
    
    # re-center the features before visualization
    f_mu_fnn -= f_mu_fnn.mean(dim=0).reshape(1, -1)
    # type and shape formating
    feature_out -= feature_out.mean(dim=0).reshape(1, -1)
    f_mu_fnn, f_var_fnn = f_mu_fnn.detach().numpy(), f_var_fnn.detach().numpy() 
    std_fnn = np.sqrt(f_var_fnn)
    if X_train is not None and feature_out_train is not None:
        feature_out_train -= feature_out_train.mean(dim=0).reshape(1, -1)
    
    feature_out = feature_out.numpy()
    
    cols = 4
    rows = math.ceil(in_features / cols)
    figsize = (2*cols ,2*rows)  
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.ravel() 
    for index in range(in_features): 
        lconf, hconf = f_mu_fnn[:, index]-2*std_fnn[:, index], f_mu_fnn[:, index]+2*std_fnn[:, index]
        
        customize_ylim = (np.min(feature_out[:, index])-1, np.max(feature_out[:, index])+1)
        hist_scale = customize_ylim[1] - customize_ylim[0]
        axs[index].set_ylim(customize_ylim)
    
        axs[index].hist(X[:, index], bins=10, bottom=customize_ylim[0], density=True, weights= hist_scale * np.ones_like(X[:, index].numpy()), alpha=0.5, color='lightblue')
            
        axs[index].plot(X[:, index], feature_out[:, index], '--', label="targeted", color="gray")
        axs[index].plot(X[:, index], f_mu_fnn[:, index], '-', label="prediction", color="royalblue")
        #axs[index].scatter(X[:, index], samples[:, index], c='lightgray', alpha=0.3)
        axs[index].fill_between(X[:, index], lconf, hconf, alpha=0.2)

        if X_train is not None and feature_out_train is not None:
            axs[index].scatter(X_train[:, index].flatten(), feature_out_train[:, index].flatten(), alpha=0.3, color='tab:orange', label='training points')
    
    fig.suptitle('orange: training points, blue dots: targeted, blue solid: prediction')
    fig.tight_layout()
    return fig

