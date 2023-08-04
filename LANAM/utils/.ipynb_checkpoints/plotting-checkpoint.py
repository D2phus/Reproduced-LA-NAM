import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt 
import numpy as np

from typing import Sequence

import math
    
def plot_uncertainty(X, y, fnn, f_mu, f_var, f_mu_fnn, f_var_fnn, predictive_samples=None, plot_additive=False): 
    """
    Visualize the predictive posterior with confidence interval.
    Note that samples should be ordered for correct visualization.
    Args:
    -------------
    X of shape (batch_size, in_features)
    y of shape (batch_size, out_features = 1)
    fnn of shape (batch_size, in_features): 
        target for each individual feature.
    f_mu: of shape (batch_size): 
        additive predictive posterior mean
    f_var of shape (batch_size, 1): 
        additive predictive posterior variance
    f_mu_fnn of shape (batch_size, in_features):
        individual predictive posterior mean
    f_var_fnn of shape (batch_size, in_features, 1):
        individual predictive posterior variance
    predictive_samples of shape (n_samples, batch_size, out_features=1): 
        generated samples.
    plot_additive: 
        bool, plot the additive fitting if True.
    """
    in_features = f_mu_fnn.shape[1]
    # compute the feature-wise residual by subtracting the mean contribution of the other feature networks from the generated data points.
    if predictive_samples is not None: 
        n_samples = predictive_samples.shape[0]
        predictive_samples = predictive_samples.squeeze(-1)
        Ef_samples_fnn = torch.stack([f_mu_fnn]*n_samples, dim=0) # of shape (n_samples, batch_size, in_features)
        residual = torch.stack([predictive_samples - torch.cat([Ef_samples_fnn[:, :, 0:index], Ef_samples_fnn[:, :, index+1:]], dim=-1).sum(dim=-1) for index in range(in_features)], dim=-1) # of shape (n_samples, batch_size, in_features)
        residual = (residual - residual.mean(dim=1).unsqueeze(1)).numpy() # re-center 
    
    # re-center the features before visualization
    Ef_mu_fnn = f_mu_fnn.mean(dim=0).reshape(1, -1) # of shape (1, in_features)
    f_mu_fnn = f_mu_fnn - Ef_mu_fnn 
    fnn = fnn - fnn.mean(dim=0).reshape(1, -1)
    
    f_mu, f_var = f_mu.flatten().detach().numpy(), f_var.flatten().detach().numpy() # of shpe (batch_size)
    
    
    f_mu_fnn, f_var_fnn = f_mu_fnn.flatten(1).detach().numpy(), f_var_fnn.flatten(1).detach().numpy() # of shape (batch_size, in_features)
    std = np.sqrt(f_var)
    std_fnn = np.sqrt(f_var_fnn)
    
    cols = 3
    rows = math.ceil(in_features / cols)
    fig, axs = plt.subplots(rows, cols)
    axs = axs.ravel() # 
    fig.tight_layout()
    #plt.setp(axs, ylim=(-4, 4))
    for index in range(in_features): 
        lconf, hconf = f_mu_fnn[:, index]-2*std_fnn[:, index], f_mu_fnn[:, index]+2*std_fnn[:, index]
        customize_ylim = (np.min(lconf).item()-0.5, np.max(hconf).item()+0.5)
        plt.setp(axs[index], ylim=(customize_ylim))
        axs[index].plot(X[:, index], fnn[:, index], '--', label="targeted", color="gray")
        axs[index].plot(X[:, index], f_mu_fnn[:, index], '-', label="prediction", color="royalblue")
        
        axs[index].fill_between(X[:, index], lconf, hconf, alpha=0.3)
        if predictive_samples is not None:
            axs[index].plot(torch.stack([X[:, index]]*n_samples, dim=0), residual[:, :, index], 'o', color='lightgray', label='residuals', alpha=0.2)
            
    if plot_additive: 
        fig, axs = plt.subplots()
        axs.plot(X[:, 0], y, '--', label="targeted", color="gray")
        axs.plot(X[:, 0], f_mu, '-', label="prediction", color="royalblue")
        axs.fill_between(X[:, 0].flatten(), f_mu-2*std, f_mu+2*std, alpha=0.2)
        if predictive_samples is not None:
            axs.plot(torch.stack([X[:, 0]]*n_samples, dim=0), predictive_samples.numpy(), 'o', color='lightgray', label='samples', alpha=0.2)
        
def plot_mean(X, y, fnn, f_mu, f_mu_fnn, plot_additive=False): 
    Ef_mu_fnn = f_mu_fnn.mean(dim=0).reshape(1, -1) # of shape (1, in_features)
    f_mu_fnn = f_mu_fnn - Ef_mu_fnn 
    fnn = fnn - fnn.mean(dim=0).reshape(1, -1)
    
    
    in_features = f_mu_fnn.shape[1]
    cols = 3 
    rows = math.ceil(in_features / cols)
    fig, axs = plt.subplots(rows, cols)
    axs = axs.ravel() # 
    fig.tight_layout()
    plt.setp(axs, ylim=(-4, 4))
    for index in range(in_features):
        axs[index].plot(X[:, index], fnn[:, index].detach().numpy(), color='gray')
        axs[index].plot(X[:, index], f_mu_fnn[: ,index].detach().numpy(), color='royalblue')
    if plot_additive: 
        fig, axs = plt.subplots()
        axs.plot(X[:, 0], y, '--', label="targeted", color="gray")
        axs.plot(X[:, 0], f_mu, '-', label="prediction", color="royalblue")
        
def plot_predictive_posterior(model, testset, uncertainty=True, sampling=False, plot_additive=True): 
    X, y, fnn = testset.X, testset.y, testset.fnn
    f_mu, f_var, f_mu_fnn, f_var_fnn = model.predict(X)
    
    noise = model.sigma_noise.reshape(1, -1, 1)
    pred_var_fnn = f_var_fnn + noise.square()
    pred_var = f_var + model.additive_sigma_noise.square()
    samples = model.predictive_samples(X) if sampling else None 
    if uncertainty:
        plot_uncertainty(X, y, fnn, f_mu,f_var, f_mu_fnn, f_var_fnn, predictive_samples=samples, plot_additive=plot_additive)
    
    else:
        plot_mean(X, y, fnn, f_mu, f_mu_fnn, plot_additive=plot_additive)
        