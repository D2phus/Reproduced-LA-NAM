import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt 
import plotly.express as px

import numpy as np

import copy
import math 

from typing import List, Tuple

from LANAM.utils.correlation import *

def adjusted_R_squared(X: torch.Tensor, 
                       y: torch.Tensor, 
                       prediction_mean: torch.Tensor):
    num_observations, num_pred_variables = X.shape
    
    R_squared = pairwise_correlation(torch.stack([prediction_mean.squeeze(), y.squeeze()], dim=1))[0][1].square() 
    R_squared = 1- (num_observations-1)/(num_observations-num_pred_variables-1)*(1-R_squared)
    return R_squared
    
def pairwise_correlation(X: torch.Tensor, eps: float=1e-12) -> torch.Tensor: 
    '''normalized covariance matrix for a batch of features X.
    Args:
    ---- 
    X of shape (batch_size, num_variables)
        feature batch.
    eps: float
        a small value is added to the standard deviation to avoid divide zero issue. 
        
    Returns: 
    -----
    normalized covariance matrix of shape (num_variables, num_variables)
    '''
    batch_size, in_features = X.shape 
    if batch_size < 2 or in_features < 2: 
        # no sufficient samples or features for convariance 
        return torch.zeros_like(X)
    
    std = X.std(dim=0)
    std = std*std.reshape(-1, 1)
    cov = torch.cov(X.T) # unnormalized covariance matrix. NOTE torch.cov and torch.coeff requires data shape (num_variables, batch_size)
    # print(cov)
    cov /= (std + eps) 
    cov = torch.where(std==0.0, 0.0, cov)   
    
    return cov

def concurvity(X: torch.Tensor, eps: float=1e-12) -> torch.Tensor: 
    """measured concurvity for a batch of features X. 
    """
    in_features = X.shape[1]
    cov = pairwise_correlation(X, eps)
    
    R = torch.triu(cov.abs(), diagonal=1).sum()
    R /= (in_features*(in_features-1)/2)
    return R

def feature_importance(X: torch.Tensor) -> torch.Tensor: 
    '''feature importance (sensitivity). 
    Args: 
    ----
    X of shape (batch_size, num_variables)
        
    Returns: 
    ---
    importance of shape (num_variables)
        importance of each feature (variable)
    ''' 
    return torch.abs(X - X.mean(dim=0)).mean(dim=0)


def get_ensemble_prediction(models: List[nn.Module], samples: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): 
    def call_single_model(params, buffers, data):
        return torch.func.functional_call(base_model, (params, buffers), (data,))

    X, y, shape_functions, feature_names = samples
    base_model = copy.deepcopy(models[0])
    base_model.to('meta')
    params, buffers = torch.func.stack_module_state(models) 
    pred_map, fnn_map = torch.vmap(call_single_model, (0, 0, None))(params, buffers, X) # (num_ensemble, batch_size, out_features)
    return pred_map.detach(), fnn_map.detach()

def get_prediction(models: List[nn.Module], samples: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): 
    prediction_map, contributions_map = get_ensemble_prediction(models, samples)
    prediction_mean, feature_contribution_mean, prediction_var, feature_contribution_var = prediction_map.mean(dim=0), contributions_map.mean(dim=0), prediction_map.var(dim=0), contributions_map.var(dim=0) 
    
    return prediction_mean, feature_contribution_mean, prediction_mean, feature_contribution_var

def get_feature_importances(models, samples): 
    prediction_mean, feature_contribution_mean, prediction_mean, feature_contribution_var = get_prediction(models, samples)
    
    importances = feature_importance(feature_contribution_mean)
    
    importance_dict = {name: importance for name, importance in zip(samples[3], importances)}
    
    return importance_dict

def plot_feature_importance_errorbar(models, samples, width=0.5): 
    in_features = samples[0].shape[1]
    names = samples[3]
    
    prediction_map, contributions_map = get_ensemble_prediction(models, samples)
    contributions_map = contributions_map.transpose(0, 1)
    importances = feature_importance(contributions_map)
    mean, var = importances.mean(dim=0), importances.var(dim=0)
    
    fig = plt.figure(figsize=(8, 5))
    
    idx = torch.arange(in_features)
    plt.errorbar(idx, mean, var, linestyle='None', marker='^', label='NAMs')
    
    plt.xticks(idx + width / 2, names, rotation=90, fontsize='large')
    plt.ylabel('Mean Absolute Value', fontsize='x-large')
    plt.legend(loc='upper right', fontsize='large')
    plt.title(f'Overall Feature Importance ErrorBar', fontsize='x-large')
    
    plt.show()
    
    return fig
    
def plot_feature_importance(models, samples, width=0.5):
    in_features = samples[0].shape[1]
    names = samples[3]
    
    prediction_mean, feature_contribution_mean, prediction_mean, feature_contribution_var = get_prediction(models, samples)
    
    importances = feature_importance(feature_contribution_mean)
    
    fig = plt.figure(figsize=(5, 5))
    
    idx = torch.arange(in_features)
    plt.bar(idx, importances, width, label='NAMs')
    plt.xticks(idx + width / 2, names, rotation=90, fontsize='large')
    plt.ylabel('Mean Absolute Value', fontsize='x-large')
    plt.legend(loc='upper right', fontsize='large')
    plt.title(f'Overall Feature Importance', fontsize='x-large')
    plt.show()

    return fig


def plot_pairwise_contribution_correlation(models, samples, pair_idx: Tuple[int, int]): 
    X, y, shape_functions, feature_names = samples
    idx1, idx2 = pair_idx
    prediction_map, contributions_map = get_ensemble_prediction(models, X, y)
    
    fig = plt.figure(figsize=(4, 3))
    for idx, m in enumerate(models): 
        plt.scatter(contributions_map[idx, :, idx1], contributions_map[idx, :, idx2])
        
    plt.xlabel(f'Feature Contribution {idx1}')
    plt.ylabel(f'Feature Contribution {idx2}')
    
    plt.title(f'Contribution Pair Correlation', fontsize='x-large')
    plt.show()
    
    return fig
    
def plot_concurvity_versus_accuracy(lam_list, acc_list, concur_list, axs=None): 
    '''plot measured concurvity (R_perp) versus accuracy (RMSE)'''
    
    if axs is None:
        fig, axs= plt.subplots()
        
    axs.set_xlabel('Val. RMSE')
    axs.set_ylabel('Val. R_perp')
    
    # plot 
    s = axs.scatter(acc_list, concur_list, marker='D', c=lam_list, cmap='viridis', norm='log')
    cbar = fig.colorbar(s) 
    cbar.ax.set_ylabel('lambda', rotation=270)
    
    return fig
    
def plot_recovered_functions(X, y, shape_functions, feature_contribution_mean, feature_contribution_var, X_train=None, shape_functions_train=None, center: bool=True): 
    """recover shape functions with a known additive structure.
    Args: 
    center: whether to centerize the plots at zero along y-axis.
    """
    def sort_by_indices(x: torch.Tensor, indices: List):
        """sort x by given indices of the same shape."""
        d1, d2 = x.size()
        ret = torch.stack([x[indices[:, idx], idx] for idx in range(x.shape[1])] ,dim=1)
        return ret
    
    in_shape_functions = X.shape[1]
    
    # for visualization, data is sorted 
    X, indices = torch.sort(X, dim=0) # sort input shape_functions along each dimension
    # sort known function values, prediction mean, and variance according to permutation indices of X
    shape_functions = sort_by_indices(shape_functions, indices)
    feature_contribution_mean = sort_by_indices(feature_contribution_mean, indices)
    feature_contribution_var = sort_by_indices(feature_contribution_var, indices)
    
    # center at zeros
    if center: 
        feature_contribution_mean -= feature_contribution_mean.mean(dim=0).reshape(1, -1)
        shape_functions -= shape_functions.mean(dim=0).reshape(1, -1)
    
        if X_train is not None and shape_functions_train is not None:
            shape_functions_train -= shape_functions_train.mean(dim=0).reshape(1, -1)
    
    std_fnn = torch.sqrt(feature_contribution_var)
    
    # plot configuration 
    if in_shape_functions < 5: 
        cols, rows = in_shape_functions, 1 
    else: 
        cols = 4
        rows = math.ceil(in_shape_functions / cols)
    figsize = (2*cols ,2*rows)  
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.ravel() 
    # fig.supxlabel('common x label')
    
    for index in range(in_shape_functions): 
        lconf, hconf = feature_contribution_mean[:, index]-2*std_fnn[:, index], feature_contribution_mean[:, index]+2*std_fnn[:, index]
        customize_ylim = (torch.min(shape_functions) - 0.5, torch.max(shape_functions) + 0.5)
        # customize_ylim = (torch.min(shape_functions[:, index])-1, torch.max(shape_functions[:, index])+1)
        hist_scale = customize_ylim[1] - customize_ylim[0]
        axs[index].set_ylim(customize_ylim)
        
        axs[index].hist(X[:, index], bins=10, bottom=customize_ylim[0], density=True, weights= hist_scale * np.ones_like(X[:, index].numpy()), alpha=0.5, color='lightblue', label='training points')
            
        axs[index].plot(X[:, index], shape_functions[:, index], '--', label="targeted", color="gray")
        axs[index].plot(X[:, index], feature_contribution_mean[:, index], '-', label="prediction", color="royalblue")
        axs[index].fill_between(X[:, index], lconf, hconf, alpha=0.2)

        if X_train is not None and shape_functions_train is not None:
            axs[index].scatter(X_train[:, index].flatten(), shape_functions_train[:, index].flatten(), alpha=0.3, color='grey', label='training points')
    
    # fig.suptitle('orange: training points, blue dots: targeted, blue solid: prediction')
    fig.tight_layout()
    return fig

