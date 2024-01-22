import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt 
import plotly.express as px

import numpy as np

import copy
import math 

from LANAM.utils.regularizer import *

from typing import List, Tuple, Dict


def adjusted_R_squared(X: torch.Tensor, 
                       y: torch.Tensor, 
                       prediction_mean: torch.Tensor):
    """adjusted Coefficient of determination for measuring dependence between independent and dependent variables. """
    num_observations, num_pred_variables = X.shape
    
    R_squared = pairwise_correlation(torch.stack([prediction_mean.squeeze(), y.squeeze()], dim=1))[0][1].square() 
    R_squared = 1- (num_observations-1)/(num_observations-num_pred_variables-1)*(1-R_squared)
    return R_squared.detach()


def feature_importance(X: torch.Tensor) -> torch.Tensor: 
    '''feature importance (sensitivity?). 
    Args: 
    ---
    X of shape (batch_size, num_variables)
    Returns: 
    ---
    importance of shape (num_variables)
    ''' 
    return torch.abs(X - X.mean(dim=0)).mean(dim=0)


def get_feature_importances(models: List[nn.Module], features: torch.tensor, feature_names: List, targets: torch.tensor, feature_targets: torch.tensor = None) -> Dict: 
    '''dictionary of feature importance.'''
    prediction_mean, feature_contribution_mean, prediction_mean, feature_contribution_var = get_prediction(models, features)
    importances = feature_importance(feature_contribution_mean)
    importance_dict = {name: importance for name, importance in zip(feature_names, importances)}
    
    return importance_dict


def get_ensemble_prediction(models: List[nn.Module], features: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]: 
        '''get ensemble members' prediction. 
        Returns: 
        ----
        predction_map of shape (num_ensemble, batch_size)
        contributions_map of shape (num_ensemble, batch_size, out_features)
        ''' 
        def call_single_model(params, buffers, data):
            return torch.func.functional_call(base_model, (params, buffers), (data,))
    
        params, buffers = torch.func.stack_module_state(models) 
        base_model = copy.deepcopy(models[0])
        base_model = base_model.to('meta')
        
        prediction_map, contributions_map = torch.vmap(call_single_model, (0, 0, None))(params, buffers, features) 
        return prediction_map.detach(), contributions_map.detach()


def get_prediction(models: List[nn.Module], features: torch.tensor): 
    """get prediction averaged over ensemble members."""
    if isinstance(models, list):
        # ensemble
        prediction_map, contributions_map = get_ensemble_prediction(models, features)
        prediction_mean, feature_contribution_mean, prediction_var, feature_contribution_var = prediction_map.mean(dim=0), contributions_map.mean(dim=0), prediction_map.var(dim=0), contributions_map.var(dim=0) 
    else:
        # probabilistic model 
        prediction_mean, prediction_var, feature_contribution_mean, feature_contribution_var = models.predict(features)
    return prediction_mean, feature_contribution_mean, prediction_var, feature_contribution_var


def plot_feature_importance_errorbar(models, features: torch.tensor, feature_names: List, width: float=0.5): 
    """plot the importance of features with one standard deviation.  
    """
    in_features = features.shape[1]
    
    prediction_map, contributions_map = get_ensemble_prediction(models, features)
    contributions_map = contributions_map.transpose(0, 1)
    importances = feature_importance(contributions_map)    
    
    mean, var = importances.mean(dim=0), importances.var(dim=0)
    err = torch.sqrt(var) # one standard deviation
    
    fig = plt.figure(figsize=(8, 5))
    idx = torch.arange(in_features)
    plt.errorbar(mean, idx, xerr=err, linestyle='None', fmt='o', label='1 std', color='red')
    plt.scatter(importances.transpose(1, 0).flatten(), idx.unsqueeze(1).repeat(1, importances.shape[0]).flatten(), alpha=.5) # each initialization
    
    plt.yticks(idx+width/2, feature_names, fontsize='large')
    plt.xlabel('importance', fontsize='x-large')
    # plt.xlim((0, 1))
    # plt.legend(loc='upper right', fontsize='large')
    plt.title(f'Feature Importance', fontsize='x-large')
    plt.legend()
    plt.show()
    
    return fig
    
def plot_feature_importance(models, features: torch.tensor, feature_names: List, width: float=0.5): 
    """plot the importance of features. 
    """
    in_features = features.shape[1]
    
    prediction_mean, feature_contribution_mean, prediction_var, feature_contribution_var = get_prediction(models, features)
    
    print(feature_contribution_mean.shape)
    importances = feature_importance(feature_contribution_mean)
    
    fig = plt.figure(figsize=(3, 3))
    
    idx = torch.arange(in_features)
    plt.barh(idx, importances, width, label='NAM')
    # plt.xticks(idx + width / 2, feature_names, rotation=45, fontsize='large')
    
    plt.yticks(idx+width/2, feature_names, fontsize='large')
    plt.xlabel('importance', fontsize='x-large')
    plt.xlim((0, 1))
    # plt.legend(loc='upper right', fontsize='large')
    plt.title(f'Feature Importance', fontsize='x-large')
    plt.show()

    return fig


def plot_feature_correlation_heatmap(features: torch.tensor, feature_names: List): 
    """plot feature correlation heatmap on given data."""
    in_features = features.shape[1]
    
    corr_matrix = pairwise_correlation(features)
    
    fig = plt.figure(figsize=(4, 3))
    hm = plt.imshow(corr_matrix, cmap='RdYlGn')
    fig.colorbar(hm)
    
    idx = torch.arange(in_features)
    plt.title('Average feature corrlelation')
    plt.xticks(idx, feature_names, rotation=90, fontsize='large')
    plt.yticks(idx, feature_names, fontsize='large')
    
    return fig

    
def plot_shape_function(X_gt: torch.tensor, 
                        feature_contribution_gt: torch.tensor, 
                        feature_names: List, 
                        feature_contribution_mean: torch.tensor, 
                        feature_contribution_var: torch.tensor = None):
    """plot the prediction and ground truth of shape functions."""
    in_features = X_gt.shape[1]
    cols = 4
    rows = math.ceil(in_features / cols)
    
    # centering 
    feature_contribution_mean -= feature_contribution_mean.mean(dim=0).reshape(1, -1)
    feature_contribution_gt -= feature_contribution_gt.mean(dim=0).reshape(1, -1)
    # error bar 
    std = torch.sqrt(feature_contribution_var) if feature_contribution_var is not None else torch.zeros_like(feature_contribution_var)
    
    fig, axs = plt.subplots(rows, cols, figsize=(3*cols ,2*rows))
    axs=axs.ravel()
    for idx in range(in_features):
        axs[idx].set_ylim((-2, 2))
        axs[idx].scatter(X_gt[:, idx], feature_contribution_gt[:, idx], s=10, label='ground truth', alpha=0.5, color='red')
        # axs[idx].scatter(X_gt[:, idx], feature_contribution_mean[:, idx], s=10,label='prediction', alpha=0.5)
        axs[idx].errorbar(X_gt[:, idx], feature_contribution_mean[:, idx], yerr=std[:, idx], fmt='.', label='prediction', alpha=0.2, ecolor='gainsboro', color='cornflowerblue')
        axs[idx].set_title(f'{feature_names[idx]}')

    plt.tight_layout()
    return fig

    
