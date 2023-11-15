"""epoch for model training, evaluation, and test"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import vmap

from .losses import penalized_loss
from .metrics import accuracy
from .metrics import mae
from LANAM.utils.plotting import * 

from types import SimpleNamespace
from typing import Mapping
from typing import Sequence

import copy


def ensemble_train_epoch(
    criterion, 
    metrics, 
    optimizers: torch.optim.Adam, 
    models: nn.Module, 
    device: str, 
    dataloader: torch.utils.data.DataLoader, 
    conc_reg: bool=True
) -> torch.Tensor: 
    """
    train models with different initialization. 
    Seuquentially.
    
    Args:
    ---------
    optimizers: list
        optimizers for each model.
    models: list
        ensembling members.
    conc_reg: bool
        whether to apply concurvity regularization.
        
    Returns: 
    -----
    loss, metrics, measured concurvity (R_perp), estimated feature importance  
    
    """
    num_ensemble = len(models)
    in_features = len(dataloader.dataset[0][0])
    
    for model in models:
        model.train()
    
    losses = [0.0]*len(models)
    metrs = [0.0]*len(models)
    Rs = [0.0]*len(models) # R_perp: pairiwise correlation between trasnformed features
    importance = [torch.zeros(in_features)]*len(models) # feature importance estimated on the training set 
    
        
    for batch in dataloader:
        features, targets = batch
        features, targets = features.to(device), targets.to(device)

        for idx, model in enumerate(models):  
            optimizer = optimizers[idx]
            
            optimizer.zero_grad()
            preds, fnn_out = model(features)
            
            step_loss = criterion(preds, fnn_out, model, targets, conc_reg)
            step_metrics = metrics(preds, targets)
            step_R = concurvity(fnn_out)
            step_im = feature_importance(fnn_out)

            step_loss.backward()
            optimizer.step()

            losses[idx] += step_loss
            metrs[idx] += step_metrics
            Rs[idx] += step_R
            importance[idx] += step_im
                
    return sum(losses) / num_ensemble / len(dataloader), sum(metrs) / num_ensemble / len(dataloader), sum(Rs) / num_ensemble / len(dataloader), torch.stack(importance, dim=1).detach() / len(dataloader)


def ensemble_evaluate_epoch(
    criterion, 
    metrics, 
    models: nn.Module, 
    device: str, 
    dataloader: torch.utils.data.DataLoader, 
    conc_reg: bool=True
) -> torch.Tensor: 
    """
    train an ensemble of models with the same minibatch.
    Use vmap to speed up.
    
    Args:
    ---------
    optimizers: list
    models: list
    """
    def call_single_model(params, buffers, data):
        return torch.func.functional_call(base_model, (params, buffers), (data,))

    
    num_ensemble = len(models)
    for model in models:
        model.eval()
    
    base_model = copy.deepcopy(models[0])
    base_model.to('meta')
    
    losses = [0.0]*len(models)
    metrs = [0.0]*len(models)
    Rs = [0.0]*len(models) # R_perp: pairiwise correlation between trasnformed features
    
    params, buffers = torch.func.stack_module_state(models) # all modules being stacked together must be the same, including the mode.
    
    for (X, y) in dataloader:
        X, y =  X.to(device), y.to(device)
            
        pred_map, fnn_map = torch.vmap(call_single_model, (0, 0, None))(params, buffers, X) # (num_ensemble, batch_size, out_features)
        for idx, model in enumerate(models):
            step_loss = criterion(pred_map[idx], fnn_map[idx], model, y, conc_reg)
            step_metrics = metrics(pred_map[idx], y)
            step_R = concurvity(fnn_map[idx])
            
            losses[idx] += step_loss
            metrs[idx] += step_metrics
            Rs[idx] += step_R
            
    return sum(losses) / num_ensemble / len(dataloader), sum(metrs) / num_ensemble / len(dataloader), sum(Rs) / num_ensemble / len(dataloader)


def train_epoch(
    criterion, 
    metrics, 
    optimizers: torch.optim.Adam, 
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    scheduler=None, 
) -> torch.Tensor: 
    """
    Perform an epoch of training on dataloader 
    """
    model.train()
    avg_loss = 0.0
    avg_metrics = 0.0
            
    for batch in dataloader:
        features, targets = batch

        optimizer.zero_grad()

        preds, fnn_out = model(features)

        step_loss = criterion(preds, fnn_out, model, targets)
        step_metrics = metrics(preds, targets)

        step_loss.backward()
        optimizer.step()
        
        avg_loss += step_loss
        avg_metrics += step_metrics
        
        if scheduler is not None: 
            scheduler.step()
                
    return avg_loss / len(dataloader), avg_metrics / len(dataloader)


def  evaluate_epoch(
    criterion, 
    metrics, 
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
) -> torch.Tensor: 
    """
    Perform an epoch of evaluation on dataloader 
    """
    model.eval()
    avg_loss = 0.0
    avg_metrics = 0.0
    for batch in dataloader:
                # Accumulates loss in dataset.
        with torch.no_grad():
            features, targets = batch
    
            preds, fnn_out = model(features)

            step_loss = criterion(preds, fnn_out, model, targets)
            step_metrics = metrics(preds, targets)
            avg_loss += step_loss
            avg_metrics += step_metrics

    return avg_loss / len(dataloader), avg_metrics / len(dataloader)

