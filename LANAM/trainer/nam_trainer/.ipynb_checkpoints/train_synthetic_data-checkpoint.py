"""trainer class for model training and evaluation"""
import random 
import copy
import math 
import logging 

from types import SimpleNamespace
from typing import Mapping, Sequence, Tuple, List 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from functorch import vmap

import matplotlib.pyplot as plt 

import wandb
from tqdm import tqdm

from .losses import penalized_loss
from .metrics import * 


from LANAM.models import NAM
from LANAM.utils import * 


log = logging.getLogger(__name__)

def train_synthetic_data(config, 
          train_loader: DataLoader, 
          val_loader: DataLoader, 
          test_samples: Tuple=None, 
          ensemble: bool=True) -> List: 
        """trainer for (ensembled) vanilla NAM. 
        Args: 
        -----
        config: 
            configuration. 
        test_samples: Tuple(features, targets, feature_targets, feature_names)
        ensemble: bool
            whether to use ensemble members for uncertainty estimation
        
        Returns: 
        -----
        models: 
            ensemble members.
        """
        if config.wandb.use: 
            run = wandb.init(entity=config.wandb.entity, project=config.wandb.project) 

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        in_features = len(train_loader.dataset[0][0])
        
        # model ensembling
        num_ensemble = config.num_ensemble if ensemble else 1 
        seeds = [random.randint(0, 1000) for _ in range(num_ensemble)]
        # seeds = [*range(num_ensemble)]
        
        # set up criterion and metrics
        criterion = lambda nam_out, fnn_out, model, targets, concurvity_regularization: penalized_loss(config, nam_out, fnn_out, model, targets, concurvity_regularization)
        metrics = lambda nam_out, targets: mse(nam_out, targets) if config.likelihood == 'regression' else accuracy(nam_out, targets)
        # annotation
        metrics_name = "RMSE" if config.likelihood == 'regression' else "Accuracy"
        train_metrics_name, val_metrics_name = 'Train_' + metrics_name, 'Val_' + metrics_name
        train_loss_name, val_loss_name = 'Train_Loss', 'Val_Loss'
        R_name = 'R_perp'
        train_R_name, val_R_name = 'Train_' + R_name, 'Val_' + R_name
        
        # set up model, optimizer, and criterion for ensemble members
        models, optimizers, schedulers = list(), list(), list()
        for idx in range(num_ensemble): 
            setup_seeds(seeds[idx])
            model = NAM(config=config, name=f'NAM-{config.activation_cls}-{idx}', in_features=in_features)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.decay_rate) 
            # decrease learning rate during the whole training process (half cosine): https://zhuanlan.zhihu.com/p/611364321, https://zhuanlan.zhihu.com/p/261134624
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(config.num_epochs * len(train_loader)))
            
            models.append(model)
            optimizers.append(optimizer)
            schedulers.append(scheduler)
            
        # setup early stopper 
        early_stopper = EarlyStopper(patience=config.early_stopping_patience, delta=config.early_stopping_delta)
        
        # training and validation
        for epoch in range(1, config.num_epochs+1):
            if epoch < int(config.num_epochs*config.perctile_epochs_burnin):
                # for stability reason, apply the concurvity regularization after 5% training steps
                loss_train, metrics_train, R_train, im_train, rmse_d_train = _ensemble_train_epoch(epoch, criterion, metrics, optimizers, schedulers, models, device, train_loader, concurvity_regularization=False)
                loss_val, metrics_val, R_val, rmse_d_val = _ensemble_evaluate_epoch(criterion, metrics, models, device, val_loader, concurvity_regularization=False)
            else: 
                loss_train, metrics_train, R_train, im_train, rmse_d_train = _ensemble_train_epoch(epoch, criterion, metrics, optimizers, schedulers, models, device, train_loader, concurvity_regularization=True)
                loss_val, metrics_val, R_val, rmse_d_val = _ensemble_evaluate_epoch(criterion, metrics, models, device, val_loader, concurvity_regularization=True)
                
                # early stopping on the penalized loss
                if early_stopper.early_stop(loss_val):
                    print(f'[EPOCH={epoch}]: {train_loss_name}: {loss_train: .6f}, {val_loss_name}: {loss_val: .6f},  {train_metrics_name}: {math.sqrt(metrics_train): .6f}, {val_metrics_name}: {math.sqrt(metrics_val): .6f}, {train_R_name}: {R_train: .6f}, {val_R_name}: {R_val: .6f}')
                    log.info(f'[EPOCH={epoch}]: Early stopping with {val_loss_name}: {loss_val: .6f}, {val_metrics_name}: {metrics_val: .6f}, {val_R_name}: {R_val: .6f}')
                    break
                 
            # logging
            val_rmsed = {f'Val_RMSE_d_{idx+1}': rmse_d_val[idx].detach().item() for idx in range(in_features)}
            train_rmsed = {f'Train_RMSE_d_{idx+1}': rmse_d_train[idx].detach().item() for idx in range(in_features)}
            if config.wandb.use:
                wandb.log({
                    train_metrics_name: math.sqrt(metrics_train), 
                    val_metrics_name: math.sqrt(metrics_val), 
                    train_loss_name: loss_train, 
                    val_loss_name: loss_val, 
                    train_R_name: R_train, 
                    val_R_name: R_val, 
                    **val_rmsed, 
                    **train_rmsed,
                })

            if epoch % config.log_loss_frequency == 0: 
                print(f'[EPOCH={epoch}]: {train_loss_name}: {loss_train: .6f}, {val_loss_name}: {loss_val: .6f},  {train_metrics_name}: {math.sqrt(metrics_train): .6f}, {val_metrics_name}: {math.sqrt(metrics_val): .6f}, {train_R_name}: {R_train: .6f}, {val_R_name}: {R_val: .6f}, {train_rmsed}, {val_rmsed}')
                log.info(f'[EPOCH={epoch}]: {train_loss_name}: {loss_train: .6f}, {val_loss_name}: {loss_val: .6f},  {train_metrics_name}: {math.sqrt(metrics_train): .6f}, {val_metrics_name}: {math.sqrt(metrics_val): .6f}, {train_R_name}: {R_train: .6f}, {val_R_name}: {R_val: .6f}, {train_rmsed}, {val_rmsed}')
                
                if test_samples is not None:
                    _test_logging(config, test_samples, models)
                          
        if test_samples is not None: 
            _test_logging(config, test_samples, models)
            
        return models
    
    
def _test_logging(config, test_samples, models): 
    features, targets, feature_targets, feature_names = test_samples
    prediction_mean, feature_contribution_mean, prediction_var, feature_contribution_var = get_prediction(models, features)
    
    R_squared = adjusted_R_squared(features, targets, prediction_mean) 
    
    importance_fig = plot_feature_importance_errorbar(models, features, feature_names)
    recover_fig = plot_shape_function(features, feature_targets, feature_names, feature_contribution_mean, feature_contribution_var)
    # recover_fig = plot_recovered_functions(X, y, shape_functions, feature_contribution_mean, feature_contribution_var, center=False) # recover shape functions      
                    
    if config.wandb.use:
        wandb.log({
                'Recover_Function': wandb.Image(recover_fig),
                'Overall_Feature_Importance': wandb.Image(importance_fig), 
                'R_squared': R_squared.item()
        })
    else: 
        print(f'R_squared: {R_squared.item(): .4f}')
          

def _ensemble_train_epoch(
    epoch: int, 
    criterion, 
    metrics, 
    optimizers: torch.optim.Adam, 
    schedulers: torch.optim.lr_scheduler, 
    models: nn.Module, 
    device: str, 
    dataloader: torch.utils.data.DataLoader, 
    concurvity_regularization: bool=True
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
    concurvity_regularization: bool
        whether to apply concurvity regularization.
        
    Returns: 
    -----
    loss, metrics, measured concurvity (R_perp), estimated feature importance  
    
    """
    num_ensemble = len(models)
    in_features = len(dataloader.dataset[0][0])
    
    for model in models:
        model.train()
    
    losses, metrs, Rs = [0.0]*len(models), [0.0]*len(models), [0.0]*len(models) 
    rmseds = [0.0]*len(models)
    importance = [torch.zeros(in_features)]*len(models) # feature importance estimated on the training set 
        
    for features, target, feature_targets in tqdm(dataloader, desc=f"Epoch: {epoch}"):
        features, target, feature_targets = features.to(device), target.to(device), feature_targets.to(device)
        
        for idx, model in enumerate(models):  
            optimizer = optimizers[idx]
            scheduler = schedulers[idx]
            
            optimizer.zero_grad()
            preds, fnn_out = model(features)
            
            step_loss = criterion(preds, fnn_out, model, target, concurvity_regularization)
            step_loss.backward()
            optimizer.step()
            scheduler.step()
            
            losses[idx] += step_loss
            metrs[idx] +=  metrics(preds, target)
            Rs[idx] += concurvity(fnn_out)
            rmseds[idx] += rmse_d(fnn_out, feature_targets)
            importance[idx] += feature_importance(fnn_out)
                
    return sum(losses) / num_ensemble / len(dataloader), sum(metrs) / num_ensemble / len(dataloader), sum(Rs) / num_ensemble / len(dataloader), torch.stack(importance, dim=1).detach() / len(dataloader), torch.stack(rmseds, dim=0).sum(0) / num_ensemble / len(dataloader)


def _ensemble_evaluate_epoch(
    criterion, 
    metrics, 
    models: nn.Module, 
    device: str, 
    dataloader: torch.utils.data.DataLoader, 
    concurvity_regularization: bool=True
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
    
    losses, metrs, Rs = [0.0]*len(models), [0.0]*len(models), [0.0]*len(models) 
    rmseds = [0.0]*len(models)
    params, buffers = torch.func.stack_module_state(models) # all modules being stacked together must be the same, including the mode.
    
    for (X, y, feature_targets) in dataloader:
        X, y, feature_targets =  X.to(device), y.to(device), feature_targets.to(device)
            
        pred_map, fnn_map = torch.vmap(call_single_model, (0, 0, None))(params, buffers, X) # (num_ensemble, batch_size, out_features)
        for idx, model in enumerate(models):
            losses[idx] += criterion(pred_map[idx], fnn_map[idx], model, y, concurvity_regularization)
            metrs[idx] += metrics(pred_map[idx], y)
            Rs[idx] += concurvity(fnn_map[idx])
            rmseds[idx] += rmse_d(fnn_map[idx], feature_targets)
            
    return sum(losses) / num_ensemble / len(dataloader), sum(metrs) / num_ensemble / len(dataloader), sum(Rs) / num_ensemble / len(dataloader),  torch.stack(rmseds, dim=0).sum(0) / num_ensemble / len(dataloader)


