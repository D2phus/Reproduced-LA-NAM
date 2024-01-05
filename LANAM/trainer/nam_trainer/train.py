"""trainer class for model training and evaluation"""
import random 
import copy
import math 

from types import SimpleNamespace
from typing import Mapping, Sequence, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt 

import wandb

from .losses import penalized_loss
from .metrics import accuracy
from .metrics import mae, mse, rmse
from .epoch import *


from LANAM.models import NAM
from LANAM.utils.plotting import *
from LANAM.utils.earlystopping import EarlyStopper
from LANAM.config import Config


def train(config: Config, 
          train_loader: DataLoader, 
          val_loader: DataLoader, 
          test_samples: Tuple=None, 
          ensemble=True, 
          use_wandb=False): 
        """trainer for the ensembled vanilla NAM. 
        Args: 
        -----
        test_samples: Tuple(features, targets, feature_targets)
            samples for testing
        ensemble: bool
            whether to use ensemble members for uncertainty estimation
        use_wandb: bool
            whether to use W&B
        
        Returns: 
        -----
        models: List[nn.Module]
            ensemble models.
        """
        # get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        in_features = len(train_loader.dataset[0][0])
        if use_wandb:
            # initialize sweeps
            run = wandb.init()
            # update configurations based on sweeps
            config.update(**wandb.config)
            # type check
            if not isinstance(config.hidden_sizes, list): 
                config.hidden_sizes = [config.hidden_sizes]
            
        # model ensembling
        num_ensemble = config.num_ensemble if ensemble else 1 
        seeds = [*range(num_ensemble)]
        
        # set up criterion and metrics
        criterion = lambda nam_out, fnn_out, model, targets, conc_reg: penalized_loss(config, nam_out, fnn_out, model, targets, conc_reg)
        metrics = lambda nam_out, targets: mse(nam_out, targets) if config.regression else accuracy(nam_out, targets)
        
        # annotation
        metrics_name = "RMSE" if config.regression else "Accuracy"
        train_metrics_name, val_metrics_name = 'Train_' + metrics_name, 'Val_' + metrics_name
        train_loss_name, val_loss_name = 'Train_Loss', 'Val_Loss'
        R_name = 'R_perp'
        train_R_name, val_R_name = 'Train_' + R_name, 'Val_' + R_name
        
        # set up model, optimizer, and criterion for ensemble members
        models = list()
        optimizers = list()
        schedulers = list()
        schedulers = list()
        for idx in range(num_ensemble): 
            setup_seeds(seeds[idx])
            model = NAM(
              config=config,
              name=f'NAM-{config.activation_cls}-{idx}',
              in_features=in_features)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.decay_rate) 
            # https://zhuanlan.zhihu.com/p/611364321, https://zhuanlan.zhihu.com/p/261134624
            # decrease learning rate during the whole training process: half the cosine. 
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(config.num_epochs * len(train_loader)))
            
            models.append(model)
            optimizers.append(optimizer)
            schedulers.append(scheduler)
            
        # setup early stopper 
        early_stopper = EarlyStopper(patience=config.early_stopping_patience, delta=config.early_stopping_delta)
        
        # training and validation
        for epoch in range(1, config.num_epochs+1):
            # forward + backward + optimize 
            if epoch < int(config.num_epochs*config.perctile_epochs_burnin):
                # for stability reason, only after 5% training steps, the concurvity regularization will be applied. 
                loss_train, metrics_train, R_train, im_train = ensemble_train_epoch(criterion, metrics, optimizers, schedulers, models, device, train_loader, conc_reg=False)
                loss_val, metrics_val, R_val = ensemble_evaluate_epoch(criterion, metrics, models, device, val_loader, conc_reg=False)
            else: 
                loss_train, metrics_train, R_train, im_train = ensemble_train_epoch(criterion, metrics, optimizers, schedulers, models, device, train_loader, conc_reg=True)
                loss_val, metrics_val, R_val = ensemble_evaluate_epoch(criterion, metrics, models, device, val_loader, conc_reg=True)
                
                # early stopping on the penalized loss
                if early_stopper.early_stop(loss_val):  
                    print(f'[EPOCH={epoch}]: Early stopping with {val_loss_name}: {loss_val: .6f}, {val_metrics_name}: {metrics_val: .6f}, {val_R_name}: {R_val: .6f}')
                    break
                 
            # logging
            if use_wandb:
                wandb.log({
                    train_metrics_name: math.sqrt(metrics_train), 
                    val_metrics_name: math.sqrt(metrics_val), 
                    train_loss_name: loss_train, 
                    val_loss_name: loss_val, 
                    train_R_name: R_train, 
                    val_R_name: R_val, 
                })

            if epoch % config.log_loss_frequency == 0: 
                print(f'[EPOCH={epoch}]: {train_loss_name}: {loss_train: .6f}, {val_loss_name}: {loss_val: .6f},  {train_metrics_name}: {metrics_train: .6f}, {val_metrics_name}: {metrics_val: .6f}, {train_R_name}: {R_train: .6f}, {val_R_name}: {R_val: .6f}')
                
                # fitting for each shape functoin 
                if test_samples is not None:
                    test_logging(test_samples, models, use_wandb)
                          
        if test_samples is not None: 
            test_logging(test_samples, models, use_wandb)
            
        return models
    
    
def setup_seeds(seed):
    """Set seeds for everything."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_logging(test_samples, models, use_wandb): 
    X, y, shape_functions, names = test_samples
    
    prediction_mean, feature_contribution_mean, prediction_mean, feature_contribution_var = get_prediction(models, test_samples)
    
    importance_fig = plot_feature_importance(models, test_samples)
    # importance_errorbar_fig = plot_feature_importance_errorbar(models, test_samples)
    # pairwise_corr_fig = plot_pairwise_contribution_correlation(models, test_samples, (0,1))

    R_squared = adjusted_R_squared(X, y, prediction_mean)
    
    # recover shape function
    recover_fig = plot_recovered_functions(X, y, shape_functions, feature_contribution_mean, feature_contribution_var, center=False)       
                    
    if use_wandb:
        wandb.log({
                'Recover_Function': wandb.Image(recover_fig),
                'Overall_Feature_Importance': wandb.Image(importance_fig), 
                'R_squared': R_squared.item()
        })
    else: 
        print(f'R_squared: {R_squared.item(): .4f}')
        
        