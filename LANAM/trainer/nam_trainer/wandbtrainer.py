"""trainer class for model training and evaluation"""
import random 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir))) # add parent folder to system paths

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
from .metrics import mae, mse
from .epoch import *
from LANAM.models import NAM
from LANAM.utils.plotting import *
from LANAM.config import Config

import copy


def setup_seeds(seed):
    """Set seeds for everything."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config: Config, 
          train_loader: DataLoader, 
          val_loader: DataLoader, 
          test_samples: Tuple=None, 
          ensemble=True, 
          use_wandb=False): 
        """trainer for the vanilla NAM.
        Args: 
        -----
        test_samples: Tuple(features, targets, feature_targets)
            samples for testing
        ensemble: bool
            whether to use ensemble members for uncertainty estimation
        use_wandb: bool
            whether to use W&B
        """
        # get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if use_wandb:
            # initialize sweeps
            run = wandb.init()
            # update configurations based on sweeps
            config.update(**wandb.config)
            config.hidden_sizes = [config.hidden_sizes]
            
        print(f"Configuration: {config}")
        # model ensembling
        num_ensemble = config.num_ensemble if ensemble else 1 
        seeds = [*range(num_ensemble)]
        
        # set up criterion and metrics
        criterion = lambda nam_out, fnn_out, model, targets: penalized_loss(config, nam_out, fnn_out, model, targets)
        metrics = lambda nam_out, targets: mse(nam_out, targets) if config.regression else accuracy(nam_out, targets)
        metrics_name = "MSE" if config.regression else "Accuracy"
        val_metrics_name = "Val_" + metrics_name
        train_metrics_name = "Train_" + metrics_name
        
        
        # set up model, optimizer, and criterion
        models = list()
        optimizers = list()
        for idx in range(num_ensemble): 
            setup_seeds(seeds[idx])
            model = NAM(
              config=config,
              name=f'NAM-{config.activation_cls}-{idx}',
              in_features=len(train_loader.dataset[0][0])).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.decay_rate) 
        
            models.append(model)
            optimizers.append(optimizer)
            
        # save the gradients information of the first model
        if use_wandb:
            for fnn in models[0].feature_nns: 
                wandb.watch(fnn, log_freq=config.log_loss_frequency) # log gradients; note that wandb.watch only supports nn.Module object.(not for ModuleList, Tuple, ...)print(f"Model summary: {model}")
        
        # loop the dataset multiple epochs
        for epoch in range(1, config.num_epochs+1):
            # forward + backward + optimize 
            loss_train, metrics_train = ensemble_train_epoch(criterion, metrics, optimizers, models, device, train_loader)
            loss_val, metrics_val = ensemble_evaluate_epoch(criterion, metrics, models, device, val_loader)
            
            if use_wandb:
                wandb.log({
                    metrics_name: metrics_val,
                })
                #wandb.log({
                #    "Train_Loss": loss_train, 
                #    "Val_Loss": loss_val, 
                #    train_metrics_name: metrics_train, 
                #    val_metrics_name: metrics_val, # same for swweep configuration and log 
                #})
            else:
                if epoch % config.log_loss_frequency == 0: 
                    print(f'[EPOCH={epoch}]: Train_Loss: {loss_train}, Val_Loss: {loss_val}, {train_metrics_name}: {metrics_train}, {val_metrics_name}: {metrics_val}')
            
            # https://docs.wandb.ai/ref/python/log
            # https://docs.wandb.ai/guides/track/limits, about the log frequency rule
            # fitting for individual features
            if test_samples is not None and epoch % config.log_loss_frequency == 0: 
                features, targets, feature_targets = test_samples
                pred_map, fnn_map = ensemble_test(models, features, targets, feature_targets)
                f_mu, f_mu_fnn, f_var, f_var_fnn = pred_map.mean(dim=0), fnn_map.mean(dim=0), pred_map.var(dim=0), fnn_map.var(dim=0) 
                fig = plot_recovered_functions(features, targets, feature_targets, f_mu_fnn, f_var_fnn)
                if use_wandb:
                    wandb.log({
                        'Recovery_functions': wandb.Image(fig),
                    })
                
        # save model in wandb.run.dir, then the file will be uploaded at the end of training.
        if use_wandb:
            torch.save(models[0].state_dict(), os.path.join(wandb.run.dir, f'model_0.pt'))
        #    for idx, m in enumerate(models):
        #        torch.save(m.state_dict(), os.path.join(wandb.run.dir, f'model_{idx}.pt'))
        
        print("Finished Training.")
        return model
        
