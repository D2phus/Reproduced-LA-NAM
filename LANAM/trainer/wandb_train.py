import time 
from copy import deepcopy

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils import parameters_to_vector
import numpy as np

from LANAM.trainer import marglik_training
from LANAM.models import LaNAM
from LANAM.utils.plotting import *
from LANAM.config.default import defaults

from laplace.curvature import BackPackGGN

import wandb

def wandb_training(config,
                   dataset,
                   ): 
    """Hyper-parameter tuning with W&B."""
    run = wandb.init()
    
    config.update(**wandb.config)
    config.hidden_sizes = [config.hidden_sizes]
    print(f'Configuration: \n {config}')
    
    # data
    train_loader, loader_fnn, _, _ = dataset.train_dataloaders()
    test_loader, _ = dataset.test_dataloaders()
    test_samples = dataset.get_test_samples()
    
    likelihood = config.likelihood
    optimizer_kwargs = {'lr': config.lr}
    lr_hyp = config.lr_hyp
    n_epochs_burnin = config.n_epochs_burnin
    n_hypersteps = config.n_hypersteps
    marglik_frequency = config.marglik_frequency
    
    in_features = dataset.in_features
    model = LaNAM(config=config, name=f'LA-NAM-{config.activation_cls}', in_features=in_features)
    
    print(f'Model summary: \n {model}')
    
    model, margliks, losses, perfs = marglik_training(model, 
                                                      train_loader, 
                                                      loader_fnn, 
                                                      test_loader,
                                                      likelihood=likelihood,
                                                      use_wandb=True, 
                                                      test_samples=test_samples,
                                                      optimizer_kwargs=optimizer_kwargs, 
                                                      lr_hyp=lr_hyp, 
                                                      n_epochs_burnin=n_epochs_burnin, 
                                                      n_hypersteps=n_hypersteps, 
                                                      marglik_frequency=marglik_frequency, 
                                                      plot_recovery=True)
    