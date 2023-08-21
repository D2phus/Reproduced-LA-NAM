import time 
from copy import deepcopy
from .marglik_training import *
from LANAM.models import LaNAM
from LANAM.utils.plotting import *
from LANAM.config.default import defaults

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils import parameters_to_vector
import numpy as np

from laplace.curvature import BackPackGGN
from laplace import Laplace

import wandb

def wandb_training(config,
                   train_loader, 
                   loader_fnn,
                   testset, 
                   ): 
    """Hyper-parameter tuning with W&B."""
    run = wandb.init()
    
    config.update(**wandb.config)
    config.hidden_sizes = [config.hidden_sizes]
    print(f'Configuration: \n {config}')
    
    likelihood = config.likelihood
    optimizer_kwargs = {'lr': config.lr}
    lr_hyp = config.lr_hyp
    n_epochs_burnin = config.n_epochs_burnin
    n_hypersteps = config.n_hypersteps
    marglik_frequency = config.marglik_frequency
    
    in_features = testset.in_features
    model = LaNAM(config=config, name=f'LA-NAM-{config.activation_cls}', in_features=in_features)
    
    print(f'Model summary: \n {model}')
    
    model, margliks, losses, perfs = marglik_training(model, train_loader, loader_fnn, use_wandb=True, testset=testset, likelihood=likelihood, optimizer_kwargs=optimizer_kwargs, lr_hyp=lr_hyp, n_epochs_burnin=n_epochs_burnin, n_hypersteps=n_hypersteps, marglik_frequency=marglik_frequency)
    