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

from laplace.curvature import BackPackGGN

import wandb

def wandb_training(config,
                   dataset,
                   ): 
    """Hyper-parameter tuning with W&B."""
    run = wandb.init()
    
    config.update(**wandb.config)
    if not isinstance(config.hidden_sizes, list):
        config.hidden_sizes = [config.hidden_sizes]
    print(f'Configuration: \n {config}')
    
    # data
    train_loader, loader_fnn, val_loader, _ = dataset.train_dataloaders()
    test_samples = dataset.get_test_samples()
    
    likelihood = config.likelihood
    optimizer_kwargs = {'lr': config.lr}
    n_epochs = config.num_epochs
    lr_hyp = config.lr_hyp
    n_epochs_burnin = config.n_epochs_burnin
    n_hypersteps = config.n_hypersteps
    marglik_frequency = config.marglik_frequency
    concurvity_regularization = config.concurvity_regularization
    log_loss_frequency = config.log_loss_frequency 
    perctile_epochs_burnin = config.perctile_epochs_burnin
    prior_prec_init = config.prior_prec_init
    
    in_features = dataset.in_features
    
    model = LaNAM(config=config, name=f'LA-NAM-{config.activation_cls}', in_features=in_features, hessian_structure='kron', subset_of_weights='last_layer')
    
    print(f'Model summary: \n {model}')
    
    model, margliks, losses, perfs = marglik_training(model, 
                                                      train_loader, 
                                                      loader_fnn, 
                                                      val_loader,
                                                      likelihood=likelihood,
                                                      use_wandb=True, 
                                                      test_samples=test_samples,
                                                      optimizer_kwargs=optimizer_kwargs, 
                                                      n_epochs=n_epochs,
                                                      lr_hyp=lr_hyp, 
                                                      n_epochs_burnin=n_epochs_burnin, 
                                                      n_hypersteps=n_hypersteps, 
                                                      marglik_frequency=marglik_frequency, 
                                                      concurvity_regularization=concurvity_regularization, 
                                                      reg_perctile_epochs_burnin=perctile_epochs_burnin,
                                                      plot_recovery=True, 
                                                      log_loss_frequency=log_loss_frequency,
                                                      prior_prec_init=prior_prec_init,
                                                     )
    