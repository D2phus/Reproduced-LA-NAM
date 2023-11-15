import os
from functools import partial
import argparse
import uuid 
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import matplotlib.pyplot as plt 
import numpy as np

import sys 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir))) # add `LANAM` to system paths
print(sys.path)

from LANAM.models import NAM 
from LANAM.trainer.nam_trainer import *
from LANAM.config.default import toy_default
from LANAM.data.dataset import *
from LANAM.utils import *
from LANAM.utils.correlation import pairwise_correlation, feature_importance

from experiments.wandbconfig import WandbConfiguration 

import wandb 

# wandb_cfg = WandbConfiguration()
cfg = toy_default()

def multicolinearity(config, ensemble, use_wandb):
    # setup dataset
    scale = None
    if use_wandb: 
        run = wandb.init()
        scale = wandb.config['scale']
        
    if scale is None: 
        scale = 1
        
    generate_function = [lambda x: torch.zeros_like(x), lambda x: x]
    data = load_nonlinearly_dependent_2D_examples(num_samples=1000, 
                                             
                                                  generate_functions=generate_function, 
                                                  dependent_functions=lambda x: scale*x) 
    
    train_dl, _, val_dl, _ = data.train_dataloaders()
    test_samples = data.get_test_samples()

    # untransformed features 
    untransformed_feature_im = feature_importance(data.features)
    untransformed_feature_corr = pairwise_correlation(data.features)
    print(f'untransformed feature importance: {untransformed_feature_im}')
    print(f'corr(X1, X2): {untransformed_feature_corr[0][1]: .6f}')
    
    train(cfg, train_dl, val_dl, test_samples, ensemble, use_wandb)
    
    
if __name__ == "__main__":
    # create W&B run
    wandb.login()
    wandb.finish()
    
    # searching space
    parameters_list = {
        'concurvity_regularization': {
            'values': [0, 0.01, 0.1, 1]
        },
        'scale': {
            'values': [0.01, 0.1, 0.5, 1, 2, 5, 10]
        }
    }
    # wandb sweep configuration 
    sweep_configuration = {
        'method': 'grid', 
        'name': 'sweep',
        'parameters': parameters_list, 
    }
    # initialize the sweep 
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='NAM_preference_multicolinearity',
    )
    # training
    wandb.agent(sweep_id, 
            function=partial(multicolinearity, 
                             config=cfg, 
                             ensemble=True, 
                             use_wandb=True))
