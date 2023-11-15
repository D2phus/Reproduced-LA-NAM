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

from experiments.wandbconfig import WandbConfiguration 

import wandb 

# wandb_cfg = WandbConfiguration()
cfg = toy_default()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--concurvity', type=bool, help="whether to use correlated or uncorrelated features", default=True)
    args = parser.parse_args()
    
    # create W&B run
    wandb.login()
    wandb.finish()
    
    # setup dataset
    data = load_concurvity_data(sigma_1=0.05, sigma_2=0.5, num_samples=1000)
    train_dl, _, val_dl, _ = data.train_dataloaders()
    test_samples = data.get_test_samples()

    # data = load_nonlinearly_dependent_2D_examples(num_samples=1000, dependent_functions=lambda x: torch.sin(4*x)) # uncorrelated features 
    # train_dl, _, val_dl, _ = data.train_dataloaders()
    # test_samples = data.get_test_samples()

    # searching space
    parameters_list = {
        'early_stopping_patience': {
            'values': [20, 40, 60],
        },
        'concurvity_regularization': {
            'values': [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        },
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
        project='concurvity_regularization_GAM',
    )
    # training
    wandb.agent(sweep_id, 
            function=partial(train, 
                             config=cfg, 
                             train_loader=train_dl, 
                             val_loader=val_dl, 
                             test_samples=test_samples, 
                             ensemble=True, 
                             use_wandb=True))
