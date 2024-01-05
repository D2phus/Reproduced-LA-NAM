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

from LANAM.models import LaNAM, NAM, BayesianLinearRegression
from LANAM.config import defaults
from LANAM.trainer import *
from LANAM.trainer.nam_trainer import *
from LANAM.data import *

from LANAM.utils.plotting import * 
from LANAM.utils.output_filter import OutputFilter
from LANAM.utils.wandb import *
from LANAM.utils.initialization import *
from experiments.wandbconfig import WandbConfiguration 

import wandb 

cfg = defaults()
    
def func(config, use_wandb): 
    rho=0
    seed=0
    if use_wandb: 
        run = wandb.init()
        rho = wandb.config['rho']
        seed = wandb.config['seed']
    
    setup_seeds(seed)
    data = interaction_example(rho=rho, min_max=False)
    wandb_training(config=cfg, dataset=data)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, help="regularization parameter exponent on 0.1", default=1)
    args = parser.parse_args()
    
    # create W&B run
    wandb.login()
    wandb.finish()

    # searching space
    if args.exp == 0:
        lams = [0, 1]
    else:
        exp = 0.1**(args.exp) 
        lams = [exp*x for x in [1, 2, 4, 6, 8]]
    
    parameters_list = {
        'concurvity_regularization': {
            'values': lams
        },
        'rho': {
            'values': [0, 0.7, 0.9, 0.95, 0.99]
        },
        'seed': {
            'values': [*range(3)] 
        },
        'prior_prec_init': {
            'values': [[1, 10]]
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
        project='interaction_example_1',
    )
    
    wandb.agent(sweep_id, 
                function=partial(func, 
                                 config=cfg, 
                                 use_wandb=True))
