import time
import datetime
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

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir))) # add `LANAM` to system paths

from LANAM.models import NAM 
from LANAM.trainer.nam_trainer import *
from LANAM.config.default import nam_defaults
from LANAM.data.generator import *
from LANAM.utils.wandb import *
from LANAM.utils.plotting import *

from experiments.wandbconfig import WandbConfiguration 

import wandb 

wandb_cfg = WandbConfiguration()
cfg = nam_defaults()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation_cls', type=str, help="activation function", default='gelu')
    # parser.add_argument('--hidden_sizes', type=int, help="size of the single hidden layer", default=64)
    parser.add_argument('--load_data', type=bool, help="load data from W&B", default=True)
    args = parser.parse_args()
    
    # create W&B run
    wandb.login()
    wandb.finish()
    
    # setup dataset
    if args.load_data: 
        ## fetch dataset from wandb
        dataset = load_dataset(project_name=wandb_cfg.data_project_name, artifact_or_name=wandb_cfg.load_artifact_name, table_name=wandb_cfg.table_name)
    else: 
        ## build dataset from scratch
        dataset = load_concurvity_data()
        # log dataset to wandb
        table_name = uuid.uuid4().hex
        log_dataset(dataset=dataset, project_name=wandb_cfg.data_project_name, artifact_name=wandb_cfg.log_artifact_name, table_name=table_name)
    
    # searching space
    parameters_list = {
        'lr': {
            'values': [0.001, 0.01]
        }, 
        'output_regularization': {
            'values': [0, 0.001]
        }, 
        'dropout':  {
            'values': [0, 0.2]
        }, 
        'feature_dropout': {
            'values': [0, 0.05]
        }, 
        'activation_cls':  {
            'values': [args.activation_cls]
        }, 
        'hidden_sizes': {
            'values': [64, 1024]
        }

    }
    # wandb sweep configuration 
    sweep_name = wandb_cfg.load_artifact_name + '_' + args.activation_cls if args.load_data else wandb_cfg.log_artifact_name + '_' + args.activation_cls
    sweep_configuration = {
        'method': 'grid', 
        'name': 'sweep',
        'parameters': parameters_list, 
    }
    # initialize the sweep 
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project=wandb_cfg.experiment_project_name,
    )
    # training
    wandb.agent(sweep_id, 
            function=partial(train, 
                             config=cfg, 
                             dataset=dataset,
                             ensemble=True, 
                             use_wandb=True))
