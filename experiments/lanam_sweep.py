import time
import datetime
import os
from functools import partial
import argparse
import uuid 
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import matplotlib.pyplot as plt 

from laplace import Laplace
from laplace import marglik_training as lamt
from laplace.curvature.backpack import BackPackGGN

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir))) # add `LANAM` to system paths

from LANAM.models import LaNAM
from LANAM.trainer import marglik_training, wandb_training
from LANAM.data.generator import *
from LANAM.data.base import LANAMDataset, LANAMSyntheticDataset

from LANAM.config.default import defaults
from LANAM.utils.plotting import * 
from LANAM.utils.wandb import *
from experiments.wandbconfig import WandbConfiguration 

import wandb

wandb_cfg = WandbConfiguration()
cfg = defaults()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation_cls', type=str, help="Activation function class", default='relu')
    parser.add_argument('--load_data', type=bool, help="Fetch data from W&B or generate new data", default=True)
    
    # set configuration parameters
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
            'values': [0.1, 0.01, 0.001]
        }, 
        'hidden_sizes': {
            'values':[64, 1024] # single hidden layer of 64 / 1024 units
        }, 
        'activation_cls': {
            'values': [args.activation_cls]
        }, 
    }
    # wandb sweep configuration 
    sweep_name = wandb_cfg.load_artifact_name + '_' + args.activation_cls if args.load_data else wandb_cfg.log_artifact_name + '_' + args.activation_cls
    sweep_configuration = {
        'method': 'grid',
        'name': sweep_name, # name for this sweep, display in UI
        'parameters': parameters_list, 
    }
    # initialize the sweep 
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project=wandb_cfg.experiment_project_name,
    )
    # training
    wandb.agent(sweep_id, 
                function=partial(wandb_training, 
                                 config=cfg, 
                                 dataset=dataset
                                ),)
    