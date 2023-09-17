import time
import datetime
import os
from functools import partial
import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import numpy as np

import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from LANAM.models import NAM 
from LANAM.trainer.nam_trainer import *
from LANAM.config.default import nam_defaults
from LANAM.data.generator import *
from LANAM.utils.wandb import *
from LANAM.utils.plotting import *

import wandb 

data_project_name = 'Datasets'
log_artifact_name = 'concurvity-7'
table_name = 'concurvity-7'
load_artifact_name = 'concurvity-7:v0'

project_name = 'LANAM-concurvity'

def setup_dataset(cfg, load_data):
    if load_data:
        dataset = load_LANAMSyntheticDataset(data_project_name, load_artifact_name, table_name)
    else:
        # construct dataset from scratch
        dataset = load_synthetic_data(sigma=0.7)
        # log dataset to W&B
        log_LANAMSyntheticDataset(dataset,  data_project_name, log_artifact_name, table_name)
    
    return dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation_cls', type=str, help="activation function", default='gelu')
    parser.add_argument('--hidden_sizes', type=int, help="size of the single hidden layer", default=64)
    parser.add_argument('--load_data', type=bool, help="load data from W&B", default=True)
    args = parser.parse_args()
    
    cfg = nam_defaults()
    # create W&B run
    wandb.login()
    wandb.finish()
    # setup dataset
    dataset = setup_dataset(cfg, load_data=args.load_data)
    
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
        'values': [args.hidden_sizes]
    }

}
    sweep_configuration = {
        'method': 'grid', 
        'name': 'sweep',
        'parameters': parameters_list, 
    }
    # initialize the sweep 
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project=project_name,
    )
    # training
    wandb.agent(sweep_id, 
            function=partial(train, 
                             config=cfg, 
                             dataset=dataset,
                             ensemble=True, 
                             use_wandb=True))
