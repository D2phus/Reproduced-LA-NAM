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

from laplace import Laplace
from laplace import marglik_training as lamt
from laplace.curvature.backpack import BackPackGGN

from LANAM.models import LaNAM
from LANAM.trainer.marglik_training import *
from LANAM.trainer.wandb_train import wandb_training
from LANAM.data.generator import *
from LANAM.data.toydataset import ToyDataset

from LANAM.config.default import defaults
from LANAM.utils.plotting import * 
from LANAM.utils.wandb import *

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
    parser.add_argument('--activation_cls', type=str, help="Activation function class", default='relu')
    parser.add_argument('--load_data', type=bool, help="Fetch data from W&B or generate new data", default=True)
    
    # set configuration parameters
    args = parser.parse_args()
    cfg = defaults()
    
    # create W&B run
    wandb.login()
    wandb.finish()
    # setup dataset
    dataset = setup_dataset(cfg, load_data=args.load_data)
    
    parameters_list = {
        'lr': {
            'values': [0.1, 0.01, 0.001]
        }, 
        'hidden_sizes': {
            'values':[64] # single hidden layer
        }, 
        'activation_cls': {
            'values': [args.activation_cls]
        }, 
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
                function=partial(wandb_training, 
                                 config=cfg, 
                                 dataset=dataset
                                ),)
    