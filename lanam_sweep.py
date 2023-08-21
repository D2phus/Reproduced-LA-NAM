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


def setup_dataset(cfg, load_data):
    if load_data:
        # fetch data and construct dataset from W&B
        processed_datasets = preprocess_and_log(project_name='LANAM-grid-basic-synthetic', job_type='dataset', artifact_or_name='synthetic-4:v0')
        trainset = processed_datasets['training']
        valset = processed_datasets['validation']
        testset = processed_datasets['test']
    else:
        # construct dataset from scratch
        gen_funcs, gen_func_names = task()
        in_features = len(gen_funcs)
        sigma = cfg.prior_sigma_noise
        print(sigma)
        trainset = ToyDataset(gen_funcs, gen_func_names, num_samples=1000, sigma=sigma)
        valset = ToyDataset(gen_funcs, gen_func_names, num_samples=200, sigma=sigma)
        testset = ToyDataset(gen_funcs, gen_func_names, num_samples=50, use_test=True)
    return trainset, valset, testset

if __name__ == "__main__":
    # mem=900M, 
    parser = argparse.ArgumentParser()
    #parser.add_argument('--activation_cls', type=str, help="Activation function class", default='relu')
    parser.add_argument('--load_data', type=bool, help="Fetch data from W&B or generate new data", default=True)
    
    # set configuration parameters
    args = parser.parse_args()
    cfg = defaults()
    #cfg.activation_cls = args.activation_cls
    
    # create W&B run
    wandb.login()
    wandb.finish()
    # setup dataset
    trainset, valset, testset = setup_dataset(cfg, load_data=args.load_data)
    train_loader, train_loader_fnn = trainset.loader, trainset.loader_fnn
    val_loader, val_loader_fnn = valset.loader, valset.loader_fnn
    X_test, y_test = testset.X, testset.y
    
    parameters_list = {
        'lr': {
            'values': [0.1]
        }, 
        'hidden_sizes': {
            'values':[1024] # single hidden layer
        }, 
        'activation_cls': {
            'values': ['exu']
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
        project='LANAM-grid-basic-synthetic-3.0',
    )
    # training
    wandb.agent(sweep_id, 
                function=partial(wandb_training, 
                config=cfg, 
                train_loader=train_loader, 
                loader_fnn=train_loader_fnn,
                testset=testset)) # specify the maximum number of runs
    
