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
from LANAM.utils.plotting import *
from LANAM.utils.correlation import * 
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
    
    run = wandb.init(
    # Set the project where this run will be logged
    project='accuracy_versus_concurvity_GAM')
    
    
    # setup dataset
    data = load_concurvity_data(sigma_1=0.05, sigma_2=0.5, num_samples=1000)
    # data = load_nonlinearly_dependent_2D_examples(num_samples=1000, dependent_functions=lambda x: torch.sin(2*x)) # uncorrelated features 
    # generate_funcs =[lambda x: x, lambda x: torch.zeros_like(x)]
    # data = load_multicollinearity_data(generate_functions=generate_funcs, x_lims=(-1, 1), num_samples=1000, sigma=0, sampling_type='uniform')
    
    train_dl, _, val_dl, _ = data.train_dataloaders()
    test_samples = data.get_test_samples()
    
    # criterion 
    metrics = lambda logits, targets: (((logits.view(-1) - targets.view(-1)).pow(2)).sum() / targets.numel()).item()
    
    base = [1e-6, 1e-5,1e-4, 1e-3, 1e-2, 1e-1]
    lams = [0.0]
    for b in base: 
        lams += [b*x for x in range(1, 10)]
    lams += [1.0]
    
    acc_list, concur_list, lam_list = list(), list(), list()

    for lam in lams: 
        lam_list.append(lam)
        cfg.concurvity_regularization = lam
        
        # train model 
        model = train(config=cfg, train_loader=train_dl, val_loader=val_dl, ensemble=True)
        # testing
        prediction_mean, feature_contribution_mean, prediction_mean, feature_contribution_var = get_prediction(model, test_samples)
        
        # measured concurvity
        r = concurvity(feature_contribution_mean)
        concur_list.append(r)
        # accuracy
        f, y = prediction_mean.detach().flatten(), test_samples[1].detach().flatten() # NOTE the shape
        m = metrics(f, y)
        acc_list.append(m)
        
        # logging metrics
        wandb.log({
            "Lambda": lam, 
            "Measured_Concurvity": r, 
            "Accuracy (MSE)": m, 
        })
    
    # accuracy and measured concurvity trade-off 
    fig = plot_concurvity_versus_accuracy(lam_list, acc_list, concur_list)
    wandb.log({
        "Accuracy_vs_Concurvity_Fig": wandb.Image(fig)
    })
