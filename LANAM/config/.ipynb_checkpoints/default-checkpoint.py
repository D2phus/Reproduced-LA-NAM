import torch

from .base import Config


def defaults() -> Config:
    config = Config(
        
        experiment_name="LANAM-Grid-2",
        data_path = 'LANAM/data/datasets',
        
        likelihood='regression',
        prior_sigma_noise=0.7,
        
        num_epochs=400,
        batch_size=128,
        
        ## logs
        wandb=False, 
        log_loss_frequency=100,
        
        # for tuning
        lr=1e-2,
        lr_hyp = 1e-1,
        n_epochs_burnin=50, 
        n_hypersteps=30, 
        marglik_frequency = 100,
        hidden_sizes=[64],  #hidden linear layers' size 
        activation=True, # use activation or not   
        activation_cls='gelu', 
    )

    return config