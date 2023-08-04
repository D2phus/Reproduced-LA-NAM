import torch

from .base import Config


def defaults() -> Config:
    config = Config(
        # device='cuda' if torch.cuda.is_available() else 'cpu',
        
        # seed=2023, 
        experiment_name="nam-sparse-features-2",
        
        regression=True,
        
        num_epochs=400,
        batch_size=128,
        
        ## logs
        wandb=False, 
        log_loss_frequency=100,
        
        # for tuning
        lr=1e-2,
        hidden_sizes=[64],  #hidden linear layers' size 
        activation=True, # use activation or not   
        activation_cls='gelu', 
    )

    return config