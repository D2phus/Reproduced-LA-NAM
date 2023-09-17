import torch

from .base import Config


def defaults() -> Config:
    config = Config(
        device =torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

        experiment_name="LANAM-Grid",
        data_path = 'LANAM/data/datasets',
        
        likelihood='regression',
        prior_sigma_noise=0.7,
        
        num_epochs=500,
        batch_size=256,
        
        ## logs
        wandb=False, 
        log_loss_frequency=100,
        
        # for tuning
        lr=1e-1,
        lr_hyp = 1e-1,
        n_epochs_burnin=50, 
        n_hypersteps=30, 
        marglik_frequency = 100,
        hidden_sizes=[64],  #hidden linear layers' size 
        activation=True, # use activation or not   
        activation_cls='gelu', 
    )

    return config


def nam_defaults() -> Config:
    config = Config(
        experiment_name="NAM-grid",
        seed=2023, 
        
        prior_sigma_noise=0.7,
        
        regression=True,
        use_dnn = False, # baseline 
        
        num_epochs=500,
        batch_size=256,
        shuffle=True, # shuffle the training set or not 
        early_stopping_patience=50,  
        decay_rate=0.005, # 0.005
        
        ## logs
        logdir="./output",
        wandb=False, 
        log_loss_frequency=100,
        
        # for tuning
        lr=1e-3,
        l2_regularization=1e-5, 
        output_regularization=0, # 1e-3
        dropout=0.2, # 0.1
        feature_dropout=0.05,  #0.1
        hidden_sizes=[64],  #hidden linear layers' size 
        activation_cls='relu',  ## hidden unit type
        
        num_ensemble=10, 
        
    )

    return config