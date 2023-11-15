import torch

from .base import Config


def defaults() -> Config:
    config = Config(
        device =torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

        experiment_name="LANAM-Grid",
        data_path = 'LANAM/data/datasets',
        
        likelihood='regression',
        prior_sigma_noise=0.7,
        
        num_epochs=1000,
        batch_size=256,
        
        ## logs
        wandb=False, 
        log_loss_frequency=250,
        
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
        device =torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

        seed=2023, 
        
        prior_sigma_noise=0.7,
        
        regression=True,
        likelihood='regression', 
        use_dnn = False, # baseline 
        
        num_epochs=250,
        batch_size=256,
        shuffle=True, # shuffle the training set or not 
        early_stopping_patience=50,  
        decay_rate=0, # 0.005
        
        ## logs
        logdir="./output",
        wandb=False, 
        log_loss_frequency=250,
        
        # for tuning
        lr=1e-3,
        l2_regularization= 0, # 1e-5, 
        output_regularization=0, # 1e-3
        concurvity_regularization=0.5, 
        dropout= 0, # 0.2, # 0.1
        feature_dropout= 0, # 0.05,  #0.1
        hidden_sizes=[128, 128, 128],  #hidden linear layers' size 
        activation_cls='gelu',  ## hidden unit type
        
        num_ensemble=10, 
        
    )

    return config

def toy_default() -> Config:
    config = Config(
        experiment_name="NAM-grid",
        device =torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

        seed=2023, 
        
        # dataset, tasks
        prior_sigma_noise=0.7,
        
        regression=True,
        likelihood='regression', 
        use_dnn = False, # baseline 
        
        # model architecture 
        hidden_sizes=[128, 128, 128],  # number of units in each hidden layer 
        activation_cls='gelu',  ## hidden unit type 
        
        # training settings
        lr=1e-3,
        num_epochs=300,
        batch_size=128,
        shuffle=True, # shuffle the training set or not 
        early_stopping_patience=20,  
        decay_rate=0, 
        
        ## logs
        logdir="./output",
        wandb=False, 
        log_loss_frequency=30,
        
        # regularizations used in vanilla nam.
        l2_regularization= 0,
        output_regularization=0, 
        dropout= 0, 
        feature_dropout= 0, 
        
        # concurvity regularization
        concurvity_regularization=0.5, 
        perctile_epochs_burnin=0.05, # start concurvity regularization after a certain proportion of training steps
        
        # model ensembling 
        num_ensemble=5, 
    )

    return config
