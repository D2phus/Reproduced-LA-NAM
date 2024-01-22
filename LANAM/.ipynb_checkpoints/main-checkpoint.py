import hydra 
from omegaconf import DictConfig
import yaml
import os 
import logging 

import sys 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir))) # add `LANAM` to system paths
from LANAM.data import * 
from LANAM.trainer import * 
from LANAM.utils import * 

import wandb

log = logging.getLogger(__name__)
config_path = 'config'

@hydra.main(config_path=config_path, config_name='concurvity_train.yaml') # decorator to introduce contents from the configuration file
def tuning(cfg: DictConfig):
    # cfg.wandb.use = True 
    if cfg.wandb.use: 
        wandb.login()
        wandb.finish()
    
    name_of_dataset = cfg.dataset.name
    name_of_model = cfg.model.name 
    orig_cwd = hydra.utils.get_original_cwd() # get the original directory instead of the hydra output directory
    
    # set up dataset
    if name_of_dataset == 'california_housing': 
        data = load_california_housing_data(config=cfg, data=f'{orig_cwd}/data/datasets/california-housing-dataset/california_housing.csv')
    elif name_of_dataset == 'concurvity': 
        data = concurvity_data = load_concurvity_data(config=cfg, num_samples=10000)
    else:
        raise ValueError('Invalid name of dataset.')
        
    train_dl, _, val_dl, _ = data.train_dataloaders()
    test_dl, _ = data.test_dataloaders()
    samples = data.get_test_samples()
        
    if name_of_model == 'nam':
        model = train(config=cfg, train_loader=train_dl, val_loader=val_dl, ensemble=False)
    else: 
        raise ValueError('Invalid name of model.')
        
    
if __name__ == '__main__': 
    tuning()
