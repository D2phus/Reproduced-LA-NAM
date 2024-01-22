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

@hydra.main(config_path=config_path, config_name='concurvity_hpo.yaml') # decorator to introduce contents from the configuration file
def ensemble(cfg: DictConfig):
    cfg.wandb.use = False
    # cfg.hsic_regularization = 533.999
    # cfg.concurvity_regularization = 0.1
    log.info(f'Configuration : {cfg}')
    
    if cfg.wandb.use: 
        wandb.login()
        wandb.finish()
    
    name_of_dataset = cfg.dataset.name
    name_of_model = cfg.model.name 
    orig_cwd = hydra.utils.get_original_cwd() # get the original directory instead of the hydra output directory
    
    # set up dataset
    if name_of_dataset == 'california_housing': 
        data = load_california_housing_data(data=f'{orig_cwd}/data/datasets/california-housing-dataset/california_housing.csv')
    elif name_of_dataset == 'concurvity': 
        data = concurvity_data = load_concurvity_data()
    else:
        raise ValueError('Invalid name of dataset.')
        
    train_dl, _, val_dl, _ = data.train_dataloaders()
    test_dl, _ = data.test_dataloaders()
    samples = data.get_test_samples()
        
    if name_of_model == 'nam':
        model = train(config=cfg, train_loader=train_dl, val_loader=val_dl, ensemble=True)
    else: 
        raise ValueError('Invalid name of model.')
        
    # save models
    for idx, m in enumerate(model): 
        torch.save(m.state_dict(), f'model_{idx}.pt')
        
    # save feature importance
    feature_importance = plot_feature_importance_errorbar(model, samples, width=0.5)
    if cfg.wandb.use: 
        wandb.log({
             'Feature_Importance': wandb.Image(feature_importance)
         })
    feature_importance.savefig('feature_importance.png')
    

if __name__ == '__main__': 
    ensemble()
