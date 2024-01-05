import hydra 
from omegaconf import DictConfig
import os 
import logging 

log = logging.getLogger(__name__)

@hydra.main(config_path='config', config_name='config') # decorator to introduce contents from the configuration file
def func(cfg: DictConfig): 
    # hydra executes main.py in the output directory './outputs/ / /', so '../../../file.txt' should be used if main.py relies on an external file.
    log.info({cfg.lr})
    
if __name__ == '__main__': 
    func()