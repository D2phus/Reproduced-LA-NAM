import os 
import torch 
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Checkpointer: 
    """
    A torch model load/save wrapper
    """
    def __init__(self, 
                model: nn.Module, 
                config) -> None:
        self._model = model 
        self._config = config
        
        self._ckpts_dir = os.path.join(config.logdir, "ckpts")
        os.makedirs(self._ckpts_dir, exist_ok=True)
        
    def save(self, 
            epoch: int) -> str: 
        """ 
        Save the model to file 'ckpts_dir/epoch/model.pt'
        """
        ckpts_path = os.path.join(self._ckpts_dir, "model-{}.pt".format(epoch))
        torch.save(self._model.state_dict(), ckpts_path)
        return ckpts_path
    
    def load(self, 
            epoch: int) -> nn.Module: 
        """ 
        Load the model from file 'ckpts_dir/epoch/model.pt'
        """
        ckpts_path = os.path.join(self._ckpts_dir, "model-{}.pt".format(epoch))
        self._model.load_state_dict(torch.load(ckpts_path, map_location=device))
        return self._model

