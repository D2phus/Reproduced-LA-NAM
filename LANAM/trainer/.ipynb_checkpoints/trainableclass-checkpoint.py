"""ray trainable class for fine-tuning hyper-parameters"""
import os 
import sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd()))) 
from config import Config
from utils.plotting import *
from models.nam import NAM
from models.utils import *

import torch 
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace
from typing import Mapping
from typing import Sequence

from .losses import penalized_loss
from .metrics import accuracy
from .metrics import mae
from .saver import Checkpointer
from .epoch import *

import ray
from ray import tune

class trainableClass(tune.Trainable): 
    def setup(self, 
              config: SimpleNamespace,
              dataset: torch.utils.data.Dataset, 
              model: nn.Module
             ):
        """
        set up training.
        Args: 
        config: a dict of hyperparameters for fine-tuning
        
        static_config: fixed hyperparameters for model setting and training
        dataset: the whole dataset on which model is trained, validated, and tested
        """
        config = Config(**config)
        self.model = model
        # self.model = NAM(config=config, name="NAM_FINE_TUNE", in_features=len(dataset[0][0]), num_units=get_num_units(config, dataset.X))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.decay_rate) 
        self.dataset = dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = self.dataset.get_dataloaders()
        
        self.criterion = lambda preds, fnn_out, model, targets: penalized_loss(config, preds, fnn_out, self.model, targets)
        self.metrics = lambda preds, targets: mae(preds, targets) if config.regression else accuracy(preds, targets)
        self.metrics_name = "MAE" if config.regression else "Accuracy"
        
        # self.ckpts_dir = os.path.join(config.logdir, "ckpts")
        
        
    def step(self):
        """
        A single trial.Each trial is placed into a Ray actor process and runs in parallel.
        """
        loss_train, metrics_train = train_epoch(self.criterion, self.metrics, self.optimizer, self.model, self.dataloader_train)
        loss_val, metrics_val = evaluate_epoch(self.criterion, self.metrics, self.model, self.dataloader_val)
        
        result = {"loss": loss_val, self.metrics_name : metrics_val}
        return result
    
    
    # def save_checkpoint(self, ckpts_dir):
        """
        Save model to file "ckpts_dir/model.pth"
        """
        # ckpts_dir = os.path.join(self.static_config.logdir, "ckpts")
      #  ckpts_path = os.path.join(self.ckpts_dir, "model.pth")
       # torch.save(self.model.state_dict(), ckpts_path)
        #return self.ckpts_dir
    
    
    #def load_checkpoint(self, ckpts_path):
        """
        Load model from file "ckpts_path"
        """
     #   ckpts_path = os.path.join(self.ckpts_dir, "model.pth")
      #  self.model.load_state_dict(torch.load(ckpts_path))