"""Dataset class for synthetic data"""
import math 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from typing import Tuple
from typing import Sequence

import matplotlib.pyplot as plt 
from .generator import *

class GAMDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, fnn, batch_size=64, use_test=False)-> None:
        """
        dataset for generalized additive models.
        
        Args:
        -----------
        X of shape (batch_size, in_features)
        y of shape (batch_size)
        fnn of shape (batch_size, in_features): output for each feature dimension
        batch_size, int
        use_test, bool: the dataloader won't be shuffled if `use_test` is True. 
        
        """
        super(GAMDataset, self).__init__()
        self.X = X
        self.y = y
        self.fnn = fnn
        self.batch_size = batch_size
        self.use_test = use_test
        self.in_features = X.shape[1]
        
        self.get_loaders()
       
    
    @property
    def tensors(self):
        """Returns data."""
        return self.X, self.y, self.fnn
    
    
    def __len__(self):
        return len(self.X)

    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return self.X[idx], self.y[idx]

    
    def plot(self, additive=True):
        cols = 4
        rows = math.ceil(self.in_features / cols)
        figsize = (2*cols ,2*rows)
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axs = axs.ravel() # 
        fig.tight_layout()
        for index in range(self.in_features): 
            axs[index].plot(self.X[:, index], self.fnn[:, index], '.', color='royalblue')
            axs[index].set_title(f"X{index}")
        
        if additive:
            fig, axs = plt.subplots()
            axs.plot(self.X[:, 0], self.y, '.', color='royalblue')
    
    
    def get_loaders(self): 
        """
        Returns:
        ---------------
        loader 
        loader_fnn, list
        """
        dataset = TensorDataset(self.X, self.y) # X, y: of shape (batch_size, 1)
        dataset_fnn = [TensorDataset(self.X[:, index].reshape(-1, 1), self.fnn[:, index].reshape(-1, 1)) for index in range(self.in_features)]
        
        shuffle = False if self.use_test else True
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        self.loader_fnn = [DataLoader(dataset_fnn[index], batch_size=self.batch_size, shuffle=shuffle) for index in range(self.in_features)]

     
