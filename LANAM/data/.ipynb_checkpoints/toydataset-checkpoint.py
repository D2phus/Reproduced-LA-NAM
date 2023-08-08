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

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 gen_funcs: Sequence,
                 gen_func_names: Sequence, 
                 x_start=0, 
                 x_end=1,
                 num_samples=200, 
                 batch_size=64,
                 sigma=1,
                 use_test=False,
                 )-> None:
        """
        dataset generated with additive model consisted of synthetic functions. 
        
        Args:
        -----------
        task_name: indicates the name for this task
        num_samples: the number of samples
        [x_start, x_end]: the x-value range for sampling. X is sampled uniformly.
        in_features: the size of input samples
        gen_funcs: list of synthetic functions for input features
        gen_func_names: list of synthetic function names
        
        Attrs:
        -----------
        X of shape (batch_size, in_features)
        y of shape (batch_size)
        fnn of shape (batch_size, in_features)
        loader: data loader for X, y
        loader_fnn: list, data loader for each input dimensional
        """
        super(ToyDataset, self).__init__()
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.sigma = sigma
        self.gen_funcs = gen_funcs
        self.gen_func_names = gen_func_names
        self.in_features = len(gen_func_names)
        self.use_test = use_test
        
        # uniformly sampled X
        self.X = torch.FloatTensor(num_samples, self.in_features).uniform_(x_start, x_end)
        if use_test:
            self.X, _ = torch.sort(self.X, dim=0)
        
        self.fnn = torch.stack([gen_funcs[index](x_i) for index, x_i in enumerate(torch.unbind(self.X, dim=1))], dim=1) # (batch_size, in_features) 
        
        self.y = self.fnn.sum(dim=1).reshape(-1, 1) # of shape (batch_size)
        if not use_test:
            noise = torch.randn_like(self.y) * self.sigma
            self.y = self.y + noise
        self.get_loaders()
        
        
       
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return self.X[idx], self.y[idx]

    def plot(self, additive=True):
        """
        plot each features on the whole dataset.
        """
        cols = 3
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
        loader, 
        loader_fnn, list:
        """
        dataset = TensorDataset(self.X, self.y)
        # X, y: of shape (batch_size, 1)
        dataset_fnn = [TensorDataset(self.X[:, index].reshape(-1, 1), self.fnn[:, index].reshape(-1, 1)) for index in range(self.in_features)]
        shuffle = False if self.use_test else True
        self.loader = DataLoader(dataset, 
                                 batch_size=self.batch_size, 
                                 shuffle=shuffle)
        self.loader_fnn = [DataLoader(dataset_fnn[index], batch_size=self.batch_size, shuffle=shuffle) 
                          for index in range(self.in_features)]
     