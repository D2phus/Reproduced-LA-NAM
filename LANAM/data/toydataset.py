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

class ConcurvityDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=200, batch_size=64, sigma_1=0.05, sigma_2=0.5, use_test=False):
        """Simulated dataset built from a known concurvity structure. 
        Refer to Section 6 of the paper: Feature selection algorithms in generalized additive models under concurvity."""
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.use_test = False

        self.in_features = 7
        self.generate_concuvity_data()
        self.get_loaders()
        
        
    def generate_concuvity_data(self):
        """Generate features with concurvity, where X1, X2, X3 are uniformly sampled, while feature X4, X5, X6, X7 are generated based on X1-X3. y is computed only based on X1, X5, X6."""
        def f4(X, sigma): 
            return torch.pow(X[1], 3) + torch.pow(X[2], 2) + torch.randn(X[1].shape)*sigma
        
        def f5(X, sigma):
            return torch.pow(X[2], 2) + torch.randn(X[2].shape)*sigma
        
        def f6(X, sigma):
            return torch.pow(X[1], 2) + torch.pow(X[3], 3) + torch.randn(X[1].shape)*sigma
        
        def f7(X, sigma):
            return X[0]*X[1] + torch.randn(X[0].shape)*sigma
        
        X = [torch.FloatTensor(self.num_samples, 1).uniform_(0, 1) for _ in range(3)]
        gen_funcs = [f4, f5, f6, f7]
        for f in gen_funcs:
            X.append(f(X, self.sigma_1))
        self.X = torch.cat(X, dim=1) 
        self.fnn = torch.zeros_like(self.X)
        self.fnn[:, 0], self.fnn[:, 4], self.fnn[:, 5] = 2*torch.pow(X[0], 2).squeeze(), torch.pow(X[4], 3).squeeze(), 2*torch.sin(X[5]).squeeze()
        self.y = self.fnn.sum(dim=1).reshape(-1, 1) + torch.randn(X[0].shape)*self.sigma_2
        
        
    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return self.X[idx], self.y[idx]

    
    def plot(self):
        """
        plot scatterplot matrix of the simulated variables.
        """
        cols = 8
        rows = 8
        figsize = (1.5*cols ,1.5*rows)
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axs = axs.ravel() # 
        fig.tight_layout()
        text_kwargs = dict(ha='center', va='center', fontsize=28)
        for index in range(self.in_features): 
            for idx in range(self.in_features):
                #print(index, idx)
                ax = axs[index*cols+idx]
                if index == idx:
                    ax.text(0.5, 0.5, f'X{index}', **text_kwargs)
                    continue
                ax.plot(self.X[:, idx], self.X[:, index], '.', color='royalblue')
            axs[cols*(index+1)-1].plot(self.y, self.X[:, index], '.', color='royalblue')
                #axs[index].set_title(f"X{index}")
        for idx in range(self.in_features):
            axs[cols*(rows-1)+idx].plot(self.X[:, idx], self.y, '.', color='royalblue')
        axs[cols*rows-1].text(0.5, 0.5, 'y', **text_kwargs)
        
        
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
     
        
class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 gen_funcs: Sequence,
                 gen_func_names: Sequence, 
                 x_start=0, 
                 x_end=1,
                 dist_type='uniform',
                 num_samples=200, 
                 batch_size=64,
                 sigma=1,
                 use_test=False,
                 )-> None:
        """
        dataset generated with additive model consisted of given synthetic functions. 
        
        Args:
        -----------
        gen_funcs: list 
            synthetic functions for input features
        gen_func_names: list 
            synthetic function names
        x_start, x_end: float 
        dist_type: str
            The data samples are generated within the domain [x_start, x_end]^{in_features} with `dist_type` distribution:
            `uniform`: U(x_start, x_end)
            `normal`: N(x_start + (x_end-x_start)/2, [(x_end-x_start)/5]^2) truncated by [x_start, x_end]
            `exp`: Exp(0.5)
        num_samples: int
            sample sizes.
        Attrs:
        -----------
        X of shape (batch_size, in_features)
        y of shape (batch_size)
        fnn of shape (batch_size, in_features)
        loader: data loader for X, y
        loader_fnn: list, data loader for each input dimensional
        """
        if dist_type not in ['uniform', 'normal', 'exp']:
            raise ValueError('`dist_type` should be `uniform`, `normal`, or `exp`.')
            
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.sigma = sigma
        self.gen_funcs = gen_funcs
        self.gen_func_names = gen_func_names
        self.dist_type = dist_type
        self.use_test = use_test
        
        self.in_features = len(gen_func_names)
        if self.dist_type == 'uniform':
            # uniformly sampled X
            self.X = torch.FloatTensor(num_samples, self.in_features).uniform_(x_start, x_end)
        elif self.dist_type == 'normal':
            # normally sampled X truncated by [x_start, x_end]
            pass
        else:
            # X sampled from exp distribution
            pass
            
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
     