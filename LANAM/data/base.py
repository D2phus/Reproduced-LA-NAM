"""Dataset class for synthetic data"""
import math 
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from typing import Tuple, Sequence, Union, List, Dict
from LANAM.data.utils import *

class LANAMDataset(torch.utils.data.Dataset): 
    """The dataset base. 
    Reference: https://github.com/AmrMKayid/nam/blob/main/nam/data/utils.py
    """
    def __init__(self, 
                 config,
                 data: pd.DataFrame,
                 features_columns: List,
                 targets_column: List,
                 min_max: bool=True):
        """
        Args: 
        ------
        config: project configuration
        data: raw data
        feature_columns: the column labels for features
        targets_column: the column labels for target
        
        Attrs:
        -----
        features: Tensor, (num_samples, in_features)
            processed features, numeric and normalized.
        targets: Tensor, (num_samples, output_size)
            processed targets, categorical for classification, numeric for regression
        in_features: int
            number of input features
        feature_names: List[str]
            the feature labels
        train_dl, val_dl, test_dl: DataLoader
            dataloaders for training, validation, and test set on the whole data     
        """
        super().__init__()
        self.config = config
        self.raw_data = data
        self.X, self.y = self.raw_data[features_columns].copy(), self.raw_data[targets_column].copy()
        
        # data transformation: categorical column -> one-hot; normalize to (-1, 1)
        self.min_max = min_max # whether to normalize
        self.features, self.feature_names = transform_data(self.X, min_max=min_max)
        self.in_features = len(self.feature_names)
        
        # for classification task, transform  categorical target into one-hot
        self.X, self.y = self.X.to_numpy(), self.y.to_numpy()
        if (config.likelihood == 'classification') and (not isinstance(self.y, np.ndarray)): 
            targets = pd.get_dummies(self.y).values
            targets = np.array(np.argmax(targets, axis=-1)) 
        else:
            targets = self.y

        self.features = torch.from_numpy(self.features).float().to(config.device)
        self.targets = torch.from_numpy(targets).view(-1, 1).float().to(config.device)

        self.setup_dataloaders()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
        return self.features[idx], self.targets[idx]


    def setup_dataloaders(self, val_split: float = 0.1, test_split: float = 0.2) -> Tuple[DataLoader, ...]:
        test_size = int(test_split * len(self))
        val_size = int(val_split * (len(self) - test_size))
        train_size = len(self) - val_size - test_size

        train_subset, val_subset, test_subset = random_split(self, [train_size, val_size, test_size])
    
        self.train_dl = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True) if len(train_subset) > 0 else None
        self.val_dl = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False) if len(val_subset) > 0 else None
        self.test_dl = DataLoader(test_subset, batch_size=self.config.batch_size, shuffle=False) if len(test_subset) > 0 else None
        
    def train_dataloaders(self) -> Tuple[DataLoader, ...]:
        return self.train_dl, self.val_dl

    def test_dataloaders(self) -> DataLoader:
        return self.test_dl
    
    def get_test_samples(self): 
        """
        get all the samples from the testing subset.
        
        Returns:
        -----
        features: Tensor, (num_testing_samples, in_features)
        targets: Tensor, (num_testing_samples, 1)
        feature_targets: Tensor, (num_testing_samples, in_features)
        """
        return *self.test_dl.dataset[:], None, self.feature_names
    
    def plot_scatterplot_matrix(self):
        """Plot scatterplot matrix on test set."""
        subset = self.test_dl.dataset
        indices = subset.indices
        features = self.features[indices]
        y = self.y[indices]
            
        cols = self.in_features + 1
        rows = cols
        figsize = (1.5*cols ,1.5*rows)
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axs = axs.ravel() # 
        #text_kwargs = dict(ha='center', va='center', fontsize=28)
        for index in range(self.in_features): 
            for idx in range(self.in_features):
                #print(index, idx)
                ax = axs[index*cols+idx]
                if index == idx:
                    ax.set_title(f'{self.feature_names[idx]}')
                    continue
                ax.plot(features[:, idx], features[:, index], '.', color='royalblue', alpha=0.4)
            axs[cols*(index+1)-1].plot(y, features[:, index], '.', color='royalblue', alpha=0.4)
                #axs[index].set_title(f"X{index}")
                
        for idx in range(self.in_features):
            axs[cols*(rows-1)+idx].plot(features[:, idx], y, '.', color='royalblue', alpha=0.4)
        axs[cols*rows-1].set_title('y')
        
        fig.tight_layout()
        
    
class LANAMSyntheticDataset(LANAMDataset):
    """The dataset for synthetic data with a known additive structure.
    Args: 
    -----
    feature_targets: Tensor, (num_samples, in_features)
        the targets for each dimension of inputs (features)
    sigma: float
        the observation noise
    """
    def __init__(self, config, data, features_columns, targets_column, feature_targets, sigma, min_max: bool=True):
        self.feature_targets = feature_targets
        self.sigma = sigma
        
        super().__init__(config=config,
                         data=data,
                         features_columns=features_columns,
                         targets_column=targets_column,
                         min_max=min_max)
        
            
    def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
        return self.features[idx], self.targets[idx], self.feature_targets[idx]

    def plot_dataset(self, subset=None):
        """plot shape functions. 
        """
        features, feature_targets = self.features, self.feature_targets
        num_samples = features.shape[1]
        
        # figure setting 
        if self.in_features < 5: 
            cols, rows = self.in_features, 1
        else: 
            cols = 4
            rows = math.ceil(self.in_features / cols)
        
        figsize = (3*cols ,2*rows)
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axs = axs.ravel()  
            
        for index in range(self.in_features): 
            customize_ylim = (torch.min(feature_targets[:, index]).numpy()-1, torch.max(feature_targets[:, index]).numpy()+1)
            hist_scale = customize_ylim[1] - customize_ylim[0]
            axs[index].set_ylim(customize_ylim)
            
            axs[index].hist(features[:, index].numpy(), bins=10, bottom=customize_ylim[0], density=True, weights= hist_scale * np.ones_like(features[:, index].numpy()), alpha=0.5, color='lightblue')
            
            axs[index].plot(features[:, index], feature_targets[:, index], '.', color='royalblue')  
            axs[index].set_xlabel(self.feature_names[index])
            axs[index].set_ylabel(f'f{index+1}')
            
        fig.tight_layout()
     
    
    def get_test_samples(self): 
        """
        get all the samples from the testing subset.
        
        Returns:
        -----
        features: Tensor, (num_testing_samples, in_features)
        targets: Tensor, (num_testing_samples, 1)
        feature_targets: Tensor, (num_testing_samples, in_features)
        """
                                    
        return *self.test_dl.dataset[:], self.feature_names

       

    def get_train_samples(self): 
        
        return *self.train_dl.dataset[:], self.feature_names
