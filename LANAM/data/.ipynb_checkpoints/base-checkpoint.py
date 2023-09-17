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

#from nam.data import transform_data
from nam.data import CSVDataset

from LANAM.config import Config
from LANAM.data.utils import *

class LANAMDataset(CSVDataset): 
    """The dataset base. Reference: https://github.com/AmrMKayid/nam/blob/main/nam/data/utils.py
    """
    def __init__(self, config: Config,
                         data: pd.DataFrame,
                         features_columns: List,
                         targets_column: List,
                         weights_column: List=None):
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
        features_names: List[str]
            the feature labels
        single_features: Dict, {str: List}
            sorted single features, str: column label, List: sorted value of this column.
        ufo: Dict, {str: List}
            sorted single features with only unique values.
        train_dl, val_dl, test_dl: DataLoader
            dataloaders for training, validation, and test set on the whole data
        train_dl_fnn, val_dl_fnn, test_dl_fnn: DataLoader
            dataloaders for training, validation, and test set on each dimensional (feature)
            note that the dimensional target means nothing.
        
        """
        super().__init__(config, data, features_columns, targets_column, weights_column)
        self.raw_data = data
        
        self.col_min_max = self.get_col_min_max()

        self.features, self.features_names = transform_data(self.raw_X) # convert categorical data into numeric
        self.compute_features() # compute sorted features
        self.in_features = len(self.features_names)

        # for classification task, concert numeric target into categorical
        if (config.likelihood == 'classification') and (not isinstance(self.raw_y, np.ndarray)): 
            targets = pd.get_dummies(self.raw_y).values
            targets = np.array(np.argmax(targets, axis=-1)) 
        else:
            targets = self.y

        self.features = torch.from_numpy(self.features).float().to(config.device)
        self.targets = torch.from_numpy(targets).view(-1, 1).float().to(config.device)
        self.wgts = torch.from_numpy(self.wgts).to(config.device)

        self.setup_dataloaders()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
        return self.features[idx], self.targets[idx]  #, self.wgts[idx]

    def get_col_min_max(self):
        col_min_max = {}
        for col in self.raw_X:
            unique_vals = self.raw_X[col].unique()
            col_min_max[col] = (np.min(unique_vals), np.max(unique_vals))

        return col_min_max

    def compute_features(self):
        """Compute sorted single features and sorted unique single features.
        """
        single_features = np.split(np.array(self.features), self.features.shape[1], axis=1)
        self.unique_features = [np.unique(f, axis=0) for f in single_features]

        self.single_features = {col: sorted(self.raw_X[col].to_numpy()) for col in self.raw_X}
        self.ufo = {col: sorted(self.raw_X[col].unique()) for col in self.raw_X}

    def setup_dataloaders(self, val_split: float = 0.1, test_split: float = 0.2) -> Tuple[DataLoader, ...]:
        def setup_feature_dataloaders(subset: torch.utils.data.Subset):
            """set up dataloader for each feature on given subset"""
            dl_fnn = list()
            for idx in range(self.in_features):
                features, targets = subset[:]
                feature = features[:, idx].reshape(-1, 1) # (num_samples, 1)
                targets = targets.reshape(-1, 1)
                dataset = TensorDataset(feature, targets)
                loader = DataLoader(dataset, batch_size=self.config.batch_size)
                dl_fnn.append(loader)
                
            return dl_fnn
                
        test_size = int(test_split * len(self))
        val_size = int(val_split * (len(self) - test_size))
        train_size = len(self) - val_size - test_size

        train_subset, val_subset, test_subset = random_split(self, [train_size, val_size, test_size])
    
        self.train_dl_fnn = setup_feature_dataloaders(train_subset)
        self.val_dl_fnn = setup_feature_dataloaders(val_subset)
        self.test_dl_fnn = setup_feature_dataloaders(test_subset)

        self.train_dl = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True) if len(train_subset) > 0 else None
        self.val_dl = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False) if len(val_subset) > 0 else None
        self.test_dl = DataLoader(test_subset, batch_size=self.config.batch_size, shuffle=False) if len(test_subset) > 0 else None
        
    def train_dataloaders(self) -> Tuple[DataLoader, ...]:
        return self.train_dl, self.train_dl_fnn, self.val_dl, self.val_dl_fnn

    def test_dataloaders(self) -> DataLoader:
        return self.test_dl, self.test_dl_fnn
    
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
                    ax.set_title(f'{self.features_names[idx]}')
                    continue
                ax.plot(features[:, idx], features[:, index], '.', color='royalblue')
            axs[cols*(index+1)-1].plot(y, features[:, index], '.', color='royalblue')
                #axs[index].set_title(f"X{index}")
                
        for idx in range(self.in_features):
            axs[cols*(rows-1)+idx].plot(features[:, idx], y, '.', color='royalblue')
        axs[cols*rows-1].set_title('y')
        
        fig.tight_layout()
        
    
class LANAMSyntheticDataset(LANAMDataset):
    """The dataset for synthetic data, whose additive structure is known.
    Args: 
    -----
    feature_targets: Tensor, (num_samples, in_features)
        the targets for each dimension of inputs (features)
    sigma: float
        the observation noise
    
    Attrs:
    -----
    sigma: float
        the observation noise
    """
    def __init__(self, config, data, features_columns, targets_column, feature_targets, sigma, weights_column=None):
        self.feature_targets = feature_targets
        self.sigma = sigma
        
        super().__init__(config=config,
                         data=data,
                         features_columns=features_columns,
                         targets_column=targets_column,
                         weights_column=weights_column)
        
    def plot_dataset(self, subset=None):
        """
        plot features and corresponding ground truth target functions.
        Args: 
        ----
        subset: str in ['training', 'validation', 'test']. default=None
            the dataset to plot. if None, the whole dataset will be used.
        """
        if subset is not None and subset not in ['training', 'validation', 'test']: 
            raise ValueError('The subset should be `training`, `validation`, or `test`.')
        if subset == 'training':
            if self.train_dl is not None:
                subset = self.train_dl.dataset
            else:
                raise ValueError('No training dataset.')
        elif subset == 'validation':
            if self.val_dl is not None:
                subset = self.val_dl.dataset
            else:
                raise ValueError('No validation dataset.')
        else: 
            if self.test_dl is not None:
                subset = self.test_dl.dataset
            else:
                raise ValueError('No test dataset.')
        
        num_bin = 10
        cols = 4
        rows = math.ceil(self.in_features / cols)
        figsize = (2*cols ,2*rows)
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axs = axs.ravel()  
        if subset is None:
            features = self.features
            feature_targets = self.feature_targets 
        else:
            indices = subset.indices
            features = self.features[indices]
            feature_targets = self.feature_targets[indices]
        
        num_samples = features.shape[1]    
        for index in range(self.in_features): 
            customize_ylim = (torch.min(feature_targets[:, index]).numpy()-1, torch.max(feature_targets[:, index]).numpy()+1)
            hist_scale = customize_ylim[1] - customize_ylim[0]
            axs[index].set_ylim(customize_ylim)
            
            axs[index].hist(features[:, index].numpy(), bins=20, bottom=customize_ylim[0], density=True, weights= hist_scale * np.ones_like(features[:, index].numpy()), alpha=0.5, color='lightblue')
            
            axs[index].plot(features[:, index], feature_targets[:, index], '.', color='royalblue')  
            axs[index].set_xlabel(self.features_names[index])
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
        test_subset = self.test_dl.dataset
        indices = test_subset.indices
        features, targets = test_subset[:]
        feature_targets = self.feature_targets[indices, :]
        
        return features, targets, feature_targets
               
        
    def setup_dataloaders(self, val_split: float = 0.1, test_split: float = 0.2) -> Tuple[DataLoader, ...]:
        def setup_feature_dataloaders(subset, feature_targets):
            dl_fnn = list()
            for idx in range(self.in_features):
                features, _ = subset[:]
                feature = features[:, idx].reshape(-1, 1) # (num_samples, 1)
                targets = feature_targets[:, idx].reshape(-1, 1)
                dataset = TensorDataset(feature, targets)
                loader = DataLoader(dataset, batch_size=self.config.batch_size)
                dl_fnn.append(loader)
                
            return dl_fnn
        
        test_size = int(test_split * len(self))
        val_size = int(val_split * (len(self) - test_size))
        train_size = len(self) - val_size - test_size

        train_subset, val_subset, test_subset = random_split(self, [train_size, val_size, test_size])
    
        self.train_dl_fnn = setup_feature_dataloaders(train_subset, self.feature_targets[train_subset.indices])
        self.val_dl_fnn = setup_feature_dataloaders(val_subset, self.feature_targets[val_subset.indices])
        self.test_dl_fnn = setup_feature_dataloaders(test_subset, self.feature_targets[test_subset.indices])

        self.train_dl = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True) if len(train_subset) > 0 else None
        self.val_dl = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False) if len(val_subset) > 0 else None
        self.test_dl = DataLoader(test_subset, batch_size=self.config.batch_size, shuffle=False) if len(test_subset) > 0 else None
        