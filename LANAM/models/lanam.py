"""Neural additive model"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split


import numpy as np

from math import sqrt, pi

from typing import Tuple
from typing import Sequence
from typing import List

from .featurenn import FeatureNN
from .utils import *

from LANAM.extensions.backpack.custom_modules import BackPackGGNExt

from laplace import Laplace


class LaNAM(nn.Module):
    def __init__(
        self, 
        config, 
        name: str, 
        in_features: int,
        
        subset_of_weights='all', 
        hessian_structure='full',
        backend=BackPackGGNExt, 
        prior_mean=0.0, 
        prior_precision=1.0, 
        sigma_noise=1.0, 
        temperature=1.0, 
        ) -> None: # type-check
            """Laplace-approximated additive models.
            Attrs:
            ------
            k_interactions: List[Tuple]
                top-k feature interaction.
            """
            super().__init__()
            self.config = config
            self.in_features = in_features
            self.k_interactions = None
        
            self.subset_of_weights = subset_of_weights 
            self.hessian_structure = hessian_structure
            self.backend = backend
            
            self.temperature = temperature
            # individual prior for each feature network
            self.prior_mean = prior_mean
            self.prior_precision = prior_precision
            self.sigma_noise = sigma_noise
            
            self.n_outputs = 1
             
            
            self._feature_nns = nn.ModuleList([
                FeatureNN(self.config, 
                          name=f"FeatureNN_{index}", 
                          in_features=1, 
                          feature_index=index, 
                          subset_of_weights=self.subset_of_weights, 
                          hessian_structure=self.hessian_structure,
                          backend=self.backend,
                          prior_mean=self.prior_mean[index], 
                          prior_precision=self.prior_precision[index], 
                          sigma_noise=self.sigma_noise[index], 
                         temperature=self.temperature)
                for index in range(self.in_features)
            ])
            
            self.likelihood = self.config.likelihood
            self.factor = 0.5 if self.likelihood=='regression' else 1# constant factor from loss to log likelihood 
            
            
    def extra_repr(self):
        self.feature_nns
        
        
    @property
    def num_nets(self):
        return self.in_features if self.k_interactions is None else self.in_features + len(self.k_interactions)
    
    @property
    def sigma_noise(self):
        return self._sigma_noise
    
    @sigma_noise.setter
    def sigma_noise(self, sigma_noise): 
        """The setter for sigma_noise.
        We have individual obervation noise for each feature neural network. 
        """
        num_feature_net = self.in_features
        if np.isscalar(sigma_noise) and np.isreal(sigma_noise):
            self._sigma_noise = torch.ones(num_feature_net, dtype=torch.float32) * sigma_noise
        elif torch.is_tensor(sigma_noise):
            if sigma_noise.ndim > 1: 
                raise ValueError('The dimension of sigma noise has to be in [0, 1].')
            if len(sigma_noise) == 1: 
                self._sigma_noise = torch.ones(num_feature_net, dtype=torch.float32) * sigma_noise
            elif len(sigma_noise) == num_feature_net: 
                self._sigma_noise = sigma_noise
            else:
                raise ValueError('Invalid length of sigma noise.')
        else: 
            raise ValueError("Invalid data type for sigma noise.")
            
    @property
    def prior_mean(self): 
        return self._prior_mean
    
    @prior_mean.setter
    def prior_mean(self, prior_mean): 
        """The setter for prior_mean.
        We have individual prior for each feature neural network. 
        Args:
        prior_mean: real scalar, torch.Tensor of shape (n_features)
        """
        num_feature_net = self.in_features
        
        if np.isscalar(prior_mean) and np.isreal(prior_mean):
            self._prior_mean = torch.ones(num_feature_net, dtype=torch.float32) * prior_mean
        elif torch.is_tensor(prior_mean):
            if prior_mean.ndim > 1: 
                raise ValueError('The dimension of prior mean has to be in [0, 1].')
            if len(prior_mean) == 1: 
                self._prior_mean = torch.ones(num_feature_net, dtype=torch.float32) * prior_mean
            elif len(prior_mean) == num_feature_net: 
                self._prior_mean = prior_mean
            else:
                raise ValueError('Invalid length of prior mean.')
        else: 
            raise ValueError("Invalid data type for prior mean.")
            
    @property 
    def prior_precision(self): 
        return self._prior_precision
    
    @prior_precision.setter
    def prior_precision(self, prior_precision):
        """The setter for prior precision.
        We have individual prior for each feature neural network."""
        num_feature_net = self.in_features
        
        if np.isscalar(prior_precision) and np.isreal(prior_precision):
            self._prior_precision = torch.ones(num_feature_net, dtype=torch.float32) * prior_precision
        elif torch.is_tensor(prior_precision):
            if prior_precision.ndim == 0:
                self._prior_precision = prior_precision.reshape(-1)
            elif prior_precision.ndim == 1:
                if len(prior_precision) == 1:
                    self._prior_precision = torch.ones(num_feature_net, dtype=torch.float32) * prior_precision
                elif len(prior_precision) == num_feature_net: 
                    self._prior_precision = prior_precision
                else:
                    raise ValueError('Length of prior precision does not align with architecture.')
            else:
                raise ValueError('Prior precision needs to be at most one-dimensional tensor.')
        else:
            raise ValueError('Prior precision either scalar or torch.Tensor up to 1-dim.')
        
    @property
    def feature_nns(self): 
        if self._feature_nns is None: # initialize
            self._feature_nns = nn.ModuleList([
                FeatureNN(self.config, 
                          name=f"FeatureNN_{index}", 
                          in_features=1, 
                          feature_index=index, 
                          subset_of_weights=self.subset_of_weights, 
                          hessian_structure=self.hessian_structure,
                          backend=self.backend,
                          prior_mean=self.prior_mean[index], 
                          prior_precision=self.prior_precision[index], 
                          sigma_noise=self.sigma_noise[index], 
                         temperature=self.temperature)
                for index in range(self.in_features)
            ])
        else: # update prior
            for idx, fnn in enumerate(self._feature_nns): 
                fnn.prior_precision = self.prior_precision[idx]
                fnn.sigma_noise = self.sigma_noise[idx]
        
        return self._feature_nns
    
    @property 
    def joint_nns(self): 
        if self._joint_nns is None: 
            self._joint_nns = nn.ModuleList([
            FeatureNN(self.config, 
                  name=f'FeatureNN_{str(indice)}', 
                  in_features=len(indice),
                  feature_index=indice,
                  subset_of_weights=self.subset_of_weights, 
                  hessian_structure=self.hessian_structure, 
                  backend=self.backend, 
                  prior_mean=self.prior_mean[self.in_features+idx], 
                  prior_precision=self.prior_precision[self.in_features+idx], 
                  sigma_noise=self.sigma_noise[self.in_features+idx], 
                  temperature=self.temperature) 
            for idx, indice in enumerate(self.k_interactions)
            ])
            
        else:
            for idx, inn in enumerate(self._joint_nns): 
                inn.prior_precision = self.prior_precision[self.in_features+idx]
                inn.sigma_noise = self.sigma_noise[self.in_features+idx]
        
        return self._joint_nns 
    
    @property 
    def k_interactions(self):
        return self._k_interactions
    
    @k_interactions.setter
    def k_interactions(self, k_interactions):
        """The setter for k_interactions."""
        
        if k_interactions is None or isinstance(k_interactions, list):
            self._k_interactions = k_interactions
        else: 
            raise ValueError('Invalid data type for `k_interactions`.')
            
    def clear_up_joint_feature_nets(self):
        self.k_interactions = None
        self._joint_nns = None
        self.sigma_noise = self.sigma_noise[:self.in_features]
        self.prior_mean = self.prior_mean[:self.in_features]
        self.prior_precision = self.prior_precision[:self.in_features]
        
    def extend_joint_feature_nets(self, k_interactions, override=True):
        if override:
            # clear up former feature nets
            self.clear_up_joint_feature_nets()
            
        if self.k_interactions is None: 
            self.k_interactions = k_interactions
        else:
            self.k_interactions.extend(k_interactions)
        # TODO
        pass
        
    @property
    def _H_factor(self): 
        return 1 / self.sigma_noise.square().sum() / self.temperature
    
    @property
    def additive_sigma_noise(self): 
        """additive normal distribution, \sigma^2 = \sum_d \sigma_d ^2"""
        return self.sigma_noise.square().sum().sqrt()
    
    @property
    def log_likelihood(self): 
        """compute log likelihood after method `fit` has been called. """
        factor = - self._H_factor
        if self.likelihood == 'regression':
            c = self.n_data * self.n_outputs * torch.log(self.additive_sigma_noise * sqrt(2 * pi))
            return factor * self.loss - c # additive loss
        else:
            # for classification Xent == log Cat
            return factor * self.loss
    
    def _features_output(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """
        Returns: 
        ---------
        list [torch.Tensor of shape (batch_size, 1)]: outputs of feature neural nets
        """
        return [self.feature_nns[index](inputs[:, index]) for index in range(self.in_features)] 
            
    def forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
        inputs of shape (batch_size, in_features): input samples, 
        
        Returns: 
        ----------
        out of shape (batch_size, 1): 
            output of additive models.
        fnn of shape (batch_size, in_features): 
            output of each feature net
        """
        fnn = self._features_output(inputs) # list [Tensor(batch_size,  1)]
        #fnn = [self.feature_nns[index](inputs[:, index]) for index in subset_of_features]
            
        out = torch.stack(fnn, dim=-1).sum(dim=-1) # sum along the features => of shape (batch_size)
        return out, fnn
    
    def fit(self, loss, train_loader, override=True): 
        if override: 
            self.loss = 0
            self.n_data = 0 
        
        N = len(train_loader.dataset)
        features, target = train_loader.dataset[:]
        self.n_data += N
        if np.isscalar(loss) and np.isreal(loss):
            loss = torch.tensor(loss)
        self.loss += self.factor*loss # NOTE: +=, accumulated.
        
        for idx, fnn in enumerate(self.feature_nns): 
            fnn.fit(DataLoader(TensorDataset(features[:, idx:idx+1], target), batch_size=self.config.batch_size), override)
        
        
   # def fit(self, loss, loader_fnn, override=True): 
   #     """fit Laplace approximation for each feature neural net.
   #     Args:
   #     --------
   #     loss: 
   #         the summed loss of additive models on the training data. 
   #     loader_fnn: list 
   #         data loaders for feature networks.  
   #     """
   #     if override:
   #         self.loss = 0
   #         self.n_data = 0
            
   #     if type(loader_fnn) is not list: 
   #         loader_fnn = [loader_fnn]
        
   #     if np.isscalar(loss) and np.isreal(loss):
   #         loss = torch.tensor(loss)
   #     self.loss = self.factor*loss 
        
   #     N = len(loader_fnn[0].dataset)
        
   #     for idx, fnn in enumerate(self.feature_nns): 
   #         fnn.fit(loader_fnn[idx], override=override) 
            
   #     self.n_data += N
            
    #@property 
    #def posterior_covariance(self):
    #    """block-diagonal posterior covariance. """
    #    factor = self.pos_cov_scale
    #    return factor@factor.T
    
    #@property
    #def pos_cov_scale(self):
    #    """the lower-triangular factor of posterior covariance."""
    #    return _precision_to_scale_tril(self.posterior_precision)
    
    #@property
    #def posterior_precision(self): 
    #    """block-diagonal posterior precision. """
    #    pos_prec = [self.feature_nns[index].posterior_precision for index in range(self.in_features)] 
    #    return torch.block_diag(*pos_prec)
    
    def predict(self, x, pred_type='glm', link_approx='probit'): 
        """can only be called after calling `fit` method.
        predictive posterior which can be decomposed across individual feature networks.
        Note that the predictive posterior of features networks may shift to accommodate for a global intercept and should be re-centered around zero before visualization.
        
        Args:
        ----------------
        x of shape (batch_size, in_features)
        
        Returns: 
        -----------------
        f_mu of shape (batch_size)
        f_var of shape (batch_size)
        f_mu_fnn, f_var_fnn of shape (batch_size, in_features)
        """
        f_mu_fnn, f_var_fnn = list(), list()
        
        for index in range(self.in_features): 
            x_fnn = x[:, index].reshape(-1, 1) # of shape (batch_size, 1)
            f_mu_index, f_var_index = self.feature_nns[index].la(x_fnn, pred_type=pred_type, link_approx=link_approx) 
            f_mu_fnn.append(f_mu_index) 
            f_var_fnn.append(f_var_index)
            
        f_mu_fnn = torch.cat(f_mu_fnn, dim=1)
        f_var_fnn = torch.cat(f_var_fnn, dim=1).flatten(1)
        return torch.sum(f_mu_fnn, dim=1), torch.sum(f_var_fnn, dim=1), f_mu_fnn, f_var_fnn
        
    def log_marginal_likelihood(self, prior_precision=None, sigma_noise=None): 
        """feature-wise log marignal likelihood approximation.
        """
        log_marglik = 0.0
        if prior_precision is not None:
            self.prior_precision = prior_precision
        if sigma_noise is not None:
            if self.likelihood != 'regression':
                raise ValueError('Can only change sigma_noise for regression.')
            self.sigma_noise = sigma_noise
        
        log_marglik += torch.stack([0.5*(fnn.log_det_ratio+fnn.scatter) for fnn in self.feature_nns]).sum()
        
        return self.log_likelihood - log_marglik
    
    def predictive_samples(self, x, n_samples=5):
        """Sample from the posterior predictive of NAM with linearized Laplace on input data x. 

        Args:
        ----------
        x of shape (batch_size, in_features)
        n_samples: number of samples
        
        Returns:
        -------
        samples of shape (n_samples, batch_size, out_features=1)
        """
        f_mu, f_var, _, _ = self.predict(x)
        f_mu = f_mu.unsqueeze(-1) if f_mu.ndim == 1 else f_mu
        f_samples = normal_samples(f_mu, f_var, n_samples)
        if self.likelihood == 'regression':
            return f_samples
        return torch.softmax(f_samples, dim=-1)