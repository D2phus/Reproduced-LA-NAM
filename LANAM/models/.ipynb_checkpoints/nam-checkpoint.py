"""Neural additive model"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Tuple
from typing import Sequence
from typing import List

from .featurenn import FeatureNN

class NAM(nn.Module):
    def __init__(
        self, 
        config, 
        name: str, 
        in_features: int,
        
        subset_of_weights='all', 
        hessian_structure='full',
        prior_mean=0.0, 
        prior_precision=1.0, 
        sigma_noise=1.0
        ) -> None: # type-check
            """
            The neural additive model learns a linear combination of nerual networks each of which attends to a single input feature. The outputs of subnets are added up, with a scalar bias, and passed through a link function for prediction. 
            Args:
            in_features: size of each input sample 
            """
            super(NAM, self).__init__()
            self.config = config
            self.in_features = in_features
            self.subset_of_weights = subset_of_weights 
            self.hessian_structure = hessian_structure
            # we have individual prior for each feature neural networks
            self.prior_mean = prior_mean
            self.prior_precision = prior_precision
            self.sigma_noise = sigma_noise
            # note 
            self._feature_nns = None
            #self.feature_nns = self.init_feature_nns()
            
    def extra_repr(self) -> str:
        return f'feature_nns={self.feature_nns}'
    
    @property
    def sigma_noise(self):
        return self._sigma_noise
    
    @sigma_noise.setter
    def sigma_noise(self, sigma_noise): 
        """The setter for sigma_noise.
        We have individual obervation noise for each feature neural network. 
        """
        if np.isscalar(sigma_noise) and np.isreal(sigma_noise):
            self._sigma_noise = torch.ones(self.in_features, dtype=torch.float32) * sigma_noise
        elif torch.is_tensor(sigma_noise):
            if sigma_noise.ndim > 1: 
                raise ValueError('The dimension of sigma noise has to be in [0, 1].')
            if len(sigma_noise) == 1: 
                self._sigma_noise = torch.ones(self.in_features, dtype=torch.float32) * sigma_noise
            elif len(sigma_noise) == self.in_features: 
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
        if np.isscalar(prior_mean) and np.isreal(prior_mean):
            self._prior_mean = torch.ones(self.in_features, dtype=torch.float32) * prior_mean
        elif torch.is_tensor(prior_mean):
            if prior_mean.ndim > 1: 
                raise ValueError('The dimension of prior mean has to be in [0, 1].')
            if len(prior_mean) == 1: 
                self._prior_mean = torch.ones(self.in_features, dtype=torch.float32) * prior_mean
            elif len(prior_mean) == self.in_features: 
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
        if np.isscalar(prior_precision) and np.isreal(prior_precision):
            self._prior_precision = torch.ones(self.in_features, dtype=torch.float32) * prior_precision
        elif torch.is_tensor(prior_precision):
            if prior_precision.ndim == 0:
                self._prior_precision = prior_precision.reshape(-1)
            elif prior_precision.ndim == 1:
                if len(prior_precision) == 1:
                    self._prior_precision = torch.ones(self.in_features, dtype=torch.float32) * prior_precision
                elif len(prior_precision) == self.in_features: 
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
                          name=f"FeatureNN_{feature_index}", 
                          in_features=1, 
                          feature_index=feature_index, 
                          subset_of_weights=self.subset_of_weights, 
                          hessian_structure=self.hessian_structure,
                          prior_mean=self.prior_mean[feature_index], 
                          prior_precision=self.prior_precision[feature_index], 
                          sigma_noise=self.sigma_noise[feature_index]
                          )
                for feature_index in range(self.in_features)
            ])
        else: # update
            for index in range(self.in_features): 
                self._feature_nns[index].update(prior_precision=self.prior_precision[index], 
                                                sigma_noise=self.sigma_noise[index])
        return self._feature_nns
    
    def _features_output(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """
        Return list [torch.Tensor of shape (batch_size, 1)]: the outputs of feature neural nets
        """
        return [self.feature_nns[feature_index](inputs[:, feature_index]) for feature_index in range(self.in_features)] # feature of shape (1, batch_size)
            
    def forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
        inputs of shape (batch_size, in_features): input samples, 
        
        Returns: 
        nam output of shape (batch_size, 1): add up the outputs of feature nets and bias
        fnn outputs of shape (batch_size, in_features): output of each feature net
        """
        fnn = self._features_output(inputs) # list [Tensor(batch_size,  1)]
        out = torch.stack(fnn, dim=-1).sum(dim=-1) # sum along the features => of shape (batch_size)
        return out, fnn
    
    def fit(self, loader_fnn, override=True): 
        """fit Laplace approximation for each feature neural net."""
        for feature_index in range(self.in_features):
            self.feature_nns[feature_index].fit(loader_fnn[feature_index]) 
    
    @property 
    def posterior_covariance(self):
        """block-diagonal posterior covariance. """
        pos_cov = [self.feature_nns[feature_index].posterior_covariance for feature_index in range(self.in_features)] 
        return torch.block_diag(*pos_cov)
    
    def predict(self, x, pred_type='glm', link_approx='probit'): 
        """predictive posterior which can be decomposed across individual feature networks.
        Note that the predictive posterior of features networks may shift to accommodate for a global intercept and should be re-centered around zero before visualization.
        Returns: 
        f_mu of shape (batch_size)
        f_var of shape (batch_size, 1)
        f_mu_fnn, f_var_fnn of shape (batch_size, in_features)
        """
        f_mu_fnn, f_var_fnn = list(), list()
        for feature_index in range(self.in_features): 
            x_fnn = x[:, feature_index].reshape(-1, 1) # of shape (batch_size, 1)
            f_mu_index, f_var_index = self.feature_nns[feature_index].la(x_fnn, pred_type=pred_type, link_approx=link_approx) 
            f_mu_fnn.append(f_mu_index) 
            f_var_fnn.append(f_var_index)
        f_mu_fnn = torch.cat(f_mu_fnn, dim=1)
        f_var_fnn = torch.cat(f_var_fnn, dim=1)
        return torch.sum(f_mu_fnn, dim=1), torch.sum(f_var_fnn, dim=1), f_mu_fnn, f_var_fnn
        
    def log_marginal_likelihood(self, prior_precision=None, sigma_noise=None): 
        """feature-wise log marignal likelihood approximation."""
        log_marglik = 0.0
        if prior_precision is not None:
            self.prior_precision = prior_precision
        if sigma_noise is not None:
            self.sigma_noise = sigma_noise
            
        for index in range(self.in_features):
            log_marglik += self.feature_nns[index].log_marginal_likelihood(self.prior_precision[index], self.sigma_noise[index])
        return log_marglik
    
    def predictive_samples(self, x, pred_type='glm', n_samples=5,  diagonal_output=False, generator=None):
        """
        Returns: 
        samples : torch.Tensor
            samples `(n_samples, batch_size, output_shape)`
        """ 
        samples = list()
        for index in range(self.in_features):
            fs = self.feature_nns[index].la.predictive_samples(x[: , index].reshape(-1, 1), pred_type=pred_type, n_samples=n_samples) # (n_samples, batch_size, output_shape)
            samples.append(fs)
        samples = torch.cat(samples, dim=-1)
        return samples
        
            
    