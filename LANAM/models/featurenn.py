"""DNN-based sub net for each input feature."""
import copy

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from .activation import ExU

from laplace import Laplace
from LANAM.extensions.backpack.custom_modules import BackPackGGNExt


class FeatureNN(nn.Module):
    def __init__(
        self, 
        config, 
        name: str, 
        in_features: int,
        feature_index: int, 
        
        subset_of_weights='all', 
        hessian_structure='full',
        backend=BackPackGGNExt, 
        prior_mean=0.0, 
        prior_precision=1.0, 
        sigma_noise=1.0, 
        temperature=1.0
        ) -> None: # type-check
            """
            Args:
            -----------
            in_features: int
                size of each input sample; default value = 1
            num_units: int 
                number of ExU/LinearReLU hidden units in the first hidden layer 
            feature_index: 
                indicate which feature is learn in this subnet
            prior_precision: float
                the prior precision of model parameters.
            sigma_noise: float
                the observation noise of feature networks. 
            
            """
            super(FeatureNN, self).__init__()
            self.config = config
            if self.config.activation_cls not in ['relu', 'exu', 'gelu', 'elu', 'leakyrelu']:
                raise ValueError('Activation unit should be `gelu`, `relu`, `exu`, `elu`, or `leakyrelu`.')
                
            self.name = name
            self.in_features = in_features
            self.feature_index = feature_index
            self.subset_of_weights = subset_of_weights
            self.hessian_structure = hessian_structure
            self.backend = backend
            self.prior_mean = prior_mean
            self.prior_precision = prior_precision
            self.sigma_noise = sigma_noise
            self.temperature = temperature
            
            self.hidden_sizes = [config.num_units for _ in range(config.num_layers)]
            # self.hidden_sizes = self.config.hidden_sizes
            self.activation = self.config.activation
            self.activation_cls = self.config.activation_cls
            self.likelihood = self.config.likelihood
            
            self.model = self.setup_model()
            self._la = None
            
            
    def setup_la(self): 
        self.la  = Laplace(model=self.model, likelihood=self.likelihood,
                      subset_of_weights=self.subset_of_weights, 
                      hessian_structure=self.hessian_structure, 
                      sigma_noise=self.sigma_noise, 
                      prior_precision=self.prior_precision, 
                      prior_mean=self.prior_mean, 
                      temperature=self.temperature, 
                      backend=self.backend)

    def setup_model(self):
        """set up DNN model. 
        1. With ExU activation unit: first hidden layer with ExU units and the rest with ReLU. 
        2. With ReLU / GeLU activation unit: hidden layers with ReLU / GeLU units. """
        layers = list()
        if len(self.hidden_sizes) == 0 or self.activation is not True: 
            hidden_sizes = [self.in_features] + self.hidden_sizes + [1]
            for in_f, out_f in zip(hidden_sizes[:], hidden_sizes[1:]):
                layers.append(nn.Linear(in_f, out_f,  bias=True))
        else:
            if self.activation_cls == 'exu': 
                hidden_sizes = [self.in_features] + self.hidden_sizes
                for index, (in_f, out_f) in enumerate(zip(hidden_sizes[:], hidden_sizes[1:])):
                    if index == 0: 
                        layers.append(ExU(in_f, out_f))
                    else:
                        layers.append(nn.Linear(in_f, out_f,  bias=True))
                        layers.append(nn.ReLU())
                        #layers.append(LinReLU(in_f, out_f))
            else:
                hidden_sizes = [self.in_features] + self.hidden_sizes
                if self.activation_cls == 'gelu': 
                    activation_cls = nn.GELU
                elif self.activation_cls == 'relu':
                    activation_cls = nn.ReLU
                elif self.activation_cls == 'elu': 
                    activation_cls = nn.ELU
                elif self.activation_cls == 'leakyrelu':
                    activation_cls = nn.LeakyReLU
                for in_f, out_f in zip(hidden_sizes[:], hidden_sizes[1:]):
                    layers.append(nn.Linear(in_f, out_f,  bias=True))
                    layers.append(activation_cls())
            layers.append(nn.Linear(self.hidden_sizes[-1], 1, bias=True))
        return nn.Sequential(*layers)
            
        
    def forward(self, inputs) -> torch.Tensor:
        """
        Args: 
        inputs of shape (batch_size): a batch of inputs 
        Return of shape (batch_size, out_features) = (batch_size, 1): a batch of outputs 
        
        """
        outputs = inputs.unsqueeze(1)
        outputs = self.model(outputs)
        return outputs
    
    
    @property 
    def la(self): 
        if self._la is None:
            self._la = Laplace(model=self.model, likelihood=self.likelihood,
                      subset_of_weights=self.subset_of_weights, 
                      hessian_structure=self.hessian_structure, 
                      sigma_noise=self.sigma_noise, 
                      prior_precision=self.prior_precision, 
                      prior_mean=self.prior_mean, 
                      temperature=self.temperature, 
                      backend=self.backend)
        else:
            self._la.prior_precision = self.prior_precision
            self._la.sigma_noise = self.sigma_noise
            
        return self._la
    
    
    def fit(self, loader, override=True): 
        self.la.fit(loader, override=override)

    
    @property
    def posterior_precision(self): 
        return self.la.posterior_precision
    
    @property
    def mean(self):
        return self.la.mean
    
    
    @property 
    def log_det_ratio(self): 
        return self.la.log_det_ratio
    
    @property
    def log_det_posterior_precision(self):
        return self.la.log_det_posterior_precision
    
    @property 
    def scatter(self): 
        return self.la.scatter 
    
    
    def log_marginal_likelihood(self, prior_precision=None, sigma_noise=None):
        if prior_precision is not None: 
            self.prior_precision = prior_precision
        if sigma_noise is not None: 
            self.sigma_noise = sigma_noise
            
        return self.la.log_marginal_likelihood(self.prior_precision, self.sigma_noise)
    
    
    def predictive_samples(self, x, pred_type='glm', n_samples=100):
        return self.la.predictive_samples(x, pred_type=pred_type, n_samples=n_samples)
        