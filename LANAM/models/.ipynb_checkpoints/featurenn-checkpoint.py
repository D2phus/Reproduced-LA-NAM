"""DNN-based sub net for each input feature."""
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from laplace import Laplace


class FeatureNN(nn.Module):
    def __init__(
        self, 
        config, 
        name: str, 
        in_features: int,
        feature_index: int, 
        
        subset_of_weights='all', 
        hessian_structure='full',
        prior_mean=0.0, 
        prior_precision=1.0, 
        sigma_noise=1.0, 
        temperature=1.0
        ) -> None: # type-check
            """
            Args:
            -----------
            in_features: scalar, size of each input sample; default value = 1
            num_units: scalar, number of ExU/LinearReLU hidden units in the first hidden layer 
            feature_index: indicate which feature is learn in this subnet
            prior_precision: the prior precision of model parameters.
            sigma_noise: the observation noise of feature networks. 
            
            """
            super(FeatureNN, self).__init__()
            self.name = name
            self.config = config
            self.in_features = in_features
            self.feature_index = feature_index
            self.hidden_sizes = self.config.hidden_sizes
            self.subset_of_weights = subset_of_weights
            self.hessian_structure = hessian_structure
            self.prior_mean = prior_mean
            self.prior_precision = prior_precision
            self.sigma_noise = sigma_noise
            self.temperature = temperature
            
            self.model = self.init_model()
            
            # Laplace approximation
            self.likelihood = 'regression' if self.config.regression else 'classification'
            
            self._la = None

    def init_model(self):
        layers = list()
        if len(self.hidden_sizes) == 0:
            layers.append(nn.Linear(self.in_features, 1, bias=True))
        else:
            hidden_sizes = [self.in_features] + self.hidden_sizes
            for in_f, out_f in zip(hidden_sizes[:], hidden_sizes[1:]):
                    layers.append(nn.Linear(in_f, out_f,  bias=True))
                    if self.config.activation: 
                        layers.append(nn.GELU())
            # no activation and bias for the output layer
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
            self._la = Laplace(self.model, self.likelihood,
                             subset_of_weights=self.subset_of_weights, 
                             hessian_structure=self.hessian_structure, 
                             sigma_noise=self.sigma_noise, # 
                             prior_precision=self.prior_precision, 
                             prior_mean=self.prior_mean, 
                             temperature=self.temperature)
            
        else: 
            self._la.sigma_noise = self.sigma_noise
            self._la.prior_precision = self.prior_precision
            #self._la.__dict__.update(_sigma_noise=self.sigma_noise, 
             #                _prior_precision=self.prior_precision)
        return self._la
    
    def fit(self, loader, override=True): 
        self.la.fit(loader, override=override)
    
    @property
    def posterior_covariance(self): 
        return self.la.posterior_covariance
    
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
        