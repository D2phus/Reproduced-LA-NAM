import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils import parameters_to_vector

from copy import deepcopy

import numpy as np
from laplace.curvature import BackPackGGN
from laplace import Laplace

from LANAM.utils.plotting import *

def marglik_training(model,
                     train_loader,
                     loader_fnn, 
                     likelihood, 
                     
                     backend=BackPackGGN,
                     hessian_structure='full',
                     
                     optimizer_cls=torch.optim.Adam, 
                     optimizer_kwargs=None, 
                     scheduler_cls=None,
                     scheduler_kwargs=None,
                     
                     n_epochs = 300,
                     lr_hyp = 1e-1,
                     n_epochs_burnin=0, 
                     n_hypersteps=30, 
                     marglik_frequency = 20,
                     
                     prior_prec_init=1.0, 
                     sigma_noise_init=1.0, 
                     temperature=1.0,
                     ): 
    """
    online learning the hyper-parameters.
    the prior p(\theta_i)=N(\theta_i; 0, \gamma^2)
    Args:
    -----------
    temperature: higher temperature leads to a more concentrated prior.
    """    
    log_frequency = 50
    
    in_features = model.in_features
    
    model.temperature = temperature
    
    N = len(train_loader.dataset)
    P = torch.stack([torch.tensor(len(parameters_to_vector(fnn.parameters()))) for fnn in model.feature_nns]) # (in_features), number of parameters in each feature network 
    
    # set up hyperparameters and loss function
    hyperparameters = list()
    log_prior_prec_init = np.log(temperature*prior_prec_init)
    log_prior_prec = torch.ones(in_features) * log_prior_prec_init
    log_prior_prec.requires_grad = True # note to require grad
    hyperparameters.append(log_prior_prec)

    if likelihood == 'regression': 
        criterion = nn.MSELoss(reduction='mean')
        log_sigma_noise_init = np.log(sigma_noise_init)
        log_sigma_noise = torch.ones(in_features)*log_sigma_noise_init
        log_sigma_noise.requires_grad = True
        hyperparameters.append(log_sigma_noise)
    elif likelihood == 'classification':
        criterion = nn.CrossEntropyLoss(reduction='mean')
        sigma_noise = 1.
    else: 
        raise ValueError('likelihood should be `regression` or `classification`. ')
        
    # set up model optimizer
    if optimizer_kwargs is None:
        optimizer_kwargs = dict()
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    
    # set up scheduler
    if scheduler_cls is not None:
        if scheduler_kwargs is None:
            scheduler_kwargs = dict()
        scheduler = torch.optim.scheduler_cls(optimizer, **scheduler_kwargs)
    
    # set up hyperparameter optimizer
    hyper_optimizer = torch.optim.Adam(hyperparameters, lr=lr_hyp)
    
    best_marglik = np.inf
    best_model_dict = None
    margliks = list()
    losses = list()
    perfs = list()
    for epoch in range(n_epochs):
        epochs_loss = 0.0
        epoch_perf = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            if likelihood == 'regression':
                sigma_noise = log_sigma_noise.exp().detach()
                crit_factor = temperature / (2 * sigma_noise.square()) #  of shape (in_features)
                #crit_factor = temperature / (2 * sigma_noise.sum().square()) #  
            else:
                crit_factor = temperature
            prior_prec = log_prior_prec.exp().detach()
            theta = torch.stack([parameters_to_vector(fnn.parameters()) for fnn in model.feature_nns]) # parameters, of shape (in_features, num_params)
            delta = expand_prior_precision(prior_prec, model) # prior precision, of shape (in_features, num_params)
            
            f, _= model(X)
            #step_loss = crit_factor*criterion(f, y) + 0.5*torch.sum((delta * theta * theta).sum(dim=1) / N)
            step_loss = criterion(f, y) + 0.5 * torch.sum((delta * theta * theta).sum(dim=1) / N / crit_factor)
            step_loss.backward()
            optimizer.step()
                
            epochs_loss += step_loss*len(y)
            if likelihood == 'regression': 
                epoch_perf += (f.detach() - y).square().sum()
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y) # the number of correct prediction.
            if scheduler_cls is not None:
                scheduler.step()
            
        losses.append(epochs_loss / N)  
        perfs.append(epoch_perf / N)
        # optimize hyper-parameters
        if epoch >= n_epochs_burnin and epoch % marglik_frequency == 0:
            # fit laplace approximation 
            sigma_noise = 1 if likelihood == 'classification' else log_sigma_noise.exp()
            prior_prec = log_prior_prec.exp()
            
            model.sigma_noise = sigma_noise
            model.prior_precision = prior_prec
            model.fit(epoch_perf, loader_fnn)
            
            # maximize the marginal likelihood
            for _ in range(n_hypersteps):
                hyper_optimizer.zero_grad()
                if likelihood == 'classification': # sigma_noise will be constant 1 for classification. 
                    sigma_noise = None 
                else:
                    sigma_noise = log_sigma_noise.exp()
                    #sigma_noise = None
                    
                prior_prec = log_prior_prec.exp()
                neg_log_marglik = -model.log_marginal_likelihood(prior_prec, sigma_noise)
                neg_log_marglik.backward()
                hyper_optimizer.step()
                margliks.append(neg_log_marglik.item())
            
            # best model selection.
            if margliks[-1] < best_marglik:
                best_model_dict = deepcopy(model.state_dict())
                best_marglik = margliks[-1]
        if epoch % log_frequency == 0:
            print(f'EPOCH={epoch+1}: epoch_loss={losses[-1]: .3f}, epoch_perf={perfs[-1]: .3f}')
            
    print('MARGLIK: finished training. Recover best model and fit Laplace.')
    if best_model_dict is not None: 
        model.load_state_dict(best_model_dict)
    return model, margliks, losses


def expand_prior_precision(prior_prec, model):
    """expand the prior precision variable to shape (in_features, num_params)
    """
    assert prior_prec.ndim == 1 
    P = torch.stack([parameters_to_vector(fnn.parameters()) for fnn in model.feature_nns]) # (in_features, num_params)
    prior_prec = torch.stack([torch.ones_like(param)*prec for prec, param in zip(prior_prec, P)])
    return prior_prec