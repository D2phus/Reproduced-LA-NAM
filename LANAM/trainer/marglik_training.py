import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils import parameters_to_vector
from torch.utils.data import Dataset, DataLoader

from copy import deepcopy

import numpy as np
from laplace.curvature import BackPackGGN
from laplace import Laplace

from LANAM.utils.plotting import *
from LANAM.trainer.training import test
import wandb

import os

def marglik_training(model,
                     train_loader,
                     train_loader_fnn, 
                     test_loader,
                     likelihood, 
                     
                     use_wandb=False,
                     test_samples=None,
                     backend=BackPackGGN,
                     
                     optimizer_cls=torch.optim.Adam, 
                     optimizer_kwargs=None, 
                     scheduler_cls=None,
                     scheduler_kwargs=None,
                     
                     n_epochs = 400,
                     lr_hyp = 1e-1,
                     n_epochs_burnin=50, 
                     n_hypersteps=30, 
                     marglik_frequency = 100,
                     
                     prior_prec_init=1.0, 
                     sigma_noise_init=1.0, 
                     temperature=1.0,
                     plot_recovery=False,
                     ): 
    """
    online learning the hyper-parameters.
    the prior p(\theta_i)=N(\theta_i; 0, \gamma^2)
    Args:
    -----------
    temperature: float
        higher temperature leads to a more concentrated prior.
    """    
    # get device
    device = parameters_to_vector(model.parameters()).device
    log_frequency = 50
    in_features = model.in_features
    model.temperature = temperature
    if use_wandb:
        for fnn in model.feature_nns: 
            wandb.watch(fnn, log_freq=log_frequency) # log gradients; note that wandb.watch only supports nn.Module object. (not for ModuleList, Tuple, ...)
        
    N = len(train_loader.dataset)
    metrics_name = 'MSE' if likelihood=='regression' else 'Cross_Entropy_Loss'
    
    # set up hyperparameters and loss function
    hyperparameters = list()
    log_prior_prec_init = np.log(temperature*prior_prec_init)
    log_prior_prec = torch.ones(in_features, device=device) * log_prior_prec_init
    log_prior_prec.requires_grad = True # note to require grad
    hyperparameters.append(log_prior_prec)

    if likelihood == 'regression': 
        criterion = nn.MSELoss(reduction='mean')
        log_sigma_noise_init = np.log(sigma_noise_init)
        log_sigma_noise = torch.ones(in_features, device=device)*log_sigma_noise_init
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
                
                
    for epoch in range(1, n_epochs+1):
        epochs_loss = 0.0
        epoch_perf = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            if likelihood == 'regression':
                sigma_noise = log_sigma_noise.exp().detach()
                crit_factor = temperature / (2 * sigma_noise.square()) #  of shape (in_features)
            else:
                crit_factor = temperature
            prior_prec = log_prior_prec.exp().detach()
            theta = torch.stack([parameters_to_vector(fnn.parameters()) for fnn in model.feature_nns]) # parameters, of shape (in_features, num_params)
            delta = expand_prior_precision(prior_prec, model) # prior precision, of shape (in_features, num_params)
            
            f, _= model(X)
            step_loss = criterion(f, y) + 0.5 * torch.sum((delta * theta * theta).sum(dim=1) / N / crit_factor)
            step_loss.backward()
            optimizer.step()
                
            epochs_loss += step_loss*len(y)
            if likelihood == 'regression': # MSE
                epoch_perf += (f.detach() - y).square().sum()
            else: # Accuracy
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y) # the number of correct prediction.
            if scheduler_cls is not None:
                scheduler.step()
            
        losses.append(epochs_loss / N)  
        perfs.append(epoch_perf / N)
          
        if use_wandb:
            # saving model training loss and metrics to W&B
            wandb.log({
                    'Loss': losses[-1], 
                    metrics_name: perfs[-1], 
            })
                
        # optimize hyper-parameters when epoch >= n_epochs_burnin and epoch == marglik_frequency
        if (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
            continue

        # fit laplace approximation 
        sigma_noise = 1 if likelihood == 'classification' else log_sigma_noise.exp()
        prior_prec = log_prior_prec.exp()
            
        model.sigma_noise = sigma_noise
        model.prior_precision = prior_prec
        for fnn in model.feature_nns:
            fnn._la = None # Re-init laplace for each feature network.
        model.fit(epoch_perf, train_loader_fnn)
        
        if test_samples is not None and plot_recovery: 
            features, targets, feature_targets = test_samples
            _, _, f_mu_fnn, f_var_fnn = model.predict(features) # epistemic uncertainty
            fig = plot_recovered_functions(features, targets, feature_targets, f_mu_fnn, f_var_fnn)    
                
                
        # maximize the marginal likelihood
        for idx in range(n_hypersteps):
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
            
            #print(f'[Epoch={epoch}, n_hypersteps={idx}]: prior precision: {prior_prec.detach().numpy()}, sigma noise: {sigma_noise.detach().numpy()}')
            #print(margliks[-1])
            if use_wandb:
                # saving negative marginal likelihood and sigma noise to W&B
                wandb.log({
                        'Negative_marginal_likelihood': margliks[-1], 
                        'Sigma_noise': model.additive_sigma_noise.detach().numpy().item(),
                })
        
        print(f'[Epoch={epoch}, MSE: {perfs[-1]}, n_hypersteps={idx}]: prior precision: {prior_prec.detach().numpy()}, sigma noise: {sigma_noise.detach().numpy()}')
        
        # best model selection.
        if margliks[-1] < best_marglik:
            best_model_dict = deepcopy(model.state_dict())
            best_marglik = margliks[-1]
        
        if test_samples is not None and plot_recovery and epoch%15==0: 
            features, targets, feature_targets = test_samples
            _, _, f_mu_fnn, f_var_fnn = model.predict(features) # epistemic uncertainty
            fig = plot_recovered_functions(features, targets, feature_targets, f_mu_fnn, f_var_fnn)    
            
            if use_wandb:
                wandb.log({
                    'Recovery_functions': wandb.Image(fig),
                })
        
    print('MARGLIK: finished training. Recover best model and fit Laplace.')
    if best_model_dict is not None: 
        model.load_state_dict(best_model_dict)
    best_metrics = test(likelihood, device, model, test_loader)
    
    # saving model to W&b
    # 4. Log an artifact to W&B
    if use_wandb:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
        wandb.log({f'Best_{metrics_name}': best_metrics})
        
    return model, margliks, losses, perfs


def expand_prior_precision(prior_prec, model):
    """expand the prior precision variable to shape (in_features, num_params)
    """
    assert prior_prec.ndim == 1 
    P = torch.stack([parameters_to_vector(fnn.parameters()) for fnn in model.feature_nns]) # number of parameters in each feature net, (in_features, num_params)
    prior_prec = torch.stack([torch.ones_like(param)*prec for prec, param in zip(prior_prec, P)])
    return prior_prec