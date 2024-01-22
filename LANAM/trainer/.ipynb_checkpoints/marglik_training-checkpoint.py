import math 

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils import parameters_to_vector
from torch.utils.data import Dataset, DataLoader

from copy import deepcopy

import numpy as np
from laplace.curvature import BackPackGGN
from laplace import Laplace

from LANAM.utils import * 
import wandb
from tqdm import tqdm

import os

def marglik_training(model,
                     train_loader,
                     val_loader,
                     likelihood, 
                     test_samples=None,
                     
                     use_wandb=False,
                     backend=BackPackGGN,
                     
                     optimizer_cls=torch.optim.Adam, 
                     optimizer_kwargs=None, 
                     scheduler_cls=None,
                     scheduler_kwargs=None,
                     
                     n_epochs = 500,
                     lr_hyp = 1e-1,
                     n_epochs_burnin=50, 
                     n_hypersteps=30, 
                     marglik_frequency = 100,
                     
                     prior_prec_init=1.0, 
                     sigma_noise_init=1.0, 
                     temperature=1.0,
                     plot_recovery=False,
                     
                     log_loss_frequency=100, 
                     
                     
                     reg_perctile_epochs_burnin=0.05, 
                     concurvity_regularization=0, 
                     hsic_regularization=0, 
                     l1_regularization=0, 
                     ): 
    """
    online learning the hyper-parameters.
    Gaussian prior for model parameters: p(\theta_i)=N(\theta_i; 0, \gamma^2)
    Args:
    -----------
    temperature: float
        higher temperature leads to a more concentrated prior.
    """    
    device = parameters_to_vector(model.parameters()).device
    in_features = model.in_features
    model.temperature = temperature
    
    N = len(train_loader.dataset)
    metrics_name = 'RMSE' if likelihood=='regression' else 'Accuracy'
    
    # set up prior: precision 
    hyperparameters = list()
    if np.isscalar(prior_prec_init):
        log_prior_prec_init = np.log(temperature*prior_prec_init)
        log_prior_prec = torch.ones(in_features, device=device) * log_prior_prec_init
    else: 
        if len(prior_prec_init)!= in_features:
            raise ValueError('`prior_prec_init` should be either scalar or tensor of length `in_features`') 
            
        if torch.is_tensor(prior_prec_init):
            log_prior_prec = torch.log(prior_prec_init)
        else:
            log_prior_prec = torch.log(torch.Tensor(prior_prec_init))
        
    log_prior_prec.requires_grad = True # note to require grad
    hyperparameters.append(log_prior_prec)

    # set up prior: observed noise
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
    hyper_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(hyper_optimizer, int(len(train_loader)))
            
    best_marglik = np.inf
    best_model_dict = None
    margliks = list()
    losses = list()
    perfs = list()
    R_perps = list()
                
    for epoch in range(1, n_epochs+1):
        epochs_loss = 0.0
        epoch_perf = 0
        epoch_R_perp = 0.0 
        
        # training 
        for X, y in tqdm(train_loader, desc=f"Epoch: {epoch}"):
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
            
            f, contributions = model(X)
            contributions = torch.cat(contributions, dim=1)
            
            # gaussian prior 
            step_loss = criterion(f, y) + 0.5 * torch.sum((delta * theta * theta).sum(dim=1) / N / crit_factor)
            
            # laplace prior 
            # step_loss = criterion(f, y) + torch.sum((delta * theta.abs()).sum(dim=1) / N / crit_factor * 2 / math.sqrt(2))
            
            # regularization
            if concurvity_regularization > 0 and epoch > int(reg_perctile_epochs_burnin*n_epochs):
                step_loss += concurvity_regularization*concurvity(contributions)
            
            if hsic_regularization > 0 and epoch > int(reg_perctile_epochs_burnin*n_epochs):
                step_loss += hsic_regularization*estimate_hsic(contributions)
            
            if l1_regularization > 0 and epoch > int(reg_perctile_epochs_burnin*n_epochs): 
                step_loss += l1_regularization*Ld_norm(contributions)
                
            step_loss.backward()
            optimizer.step()
                
            epochs_loss += step_loss*len(y)
            
            if likelihood == 'regression': # MSE
                epoch_perf += (f.detach() - y).square().sum()
            else: # Accuracy
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y) # the number of correct prediction.
                
            # concurvity metrics
            epoch_R_perp += concurvity(contributions)*len(y)
            
            if scheduler_cls is not None:
                scheduler.step()
                
        losses.append(epochs_loss / N)  
        perfs.append(epoch_perf / N)
        R_perps.append(epoch_R_perp / N)
        
        # validation 
        val_loss = 0
        val_perf = 0
        val_R_perp = 0
        for X, y in val_loader: 
            X, y = X.to(device), y.to(device)
            
            if likelihood == 'regression':
                sigma_noise = log_sigma_noise.exp().detach()
                crit_factor = temperature / (2 * sigma_noise.square()) #  of shape (in_features)
            else:
                crit_factor = temperature
            prior_prec = log_prior_prec.exp().detach()
            theta = torch.stack([parameters_to_vector(fnn.parameters()) for fnn in model.feature_nns]) # parameters, of shape (in_features, num_params)
            delta = expand_prior_precision(prior_prec, model) # prior precision, of shape (in_features, num_params)
            f, contributions = model(X)
            contributions = torch.cat(contributions, dim=1)
            step_loss = criterion(f, y) + 0.5 * torch.sum((delta * theta * theta).sum(dim=1) / N / crit_factor)
            
            # regularization
            if concurvity_regularization > 0 and epoch > int(reg_perctile_epochs_burnin*n_epochs):
                step_loss += concurvity_regularization*concurvity(contributions)
            
            val_loss += step_loss*len(y)
            
            if likelihood == 'regression': # MSE
                val_perf += (f.detach() - y).square().sum()
            else: # Accuracy
                val_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y) # the number of correct prediction.
            val_R_perp += concurvity(contributions)*len(y)
              
        val_loss /= N
        val_perf /= N
        val_R_perp /= N
        
        if use_wandb:
            # saving model training loss and metrics to W&B
            wandb.log({
                    'Train_Loss': losses[-1], 
                    'Val_Loss': val_loss, 
                    
                    f'Train_{metrics_name}': torch.sqrt(perfs[-1]), 
                    f'Val_{metrics_name}': torch.sqrt(val_perf), 
                    'Train_R_perp': R_perps[-1], 
                    'Val_R_perp': val_R_perp, 
            })
                
        # optimize hyper-parameters when epoch >= n_epochs_burnin and epoch == marglik_frequency
        if (epoch % marglik_frequency) != 0 or epoch < int(reg_perctile_epochs_burnin*n_epochs + n_epochs_burnin):
            continue

        # fit laplace approximation 
        sigma_noise = 1 if likelihood == 'classification' else log_sigma_noise.exp()
        prior_prec = log_prior_prec.exp()
            
        model.sigma_noise = sigma_noise
        model.prior_precision = prior_prec
        for fnn in model.feature_nns:
            fnn._la = None # Re-init laplace for each feature network.
        model.fit(epoch_perf, train_loader)
                
        # maximize the marginal likelihood
        for idx in range(n_hypersteps):
            hyper_optimizer.zero_grad()
            # print(hyper_scheduler.get_last_lr())
            if likelihood == 'classification': # sigma_noise will be constant 1 for classification. 
                sigma_noise = None 
            else:
                sigma_noise = log_sigma_noise.exp()
                #sigma_noise = None
                    
            prior_prec = log_prior_prec.exp()
            neg_log_marglik = -model.log_marginal_likelihood(prior_prec, sigma_noise)
            neg_log_marglik.backward()
            hyper_optimizer.step()
            hyper_scheduler.step()
            margliks.append(neg_log_marglik.item())
            
            if use_wandb:
                # saving negative marginal likelihood and sigma noise to W&B
                wandb.log({
                        'Negative_marginal_likelihood': margliks[-1], 
                        'Sigma_noise': model.additive_sigma_noise.detach().numpy().item(),
                        'Prior_precision_0': prior_prec[0].item(), 
                        'Prior_precision_1': prior_prec[1].item(),
                })
        
        print(f'[Epoch={epoch}], Train_RMSE: {torch.sqrt(perfs[-1]): .3f}, Train_R_perp: {R_perps[-1]: .3f}, Val_RMSE: {torch.sqrt(val_perf): .3f}, Val_R_perp: {val_R_perp: .3f}, n_hypersteps={idx}]: observed noise={model.additive_sigma_noise.detach().item(): .3f}, prior precision={model.prior_precision.detach()}')
        
        # best model selection.
        if margliks[-1] < best_marglik:
            best_model_dict = deepcopy(model.state_dict())
            best_marglik = margliks[-1]
        
        if test_samples is not None and plot_recovery and epoch%log_loss_frequency==0: 
            _test_logging(test_samples, model, use_wandb)
        
            
    print('MARGLIK: finished training. Recover best model and fit Laplace.')
    if best_model_dict is not None: 
        model.load_state_dict(best_model_dict)
        
    if test_samples is not None and plot_recovery: 
        _test_logging(test_samples, model, use_wandb)
    
    # saving model to W&b
    # 4. Log an artifact to W&B
#     if use_wandb:
#         torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
#         wandb.log({f'Best_{metrics_name}': best_metrics})
        
    return model, margliks, losses, perfs


def expand_prior_precision(prior_prec, model):
    """expand the prior precision variable to shape (in_features, num_params)
    """
    assert prior_prec.ndim == 1 
    P = torch.stack([parameters_to_vector(fnn.parameters()) for fnn in model.feature_nns]) # number of parameters in each feature net, (in_features, num_params)
    prior_prec = torch.stack([torch.ones_like(param)*prec for prec, param in zip(prior_prec, P)])
    return prior_prec


def _test_logging(test_samples, model, use_wandb): 
    features, targets, feature_targets, feature_names = test_samples
    f_mu, _, f_mu_fnn, f_var_fnn = model.predict(features) # epistemic uncertainty
            
    fig = plot_recovered_functions(features, targets, feature_targets, f_mu_fnn.flatten(start_dim=1), f_var_fnn.flatten(start_dim=1))  
    importance_fig = plot_feature_importance_errorbar(model, features, feature_names)
    fig_3d = plot_3d(features[:, 0], features[:, 1], f_mu, targets)
    
    R_squared = adjusted_R_squared(features, targets, f_mu)
    if use_wandb:
        wandb.log({
            'Recover_Function': wandb.Image(fig),
            'Overall_Feature_Importance': wandb.Image(importance_fig), 
            'Prediction': wandb.Image(fig_3d), 
            'R_squared': R_squared.item()
            })
    else: 
        print(f'R_squared: {R_squared.item(): .4f}')
        