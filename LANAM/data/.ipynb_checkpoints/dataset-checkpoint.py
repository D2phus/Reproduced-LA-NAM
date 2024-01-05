from typing import Dict, List

import copy
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_diabetes

from LANAM.data.base import LANAMDataset, LANAMSyntheticDataset
from LANAM.data.generator import  *
from LANAM.config.default import defaults

"""build dataset from raw data."""
cfg = defaults()

def linear_regression_example(rho, sigma=0, config=cfg, num_samples=1000, min_max=False):
    """
    y = X_1 + X_2, X_1, X_2 \sim N(\boldsymbol{0}, \Sigma^{-1}) 
    \Sigma = [[1, \rho], [\rho, 1]]
    
    """
    input_0, input_1 = np.random.multivariate_normal(np.zeros(2), 2*np.array([[1, rho], [rho, 1]]), size=num_samples).T
    output_0, output_1 = input_0, input_1
    y = output_0 + output_1 
    noise = np.random.randn(*y.shape)*sigma
    y += noise
    
    data = pd.DataFrame()
    data['input_0'] = pd.Series(input_0)
    data['input_1'] = pd.Series(input_1)
    data['target'] = pd.Series(y)
    
    return LANAMSyntheticDataset(config, data=data, features_columns=data.columns[:-1], targets_column=data.columns[-1], feature_targets=torch.Tensor(np.stack([output_0, output_1], axis=1)), sigma=sigma, min_max=min_max)

def interaction_example(rho, config=cfg, num_samples=1000, min_max=True): 
    """
    y = min(x1, x2) + 0.1*x1 + 0.1*x2, 
    x1, x2 \sim N(0, [[1, \rho], [\rho, 1]])
    """
    input_0, input_1 = np.random.multivariate_normal(np.zeros(2), 2*np.array([[1, rho], [rho, 1]]), size=num_samples).T
    
    output_0, output_1 = -0.1*input_0, -0.1*input_1
    y = np.minimum(input_0, input_1) + output_0 + output_1
    
    data = pd.DataFrame()
    data['input_0'] = pd.Series(input_0)
    data['input_1'] = pd.Series(input_1)
    data['target'] = pd.Series(y)
    
    return LANAMSyntheticDataset(config, data=data, features_columns=data.columns[:-1], targets_column=data.columns[-1], feature_targets=torch.Tensor(np.stack([output_0, output_1], axis=1)), sigma=0, min_max=min_max)


def load_nonlinearly_dependent_2D_examples(config=cfg, 
                                           num_samples: int=10000, 
                                           sigma: float=0,
                                           sampling_type: str='uniform',      
                                           generate_functions: List=[lambda x: torch.zeros_like(x), lambda x: x],                             
                                           dependent_functions=torch.abs) -> LANAMSyntheticDataset:
    """generate nonlinearly dependent 2D examples. 
    Args: 
    generate_functions: the additive structure. 
    dependent_functions: how X_2 is generated from X_1. 
    """
    if sampling_type not in ['normal', 'uniform']:
        raise ValueError('Invalid input type for `sampling_type`.')
    
    in_features = len(generate_functions)
    feature_names = [f'X{idx+1}' for idx in range(in_features)]
    
    if sampling_type == 'uniform':
        Z = torch.FloatTensor(num_samples).uniform_(-1, 1)
    else: 
        Z = torch.zeros(num_samples)
        torch.nn.init.trunc_normal_(Z, mean=0, std=1, a=-1, b=1) 
    
    X1 = copy.deepcopy(Z)
    X2 = dependent_functions(copy.deepcopy(Z))
    X = torch.stack([X1, X2], dim=1)
    f = torch.stack([generate_functions[idx](x) for idx, x in enumerate(torch.unbind(X, dim=1))], dim=1) 
    y = f.sum(dim=1)
    noise = torch.randn_like(y)*sigma
    y += noise

    data = pd.DataFrame(X, columns = feature_names)
    data['target'] = pd.Series(y)

    return LANAMSyntheticDataset(config,
                                 data=data, 
                                 features_columns=data.columns[:-1], 
                                 targets_column=data.columns[-1], 
                                 feature_targets=f, 
                                 sigma=0)


def load_multicollinearity_data(generate_functions, 
                                config=cfg, 
                                x_lims=(-1, 1), 
                                num_samples=10000, 
                                sigma=0, 
                                sampling_type='uniform'): 
    '''generate dataset with a known additive structure. features are perfectly correlated, where Xi are fixed to identical samples.
    
    NOTE: redundant; can be implemented by method `load_nonlinearly_dependent_2D_examples` above. '''
    if sampling_type not in ['normal', 'uniform']:
        raise ValueError('Invalid input type for `sampling_type`.')
    
    in_features = len(generate_functions)
    feature_names = [f'X{idx+1}' for idx in range(in_features)]
    
    if sampling_type == 'uniform':
        X1 = torch.FloatTensor(num_samples).uniform_(-1, 1)
    else: 
        X1 = torch.zeros(num_samples)
        torch.nn.init.trunc_normal_(X1, mean=x_lims[0]+(x_lims[1]-x_lims[0])/2, std=1, a=x_lims[0], b=x_lims[1])
        
    X2 = [copy.deepcopy(X1) for _ in range(in_features-1)] # NOTE: deepcopy!
    X = torch.stack([X1] + X2, dim=1)
    f = torch.stack([generate_functions[idx](x) for idx, x in enumerate(torch.unbind(X, dim=1))], dim=1)
    y = f.sum(dim=1) 
    noise = torch.randn_like(y)*sigma
    y += noise

    data = pd.DataFrame(X, columns = feature_names)
    data['target'] = pd.Series(y)

    return LANAMSyntheticDataset(config,
                                 data=data, 
                                 features_columns=data.columns[:-1], 
                                 targets_column=data.columns[-1], 
                                 feature_targets=f, 
                                 sigma=0)

def load_synthetic_data(config=cfg, 
                        x_lims=(0, 1),
                       num_samples=1000,
                       generate_functions=synthetic_example(),
                       sigma=1, 
                       sampling_type='uniform',
                       ): 
    """build dataset with a known additive structure, with uncorrelated features.
    Args: 
    -----
    sampling_type: str
        the distribution for X sampling. 
        - uniform: U[*x_lims]^in_features, 
        - normal: N(x_lims[0]+(x_lims[1]-x_lims[0])/2, 1) truncated by x_lims
    """
    if sampling_type not in ['normal', 'uniform']:
        raise ValueError('Invalid input type for `sampling_type`.')
    config.likelihood = 'regression'
    in_features = len(generate_functions)
    feature_names = [f'X{idx+1}' for idx in range(in_features)]
    
    if sampling_type == 'uniform':
        X = torch.FloatTensor(num_samples, in_features).uniform_(x_lims[0], x_lims[1])
    else:
        X = torch.zeros(num_samples, in_features)
        torch.nn.init.trunc_normal_(X, mean=x_lims[0]+(x_lims[1]-x_lims[0])/2, std=1, a=x_lims[0], b=x_lims[1])
    
    feature_targets = torch.stack([generate_functions[index](x_i) for index, x_i in enumerate(torch.unbind(X, dim=1))], dim=1) # (batch_size, in_features) 
    y = feature_targets.sum(dim=1) # of shape (batch_size, 1)
    noise = torch.randn_like(y)*sigma
    y += noise

    data = pd.DataFrame(X, columns = feature_names)
    data['target'] = pd.Series(y)
    
    return LANAMSyntheticDataset(config,
                        data=data,
                        features_columns=data.columns[:-1],
                        targets_column=data.columns[-1], 
                        feature_targets=feature_targets, 
                        sigma=sigma)
    
    

def load_concurvity_data(config=cfg,
                         num_samples=1000,
                         sigma_1=0.05, 
                         sigma_2=0.5):
    
    """likelihood: regression.
    build dataset with a known additive structure and strong concurvity among features.
    Refer to Sec 6 of this paper: Feature selection algorithms in generalized additive models under concurvity."""
    
    def f4(X, sigma=sigma_1): 
        return torch.pow(X[1], 3) + torch.pow(X[2], 2) + torch.randn(X[1].shape)*sigma
        
    def f5(X, sigma=sigma_1):
        return torch.pow(X[2], 2) + torch.randn(X[2].shape)*sigma
        
    def f6(X, sigma=sigma_1):
        return torch.pow(X[1], 2) + torch.pow(X[3], 3) + torch.randn(X[1].shape)*sigma
        
    def f7(X, sigma=sigma_1):
        return X[0]*X[1] + torch.randn(X[0].shape)*sigma
    
    
    config.likelihood = 'regression'
    in_features = 7
    feature_names = [f'X{idx+1}' for idx in range(in_features)]
    
    X = [torch.FloatTensor(num_samples, 1).uniform_(0, 1) for _ in range(3)]
    generate_functions = [f4, f5, f6, f7]
    for f in generate_functions:
        X.append(f(X, sigma_1))
    X = torch.cat(X, dim=1) 
    feature_targets = torch.zeros_like(X) 
    print(feature_targets.shape)
    feature_targets[:, 0], feature_targets[:, 4], feature_targets[:, 5] = 2*torch.pow(X[:, 0], 2), torch.pow(X[:, 4], 3), 2*torch.sin(X[:, 5])
    y = feature_targets.sum(dim=1)
    y+= torch.randn_like(y)*sigma_2
    
    data = pd.DataFrame(X, columns = feature_names)
    data['target'] = pd.Series(y)
    
    return LANAMSyntheticDataset(config,
                        data=data,
                        features_columns=data.columns[:-1],
                        targets_column=data.columns[-1], 
                        feature_targets=feature_targets, 
                        sigma=sigma_2)

def load_breast_data(config=cfg):
    """likelihood: classification."""
    breast_cancer = load_breast_cancer()
    dataset = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
    dataset['target'] = breast_cancer.target
    
    config.likelihood = 'classification'
        
    return LANAMDataset(config,
                        data=dataset,
                        features_columns=dataset.columns[:-1],
                        targets_column=dataset.columns[-1])


def load_sklearn_housing_data(config=cfg): 
    """likelihood: regression, N: 20640, D: 8"""
    housing = fetch_california_housing()

    dataset = pd.DataFrame(data=housing.data, columns=housing.feature_names)
    dataset['target'] = housing.target
    
    config.likelihood = 'regression'
     
    return LANAMDataset(config,
                        data=dataset,
                        features_columns=dataset.columns[:-1],
                        targets_column=dataset.columns[-1])


def load_diabetes_data(config=cfg): 
    diabetes = load_diabetes()
    dataset = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    dataset['target'] = diabetes.target
    
    config.likelihood = 'regression'
     
    return LANAMDataset(config,
                        data=dataset,
                        features_columns=dataset.columns[:-1],
                        targets_column=dataset.columns[-1])

def load_autompg_data(config=cfg, 
                      data: str='LANAM/data/datasets/autompg-dataset/auto-mpg.csv', 
                      features_columns: list = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin'], 
                      targets_column: str = 'mpg'):
    config.likelihood = 'regression'
    
    data = pd.read_csv(data)
    data.drop(['car name'], axis=1, inplace=True)
    data['horsepower'].replace('?',np.nan,inplace=True)
    data['horsepower'] = pd.to_numeric(data['horsepower'])
    data.fillna(data['horsepower'].mean(skipna=True),axis=0,inplace=True)
    
    data[targets_column] = data[targets_column]
    return LANAMDataset(config,
                        data=data,
                        features_columns=features_columns,
                        targets_column=targets_column)


def load_gap_data(config=cfg, 
                   num_samples=1000,
                   generate_functions=synthetic_example(),
                   sigma=1, 
                   sampling_type='uniform'
                   ): 
    if sampling_type not in ['normal', 'uniform']:
        raise ValueError('Invalid input type for `sampling_type`.')
    config.likelihood = 'regression'
    in_features = len(generate_functions)
    feature_names = [f'X{idx+1}' for idx in range(in_features)]
    
    if sampling_type == 'uniform':
        X_1 = torch.FloatTensor(int(np.ceil(num_samples*0.495)), in_features).uniform_(0, 1)*0.35
        X_2 = torch.FloatTensor(int(np.ceil(num_samples*0.01)), in_features).uniform_(0, 1)*0.3+0.4
        X_3 = torch.FloatTensor(int(np.floor(num_samples*0.495)), in_features).uniform_(0, 1)*0.35+0.65
        X = torch.cat([X_1, X_2, X_3], dim=0)
        print(X.shape)
    else: 
        X = torch.zeros(num_samples, in_features)
        torch.nn.init.trunc_normal_(X, mean=0.5, std=0.2, a=0, b=1)
        
    feature_targets = torch.stack([generate_functions[index](x_i) for index, x_i in enumerate(torch.unbind(X, dim=1))], dim=1) # (batch_size, in_features) 
    y = feature_targets.sum(dim=1) # of shape (batch_size, 1)
    noise = torch.randn_like(y)*sigma
    y += noise
    
    data = pd.DataFrame(X, columns = feature_names)
    data['target'] = pd.Series(y)
    
    return LANAMSyntheticDataset(config,
                        data=data,
                        features_columns=data.columns[:-1],
                        targets_column=data.columns[-1], 
                        feature_targets=feature_targets, 
                        sigma=sigma)
    
    
 