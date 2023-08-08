from itertools import combinations

import torch 

def mutual_information(model):
    if model.subset_of_weights == 'full':
    elif model.subset_of_weights == 'last_layer':
    else:
        raise ValueError('In valid type for `subset_of_weights`. Only `full` and `last_layer` are supported.')
def mi_interaction_selection(model, 
                            ):
    in_features = model.in_features 
    combi = combinations([*range(in_features)])
    for (idx1, idx2) in combi: 
        pos_cov1 = model.feature_nns[idx1].posterior_covariance
        pos_cov2 = model.feature_nns[idx2].posterior_covariance
        