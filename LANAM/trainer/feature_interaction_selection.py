from itertools import combinations
import torch 
from torch.nn.utils import parameters_to_vector
from LANAM.utils.output_filter import OutputFilter
from LANAM.extensions.backpack import BackPackGGNExt
from LANAM.models.featurenn import FeatureNN
from laplace import Laplace

        
def feature_interaction_selection(model, train_loader, k=5, subset_of_weights='all'):
    """
    Select top k feature interactions among feature pairs. 
    Use fully Hessian approximation: joint posterior information is exactly what we want.
    
    Args: 
    -------
    model: 
        trained LaNAM under the presumption that each feature is independent.
    train_loader: 
        trainining data loader.
    k: 
        top-k pairs.
    subset_of_weights: 
        only `all` is supported now
    
    Returns: 
    --------
    k_pairs: list
        top-k feature pairs with more mutual information. k = min(num_all_pairs, k)
    sorted_mis: Dict{Tuple: float}
        feature pairs with their mutual information, order by descending mutual information value.
    """
    if subset_of_weights not in ['all', 'last_layer']:
        raise ValueError('`subset_of_weights` type is not supported.')
        
    likelihood = model.likelihood
    in_features = model.in_features 
    num_pairs = int(in_features*(in_features-1)/2) # number of different feature pairs
    k = min(k, num_pairs)
    
    # compute the MI for all pairs
    combi = combinations([*range(in_features)], 2) # pairs of interaction
    mis = dict()
    for c in combi: 
        idx1, idx2 = c # idx1 < idx2
        # independent posterior 
        log_det_pos_prec1 = model.feature_nns[idx1].log_det_posterior_precision.detach().item()
        log_det_pos_prec2 = model.feature_nns[idx2].log_det_posterior_precision.detach().item()
        
        # joint posterior 
        _model = OutputFilter(model, list(c)) # customize model to have single output for Laplace 
        la = Laplace(_model, likelihood, subset_of_weights=subset_of_weights, hessian_structure='full', backend=BackPackGGNExt) # post-hoc laplace. Note that customized backennd is used, which does not clean up hooks
        la.fit(train_loader) 
        joint_pos_cov = la.posterior_covariance
        log_det_joint_pos_cov = joint_pos_cov.logdet().detach().item() # for `full` hessian structure
        
        # calculate mutual information
        mi = -0.5 * (log_det_joint_pos_cov+log_det_pos_prec1+log_det_pos_prec2)
        mis[c] = mi
    
    sorted_mis = {k: v for k, v in sorted(mis.items(), key=lambda item: item[1], reverse=True)}
    k_pairs = list(sorted_mis)[:k] # top-k
    return k_pairs, sorted_mis

    
def feature_net_params_index(model):
    """The index range of each feature net in the `full`, `all` parameters model posterior precision matrix.
    
    Returns: 
    ------
    fnn_param_index, list[Tuple]: start and end index for each feature net.
    """
    in_features = model.in_features 
    P = torch.stack([torch.tensor(len(parameters_to_vector(fnn.parameters()))) for fnn in model.feature_nns]) # (in_features), number of parameters in each feature network 
    fnn_param_index = list()
    for idx in range(in_features):
        s, e = P[:idx].sum().item(), P[:idx+1].sum().item()
        fnn_param_index.append((s, e))
        
    return fnn_param_index
        