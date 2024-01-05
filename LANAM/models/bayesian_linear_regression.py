import torch 
from math import pi, log

def identity_basis_function(x):
    return x

def rbf_basis_function(x, mu, sigma=0.1):
    return torch.exp(-0.5*(x-mu)**2/sigma**2)
   
def polynomial_basis_function(x, power):
    return x**power

class BayesianLinearRegression(): 
    def __init__(self, X, y, bf='identity', bf_args=None, sigma_noise=1.0, prior_var=1.0):
        """
        Bayesian linear basis regression.  \hat{y} = \sum \omega^T \Phi(x) + \epsilon.
        
        Args: 
        X of shape (num_samples, num_params-1), y of shape (num_samples, out_features): the training samples
        sigma_noise, scalar: the standard deviation of centered Gaussian noise
        prior_var, scalar: the variance of parameter prior, same for all parameters. 
        
        Attrs: 
        X of shape (num_samples, num-params)
        num_samples, scalar: the number of training samples
        num_params, scalar: the number of parameters
        """
        if bf not in ['rbf', 'polynomial', 'identity']: 
            raise ValueError('the basis function has to be identity, rbf, or polynomial')
        if bf != 'identity': 
            raise NotImplementedError
        
        self.bf = bf 
        self.bf_args = bf_args
        if self.bf == 'rbf':
            self.basis_function = rbf_basis_function
        elif self.bf == 'polynomial': 
            self.basis_function = polynomial_basis_function
        else: 
            self.basis_function = identity_basis_function
        
        self.Phi = self.basis_function(X)
        
        # self.Phi = self.expand(X)
        self.num_samples, self.num_params = self.Phi.shape
        self.y = y
        
        self.sigma_noise = torch.tensor(sigma_noise)
        self.prior_var = torch.tensor(prior_var)
        
    def expand(self, x): 
        # expand a column vector \Phi(mathbf{1}) for the bias.
        bias = torch.ones(x.shape[0], 1)
        if self.bf_args is None:
            return torch.cat((self.basis_function(x), bias), 1)
        else:
            return torch.cat([self.basis_function(x, bf_arg) for bf_arg in self.bf_args] + [bias], 1)
        
    @property 
    def noise_precision(self): 
        return 1/(self.sigma_noise**2)
    
    @property
    def prior_precision(self):
        return 1/self.prior_var
    
    @property
    def prior_precision_diag(self):
        """the diagonal of prior precision of shape (num_params)"""
        return torch.ones(self.num_params)*self.prior_precision
    
    @property 
    def posterior_precision(self): 
        """the parameter posterior precision of shape (num_params, num_params)"""
        cov_lik = self.noise_precision*(self.Phi.T@self.Phi)
        cov_prior = torch.diag(self.prior_precision_diag)
        return cov_lik + cov_prior
    
    @property
    def posterior_cov(self): 
        """the parameter variance of shape (num_params, num_params)"""
        return torch.linalg.inv(self.posterior_precision)
        
    @property 
    def mean(self): 
        """the parameter posterior mean of shape (num_params, 1)"""
        return (self.noise_precision*self.posterior_cov @self.Phi.T@self.y)
    
    @property
    def log_det_posterior_precision(self): 
        return self.posterior_precision.logdet()
    
    def log_marginal_likelihood(self, prior_var=None, sigma_noise=None):
        """the logarithm of marignal likelihood, i.e. evidence, p(D) """
        
        # M: num_params, N: num_samples, mN: mean, SN: variance, alphas: prior_precision, beta: noise_precision
        if prior_var is not None: 
            self.prior_var = prior_var
        if sigma_noise is not None: 
            self.sigma_noise = sigma_noise
        
        diff = self.y - self.Phi@self.mean
        scatter_lik = self.noise_precision*torch.sum(diff**2)
        scatter_prior= self.prior_precision*torch.sum(self.mean**2)
        scatter = scatter_lik + scatter_prior
        
        # note that to compute gradients of parameters, torch.log which has grad attribute should be used, instead of math.log which returns float type!!!
        log_marg_lik = self.num_params*torch.log(self.prior_precision)+self.num_samples*torch.log(self.noise_precision)-scatter-self.log_det_posterior_precision-self.num_samples*log(2*pi) 
        return 0.5*log_marg_lik
    
    def predict(self, X_test): 
        """the prediction p(y|\omega, D)
        Args: 
        X_test of shape (batch_size, num_params-1)
        Returns: 
        pred_mean of shape (batch_size, 1)
        pred_var of shape (batch_size, 1)
        """
        Phi_test = self.expand(X_test)
        pred_mean = Phi_test@self.mean
        pred_var = self.sigma_noise**2 + torch.diagonal(Phi_test@self.posterior_cov@Phi_test.T)
        return pred_mean, pred_var

    
