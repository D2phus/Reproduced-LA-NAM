"""Partial derivatives for the ExU activation function."""
from torch import exp, le, ge, zeros_like, zeros_like

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives

from backpack.utils.subsampling import subsample

class ExUDerivatives(BaseParameterDerivatives):
    """Implement first- and second-order partial derivatives of ExU."""
    def df(
        self, 
        module, 
        g_inp, 
        g_out, 
        subsampling
    ): 
        """First ExU derivative: `ExU'(x) = 0 if e^\omega (x-b) < 0 \ or \ e^\omega (x-b) > n else e^omega x`"""
        input0 = subsample(module.input0, subsampling=subsampling)
        exp_input0 = (input0 - module.bias).matmul(exp(module.weights))
        non_neg = ge(exp_input0, 0)
        le_n = le(exp_input0, module.n)
        non_zero_deriv = torch.logical_and(non_neg, le_n)
        
        result = zeros_like(exp_input0)
        result[non_zero_deriv] = input0.matmul(exp(module.weights))
        
        return result

    def d2f(self, module, g_inp, g_out): 
        """Second ELU derivative: `ExU''(x) = 0 if e^\omega (x-b) < 0 \ or \ e^\omega (x-b) > n else e^omega x`"""
        exp_input0 = (module.input0 - module.bias).matmul(exp(module.weights))
        non_neg = ge(exp_input0, 0)
        le_n = le(exp_input0, module.n)
        non_zero_deriv = torch.logical_and(non_neg, le_n)
        
        result = zeros_like(exp_input0)
        result[non_zero_deriv] = module.input0.matmul(exp(module.weights))
        
        return result