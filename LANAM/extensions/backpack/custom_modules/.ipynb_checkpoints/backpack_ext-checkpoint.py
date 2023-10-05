import torch

from backpack import backpack, extend, memory_cleanup
from backpack.extensions import DiagGGNExact, DiagGGNMC, KFAC, KFLR, SumGradSquared, BatchGrad
from backpack.context import CTX

from laplace.curvature import BackPackInterface, BackPackGGN
from laplace.utils import Kron

from LANAM.extensions.backpack.firstorder.batchgrad import BatchGradExU
from LANAM.models.activation import ExU

   
class BackPackGGNExt(BackPackGGN): 
    """extended implementation of the `BackPackGGN`"""
    def __init__(self, model, likelihood, last_layer=False, subnetwork_indices=None, stochastic=False):
        super().__init__(model, likelihood, last_layer, subnetwork_indices)
        self.stochastic = stochastic
        # register module-computation mapping
        self.BatchGrad_ext = BatchGrad()
        self.BatchGrad_ext.set_module_extension(ExU, BatchGradExU())
    
    
    def jacobians(self, x, enable_backprop=False):
        """Compute Jacobians \\(\\nabla_{\\theta} f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using backpack's BatchGrad per output dimension.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.
        enable_backprop : bool, default = False
            whether to enable backprop through the Js and f w.r.t. x

        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, parameters, outputs)`
        f : torch.Tensor
            output function `(batch, outputs)`
        """
        model = extend(self.model)
        to_stack = []
        for i in range(model.output_size):
            model.zero_grad()
            out = model(x)
            with backpack(self.BatchGrad_ext):
                if model.output_size > 1:
                    out[:, i].sum().backward(
                        create_graph=enable_backprop, 
                        retain_graph=enable_backprop
                    )
                else:
                    out.sum().backward(
                        create_graph=enable_backprop, 
                        retain_graph=enable_backprop
                    )
                
                #for idx, (name, param) in enumerate(model.named_parameters()):
                #    if idx > 0: 
                #        break
                #    print(name, vars(param))
                to_cat = []
                for param in model.parameters():
                    to_cat.append(param.grad_batch.reshape(x.shape[0], -1))
                    delattr(param, 'grad_batch')
                Jk = torch.cat(to_cat, dim=1)
                if self.subnetwork_indices is not None:
                    Jk = Jk[:, self.subnetwork_indices]
                    
            to_stack.append(Jk)
            if i == 0:
                f = out

        model.zero_grad()
        #CTX.remove_hooks()
        #_cleanup(model)
        
        if model.output_size > 1:
            return torch.stack(to_stack, dim=2).transpose(1, 2), f
        else:
            return Jk.unsqueeze(-1).transpose(1, 2), f
        
        
def _cleanup(module):
    for child in module.children():
        _cleanup(child)

    setattr(module, "_backpack_extend", False)
    memory_cleanup(module)
    
    