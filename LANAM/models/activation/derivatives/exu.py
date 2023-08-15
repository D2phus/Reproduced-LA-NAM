from typing import Tuple, List

import torch
from torch import exp, le, ge, zeros_like, zeros_like, einsum, Tensor, Size
from torch.nn import Module

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils.subsampling import subsample


class ExUDerivatives(BaseParameterDerivatives):
    """Partial derivatives of ExU hidden unit.
    Parameters are saved as `.weight` and `.bias` fields.
    Index conventions:
    ------------------
    * v: Free dimension
    * n: Batch dimension
    * o: Output dimension
    * i: Input dimension
    
    `param_mjp` methods: compute matrix-Jacobian products (MJPs) of the module w.r.t. a parameter. Internally calls out to `_{param_str}_jac_t_mat_prod` function that must be immplemented by descendants.
    
    References: 
    -------------------
    https://github.com/f-dangel/backpack/blob/master/backpack/core/derivatives/linear.py
    """ 
    def hessian_is_zero(self, module: Module) -> bool: 
        """ExU hidden unit output is linear w.r.t. to its input.
        Args:
            module: current module.
        """
        return True
    
    def _weight_jac_t_mat_prod(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: int = True,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Batch-apply transposed Jacobian of the output w.r.t. the weight.

        Args:
            module: Linear layer.
            g_inp: Gradients w.r.t. module input. Not required by the implementation.
            g_out: Gradients w.r.t. module output. Not required by the implementation.
            mat: Batch of ``V`` vectors of same shape as the layer output
                (``[N, *, out_features]``) to which the transposed output-input Jacobian
                is applied. Has shape ``[V, N, *, out_features]`` if subsampling is not
                used, otherwise ``N`` must be ``len(subsampling)`` instead.
            sum_batch: Sum the result's batch axis. Default: ``True``.
            subsampling: Indices of samples along the output's batch dimension that
                should be considered. Defaults to ``None`` (use all samples).

        Returns:
            Batched transposed Jacobian vector products. Has shape
            ``[V, N, *module.weight.shape]`` when ``sum_batch`` is ``False``. With
            ``sum_batch=True``, has shape ``[V, *module.weight.shape]``. If sub-
            sampling is used, ``N`` must be ``len(subsampling)`` instead.
        """
        sin = subsample(module.input0, subsampling=subsampling) # (n...i)
        t = (sin - module.bias).matmul(exp(module.weight))  # (n...o)
        dtw = einsum("n...i, io->n...io", sin-module.bias, exp(module.weight)) 
        
        non_neg = ge(t, 0)
        le_n = le(t, module.n)
        dft = torch.logical_and(non_neg, le_n).float() # (n...o)
        
        dfw = einsum("n...io, n...o->n...io", dtw, dft)
        equation = f"vn...o,n...io->v{'' if sum_batch else 'n'}io"
        res = einsum(equation, mat, dfw)
        return res

    def _bias_jac_t_mat_prod(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: int = True,
        subsampling: List[int] = None,
    ) -> Tensor:
        """Batch-apply transposed Jacobian of the output w.r.t. the bias.

        Args:
            module: Linear layer.
            g_inp: Gradients w.r.t. module input. Not required by the implementation.
            g_out: Gradients w.r.t. module output. Not required by the implementation.
            mat: Batch of ``V`` vectors of same shape as the layer output
                (``[N, *, out_features]``) to which the transposed output-input Jacobian
                is applied. Has shape ``[V, N, *, out_features]``.
            sum_batch: Sum the result's batch axis. Default: ``True``.
            subsampling: Indices of samples along the output's batch dimension that
                should be considered. Defaults to ``None`` (use all samples).

        Returns:
            Batched transposed Jacobian vector products. Has shape
            ``[V, N, *module.bias.shape]`` when ``sum_batch`` is ``False``. With
            ``sum_batch=True``, has shape ``[V, *module.bias.shape]``. If sub-
            sampling is used, ``N`` is replaced by ``len(subsampling)``.
        """
        sin = subsample(module.input0, subsampling=subsampling) # (n...i)
        t = (sin - module.bias).matmul(exp(module.weight))  # (n...o)
        neg_dtb = einsum("io->i", exp(module.weight)) # i
        
        N = module.input0.shape[0]
        additional_dims = list(self._get_additional_dims(module))
        for _ in range(len(additional_dims) + 1):
            neg_dtb = neg_dtb.unsqueeze(0)
        expand = [N] + additional_dims + [-1]
        neg_dtb = neg_dtb.expand(*expand) # (n...i)
        
        non_neg = ge(t, 0)
        le_n = le(t, module.n)
        dft = torch.logical_and(non_neg, le_n).float() # (n...o)
        
        dfb = einsum("n...o, n...i->n...i", dft, -neg_dtb)
        equation = f"vn...o,n...i->v{'' if sum_batch else 'n'}i"
        res = einsum(equation, mat, dfb)
        
        return res

    @staticmethod
    def _get_additional_dims(module: Module) -> Size:
        """Return the shape of additional dimensions in the input to a linear layer.

        Args:
            module: A linear layer.

        Returns:
            Shape of the additional dimensions. Corresponds to ``*`` in the
            input shape ``[N, *, out_features]``.
        """
        return module.input0.shape[1:-1]
    
    