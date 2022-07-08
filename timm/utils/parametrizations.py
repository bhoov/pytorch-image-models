import torch
import numpy as np
from torch import nn
import torch.nn.utils.parametrize as parametrize
from typing import *

def _is_orthogonal(Q, eps=None):
    n, k = Q.size(-2), Q.size(-1)
    Id = torch.eye(k, dtype=Q.dtype, device=Q.device)
    # A reasonable eps, but not too large
    eps = 10. * n * torch.finfo(Q.dtype).eps
    return torch.allclose(Q.T @ Q, Id, atol=eps)


def _make_orthogonal(A):
    """ Assume that A is a tall matrix.
    Compute the Q factor s.t. A = QR  and diag(R) is real and non-negative
    """
    X, tau = torch.geqrf(A)
    Q = torch.linalg.householder_product(X, tau)
    # The diagonal of X is the diagonal of R (which is always real) so we normalise by its signs
    Q *= X.diagonal(dim1=-2, dim2=-1).sgn().unsqueeze(-2)
    return Q

class OrthogonalParametrization(nn.Module):
    def forward(self, X):
        # print("MAKING ORTHO")
        return _make_orthogonal(X)

class Orthogonal(nn.Module):
    """Shows how to wrap a linear module with an orthogonal parameterization."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)
        parametrize.register_parametrization(self.linear, "weight", OrthogonalParametrization())
        
    def forward(self, x):
        return self.linear(x)
    
class ScaledOrthogonalParametrization(nn.Module):
    def __init__(self, N:int, Lclip_min=None, Lclip_max=None):
        """A simple parameterization of an orthogonal matrix, that has learnable eigenvalues that can be clipped. Only works on square matrices of shape `NxN`.
        
        If both Lclip_min and Lclip_max are given, initialize L with a normal distribution between the given limits.
        """
        super().__init__()
        
        # If both Lclip_min and Lclip_max are given
        self.Lclip_min = Lclip_min
        self.Lclip_max = Lclip_max
        self.has_clipping = Lclip_min is not None or Lclip_max is not None
        both_limits = Lclip_min is not None and Lclip_max is not None
        diff_range = Lclip_max - Lclip_min if both_limits else None
        init_mean = 0.5*(Lclip_max - Lclip_min) + Lclip_min if both_limits else 1
        
        init_std = diff_range / 6 if both_limits else 0.25
        
        if self.has_clipping:
            init_mean = np.clip(init_mean, Lclip_min, Lclip_max)
        self.L = nn.Parameter(torch.empty(N))
        nn.init.normal_(self.L, init_mean, init_std)

    @classmethod
    def from_weight(cls, weight, Lclip_min=None, Lclip_max=None):
        """Initialize from the shape of the weight"""
        
        assert len(weight.shape) == 2, "Only works with 2D square matrices"
        m,n = weight.shape
        assert m == n, "Only works with 2D square matrices"
        return cls(m, Lclip_min, Lclip_max)
        
    def forward(self, B):
        A = (0.5 * (B - B.T))
        Q = torch.matrix_exp(A)
        if self.has_clipping:
            L = self.L.clip(self.Lclip_min, self.Lclip_max)
        else:
            L = self.L
            
        M = torch.einsum("ml,l,ln->mn", Q, L, Q.T)
        return M
    
class ScaledOrthogonal(nn.Module):
    """Example of how to use the Scaled Orthogonal parameterization"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)
        parametrize.register_parametrization(self.linear, "weight", OrthogonalParametrization.from_weight(self.linear.weight, 0.3, 1.))
        
    def forward(self, x):
        return self.linear(x)