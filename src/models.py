from dataclasses import dataclass

from dataclasses import dataclass
from functools import cached_property
from torch import Tensor
import torch

@torch.jit.script
@dataclass
class GaussianMixture:
    """
    This class represents a Gaussian Mixture.

    weights: (K,)
    means: (K, dim)
    covariances: (K, dim, dim)
    """

    weights: Tensor  # shape: (K,)
    means: Tensor
    covariances: Tensor

    @cached_property
    def cov_colesky(self):
        return torch.linalg.cholesky(self.covariances)

    @property
    def mean(self):
        """Mean of the whole GM"""
        return (self.weights[:, None] * self.means).sum(dim=0)

    @property
    def count(self):
        """Number of gaussian components"""
        return len(self.means)

    @property
    def dim(self):
        """Dimension of the random vector"""
        return self.means.shape[1]
    
@dataclass
class PCDSamplingConfig:
    dim: int
    number_unit_vectors: int
    number_samples: int
    threshold: float = 0.1
    steps: int = 40
    sorting: bool = True
