import torch
from torch import Tensor
from pcd_sampling_py.sampling_utils import (
    gm_cdf_1d,
    gm_pdf_1d)

class LookupTable():
    """
    Lookup-table approximation of a 1D Gaussian Mixture Model PDF/CDF
    using linear interpolation.

    Parameters
    ----------
    weights : Tensor, shape (K,)
        Mixture weights (should sum to 1).
    means : Tensor, shape (K,)
        Component means.
    stds : Tensor, shape (K,)
        Component standard deviations.
    num_points : int
        Number of lookup-table points.
    xmin : float, optional
        Lower grid bound. If None, computed automatically.
    xmax : float, optional
        Upper grid bound. If None, computed automatically.
    nsigma : float
        Range for automatic bounds:
        [min(mu - nsigma*sigma), max(mu + nsigma*sigma)].
    """

    def __init__(
        self,
        weights: Tensor,
        means: Tensor,
        stds: Tensor,
        num_points,
        xmin=None,
        xmax=None,
        nsigma=6.0,
    ):
        super().__init__()

        weights = weights / weights.sum()

        if xmin is None:
            xmin = torch.min(means - nsigma * stds).item()

        if xmax is None:
            xmax = torch.max(means + nsigma * stds).item()

        grid = torch.linspace(xmin, xmax, num_points)

#TODO: is this correct? Work with distributions instead of weights, means, std
        self.pdf = gm_pdf_1d(grid, weights, means, stds)
        self.cdf = gm_cdf_1d(grid, weights, means, stds)
    
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.num_points = int(num_points)
        self.dx = (self.xmax - self.xmin) / (self.num_points - 1)


    def pdf_cdf(self, x: Tensor):
        """
        Returns (pdf, cdf).
        """

        x_clamped = x.clamp(self.xmin, self.xmax)

        pos = (x_clamped - self.xmin) / self.dx

        idx0 = torch.floor(pos).long()
        idx1 = (idx0 + 1).clamp(max=self.num_points - 1)

        frac = pos - idx0.float()

        return (self.__lin_int(self.pdf[idx0], self.pdf[idx1], frac), self.__lin_int(self.cdf[idx0], self.cdf[idx1], frac))

    @staticmethod
    def __lin_int(y0, y1, frac):
        return y0 + frac * (y1-y0)

