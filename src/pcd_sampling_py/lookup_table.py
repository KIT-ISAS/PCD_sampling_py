import torch
from torch import Tensor
from torch.distributions import Distribution
from pcd_sampling_py.sampling_utils import pdf_cdf_dist

class LookupTable(torch.nn.Module):
    """
    Lookup-table approximation of a 1D distribution's PDF and CDF
    using linear interpolation.

    Parameters
    ----------
    dist : Distribution
        Distribution providing log_prob() and cdf().
    num_points : int
        Number of lookup-table points.
    xmin, xmax : float, optional
        Grid bounds. If omitted they are chosen as
        mean ± nsigma * stddev.
    nsigma : float
        Range used when bounds are computed automatically.
    """

    def __init__(
        self,
        dist: Distribution,
        num_points,
        xmin: Tensor = None,
        xmax: Tensor = None,
        nsigma=3.0,
    ):
        super().__init__()

        if xmin is None:
            xmin = dist.mean - nsigma * dist.stddev
        self.xmin = torch.atleast_1d(xmin)

        if xmax is None:
            xmax = dist.mean + nsigma * dist.stddev
        self.xmax = torch.atleast_1d(xmax)

        t = torch.linspace(0, 1, num_points, device=dist.mean.device, dtype=dist.mean.dtype)
        grid = self.xmin[:, None] + (self.xmax - self.xmin)[:, None] * t
        table = pdf_cdf_dist(dist, grid)
        self.register_buffer("table", table)
        
        self.num_points = int(num_points)
        self.inv_dx = (num_points - 1) / (self.xmax - self.xmin)


    def pdf_cdf(self, x: Tensor):
        """
        Returns (pdf, cdf).
        """

        x_clamped = x.clamp(self.xmin[:, None], self.xmax[:, None])
        pos = (x_clamped - self.xmin[:, None]) * self.inv_dx[:, None]

        idx0 = torch.floor(pos).long()
        idx1 = (idx0 + 1).clamp(max=self.num_points - 1)

        frac = pos - idx0.float()
        rows = torch.arange(self.table.shape[0], device=self.table.device)[:, None]
        
        return torch.lerp(self.table[rows, idx0], self.table[rows, idx1], frac[..., None])

@torch.compile
def pdf_cdf_lut(luts: LookupTable, R: Tensor):
    return luts.pdf_cdf(R)