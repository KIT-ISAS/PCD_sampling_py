import torch
from torch import Tensor
from torch.distributions import Distribution

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
        xmin=None,
        xmax=None,
        nsigma=6.0,
    ):
        super().__init__()

        if xmin is None:
            xmin = dist.mean - nsigma * dist.stddev
        self.xmin = float(xmin)

        if xmax is None:
            xmax = dist.mean + nsigma * dist.stddev
        self.xmax = float(xmax)

        grid = torch.linspace(xmin, xmax, num_points, device=dist.mean.device, dtype=dist.mean.dtype)
        table = torch.stack((torch.exp(dist.log_prob(grid)), dist.cdf(grid)), dim=-1)
        self.register_buffer("table", table)
        
        self.num_points = int(num_points)
        self.inv_dx = (num_points - 1) / (self.xmax - self.xmin)


    def pdf_cdf(self, x: Tensor):
        """
        Returns (pdf, cdf).
        """

        x_clamped = x.clamp(self.xmin, self.xmax)
        pos = (x_clamped - self.xmin) * self.inv_dx

        idx0 = torch.floor(pos).long()
        idx1 = (idx0 + 1).clamp(max=self.num_points - 1)

        frac = pos - idx0.float()

        return torch.lerp(self.table[idx0], self.table[idx1], frac[..., None])

