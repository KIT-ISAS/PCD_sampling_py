"""
Utils for computing the Cramer von Mises distance between the samples and the target distribution.
"""

import math
import torch


def sample_unit_vectors(K: int, D: int, device=None, dtype=None):
    u = torch.randn(K, D, device=device, dtype=dtype)
    u = u / torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(1e-12)
    return u


def gm_cdf_1d(x: torch.Tensor,
              weights: torch.Tensor,
              means: torch.Tensor,
              stds: torch.Tensor) -> torch.Tensor:
    """
    1D Gaussian mixture CDF.

    x:       (...,) evaluation points
    weights: (M,)
    means:   (M,)
    stds:    (M,)

    returns: (...,)
    """
    z = (x[..., None] - means) / stds.clamp_min(1e-8)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return torch.sum(weights * Phi, dim=-1)


def calculate_distance_on_interval(
    r1: torch.Tensor,
    r2: torch.Tensor,
    j: int,
    L: int,
    projected_means: torch.Tensor,
    projected_stds: torch.Tensor,
    weights: torch.Tensor,
    num_points: int = 1000,
):
    """
    Numerically computes
        ∫_{r1}^{r2} (F_tilda(r) - j/L)^2 dr

    where F_tilda is the projected Gaussian mixture CDF, and j/L is the
    constant Dirac-mixture CDF level on this interval.

    Parameters
    ----------
    r1, r2 : scalar tensors or floats
        Interval boundaries.
    j : int
        Step index of the empirical/Dirac CDF on [r1, r2].
        For sorted samples:
            interval (-inf, r_(1))  -> j = 0
            interval [r_(1), r_(2)) -> j = 1
            ...
            interval [r_(L), +inf)  -> j = L
    L : int
        Number of Dirac samples.
    projected_means : (M,)
        Means of projected Gaussian mixture.
    projected_stds : (M,)
        Standard deviations of projected Gaussian mixture.
    weights : (M,)
        Gaussian mixture weights.
    num_points : int
        Number of grid points for trapezoidal integration.

    Returns
    -------
    distance : scalar tensor
    """
    #commented out because cannot be used in vmap
    # # Avoid pathological empty intervals 
    # if torch.as_tensor(r2) <= torch.as_tensor(r1):
    #     return torch.zeros((), device=projected_means.device, dtype=projected_means.dtype)

    t = torch.linspace(0.0, 1.0, num_points, device=projected_means.device, dtype=projected_means.dtype)
    x = r1 + (r2 - r1) * t

    F_tilda = gm_cdf_1d(x, weights, projected_means, projected_stds)

    c = torch.as_tensor(j / L, device=x.device, dtype=x.dtype)

    # Full squared difference:
    # (F - c)^2 = F^2 - 2cF + c^2
    integrand = (F_tilda - c) ** 2

    return torch.trapz(integrand, x)


def _calculate_projections(unit_vectors, means, covariances):
    """
    Projects a Gaussian mixture onto many unit vectors.

    unit_vectors: (K, D)
    means:        (M, D)
    covariances:  (M, D, D)

    Returns
    -------
    projected_means: (K, M)
    projected_stds:  (K, M)
    """
    projected_means = unit_vectors @ means.T  # (K, M)
    sigma2 = torch.einsum(
        "kd,mde,ke->km", unit_vectors, covariances, unit_vectors
    )
    sigma2 = torch.clamp(sigma2, min=1e-12)  # Avoid negative variances due to numerical issues
    projected_stds = torch.sqrt(sigma2)

    return projected_means, projected_stds

def calculate_for_one_projection(mu, std, s, weights, L, num_interval_points, tail_sigma_mult, tail_margin):
    device = mu.device
    dtype = mu.dtype
    s_sorted, _ = torch.sort(s)

    # Truncation bounds for the tails
    gm_left = torch.min(mu - tail_sigma_mult * std)
    gm_right = torch.max(mu + tail_sigma_mult * std)
    dirac_left = s_sorted[0]
    dirac_right = s_sorted[-1]

    r_min = torch.minimum(gm_left, dirac_left) - tail_margin
    r_max = torch.maximum(gm_right, dirac_right) + tail_margin

    D1 = torch.zeros((), device=device, dtype=dtype)

    # Left tail: (-inf, s_1) approximated by [r_min, s_1], step level 0
    D1 = D1 + calculate_distance_on_interval(
        r_min, s_sorted[0], 0, L, mu, std, weights,
        num_points=num_interval_points
    )

    # Middle intervals: [s_j, s_{j+1}], step level j/L
    for j in range(1, L):
        D1 = D1 + calculate_distance_on_interval(
            s_sorted[j - 1], s_sorted[j], j, L, mu, std, weights,
            num_points=num_interval_points
        )

    # Right tail: [s_L, +inf) approximated by [s_L, r_max], step level 1
    D1 = D1 + calculate_distance_on_interval(
        s_sorted[-1], r_max, L, L, mu, std, weights,
        num_points=num_interval_points
    )
    
    del s_sorted, gm_left, gm_right, dirac_left, dirac_right, r_min, r_max # Cleanup intermediate tensors to save memory

    return D1
    

def calculate_cvm(
    samples: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
    K: int = 256,
    num_interval_points: int = 128,
    tail_sigma_mult: float = 6.0,
    tail_margin: float = 1.0,
    unit_vectors: torch.Tensor | None = None,
):
    """
    Numerically estimates the multivariate CvM distance by Monte Carlo over
    projection directions.

    Parameters
    ----------
    samples : (L, D)
        Dirac sample locations.
    means : (M, D)
        Gaussian mixture means.
    covariances : (M, D, D)
        Gaussian mixture covariances.
    weights : (M,)
        Gaussian mixture weights (should sum to 1).
    K : int
        Number of random unit vectors.
    num_interval_points : int
        Number of quadrature points per interval.
    tail_sigma_mult : float
        How many stds to include for truncating the infinite tails.
    tail_margin : float
        Extra margin on both sides.

    Returns
    -------
    cvm : scalar tensor
        Estimated CvM distance.
    """
   

    samples = samples.double()
    means = means.double()
    covariances = covariances.double()
    weights = weights.double()
    device = samples.device
    dtype = samples.dtype
    L, D = samples.shape
    M = means.shape[0]

    # Normalize weights just in case
    weights = weights / weights.sum()

    # Project GM
    projected_means, projected_stds = _calculate_projections(unit_vectors, means, covariances)  # (K,M), (K,M,M)

    # Project Dirac samples
    projected_samples = unit_vectors @ samples.T  # (K, L)

    # create a vmap of calculate_for_one_projection over the K projections
    vmap_func = torch.vmap(calculate_for_one_projection, in_dims=(0, 0, 0, None, None, None, None, None,))
    D1_values = vmap_func(
        projected_means, projected_stds, projected_samples,
        weights, L, num_interval_points, tail_sigma_mult, tail_margin
    )

    return D1_values.mean()
