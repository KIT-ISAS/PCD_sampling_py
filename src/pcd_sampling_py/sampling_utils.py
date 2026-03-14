import torch
from torch import Tensor


@torch.compile
def sample_gm_cached_cholesky(
    weights: Tensor,
    means: Tensor,
    covs_cholesky: Tensor,
    number_samples: int,
):
    """
    This function is used to sample from Gaussian Mixture without calculating cholesky decomposition (minor optimization)

    Maybe for higher dimensional Covariances it makes sense (try it out)

    :param weights: Weights of the components (T,)
    :param means: Means of the components (T, N)
    :param covs_cholesky: Precomputed Cholesky decomposition of the covariances (T, N, N)
    :param number_samples: Number of samples L (int)
    """

    # 1. Sample component indices
    z = torch.multinomial(weights, number_samples, replacement=True)  # (L,)

    # 2. Gather parameters
    means = means[z]  # (L, N)

    # 3. Cholesky (batched)
    L = covs_cholesky[z]

    # 4. Sample standard normals
    eps = torch.randn(number_samples, means.shape[1], device=means.device)

    # 5. Affine transform
    samples = means + torch.einsum("nij,nj->ni", L, eps)  # (L, N)

    return samples


@torch.compile
def sample_gm(
    weights: Tensor,
    means: Tensor,
    covs: Tensor,
    number_samples: int,
):
    """
    This function is used to sample from Gaussian Mixture.

    :param weights: Weights of the components (T,)
    :param means: Means of the components (T, N)
    :param covs: Covs of the components (T, N, N)
    :param number_samples: Number of samples L (int)
    """
    cholesky = torch.linalg.cholesky(covs)
    return sample_gm_cached_cholesky(
        weights, means, cholesky, number_samples=number_samples
    )


def gm_pdf_1d(x: Tensor, weights: Tensor, means: Tensor, stds: Tensor):
    """
    This function returns a value of a CDF of a 1D Gaussian Mixture
    """
    mixture = torch.distributions.Categorical(probs=weights)
    components = torch.distributions.Normal(means, stds)
    gm = torch.distributions.MixtureSameFamily(mixture, components)
    return torch.exp(gm.log_prob(x))


def gm_cdf_1d(x: Tensor, weights: Tensor, means: Tensor, stds: Tensor):
    """
    This function returns CDF ( P(y <= x) ) of the provided 1D Gaussian Mixture

    :param x: scalar (0-dim tensor)
    :type x: Tensor
    :param weights: Weights of the GM components (T,)
    :type weights: Tensor
    :param means: Means of the GM components (T,)
    :type means: Tensor
    :param stds: Stds of the GM components (T, )
    :type stds: Tensor
    """
    mixture = torch.distributions.Categorical(probs=weights)
    components = torch.distributions.Normal(means, stds)
    gm = torch.distributions.MixtureSameFamily(mixture, components)
    return gm.cdf(x)


def heaviside_mean(x: Tensor, r: Tensor):
    """
    Computes mean of heavisides for scalar x and vector r by comparing x with every element

    :param x: scalar that has to be compared to every element in vector r.
    :param r: vector
    """
    return ((r < x).float().sum() + 0.5 * (r == x).float().sum()) / r.numel()


def reduce_gm(weights: Tensor, means: Tensor, covariances: Tensor):
    """
    This function reduces a Gaussian Mixture to a single Gaussian by matching the first two moments.

    :param weights: Weights of the components (T,)
    :param means: Means of the components (T, N)
    :param covariances: Covs of the components (T, N, N)

    :return: mean and covariance of the reduced Gaussian
    """
    mixture_mean = torch.sum(weights[:, None] * means, dim=0)
    centered = means - mixture_mean
    mixture_cov = torch.sum(
        weights[:, None, None]
        * (covariances + torch.einsum("bi,bj->bij", centered, centered)),
        dim=0,
    )
    return mixture_mean, mixture_cov


def sample_ut(mean: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    """
    Sample from a Gaussian using the Unscented Transform (UT).

    :param mean: Mean of the Gaussian with shape (n,) or (batch_size, n).
    :param covariance: Covariance matrix of the Gaussian with shape (n, n) or (batch_size, n, n).
    :return: Sigma points sampled from the Gaussian with shape (2n+1, n) or (batch_size, 2n+1, n).
    """
    # Ensure mean has shape (..., n)
    alpha = 0.99
    beta = 2.0
    kappa = 0.0
    # First reduce to a single gaussian then sample

    if mean.ndim == 1:
        mean_ = mean.unsqueeze(0)  # (1, n)
        cov_ = covariance.unsqueeze(0)  # (1, n, n)
        squeeze_batch = True
    else:
        mean_ = mean
        cov_ = covariance
        squeeze_batch = False

    n = mean_.shape[-1]
    lambda_ = alpha**2 * (n + kappa) - n

    # Cholesky of scaled covariance, supports batching
    # L: (..., n, n)
    L = torch.linalg.cholesky((n + lambda_) * cov_)

    # UT uses columns of sqrt matrix as deviations.
    # Make them rows by transposing: cols -> rows
    # rows: (..., n, n) where rows[..., i, :] = column_i(L)^T
    rows = L.transpose(-1, -2)

    zeros = torch.zeros(
        (*mean_.shape[:-1], 1, n), device=mean_.device, dtype=mean_.dtype
    )
    plus = rows
    minus = -rows

    # deviations: (..., 2n+1, n)
    deviations = torch.cat([zeros, plus, minus], dim=-2)

    # sigma_points: (..., 2n+1, n)
    sigma_points = mean_.unsqueeze(-2) + deviations

    if squeeze_batch:
        sigma_points = sigma_points.squeeze(0)  # back to (2n+1, n)

    return sigma_points


def sample_sot_unit_vectors(number_unit_vectors: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Sample unit vectors uniformly on the unit sphere using Spherical Optimal Transport (SOT).

    :param number_unit_vectors: Number of unit vectors to sample.
    :param dim: Dimensionality of the space.
    :param device: Device to create the tensor on.
    :return: Tensor of shape (number_unit_vectors, dim) containing unit vectors.
    """
    # Sample from a standard normal distribution
    random_vectors = torch.randn(number_unit_vectors, dim, device=device)

    # Normalize to lie on the unit sphere
    unit_vectors = random_vectors / random_vectors.norm(dim=1, keepdim=True)

    return unit_vectors

def sample_random_sphere(n: int, d: int = 3, device: str = "cpu") -> torch.Tensor:
    """Random uniform points on S^{d-1}."""
    pts = torch.randn(n, d, device=device)
    return pts / pts.norm(dim=1, keepdim=True)

def sot_step(points: torch.Tensor, K: int) -> torch.Tensor:
    """
    One SOT iteration on the sphere.
 
    For each of K random directions:
      1. Project all points onto that direction (dot product)
      2. Sort projections (= 1D optimal transport)
      3. Generate target positions by sampling from the projected
         density (Gaussian -> normalize -> project = same 1D marginal)
         and sorting those too
      4. Nudge each point toward its matched target along that direction
 
    Instead of an inverse-CDF table, we just draw n fresh samples from
    the *same* distribution (uniform on sphere), project them onto theta,
    and sort. The sorted projections of a large uniform sample converge
    to the quantiles of the projected density -- this IS the inverse CDF,
    computed by sampling.
    """
    n, d = points.shape
    displacements = torch.zeros_like(points)
 
    for _ in range(K):
        # Random slice direction
        theta = torch.randn(d, device=points.device)
        theta = theta / theta.norm()
 
        # Project current points onto theta
        proj = points @ theta                          # (n,)
        order = torch.argsort(proj)
 
        # Target: project n fresh uniform-sphere samples onto theta, sort them
        # This gives us the empirical quantiles of the 1D projected density
        target_pts = torch.randn(n, d, device=points.device)
        target_pts = target_pts / target_pts.norm(dim=1, keepdim=True)
        target_proj = (target_pts @ theta).sort().values
 
        # Displacement along theta
        d_ij = target_proj - proj[order]               # (n,)
        displacements[order] += d_ij.unsqueeze(1) * theta.unsqueeze(0) / K
 
    # Apply and re-project onto sphere
    points_new = points + displacements
    return points_new / points_new.norm(dim=1, keepdim=True)
 
 
def sot_sphere(
    n_points: int,
    d: int = 3,
    K: int = 64,
    iterations: int = 300,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate well-distributed points on S^{d-1} via SOT."""
    points = sample_random_sphere(n_points, d, device)
    for i in range(iterations):
        points = sot_step(points, K)
    return points