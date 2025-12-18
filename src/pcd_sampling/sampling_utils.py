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
