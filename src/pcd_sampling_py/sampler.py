from typing import Callable
import torch
from torch import Tensor
from pcd_sampling_py.models import PCDSamplerConfig
from pcd_sampling_py.sampling_utils import (
    gm_cdf_1d,
    gm_pdf_1d,
    heaviside_mean,
    reduce_gm,
    sample_gm,
    sample_gm_cached_cholesky,
    sample_ut,
    sot_sphere,
)


class PCDSampler:
    """
    Deterministic sampling using Projected Cumulative Distribution.

    L - number of samples
    N - dimension
    K - number of unit vectors
    T - number of components in a GM
    """

    def __init__(self, config: PCDSamplerConfig):
        self.number_samples = config.number_samples
        self.number_unit_vectors = config.number_unit_vectors
        self.threshold = config.threshold
        self.dim = config.dim
        self.steps = config.steps
        self.sorting = config.sorting
        self.initial_sampling_method = config.initial_sampling_method
        self.unit_vectors_method = config.unit_vectors_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.initial_sampling_method == "ut":
            assert (
                self.number_samples == self.dim * 2 + 1
            ), "UT sampling requires 2 * N + 1 samples."

        self.alpha_max = 2
        self.alpha_min = 0.01

        # Batch functions for calculating gain.
        self._delta_r_vmap = torch.func.vmap(
            self.calculate_delta_r, in_dims=(0, None, None, None, None)
        )
        self._delta_x_vmap = torch.func.vmap(
            self.calculate_delta_x, in_dims=(0, 0, 0, 0, None)
        )
        self._delta_r_sorted_vmap = torch.func.vmap(
            self.calculate_delta_r_sorted, in_dims=(0, 0, None, None, None)
        )
        self._delta_x_sorted_vmap = torch.func.vmap(
            self.calculate_delta_x_sorted, in_dims=(0, 0, 0, 0, None)
        )

        # If sorting of the projections is enabled use the correct impelementation
        if self.sorting:
            self.compute_delta_x: Callable = self._delta_x_sorted_vmap
        else:
            self.compute_delta_x: Callable = self._delta_x_vmap

        self.numbers = torch.arange(0, self.number_samples)

        # Pre-allocate unit vectors, so that we don't calculate them in every step.
        # This can also be done deterministicaly with vectors uniformally covering the unit sphere. #TODO: later
        
        if self.unit_vectors_method == "random":
            self.create_unit_vectors_random()
        elif self.unit_vectors_method == "deterministic":
            self.create_unit_vectors_deterministic()

    def create_unit_vectors_random(self):
        """
        Randomly creates unit vectors to project the distribution onto.
        """

        self.unit_vectors = torch.distributions.uniform.Uniform(-1.0, 1.0).sample(
            (self.number_unit_vectors, self.dim)
        )
        self.unit_vectors = self.unit_vectors / self.unit_vectors.norm(
            dim=1, keepdim=True
        )

    def create_unit_vectors_deterministic(self):
        """
        Creates unit vectors to project the distribution onto with a deterministic method, so that they cover the unit sphere uniformly.
        """

        self.unit_vectors = sot_sphere(
            self.number_unit_vectors, d=self.dim, K=64, iterations=300, device=self.device)

    @torch.compile
    def calculate_delta_r(
        self, x: Tensor, r: Tensor, weights: Tensor, means: Tensor, stds: Tensor
    ):
        """
        This function is used to compute the gain between real and approximation projected distributions

        :param x: scalar
        :param r: (L, ) all projections of samples onto the unit vector.
        :param weights: (T, ) weigths of the components of the projections of the real GM
        :param means: (T, )
        :param stds: (T, T)

        L - number of sampoles, T - number of components in GM
        """
        F_tilda = gm_cdf_1d(x, weights, means, stds)
        f_tilda = gm_pdf_1d(x, weights, means, stds)

        F = heaviside_mean(x, r)

        eps = 1e-3  # to avoid division by zero.
        return -((F_tilda - F) / torch.clamp(f_tilda, min=eps))

    @torch.compile
    def calculate_delta_r_sorted(
        self, x: Tensor, i: Tensor, weights: Tensor, means: Tensor, stds: Tensor
    ):
        """
        This function is used to compute the gain between real and approximation projected distributions for sorted projections.

        :param x: scalar
        :param i: index of the sample in the sorted projection
        :param weights: (T, ) weigths of the components of the projections of the real GM
        :param means: (T, )
        :param stds: (T, T)

        L - number of sampoles, T - number of components in GM
        """
        F_tilda = gm_cdf_1d(x, weights, means, stds)
        f_tilda = gm_pdf_1d(x, weights, means, stds)

        eps = 1e-3  # to avoid division by zero.
        return -(
            (F_tilda - (2 * i - 1) / 2 / self.number_samples)
            / torch.clamp(f_tilda, min=eps)
        )

    @torch.compile
    def calculate_delta_x(
        self, r: Tensor, u: Tensor, means: Tensor, stds: Tensor, weights: Tensor
    ):
        """
        This function is used to compute the gain for all samples with respect to the provided unit vector.

        :param r: projections of all samples onto the provided unit vector
        :param u: unit vector
        :param means: projected means of the original GM.
        :param stds: projection stds of the original GM.
        :param weights: weights of the original GM.
        """
        delta_r = self._delta_r_vmap(r, r, weights, means, stds)
        delta_x = (u[None, :] * delta_r[:, None]) / self.number_unit_vectors

        return delta_x

    @torch.compile
    def calculate_delta_x_sorted(
        self, r: Tensor, u: Tensor, means: Tensor, stds: Tensor, weights: Tensor
    ):
        """
        This function is used to compute the gain for all samples with respect to the provided unit vector.
        It first sorts the projections for the unit vector

        :param r: projections of all samples onto the provided unit vector
        :param u: unit vector
        :param means: projected means of the original GM.
        :param stds: projection stds of the original GM.
        :param weights: weights of the original GM.
        """

        sorted_r, idx = torch.sort(r)
        delta_r_sorted = self._delta_r_sorted_vmap(
            sorted_r, self.numbers, weights, means, stds
        )

        # Returning to the original order
        delta_r_original = delta_r_sorted[torch.argsort(idx)]
        delta_x = (u[None, :] * delta_r_original[:, None]) / self.number_unit_vectors
        return delta_x

    def _calculate_gain(self, step, X, projected_means, projected_stds, weights):
        coef = (
            self.alpha_min * step / self.steps
            + self.alpha_max * (self.steps - step) / self.steps
        )

        # Calculate projections onto unit vectors before hand
        R = self.unit_vectors @ X.T  # -> (K, L)

        # Calculate the gain for the samples based on the differences between projections of real and approximate distributions
        delta_x: Tensor = self.compute_delta_x(
            R, self.unit_vectors, projected_means, projected_stds, weights
        ).sum(
            dim=0
        )  # -> (L, N)

        return coef, delta_x

    def _calculate_projections(self, means, covariances):
        projected_means = self.unit_vectors @ means.T  # -> (K, L)

        sigma2 = torch.einsum(
            "kd,lde,ke->kl", self.unit_vectors, covariances, self.unit_vectors
        )
        sigma2 = torch.clamp(sigma2, min=1e-3)
        projected_stds = torch.sqrt(sigma2)

        return projected_means, projected_stds

    def _initial_samples(self, weights: Tensor, means: Tensor, covariances: Tensor):
        if self.initial_sampling_method == "mean":
            # We just iterate through means and get samples.
            K = means.shape[0]
            N = self.number_samples

            idx = torch.arange(N, device=means.device) % K  # (N,)
            samples = means[idx]  # (N, d)
            return samples

        elif self.initial_sampling_method == "ut":
            # We use the unscented transform to get the initial samples.
            reduced_mean, reduced_cov = reduce_gm(weights, means, covariances)
            samples = sample_ut(reduced_mean, reduced_cov)
            return samples

        return sample_gm(weights, means, covariances, self.number_samples)

    @torch.compile
    def sample(self, weights: Tensor, means: Tensor, covariances: Tensor):
        """
        Sample from the Gaussian Mixture. Returns a (L, N) tensor of samples.
        Stop when the final steps is reachend. Can be used in a vmap.

        :param weights: (T,)
        :type weights: Tensor
        :param means: (T, N)
        :type means: Tensor
        :param covariances: (T, N, N)
        :type covariances: Tensor
        """

        # 1. Create projections of the original GM onto the unit vectors.
        projected_means, projected_stds = self._calculate_projections(
            means, covariances
        )

        # 2. Create some random starting samples from the provided GM
        X = self._initial_samples(weights, means, covariances)

        # 3. Start the minimization loop
        for _ in range(self.steps):

            coef, delta_x = self._calculate_gain(
                _, X, projected_means, projected_stds, weights
            )
            X += coef * delta_x
        return X

    @torch.compile
    def sample_threshold(self, weights: Tensor, means: Tensor, covariances: Tensor):
        """
        Sample from the Gaussian Mixture. Returns a (L, N) tensor of samples. Stop when threshold is reached
        ATTENTION: Cannot be used in vmap.

        :param weights: (T,)
        :type weights: Tensor
        :param means: (T, N)
        :type means: Tensor
        :param covariances: (T, N, N)
        :type covariances: Tensor
        """

        # 1. Create projections of the original GM onto the unit vectors.
        projected_means, projected_stds = self._calculate_projections(
            means, covariances
        )

        # 2. Create some random starting samples from the provided GM
        X = self._initial_samples(weights, means, covariances)

        # 3. Start the minimization loop
        for _ in range(self.steps):

            coef, delta_x = self._calculate_gain(
                _, X, projected_means, projected_stds, weights
            )
            X += coef * delta_x

            if torch.linalg.vector_norm(delta_x, dim=1).mean() < self.threshold:
                return X

        return X

    def benchmark_steps(self, weights: Tensor, means: Tensor, covariances: Tensor):
        # 1. Create projections of the original GM onto the unit vectors.
        projected_means, projected_stds = self._calculate_projections(
            means, covariances
        )

        # 2. Create some random starting samples from the provided GM
        X = self._initial_samples(weights, means, covariances)

        # Create array of norms
        norms = torch.empty((self.steps))

        # 3. Start the minimization loop
        for _ in range(self.steps):

            coef, delta_x = self._calculate_gain(
                _, X, projected_means, projected_stds, weights
            )
            X += coef * delta_x
            norms[_] = torch.linalg.vector_norm(delta_x, dim=1).mean()

        return norms
