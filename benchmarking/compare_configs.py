"""
This script compares different initialization configs for PCD such as: initial sampling method, unit vector method etc.
It uses Cramer von Mises distance between the samples and the target distribution as a metric to compare the convergence of different configs.
It compares CvM distance of the config with the best possible PCD for a fixed number of samples (A large number of steps and unit vectors).
"""

from sympy import plot
import torch

from benchmarking.cvm import calculate_cvm, sample_unit_vectors
from pcd_sampling_py.models import PCDSamplerConfig
from pcd_sampling_py.sampler import PCDSampler
from pcd_sampling_py.sampling_utils import sot_sphere

UNIT_VECTORS = 50
STEPS = 30
SEEDS = 70
SAMPLES = 5


def plot_convergence(all_distances: list, gt: float):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 20})

    all_distances = np.array(
        all_distances
    )  # Shape: (num_configs, num_seeds, num_steps)
    mean_distances = np.mean(all_distances, axis=1)  # Shape: (num_configs, num_steps)
    medians = np.median(all_distances, axis=1)  # Shape: (num_configs, num_steps)
    iqrs = np.percentile(
        all_distances, [25, 75], axis=1
    )  # Shape: (2, num_configs, num_steps)

    config_labels = [
        "Random Init + Random UV",
        # "Mean Init + Random UV",
        # "Random Init + Deterministic UV",
        "Mean Init + Deterministic UV",
    ]

    for i in range(mean_distances.shape[0]):
        # plt.plot(mean_distances[i], label=config_labels[i])
        plt.plot(medians[i], label=f"{config_labels[i]} Median", linestyle="--", linewidth=2)
        # plt.fill_between(
        #     range(STEPS), medians[i] - iqrs[0, i], medians[i] + iqrs[1, i], alpha=0.2
        # )

    plt.axhline(gt, color="red", linestyle="--", label="Ground truth CvM distance")
    plt.xlabel("Steps")
    plt.ylabel("CvM distance")
    plt.title("Convergence of PCD with different configs")
    plt.legend()
    plt.grid()

    # save the plot to a file instead of showing it
    plt.savefig("pcd_convergence_comparison.pdf")

    plt.show()


# def plot_convergence(distances: list, gt: float):
#     import matplotlib.pyplot as plt

#     plt.plot(distances, label="Config CvM distance")
#     plt.axhline(gt, color="red", linestyle="--", label="Ground truth CvM distance")
#     plt.xlabel("Steps")
#     plt.ylabel("CvM distance")
#     plt.title("Convergence of PCD with different configs")
#     plt.legend()
#     plt.grid()
#     plt.show()

# Pre calculate deterministic Unit Vecotors for faster computation:

deterministic_unit_vectors = sot_sphere(
    UNIT_VECTORS,
    d=2,
    K=64,
    iterations=300,
    device=torch.device("cuda"),
)


def calc_cvms_for_seed(
    seed: int,
    cvm_unit_vectors: torch.Tensor,
    weights: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    initial_sampling_method: str,
    unit_vectors_method: str,
):
    torch.manual_seed(seed)

    distances = []
    for i in range(STEPS):

        sampling_config = PCDSamplerConfig(
            number_samples=SAMPLES,
            dim=2,
            number_unit_vectors=UNIT_VECTORS,
            steps=i,
            sorting=True,
            initial_sampling_method=initial_sampling_method,
            unit_vectors_method="random",
        )

        sampler = PCDSampler(sampling_config)

        # Override, because there is no way to initialize.
        if unit_vectors_method == "deterministic":
            sampler.unit_vectors = deterministic_unit_vectors

        samples = sampler.sample(weights, means, covariances)
        cvm = calculate_cvm(
            samples, means, covariances, weights, unit_vectors=cvm_unit_vectors
        )
        print(f"Config CvM distance: {cvm.item()}")
        distances.append(cvm.item())
    return distances


def compare_configs():
    weights = torch.tensor([0.5, 0.5])
    means = torch.tensor([[2.0, 0.0], [-2.0, 0.0]])
    covariances = torch.tensor([[[3.0, 2.8], [2.8, 3.0]], [[3.0, -2.8], [-2.8, 3.0]]])

    cvm_unit_vectors = sample_unit_vectors(
        1000, 2, device=means.device, dtype=means.dtype
    ).double()

    sampling_config = PCDSamplerConfig(
        number_samples=SAMPLES,
        dim=2,
        number_unit_vectors=10000,
        steps=300,
        # threshold=1e-6,
        sorting=True,
        initial_sampling_method="mean",
        unit_vectors_method="deterministic",
    )
    sampler = PCDSampler(sampling_config)

    # Ground truth samples with a large number of steps and unit vectors to approximate the best possible PCD samples.
    gt_samples = sampler.sample(weights, means, covariances)

    print(f"Samples shape: {gt_samples.shape}")

    gt_cvm = calculate_cvm(
        gt_samples, means, covariances, weights, unit_vectors=cvm_unit_vectors
    )
    print(f"Ground truth CvM distance: {gt_cvm.item()}")

    # Random initial samples + random unit vectors for each config:
    rand_init_rand_uv = []
    mean_init_rand_uv = []
    rand_init_det_uv = []
    mean_init_det_uv = []

    for i in range(SEEDS):
        print(f"Running seed {i+1}/{SEEDS}")
        
        print("Config: Random Init + Random UV")
        distances = calc_cvms_for_seed(
            seed=i,
            cvm_unit_vectors=cvm_unit_vectors,
            weights=weights,
            means=means,
            covariances=covariances,
            initial_sampling_method="random",
            unit_vectors_method="random",
        )
        rand_init_rand_uv.append(distances)

        # print("Config: Mean Init + Random UV")
        # distances = calc_cvms_for_seed(
        #     seed=i,
        #     cvm_unit_vectors=cvm_unit_vectors,
        #     weights=weights,
        #     means=means,
        #     covariances=covariances,
        #     initial_sampling_method="mean",
        #     unit_vectors_method="random",
        # )
        # mean_init_rand_uv.append(distances)

        # print("Config: Random Init + Deterministic UV")
        # distances = calc_cvms_for_seed(
        #     seed=i,
        #     cvm_unit_vectors=cvm_unit_vectors,
        #     weights=weights,
        #     means=means,
        #     covariances=covariances,
        #     initial_sampling_method="random",
        #     unit_vectors_method="deterministic",
        # )
        # rand_init_det_uv.append(distances)

        print("Config: Mean Init + Deterministic UV")
        distances = calc_cvms_for_seed(
            seed=i,
            cvm_unit_vectors=cvm_unit_vectors,
            weights=weights,
            means=means,
            covariances=covariances,
            initial_sampling_method="mean",
            unit_vectors_method="deterministic",
        )
        mean_init_det_uv.append(distances)

    plot_convergence(
        # [rand_init_rand_uv, mean_init_rand_uv, rand_init_det_uv, mean_init_det_uv],
        [rand_init_rand_uv, mean_init_det_uv],
        gt_cvm.item(),
    )


if __name__ == "__main__":
    torch.set_default_device("cuda")
    with torch.no_grad():
        compare_configs()
