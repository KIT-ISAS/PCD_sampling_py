import time

import torch
import matplotlib.pyplot as plt

# from benchmarking.cvm import sample_unit_vectors
from pcd_sampling_py import sampler
from pcd_sampling_py.models import PCDSamplerConfig
from pcd_sampling_py.sampler import PCDSampler
from pcd_sampling_py.sampling_utils import sample_ut

"""
A simple example of a 2 D Gaussian mixture
"""


def sample():
    # weights = torch.tensor([0.5, 0.5])
    # means = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    # covariances = torch.tensor([[[3.0, 2.8], [2.8, 3.0]], [[3.0, -2.8], [-2.8, 3.0]]])
    
    weights = torch.tensor([0.5, 0.5])
    means = torch.tensor([[2.0, 0.0], [-2., 0.0]])
    covariances = torch.tensor([[[1.0, 0], [0., 1.0]], [[1.0, 0.], [0., 1.0]]])
    
    # weights = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    # means = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    # # covariances = torch.tensor([[[3.0, 2.8], [2.8, 3.0]], [[3.0, -2.8], [-2.8, 3.0]], [[3.0, 0.0], [0.0, 3.0]], [[3.0, 0.0], [0.0, 3.0]], [[3.0, 0.0], [0.0, 3.0]]])
    # weights = torch.tensor([1.])
    # means = torch.tensor([[0.0, 0.0]])
    # covariances = torch.tensor([[[3.0, 1.0], [1.0, 3.0]]])

    torch.manual_seed(4)
    # unit_vectors = sample_unit_vectors(10, 2, device=means.device, dtype=means.dtype).double()

    sum_time = 0.0
    last_samples = None

    # sampling_config = PCDSamplerConfig(
    #     number_samples=5,
    #     dim=2,
    #     number_unit_vectors=2000,
    #     steps=28,
    #     # threshold=1e-6,
    #     sorting=True,
    #     initial_sampling_method="random",
    #     unit_vectors_method="random",
    # )
    # sampler = PCDSampler(sampling_config)

    for i in range(1):
        print(f"Sampling step {i}")
        start = time.time()
        sampling_config = PCDSamplerConfig(
            number_samples=2,
            dim=2,
            number_unit_vectors=100,
            steps=100,
            # threshold=1e-6,
            sorting=True,
            initial_sampling_method="random",
            unit_vectors_method="deterministic",
        )
        sampler = PCDSampler(sampling_config)

        X = sampler.sample(weights, means, covariances)
        # X = sample_ut(mean=means[0], covariance=covariances[0])
        
        last_samples = X
        diff = time.time() - start
        if i != 0:
            sum_time += diff

        print(f"Step {i}, elapsed: {diff}")
    print(f"Total no comp: {sum_time}")

    # last_samples = sampler.sample(weights, means, covariances)
    
    # vectors = sampler.unit_vectors.detach().cpu().numpy()
    # for v in vectors:
    #     plt.plot([0, v[0]], [0, v[1]], color="red", alpha=0.3, linewidth=0.8)
    if last_samples is not None:
        plot_gaussian_mixture_and_samples(
            weights,
            means,
            covariances,
            last_samples,
            grid_size=200,
            save_paths=["examples/gm_samples.pdf", "examples/result.pdf"],
            show=True,
            # vectors=vectors,
        )
        print("Saved plots to examples/gm_samples.pdf and examples/result.pdf")

def plot_gaussian_mixture_and_samples(
    weights: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    samples: torch.Tensor,
    grid_size: int = 150,
    save_paths: list[str] | None = None,
    show: bool = False,
    # vectors: torch.Tensor | None = None,
):
    """Plot a 2D Gaussian mixture density and sample locations.

    Args:
        weights: Mixture weights with shape (T,).
        means: Component means with shape (T, 2).
        covariances: Component covariance matrices with shape (T, 2, 2).
        samples: Sample locations with shape (L, 2).
        grid_size: Resolution of the evaluation grid.
        save_paths: Optional list of paths to save the figure.
        show: Whether to display the figure interactively.
    """

    device = weights.device
    

    # Ensure float precision for distributions
    weights = weights.to(dtype=torch.float32)
    means = means.to(dtype=torch.float32)
    covariances = covariances.to(dtype=torch.float32)
    samples = samples.detach().to(dtype=torch.float32, device=device)

    # Determine plotting bounds using component means and variances
    diag_vars = torch.diagonal(covariances, dim1=-2, dim2=-1)
    stds = torch.sqrt(torch.clamp(diag_vars, min=1e-6))
    mins = torch.min(means - 3 * stds, dim=0).values
    maxs = torch.max(means + 3 * stds, dim=0).values

    xs = torch.linspace(mins[0].item(), maxs[0].item(), grid_size, device=device)
    ys = torch.linspace(mins[1].item(), maxs[1].item(), grid_size, device=device)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    # Compute mixture density on the grid
    log_probs = []
    for w, mean, cov in zip(weights, means, covariances):
        mv = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        log_probs.append(torch.log(w) + mv.log_prob(grid))

    log_density = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0)
    density = torch.exp(log_density).reshape(grid_size, grid_size).cpu()

    # Move tensors to CPU for plotting
    xx_cpu = xx.cpu()
    yy_cpu = yy.cpu()
    means_cpu = means.cpu()
    samples_cpu = samples.cpu()
    
    # Use latex rendering for better text quality
    plt.rcParams.update({"font.size": 20})

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xx_cpu, yy_cpu, density, levels=30, cmap="Blues", alpha=0.85)
    plt.contour(
        xx_cpu, yy_cpu, density, levels=10, colors="black", linewidths=0.6, alpha=0.4
    )
    plt.colorbar(contour, label="Density")

    plt.scatter(
        samples_cpu[:, 0],
        samples_cpu[:, 1],
        c="white",
        s=200,
        label="Optimal samples",
        edgecolors="black",
        linewidths=1.5,
    )
    # for v in vectors:
    #     plt.plot([0, v[0]], [0, v[1]], color="red", alpha=0.3, linewidth=0.8)

    # Samples
  
    plt.scatter(
        means_cpu[:, 0],
        means_cpu[:, 1],
        c="darkorange",
        s=200,
        label="Component means",
        edgecolors="black",
        linewidths=0.8,
        marker="X",
    )

    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("Gaussian Mixture and Samples")
    plt.legend(loc="upper left")
    plt.tight_layout()

    if save_paths:
        for path in save_paths:
            plt.savefig(path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    torch.set_default_device("cuda")
    with torch.no_grad():
        sample()
