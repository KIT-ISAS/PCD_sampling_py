"""Benchmark convergence of PCD sampling using moment matching errors per step."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pcd_sampling_py.models import PCDSamplerConfig
from pcd_sampling_py.sampler import PCDSampler


def build_gaussian_mixture(
    dim: int = 2, components: int = 2, device: torch.device | None = None
):
    device = device or torch.device("cpu")
    weights = torch.full((components,), 1.0 / components, device=device, dtype=torch.float32)
    means = torch.zeros((components, dim), device=device, dtype=torch.float32)
    means[:, 0] = torch.linspace(-2.0, 2.0, components, device=device)
    covariances = torch.stack(
        [torch.eye(dim, device=device, dtype=torch.float32) for _ in range(components)],
        dim=0,
    )
    return weights, means, covariances


def compute_mixture_moments(weights: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor):
    """Return population mean and covariance of a Gaussian mixture."""
    mixture_mean = torch.sum(weights[:, None] * means, dim=0)
    centered = means - mixture_mean
    mixture_cov = torch.sum(
        weights[:, None, None]
        * (covariances + torch.einsum("bi,bj->bij", centered, centered)),
        dim=0,
    )
    return mixture_mean, mixture_cov


def ensure_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    torch.set_default_device(device)
    return device


def run_benchmark(
    number_samples: int,
    max_steps: int,
    number_unit_vectors: int,
    device: torch.device,
    output_dir: Path,
    dim: int = 2,
    components: int = 2,
    sorting: bool = True,
    threshold: float = 1e-4,
):
    torch.manual_seed(0)

    weights, means, covariances = build_gaussian_mixture(
        dim=dim, components=components, device=device
    )
    target_mean, target_cov = compute_mixture_moments(weights, means, covariances)

    records: list[tuple[int, float, float]] = []

    for steps in range(1, max_steps + 1):
        sampler_cfg = PCDSamplerConfig(
            dim=dim,
            number_unit_vectors=number_unit_vectors,
            number_samples=number_samples,
            steps=steps,
            sorting=sorting,
            threshold=threshold,
        )
        sampler = PCDSampler(sampler_cfg)

        with torch.no_grad():
            samples = sampler.sample(weights, means, covariances)

        sample_mean = samples.mean(dim=0)
        centered = samples - sample_mean
        sample_cov = centered.T @ centered / samples.shape[0]

        mean_error = torch.linalg.vector_norm(sample_mean - target_mean).item()
        cov_error = torch.linalg.matrix_norm(sample_cov - target_cov, ord="fro").item()

        records.append((steps, mean_error, cov_error))

        print(
            f"Step {steps:3d} | mean error: {mean_error:.6f} | cov error: {cov_error:.6f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"pcd_moment_errors_dim={dim}_comp={components}_steps={max_steps}.csv"
    png_path = output_dir / f"pcd_moment_errors_dim={dim}_comp={components}_steps={max_steps}.pdf"

    np.savetxt(
        csv_path,
        np.array(records),
        delimiter=",",
        header="step,mean_error,cov_error",
        comments="",
    )

    steps_arr = [r[0] for r in records]
    mean_errs = [r[1] for r in records]
    cov_errs = [r[2] for r in records]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps_arr, mean_errs, marker="o", label="Mean error (L2)")
    ax.plot(steps_arr, cov_errs, marker="s", label="Covariance error (Frobenius)")
    ax.set_xlabel("Optimization steps")
    ax.set_ylabel("Moment error")
    ax.set_title("Moment matching convergence")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close(fig)

    print(f"Saved CSV to {csv_path}")
    print(f"Saved plot to {png_path}")

    return csv_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark convergence via moment matching errors."
    )
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples (L).")
    parser.add_argument(
        "--max-steps", type=int, default=40, help="Maximum optimization steps to evaluate."
    )
    parser.add_argument(
        "--unit-vectors", type=int, default=1000, help="Number of unit vectors (K)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (e.g., 'cpu' or 'cuda'). Falls back to CPU if CUDA unavailable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarking/benchmarking_results"),
        help="Directory to store CSV and plot.",
    )
    parser.add_argument("--dim", type=int, default=2, help="Dimensionality of the Gaussian mixture.")
    parser.add_argument("--components", type=int, default=2, help="Number of Gaussian mixture components.")
    parser.add_argument(
        "--sorting",
        action="store_true",
        help="Enable sorting-based update (same as sampler default). If omitted, disables sorting.",
    )
    parser.add_argument(
        "--no-sorting",
        dest="sorting",
        action="store_false",
        help="Disable sorting-based update.",
    )
    parser.set_defaults(sorting=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-4,
        help="Threshold passed to sampler config (not used when steps are fixed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = ensure_device(args.device)
    run_benchmark(
        number_samples=args.samples,
        max_steps=args.max_steps,
        number_unit_vectors=args.unit_vectors,
        device=device,
        output_dir=args.output_dir,
        dim=args.dim,
        components=args.components,
        sorting=args.sorting,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
