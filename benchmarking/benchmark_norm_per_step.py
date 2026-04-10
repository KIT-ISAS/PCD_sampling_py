"""Benchmark gain norm per step using PCDSampler.benchmark_steps for a 2D, 2-component GM."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pcd_sampling_py.models import PCDSamplerConfig
from pcd_sampling_py.sampler import PCDSampler


def build_gaussian_mixture(dim: int = 10, components: int = 2, device: torch.device | None = None):
    device = device or torch.device("cpu")
    weights = torch.full((components,), 1.0 / components, device=device, dtype=torch.float32)
    means = torch.zeros((components, dim), device=device, dtype=torch.float32)
    means[:, 0] = torch.linspace(-2.0, 2.0, components, device=device)
    covariances = torch.stack(
        [torch.eye(dim, device=device, dtype=torch.float32) for _ in range(components)], dim=0
    )
    return weights, means, covariances


def ensure_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    torch.set_default_device(device)
    return device


def run_benchmark(
    number_samples: int,
    steps: int,
    number_unit_vectors: int,
    device: torch.device,
    output_dir: Path,
):
    sampler_cfg = PCDSamplerConfig(
        dim=2,
        number_unit_vectors=number_unit_vectors,
        number_samples=number_samples,
        steps=steps,
        sorting=True,
        # threshold=1e-4,
        initial_sampling_method="mean"
        
    )

    sampler = PCDSampler(sampler_cfg)

    weights, means, covariances = build_gaussian_mixture(dim=sampler_cfg.dim, components=2, device=device)

    with torch.no_grad():
        norms = sampler.benchmark_steps(weights, means, covariances).cpu()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"pcd_norms_dim=2_comp=2_steps={steps}.csv"
    png_path = output_dir / f"pcd_norms_dim=2_comp=2_steps={steps}.pdf"

    np.savetxt(csv_path, norms.numpy(), delimiter=",", header="mean_gain_norm", comments="")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(norms)), norms.numpy(), marker="o", label="mean gain norm")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean gain norm")
    ax.set_title("Gain norm per step (dim=2, components=10, samples=10, uv=40)")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close(fig)

    print(f"Saved CSV to {csv_path}")
    print(f"Saved plot to {png_path}")

    return csv_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark gain norm per step for 2D, 2-component GM.")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples (L).")
    parser.add_argument("--steps", type=int, default=40, help="Number of optimization steps.")
    parser.add_argument("--unit-vectors", type=int, default=30, help="Number of unit vectors (K).")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = ensure_device(args.device)
    run_benchmark(
        number_samples=args.samples,
        steps=args.steps,
        number_unit_vectors=args.unit_vectors,
        device=device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    main()
