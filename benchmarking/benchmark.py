"""Benchmark PCD sampling across seeds/sample sizes using YAML configs."""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.pcd_sampling_py.models import PCDSamplerConfig
from src.pcd_sampling_py.sampler import PCDSampler


def load_config(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def build_gaussian_mixture(dim: int, components: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic Gaussian mixture for benchmarking with configurable shape."""

    weights = torch.full((components,), 1.0 / components, device=device, dtype=torch.float32)
    means = torch.zeros((components, dim), device=device, dtype=torch.float32)
    means[:, 0] = torch.linspace(-2.0, 2.0, components, device=device)
    covariances = torch.stack([torch.eye(dim, device=device, dtype=torch.float32) for _ in range(components)], dim=0)
    return weights, means, covariances


def ensure_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    torch.set_default_device(device)
    return device


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def sample_once(
    number_samples: int,
    seed: int,
    sampler_cfg: Dict,
    gm: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> float:
    torch.manual_seed(seed)

    config = PCDSamplerConfig(
        dim=sampler_cfg["dimension"],
        number_unit_vectors=sampler_cfg["unit_vectors"],
        number_samples=number_samples,
        threshold=sampler_cfg.get("threshold", 0.1),
        steps=sampler_cfg.get("steps", 40),
        sorting=True,
    )

    sampler = PCDSampler(config)
    weights, means, covariances = gm
    # Warm up sampling:
    sampler.sample(weights, means, covariances)
    
    start = time.perf_counter()
    if "threshold" in sampler_cfg and hasattr(sampler, "sample_threshold"):
        _ = sampler.sample_threshold(weights, means, covariances)
    else:
        _ = sampler.sample(weights, means, covariances)
    maybe_synchronize(device)
    return time.perf_counter() - start


def run_benchmark(cfg: Dict) -> Tuple[Sequence[int], Sequence[int], List[List[float]]]:
    device = ensure_device(cfg.get("device", "cpu"))
    seeds = list(range(cfg["start_seed"], cfg["end_seed"] + 1))
    sample_counts = list(range(cfg["start_samples"], cfg["end_samples"] + 1, cfg["step_samples"]))

    gm = build_gaussian_mixture(cfg["dimension"], cfg["components"], device)

    all_times: List[List[float]] = []
    for n_samples in sample_counts:
        seed_times: List[float] = []
        for seed in seeds:
            elapsed = sample_once(n_samples, seed, cfg, gm, device)
            seed_times.append(elapsed)
            print(f"samples={n_samples}, seed={seed}, elapsed={elapsed:.6f}s")
        all_times.append(seed_times)

    return sample_counts, seeds, all_times


def save_csv(sample_counts: Sequence[int], seeds: Sequence[int], times: List[List[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["num_samples"] + [f"seed_{seed}_time" for seed in seeds]
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for n, row in zip(sample_counts, times):
            writer.writerow([n] + row)
    print(f"Saved CSV to {output_path}")


def plot_results(sample_counts: Sequence[int], times: List[List[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.array(times, dtype=float)  # shape: (num_counts, num_seeds)
    mean = data.mean(axis=1)
    std = data.std(axis=1)

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 5))

    def _plot(ax, log_scale: bool) -> None:
        lower = np.clip(mean - std, a_min=1e-12, a_max=None) if log_scale else mean - std
        upper = mean + std

        ax.plot(sample_counts, mean, color="tab:blue", label="mean")
        ax.fill_between(sample_counts, lower, upper, color="tab:blue", alpha=0.2, label="std")
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title("Sampling time (log-log scale)")
        else:
            ax.set_title("Sampling time (linear scale)")
        ax.set_xlabel("Number of samples")
        ax.set_ylabel("Elapsed time [s]")
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.legend()

    _plot(ax_lin, log_scale=False)
    _plot(ax_log, log_scale=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved plots to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PCD sampling with YAML configs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmarking/benchmarking_configs/config_1.yaml"),
        help="Path to benchmarking config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path("benchmarking/benchmarking_results")
    csv_path = output_dir / cfg["output_csv"]
    png_path = output_dir / cfg["output_png"]

    with torch.no_grad():
        sample_counts, seeds, times = run_benchmark(cfg)

    save_csv(sample_counts, seeds, times, csv_path)
    plot_results(sample_counts, times, png_path)


if __name__ == "__main__":
    main()
