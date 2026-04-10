import time

import torch
import matplotlib.pyplot as plt

from pcd_sampling_py.models import PCDSamplerConfig
from pcd_sampling_py.sampler import PCDSampler
from pcd_sampling_py.sampling_utils import reduce_gm, sample_ut

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_RUNS = 5
N_SAMPLES = 5
DIM = 2

weights = torch.tensor([0.25, 0.25, 0.25, 0.25], device=DEVICE)
means = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]], device=DEVICE)
covariances = torch.tensor(
    [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]], device=DEVICE
)

def vmapped_sample_pcd(weights, means, covariances, sampler: PCDSampler):
    batch_size = 200
    weights_batched = weights.unsqueeze(0).expand(batch_size, -1)
    means_batched = means.unsqueeze(0).expand(batch_size, -1, -1)
    covariances_batched = covariances.unsqueeze(0).expand(batch_size, -1, -1, -1)

    samples = torch.vmap(sampler.sample, randomness="different")(weights_batched, means_batched, covariances_batched)
    return samples
    


def time_pcd(steps: int, num_uvs: int) -> float:
    config = PCDSamplerConfig(
        number_samples=N_SAMPLES,
        dim=DIM,
        number_unit_vectors=num_uvs,
        steps=steps,
        sorting=True,
        initial_sampling_method="mean",
        unit_vectors_method="deterministic",
    )
    s = PCDSampler(config)
    # warmup
    with torch.no_grad():
        s.sample(weights, means, covariances)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed = 0.0
    with torch.no_grad():
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            vmapped_sample_pcd(weights, means, covariances, s)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0
    return elapsed / N_RUNS * 1000  # ms


def time_ut() -> float:
    gm_mean, gm_cov = reduce_gm(weights, means, covariances)
    # warmup
    with torch.no_grad():
        sample_ut(gm_mean, gm_cov)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed = 0.0
    with torch.no_grad():
        for _ in range(N_RUNS):
            # Must be measured together, since unavoidable.
            gm_mean, gm_cov = reduce_gm(weights, means, covariances)

            t0 = time.perf_counter()
            sample_ut(gm_mean, gm_cov)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0
    return elapsed / N_RUNS * 1000  # ms


# ---------- benchmark ----------
ut_time = time_ut()

FIXED_UVS = 50
steps_range = [1, 2, 3, 4, 5, 10, 15, 20]
times_steps = [time_pcd(s, FIXED_UVS) for s in steps_range]

FIXED_STEPS = 5
uvs_range = [20, 30, 40, 50, 100, 500, 1000,]
times_uvs = [time_pcd(FIXED_STEPS, u) for u in uvs_range]

# ---------- plot ----------
plt.rcParams.update({"font.size": 20})

plt.figure(figsize=(7, 5))
plt.plot(steps_range, times_steps, marker="o", label=f"PCD with {FIXED_UVS} unit vectors")
plt.axhline(ut_time, color="red", linestyle="--", label=f"UT ({ut_time:.3f} ms)")
plt.xlabel(f"Steps")
plt.ylabel("Time (ms)")
plt.title("PCD time vs. number of steps")
plt.legend()
plt.tight_layout()
plt.savefig("benchmarking/benchmark_steps.pdf", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(7, 5))
plt.plot(uvs_range, times_uvs, marker="o", label=f"PCD with {FIXED_STEPS} steps")
plt.axhline(ut_time, color="red", linestyle="--", label=f"UT ({ut_time:.3f} ms)")
plt.xlabel(f"Unit vectors")
plt.ylabel("Time (ms)")
plt.title("PCD time vs. number of unit vectors")
plt.legend()
plt.tight_layout()
plt.savefig("benchmarking/benchmark_unit_vectors.pdf", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

print(f"UT time: {ut_time:.4f} ms")
print("Steps benchmark:", list(zip(steps_range, [f'{t:.4f}' for t in times_steps])))
print("Unit-vectors benchmark:", list(zip(uvs_range, [f'{t:.4f}' for t in times_uvs])))
