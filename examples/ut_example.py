import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from pcd_sampling_py.sampling_utils import reduce_gm, sample_ut


"""
Plot a 2D Gaussian Mixture with Unscented Transform sigma points
and an ellipse of the moment-matched (reduced) Gaussian.
"""


def covariance_ellipse(mean, cov, n_std=2.0, **kwargs):
    """Return a matplotlib Ellipse patch for a 2D Gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    return Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)


def plot_gm_and_ut(
    weights: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    grid_size: int = 200,
    save_path: str | None = None,
    show: bool = True,
):
    weights_f = weights.to(dtype=torch.float32)
    means_f = means.to(dtype=torch.float32)
    covs_f = covariances.to(dtype=torch.float32)

    # --- Reduced Gaussian & UT sigma points ---
    red_mean, red_cov = reduce_gm(weights_f, means_f, covs_f)
    sigma_pts = sample_ut(red_mean, red_cov)  # (2n+1, 2)

    # --- Grid for density ---
    diag_vars = torch.diagonal(covs_f, dim1=-2, dim2=-1)
    stds = torch.sqrt(diag_vars.clamp(min=1e-6))
    mins = (means_f - 4 * stds).min(0).values
    maxs = (means_f + 4 * stds).max(0).values

    xs = torch.linspace(mins[0].item(), maxs[0].item(), grid_size)
    ys = torch.linspace(mins[1].item(), maxs[1].item(), grid_size)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    log_probs = []
    for w, m, c in zip(weights_f, means_f, covs_f):
        mv = torch.distributions.MultivariateNormal(m, covariance_matrix=c)
        log_probs.append(torch.log(w) + mv.log_prob(grid))
    density = torch.exp(torch.logsumexp(torch.stack(log_probs), 0))
    density = density.reshape(grid_size, grid_size).cpu().numpy()

    xx_np = xx.cpu().numpy()
    yy_np = yy.cpu().numpy()
    sigma_np = sigma_pts.detach().cpu().numpy()
    red_mean_np = red_mean.cpu().numpy()
    red_cov_np = red_cov.cpu().numpy()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams.update({"font.size": 20})
    contour = ax.contourf(xx_np, yy_np, density, levels=30, cmap="Blues", alpha=0.85)
    ax.contour(xx_np, yy_np, density, levels=10, colors="black", linewidths=0.6, alpha=0.4)
    plt.colorbar(contour, ax=ax, label="Density")

    # Reduced-GM ellipse 2σ)
    for n_std, alpha in [(2.0, 0.4)]:
        ell = covariance_ellipse(
            red_mean_np, red_cov_np, n_std=n_std,
            edgecolor="black", facecolor="none",
            linewidth=2.0, linestyle="--", alpha=alpha,
            label=f"Reduced GM {int(n_std)}σ ellipse" if n_std == 2.0 else None,
            zorder=3,
        )
        ax.add_patch(ell)

    # Reduced-GM mean
    # ax.scatter(*red_mean_np, c="darkorange", s=80, zorder=5,
    #            edgecolors="black", linewidths=0.8, label="Reduced GM mean")

    # UT sigma points
    ax.scatter(sigma_np[:, 0], sigma_np[:, 1],
               c="white", s=80, zorder=5,
               edgecolors="black", linewidths=1.5, label="UT sigma points")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Gaussian Mixture — Unscented Transform")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {save_path}")
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    torch.set_default_device("cpu")

    weights = torch.tensor([0.5, 0.5])
    means = torch.tensor([[2.0, 0.0], [-2.0, 0.0]])
    covariances = torch.tensor([[[3.0, 2.8], [2.8, 3.0]], [[3.0, -2.8], [-2.8, 3.0]]])
    

    with torch.no_grad():
        plot_gm_and_ut(
            weights, means, covariances,
            save_path="examples/ut_example.pdf",
            show=True,
        )
