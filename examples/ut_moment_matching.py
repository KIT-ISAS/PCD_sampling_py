"""
Three-panel visualisation of Gaussian-mixture moment matching + UT sampling.

Panel 1: Original 4-component GMM  (coloured 2σ ellipses + means)
Panel 2: Moment-matched (reduced) single Gaussian  (red ellipse + mean cross)
Panel 3: Reduced Gaussian + Unscented-Transform sigma points

Requires:  torch, matplotlib, numpy
           pcd_sampling_py.sampling_utils  (reduce_gm, sample_ut)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

from pcd_sampling_py.sampling_utils import reduce_gm, sample_ut


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def covariance_ellipse(mean, cov, n_std=2.0, **kwargs):
    """Return a matplotlib Ellipse for a 2-D Gaussian (n_std standard deviations)."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    return Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)


# ---------------------------------------------------------------------------
# GMM definition — 4 components in 2-D
# ---------------------------------------------------------------------------

weights = torch.tensor([0.25, 0.25, 0.25, 0.25])

means = torch.tensor(
    [
        [-4.0, 3.0],  # comp 0  (top-left)
        [-2.0, 1.0],  # comp 1  (bottom-left)
        [2.0, 1.0],  # comp 2  (centre)
        [4.0, 3.0],  # comp 3  (top-right)
    ]
)

covariances = torch.tensor(
    [
        [[1.5, -1.0], [-1.0, 1.5]],
        [[1.5, -1.0], [-1.0, 1.5]],
        [[1.5, 1.0], [1.0, 1.5]],
        [[1.5, 1.0], [1.0, 1.5]],
    ]
)

comp_colours = ["blue", "red", "orange", "purple"]


# ---------------------------------------------------------------------------
# moment-match + UT
# ---------------------------------------------------------------------------

with torch.no_grad():
    red_mean, red_cov = reduce_gm(weights, means, covariances)
    sigma_pts = sample_ut(red_mean, red_cov)  # (2d+1, 2)

red_mean_np = red_mean.cpu().numpy()
red_cov_np = red_cov.cpu().numpy()
sigma_np = sigma_pts.detach().cpu().numpy()
means_np = means.cpu().numpy()
covs_np = covariances.cpu().numpy()

# ---------------------------------------------------------------------------
# figure — three panels
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# axis limits (shared)
xlim = (-7.5, 7.5)
ylim = (-1.5, 5.5)


# ---- Panel 1: Original 4-component GMM ------------------------------------
ax = axes[0]
ax.set_title("Original 4-component GMM", fontsize=15, fontweight="bold")

for i, (m, c, col) in enumerate(zip(means_np, covs_np, comp_colours)):
    ell = covariance_ellipse(
        m,
        c,
        n_std=2.0,
        edgecolor=col,
        facecolor="none",
        linewidth=1.,
        zorder=2,
    )
    ax.add_patch(ell)
    # ax.scatter(*m, marker="x", c=col, s=100, linewidths=2, zorder=4, label=f"comp {i}")

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("x")
ax.set_ylabel("y")
# ax.legend(loc="upper right", fontsize=20, framealpha=0.9)
ax.set_aspect("equal")


# ---- Panel 2: Moment-matched reduced Gaussian -----------------------------
ax = axes[1]
ax.set_title("Moment-matched reduction", fontsize=20, fontweight="bold")
for i, (m, c, col) in enumerate(zip(means_np, covs_np, comp_colours)):
    ell = covariance_ellipse(
        m,
        c,
        n_std=2.0,
        edgecolor=col,
        facecolor="none",
        linewidth=1.,
        zorder=2,
    )
    ax.add_patch(ell)
    # ax.scatter(*m, marker="x", c=col, s=100, linewidths=2, zorder=4, label=f"comp {i}")
ell = covariance_ellipse(
    red_mean_np,
    red_cov_np,
    n_std=2.0,
    edgecolor="black",
    facecolor="none",
    linewidth=3.0,
    zorder=2,
)
ax.add_patch(ell)
# ax.scatter(
#     *red_mean_np,
#     marker="x",
#     c="green",
#     s=120,
#     linewidths=2.5,
#     zorder=4,
#     label="reduced mean",
# )

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("x")
ax.set_ylabel("y")
# ax.legend(loc="upper right", fontsize=20, framealpha=0.9)
ax.set_aspect("equal")


# ---- Panel 3: Reduced Gaussian + UT sigma points --------------------------
ax = axes[2]
ax.set_title(
    "Unscented sigma points (reduced Gaussian)", fontsize=20, fontweight="bold"
)
for i, (m, c, col) in enumerate(zip(means_np, covs_np, comp_colours)):
    ell = covariance_ellipse(
        m,
        c,
        n_std=2.0,
        edgecolor=col,
        facecolor="none",
        linewidth=1.,
        zorder=2,
    )
    ax.add_patch(ell)
ell = covariance_ellipse(
    red_mean_np,
    red_cov_np,
    n_std=2.0,
    edgecolor="black",
    facecolor="none",
    linewidth=3.0,
    zorder=2,
)
ax.add_patch(ell)
ax.scatter(
    sigma_np[:, 0], sigma_np[:, 1], c="black", s=80, zorder=5, label="sigma points"
)
# ax.scatter(
#     *red_mean_np,
#     marker="x",
#     c="black",
#     s=120,
#     linewidths=2.5,
#     zorder=4,
#     label="reduced mean",
# )

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("x")
ax.set_ylabel("y")
# ax.legend(loc="upper right", fontsize=20, framealpha=0.9)
ax.set_aspect("equal")


# ---- Arrows between panels -------------------------------------------------
# for i in range(2):
# arrow = FancyArrowPatch(
#     posA=(1.0, 0.5), posB=(0.0, 0.5),
#     coordsA=axes[i].transAxes, coordsB=axes[i + 1].transAxes,
#     arrowstyle="-|>", mutation_scale=20,
#     color="darkslategrey", linewidth=2.0,
#     transform=fig.transFigure, clip_on=False,
# )
# fig.patches.append(arrow)

plt.tight_layout()
plt.savefig("gm_reduction_pipeline.pdf", dpi=200, bbox_inches="tight")
plt.savefig("gm_reduction_pipeline.png", dpi=200, bbox_inches="tight")
print("Saved  gm_reduction_pipeline.pdf / .png")
plt.show()
