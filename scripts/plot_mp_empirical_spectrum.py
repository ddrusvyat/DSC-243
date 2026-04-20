"""
Animate the empirical spectral distribution of A = (1/n) X^T X
converging to the Marchenko-Pastur law, for three aspect ratios
gamma = d/n in {0.5, 1.0, 2.0}.

Convention: X in R^{n x d} with iid standard Gaussian entries.
As n -> infinity with d = round(gamma * n), the empirical spectral
distribution of A converges to the MP density on
  [(1 - sqrt(gamma))^2,  (1 + sqrt(gamma))^2].
For gamma > 1 there is an additional atom of mass 1 - 1/gamma at 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def mp_density(lam: np.ndarray, gamma: float) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    lam_m = (1.0 - np.sqrt(gamma)) ** 2
    lam_p = (1.0 + np.sqrt(gamma)) ** 2
    inside = np.clip((lam_p - lam) * (lam - lam_m), 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.sqrt(inside) / (2.0 * np.pi * gamma * lam)
    rho = np.where(np.isfinite(rho), rho, 0.0)
    rho[(lam < lam_m) | (lam > lam_p)] = 0.0
    return rho


def sample_eigs(n: int, gamma: float, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    d = max(int(round(gamma * n)), 1)
    X = rng.standard_normal(size=(n, d))
    A = (X.T @ X) / n
    return np.linalg.eigvalsh(A), d


def main() -> None:
    gammas = [0.5, 1.0, 2.0]
    bulk_colors = ["#2980b9", "#e67e22", "#27ae60"]
    n_values = [30, 60, 120, 250, 500, 1000]
    rng = np.random.default_rng(2026)

    lam_max = max((1 + np.sqrt(g)) ** 2 for g in gammas) * 1.08
    lam_grid = np.linspace(1e-4, lam_max, 800)
    densities = [mp_density(lam_grid, g) for g in gammas]
    # Per-panel y-axis: gamma = 1 diverges like 1/sqrt(lambda) at the
    # origin and needs extra headroom; the bounded gamma = 0.5 and
    # gamma = 2 panels get tighter y-limits so their histograms and
    # density curves are easier to compare.
    y_tops = {0.5: 1.6, 1.0: 2.2, 2.0: 0.8}

    frames_data = []
    for n in n_values:
        per_gamma = [sample_eigs(n, g, rng) for g in gammas]
        frames_data.append((n, per_gamma))

    n_bins = 70
    bin_edges = np.linspace(0.0, lam_max, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for ax, g, rho in zip(axes, gammas, densities):
        ax.plot(lam_grid, rho, color="black", linewidth=2.0, zorder=5)
        ax.set_xlim(-0.05, lam_max)
        ax.set_ylim(0, y_tops[g])
        ax.set_xlabel(r"$\lambda$", fontsize=11)
        ax.grid(True, alpha=0.25)
        if g > 1.0:
            ax.axvline(0, color="black", linewidth=1.4, linestyle=":", zorder=4)
            ax.annotate(
                rf"atom of mass $1-\frac{{1}}{{\gamma}}={1 - 1/g:.2g}$ at $\lambda=0$",
                xy=(0.03, 0.93), xycoords="axes fraction",
                fontsize=9, ha="left", va="top",
            )

    axes[0].set_ylabel("density", fontsize=11)
    legend_handles = [
        Line2D([0], [0], color="black", linewidth=2.0, label="Marchenko-Pastur density"),
        Patch(facecolor=bulk_colors[0], alpha=0.55, edgecolor="white",
              label=r"empirical spectrum of $A$"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle(
        r"Empirical spectrum of $A=\frac{1}{n}X^{\top} X$"
        r" vs. Marchenko-Pastur law   ($d/n\to\gamma$,  $n\to\infty$)",
        fontsize=13.5, y=1.01,
    )

    def draw(frame_idx: int):
        n, per_gamma = frames_data[frame_idx]
        for ax, g, color, (eigs, d) in zip(axes, gammas, bulk_colors, per_gamma):
            for patch in list(ax.patches):
                patch.remove()
            # For gamma > 1 exclude the (d-n) forced zero eigenvalues
            # and normalize by the total count so the histogram matches
            # the bulk mass 1/gamma of the MP density.
            if g > 1.0:
                bulk_eigs = eigs[eigs > 1e-8]
            else:
                bulk_eigs = eigs
            total = len(eigs)
            weights = np.full(bulk_eigs.shape, 1.0 / (total * bin_width))
            ax.hist(
                bulk_eigs, bins=bin_edges, weights=weights,
                color=color, alpha=0.55, edgecolor="white", zorder=2,
            )
            ax.set_title(rf"$\gamma={g:g}$,   $n={n}$,   $d={d}$", fontsize=12)
        return ()

    anim = FuncAnimation(
        fig, draw, frames=len(frames_data),
        interval=1100, repeat_delay=2500, blit=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    gif_path = FIGURES_DIR / "mp_empirical_spectrum.gif"
    anim.save(gif_path, writer=PillowWriter(fps=1.1))
    plt.close(fig)
    print(f"Saved {gif_path}")


if __name__ == "__main__":
    main()
