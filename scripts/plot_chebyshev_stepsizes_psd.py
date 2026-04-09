"""
Plot PSD Chebyshev stepsizes in the same style as the PD stepsize figure:
one row of dots per k, with stepsize on the x-axis and k on the y-axis.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"

beta = 50.0
K_MAX = 20


def psd_chebyshev_stepsizes(k, beta):
    j = np.arange(1, k + 1)
    return 1.0 / (beta * np.sin(j * np.pi / (2 * k)) ** 2)


def main():
    fig, ax = plt.subplots(figsize=(10, 7))

    cmap = plt.cm.viridis
    norm = Normalize(vmin=1, vmax=K_MAX)

    for k in range(1, K_MAX + 1):
        etas = psd_chebyshev_stepsizes(k, beta)
        color = cmap(norm(k))
        ax.scatter(etas, np.full_like(etas, k), s=28, color=color, zorder=3)

    ax.set_xlabel(r"Stepsize $\eta_j$", fontsize=13)
    ax.set_ylabel(r"Number of steps $k$", fontsize=13)
    ax.set_title(rf"Chebyshev stepsizes (PSD): $\beta = {int(beta)}$", fontsize=14)
    ax.set_ylim(K_MAX + 0.5, 0.5)
    ax.set_yticks(range(1, K_MAX + 1))
    ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "chebyshev_stepsizes_psd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ chebyshev_stepsizes_psd.png")


if __name__ == "__main__":
    main()
