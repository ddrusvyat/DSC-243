"""
Plot the minimizer z^*_i = 1 - i/(d+1) of the chain quadratic
    bar f(z) = (1/2) z^T T z - e_1^T z
together with the truncated minimizers
    hat z^{(2k+1)}_i = 1 - i/(2k+2)   for i <= 2k+1, else 0,
which are the best points any deterministic first-order method can reach
inside E_{2k+1} after k gradient queries (by Lemma 9.1). The shaded tail
between hat z^{(2k+1)} and z^* on coordinates i > 2k+1 is exactly what
Theorem 9.2 / Theorem 9.3 lower-bound.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def main() -> None:
    d = 50
    coords = np.arange(1, d + 1)
    z_star = 1.0 - coords / (d + 1)

    fig, ax = plt.subplots(figsize=(10.0, 5.2), dpi=130)

    # Gray reference line at zero.
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.6)

    # True minimizer.
    ax.plot(
        coords, z_star, "-", color="#222222", linewidth=2.4,
        label=r"$z^\ast_i = 1 - i/(d+1)$",
        zorder=6,
    )
    ax.plot(coords, z_star, "o", color="#222222", markersize=3.5, zorder=6)

    # Truncated minimizers for several k.
    ks = [2, 6, 12, 20]
    colors = ["#3182bd", "#fd8d3c", "#31a354", "#9e3a9e"]

    for k, color in zip(ks, colors):
        m = 2 * k + 1
        z_trunc = np.zeros(d)
        z_trunc[:m] = 1.0 - np.arange(1, m + 1) / (m + 1)
        ax.plot(
            coords, z_trunc, "-", color=color, linewidth=1.5, alpha=0.95,
            label=rf"$\hat z^{{(2k+1)}}\in E_{{{m}}}\;(k={k})$",
        )
        if m < d:
            ax.fill_between(
                coords[m - 1:], z_trunc[m - 1:], z_star[m - 1:],
                color=color, alpha=0.13, linewidth=0,
            )

    ax.set_xlabel(r"coordinate $i$", fontsize=12)
    ax.set_ylabel(r"value", fontsize=12)
    ax.set_title(
        r"Tail gap between truncated minimizers $\hat z^{(2k+1)}\in E_{2k+1}$ and $z^\ast$ "
        rf"($d = {d}$)",
        fontsize=12,
    )
    ax.set_xlim(0.5, d + 0.5)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)

    fig.tight_layout()
    out = FIGURES_DIR / "chain_minimizers.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
