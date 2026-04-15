"""
Bach-style scaling law comparison for power-law model.

Plants eigenvalues lambda_i = L/i^alpha and initial error delta_i = Delta/i^{beta/2},
computes exact GD gaps, and compares with:
  - The classical O(1/k) upper bound
  - The asymptotic equivalent from Theorem 7.2
Two panels: (alpha=2, beta=2) where rate > 1/k, and (alpha=2, beta=0) where rate < 1/k.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import gamma as gammafn
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def gd_gaps_powerlaw(alpha, beta, L, Delta, d, k_max):
    """Compute exact GD function gaps for the power-law model via spectral formula."""
    idx = np.arange(1, d + 1, dtype=np.float64)
    lam = L * idx ** (-alpha)
    delta_sq = Delta ** 2 * idx ** (-beta)

    bta = lam[0]  # largest eigenvalue = L
    weights = 0.5 * delta_sq * lam
    ratios = (1.0 - lam / bta) ** 2

    gaps = np.zeros(k_max + 1)
    power = np.ones(d)
    for k in range(k_max + 1):
        gaps[k] = np.dot(weights, power)
        power *= ratios
    return gaps


def asymptotic_equivalent(alpha, beta, L, Delta, k_arr):
    """Bach's asymptotic equivalent: L*Delta^2/(2*alpha) * Gamma(omega) / (2k)^omega."""
    omega = (beta - 1) / alpha + 1
    coeff = L * Delta ** 2 / (2 * alpha) * gammafn(omega)
    return coeff / (2.0 * k_arr) ** omega


def nsc_bound(alpha, beta, L, Delta, d, k_arr):
    """Non-strongly-convex O(1/k) upper bound: L/(4e) * ||e_0||^2 / k."""
    idx = np.arange(1, d + 1, dtype=np.float64)
    e0_sq = Delta ** 2 * np.sum(idx ** (-beta))
    return L * e0_sq / (4 * np.e * k_arr)


def main():
    d = 5000
    k_max = 3000
    L = 1.0
    Delta = 1.0

    configs = [
        {
            "alpha": 2, "beta": 2,
            "title": r"$\alpha=2,\;\beta=2$"
                     "\n"
                     r"(rate $k^{-3/2}$, faster than $1/k$)",
        },
        {
            "alpha": 2, "beta": 0,
            "title": r"$\alpha=2,\;\beta=0$"
                     "\n"
                     r"(rate $k^{-1/2}$, slower than $1/k$)",
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, cfg in zip(axes, configs):
        alpha = cfg["alpha"]
        beta = cfg["beta"]
        omega = (beta - 1) / alpha + 1

        gaps = gd_gaps_powerlaw(alpha, beta, L, Delta, d, k_max)

        k_arr = np.arange(1, k_max + 1)
        gd_vals = gaps[1:]

        ax.loglog(k_arr, gd_vals, "-", color="#2c7bb6", linewidth=1.8,
                  label="Actual GD", zorder=3)

        asy = asymptotic_equivalent(alpha, beta, L, Delta, k_arr.astype(float))
        ax.loglog(k_arr, asy, "--", color="#d7191c", linewidth=1.5,
                  label=rf"Asymptotic $\sim k^{{-{omega:.1f}}}$  (Thm 7.2)",
                  zorder=2)

        if beta > 1:
            nsc = nsc_bound(alpha, beta, L, Delta, d, k_arr.astype(float))
            ax.loglog(k_arr, nsc, "--", color="gray", linewidth=1.2,
                      alpha=0.7, label=r"$O(1/k)$ bound (Thm 6.1)", zorder=1)
        else:
            ax.text(0.97, 0.97,
                    r"$\|e_0\|^2 = \infty$" "\n" r"$O(1/k)$ bound vacuous",
                    transform=ax.transAxes, fontsize=9.5,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7))

        ax.set_xlabel("Iteration $k$", fontsize=12)
        ax.set_ylabel(r"$f(x_k) - f^\ast$", fontsize=12)
        ax.set_title(cfg["title"], fontsize=12)
        ax.legend(fontsize=9.5, framealpha=0.9, loc="best")
        ax.grid(True, alpha=0.25, which="both")
        ax.set_xlim(1, k_max)

    fig.suptitle(
        r"Power-law scaling: $\lambda_i = i^{-\alpha}$, "
        r"$\delta_i = i^{-\beta/2}$, "
        rf"$d = {d}$",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "scaling_laws_powerlaw.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


if __name__ == "__main__":
    main()
