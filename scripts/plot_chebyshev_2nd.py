"""
Plot Chebyshev polynomials of the second kind U_1 ... U_5,
matching the style of the first-kind figure (chebyshev_polynomials.png).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"


def chebyshev_U(n, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    else:
        U_prev2 = np.ones_like(x)
        U_prev1 = 2.0 * x
        for _ in range(2, n + 1):
            U_curr = 2.0 * x * U_prev1 - U_prev2
            U_prev2 = U_prev1
            U_prev1 = U_curr
        return U_curr


def main():
    x = np.linspace(-2, 2, 2000)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for k, color in zip(range(1, 6), colors):
        y = chebyshev_U(k, x)
        ax.plot(x, y, linewidth=1.8, color=color, label=rf"$U_{{{k}}}$")

    ax.axhline(1, color="gray", linewidth=0.7, linestyle="--")
    ax.axhline(-1, color="gray", linewidth=0.7, linestyle="--")

    ax.set_xlim(-2, 2)
    ax.set_ylim(-6, 6)
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$U_k(x)$", fontsize=13)
    ax.set_title("Chebyshev Polynomials of the Second Kind", fontsize=14)
    ax.legend(fontsize=12, framealpha=0.9, ncol=3, loc="lower right")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "chebyshev_polynomials_2nd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ chebyshev_polynomials_2nd.png")


if __name__ == "__main__":
    main()
