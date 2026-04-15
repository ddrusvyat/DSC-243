"""
Compare gradient descent convergence on kernel regression across the
Matérn family: Laplace (m=1/2), Matérn 3/2 (m=3/2), Matérn 5/2 (m=5/2),
and Gaussian (m=∞).  The target function is the same for all kernels.

Gaps are computed via eigendecomposition to avoid numerical issues with
np.linalg.solve on ill-conditioned systems.  The y-axis shows the relative
gap (f(x_k)-f*)/(f(x_0)-f*).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def laplace_kernel(X, sigma):
    D = np.abs(X[:, None] - X[None, :])
    return np.exp(-D / sigma)


def matern32_kernel(X, sigma):
    D = np.abs(X[:, None] - X[None, :])
    r = np.sqrt(3) * D / sigma
    return (1 + r) * np.exp(-r)


def matern52_kernel(X, sigma):
    D = np.abs(X[:, None] - X[None, :])
    r = np.sqrt(5) * D / sigma
    return (1 + r + r ** 2 / 3) * np.exp(-r)


def gaussian_kernel(X, sigma):
    D = np.abs(X[:, None] - X[None, :])
    return np.exp(-D ** 2 / (2 * sigma ** 2))


def gd_relative_gaps_spectral(eigs, d_coeffs, beta, k_max):
    """
    Compute relative gap (f(x_k)-f*)/(f(x_0)-f*) using the spectral formula:
      gap_k = (1/2) sum_i (d_i^2 / lam_i) * (1 - lam_i/beta)^{2k}
    where d_i = v_i^T b.
    """
    ratios = 1.0 - eigs / beta
    weights = 0.5 * d_coeffs ** 2 / eigs
    gap0 = np.sum(weights)
    gaps = np.zeros(k_max + 1)
    power = np.ones_like(eigs)
    for k in range(k_max + 1):
        gaps[k] = np.dot(weights, power) / gap0
        power *= ratios ** 2
    return gaps


def main():
    np.random.seed(0)
    n = 200
    sigma = 0.15
    x = np.sort(np.random.rand(n))
    k_max = 5000

    y = np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)

    kernels = [
        (r"Laplace ($m=\frac{1}{2}$)", laplace_kernel(x, sigma), "#d7191c"),
        (r"Matérn 3/2 ($m=\frac{3}{2}$)", matern32_kernel(x, sigma), "#e66101"),
        (r"Matérn 5/2 ($m=\frac{5}{2}$)", matern52_kernel(x, sigma), "#5e3c99"),
        (r"Gaussian ($m=\infty$)", gaussian_kernel(x, sigma), "#2c7bb6"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    iters = np.arange(1, k_max + 1)

    for label, K, color in kernels:
        A = K / n
        eigs, V = np.linalg.eigh(A)
        eigs = np.maximum(eigs, 1e-30)
        beta = eigs[-1]
        b = y / n
        d_coeffs = V.T @ b

        rel_gaps = gd_relative_gaps_spectral(eigs, d_coeffs, beta, k_max)
        gd_vals = rel_gaps[1:]
        pos = gd_vals > 0
        ax.loglog(iters[pos], gd_vals[pos], "-", color=color, linewidth=1.5,
                  label=label)

        e0_sq = np.sum(d_coeffs ** 2 / eigs ** 2)
        gap0 = np.sum(0.5 * d_coeffs ** 2 / eigs)
        bound_const = 0.5 * beta * e0_sq / gap0
        kappa = eigs[-1] / eigs[0]
        print(f"{label:30s}  kappa={kappa:.2e}"
              f"  C_rel={bound_const:.2e}")

    ax.set_xlabel("Iteration $k$", fontsize=12)
    ax.set_ylabel(r"$(f(x_k) - f^\ast)\,/\,(f(x_0) - f^\ast)$", fontsize=12)
    ax.set_title(
        r"GD convergence across Matérn family "
        rf"($n={n}$, $\sigma={sigma}$, "
        r"$h(x)=\sin(2\pi x)+\frac{1}{2}\cos(4\pi x)$)",
        fontsize=11.5,
    )
    ax.legend(fontsize=9.5, framealpha=0.9, loc="best")
    ax.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    out = FIGURES_DIR / "convergence_matern.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


if __name__ == "__main__":
    main()
