"""
Run gradient descent and conjugate gradient on the kernel regression problem
(1/n)K alpha = (1/n)y for a smooth target, and produce a log-log convergence
plot showing f(x_k) - f* vs iteration k with predicted rate lines.
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


def gd_quadratic(A, b, eta, k_max):
    """GD on f(x) = 0.5 x^T A x - b^T x with fixed stepsize eta."""
    d = len(b)
    x = np.zeros(d)
    x_star = np.linalg.solve(A, b)
    f_star = 0.5 * x_star @ A @ x_star - b @ x_star
    gaps = np.zeros(k_max + 1)
    for k in range(k_max + 1):
        fk = 0.5 * x @ A @ x - b @ x
        gaps[k] = fk - f_star
        if k < k_max:
            x = x - eta * (A @ x - b)
    return gaps


def cg_quadratic(A, b, k_max):
    """CG on A x = b (equivalently minimizing f(x) = 0.5 x^T A x - b^T x)."""
    d = len(b)
    x = np.zeros(d)
    r = b.copy()
    p = r.copy()
    x_star = np.linalg.solve(A, b)
    f_star = 0.5 * x_star @ A @ x_star - b @ x_star
    gaps = np.zeros(k_max + 1)
    gaps[0] = -f_star

    for k in range(k_max):
        Ap = A @ p
        rr = r @ r
        alpha_cg = rr / (p @ Ap)
        x = x + alpha_cg * p
        r = r - alpha_cg * Ap
        fk = 0.5 * x @ A @ x - b @ x
        gaps[k + 1] = fk - f_star
        if gaps[k + 1] < 1e-30:
            gaps[k + 1:] = gaps[k + 1]
            break
        beta_cg = (r @ r) / rr
        p = r + beta_cg * p

    return gaps


def main():
    np.random.seed(0)
    n = 200
    sigma = 0.15
    x = np.sort(np.random.rand(n))

    K = laplace_kernel(x, sigma)
    A = K / n
    beta = np.max(np.linalg.eigvalsh(A))
    eta = 1.0 / beta
    k_max = 5000

    y_smooth = np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)
    b_smooth = y_smooth / n
    gaps_gd = gd_quadratic(A, b_smooth, eta, k_max)
    gaps_cg = cg_quadratic(A, b_smooth, k_max)

    fig, ax = plt.subplots(figsize=(7, 5))

    iters = np.arange(1, k_max + 1)
    gd_vals = gaps_gd[1:]
    cg_vals = gaps_cg[1:]
    gd_pos = gd_vals > 0
    cg_pos = cg_vals > 0

    ax.loglog(iters[gd_pos], gd_vals[gd_pos],
              "-", color="#2c7bb6", linewidth=1.5,
              label="Gradient Descent")
    ax.loglog(iters[cg_pos], cg_vals[cg_pos],
              "-", color="#d7191c", linewidth=1.5,
              label="Conjugate Gradient")

    k_ref = np.logspace(0, np.log10(k_max), 200)

    # GD: O(k^{-1.8}) from Thm 7.1 with s'=0.4
    gd_anchor_k = 500
    gd_anchor_val = gd_vals[gd_anchor_k - 1]
    ref_gd = gd_anchor_val * (gd_anchor_k / k_ref) ** 1.8
    ax.loglog(k_ref, ref_gd, "--", color="#2c7bb6", linewidth=1.0,
              alpha=0.6, label=r"$O(k^{-1.8})$ (Thm 7.1)")

    # CG: O(k^{-3.6}) from Cor 7.1 with s'=0.4
    cg_anchor_k = 50
    cg_anchor_val = cg_vals[cg_anchor_k - 1]
    ref_cg = cg_anchor_val * (cg_anchor_k / k_ref) ** 3.6
    ax.loglog(k_ref, ref_cg, "--", color="#d7191c", linewidth=1.0,
              alpha=0.6, label=r"$O(k^{-3.6})$ (Cor 7.1)")

    ax.set_xlabel("Iteration $k$", fontsize=12)
    ax.set_ylabel(r"$f(x_k) - f^\ast$", fontsize=12)
    ax.set_title(
        rf"Convergence on kernel regression "
        rf"(Laplace, $n={n}$, $\sigma={sigma}$, "
        rf"$h(x) = \sin(2\pi x) + \frac{{1}}{{2}}\cos(4\pi x)$)",
        fontsize=12,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "convergence_kernel.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ convergence_kernel.png")


if __name__ == "__main__":
    main()
