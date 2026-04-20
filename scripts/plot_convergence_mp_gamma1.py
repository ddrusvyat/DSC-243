"""
Run gradient descent and conjugate gradient on a random linear least-squares
problem in the Marchenko--Pastur critical regime gamma = d/n = 1, and produce
a log-log convergence plot showing f(x_k) - f* vs iteration k with the
predicted sublinear rate lines:
    GD:  O(k^{-3/2})   (Section 7, Marchenko--Pastur asymptotics)
    CG:  O(k^{-3})     (Theorem 7.5)

Setup. Draw D in R^{n x n} with iid standard Gaussian entries (so gamma = 1),
form A = (1/n) D^T D (whose limiting spectrum is MP on [0, 4]), pick a
ground-truth x_star uniformly on the sphere, set b = A x_star, and start
GD/CG from x_0 = 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def gd_quadratic(A, b, eta, k_max):
    """GD on f(x) = 0.5 x^T A x - b^T x with fixed stepsize eta."""
    d = len(b)
    x = np.zeros(d)
    x_star = np.linalg.solve(A, b)
    f_star = 0.5 * x_star @ A @ x_star - b @ x_star
    gaps = np.zeros(k_max + 1)
    for k in range(k_max + 1):
        gaps[k] = (0.5 * x @ A @ x - b @ x) - f_star
        if k < k_max:
            x = x - eta * (A @ x - b)
    return gaps


def cg_quadratic(A, b, k_max):
    """CG on A x = b (equivalently minimize f(x) = 0.5 x^T A x - b^T x)."""
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
        denom = p @ Ap
        if denom <= 0:
            gaps[k + 1:] = gaps[k]
            break
        alpha_cg = rr / denom
        x = x + alpha_cg * p
        r = r - alpha_cg * Ap
        gaps[k + 1] = (0.5 * x @ A @ x - b @ x) - f_star
        if gaps[k + 1] < 1e-30:
            gaps[k + 1:] = gaps[k + 1]
            break
        beta_cg = (r @ r) / rr
        p = r + beta_cg * p
    return gaps


def main():
    rng = np.random.default_rng(0)
    n = 1500
    n_trials = 30
    k_max = 800

    gd_runs = np.zeros((n_trials, k_max + 1))
    cg_runs = np.zeros((n_trials, k_max + 1))

    for t in range(n_trials):
        D = rng.standard_normal((n, n))
        A = (D.T @ D) / n
        beta = float(np.max(np.linalg.eigvalsh(A)))
        eta = 1.0 / beta

        x_star = rng.standard_normal(n)
        x_star /= np.linalg.norm(x_star) / np.sqrt(n)
        b = A @ x_star

        gd_runs[t] = gd_quadratic(A, b, eta, k_max)
        cg_runs[t] = cg_quadratic(A, b, k_max)
        print(f"  trial {t + 1}/{n_trials} done (beta = {beta:.3f})")

    gd_med = np.median(gd_runs, axis=0)
    cg_med = np.median(cg_runs, axis=0)
    gd_lo, gd_hi = np.quantile(gd_runs, [0.1, 0.9], axis=0)
    cg_lo, cg_hi = np.quantile(cg_runs, [0.1, 0.9], axis=0)

    iters = np.arange(1, k_max + 1)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.fill_between(iters, gd_lo[1:], gd_hi[1:],
                    color="#2c7bb6", alpha=0.18, linewidth=0)
    ax.loglog(iters, gd_med[1:], "-", color="#2c7bb6", linewidth=1.7,
              label=f"Gradient Descent (median over {n_trials} trials)")

    ax.fill_between(iters, cg_lo[1:], cg_hi[1:],
                    color="#d7191c", alpha=0.18, linewidth=0)
    ax.loglog(iters, cg_med[1:], "-", color="#d7191c", linewidth=1.7,
              label=f"Conjugate Gradient (median over {n_trials} trials)")

    k_ref = np.logspace(0, np.log10(k_max), 200)

    # GD reference: O(k^{-3/2}) from MP gamma=1 GD asymptotic
    gd_anchor_k = 30
    gd_anchor_val = gd_med[gd_anchor_k]
    ref_gd = gd_anchor_val * (gd_anchor_k / k_ref) ** 1.5
    ax.loglog(k_ref, ref_gd, "--", color="#2c7bb6", linewidth=1.0, alpha=0.8,
              label=r"$O(k^{-3/2})$ (GD, MP $\gamma=1$)")

    # CG reference: O(k^{-3}) from Theorem 7.5
    cg_anchor_k = 10
    cg_anchor_val = cg_med[cg_anchor_k]
    ref_cg = cg_anchor_val * (cg_anchor_k / k_ref) ** 3.0
    ax.loglog(k_ref, ref_cg, "--", color="#d7191c", linewidth=1.0, alpha=0.8,
              label=r"$O(k^{-3})$ (Theorem 7.5)")

    ax.set_xlabel("Iteration $k$", fontsize=12)
    ax.set_ylabel(r"$f(x_k) - f^\ast$", fontsize=12)
    ax.set_title(
        rf"Random least squares at the MP critical aspect ratio "
        rf"($n = d = {n}$, $A = \frac{{1}}{{n}} D^\top D$)",
        fontsize=11.5,
    )
    ax.set_ylim(top=max(gd_hi[1], cg_hi[1]) * 2)
    ax.legend(fontsize=10, framealpha=0.9, loc="lower left")
    ax.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    out = FIGURES_DIR / "convergence_mp_gamma1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


if __name__ == "__main__":
    main()
