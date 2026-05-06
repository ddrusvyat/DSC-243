"""
Compare randomized Kaczmarz vs constant-stepsize SGD on interpolation least squares.

Model: D in R^{n x d} with rows of varying norms; w_* is a random unit vector and
y = D w_* (interpolation, so the noise floor in Theorem 8.1 vanishes).

Algorithms compared:
  - SGD on D with uniform sampling and stepsize gamma = 1/max_i ||d_i||^2
    (the largest stepsize allowed by assumption (25) with R^2 = max_i ||d_i||^2);
    rate sigma_min^2(D) / (n * max_i ||d_i||^2).
  - SGD on the row-rescaled system tilde D, tilde y with unit-norm rows,
    uniform sampling, and stepsize gamma = 1; rate sigma_min^2(tilde D) / n.
    The update coincides with the Kaczmarz projection; only the sampling differs.
  - Randomized Kaczmarz (Strohmer-Vershynin): importance sampling
    p_i = ||d_i||^2 / ||D||_F^2 plus the projection update; rate
    sigma_min^2(D) / ||D||_F^2 = sigma_min^2(D) / (n * mean_i ||d_i||^2).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def make_problem(
    n: int,
    d: int,
    row_scale_ratio: float,
    rng: np.random.Generator,
):
    """Generate an interpolation least-squares instance with prescribed
    multiplicative spread of row norms."""
    raw = rng.standard_normal((n, d))
    scales = np.exp(np.linspace(0.0, np.log(row_scale_ratio), n))
    rng.shuffle(scales)
    D = raw * scales[:, None]
    w_star = rng.standard_normal(d)
    w_star /= np.linalg.norm(w_star)
    y = D @ w_star
    return D, y, w_star


def run_sgd_uniform(
    D: np.ndarray,
    y: np.ndarray,
    w_star: np.ndarray,
    gamma: float,
    n_iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n, d = D.shape
    w = np.zeros(d)
    err = np.empty(n_iters + 1)
    err[0] = float(np.dot(w - w_star, w - w_star))
    for t in range(n_iters):
        i = int(rng.integers(n))
        residual = y[i] - float(D[i] @ w)
        w = w + gamma * residual * D[i]
        err[t + 1] = float(np.dot(w - w_star, w - w_star))
    return err


def run_kaczmarz(
    D: np.ndarray,
    y: np.ndarray,
    w_star: np.ndarray,
    n_iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n, d = D.shape
    row_norms_sq = np.einsum("ij,ij->i", D, D)
    probs = row_norms_sq / row_norms_sq.sum()
    w = np.zeros(d)
    err = np.empty(n_iters + 1)
    err[0] = float(np.dot(w - w_star, w - w_star))
    for t in range(n_iters):
        i = int(rng.choice(n, p=probs))
        residual = y[i] - float(D[i] @ w)
        w = w + (residual / row_norms_sq[i]) * D[i]
        err[t + 1] = float(np.dot(w - w_star, w - w_star))
    return err


def run_sgd_rescaled_uniform(
    D: np.ndarray,
    y: np.ndarray,
    w_star: np.ndarray,
    n_iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """SGD with uniform sampling on the row-rescaled system tilde D w = tilde y,
    where tilde d_i = d_i / ||d_i||. With stepsize gamma = 1 the update reduces
    to the Kaczmarz projection on the original system; only the sampling differs."""
    n, d = D.shape
    row_norms_sq = np.einsum("ij,ij->i", D, D)
    w = np.zeros(d)
    err = np.empty(n_iters + 1)
    err[0] = float(np.dot(w - w_star, w - w_star))
    for t in range(n_iters):
        i = int(rng.integers(n))
        residual = y[i] - float(D[i] @ w)
        w = w + (residual / row_norms_sq[i]) * D[i]
        err[t + 1] = float(np.dot(w - w_star, w - w_star))
    return err


def main() -> None:
    n, d = 500, 50
    row_scale_ratio = 8.0
    n_iters = 4000
    n_trials = 25

    rng_problem = np.random.default_rng(7)
    D, y, w_star = make_problem(n, d, row_scale_ratio, rng_problem)

    row_norms_sq = np.einsum("ij,ij->i", D, D)
    R2_max = float(row_norms_sq.max())
    avg_norm_sq = float(row_norms_sq.mean())
    sing_values = np.linalg.svd(D, compute_uv=False)
    sigma_min_sq = float(sing_values[-1] ** 2)
    fro_sq = float((D ** 2).sum())

    # Spectrum of the row-rescaled matrix tilde D (unit-norm rows).
    D_tilde = D / np.sqrt(row_norms_sq[:, None])
    sing_tilde = np.linalg.svd(D_tilde, compute_uv=False)
    sigma_min_sq_tilde = float(sing_tilde[-1] ** 2)

    # Per-step exponential rates (smaller = slower).
    rate_sgd = sigma_min_sq / (n * R2_max)
    rate_resc = sigma_min_sq_tilde / n
    rate_kac = sigma_min_sq / fro_sq

    print(f"row-norm spread max/avg     = {R2_max / avg_norm_sq:.3f}")
    print(f"R^2 = max_i ||d_i||^2       = {R2_max:.4g}")
    print(f"avg_i ||d_i||^2             = {avg_norm_sq:.4g}")
    print(f"sigma_min^2(D)              = {sigma_min_sq:.4g}")
    print(f"sigma_min^2(tilde D)        = {sigma_min_sq_tilde:.4g}")
    print(f"||D||_F^2                   = {fro_sq:.4g}")
    print(f"SGD per-step rate           = {rate_sgd:.4g}")
    print(f"Rescaled-uniform rate       = {rate_resc:.4g}")
    print(f"Kaczmarz per-step rate      = {rate_kac:.4g}")
    print(f"Kaczmarz / SGD speedup      = {rate_kac / rate_sgd:.3f}")
    print(f"Kaczmarz / Rescaled speedup = {rate_kac / rate_resc:.3f}")

    sgd_runs = np.empty((n_trials, n_iters + 1))
    resc_runs = np.empty((n_trials, n_iters + 1))
    kac_runs = np.empty((n_trials, n_iters + 1))
    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        sgd_runs[trial] = run_sgd_uniform(
            D, y, w_star, gamma=1.0 / R2_max, n_iters=n_iters, rng=rng
        )
        rng = np.random.default_rng(3000 + trial)
        resc_runs[trial] = run_sgd_rescaled_uniform(
            D, y, w_star, n_iters=n_iters, rng=rng
        )
        rng = np.random.default_rng(2000 + trial)
        kac_runs[trial] = run_kaczmarz(D, y, w_star, n_iters=n_iters, rng=rng)
        if (trial + 1) % 5 == 0:
            print(f"  trial {trial + 1}/{n_trials} done")

    sgd_med = np.median(sgd_runs, axis=0)
    sgd_lo = np.quantile(sgd_runs, 0.1, axis=0)
    sgd_hi = np.quantile(sgd_runs, 0.9, axis=0)
    resc_med = np.median(resc_runs, axis=0)
    resc_lo = np.quantile(resc_runs, 0.1, axis=0)
    resc_hi = np.quantile(resc_runs, 0.9, axis=0)
    kac_med = np.median(kac_runs, axis=0)
    kac_lo = np.quantile(kac_runs, 0.1, axis=0)
    kac_hi = np.quantile(kac_runs, 0.9, axis=0)

    iters = np.arange(n_iters + 1)
    e0 = float(np.dot(-w_star, -w_star))  # ||w_0 - w_*||^2 with w_0 = 0
    t_grid = np.linspace(1, n_iters, 400)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=120)
    color_sgd = "#d7191c"
    color_resc = "#2ca02c"
    color_kac = "#2c7bb6"

    ax.fill_between(iters, sgd_lo, sgd_hi, color=color_sgd, alpha=0.15, linewidth=0)
    ax.semilogy(
        iters, sgd_med, "-", color=color_sgd, linewidth=1.8,
        label=r"SGD on $D$ (uniform, $\gamma=1/R^2$)",
    )

    ax.fill_between(iters, resc_lo, resc_hi, color=color_resc, alpha=0.15, linewidth=0)
    ax.semilogy(
        iters, resc_med, "-", color=color_resc, linewidth=1.8,
        label=r"SGD on rescaled $\tilde D$ (uniform, $\gamma=1$)",
    )

    ax.fill_between(iters, kac_lo, kac_hi, color=color_kac, alpha=0.15, linewidth=0)
    ax.semilogy(
        iters, kac_med, "-", color=color_kac, linewidth=1.8,
        label="Randomized Kaczmarz",
    )

    ax.set_xlabel(r"iteration $t$", fontsize=12)
    ax.set_ylabel(r"$\|w_t - w_\ast\|^2$", fontsize=12)
    ax.set_ylim(1e-22, 5.0)
    ax.set_title(
        rf"SGD variants vs Randomized Kaczmarz on interpolation LS"
        rf"  ($n={n},\ d={d}$, row-norm ratio $={row_scale_ratio:.0f}$)",
        fontsize=11.5,
    )
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    fig.tight_layout()

    out = FIGURES_DIR / "kaczmarz_vs_sgd.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
