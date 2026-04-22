"""
Illustrate Theorem 9.9 (Volterra risk curve as the high-dimensional limit
of streaming SGD on correlated least squares) from Section 9.

Setup:
    H = diag(lambda_1, ..., lambda_d) with lambda_i = i/d on (0, 1]
    x ~ N(0, H)  (sampled as x_i = sqrt(lambda_i) * z_i, z ~ N(0, I_d))
    y = <w_*, x> + sigma * eta,   eta ~ N(0, 1)
    L(w) - L(w_*) = (1/2) (w - w_*)^T H (w - w_*)
    Streaming SGD with stepsize gamma/d.

Volterra model (57):
    F(t) = L(Y_t) = sigma^2/2 + (1/2) sum_i lambda_i e^{-2 t lambda_i} (w_{0,i} - w_{*,i})^2
    K(t) = (gamma^2 / d) sum_i lambda_i^2 e^{-2 gamma t lambda_i}
    Psi(t) = F(gamma t) + integral_0^t K(t - s) Psi(s) ds.

For several d, we run n_trials independent SGD trajectories of
L(w_{[td]}) - L(w_*) vs epoch t and plot the median together with the 10-90%
interquantile band, against the Volterra solution Psi(t) - sigma^2/2 obtained
by trapezoidal-rule discretization of (57). As d grows the bands shrink around
the deterministic Volterra curve.

To make the SGD trajectories directly comparable across dimensions and to the
single Volterra reference curve, we choose w_* deterministically so that the
per-coordinate squared difference (w_{0,i} - w_{*,i})^2 = 1/d for every i.
With w_0 = 0 this is achieved by w_*_i = 1/sqrt(d). Then the initial excess
risk equals (1/2) * mean(lambda) and the Volterra forcing F(gamma t) is fully
determined by the spectrum of H and gamma.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def linear_ramp_spectrum(d: int) -> np.ndarray:
    """Return lambda_i = i/d for i = 1, ..., d (uniformly spaced eigenvalues
    in (0, 1])."""
    return np.arange(1, d + 1, dtype=float) / d


def run_streaming_sgd_diag(
    d: int,
    lam: np.ndarray,
    sigma: float,
    gamma: float,
    w_star: np.ndarray,
    w0: np.ndarray,
    n_epochs: float,
    rng: np.random.Generator,
):
    """One trial of streaming SGD on least squares with diagonal feature
    covariance H = diag(lam) and stepsize gamma/d. Returns epochs t_k = k/d
    and excess risks (1/2) sum_i lam_i (w_{k,i} - w_{*,i})^2."""
    n_steps = int(round(n_epochs * d))
    R = np.empty(n_steps + 1)
    w = w0.copy()
    sqrt_lam = np.sqrt(lam)
    diff = w - w_star
    R[0] = 0.5 * float(lam @ (diff * diff))
    for k in range(n_steps):
        z = rng.standard_normal(d)
        x = sqrt_lam * z
        eta = sigma * rng.standard_normal()
        y = float(w_star @ x) + eta
        residual = float(w @ x) - y
        w -= (gamma / d) * residual * x
        diff = w - w_star
        R[k + 1] = 0.5 * float(lam @ (diff * diff))
    epochs = np.arange(n_steps + 1) / d
    return epochs, R


def volterra_solve(
    t_grid: np.ndarray,
    lam: np.ndarray,
    sigma: float,
    gamma: float,
    w0_diff_sq: np.ndarray,
):
    """Solve Psi(t) = F(gamma t) + integral_0^t K(t - s) Psi(s) ds on a
    uniform grid t_grid by the trapezoidal rule. Returns (Psi, F_gt) of length
    t_grid.size."""
    h = float(t_grid[1] - t_grid[0])
    d = lam.size
    # Forcing F(gamma t):
    Eg = np.exp(-2.0 * gamma * np.outer(t_grid, lam))   # shape (N, d)
    F_gt = 0.5 * sigma ** 2 + 0.5 * (Eg * (lam * w0_diff_sq)).sum(axis=1)
    # Memory kernel K(t):
    K = (gamma ** 2 / d) * (Eg * (lam ** 2)).sum(axis=1)

    N = t_grid.size
    Psi = np.empty(N)
    Psi[0] = F_gt[0]
    for n in range(1, N):
        Kr = K[: n + 1][::-1]                # K(t_n - t_m) for m = 0..n
        w_trap = np.ones(n + 1)
        w_trap[0] *= 0.5
        w_trap[n] *= 0.5
        diag_w = h * w_trap[n] * Kr[n]       # coefficient of unknown Psi[n]
        rest = h * float(np.dot(w_trap[:n] * Kr[:n], Psi[:n]))
        Psi[n] = (F_gt[n] + rest) / (1.0 - diag_w)
    return Psi, F_gt


def main():
    sigma = 0.1
    gamma = 0.5
    n_epochs = 12.0
    dims = [50, 200, 800, 3200]
    n_trials = 30

    rng_master = np.random.default_rng(2)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(dims)))

    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    for d, color in zip(dims, colors):
        lam = linear_ramp_spectrum(d)
        # w_* with per-coordinate (w_{*,i})^2 = 1/d so that ||w_*||^2 = 1.
        w_star = np.full(d, 1.0 / np.sqrt(d))
        w0 = np.zeros(d)

        n_steps = int(round(n_epochs * d))
        runs = np.empty((n_trials, n_steps + 1))
        for trial in range(n_trials):
            rng = np.random.default_rng(rng_master.integers(0, 2 ** 31))
            epochs, R = run_streaming_sgd_diag(
                d, lam, sigma, gamma, w_star, w0, n_epochs, rng
            )
            runs[trial] = R

        med = np.median(runs, axis=0)
        lo, hi = np.quantile(runs, [0.1, 0.9], axis=0)

        ax.fill_between(epochs, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(
            epochs, med, "-", color=color, linewidth=1.4,
            label=f"streaming SGD, $d={d}$",
        )
        print(
            f"  d={d}: {n_steps} steps x {n_trials} trials, "
            f"R_init median = {med[0]:.3e}, R_final median = {med[-1]:.3e}"
        )

    # Volterra reference: a single deterministic curve evaluated at the largest
    # dimension's spectrum (the empirical spectral measure of lambda_i = i/d
    # converges to U[0,1] as d -> infty; with d = 1024 the discretization is
    # already very close to the limit kernel).
    d_ref = 1024
    lam_ref = linear_ramp_spectrum(d_ref)
    w0_diff_sq_ref = np.full(d_ref, 1.0 / d_ref)        # matches w_* used above
    t_grid = np.linspace(0.0, n_epochs, 601)
    Psi, F_gt = volterra_solve(t_grid, lam_ref, sigma, gamma, w0_diff_sq_ref)
    excess_Psi = Psi - 0.5 * sigma ** 2
    ax.plot(t_grid, excess_Psi, "--", color="black", linewidth=1.7,
            label=r"Volterra $\Psi(t) - \frac{1}{2}\sigma^2$ from (57)")

    ax.set_xlabel(r"epoch $t = k/d$", fontsize=12)
    ax.set_ylabel(r"excess risk $L(w_{[td]}) - L(w_\ast)$", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(
        rf"Streaming SGD with linear-ramp covariance $\lambda_i = i/d$ "
        rf"($\sigma={sigma}$, $\gamma={gamma}$, $w_0=0$, "
        rf"median and 10-90% band over {n_trials} trials)",
        fontsize=11.5,
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax.set_xlim(0, n_epochs)

    fig.tight_layout()
    out = FIGURES_DIR / "sgd_volterra_limit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
