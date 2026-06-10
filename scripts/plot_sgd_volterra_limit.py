"""
Illustrate Theorem 10.8 (Volterra risk curve as the high-dimensional limit of
streaming SGD on correlated least squares) from Section 10.

Model (same correlated-feature setup as the SGD-vs-SDE figure):
    H = Q diag(lambda_1, ..., lambda_d) Q^T,  lambda_i = i/d on (0, 1],
        Q a Haar-random orthogonal matrix (so H is NOT diagonal -- the features
        are correlated across all coordinates, not axis-aligned).
    x ~ N(0, H),   y = <w_*, x> + sigma * eta,  eta ~ N(0, 1),  ||w_*|| = 1.
    L(w) - L(w_*) = (1/2) (w - w_*)^T H (w - w_*),  w_0 = 0.
    Streaming SGD with stepsize gamma/d.

Why we can simulate in the eigenbasis. The model is orthogonally invariant:
under v = Q^T w the rotated-H problem with signal w_* becomes EXACTLY the
diagonal problem with covariance diag(lambda) and signal s = Q^T w_*. For
Haar-random Q and ||w_*|| = 1, s is uniform on the unit sphere. We therefore
run the fast diagonal dynamics in the eigenbasis with a fresh random unit signal
per trial -- this is the rotated-H experiment, computed in the smart basis.

Volterra model (Section 10):
    F(s) = sigma^2/2 + (1/2) sum_i lambda_i e^{-2 s lambda_i} c_i^2,  c_i = s_i,
    K(t) = (gamma^2 / d) sum_i lambda_i^2 e^{-2 gamma t lambda_i},        (56)
    Psi(t) = F(gamma t) + integral_0^t K(t - s) Psi(s) ds.               (54)

For several d we run n_trials independent SGD trajectories (each with its own
random rotation, i.e. random unit signal) of L(w_{[td]}) - L(w_*) vs epoch t and
plot the median together with the 10-90% interquantile band, against the single
deterministic Volterra solution Psi(t) - sigma^2/2. The deterministic limit uses
the signal-weighted measure c_i^2 = E[s_i^2] = 1/d (total mass ||w_*||^2 = 1),
the high-dimensional limit of the random per-trial signal. As d grows the bands
shrink around the deterministic Volterra curve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def ramp_spectrum(d: int) -> np.ndarray:
    """lambda_i = i/d for i = 1, ..., d (eigenvalues of H, uniform on (0, 1])."""
    return np.arange(1, d + 1, dtype=float) / d


def random_unit_signals(n_trials: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Per-trial signal in H's eigenbasis, s = Q^T w_* with Q Haar-random and
    ||w_*|| = 1. For Haar Q this is uniform on the unit sphere; draw directly as
    normalized Gaussians. Returns an array of shape (n_trials, d)."""
    G = rng.standard_normal((n_trials, d))
    return G / np.linalg.norm(G, axis=1, keepdims=True)


def run_streaming_sgd(
    d: int,
    lam: np.ndarray,
    sigma: float,
    gamma: float,
    S: np.ndarray,
    n_epochs: float,
    rng: np.random.Generator,
):
    """n_trials streaming-SGD trajectories in the eigenbasis (vectorized over
    trials), each with its own signal S[trial] and start v_0 = 0. Stepsize
    gamma/d. Returns epochs t_k = k/d and excess risks R[trial, k]."""
    n_trials = S.shape[0]
    n_steps = int(round(n_epochs * d))
    sqrt_lam = np.sqrt(lam)
    V = np.zeros((n_trials, d))
    R = np.empty((n_trials, n_steps + 1))
    diff = V - S
    R[:, 0] = 0.5 * ((diff * diff) @ lam)
    for k in range(n_steps):
        Z = rng.standard_normal((n_trials, d))
        X = Z * sqrt_lam
        eta = sigma * rng.standard_normal(n_trials)
        y = np.sum(X * S, axis=1) + eta
        residual = np.sum(V * X, axis=1) - y
        V -= (gamma / d) * residual[:, None] * X
        diff = V - S
        R[:, k + 1] = 0.5 * ((diff * diff) @ lam)
    epochs = np.arange(n_steps + 1) / d
    return epochs, R


def volterra_solve(
    t_grid: np.ndarray,
    lam: np.ndarray,
    sigma: float,
    gamma: float,
    c_sq: np.ndarray,
):
    """Solve Psi(t) = F(gamma t) + integral_0^t K(t - s) Psi(s) ds on a uniform
    grid t_grid by the trapezoidal rule, with forcing weights c_sq = c_i^2.
    Returns (Psi, F_gt) of length t_grid.size."""
    h = float(t_grid[1] - t_grid[0])
    d = lam.size
    Eg = np.exp(-2.0 * gamma * np.outer(t_grid, lam))   # shape (N, d)
    F_gt = 0.5 * sigma ** 2 + 0.5 * (Eg * (lam * c_sq)).sum(axis=1)
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


def deterministic_volterra(d_ref, sigma, gamma, n_epochs, n_grid=601):
    """Single deterministic Volterra curve Psi(t) - sigma^2/2 at reference
    dimension d_ref, using the limiting signal-weighted measure c_i^2 = 1/d_ref
    (= E[s_i^2]). Returns (t_grid, excess_Psi)."""
    lam_ref = ramp_spectrum(d_ref)
    c_sq_ref = np.full(d_ref, 1.0 / d_ref)
    t_grid = np.linspace(0.0, n_epochs, n_grid)
    Psi, _ = volterra_solve(t_grid, lam_ref, sigma, gamma, c_sq_ref)
    return t_grid, Psi - 0.5 * sigma ** 2


def panel_vary_gamma(ax, sigma, n_epochs, d, gammas, n_trials, rng_master):
    """Fixed d, several stepsizes: SGD median+band (colored) vs the deterministic
    Volterra curve (dashed black) for each gamma."""
    colors = plt.cm.viridis(np.linspace(0.15, 0.80, len(gammas)))
    lam = ramp_spectrum(d)
    for gamma, color in zip(gammas, colors):
        S = random_unit_signals(
            n_trials, d, np.random.default_rng(rng_master.integers(0, 2 ** 31))
        )
        rng = np.random.default_rng(rng_master.integers(0, 2 ** 31))
        epochs, runs = run_streaming_sgd(d, lam, sigma, gamma, S, n_epochs, rng)
        med = np.median(runs, axis=0)
        lo, hi = np.quantile(runs, [0.1, 0.9], axis=0)
        ax.fill_between(epochs, lo, hi, color=color, alpha=0.22, linewidth=0)
        ax.plot(epochs, med, "-", color=color, linewidth=1.6,
                label=rf"$\gamma={gamma}$")
        t_grid, excess_Psi = deterministic_volterra(1024, sigma, gamma, n_epochs)
        ax.plot(t_grid, excess_Psi, "-", color="black", linewidth=1.3, alpha=0.9)
        gap = float(np.max(np.abs(np.interp(epochs, t_grid, excess_Psi) - med)))
        print(f"  gamma={gamma}: max |median SGD - Volterra| = {gap:.3e}")
    ax.set_yscale("log")
    ax.set_xlim(0, n_epochs)
    ax.set_xlabel(r"epoch $t = k/d$", fontsize=12)
    ax.set_ylabel(r"excess risk $L(w_{[td]}) - L(w_\ast)$", fontsize=12)
    ax.set_title(rf"Vary $\gamma$ at fixed $d={d}$ "
                 rf"(color: SGD, black: Volterra $\Psi$)", fontsize=11.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)


def panel_vary_d(ax, sigma, n_epochs, gamma, dims, n_trials, rng_master):
    """Fixed stepsize, several dimensions: SGD median+band (colored) concentrating
    around the single deterministic Volterra curve (dashed black)."""
    colors = plt.cm.viridis(np.linspace(0.15, 0.80, len(dims)))
    for d, color in zip(dims, colors):
        lam = ramp_spectrum(d)
        S = random_unit_signals(
            n_trials, d, np.random.default_rng(rng_master.integers(0, 2 ** 31))
        )
        rng = np.random.default_rng(rng_master.integers(0, 2 ** 31))
        epochs, runs = run_streaming_sgd(d, lam, sigma, gamma, S, n_epochs, rng)
        med = np.median(runs, axis=0)
        lo, hi = np.quantile(runs, [0.1, 0.9], axis=0)
        ax.fill_between(epochs, lo, hi, color=color, alpha=0.20, linewidth=0)
        ax.plot(epochs, med, "-", color=color, linewidth=1.5,
                label=rf"$d={d}$")
        band = float(np.max((hi - lo)[epochs >= 2.0]))
        print(f"  d={d}: R_final median = {med[-1]:.3e}, "
              f"max band (t>=2) = {band:.3e}")
    t_grid, excess_Psi = deterministic_volterra(1024, sigma, gamma, n_epochs)
    ax.plot(t_grid, excess_Psi, "-", color="black", linewidth=1.8,
            label=r"Volterra $\Psi(t) - \frac{1}{2}\sigma^2$ from (54)")
    ax.set_yscale("log")
    ax.set_xlim(0, n_epochs)
    ax.set_xlabel(r"epoch $t = k/d$", fontsize=12)
    ax.set_ylabel(r"excess risk $L(w_{[td]}) - L(w_\ast)$", fontsize=12)
    ax.set_title(rf"Fix $\gamma={gamma}$, vary $d$ "
                 rf"(color: SGD, black: Volterra limit)", fontsize=11.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)


def main():
    sigma = 0.1
    n_epochs = 8.0

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.0, 5.4))

    print("left panel (vary gamma, d=512):")
    panel_vary_gamma(ax_left, sigma, n_epochs, d=512, gammas=[0.5, 1.0, 1.5],
                     n_trials=40, rng_master=np.random.default_rng(0))
    print("right panel (fix gamma=1, vary d):")
    panel_vary_d(ax_right, sigma, n_epochs, gamma=1.0, dims=[50, 200, 800, 3200],
                 n_trials=30, rng_master=np.random.default_rng(1))

    fig.suptitle(
        rf"Streaming SGD vs. the deterministic Volterra risk curve, "
        rf"$H=Q\Lambda Q^\top$ with $\lambda_i=i/d$ and Haar-random $Q$ "
        rf"($\sigma={sigma}$, $w_0=0$)",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIGURES_DIR / "sgd_volterra_limit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
