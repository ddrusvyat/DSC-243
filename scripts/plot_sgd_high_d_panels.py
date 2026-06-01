"""
Combined two-panel figure for the high-dimensional limit of streaming SGD on
the isotropic Gaussian model (Section 10).

Left panel  (vary the stepsize, fix d): excess-risk trajectories for several
    stepsizes gamma at a single large d, illustrating the bias-variance
    trade-off of the ODE limit (decay rate maximized at gamma=1, stationary
    risk increasing in gamma).
Right panel (fix the stepsize, vary d): excess-risk trajectories for fixed
    gamma=1 across several dimensions d. The ODE limit is dimension-independent,
    so all medians follow the same dashed curve while the 10-90% band narrows
    like 1/sqrt(d) as d grows -- the concentration of Theorem 10.2.

Both panels share the same model:
    x ~ N(0, I_d),  eta ~ N(0, sigma^2),  y = <w_*, x> + eta
    L(w) - L(w_*) = (1/2) ||w - w_*||^2
    w_{k+1} = w_k - (gamma/d) (<w_k, x_{k+1}> - y_{k+1}) x_{k+1}.

Rendering both panels in one image guarantees they appear side by side in any
Markdown/HTML viewer (no reliance on CSS flexbox support).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def run_streaming_sgd_batch(
    d: int,
    sigma: float,
    gamma: float,
    w_star: np.ndarray,
    w0: np.ndarray,
    n_epochs: float,
    n_trials: int,
    rng: np.random.Generator,
):
    """n_trials independent streaming-SGD trajectories, vectorized over trials.
    Returns epochs t_k = k/d and excess risks R[trial, k] = (1/2)||w_k - w_*||^2."""
    n_steps = int(round(n_epochs * d))
    W = np.tile(w0, (n_trials, 1)).astype(float)
    R = np.empty((n_trials, n_steps + 1))
    diff = W - w_star
    R[:, 0] = 0.5 * np.sum(diff * diff, axis=1)
    for k in range(n_steps):
        X = rng.standard_normal((n_trials, d))
        eta = sigma * rng.standard_normal(n_trials)
        y = X @ w_star + eta
        pred = np.sum(W * X, axis=1)
        residual = pred - y
        W -= (gamma / d) * residual[:, None] * X
        diff = W - w_star
        R[:, k + 1] = 0.5 * np.sum(diff * diff, axis=1)
    epochs = np.arange(n_steps + 1) / d
    return epochs, R


def ode_solution(t: np.ndarray, gamma: float, sigma: float, R0: float):
    """Closed-form solution of dot psi = (gamma^2 - 2 gamma) psi + gamma^2 sigma^2/2."""
    a = gamma ** 2 - 2 * gamma
    if abs(a) < 1e-12:
        return R0 + 0.5 * gamma ** 2 * sigma ** 2 * t
    psi_inf = -0.5 * gamma ** 2 * sigma ** 2 / a
    return psi_inf + (R0 - psi_inf) * np.exp(a * t)


def panel_vary_gamma(ax, sigma, n_epochs, d, gammas, n_trials, rng_master):
    colors = plt.cm.viridis(np.linspace(0.15, 0.80, len(gammas)))
    t_grid = np.linspace(0, n_epochs, 600)
    for gamma, color in zip(gammas, colors):
        w_star = np.zeros(d)
        w_star[0] = 1.0
        w0 = np.zeros(d)
        rng = np.random.default_rng(rng_master.integers(0, 2 ** 31))
        epochs, R = run_streaming_sgd_batch(
            d, sigma, gamma, w_star, w0, n_epochs, n_trials, rng
        )
        med = np.median(R, axis=0)
        lo, hi = np.quantile(R, [0.1, 0.9], axis=0)
        ax.fill_between(epochs, lo, hi, color=color, alpha=0.28, linewidth=0)
        ax.plot(epochs, med, "-", color=color, linewidth=1.5,
                label=rf"streaming SGD, $\gamma={gamma}$")
        psi = ode_solution(t_grid, gamma, sigma, 0.5)
        ax.plot(t_grid, psi, "--", color=color, linewidth=1.5, alpha=0.9)
        psi_inf = 0.5 * gamma * sigma ** 2 / (2 - gamma)
        ax.axhline(psi_inf, color=color, linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlim(0, n_epochs)
    ax.set_xlabel(r"epoch $t = k/d$", fontsize=12)
    ax.set_ylabel(r"excess risk $L(w_{[td]}) - L(w_\ast)$", fontsize=12)
    ax.set_title(
        rf"Vary $\gamma$ at fixed $d={d}$ "
        rf"(dashed: ODE limit $\psi(t)$)",
        fontsize=11.5,
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9.0, loc="upper right", framealpha=0.95)


def panel_vary_d(ax, sigma, n_epochs, gamma, dims, n_trials, rng_master):
    colors = plt.cm.viridis(np.linspace(0.15, 0.80, len(dims)))
    t_grid = np.linspace(0, n_epochs, 600)
    psi = ode_solution(t_grid, gamma, sigma, 0.5)
    psi_inf = 0.5 * gamma * sigma ** 2 / (2 - gamma)
    for d, color in zip(dims, colors):
        w_star = np.zeros(d)
        w_star[0] = 1.0
        w0 = np.zeros(d)
        rng = np.random.default_rng(rng_master.integers(0, 2 ** 31))
        epochs, R = run_streaming_sgd_batch(
            d, sigma, gamma, w_star, w0, n_epochs, n_trials, rng
        )
        med = np.median(R, axis=0)
        lo, hi = np.quantile(R, [0.1, 0.9], axis=0)
        ax.fill_between(epochs, lo, hi, color=color, alpha=0.28, linewidth=0)
        ax.plot(epochs, med, "-", color=color, linewidth=1.6,
                label=rf"streaming SGD, $d={d}$")
        band_width = float(np.max((hi - lo)[epochs >= 2.0]))
        print(f"  d={d}: max band width (t>=2) = {band_width:.3e}")
    ax.plot(t_grid, psi, "--", color="black", linewidth=1.6,
            label=r"ODE limit $\psi(t)$")
    ax.axhline(psi_inf, color="black", linewidth=0.9, linestyle=":",
               alpha=0.7, label=r"$\psi_\infty$")
    ax.set_yscale("log")
    ax.set_xlim(0, n_epochs)
    ax.set_xlabel(r"epoch $t = k/d$", fontsize=12)
    ax.set_ylabel(r"excess risk $L(w_{[td]}) - L(w_\ast)$", fontsize=12)
    ax.set_title(
        rf"Fix $\gamma={gamma}$, vary $d$ "
        rf"(band $\sim 1/\sqrt{{d}}$)",
        fontsize=11.5,
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9.0, loc="upper right", framealpha=0.95)


def main():
    sigma = 0.1
    n_epochs = 8.0

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.0, 5.4))

    print("left panel (vary gamma, d=400):")
    panel_vary_gamma(
        ax_left, sigma, n_epochs, d=400, gammas=[0.5, 1.0, 1.5],
        n_trials=30, rng_master=np.random.default_rng(0),
    )
    print("right panel (fix gamma=1, vary d):")
    panel_vary_d(
        ax_right, sigma, n_epochs, gamma=1.0, dims=[50, 200, 800],
        n_trials=200, rng_master=np.random.default_rng(1),
    )

    fig.suptitle(
        rf"Streaming SGD on isotropic Gaussian regression "
        rf"($\sigma={sigma}$, $w_0=0$): concentration around the ODE limit",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIGURES_DIR / "sgd_high_d_panels.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
