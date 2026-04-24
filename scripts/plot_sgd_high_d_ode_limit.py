"""
Illustrate Theorem 9.3 (autonomous ODE limit of streaming SGD) on the
isotropic Gaussian linear regression model of Section 9.

Setup:
    x ~ N(0, I_d),   eta ~ N(0, sigma^2),   y = <w_*, x> + eta
    L(w) - L(w_*) = (1/2) ||w - w_*||^2

Streaming SGD with stepsize gamma/d:
    w_{k+1} = w_k - (gamma/d) * (<w_k, x_{k+1}> - y_{k+1}) * x_{k+1}.

ODE limit (44):
    dot psi = (gamma^2 - 2 gamma) psi + (gamma^2 sigma^2)/2,
    psi(0) = (1/2) ||w_0 - w_*||^2.

For several values of gamma we overlay streaming SGD (median + 10-90%
interquantile band over n_trials independent trajectories at a single
large d) against the deterministic ODE curve psi(t) and its stationary
value psi_inf = gamma sigma^2 / (2 (2 - gamma)).  The three regimes
gamma < 1, gamma = 1, gamma > 1 illustrate the bias-variance trade-off
in the limiting ODE: the decay rate 2 gamma - gamma^2 is maximized at
gamma = 1, whereas psi_inf increases monotonically in gamma on (0, 2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def run_streaming_sgd(
    d: int,
    sigma: float,
    gamma: float,
    w_star: np.ndarray,
    w0: np.ndarray,
    n_epochs: float,
    rng: np.random.Generator,
):
    """One trial of streaming SGD on isotropic Gaussian regression with
    stepsize gamma/d. Returns epochs t_k = k/d and excess risks
    R(w_k) = (1/2) ||w_k - w_*||^2 at every step."""
    n_steps = int(round(n_epochs * d))
    R = np.empty(n_steps + 1)
    w = w0.copy()
    diff = w - w_star
    R[0] = 0.5 * float(diff @ diff)
    for k in range(n_steps):
        x = rng.standard_normal(d)
        eta = sigma * rng.standard_normal()
        y = float(w_star @ x) + eta
        residual = float(w @ x) - y
        w -= (gamma / d) * residual * x
        diff = w - w_star
        R[k + 1] = 0.5 * float(diff @ diff)
    epochs = np.arange(n_steps + 1) / d
    return epochs, R


def ode_solution(t: np.ndarray, gamma: float, sigma: float, R0: float):
    """Closed-form solution of dot psi = (gamma^2 - 2 gamma) psi + gamma^2 sigma^2/2."""
    a = gamma ** 2 - 2 * gamma
    if abs(a) < 1e-12:
        return R0 + 0.5 * gamma ** 2 * sigma ** 2 * t
    psi_inf = -0.5 * gamma ** 2 * sigma ** 2 / a
    return psi_inf + (R0 - psi_inf) * np.exp(a * t)


def main():
    sigma = 0.1
    n_epochs = 8.0
    gammas = [0.5, 1.0, 1.5]
    d = 400
    n_trials = 30

    rng_master = np.random.default_rng(0)
    colors = plt.cm.viridis(np.linspace(0.15, 0.80, len(gammas)))

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    for gamma, color in zip(gammas, colors):
        w_star = np.zeros(d)
        w_star[0] = 1.0
        w0 = np.zeros(d)

        n_steps = int(round(n_epochs * d))
        runs = np.empty((n_trials, n_steps + 1))
        for trial in range(n_trials):
            rng = np.random.default_rng(rng_master.integers(0, 2 ** 31))
            epochs, R = run_streaming_sgd(d, sigma, gamma, w_star, w0, n_epochs, rng)
            runs[trial] = R

        med = np.median(runs, axis=0)
        lo, hi = np.quantile(runs, [0.1, 0.9], axis=0)

        ax.fill_between(epochs, lo, hi, color=color, alpha=0.28, linewidth=0)
        ax.plot(epochs, med, "-", color=color, linewidth=1.5,
                label=rf"streaming SGD, $\gamma={gamma}$")

        t_grid = np.linspace(0, n_epochs, 600)
        psi = ode_solution(t_grid, gamma, sigma, 0.5)
        ax.plot(t_grid, psi, "--", color=color, linewidth=1.5, alpha=0.9,
                label=rf"ODE limit $\psi(t)$, $\gamma={gamma}$")

        psi_inf = 0.5 * gamma * sigma ** 2 / (2 - gamma)
        ax.axhline(psi_inf, color=color, linewidth=0.8, linestyle=":", alpha=0.7)

        print(
            f"  gamma={gamma}, d={d}: {n_steps} steps x {n_trials} trials, "
            f"R_final median = {med[-1]:.3e}, psi_inf = {psi_inf:.3e}"
        )

    ax.set_xlabel(r"epoch $t = k/d$", fontsize=12)
    ax.set_ylabel(r"excess risk $L(w_{[td]}) - L(w_\ast)$", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(
        rf"Streaming SGD on isotropic Gaussian regression "
        rf"($\sigma={sigma}$, $w_0=0$, $d={d}$, "
        rf"median and 10-90% band over {n_trials} trials)",
        fontsize=11.5,
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95, ncol=2)
    ax.set_xlim(0, n_epochs)

    fig.tight_layout()
    out = FIGURES_DIR / "sgd_high_d_ode_limit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
