"""
Run gradient descent and conjugate gradient on a synthetic diagonal quadratic
whose spectral error density is the power law

    phi(lambda) = M * lambda^{a-1}    on (0, beta],

and produce a log-log convergence plot showing f(x_k) - f* vs iteration k for
several exponents a, overlaid with the predicted asymptotic rates

    GD:  E_k ~ M Gamma(a+1) beta^{a+1} / (2 (2k)^{a+1})            (Thm 7.2)
    CG:  E_k ~ M Gamma(a+1) Gamma(a+2) beta^{a+1} / (2 k^{2(a+1)}) (Thm 7.6)

Setup. For each trial we draw a fresh diagonal problem in the eigenbasis:

  * Eigenvalues lambda_i are sampled iid from the density rho = phi/integral(phi)
    on (0, beta]; concretely lambda_i = beta * U_i^{1/a} for U_i ~ Uniform(0,1).
  * Initial-error coordinates c_i are sampled iid as c_i = sigma * xi_i with
    xi_i ~ N(0, 1) and sigma^2 = M * beta^a / (a * d). This normalisation is
    chosen so that for every continuous test function g,
        E[sum_i c_i^2 g(lambda_i)] = integral M lambda^{a-1} g(lambda) dlambda,
    matching the Riemann discretisation used by the deterministic version of
    this experiment.

We then plot the median curve over n_trials draws and shade the 10%-90%
interquantile band (same convention as the MP gamma=1 figure).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import gamma as Gamma


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def sample_powerlaw_problem(
    a: float,
    beta: float,
    d: int,
    rng: np.random.Generator,
    M: float = 1.0,
):
    """
    Sample one random diagonal problem in the eigenbasis with the prescribed
    power-law spectral error density phi(lambda) = M lambda^{a-1} on (0, beta].

    Eigenvalues are drawn iid from rho = phi/integral(phi) (CDF
    F(lambda) = (lambda/beta)^a, inverse CDF lambda = beta * u^{1/a}).
    Coefficients c_i are drawn iid Gaussian with variance M beta^a / (a d), so
    that the discrete spectral measure sum_i c_i^2 delta_{lambda_i} converges
    in expectation to phi(lambda) d lambda as d -> infinity.
    """
    if a <= 0:
        raise ValueError("Use a > 0 so that integral(phi) is finite.")

    u = rng.uniform(size=d)
    lam = beta * u ** (1.0 / a)

    sigma2 = M * beta ** a / (a * d)
    c = np.sqrt(sigma2) * rng.standard_normal(d)

    return lam, c


def run_gd_cg(lam: np.ndarray, c: np.ndarray, eta: float, k_max: int):
    """
    Run GD with stepsize eta and CG on f(x) = (1/2) x^T diag(lam) x - b^T x
    where b = diag(lam) c, so x_star = c and starting from x_0 = 0 gives
    e_0 = -c. f_star = -(1/2) c^T diag(lam) c.

    Returns (gaps_gd, gaps_cg) of length k_max + 1.
    """
    d = len(lam)

    # GD via spectral formula: e_k = (1 - eta lam)^k * e_0,
    # so f(x_k) - f* = (1/2) sum_i lam_i (1 - eta lam_i)^{2k} c_i^2.
    ratios = (1.0 - eta * lam) ** 2
    weights_gd = 0.5 * lam * c ** 2
    gaps_gd = np.zeros(k_max + 1)
    power = np.ones_like(lam)
    for k in range(k_max + 1):
        gaps_gd[k] = float(weights_gd @ power)
        power *= ratios

    # CG: run the standard recurrence in the diagonal basis. b = lam * c gives
    # x_star = c. Start from x_0 = 0.
    b = lam * c
    x = np.zeros(d)
    r = b.copy()
    p = r.copy()
    f_star = -0.5 * float(c @ (lam * c))
    gaps_cg = np.zeros(k_max + 1)
    gaps_cg[0] = -f_star  # f(0) - f_star = 0 - f_star
    for k in range(k_max):
        Ap = lam * p
        rr = float(r @ r)
        denom = float(p @ Ap)
        if denom <= 0:
            gaps_cg[k + 1:] = gaps_cg[k]
            break
        alpha_cg = rr / denom
        x = x + alpha_cg * p
        r = r - alpha_cg * Ap
        gaps_cg[k + 1] = 0.5 * float(x @ (lam * x)) - float(b @ x) - f_star
        if gaps_cg[k + 1] < 1e-30:
            gaps_cg[k + 1:] = max(gaps_cg[k + 1], 1e-30)
            break
        beta_cg = float(r @ r) / rr
        p = r + beta_cg * p

    return gaps_gd, gaps_cg


def main():
    beta = 1.0
    M = 1.0
    d = 20000
    k_max = 800
    n_trials = 15
    rng = np.random.default_rng(0)

    a_values = [0.5, 1.0, 1.5]
    colors = ["#2c7bb6", "#1a9641", "#d7191c"]

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    iters = np.arange(1, k_max + 1)
    floor = 1e-22

    for a, color in zip(a_values, colors):
        gd_runs = np.zeros((n_trials, k_max + 1))
        cg_runs = np.zeros((n_trials, k_max + 1))
        eta = 1.0 / beta
        for t in range(n_trials):
            lam, c = sample_powerlaw_problem(a=a, beta=beta, d=d, rng=rng, M=M)
            gaps_gd, gaps_cg = run_gd_cg(lam, c, eta, k_max)
            gd_runs[t] = np.maximum(gaps_gd, floor)
            cg_runs[t] = np.maximum(gaps_cg, floor)
            print(f"  a={a:>3}  trial {t + 1}/{n_trials} done")

        gd_med = np.median(gd_runs, axis=0)
        cg_med = np.median(cg_runs, axis=0)
        gd_lo, gd_hi = np.quantile(gd_runs, [0.1, 0.9], axis=0)
        cg_lo, cg_hi = np.quantile(cg_runs, [0.1, 0.9], axis=0)

        gd_mask = gd_med[1:] > floor
        cg_mask = cg_med[1:] > floor

        ax.fill_between(iters[gd_mask], gd_lo[1:][gd_mask], gd_hi[1:][gd_mask],
                        color=color, alpha=0.12, linewidth=0)
        ax.loglog(iters[gd_mask], gd_med[1:][gd_mask], "-", color=color,
                  linewidth=1.8, alpha=0.95,
                  label=fr"GD, $a={a}$")

        ax.fill_between(iters[cg_mask], cg_lo[1:][cg_mask], cg_hi[1:][cg_mask],
                        color=color, alpha=0.12, linewidth=0)
        ax.loglog(iters[cg_mask], cg_med[1:][cg_mask], "--", color=color,
                  linewidth=1.8, alpha=0.95,
                  label=fr"CG, $a={a}$")

        # Predicted asymptotics (dotted reference lines).
        k_ref = np.logspace(np.log10(5), np.log10(k_max), 200)

        gd_const = M * Gamma(a + 1) * beta ** (a + 1) / 2.0
        ref_gd = gd_const / (2 * k_ref) ** (a + 1)
        ax.loglog(k_ref, ref_gd, ":", color=color, linewidth=1.1, alpha=0.7)

        cg_const = M * Gamma(a + 1) * Gamma(a + 2) * beta ** (a + 1) / 2.0
        ref_cg = cg_const / k_ref ** (2 * (a + 1))
        ax.loglog(k_ref, ref_cg, ":", color=color, linewidth=1.1, alpha=0.7)

    ax.set_xlabel("Iteration $k$", fontsize=12)
    ax.set_ylabel(r"$f(x_k) - f^\ast$", fontsize=12)
    ax.set_title(
        r"GD vs CG under power-law spectral density "
        rf"$\phi(\lambda)=M\lambda^{{a-1}}$ on $(0,{beta:g}]$"
        + f"  (median over {n_trials} trials)",
        fontsize=11.5,
    )

    # Add an explanatory annotation for the dotted lines.
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color="0.4", linestyle=":", linewidth=1.1))
    labels.append("predicted rate")
    ax.legend(handles, labels, fontsize=9.5, framealpha=0.9, ncol=2,
              loc="lower left")
    ax.grid(True, alpha=0.25, which="both")
    ax.set_ylim(1e-22, 5)
    ax.set_xlim(1, k_max)

    fig.tight_layout()
    out = FIGURES_DIR / "convergence_powerlaw.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


if __name__ == "__main__":
    main()
