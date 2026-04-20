"""
Run gradient descent and conjugate gradient on a synthetic diagonal quadratic
whose spectral error density is the power law

    phi(lambda) = M * lambda^{a-1}    on (0, beta],

and produce a log-log convergence plot showing f(x_k) - f* vs iteration k for
several exponents a, overlaid with the predicted asymptotic rates

    GD:  E_k ~ M Gamma(a+1) beta^{a+1} / (2 (2k)^{a+1})            (Thm 7.2)
    CG:  E_k ~ M Gamma(a+1) Gamma(a+2) beta^{a+1} / (2 k^{2(a+1)}) (Thm 7.6)

Setup. We work with a diagonal A = diag(lambda_1, ..., lambda_d) where the
eigenvalues lambda_i are the (i - 1/2)/d quantiles of the density
phi(lambda)/integral(phi) on (0, beta] (so the empirical eigenvalue
distribution is the natural quantile discretisation of the density), and
initial error coordinates c_i are chosen so that the discrete spectral
measure sum_i c_i^2 delta_{lambda_i} is the Riemann sum approximation of
phi(lambda) d lambda. Concretely we take c_i^2 = phi(lambda_i) * w_i where
w_i is the local quantile width.
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


def build_powerlaw_problem(a: float, beta: float, d: int, M: float = 1.0):
    """
    Build A = diag(lambda) and an initial error e0 (in eigenbasis = standard
    basis) such that the discrete spectral measure sum c_i^2 delta_{lambda_i}
    is the natural Riemann discretisation of phi(lambda) = M lambda^{a-1} on
    (0, beta].

    Eigenvalues placed at quantiles of phi/integral(phi) (which has CDF
    F(lambda) = (lambda/beta)^a). Quantile widths w_i used to weight c_i^2
    so that sum c_i^2 g(lambda_i) -> integral M lambda^{a-1} g(lambda) d lambda
    for any continuous g.
    """
    if a <= 0:
        raise ValueError("Use a > 0 so that integral(phi) is finite.")

    # Use cell-centre quantiles u_i = (i - 1/2)/d of the CDF F(lambda) =
    # (lambda/beta)^a. Inverse CDF gives lambda_i = beta * u_i^{1/a}.
    u = (np.arange(1, d + 1) - 0.5) / d
    lam = beta * u ** (1.0 / a)

    # Cell edges u_{i-1} = (i-1)/d, u_i = i/d so that widths w_i = lambda_i^{up}
    # - lambda_i^{lo} match the Riemann sum normalisation:
    #   sum_i M lambda_i^{a-1} (lambda_i^{up} - lambda_i^{lo})  ~ integral phi.
    edges = beta * (np.arange(d + 1) / d) ** (1.0 / a)
    widths = np.diff(edges)
    c_sq = M * lam ** (a - 1.0) * widths
    c = np.sqrt(c_sq)

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

    a_values = [0.5, 1.0, 1.5]
    colors = ["#2c7bb6", "#1a9641", "#d7191c"]

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    iters = np.arange(1, k_max + 1)

    for a, color in zip(a_values, colors):
        lam, c = build_powerlaw_problem(a=a, beta=beta, d=d, M=M)
        eta = 1.0 / beta
        gaps_gd, gaps_cg = run_gd_cg(lam, c, eta, k_max)

        # Truncate CG at machine precision floor to avoid the cliff.
        floor = 1e-22
        gd_mask = gaps_gd[1:] > floor
        cg_mask = gaps_cg[1:] > floor
        gd_vals = gaps_gd[1:]
        cg_vals = gaps_cg[1:]

        ax.loglog(iters[gd_mask], gd_vals[gd_mask], "-", color=color,
                  linewidth=1.8, alpha=0.95,
                  label=fr"GD, $a={a}$")
        ax.loglog(iters[cg_mask], cg_vals[cg_mask], "--", color=color,
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
        rf"$\phi(\lambda)=M\lambda^{{a-1}}$ on $(0,{beta:g}]$",
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
