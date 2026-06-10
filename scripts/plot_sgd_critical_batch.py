"""
Illustrate the critical batch size B_crit = Tr(H)/lambda_max for mini-batch
streaming SGD (Section 10, consequence 4 of the Volterra equation).

Model (same correlated-feature setup as the other Section 10 figures):
    H = Q diag(lambda_1, ..., lambda_d) Q^T,  lambda_i = i/d on (0, 1],
        Q Haar-random (simulated in the eigenbasis with a random unit signal).
    x ~ N(0, H),  y = <w_*, x> + sigma * eta,  ||w_*|| = 1,  w_0 = 0.
    Mini-batch streaming SGD with batch size B and stepsize gamma/d:
        w <- w - (gamma/d) * (1/B) sum_j (<w, x_j> - y_j) x_j.

Stepsize rule (run at a fixed fraction beta of the maximal stable stepsize):
    gamma(B) = beta * min( 2B/lambda_bar,   <- noise limit (kernel mass < 1)
                           2d/lambda_max )  <- curvature limit (discrete GD)
    The two limits cross at B_crit = d*lambda_bar/lambda_max = Tr(H)/lambda_max.
    For the ramp spectrum lambda_bar ~ 1/2, lambda_max = 1, so B_crit ~ d/2.

The figure shows ONE thing: the number of updates k_eps(B) needed to reach a
fixed excess risk eps, as a function of the batch size B (log-log). Below
B_crit the count falls like 1/B (linear speedup); beyond B_crit it plateaus
and matches the update count of full-batch gradient descent run at the
curvature-limited stepsize.
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
    """lambda_i = i/d for i = 1, ..., d (eigenvalues of H)."""
    return np.arange(1, d + 1, dtype=float) / d


def gamma_rule(B: int, lam: np.ndarray, d: int, beta: float) -> float:
    """gamma(B) = beta * min(noise limit, curvature limit)."""
    return beta * min(2.0 * B / float(lam.mean()), 2.0 * d / float(lam.max()))


def run_minibatch_sgd(d, lam, sigma, gamma, B, s, n_steps, rng):
    """One trial of mini-batch streaming SGD in the eigenbasis (signal s,
    start 0). Returns excess risks R[0..n_steps]."""
    sqrt_lam = np.sqrt(lam)
    v = np.zeros(d)
    R = np.empty(n_steps + 1)
    diff = v - s
    R[0] = 0.5 * float(lam @ (diff * diff))
    for k in range(n_steps):
        Z = rng.standard_normal((B, d))
        X = Z * sqrt_lam
        y = X @ s + sigma * rng.standard_normal(B)
        resid = X @ v - y
        v -= (gamma / d) * (X.T @ resid) / B
        diff = v - s
        R[k + 1] = 0.5 * float(lam @ (diff * diff))
    return R


def gd_reference(d, lam, gamma, n_steps):
    """Full-batch GD (B = infinity, no noise) with the limiting signal weights
    c_i^2 = 1/d: excess(k) = (1/2) sum_i lam_i (1 - gamma*lam_i/d)^{2k} / d."""
    contraction = (1.0 - (gamma / d) * lam) ** 2
    R = np.empty(n_steps + 1)
    cur = np.full(d, 1.0 / d)
    R[0] = 0.5 * float(lam @ cur)
    for k in range(n_steps):
        cur *= contraction
        R[k + 1] = 0.5 * float(lam @ cur)
    return R


def first_crossing(R: np.ndarray, eps: float):
    """First index k with R[k] <= eps, or None."""
    idx = np.nonzero(R <= eps)[0]
    return int(idx[0]) if idx.size else None


def main():
    d = 1024
    sigma = 0.1
    beta = 1.0 / 8.0
    eps = 3e-3
    n_trials = 8

    lam = ramp_spectrum(d)
    B_crit = lam.sum() / float(lam.max())     # Tr(H) / lambda_max ~ d/2
    gamma_cap = beta * 2.0 * d / float(lam.max())
    print(f"d={d}, B_crit={B_crit:.0f}, gamma_cap={gamma_cap:.0f}")

    rng_master = np.random.default_rng(7)

    B_grid = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    k_eps = []
    for B in B_grid:
        gamma = gamma_rule(B, lam, d, beta)
        n_steps = int(np.ceil(30.0 * d / gamma))
        runs = np.empty((n_trials, n_steps + 1))
        for t in range(n_trials):
            rng = np.random.default_rng(rng_master.integers(0, 2 ** 31))
            g = rng.standard_normal(d)
            s = g / np.linalg.norm(g)
            runs[t] = run_minibatch_sgd(d, lam, sigma, gamma, B, s, n_steps, rng)
        med = np.median(runs, axis=0)
        k = first_crossing(med, eps)
        k_eps.append(k)
        print(f"  B={B}: gamma={gamma:.1f}, k_eps={k}")

    n_gd = int(np.ceil(30.0 * d / gamma_cap))
    k_gd = first_crossing(gd_reference(d, lam, gamma_cap, n_gd), eps)
    print(f"  full-batch GD: k_eps={k_gd}")

    Bs = np.array(B_grid, dtype=float)
    ks = np.array([np.nan if k is None else k for k in k_eps])

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.loglog(Bs, ks, "o-", color="tab:blue", linewidth=1.8, markersize=6,
              label=r"mini-batch SGD: updates to reach $\varepsilon$")
    ax.loglog(Bs, ks[0] * Bs[0] / Bs, "--", color="tab:blue", alpha=0.45,
              linewidth=1.4, label=r"linear speedup (slope $-1$)")
    ax.axhline(k_gd, color="black", linestyle=":", linewidth=1.6,
               label="full-batch gradient descent")
    ax.axvline(B_crit, color="gray", linewidth=1.4, alpha=0.8)
    ax.text(B_crit * 0.88, 3.0,
            r"$B_{\mathrm{crit}} = \operatorname{Tr}H/\lambda_{\max}$",
            color="gray", fontsize=11, rotation=90, va="bottom", ha="right")
    ax.set_xlabel("batch size $B$", fontsize=12)
    ax.set_ylabel(rf"updates to reach excess risk $\varepsilon = {eps}$",
                  fontsize=12)
    ax.set_title(
        rf"Batch size saturation of streaming SGD "
        rf"($H = Q\Lambda Q^\top$, $\lambda_i = i/d$, $d = {d}$)",
        fontsize=11.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)

    fig.tight_layout()
    out = FIGURES_DIR / "sgd_critical_batch.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
