"""
Compare GD, PSD-Chebyshev, and CG on quadratics whose spectrum
follows a power law: lambda_i = i^{-alpha}.

For PSD Chebyshev, each data point at iteration k uses the k-step
schedule eta_j = 1/(beta * sin^2(j*pi/(2k))), j=1..k, run from x_0.
The product polynomial is computed in log-space to avoid overflow.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def _logsumexp(a):
    a_max = np.max(a)
    return a_max + np.log(np.sum(np.exp(a - a_max)))


def gd_gaps(eigenvalues, c0_sq, beta, K):
    log_base = np.log(eigenvalues) + np.log(c0_sq)
    log_decay = np.log(np.abs(1.0 - eigenvalues / beta))
    gaps = np.empty(K + 1)
    gaps[0] = 0.5 * np.sum(eigenvalues * c0_sq)
    for k in range(1, K + 1):
        log_terms = log_base + 2 * k * log_decay
        gaps[k] = 0.5 * np.exp(_logsumexp(log_terms))
    return gaps


def chebyshev_psd_gaps(eigenvalues, c0_sq, beta, K):
    """For each horizon k = 1..K, compute f(x_k) - f* in log-space."""
    n = len(eigenvalues)
    log_base = np.log(eigenvalues) + np.log(c0_sq)
    gaps = np.empty(K + 1)
    gaps[0] = 0.5 * np.sum(eigenvalues * c0_sq)
    for k in range(1, K + 1):
        log_P_sq = np.zeros(n)
        for j in range(1, k + 1):
            s_sq = np.sin(j * np.pi / (2.0 * k)) ** 2
            ratio = eigenvalues / (beta * s_sq)
            log_P_sq += 2.0 * np.log(np.abs(1.0 - ratio))
        log_terms = log_base + log_P_sq
        gaps[k] = 0.5 * np.exp(_logsumexp(log_terms))
    return gaps


def cg_gaps(eigenvalues, c0, K):
    """CG on diagonal system with given spectrum and initial error coefficients."""
    d = len(eigenvalues)
    A = np.diag(eigenvalues)
    x_star = c0.copy()
    b = A @ x_star
    x = np.zeros(d)

    f_star = 0.5 * x_star @ A @ x_star - b @ x_star

    def fval(xx):
        return 0.5 * xx @ A @ xx - b @ xx

    r = b - A @ x
    p = r.copy()
    rr = r @ r
    gaps = np.empty(K + 1)
    gaps[0] = fval(x) - f_star
    for k in range(K):
        Ap = A @ p
        pAp = p @ Ap
        if pAp < 1e-30:
            gaps[k + 1:] = gaps[k]
            break
        alpha_k = rr / pAp
        x = x + alpha_k * p
        r_new = r - alpha_k * Ap
        rr_new = r_new @ r_new
        beta_k = rr_new / (rr + 1e-30)
        p = r_new + beta_k * p
        r = r_new
        rr = rr_new
        g = fval(x) - f_star
        gaps[k + 1] = max(g, 0.0)
    return gaps


def main():
    np.random.seed(42)
    d = 200
    K = 250

    alpha = 3.0
    c0 = np.ones(d)
    c0_sq = c0 ** 2

    eigenvalues = np.array([(i + 1.0) ** (-alpha) for i in range(d)])
    eigenvalues = np.sort(eigenvalues)[::-1]
    beta = eigenvalues[0]  # = 1

    g_gd = gd_gaps(eigenvalues, c0_sq, beta, K)
    g_ch = chebyshev_psd_gaps(eigenvalues, c0_sq, beta, K)
    g_cg = cg_gaps(eigenvalues, c0, K)

    iters = np.arange(K + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(iters, np.maximum(g_gd, 1e-16),
                 lw=1.8, ls="-", color="#1f77b4", label="GD")
    ax.semilogy(iters, np.maximum(g_ch, 1e-16),
                 lw=1.8, ls="--", color="#ff7f0e", label="PSD Chebyshev")
    ax.semilogy(iters, np.maximum(g_cg, 1e-16),
                 lw=1.8, ls=":", color="#2ca02c", label="CG")

    ax.set_xlabel("Iteration $k$", fontsize=13)
    ax.set_ylabel(r"$f(x_k) - f^\ast$", fontsize=13)
    ax.set_title(
        r"GD vs PSD Chebyshev vs CG: $\lambda_i = i^{-3}$, $d = 200$",
        fontsize=14,
    )
    ax.legend(fontsize=11, framealpha=0.9, loc="upper right")
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(0, K)
    fig.tight_layout()
    out = FIGURES_DIR / "gd_cheb_cg_psd.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    main()
