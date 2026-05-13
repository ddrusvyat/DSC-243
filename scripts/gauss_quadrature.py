"""Illustration of Gauss quadrature for Lemma 9.2 in week1.md.

Shows:
- a continuous measure mu on [0, 1] (here Beta(2, 4)-like density, non-uniform)
- its N-point Gauss approximation mu_N (nodes + weights)
- exactness: int P d mu = sum_j w_j P(theta_j) for any P of degree <= 2N-1

We compute the Gauss rule from scratch (Gram-Schmidt + Jacobi-matrix eigendecomposition),
so the script depends only on numpy and matplotlib.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")


# -- Density of mu on [0, 1] ----------------------------------------------
# A Beta(2, 4)-like density: rho(lambda) = 20 * lambda * (1 - lambda)^3
def rho(lam):
    return 20.0 * lam * (1.0 - lam) ** 3


# -- Compute the N-point Gauss rule for d mu = rho(lambda) d lambda --------
def gauss_rule_from_density(rho_fn, N, x_grid):
    """Return (nodes theta_j, weights w_j) for the N-point Gauss rule.

    Uses Gram-Schmidt on monomials in L^2(mu), reads three-term recurrence
    coefficients, and diagonalises the resulting Jacobi matrix.
    """
    rho_vals = rho_fn(x_grid)

    def inner(u, v):
        return np.trapezoid(u * v * rho_vals, x_grid)

    # Build orthonormal polynomial values p_0, ..., p_N on x_grid
    p_list = []
    p0 = np.ones_like(x_grid)
    p0 /= np.sqrt(inner(p0, p0))
    p_list.append(p0)

    for n in range(1, N + 1):
        pn = x_grid ** n
        for pk in p_list:
            pn = pn - inner(pn, pk) * pk
        pn /= np.sqrt(inner(pn, pn))
        p_list.append(pn)

    # Three-term recurrence: lambda p_n = beta_{n+1} p_{n+1} + alpha_n p_n + beta_n p_{n-1}
    alpha = np.array([inner(p_list[n], x_grid * p_list[n]) for n in range(N)])
    beta = np.array([inner(p_list[n + 1], x_grid * p_list[n]) for n in range(N - 1)])

    # Jacobi matrix; eigenvalues are nodes, weights from squared first eigenvector component
    J = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    eigvals, eigvecs = np.linalg.eigh(J)
    order = np.argsort(eigvals)
    theta = eigvals[order]
    weights = eigvecs[0, order] ** 2  # total mass of mu is 1
    return theta, weights


N = 5
x_grid = np.linspace(0.0, 1.0, 10001)
theta, w = gauss_rule_from_density(rho, N, x_grid)
print(f"nodes:  {theta}")
print(f"weights: {w}")
print(f"sum of weights (should = total mass of mu = 1): {w.sum():.10f}")


# -- Test polynomial of degree 2N-1 = 9 -----------------------------------
def P_test(lam):
    return lam ** 9 + (1.0 - lam) ** 9 + 0.4


# Reference integral by fine-grid numerical integration
int_mu = np.trapezoid(P_test(x_grid) * rho(x_grid), x_grid)
int_muN = float(np.sum(w * P_test(theta)))
print(f"int P d mu  (numerical) = {int_mu:.10f}")
print(f"int P d muN (Gauss)     = {int_muN:.10f}")
print(f"abs diff                = {abs(int_mu - int_muN):.2e}")


# -- Plot ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

# Panel 1: density of mu + atoms of mu_N
ax = axes[0]
rho_vals = rho(x_grid)
ax.fill_between(x_grid, 0, rho_vals, alpha=0.28, color="steelblue", label=r"density of $\mu$")
ax.plot(x_grid, rho_vals, color="steelblue", lw=2)

# Visual scale for stems: make max stem comparable to density peak
stem_scale = 0.9 * rho_vals.max() / w.max()
ml, sl, bl = ax.stem(
    theta,
    w * stem_scale,
    linefmt="crimson",
    markerfmt="o",
    basefmt="k-",
    label=r"atoms of $\mu_N$ (stem height $\propto w_j$)",
)
plt.setp(ml, markersize=9, color="crimson")
plt.setp(sl, linewidth=2.2)

# Node labels under x-axis
for j, t in enumerate(theta):
    ax.annotate(
        rf"$\theta_{{{j + 1}}}$",
        xy=(t, 0),
        xytext=(t, -0.18),
        ha="center",
        fontsize=10,
        color="crimson",
    )

ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.28, max(rho_vals.max(), (w * stem_scale).max()) * 1.18)
ax.set_xlabel(r"$\lambda$")
ax.set_yticks([])
ax.set_title(rf"$\mu$ vs $N={N}$-point Gauss rule $\mu_N = \sum_{{j=1}}^{{{N}}} w_j\,\delta_{{\theta_j}}$")
ax.legend(loc="upper right", fontsize=10, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Panel 2: a test polynomial and exactness
ax = axes[1]
P_vals = P_test(x_grid)
P_at_nodes = P_test(theta)

ax.plot(x_grid, P_vals, color="darkgreen", lw=2.2, label=r"$P(\lambda) \in \mathcal{P}_{2N-1}$")
ax.fill_between(x_grid, 0, P_vals, alpha=0.13, color="darkgreen")

# Drop lines from x-axis to (theta_j, P(theta_j))
for t, p in zip(theta, P_at_nodes):
    ax.vlines(t, 0, p, color="crimson", linestyle=":", linewidth=1.4, alpha=0.9)
ax.plot(theta, P_at_nodes, "o", markersize=10, color="crimson", label=r"$P(\theta_j)$")

ax.set_xlim(-0.03, 1.03)
ax.set_ylim(0, max(P_vals.max(), P_at_nodes.max()) * 1.18)
ax.set_xlabel(r"$\lambda$")
ax.set_yticks([])
ax.set_title(
    rf"$\int_0^1 P\,d\mu \;=\; \sum_{{j=1}}^{{{N}}} w_j\,P(\theta_j) \;=\; {int_muN:.5f}$"
)
ax.legend(loc="upper right", fontsize=10, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("figures/gauss_quadrature.png", dpi=160, bbox_inches="tight")
plt.close()

print("Saved figures/gauss_quadrature.png")
