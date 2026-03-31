"""
Generate GIF animations for Week 1 lecture notes:
  1. GD on well-conditioned vs ill-conditioned quadratic
  2. Fixed stepsize GD vs Chebyshev stepsizes
  3. Convergence rate comparison (GD vs Chebyshev vs CG)
  4. Conjugate gradient directions on a 2D quadratic
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import math

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ── Quadratic helpers ──────────────────────────────────────────────

def make_quadratic(eigenvalues):
    """Return (A, f, grad) for f(x) = 0.5 * x^T A x with given eigenvalues."""
    theta = 0.4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    A = R @ np.diag(eigenvalues) @ R.T
    f = lambda x: 0.5 * x @ A @ x
    grad = lambda x: A @ x
    return A, f, grad


def gd_iterates(grad, x0, stepsize, n_iters):
    xs = [x0.copy()]
    x = x0.copy()
    for _ in range(n_iters):
        x = x - stepsize * grad(x)
        xs.append(x.copy())
    return np.array(xs)


def chebyshev_stepsizes(L, mu, n):
    """Chebyshev stepsize schedule for a quadratic with eigenvalues in [mu, L]."""
    kappa = L / mu
    stepsizes = []
    for k in range(1, n + 1):
        cos_val = np.cos((2 * k - 1) * np.pi / (2 * n))
        lam_k = mu + 0.5 * (L - mu) * (1 + cos_val)
        stepsizes.append(1.0 / lam_k)
    return stepsizes


def gd_chebyshev_iterates(grad, x0, L, mu, n_cycles, cycle_len):
    xs = [x0.copy()]
    x = x0.copy()
    for _ in range(n_cycles):
        steps = chebyshev_stepsizes(L, mu, cycle_len)
        for s in steps:
            x = x - s * grad(x)
            xs.append(x.copy())
    return np.array(xs)


def cg_iterates(A, x0, n_iters):
    """CG on f(x) = 0.5 x^T A x (i.e., solving Ax = 0 from x0)."""
    xs = [x0.copy()]
    x = x0.copy()
    r = -A @ x  # r = b - Ax, with b = 0
    p = r.copy()
    for _ in range(n_iters):
        Ap = A @ p
        rr = r @ r
        if rr < 1e-30:
            break
        alpha = rr / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new @ r_new) / rr
        p = r_new + beta * p
        r = r_new
        xs.append(x.copy())
    return np.array(xs)


def contour_grid(A, xs_list, pad=1.5):
    all_pts = np.vstack(xs_list)
    xmin, xmax = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    ymin, ymax = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200),
                         np.linspace(ymin, ymax, 200))
    pts = np.stack([xx, yy], axis=-1)
    zz = 0.5 * np.einsum("...i,ij,...j", pts, A, pts)
    return xx, yy, zz


# ── GIF 1: Well-conditioned vs ill-conditioned ────────────────────

def make_gif_condition():
    eigs_good = [1.0, 1.5]
    eigs_bad = [0.2, 10.0]
    A_good, _, grad_good = make_quadratic(eigs_good)
    A_bad, _, grad_bad = make_quadratic(eigs_bad)

    x0 = np.array([4.0, 4.0])
    n_iters = 40

    step_good = 2.0 / (eigs_good[0] + eigs_good[1])
    step_bad = 2.0 / (eigs_bad[0] + eigs_bad[1])

    xs_good = gd_iterates(grad_good, x0, step_good, n_iters)
    xs_bad = gd_iterates(grad_bad, x0, step_bad, n_iters)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    xx1, yy1, zz1 = contour_grid(A_good, [xs_good])
    xx2, yy2, zz2 = contour_grid(A_bad, [xs_bad])

    levels1 = np.linspace(0, zz1.max() * 0.8, 20)
    levels2 = np.linspace(0, zz2.max() * 0.8, 20)

    ax1.contour(xx1, yy1, zz1, levels=levels1, cmap="Blues", alpha=0.6)
    ax2.contour(xx2, yy2, zz2, levels=levels2, cmap="Reds", alpha=0.6)

    kappa_good = max(eigs_good) / min(eigs_good)
    kappa_bad = max(eigs_bad) / min(eigs_bad)
    ax1.set_title(f"Well-conditioned ($\\kappa = {kappa_good:.1f}$)", fontsize=13)
    ax2.set_title(f"Ill-conditioned ($\\kappa = {kappa_bad:.0f}$)", fontsize=13)

    line1, = ax1.plot([], [], "o-", color="royalblue", markersize=4, linewidth=1.5)
    dot1, = ax1.plot([], [], "o", color="navy", markersize=8)
    line2, = ax2.plot([], [], "o-", color="indianred", markersize=4, linewidth=1.5)
    dot2, = ax2.plot([], [], "o", color="darkred", markersize=8)

    ax1.plot(0, 0, "*", color="gold", markersize=15, zorder=5)
    ax2.plot(0, 0, "*", color="gold", markersize=15, zorder=5)

    for ax in (ax1, ax2):
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")

    fig.suptitle("Gradient Descent: Effect of Condition Number", fontsize=14, y=1.02)
    fig.tight_layout()

    def update(i):
        k = min(i, len(xs_good) - 1)
        line1.set_data(xs_good[:k+1, 0], xs_good[:k+1, 1])
        dot1.set_data([xs_good[k, 0]], [xs_good[k, 1]])
        k2 = min(i, len(xs_bad) - 1)
        line2.set_data(xs_bad[:k2+1, 0], xs_bad[:k2+1, 1])
        dot2.set_data([xs_bad[k2, 0]], [xs_bad[k2, 1]])
        return line1, dot1, line2, dot2

    anim = FuncAnimation(fig, update, frames=n_iters + 5, interval=120, blit=True)
    anim.save(FIGURES_DIR / "gd_condition.gif", writer=PillowWriter(fps=8))
    plt.close(fig)
    print("  ✓ gd_condition.gif")


# ── GIF 2: GD vs Chebyshev ────────────────────────────────────────

def make_gif_chebyshev():
    eigs = [0.3, 8.0]
    A, _, grad = make_quadratic(eigs)
    L, mu = max(eigs), min(eigs)

    x0 = np.array([5.0, 4.0])
    n_iters = 60
    cycle_len = 8
    n_cycles = n_iters // cycle_len

    step_fixed = 2.0 / (L + mu)
    xs_gd = gd_iterates(grad, x0, step_fixed, n_iters)
    xs_cheb = gd_chebyshev_iterates(grad, x0, L, mu, n_cycles, cycle_len)

    fig, ax = plt.subplots(figsize=(8, 6))
    xx, yy, zz = contour_grid(A, [xs_gd, xs_cheb])
    levels = np.linspace(0, zz.max() * 0.7, 25)
    ax.contour(xx, yy, zz, levels=levels, cmap="Greys", alpha=0.5)

    line_gd, = ax.plot([], [], "o-", color="royalblue", markersize=3, linewidth=1.2, label="GD (fixed step)")
    dot_gd, = ax.plot([], [], "o", color="navy", markersize=7)
    line_ch, = ax.plot([], [], "s-", color="orangered", markersize=3, linewidth=1.2, label="Chebyshev steps")
    dot_ch, = ax.plot([], [], "s", color="darkred", markersize=7)

    ax.plot(0, 0, "*", color="gold", markersize=15, zorder=5)
    ax.set_title("GD with Fixed Stepsize vs. Chebyshev Stepsizes", fontsize=13)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_aspect("equal")
    fig.tight_layout()

    n_frames = max(len(xs_gd), len(xs_cheb))

    def update(i):
        k1 = min(i, len(xs_gd) - 1)
        line_gd.set_data(xs_gd[:k1+1, 0], xs_gd[:k1+1, 1])
        dot_gd.set_data([xs_gd[k1, 0]], [xs_gd[k1, 1]])
        k2 = min(i, len(xs_cheb) - 1)
        line_ch.set_data(xs_cheb[:k2+1, 0], xs_cheb[:k2+1, 1])
        dot_ch.set_data([xs_cheb[k2, 0]], [xs_cheb[k2, 1]])
        return line_gd, dot_gd, line_ch, dot_ch

    anim = FuncAnimation(fig, update, frames=n_frames + 5, interval=120, blit=True)
    anim.save(FIGURES_DIR / "gd_vs_chebyshev.gif", writer=PillowWriter(fps=8))
    plt.close(fig)
    print("  ✓ gd_vs_chebyshev.gif")


# ── GIF 3: Convergence comparison ─────────────────────────────────

def make_gif_convergence():
    eigs = [0.3, 8.0]
    A, f, grad = make_quadratic(eigs)
    L, mu = max(eigs), min(eigs)

    x0 = np.array([5.0, 4.0])
    n_iters = 60
    cycle_len = 8

    xs_gd = gd_iterates(grad, x0, 2.0 / (L + mu), n_iters)
    xs_cheb = gd_chebyshev_iterates(grad, x0, L, mu, n_iters // cycle_len, cycle_len)
    xs_cg = cg_iterates(A, x0, n_iters)

    f0 = f(x0)
    err_gd = np.array([f(x) / f0 for x in xs_gd])
    err_cheb = np.array([f(x) / f0 for x in xs_cheb])
    err_cg = np.array([f(x) / f0 for x in xs_cg])

    # Clamp to avoid log(0)
    floor = 1e-16
    err_gd = np.maximum(err_gd, floor)
    err_cheb = np.maximum(err_cheb, floor)
    err_cg = np.maximum(err_cg, floor)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, n_iters)
    ax.set_ylim(1e-15, 2)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(r"$(f(x_k)-f(x^\star))/(f(x_0)-f(x^\star))$", fontsize=12)
    ax.set_title("Convergence Comparison", fontsize=13)
    ax.grid(True, alpha=0.3)

    line_gd, = ax.plot([], [], "-", color="royalblue", linewidth=2, label="Gradient Descent")
    line_ch, = ax.plot([], [], "-", color="orangered", linewidth=2, label="Chebyshev GD")
    line_cg, = ax.plot([], [], "-", color="seagreen", linewidth=2, label="Conjugate Gradient")
    ax.legend(fontsize=11)
    fig.tight_layout()

    n_frames = n_iters + 10

    def update(i):
        k = min(i + 1, n_iters + 1)
        k_gd = min(k, len(err_gd))
        k_ch = min(k, len(err_cheb))
        k_cg = min(k, len(err_cg))
        line_gd.set_data(np.arange(k_gd), err_gd[:k_gd])
        line_ch.set_data(np.arange(k_ch), err_cheb[:k_ch])
        line_cg.set_data(np.arange(k_cg), err_cg[:k_cg])
        return line_gd, line_ch, line_cg

    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)
    anim.save(FIGURES_DIR / "convergence_comparison.gif", writer=PillowWriter(fps=10))
    plt.close(fig)
    print("  ✓ convergence_comparison.gif")


# ── GIF 4: Conjugate Gradient directions ──────────────────────────

def make_gif_cg():
    eigs = [0.5, 6.0]
    A, _, _ = make_quadratic(eigs)

    x0 = np.array([5.0, 4.0])
    n_iters = 3  # CG on 2D converges in 2 steps; show a few

    # Run CG manually to capture search directions
    xs = [x0.copy()]
    directions = []
    x = x0.copy()
    r = -A @ x
    p = r.copy()
    for _ in range(n_iters):
        Ap = A @ p
        rr = r @ r
        if rr < 1e-30:
            break
        alpha = rr / (p @ Ap)
        directions.append((x.copy(), alpha * p))
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new @ r_new) / rr
        p = r_new + beta * p
        r = r_new
        xs.append(x.copy())

    xs = np.array(xs)

    fig, ax = plt.subplots(figsize=(8, 6))
    xx, yy, zz = contour_grid(A, [xs], pad=2.0)
    levels = np.linspace(0, zz.max() * 0.7, 25)
    ax.contour(xx, yy, zz, levels=levels, cmap="Greys", alpha=0.5)
    ax.plot(0, 0, "*", color="gold", markersize=15, zorder=5)
    ax.set_title("Conjugate Gradient Method", fontsize=13)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    fig.tight_layout()

    line, = ax.plot([], [], "o-", color="seagreen", markersize=6, linewidth=2)
    dot, = ax.plot([], [], "o", color="darkgreen", markersize=10)
    arrows = []

    total_steps = len(xs) - 1
    frames_per_step = 15
    pause_frames = 8
    total_frames = total_steps * (frames_per_step + pause_frames) + pause_frames

    def update(frame):
        for arr in arrows:
            arr.remove()
        arrows.clear()

        step = min(frame // (frames_per_step + pause_frames), total_steps)
        sub = frame % (frames_per_step + pause_frames)
        t = min(sub / frames_per_step, 1.0)

        show_pts = list(xs[:step])
        if step < total_steps:
            interp = xs[step] + t * (xs[step + 1] - xs[step])
            show_pts.append(interp)
        else:
            show_pts.append(xs[step])

        pts = np.array(show_pts)
        line.set_data(pts[:, 0], pts[:, 1])
        dot.set_data([pts[-1, 0]], [pts[-1, 1]])

        for i in range(min(step + 1, len(directions))):
            if i < step or (i == step and t >= 0.3):
                origin = xs[i]
                d = directions[i]
                scale = min(t, 1.0) if i == step else 1.0
                arr = ax.annotate("", xy=origin + scale * d[1],
                                  xytext=origin,
                                  arrowprops=dict(arrowstyle="->", color="tomato",
                                                  lw=2.0))
                arrows.append(arr)

        return line, dot

    anim = FuncAnimation(fig, update, frames=total_frames, interval=80, blit=False)
    anim.save(FIGURES_DIR / "conjugate_gradient.gif", writer=PillowWriter(fps=12))
    plt.close(fig)
    print("  ✓ conjugate_gradient.gif")


# ── Plot 5: Chebyshev polynomials of various degrees ──────────────

def make_chebyshev_plot():
    x = np.linspace(-1, 1, 500)
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad", "#e67e22"]
    for k, c in zip(range(1, 6), colors):
        y = np.cos(k * np.arccos(x))
        ax.plot(x, y, color=c, linewidth=2, label=f"$T_{k}$")

    ax.axhline(1, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(-1, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("$x$", fontsize=13)
    ax.set_ylabel("$T_k(x)$", fontsize=13)
    ax.set_title("Chebyshev Polynomials of the First Kind", fontsize=14)
    ax.legend(fontsize=12, loc="lower right", ncol=3)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "chebyshev_polynomials.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ chebyshev_polynomials.png")


# ── GIF 6: Optimal polynomials p_k^* on [alpha, beta] ─────────────

def make_gif_optimal_poly():
    alpha, beta = 1.0, 50.0
    kappa = beta / alpha
    sigma = (kappa + 1) / (kappa - 1)
    k_max = 15

    lam = np.linspace(alpha, beta, 500)
    t = (beta + alpha - 2 * lam) / (beta - alpha)

    fig, (ax_poly, ax_env) = plt.subplots(1, 2, figsize=(13, 5),
                                          gridspec_kw={"width_ratios": [3, 2]})

    ax_poly.set_xlim(alpha - 0.3, beta + 0.3)
    ax_poly.set_ylim(-1.15, 1.15)
    ax_poly.set_xlabel("$\\lambda$", fontsize=13)
    ax_poly.set_ylabel("$p_k^*(\\lambda)$", fontsize=13)
    ax_poly.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
    ax_poly.axvline(alpha, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
    ax_poly.axvline(beta, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
    ax_poly.grid(True, alpha=0.2)

    ax_env.set_xlim(0, k_max + 1)
    ax_env.set_ylim(1e-7, 2)
    ax_env.set_yscale("log")
    ax_env.set_xlabel("Degree $k$", fontsize=13)
    ax_env.set_ylabel("$\\max_{\\lambda} \\, |p_k^*(\\lambda)|$", fontsize=13)
    ax_env.grid(True, alpha=0.2)

    ks_all = np.arange(1, k_max + 1)
    env_all = 1.0 / np.cosh(ks_all * np.arccosh(sigma))
    ax_env.plot(ks_all, env_all, "o--", color="grey", alpha=0.3, markersize=4, linewidth=1)

    cheb_rate = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** ks_all
    ax_env.plot(ks_all, 2 * cheb_rate, ":", color="grey", alpha=0.4, linewidth=1,
                label="$2\\rho_{\\rm Cheb}^k$")

    curve, = ax_poly.plot([], [], linewidth=2.2, color="#2980b9")
    env_line, = ax_env.plot([], [], "o-", color="#e74c3c", markersize=6, linewidth=2)
    title = fig.suptitle("", fontsize=13)

    ax_poly.set_title(
        f"$p_k^*(\\lambda)$ on "
        f"$[\\alpha, \\beta] = [{alpha:.0f},\\,{beta:.0f}]$"
        f"  ($\\kappa = {kappa:.0f}$)",
        fontsize=13)
    ax_env.set_title("Max amplitude", fontsize=13)
    ax_env.legend(fontsize=10, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    hold = 10

    def update(frame):
        k = frame // hold + 1
        Tk_sigma = np.cosh(k * np.arccosh(sigma))
        pk = np.cos(k * np.arccos(np.clip(t, -1, 1))) / Tk_sigma

        curve.set_data(lam, pk)
        curve.set_color(plt.cm.viridis(k / k_max))

        env_line.set_data(ks_all[:k], env_all[:k])
        title.set_text(f"$k = {k}$,  $\\max |p_k^*| = {1/Tk_sigma:.4f}$")
        return curve, env_line, title

    anim = FuncAnimation(fig, update, frames=k_max * hold, interval=100, blit=False)
    anim.save(FIGURES_DIR / "optimal_polynomials.gif",
              writer=PillowWriter(fps=hold))
    plt.close(fig)
    print("  ✓ optimal_polynomials.gif")


# ── Plot: Chebyshev polynomials of the second kind ─────────────────

def make_chebyshev2_plot():
    x = np.linspace(-1, 1, 500)
    theta = np.arccos(x)
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad", "#e67e22"]
    for k, c in zip(range(1, 6), colors):
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.sin((k + 1) * theta) / np.sin(theta)
        y[0] = k + 1
        y[-1] = (-1)**k * (k + 1)
        ax.plot(x, y, color=c, linewidth=2, label=f"$U_{k}$")

    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-6.5, 6.5)
    ax.set_xlabel("$x$", fontsize=13)
    ax.set_ylabel("$U_k(x)$", fontsize=13)
    ax.set_title("Chebyshev Polynomials of the Second Kind", fontsize=14)
    ax.legend(fontsize=12, loc="lower right", ncol=3)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "chebyshev_polynomials_2nd.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ chebyshev_polynomials_2nd.png")


def make_chebyshev_stepsizes_pd_plot():
    alpha, beta = 1.0, 50.0
    ks = [8, 16, 32]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = ["#2980b9", "#e67e22", "#27ae60"]

    for k, c in zip(ks, colors):
        j = np.arange(1, k + 1)
        lam_j = 0.5 * (beta + alpha) - 0.5 * (beta - alpha) * np.cos((2 * j - 1) * np.pi / (2 * k))
        eta_j = 1.0 / lam_j
        ax.plot(j, eta_j, "o-", color=c, linewidth=1.8, markersize=3.5, label=fr"$k={k}$")

    ax.set_xlabel("Index $j$")
    ax.set_ylabel(r"Stepsize $\eta_j$")
    ax.set_title(r"Chebyshev Stepsizes (PD): $\eta_j = 1/\lambda_j$")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=11, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "chebyshev_stepsizes_pd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ chebyshev_stepsizes_pd.png")


def make_chebyshev_stepsizes_psd_plot():
    beta = 50.0
    ks = [8, 16, 32]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = ["#8e44ad", "#c0392b", "#16a085"]

    for k, c in zip(ks, colors):
        j = np.arange(1, k + 1)
        eta_j = 1.0 / (beta * np.sin(j * np.pi / (2 * k)) ** 2)
        ax.plot(j, eta_j, "o-", color=c, linewidth=1.8, markersize=3.5, label=fr"$k={k}$")

    ax.set_yscale("log")
    ax.set_xlabel("Index $j$")
    ax.set_ylabel(r"Stepsize $\eta_j$")
    ax.set_title(r"Chebyshev Stepsizes (PSD): $\eta_j = 1/(\beta\sin^2(j\pi/(2k)))$")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=11, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "chebyshev_stepsizes_psd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ chebyshev_stepsizes_psd.png")


# ── Section 7 visuals: spectral structure ──────────────────────────

def _source_bound_psd(beta, s, k_vals):
    k = np.asarray(k_vals, dtype=float)
    return 0.5 * (beta ** (1 + 2 * s)) * ((1 + 2 * s) / (2 * k + 1 + 2 * s)) ** (1 + 2 * s)


def _max_source_bound_pd(alpha, beta, s, k_vals):
    k = np.asarray(k_vals, dtype=float)
    lam_star = beta * (1 + 2 * s) / (2 * k + 1 + 2 * s)
    lam_opt = np.maximum(alpha, lam_star)
    return (lam_opt ** (1 + 2 * s)) * (1 - lam_opt / beta) ** (2 * k)


def _theorem9_curve(a, s, k_vals, beta=1.0, mass=1.0):
    q = a + 2 * s + 1
    out = []
    for k in k_vals:
        # Theorem 9 exact prefactor with Beta-function ratio.
        log_val = (
            math.log(mass / 2.0)
            + q * math.log(beta)
            + math.lgamma(q)
            + math.lgamma(2 * k + 1)
            - math.lgamma(2 * k + q + 1)
        )
        out.append(math.exp(log_val))
    return np.array(out)


def _mp_density(lam, gamma):
    lam = np.asarray(lam)
    lam_minus = (1 - np.sqrt(gamma)) ** 2
    lam_plus = (1 + np.sqrt(gamma)) ** 2
    inside = np.maximum((lam_plus - lam) * (lam - lam_minus), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.sqrt(inside) / (2 * np.pi * gamma * lam)
    rho[(lam < lam_minus) | (lam > lam_plus)] = 0.0
    rho[~np.isfinite(rho)] = 0.0
    return rho


def make_source_condition_rates_plot():
    beta = 1.0
    k_vals = np.arange(1, 401)
    s_vals = [0.0, 0.5, 1.0]
    colors = ["#2980b9", "#e74c3c", "#27ae60"]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for s, color in zip(s_vals, colors):
        y = _source_bound_psd(beta=beta, s=s, k_vals=k_vals)
        y = y / y[0]
        ax.plot(k_vals, y, color=color, linewidth=2.2, label=fr"$s={s:g}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration $k$", fontsize=12)
    ax.set_ylabel("Normalized upper bound", fontsize=12)
    ax.set_title("Source Condition Improves PSD Gradient Descent Rates", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=11)
    ax.text(4, 1.4e-2, r"Slope $\approx -(1+2s)$", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source_condition_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ source_condition_rates.png")


def make_source_condition_phase_transition_plot():
    beta = 1.0
    kappa = 50.0
    alpha = beta / kappa
    s = 0.5
    k_vals = np.arange(1, 420)

    exact = _max_source_bound_pd(alpha=alpha, beta=beta, s=s, k_vals=k_vals)
    sublinear = (beta ** (1 + 2 * s)) * ((1 + 2 * s) / (2 * k_vals + 1 + 2 * s)) ** (1 + 2 * s)
    linear = (alpha ** (1 + 2 * s)) * (1 - alpha / beta) ** (2 * k_vals)
    k_trans = 0.5 * (1 + 2 * s) * (kappa - 1)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(k_vals, exact, color="#2c3e50", linewidth=2.5, label="Exact maximized bound")
    ax.plot(k_vals, sublinear, "--", color="#2980b9", linewidth=2, label="Interior sublinear envelope")
    ax.plot(k_vals, linear, "--", color="#e74c3c", linewidth=2, label="Endpoint linear envelope")
    ax.axvline(k_trans, color="#8e44ad", linestyle=":", linewidth=2, label=r"$k_{\rm trans}$")

    ax.set_yscale("log")
    ax.set_xlabel("Iteration $k$", fontsize=12)
    ax.set_ylabel("Upper bound (unnormalized)", fontsize=12)
    ax.set_title(r"Phase Transition for $\alpha>0$: Sublinear to Linear", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source_condition_phase_transition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ source_condition_phase_transition.png")


def make_power_law_density_rates_plot():
    k_vals = np.arange(1, 450)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # Left panel: vary spectral exponent a with fixed source order s=0.
    a_vals = [0.5, 1.0, 2.0]
    colors_left = ["#2980b9", "#e67e22", "#27ae60"]
    for a, color in zip(a_vals, colors_left):
        y = _theorem9_curve(a=a, s=0.0, k_vals=k_vals)
        y = y / y[0]
        ax1.plot(k_vals, y, color=color, linewidth=2.2, label=fr"$a={a:g}$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration $k$")
    ax1.set_ylabel("Normalized error proxy")
    ax1.set_title("Power-Law Spectrum: Varying $a$ (with $s=0$)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=10)

    # Right panel: vary source order s with fixed spectral exponent a=1.
    s_vals = [0.0, 0.5, 1.0]
    colors_right = ["#8e44ad", "#c0392b", "#16a085"]
    for s, color in zip(s_vals, colors_right):
        y = _theorem9_curve(a=1.0, s=s, k_vals=k_vals)
        y = y / y[0]
        ax2.plot(k_vals, y, color=color, linewidth=2.2, label=fr"$s={s:g}$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Iteration $k$")
    ax2.set_ylabel("Normalized error proxy")
    ax2.set_title("Power-Law Spectrum: Varying $s$ (with $a=1$)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=10)

    fig.suptitle("Theorem 9 Rate Exponents Add: $a + 2s + 1$", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "power_law_density_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ power_law_density_rates.png")


def make_marchenko_pastur_regimes_plot():
    gammas = [0.5, 1.0, 2.0]
    titles = [r"$\gamma<1$", r"$\gamma=1$", r"$\gamma>1$"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)

    for ax, gamma, title in zip(axes, gammas, titles):
        lam_minus = (1 - np.sqrt(gamma)) ** 2
        lam_plus = (1 + np.sqrt(gamma)) ** 2
        lam_left = max(lam_minus + 1e-3, 1e-3)
        lam = np.linspace(lam_left, lam_plus, 700)
        rho = _mp_density(lam, gamma)

        ax.plot(lam, rho, color="#2c3e50", linewidth=2.2)
        ax.fill_between(lam, rho, color="#95a5a6", alpha=0.25)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"$\lambda$")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, lam_plus * 1.05)

        if gamma > 1:
            atom_mass = 1.0 - 1.0 / gamma
            ax.vlines(0, 0, atom_mass, color="#e74c3c", linewidth=3)
            ax.plot([0], [atom_mass], "o", color="#e74c3c", markersize=5)
            ax.text(0.03 * lam_plus, atom_mass * 0.92, r"atom at $0$", color="#c0392b", fontsize=10)

    axes[0].set_ylabel(r"Density $\rho_{MP}(\lambda)$")
    fig.suptitle("Marchenko--Pastur Spectral Regimes", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "marchenko_pastur_regimes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ marchenko_pastur_regimes.png")


def make_mp_convergence_regimes_gif():
    k_max = 260
    k_vals = np.arange(1, k_max + 1)
    gammas = [0.5, 1.0, 2.0]
    labels = [r"$\gamma=0.5$ (PD)", r"$\gamma=1$ (critical)", r"$\gamma=2$ (rank-def.)"]
    colors = ["#2980b9", "#e67e22", "#27ae60"]

    def mp_rate(gamma):
        poly = k_vals.astype(float) ** (-1.5)
        if abs(gamma - 1.0) < 1e-12:
            y = poly
        else:
            if gamma < 1:
                alpha = (1 - np.sqrt(gamma)) ** 2
                beta = (1 + np.sqrt(gamma)) ** 2
            else:
                alpha = (np.sqrt(gamma) - 1) ** 2
                beta = (np.sqrt(gamma) + 1) ** 2
            r = 1.0 - alpha / beta
            y = poly * (r ** (2 * k_vals))
        return y / y[0]

    curves = [mp_rate(g) for g in gammas]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.set_yscale("log")
    ax.set_xlim(1, k_max)
    ax.set_ylim(1e-15, 1.2)
    ax.set_xlabel("Iteration $k$", fontsize=12)
    ax.set_ylabel("Normalized asymptotic proxy", fontsize=12)
    ax.set_title("Marchenko--Pastur Convergence Regimes", fontsize=13)
    ax.grid(True, alpha=0.25)

    lines = []
    for color, label in zip(colors, labels):
        line, = ax.plot([], [], color=color, linewidth=2.2, label=label)
        lines.append(line)
    ax.legend(fontsize=10, loc="upper right")

    def update(frame):
        k = min(frame + 1, k_max)
        x = k_vals[:k]
        for line, y in zip(lines, curves):
            line.set_data(x, y[:k])
        return tuple(lines)

    anim = FuncAnimation(fig, update, frames=k_max, interval=45, blit=True)
    anim.save(FIGURES_DIR / "mp_convergence_regimes.gif", writer=PillowWriter(fps=15))
    plt.close(fig)
    print("  ✓ mp_convergence_regimes.gif")

    # Static companion figure: fully drawn curves for markdown fallbacks.
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.set_yscale("log")
    ax.set_xlim(1, k_max)
    ax.set_ylim(1e-15, 1.2)
    ax.set_xlabel("Iteration $k$", fontsize=12)
    ax.set_ylabel("Normalized asymptotic proxy", fontsize=12)
    ax.set_title("Marchenko--Pastur Convergence Regimes", fontsize=13)
    ax.grid(True, alpha=0.25)
    for color, label, y in zip(colors, labels, curves):
        ax.plot(k_vals, y, color=color, linewidth=2.2, label=label)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mp_convergence_regimes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ mp_convergence_regimes.png")


def make_laplace_edge_asymptotics_gif():
    k_max = 250
    k_vals = np.arange(1, k_max + 1)
    kappa = 25.0
    r = 1.0 - 1.0 / kappa

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.set_yscale("log")
    ax.set_xlim(1, k_max)
    ax.set_ylim(1e-14, 1.2)
    ax.set_xlabel("Iteration $k$", fontsize=12)
    ax.set_ylabel("Normalized asymptotic proxy", fontsize=12)
    ax.set_title("Laplace Edge Asymptotics: effect of edge exponent $p$", fontsize=13)
    ax.grid(True, alpha=0.25)

    baseline = (r ** (2 * k_vals))
    baseline = baseline / baseline[0]
    ax.plot(k_vals, baseline, "--", color="#7f8c8d", linewidth=1.8, label="Pure exponential backbone")

    line, = ax.plot([], [], color="#8e44ad", linewidth=2.5, label="Current Laplace asymptotic curve")
    ax.legend(fontsize=10, loc="upper right")
    title = ax.text(0.03, 0.06, "", transform=ax.transAxes, fontsize=11)

    p_vals = np.linspace(0.0, 2.5, 55)

    def update(frame):
        p = p_vals[frame]
        y = (k_vals.astype(float) ** (-(p + 1))) * (r ** (2 * k_vals))
        y = y / y[0]
        line.set_data(k_vals, y)
        title.set_text(fr"$p={p:.2f}$  =>  polynomial factor $k^{{-(p+1)}}$")
        return line, title

    anim = FuncAnimation(fig, update, frames=len(p_vals), interval=120, blit=True)
    anim.save(FIGURES_DIR / "laplace_edge_asymptotics.gif", writer=PillowWriter(fps=10))
    plt.close(fig)
    print("  ✓ laplace_edge_asymptotics.gif")

    # Static companion figure: several edge exponents on one axis.
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.set_yscale("log")
    ax.set_xlim(1, k_max)
    ax.set_ylim(1e-14, 1.2)
    ax.set_xlabel("Iteration $k$", fontsize=12)
    ax.set_ylabel("Normalized asymptotic proxy", fontsize=12)
    ax.set_title("Laplace Edge Asymptotics: polynomial corrections", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.plot(k_vals, baseline, "--", color="#7f8c8d", linewidth=1.8, label="Pure exponential backbone")
    for p, color in zip([0.0, 0.5, 1.5, 2.5], ["#8e44ad", "#2980b9", "#e67e22", "#27ae60"]):
        y = (k_vals.astype(float) ** (-(p + 1))) * (r ** (2 * k_vals))
        y = y / y[0]
        ax.plot(k_vals, y, color=color, linewidth=2.1, label=fr"$p={p:g}$")
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "laplace_edge_asymptotics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ laplace_edge_asymptotics.png")


def _make_powerlaw_psd_matrix(dim=220, exponent=1.4, beta=1.0):
    idx = np.arange(1, dim + 1, dtype=float)
    eigs = beta * idx ** (-exponent)
    eigs[0] = beta
    return np.diag(eigs), eigs


def _make_source_init(A_diag, s=0.5, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=A_diag.shape[0])
    return (A_diag ** s) * w


def _relative_energy_errors(A, xs):
    # b = 0, x* = 0 => f(x) = 0.5 x^T A x.
    def f(x):
        return 0.5 * float(x @ (A @ x))
    vals = np.array([f(x) for x in xs], dtype=float)
    floor = 1e-18
    vals = np.maximum(vals, floor)
    return vals / vals[0]


def make_cg_vs_gd_powerlaw_plot():
    d = 220
    A, eigs = _make_powerlaw_psd_matrix(dim=d, exponent=1.4, beta=1.0)
    grad = lambda x: A @ x
    beta = float(np.max(eigs))
    n_iters = 140

    x0_s0 = _make_source_init(eigs, s=0.0, seed=7)
    x0_s12 = _make_source_init(eigs, s=0.5, seed=9)

    xs_gd_s0 = gd_iterates(grad, x0_s0, 1.0 / beta, n_iters)
    xs_cg_s0 = cg_iterates(A, x0_s0, n_iters)
    xs_gd_s12 = gd_iterates(grad, x0_s12, 1.0 / beta, n_iters)
    xs_cg_s12 = cg_iterates(A, x0_s12, n_iters)

    err_gd_s0 = _relative_energy_errors(A, xs_gd_s0)
    err_cg_s0 = _relative_energy_errors(A, xs_cg_s0)
    err_gd_s12 = _relative_energy_errors(A, xs_gd_s12)
    err_cg_s12 = _relative_energy_errors(A, xs_cg_s12)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.set_yscale("log")
    ax.set_xlim(0, n_iters)
    ax.set_ylim(1e-16, 1.2)
    ax.set_xlabel("Iteration $k$", fontsize=12)
    ax.set_ylabel(r"$(f(x_k)-f(x^\star))/(f(x_0)-f(x^\star))$", fontsize=12)
    ax.set_title("CG vs GD under Power-Law Spectrum", fontsize=13)
    ax.grid(True, alpha=0.25)

    ax.plot(np.arange(len(err_gd_s0)), err_gd_s0, color="#2980b9", linewidth=2, label=r"GD, $s=0$")
    ax.plot(np.arange(len(err_cg_s0)), err_cg_s0, color="#1b4f72", linewidth=2, linestyle="--", label=r"CG, $s=0$")
    ax.plot(np.arange(len(err_gd_s12)), err_gd_s12, color="#e67e22", linewidth=2, label=r"GD, $s=1/2$")
    ax.plot(np.arange(len(err_cg_s12)), err_cg_s12, color="#a04000", linewidth=2, linestyle="--", label=r"CG, $s=1/2$")
    ax.legend(fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cg_vs_gd_powerlaw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ cg_vs_gd_powerlaw.png")


def make_cg_vs_gd_mp_gif():
    d = 180
    n_iters = 120
    gammas = [0.5, 1.0, 2.0]
    colors = ["#2980b9", "#e67e22", "#27ae60"]
    rng = np.random.default_rng(123)

    all_err_gd = []
    all_err_cg = []
    labels = []

    for gamma in gammas:
        n = int(round(d / gamma))
        X = rng.normal(scale=1.0 / np.sqrt(d), size=(n, d))
        A = (X.T @ X) / n
        grad = lambda x, A=A: A @ x
        x0 = rng.normal(size=d)

        xs_gd = gd_iterates(grad, x0, 1.0 / np.linalg.eigvalsh(A).max(), n_iters)
        xs_cg = cg_iterates(A, x0, n_iters)
        all_err_gd.append(_relative_energy_errors(A, xs_gd))
        all_err_cg.append(_relative_energy_errors(A, xs_cg))
        labels.append(fr"$\gamma={gamma:g}$")

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.set_yscale("log")
    ax.set_xlim(0, n_iters)
    ax.set_ylim(1e-16, 1.2)
    ax.set_xlabel("Iteration $k$", fontsize=12)
    ax.set_ylabel(r"$(f(x_k)-f(x^\star))/(f(x_0)-f(x^\star))$", fontsize=12)
    ax.set_title("CG vs GD across Marchenko--Pastur Regimes", fontsize=13)
    ax.grid(True, alpha=0.25)

    lines_gd = []
    lines_cg = []
    for color, lab in zip(colors, labels):
        line_gd, = ax.plot([], [], color=color, linewidth=2.0, label=lab + " GD")
        line_cg, = ax.plot([], [], color=color, linewidth=2.0, linestyle="--", label=lab + " CG")
        lines_gd.append(line_gd)
        lines_cg.append(line_cg)
    ax.legend(fontsize=9, loc="upper right", ncol=2)

    def update(frame):
        k = min(frame + 1, n_iters + 1)
        for line, err in zip(lines_gd, all_err_gd):
            kk = min(k, len(err))
            line.set_data(np.arange(kk), err[:kk])
        for line, err in zip(lines_cg, all_err_cg):
            kk = min(k, len(err))
            line.set_data(np.arange(kk), err[:kk])
        return tuple(lines_gd + lines_cg)

    anim = FuncAnimation(fig, update, frames=n_iters + 10, interval=70, blit=True)
    anim.save(FIGURES_DIR / "cg_vs_gd_mp.gif", writer=PillowWriter(fps=12))
    plt.close(fig)
    print("  ✓ cg_vs_gd_mp.gif")

    # Static companion figure: fully drawn trajectories for markdown fallbacks.
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.set_yscale("log")
    ax.set_xlim(0, n_iters)
    ax.set_ylim(1e-16, 1.2)
    ax.set_xlabel("Iteration $k$", fontsize=12)
    ax.set_ylabel(r"$(f(x_k)-f(x^\star))/(f(x_0)-f(x^\star))$", fontsize=12)
    ax.set_title("CG vs GD across Marchenko--Pastur Regimes", fontsize=13)
    ax.grid(True, alpha=0.25)
    for color, lab, err_gd, err_cg in zip(colors, labels, all_err_gd, all_err_cg):
        x_gd = np.arange(len(err_gd))
        x_cg = np.arange(len(err_cg))
        ax.plot(x_gd, err_gd, color=color, linewidth=2.0, label=lab + " GD")
        ax.plot(x_cg, err_cg, color=color, linewidth=2.0, linestyle="--", label=lab + " CG")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cg_vs_gd_mp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ cg_vs_gd_mp.png")


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating Week 1 figures...")
    make_gif_condition()
    make_gif_chebyshev()
    make_gif_convergence()
    make_gif_cg()
    make_chebyshev_plot()
    make_chebyshev2_plot()
    make_chebyshev_stepsizes_pd_plot()
    make_chebyshev_stepsizes_psd_plot()
    make_gif_optimal_poly()
    make_source_condition_rates_plot()
    make_source_condition_phase_transition_plot()
    make_power_law_density_rates_plot()
    make_marchenko_pastur_regimes_plot()
    make_mp_convergence_regimes_gif()
    make_laplace_edge_asymptotics_gif()
    make_cg_vs_gd_powerlaw_plot()
    make_cg_vs_gd_mp_gif()
    print("Done! Figures saved to", FIGURES_DIR)
