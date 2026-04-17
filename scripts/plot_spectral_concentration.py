"""
Animate the integrand λ(1−λ/β)^{2k} for fixed-stepsize GD,
showing how the peak concentrates near λ* = β/(2k+1) and
shifts toward zero as k grows.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

beta = 1.0
lam = np.linspace(0, beta, 2000)

k_values = list(range(1, 8)) + list(range(8, 20, 2)) + list(range(20, 51, 5)) + \
           list(range(60, 101, 10)) + list(range(120, 201, 20))


def integrand(lam, k):
    return lam * (1 - lam / beta) ** (2 * k)


def peak_value(k):
    lam_star = beta / (2 * k + 1)
    return integrand(lam_star, k)


fig, ax = plt.subplots(figsize=(7, 4))
line, = ax.plot([], [], "royalblue", lw=2.2)
vline = ax.axvline(0, color="crimson", ls="--", lw=1.5, alpha=0.8)
peak_dot, = ax.plot([], [], "o", color="crimson", ms=7, zorder=5)
title = ax.set_title("", fontsize=13)

ax.set_xlabel(r"$\lambda$", fontsize=12)
ax.set_ylabel(r"$\lambda\,(1-\lambda/\beta)^{2k}$", fontsize=12)
ax.set_xlim(0, beta)

y_max_init = peak_value(k_values[0]) * 1.15
ax.set_ylim(0, y_max_init)


def animate(frame):
    k = k_values[frame]
    y = integrand(lam, k)
    line.set_data(lam, y)

    lam_star = beta / (2 * k + 1)
    y_star = integrand(lam_star, k)
    vline.set_xdata([lam_star])
    peak_dot.set_data([lam_star], [y_star])

    y_top = y_star * 1.15
    ax.set_ylim(0, max(y_top, 1e-6))

    title.set_text(
        rf"$k = {k}$,  $\lambda^\ast = \beta/(2k+1) = {lam_star:.4f}$"
    )
    return line, vline, peak_dot, title


anim = FuncAnimation(fig, animate, frames=len(k_values), interval=300, blit=False)
out = FIGURES_DIR / "spectral_concentration.gif"
anim.save(str(out), writer=PillowWriter(fps=4))
plt.close(fig)
print(f"Saved {out}")
