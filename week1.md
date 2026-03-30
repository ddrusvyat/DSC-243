---
layout: default
title: "Week 1: Convex Quadratics"
---

# Week 1: Convex Quadratics

[← Back to course page](./)

---

## Overview

This week we study optimization algorithms for the **convex quadratic problems**. This is the most basic and fundamental problem in numerical optimization. Surprisingly, many of the phonemona that hold for minimizng convex quadratics have direct analogues for highly nonlinear and complex models (e.g. deep learning). Since the objective function is a convex quadratic, this setting allows us to develop sharp intuition for convergence behavior using only basic linear algebraic tools. Moving beyond linear least squares will require combining linear algebra with analytic techniques---more on this later.

We cover three algorithms of increasing sophistication:

1. **Gradient descent** with a fixed stepsize
2. **Chebyshev-accelerated gradient descent**
3. **Conjugate Gradient method**

---

## 1. Problem Setup

We consider the quadratic minimization problem

$$\min_{x \in \mathbb{R}^d} \; f(x) = \tfrac{1}{2} x^\top A x - b^\top x,$$

where $A \in \mathbb{R}^{d \times d}$ is a symmetric positive semidefinite matrix, meaning $A = A^\top$ and $v^\top A v \geq 0$ for all $v \in \mathbb{R}^d$. The gradient is

$$\nabla f(x) = Ax - b.$$

In particular, the solutions of the problem are exactly the solutions of the linear system $Ax=b$. Note that this linear system is special in that $A$ is a positive definite matrix---a property with important consequences for numerical methods.

We denote the eigenvalues of $A$ by

$$
0 < \alpha = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_d = \beta
$$

and its **condition number** by $\kappa = \beta / \alpha$.

A key example of convex quadratic optimization is **linear least squares**:

$$
\min_{x \in \mathbb{R}^d} \;\tfrac{1}{2}\|Dx - y\|^2,
$$

under the correspondence $A = D^\top D$ and $b = D^\top y$. In applications, $D \in \mathbb{R}^{m \times d}$ is usually a data matrix and $y \in \mathbb{R}^m$ is a vector of observations. 


**Why convex quadratic minimization?** The linear system $Ax = b$ arises everywhere: in linear regression for inference, as a subroutine in Newton's method and interior-point algorithms, and as a building block for preconditioning. Understanding how to solve it iteratively is fundamental.

---

## 2. Gradient Descent

We will be interested in algorithms that access the matrix $A$ only by evaluating matrix-vector products $v\mapsto Av$ for any query vector $v$. This **matrix-free** abstraction is powerful for several reasons:

- **Storage.** In many applications $A$ is never formed explicitly. For instance, in least squares with $A = D^\top D$, the product $Av = D^\top(Dv)$ can be computed using two matrix-vector products with $D$ and $D^\top$, which costs $O(md)$ operations and requires storing only $D \in \mathbb{R}^{m \times d}$ rather than the $d \times d$ matrix $A$. When $m \ll d^2 / d = d$ or when $D$ is sparse or structured, this is a significant saving.

- **Structure.** Many matrices arising in practice (e.g., discrete Laplacians, convolution operators, fast transforms) admit fast matrix-vector products via the FFT or other algorithms, costing $O(d \log d)$ or even $O(d)$ per product—far less than the $O(d^2)$ cost of a general dense multiply, and enormously less than the $O(d^3)$ cost of a direct factorization.

- **Generality.** By treating $A$ as a "black box" that we can only query through products, we obtain algorithms that work unchanged whether $A$ is dense, sparse, or defined only implicitly through an operator. This abstraction cleanly separates the optimization algorithm from the problem-specific details of how $A$ acts on vectors.

All three methods studied this week—gradient descent, Chebyshev-accelerated gradient descent, and conjugate gradients—are matrix-free: their only access to $A$ is through one matrix-vector product per iteration.

### Algorithm

Starting from $x_0 \in \mathbb{R}^d$, gradient descent with stepsize $\eta > 0$ iterates

$$
\begin{aligned}
x_{k+1} = x_k - \eta \nabla f(x_k) &= x_k - \eta(Ax_k - b) \\
         &= x_k - \eta A(x_k - x^\star),
\end{aligned}
\tag{1}
$$

where $x^\star$ is any minimizer of $f$, i.e. one satisfying $Ax^\star=b$.

### Error recurrence

To analyze gradient descent, we introduce the **error vector** $e_k = x_k - x^\star$. Subtracting $x^\star$ from both sides of $(1)$ yields

$$e_{k+1} = (I - \eta A)\, e_k.$$

Unrolling the recurrence gives $e_k = (I - \eta A)^k e_0$. Next, observe that the function value gap can be expressed in terms of $e_k$ as

$$
\begin{aligned}
f(x_k) - f(x^\star)
&= \tfrac{1}{2} x_k^\top A x_k - b^\top x_k - \tfrac{1}{2} (x^\star)^\top A x^\star + b^\top x^\star \\
&= \tfrac{1}{2} x_k^\top A x_k - (Ax^\star)^\top x_k - \tfrac{1}{2} (x^\star)^\top A x^\star + (Ax^\star)^\top x^\star \\
&= \tfrac{1}{2} (x_k - x^\star)^\top A\, (x_k - x^\star) \\
&= \tfrac{1}{2}\, e_k^\top A\, e_k \\
&=: \tfrac{1}{2}\|e_k\|_A^2,
\end{aligned}
$$

where $\|v\|_A = \sqrt{v^\top A v}$ is the **$A$-norm** (or energy norm). This is the natural norm for measuring progress on quadratic problems.

### Convergence for a general stepsize

Let $v_1, \ldots, v_d$ be an orthonormal eigenbasis of $A$ with $Av_i = \lambda_i v_i$. Expanding the initial error as $e_0 = \sum_{i=1}^d c_i v_i$, the error at step $k$ is

$$
\begin{aligned}
e_k &= (I - \eta A)^k\, e_0 \\
    &= (I - \eta A)^k \sum_{i=1}^d c_i\, v_i \\
    &= \sum_{i=1}^d c_i\, (I - \eta A)^k\, v_i \\
    &= \sum_{i=1}^d c_i\, (1 - \eta\lambda_i)^k\, v_i.
\end{aligned}
$$

The $A$-norm of the error therefore satisfies

$$
\|e_k\|_A^2 = \sum_{i=1}^d \lambda_i (1 - \eta\lambda_i)^{2k}\, c_i^2 \leq \max_{1 \leq i \leq d} (1 - \eta\lambda_i)^{2k} \cdot \sum_{i=1}^d \lambda_i\, c_i^2 = \rho(\eta)^{2k}\, \|e_0\|_A^2,
$$

where we set

$$
\rho(\eta) := \max_{1 \leq i \leq d} \lvert 1 - \eta\lambda_i\rvert=\max(\lvert 1 - \eta\alpha\rvert, \lvert 1 - \eta \beta\rvert).
$$

We have thus proved the following.

<div style="background-color: #eef6fc; border-left: 4px solid #2980b9; padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">

**Theorem 1 (Gradient descent).** *For any $\eta \in (0, \tfrac{2}{\beta})$ the inclusion $\rho(\eta)\in (0,1)$ holds and the gradient descent iterates enjoy the linear rate of convergence:*

$$f(x_k) - f(x^\star) \leq \rho(\eta)^{2k}\,\bigl(f(x_0) - f(x^\star)\bigr).$$

</div>




### Optimal stepsize

The rate $\rho(\eta)$ depends on the stepsize $\eta$. To find the ``optimal'' fixed stepsize, we minimize $\rho(\eta) = \max(\lvert 1 - \eta\alpha\rvert,\; \lvert 1 - \eta \beta\rvert)$ over $\eta$. Observe that $1 - \eta\alpha$ is decreasing in $\eta$ while $\eta \beta - 1$ is increasing. These two expressions balance when $1 - \eta\alpha = \eta \beta - 1$, which gives

$$\eta^\star = \frac{2}{\beta + \alpha}.$$

<div style="background-color: #eef6fc; border-left: 4px solid #2980b9; padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">

**Corollary 1 (Optimal fixed stepsize).** *With $\eta = \eta^\star = \frac{2}{\beta+\alpha}$, gradient descent satisfies*

$$f(x_k) - f(x^\star) \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^{2k}\bigl(f(x_0) - f(x^\star)\bigr).$$

</div>

*Proof.* Substituting $\eta^\star$ into the expression for $\rho$ yields

$$\rho(\eta^\star) = \left\lvert 1 - \frac{2\alpha}{\beta+\alpha}\right\rvert = \frac{\beta - \alpha}{\beta + \alpha} = \frac{\kappa - 1}{\kappa + 1}.$$

The result follows from Theorem 1. <span style="float: right;">$\square$</span>

The rate $\rho^\star = \frac{\kappa - 1}{\kappa + 1}$ approaches $1$ as $\kappa \to \infty$, meaning convergence degrades for ill-conditioned problems. For example, when $\kappa = 100$ we get $\rho^\star \approx 0.98$, so roughly $k \approx 2{,}500$ iterations are needed to reduce the suboptimality by a factor of $e^{-50}$.

### The practical stepsize $\eta = 1/\beta$

The optimal stepsize $\eta^\star = 2/(\beta+\alpha)$ requires knowledge of both the largest and smallest eigenvalues of $A$. In practice, the smallest eigenvalue $\alpha$ is often unknown or expensive to estimate. A natural and widely used alternative is the stepsize $\eta = 1/\beta$, which requires only an upper bound on the spectrum.

<div style="background-color: #eef6fc; border-left: 4px solid #2980b9; padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">

**Corollary 2 (Stepsize $1/\beta$).** *With $\eta = 1/\beta$, gradient descent satisfies*

$$f(x_k) - f(x^\star) \leq \left(1 - \frac{1}{\kappa}\right)^{2k}\bigl(f(x_0) - f(x^\star)\bigr).$$

</div>

*Proof.* Substituting $\eta = 1/\beta$ into Theorem 1 yields:

$$\rho(1/\beta) = \max\!\big(\lvert 1 - \alpha/\beta\rvert,\; \lvert 1 - 1\rvert\big) = 1 - \frac{1}{\kappa},$$

which completes the proof. <span style="float: right;">$\square$</span>

**Comparison of the two stepsizes.** For large $\kappa$, the two rates behave as

$$\rho^\star = \frac{\kappa - 1}{\kappa + 1} = 1 - \frac{2}{\kappa} + O(\kappa^{-2}), \qquad \rho(1/\beta) = 1 - \frac{1}{\kappa}.$$

Thus the optimal stepsize is roughly twice as fast per step as $\eta = 1/\beta$---a modest price to pay for not knowing $\alpha$.

### Iteration complexity

So far we have described how the suboptimality $f(x_k)-f(x^{\star})$ decays with with the iteration counted. An equivalent and often more informative viewpoint is to ask: *how many iterations are needed to reach a target accuracy $\varepsilon$?* This is the **iteration complexity** of the algorithm.

From Theorem 1 with stepsize $\eta = 1/\beta$, we may use the elementary inequality $1 - x \leq e^{-x}$ to deduce that

$$
k \geq \kappa\cdot\ln\left(\frac{1}{\varepsilon}\right)
$$

iterations suffice to achieve $\varepsilon$-accuracy $f(x_k)-f(x^\star)\leq \varepsilon$. This is the **iteration complexity** of gradient descent on quadratics.

This change of perspective---from contraction rate to iteration count---is valuable because it separates two distinct contributions to the difficulty of the problem: the **condition number** $\kappa$, which measures the intrinsic difficulty of the problem, and the **logarithmic accuracy** $\ln(1/\varepsilon)$, which measures how precisely we need to solve it.

### Visualizing the effect of condition number

The following animation shows gradient descent on two quadratics with the same starting point. On the left, the problem is well-conditioned ($\kappa = 1.5$); on the right, it is ill-conditioned ($\kappa = 50$). Notice the zig-zagging behavior on the ill-conditioned problem.

![Gradient descent: well-conditioned vs ill-conditioned](figures/gd_condition.gif)

---

## 3. Acceleration by Chebyshev Stepsizes


The analysis of gradient descent so far was quite crude in that it was based on lower-bounding the improvement in function value in a single step. We now show that by monitoring performance across a longer time horizon, it is possible to choose a time-varying stepsize that yields a much faster rate of convergence.  To see this, consider gradient descent with *time-varying* stepsizes $\eta_0, \eta_1, \ldots, \eta_{k-1}$. We saw that the error $e_j = x_j - x^\star$ evolves as $e_{j+1} = (I - \eta_j A)\,e_j$. Therefore, after $k$ steps we have:

$$e_k = (I - \eta_{k-1}A)(I - \eta_{k-2}A)\cdots(I - \eta_0 A)\,e_0 = p_k(A)\,e_0,$$

where $p_k$ is the degree-$k$ polynomial

$$p_k(\lambda) = \prod_{j=0}^{k-1}(1 - \eta_j \lambda).$$

Note that $p_k(0) = 1$ regardless of the choice of stepsizes. Expanding $e_k$ in the eigenbasis of $A$ as before yields:

$$
\begin{aligned}
f(x_k) - f(x^\star) = \tfrac{1}{2}\|e_k\|_A^2
&= \tfrac{1}{2}\sum_{i=1}^d \lambda_i\, p_k(\lambda_i)^2\, c_i^2 \\
&\leq \max_{\lambda \in [\alpha, \beta]} p_k(\lambda)^2 \cdot \tfrac{1}{2}\|e_0\|_A^2.
\end{aligned}
$$

Rearranging yields

$$\frac{f(x_k) - f(x^\star)}{f(x_0) - f(x^\star)} \leq \max_{\lambda \in [\alpha, \beta]} p_k(\lambda)^2.$$

Fixed-stepsize gradient descent corresponds to the special case $p_k(\lambda) = (1 - \eta\lambda)^k$, but we are now free to choose *any* stepsizes. Notice that as we vary the stepsizes $\eta_0,\ldots \eta_{k-1}$, any degree $k$ polynomial $p(\lambda)$ satisfying $p(0)=1$ can be realized as $p_{k-1}(\lambda)$. Thus choosing time-varying stepsizes is equivalent to choosing such a polynomial.  The best possible convergence after $k$ steps is therefore determined by the **minimax polynomial problem**:

$$
\min_{\substack{p \in \mathcal{P}_k \\ p(0) = 1}} \max_{\lambda \in [\alpha, \beta]} p(\lambda)^2,
$$

where $\mathcal{P}_k$ denotes the set of polynomials of degree at most $k$. The solution to this classical approximation problem involves **Chebyshev polynomials**.

### Chebyshev polynomials

The **Chebyshev polynomial of the first kind** of degree $k$, denoted $T_k$, is defined recursively as follows. We set $T_0(x) = 1$ and $T_1(x) = x$ and define

$$T_{k+1}(x) = 2x\,T_k(x) - T_{k-1}(x) \qquad \forall k\geq 1.$$

An equivalent characterization of Chebychev polynomials is the equality

$$
T_k(\cos\theta) = \cos(k\theta) \qquad \forall \theta \in [0,\pi].
$$

Chebychev polynomials play a special role in numerical analysis because they solve the extremal problem: 

> Any degree-$k$ polynomial $p(x)$ with the same leading coefficient as $T_k$ satisfies
>
> $$\max_{x\in [-1,1]} \lvert p(x)\rvert\geq \max_{x\in [-1,1]} \lvert T_k(x)\rvert=1.$$

In words, among all degree-$k$ polynomials with the same leading coefficient as $T_k$, the Chebychev polynomial $T_k$ has the smallest maximum absolute value on $[-1,1]$. See the figure below.


![Chebyshev polynomials of the first kind](figures/chebyshev_polynomials.png)


Chebychev polynomials satisfy the following key properties:
1. **Boundedness:** The inequality $\lvert T_k(t)\rvert \leq 1$ holds for all $t \in [-1,1]$ with equality at $t_j = \cos(j\pi/k)$ for $j = 0, \ldots, k$. 

2. **Roots:** $T_k$ has $k$ roots in $(-1,1)$ at $t_j = \cos\!\left(\frac{(2j-1)\pi}{2k}\right)$ for $j = 1, \ldots, k$.

3. **Explosion:** For $\lvert t\rvert > 1$, we have $T_k(t) = \cosh(k\,\operatorname{arccosh}(t))$.




### The optimal polynomial
Returning to gradient descent, we rescale the interval $[\alpha, \beta]$ to $[-1,1]$ with the affine change of coordinates $\lambda\mapsto\frac{\beta + \alpha - 2\lambda}{\beta - \alpha}$. Not that this transformation sends $\lambda = 0$ to the point $\sigma := \frac{\beta + \alpha}{\beta - \alpha} = \frac{\kappa + 1}{\kappa - 1} > 1$. Thus, under this substitution, any degree-$k$ polynomial $p(\lambda)$ with $p(0) = 1$ corresponds to a degree-$k$ polynomial $q$ with $q(\sigma) = 1$, and

$$\max_{\lambda \in [\alpha, \beta]}\lvert p(\lambda)\rvert = \max_{t \in [-1,1]}\lvert q(t)\rvert.$$

We must therefore find the degree-$k$ polynomial $q$ with $q(\sigma) = 1$ that has the smallest maximum on $[-1,1]$. By properties 1 and 3 above, $T_k$ is bounded by 1 on $[-1,1]$ yet $T_k(\sigma) \gg 1$ for large $k$. This makes the rescaled polynomial

$$q_k^*(t) = \frac{T_k(t)}{T_k(\sigma)}$$

an excellent candidate: it satisfies $q_k^*(\sigma) = 1$ and $\max_{t \in [-1,1]}\lvert q_k^*(t)\rvert = 1/T_k(\sigma)$, which is small because $T_k(\sigma)$ grows exponentially in $k$. Transforming back to the $\lambda$-variable, the optimal polynomial is

$$p_k^*(\lambda) = \frac{T_k\!\left(\frac{\beta + \alpha - 2\lambda}{\beta - \alpha}\right)}{T_k\!\left(\frac{\kappa + 1}{\kappa - 1}\right)}.$$

Its roots on $[\alpha, \beta]$ are the images of the Chebyshev roots $t_j$ under the inverse map $t \mapsto \frac{\beta+\alpha}{2} - \frac{\beta-\alpha}{2}\,t$, giving the stepsizes $\eta_j = 1/\lambda_j$. We thus have arrived at the following theorem.

<div style="background-color: #eef6fc; border-left: 4px solid #2980b9; padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">

**Theorem 2 (Chebyshev stepsizes).** *Define the stepsizes*

$$\eta_j = \tfrac{1}{\lambda_j}~~ \textrm{where}~~\lambda_j = \tfrac{\beta + \alpha}{2} - \tfrac{\beta - \alpha}{2}\cos\!\left(\tfrac{(2j - 1)\pi}{2k}\right) ~~\textrm{for}~ j = 1, \ldots, k.$$

*Then the gradient descent iterates satisfy*

$$f(x_k) - f(x^\star) \leq 4\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^{2k}\bigl(f(x_0) - f(x^\star)\bigr). \tag{2}$$

</div>

*Proof.* We have already shown that

$$
\frac{f(x_k) - f(x^\star)}{f(x_0) - f(x^\star)} \leq \max_{\lambda \in [\alpha, \beta]} p_k^*(\lambda)^2 = \frac{1}{T_k(\sigma)^2},
$$

where $\sigma = \frac{\kappa+1}{\kappa-1}$. It remains to estimate $T_k(\sigma)$. Using the expression $T_k(x) = \cosh(k\,\operatorname{arccosh}(x))$ for $x > 1$ we deduce

$$
\begin{aligned}
\operatorname{arccosh}(\sigma)
&= \ln\!\big(\sigma + \sqrt{\sigma^2 - 1}\big)  = \ln\frac{\sqrt{\kappa}+1}{\sqrt{\kappa}-1}.
\end{aligned}
$$

Therefore:

$$
\begin{aligned}
T_k(\sigma) &= \cosh\!\left(k\ln\frac{\sqrt{\kappa}+1}{\sqrt{\kappa}-1}\right) = \frac{1}{2}\left[\left(\frac{\sqrt{\kappa}+1}{\sqrt{\kappa}-1}\right)^k + \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k\right] \\
&\geq \frac{1}{2}\left(\frac{\sqrt{\kappa}+1}{\sqrt{\kappa}-1}\right)^k,
\end{aligned}
$$

which completes the proof. <span style="float: right;">$\square$</span>

Thus the effective per-step contraction rate is $\rho_{\rm Cheb} = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$, a **square-root improvement** over the fixed-stepsize rate $\rho^\star = \frac{\kappa - 1}{\kappa + 1}$. For $\kappa = 100$, fixed-stepsize gradient descent has $\rho^\star \approx 0.98$ while the Chebyshev method achieves $\rho_{\rm Cheb} \approx 0.82$---a dramatic acceleration.

### Comparing trajectories

The animation below overlays gradient descent (blue) and Chebyshev-accelerated GD (red) on the same ill-conditioned quadratic. The Chebyshev method reaches the minimizer much faster.

![GD vs Chebyshev stepsizes](figures/gd_vs_chebyshev.gif)

---

## 4. The Conjugate Gradient Method

### Motivation

Gradient descent chooses the steepest descent direction at each step. The Conjugate Gradient (CG) method instead builds a sequence of **$A$-conjugate** search directions $p_0, p_1, \ldots$ satisfying

$$p_i^\top A p_j = 0 \quad \text{for } i \neq j.$$

This orthogonality condition ensures that each step makes "independent" progress, and no work is ever undone.

### Algorithm

Starting from $x_0$, set $r_0 = b - Ax_0$ and $p_0 = r_0$. For $k = 0, 1, 2, \ldots$:

$$\eta_k = \frac{r_k^\top r_k}{p_k^\top A p_k}, \qquad x_{k+1} = x_k + \eta_k p_k,$$

$$r_{k+1} = r_k - \eta_k A p_k, \qquad \beta_k = \frac{r_{k+1}^\top r_{k+1}}{r_k^\top r_k}, \qquad p_{k+1} = r_{k+1} + \beta_k p_k.$$

### Key properties

- **Finite termination:** CG solves a $d$-dimensional quadratic in at most $d$ iterations (in exact arithmetic).
- **Optimal Krylov method:** Among all methods that search in the Krylov subspace $\mathcal{K}_k(A, r_0) = \mathrm{span}\{r_0, Ar_0, \ldots, A^{k-1}r_0\}$, CG minimizes $f$ at each step.
- **Convergence rate:** Even before termination, CG satisfies

$$\frac{f(x_k) - f(x^\star)}{f(x_0) - f(x^\star)} \leq 4\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^{2k},$$

which matches the Chebyshev rate (up to a constant) without requiring knowledge of $\alpha$ and $\beta$.

### Visualizing CG

The animation below shows CG on a 2D quadratic. Notice that it converges in exactly 2 steps (the dimension of the problem). The red arrows indicate the search directions, which are $A$-conjugate.

![Conjugate Gradient method](figures/conjugate_gradient.gif)

---

## 5. Convergence Comparison

The following animation compares the convergence of all three methods on the same problem, plotting $f(x_k)/f(x_0)$ on a log scale.

- **Gradient descent** (blue): linear convergence with rate $(\kappa-1)/(\kappa+1)$
- **Chebyshev GD** (red): faster convergence with effective rate $(\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$
- **Conjugate Gradients** (green): fastest, with finite termination

![Convergence comparison](figures/convergence_comparison.gif)

---

## Summary

| Method | Per-step cost | Convergence rate | Requires $\alpha, \beta$? |
|--------|--------------|-----------------|-------------------|
| Gradient descent | One matrix-vector product | $\left(\frac{\kappa-1}{\kappa+1}\right)^2$ | Yes (for optimal step) |
| Chebyshev GD | One matrix-vector product | $\approx \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2$ | Yes |
| Conjugate Gradients | One matrix-vector product | $\leq 4\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^{2k}$, finite termination | No |

The key takeaway: on quadratics, CG achieves the accelerated rate *adaptively*, without needing to know the condition number, and terminates in at most $d$ steps.

---

[← Back to course page](./)
