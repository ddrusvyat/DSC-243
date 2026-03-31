---
layout: default
title: "Week 1: Convex Quadratics"
math:
  engine: mathjax
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
0 \leq  \alpha = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_d = \beta
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

where $\lVert v\rVert_A = \sqrt{v^\top A v}$ is the **$A$-norm** (or energy norm). This is the natural norm for measuring progress on quadratic problems.

### Convergence for a general stepsize

Let $v_1, \ldots, v_d$ be an orthonormal eigenbasis of $A$ with $Av_i = \lambda_i v_i$. Expanding the initial error as $e_0 = \sum_{i=1}^d c_i v_i$, the error at step $k$ is

$$
\begin{aligned}
e_k &= (I - \eta A)^k\, e_0 = (I - \eta A)^k \sum_{i=1}^d c_i\, v_i = \sum_{i=1}^d c_i\, (I - \eta A)^k\, v_i = \sum_{i=1}^d c_i\, (1 - \eta\lambda_i)^k\, v_i.
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

**Corollary 1 (Optimal fixed stepsize).** *Suppose $\alpha>0$ and set $\eta = \eta^\star = \frac{2}{\beta+\alpha}$. Then gradient descent satisfies*

$$f(x_k) - f(x^\star) \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^{2k}\bigl(f(x_0) - f(x^\star)\bigr).$$

</div>

*Proof.* Substituting $\eta^\star$ into the expression for $\rho$ yields

$$\rho(\eta^\star) = \left\lvert 1 - \frac{2\alpha}{\beta+\alpha}\right\rvert = \frac{\beta - \alpha}{\beta + \alpha} = \frac{\kappa - 1}{\kappa + 1}.$$

The result follows from Theorem 1. <span style="float: right;">$\square$</span>



### The practical stepsize $\eta = 1/\beta$

The optimal stepsize $\eta^\star = 2/(\beta+\alpha)$ requires knowledge of both the largest and smallest eigenvalues of $A$. In practice, the smallest eigenvalue $\alpha$ is often unknown or expensive to estimate. A natural and widely used alternative is the stepsize $\eta = 1/\beta$, which requires only an upper bound on the spectrum.

<div style="background-color: #eef6fc; border-left: 4px solid #2980b9; padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">

**Corollary 2 (Stepsize $1/\beta$).** *Suppose $\alpha>0$  and set $\eta = 1/\beta$. Then gradient descent satisfies*

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
&= \tfrac{1}{2}\sum_{i=1}^d \lambda_i\, p_k(\lambda_i)^2\, c_i^2 \leq \max_{\lambda \in [\alpha, \beta]} p_k(\lambda)^2 \cdot \tfrac{1}{2}\|e_0\|_A^2.
\end{aligned}
$$

Rearranging yields

$$\frac{f(x_k) - f(x^\star)}{f(x_0) - f(x^\star)} \leq \max_{\lambda \in [\alpha, \beta]} p_k(\lambda)^2.$$

Fixed-stepsize gradient descent corresponds to the special case $p_k(\lambda) = (1 - \eta\lambda)^k$, but we are now free to choose *any* stepsizes. Notice that as we vary the stepsizes $\eta_0,\ldots \eta_{k-1}$, any degree $k$ polynomial $p(\lambda)$ satisfying $p(0)=1$ can be realized as $p_{k-1}(\lambda)$. Thus choosing time-varying stepsizes is equivalent to choosing such a polynomial.  The best possible convergence after $k$ steps is therefore determined by the **minimax polynomial problem**:

$$
\min_{\substack{p \in \mathcal{P}_k \\ p(0) = 1}} \max_{\lambda \in [\alpha, \beta]} p(\lambda)^2,
$$

where $\mathcal{P}_k$ denotes the set of polynomials of degree at most $k$. The solution to this classical approximation problem involves Chebyshev polynomials.

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
Returning to gradient descent, we rescale the interval $[\alpha, \beta]$ to $[-1,1]$ with the affine change of coordinates $\lambda\mapsto\frac{\beta + \alpha - 2\lambda}{\beta - \alpha}$. Not that this transformation sends $\lambda = 0$ to the point $\sigma := \frac{\beta + \alpha}{\beta - \alpha} = \frac{\kappa + 1}{\kappa - 1}$, assuming $\alpha>0$. Thus, under this substitution, any degree-$k$ polynomial $p(\lambda)$ with $p(0) = 1$ corresponds to a degree-$k$ polynomial $q$ with $q(\sigma) = 1$, and

$$\max_{\lambda \in [\alpha, \beta]}\lvert p(\lambda)\rvert = \max_{t \in [-1,1]}\lvert q(t)\rvert.$$

We must therefore find the degree-$k$ polynomial $q$ with $q(\sigma) = 1$ that has the smallest maximum on $[-1,1]$. By properties 1 and 3 above, $T_k$ is bounded by 1 on $[-1,1]$ yet $T_k(\sigma) \gg 1$ for large $k$. This makes the rescaled polynomial

$$q_k^*(t) = \frac{T_k(t)}{T_k(\sigma)}$$

an excellent candidate: it satisfies $q_k^{*}(\sigma) = 1$ and $\max_{t \in [-1,1]}\lvert q_k^{*}(t)\rvert = 1/T_k(\sigma)$, which is small because $T_k(\sigma)$ grows exponentially in $k$. Transforming back to the $\lambda$-variable, the optimal polynomial is

$$p_k^*(\lambda) = \frac{T_k\!\left(\frac{\beta + \alpha - 2\lambda}{\beta - \alpha}\right)}{T_k\!\left(\frac{\kappa + 1}{\kappa - 1}\right)}.$$

The animation below shows $p_k^{*}(\lambda)$ on $[\alpha, \beta]$ for increasing degree $k$. As $k$ grows, the polynomial oscillates more rapidly yet its maximum amplitude $1/T_k(\sigma)$ shrinks exponentially---this is the mechanism behind the accelerated convergence.

![Optimal polynomials on the eigenvalue interval](figures/optimal_polynomials.gif)


Summarizing, we have the following lemma. 


<div style="background-color: #eef6fc; border-left: 4px solid #2980b9; padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">

**Lemma 1 (Chebyshev minimax).** *Suppose $\alpha>0$. Then with $\sigma = \frac{\kappa+1}{\kappa-1}$, the minimax value satisfies*

$$\min_{\substack{p \in \mathcal{P}_k \\ p(0) = 1}} \max_{\lambda \in [\alpha,\beta]} \lvert p(\lambda)\rvert \leq \frac{1}{T_k(\sigma)} \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k.$$

</div>

*Proof.* The first inequality follows from a feasible choice. Indeed, the rescaled polynomial

$$
p_k^*(\lambda) = \frac{T_k\!\big(\frac{\beta+\alpha-2\lambda}{\beta-\alpha}\big)}{T_k(\sigma)},
$$

is a polynomial of degree at most $k$ satisfying the condition $p_k^{*}(0)=1$. Moreover, the boundedness of the Chebyshev polynomial on the interval $[-1,1]$ yields the estimate

$$
\max_{\lambda \in [\alpha,\beta]} \lvert p_k^*(\lambda)\rvert = \frac{1}{T_k(\sigma)}.
$$

This proves the first inequality.

For the second inequality, we use the identity $T_k(x) = \cosh(k\,\operatorname{arccosh}(x))$ valid for every real number $x>1$. Applying this identity with the quantity $\sigma$ gives the relation

$$
\begin{aligned}
\operatorname{arccosh}(\sigma)
&= \ln\!\big(\sigma + \sqrt{\sigma^2 - 1}\big)  = \ln\frac{\sqrt{\kappa}+1}{\sqrt{\kappa}-1}.
\end{aligned}
$$

Consequently, the representation

$$
\begin{aligned}
T_k(\sigma) &= \cosh\!\left(k\ln\frac{\sqrt{\kappa}+1}{\sqrt{\kappa}-1}\right) = \frac{1}{2}\left[\left(\frac{\sqrt{\kappa}+1}{\sqrt{\kappa}-1}\right)^k + \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k\right] \geq \frac{1}{2}\left(\frac{\sqrt{\kappa}+1}{\sqrt{\kappa}-1}\right)^k,
\end{aligned}
$$

holds. Taking reciprocals gives the second claim. This completes the proof. <span style="float: right;">$\square$</span>


Returning to choosing stepsizes for gradient descent, the roots of $p^{*}_k(\lambda)$ on $[\alpha, \beta]$ are the images of the Chebyshev roots $t_j$ under the inverse map $t \mapsto \frac{\beta+\alpha}{2} - \frac{\beta-\alpha}{2}\,t$, giving the stepsizes $\eta_j = 1/\lambda_j$. We thus have arrived at the main theorem of this section.

<div style="background-color: #eef6fc; border-left: 4px solid #2980b9; padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">

**Theorem 2 (Chebyshev stepsizes).** *Define the stepsizes*

$$\eta_j = \tfrac{1}{\lambda_j}~~ \textrm{where}~~\lambda_j = \tfrac{\beta + \alpha}{2} - \tfrac{\beta - \alpha}{2}\cos\!\left(\tfrac{(2j - 1)\pi}{2k}\right) ~~\textrm{for}~ j = 1, \ldots, k.$$

*Then as long as $\alpha>0$ the gradient descent iterates satisfy*

$$f(x_k) - f(x^\star) \leq 4\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^{2k}\bigl(f(x_0) - f(x^\star)\bigr). \tag{2}$$

</div>


*Proof of Theorem 2.* The polynomial estimate established above yields

$$
\frac{f(x_k) - f(x^\star)}{f(x_0) - f(x^\star)} \leq \max_{\lambda \in [\alpha, \beta]} p_k^*(\lambda)^2 = \frac{1}{T_k(\sigma)^2}.
$$

Applying Lemma 1 gives the bound

$$
\frac{1}{T_k(\sigma)^2} \leq 4\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^{2k}.
$$

Combining the preceding two estimates yields the conclusion $(2)$. This completes the proof. <span style="float: right;">$\square$</span>

Thus, the iteration complexity of Chebyshev-accelerated gradient descent is $O(\sqrt{\kappa}\,\ln(1/\varepsilon))$---a **square-root improvement** over the $O(\kappa\,\ln(1/\varepsilon))$ complexity of fixed-stepsize gradient descent. For $\kappa = 100$, this is the difference between roughly $10$ and $100$ iterations.

### Comparing trajectories

The animation below overlays gradient descent (blue) and Chebyshev-accelerated GD (red) on the same ill-conditioned quadratic. The Chebyshev method reaches the minimizer much faster.

![GD vs Chebyshev stepsizes](figures/gd_vs_chebyshev.gif)

---

## 4. The Krylov Subspace Method and Conjugate Gradients

### From polynomials to Krylov subspaces

The Chebyshev method achieves the iteration complexity $O(\sqrt{\kappa}\,\ln(1/\varepsilon))$ by cleverly choosing time-varying stepsizes---but it requires advance knowledge of the extreme eigenvalues $\alpha$ and $\beta$. Moreover, the total number of iterations must be set in advance in order to define the stepsizes. A natural question arises:

> can we match this rate adaptively, without knowing the spectrum nor setting the time horizon?

The key observation is that gradient descent with *any* sequence of stepsizes produces iterates that lie in a specific linear subspace. Due to the recursion $x_{j+1} = x_j - \eta_j(Ax_j - b)$, one readily verifies the inclusion

$$
x_k \in x_0 + \mathcal{K}_k(A, r_0),
$$

where $r_0 := b - Ax_0$ is the initial residual and

$$
\mathcal{K}_k(A, r_0) := \mathrm{span}\{r_0,\, Ar_0,\, A^2 r_0,\, \ldots,\, A^{k-1}r_0\}
$$

is the **Krylov subspace** of order $k$. Both fixed-stepsize gradient descent and the Chebyshev method search within this subspace but do not fully exploit it. The natural idea is to search *optimally* within the Krylov subspace at each step.

### The Krylov subspace method

The **Krylov subspace method** is the idealized algorithm that, at each step $k$, sets

$$x_k = \argmin_{x \,\in\, x_0 + \mathcal{K}_k(A,\,r_0)} f(x). \tag{3}$$

Since $f$ is strictly convex (when $\alpha > 0$), the minimizer in $(3)$ is unique. The convergence analysis follows immediately from the polynomial framework developed in Section 3.

<div style="background-color: #eef6fc; border-left: 4px solid #2980b9; padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">

**Theorem 3 (Krylov method convergence).** *Assuming $\alpha > 0$, the Krylov subspace method $(3)$ satisfies*

$$
f(x_k) - f(x^\star) \leq 4\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^{2k}\bigl(f(x_0) - f(x^\star)\bigr).
\tag{4}
$$

*Moreover, the method converges in at most $m$ iterations, where $m$ is the number of distinct eigenvalues of $A$.*

</div>

*Proof.* For simplicity, let $\mathcal{K}_k$ denote $\mathcal{K}_k(A,r_0)$ throughout the proof. By definition, the iterate $x_k$ minimizes $f$ over $x_0+\mathcal{K}_k$. Recall from Section 2 that the identity

$$
f(x) - f(x^\star) = \tfrac{1}{2}\|x - x^\star\|_A^2
$$

holds for every vector $x$. Since $f(x^\star)$ is constant, minimizing $f$ over a subset is equivalent to minimizing the $A$-norm distance to $x^\star$. Therefore

$$
\tfrac{1}{2}\|e_k\|_A^2 = \min_{x\,\in\, x_0 + \mathcal{K}_k}\; \tfrac{1}{2}\|x - x^\star\|_A^2.
$$

We convert this geometric minimization into a polynomial one. A vector $x$ belongs to $x_0+\mathcal{K}_k$ if and only if there exists a polynomial $q$ of degree at most $k-1$ such that $x = x_0 + q(A)\,r_0$. Using the identity $r_0 = b - Ax_0 = -Ae_0$, we obtain

$$
x - x^\star = e_0 - q(A)\,Ae_0 = p(A)\,e_0,
$$

where the polynomial $p(\lambda):=1-\lambda q(\lambda)$ has degree at most $k$ and satisfies $p(0)=1$. The correspondence between $p$ and $q$ is one-to-one. Substituting gives

$$
\|e_k\|_A^2 = \min_{\substack{p \in \mathcal{P}_k \\ p(0) = 1}} \|p(A)\,e_0\|_A^2 \leq \min_{\substack{p \in \mathcal{P}_k \\ p(0) = 1}} \max_{\lambda \in [\alpha,\beta]} p(\lambda)^2 \cdot \|e_0\|_A^2.
$$

Applying Lemma 1 yields

$$
\|e_k\|_A^2 \leq \frac{\|e_0\|_A^2}{T_k(\sigma)^2} \leq 4\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^{2k}\|e_0\|_A^2.
$$

This proves the estimate $(4)$.

For finite termination, let $\mu_1,\ldots,\mu_m$ denote the distinct eigenvalues of $A$, and consider the polynomial

$$
p(\lambda) = \prod_{i=1}^{m}\left(1 - \lambda/\mu_i\right).
$$

This polynomial has degree $m$, satisfies $p(0)=1$, and vanishes at every eigenvalue of $A$. Hence $p(A)=0$, and the polynomial representation gives $\lVert e_m\rVert_A=0$. <span style="float: right;">$\square$</span>

The convergence bound $(4)$ is identical to the Chebyshev bound $(2)$ of Theorem 2, and the iteration complexity is the same $O(\sqrt{\kappa}\,\ln(1/\varepsilon))$. The Krylov method achieves this rate *without knowing $\alpha$ or $\beta$*, and finite termination provides an absolute guarantee of at most $d$ steps. In practice, clustered eigenvalues lead to far fewer iterations than the worst-case bound suggests.

### The Conjugate Gradient algorithm

The Krylov subspace method $(3)$ is conceptual: a direct implementation would solve a $k$-dimensional optimization at each step, with cost growing as $k$ increases. The **Conjugate Gradient (CG)** algorithm is a remarkable implementation of the Krylov method that uses only **one matrix-vector product per iteration** and $O(d)$ additional work.

The key idea is to iteratively build a basis of the Krylov subspaces that is orthogonal with respect to the inner product $\langle x,y\rangle_A=x^\top Ay$, so that each successive minimization reduces to a single line search. Conceptually, this basis is formed by a Gram--Schmidt process. The special structure of Krylov subspaces ensures that each Gram--Schmidt update requires only the immediately preceding direction---a **short recurrence**---rather than all previous directions. This is why each CG iteration costs $O(d)$ work beyond the single matrix-vector product.

Concretely, the conjugate gradient method takes the form:

<div style="background-color: #f8f8f8; border: 1px solid #ccc; padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px; font-size: 0.97em;" markdown="1">

**Algorithm 1** (Conjugate Gradient Method)

**Input:** $x_0 \in \mathbb{R}^d$

1. Set $r_0 = b - Ax_0$, $\;p_0 = r_0$
2. **For** $k = 0, 1, 2, \ldots$ do:
3. $\qquad \eta_k = \dfrac{r_k^\top r_k}{p_k^\top A p_k}$
4. $\qquad x_{k+1} = x_k + \eta_k\, p_k$
5. $\qquad r_{k+1} = r_k - \eta_k\, A p_k$
6. $\qquad \beta_k = \dfrac{\lVert r_{k+1}\rVert^2}{\lVert r_k\rVert^2}$
7. $\qquad p_{k+1} = r_{k+1} + \beta_k\, p_k$

</div>

Each iteration requires one matrix-vector product $Ap_k$, the same per-step cost as gradient descent. The vectors $r_k = b - Ax_k$ are the **residuals** satisfying $r_k = -\nabla f(x_k)$, while the vectors $p_k$ are the **search directions**. The stepsize $\eta_k$ minimizes $f$ along the ray $x_k + \eta\, p_k$, while $\beta_k$ ensures $A$-conjugacy of consecutive search directions.

### CG implements the Krylov method

We now verify that the CG algorithm produces exactly the iterates of the Krylov subspace method $(3)$. The key is to show that the search directions span the Krylov subspace and that the residuals are mutually orthogonal---two properties that together force each CG iterate to minimize $f$ over the correct affine subspace.

<div style="background-color: #eef6fc; border-left: 4px solid #2980b9; padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">

**Theorem 4 (CG correctness).** *Suppose $\alpha > 0$. The CG residuals and search directions satisfy, for all valid indices:*

1. *$r_i^\top r_j = 0$ for $i \neq j$ (mutual orthogonality of residuals),*
2. *$p_i^\top A p_j = 0$ for $i \neq j$ ($A$-conjugacy of search directions),*
3. *$\mathrm{span}\lbrace p_0, \ldots, p_{k-1}\rbrace = \mathrm{span}\lbrace r_0, \ldots, r_{k-1}\rbrace = \mathcal{K}_k(A, r_0)$.*

*Consequently, $x_k$ minimizes $f$ over $x_0 + \mathcal{K}_k(A, r_0)$, and the convergence guarantee $(4)$ of Theorem 3 applies to the CG iterates.*

</div>

*Proof.* We prove properties 1--3 by induction on the index $k$. The base case $k=0$ is immediate, since the identity $p_0=r_0$ holds and the equality $\mathcal{K}_1(A,r_0)=\mathrm{span}\lbrace r_0\rbrace$ is valid. For the inductive step, assume that properties 1--3 hold through step $k$. The CG updates give the relations

$$
r_{k+1}=r_k-\eta_kAp_k, \qquad p_{k+1}=r_{k+1}+\beta_kp_k.
$$

We first verify property 1. Fix an index $j<k$. The inductive hypothesis gives the orthogonality relation $r_k^\top r_j=0$. In addition, property 3 implies that the vector $r_j$ belongs to the span $\mathrm{span}\lbrace p_0,\ldots,p_j\rbrace$. Since the vector $p_k$ is $A$-conjugate to every earlier search direction, the relation $p_k^\top Ar_j=0$ follows. Substituting these two relations into the residual update gives the identity

$$
r_{k+1}^\top r_j = r_k^\top r_j - \eta_k\,p_k^\top Ar_j = 0.
$$

For the remaining index $j=k$, the relation $r_k=p_k-\beta_{k-1}p_{k-1}$ and the $A$-conjugacy relation $p_k^\top Ap_{k-1}=0$ imply the identity $p_k^\top Ar_k=p_k^\top Ap_k$. Using the definition of the stepsize $\eta_k$ therefore yields the relation

$$
r_{k+1}^\top r_k = r_k^\top r_k - \eta_k\,p_k^\top Ap_k = \|r_k\|^2-\|r_k\|^2=0.
$$

Thus the residuals remain mutually orthogonal.

We next verify property 3. The inductive hypothesis implies that the vector $p_k$ belongs to the Krylov subspace $\mathcal{K}_{k+1}(A,r_0)$. Hence the vector $Ap_k$ belongs to the larger Krylov subspace $\mathcal{K}_{k+2}(A,r_0)$. Since the vector $r_k$ also belongs to $\mathcal{K}_{k+1}(A,r_0)$, the residual update yields the inclusion

$$
r_{k+1}=r_k-\eta_kAp_k \in \mathcal{K}_{k+2}(A,r_0).
$$

The search-direction update then gives the inclusion $p_{k+1}\in \mathcal{K}_{k+2}(A,r_0)$ as well. Moreover, the relations $p_{k+1}=r_{k+1}+\beta_kp_k$ and $r_{k+1}=p_{k+1}-\beta_kp_k$ show that adjoining the vector $r_{k+1}$ or adjoining the vector $p_{k+1}$ produces the same span. Consequently, the identity

$$
\mathrm{span}\{r_0,\ldots,r_{k+1}\}=\mathrm{span}\{p_0,\ldots,p_{k+1}\}
$$

holds.

If the residual $r_{k+1}$ is nonzero, then property 1 shows that the vectors $r_0,\ldots,r_{k+1}$ are mutually orthogonal and therefore linearly independent. Their span therefore has dimension $k+2$. Since this span is contained in $\mathcal{K}_{k+2}(A,r_0)$, and since the Krylov subspace $\mathcal{K}_{k+2}(A,r_0)$ is generated by the $k+2$ vectors $r_0, Ar_0,\ldots,A^{k+1}r_0$, the dimension of $\mathcal{K}_{k+2}(A,r_0)$ is at most $k+2$. Hence the inclusion above is in fact an equality, and property 3 follows. If instead the residual $r_{k+1}$ vanishes, then the iterate $x_{k+1}$ already equals the minimizer and all later statements are trivial. Therefore property 3 holds in every valid case.

We now verify property 2. The residual recursion implies the identity

$$
Ap_j=\frac{r_j-r_{j+1}}{\eta_j}.
$$

For every index $j<k$, property 1 gives the orthogonality relations $r_{k+1}^\top r_j=0$ and $r_{k+1}^\top r_{j+1}=0$. Substituting these relations into the identity above yields the relation

$$
r_{k+1}^\top Ap_j = \frac{r_{k+1}^\top r_j-r_{k+1}^\top r_{j+1}}{\eta_j}=0.
$$

Combining the preceding relation with the inductive hypothesis $p_k^\top Ap_j=0$ yields the identity

$$
p_{k+1}^\top Ap_j = r_{k+1}^\top Ap_j + \beta_k\,p_k^\top Ap_j = 0.
$$

For the remaining index $j=k$, the same identity with index $k$ gives the relation

$$
r_{k+1}^\top Ap_k = \frac{r_{k+1}^\top r_k-\|r_{k+1}\|^2}{\eta_k} = -\frac{\|r_{k+1}\|^2}{\eta_k},
$$

where the orthogonality relation $r_{k+1}^\top r_k=0$ comes from property 1. Using again the definitions of $\beta_k$ and $\eta_k$, we obtain the calculation

$$
\beta_k\,p_k^\top Ap_k = \frac{\|r_{k+1}\|^2}{\|r_k\|^2}\cdot \frac{\|r_k\|^2}{\eta_k}=\frac{\|r_{k+1}\|^2}{\eta_k}.
$$

Combining the preceding two identities yields the relation

$$
p_{k+1}^\top Ap_k = r_{k+1}^\top Ap_k + \beta_k\,p_k^\top Ap_k = 0.
$$

Therefore, the search directions remain $A$-conjugate.

It remains to prove optimality. Recall that a point $x_k$ minimizes the strictly convex quadratic $f$ over an affine subspace $x_0+V$ if and only if the gradient $\nabla f(x_k)$ is orthogonal to the subspace $V$. Since the negative gradient equals the residual $r_k = -\nabla f(x_k)$, the optimality condition reads

$$
r_k \perp \mathcal{K}_k(A,r_0).
$$

By property 3, the Krylov subspace $\mathcal{K}_k(A,r_0)$ coincides with $\mathrm{span}\lbrace r_0,\ldots,r_{k-1}\rbrace$. Property 1 states that $r_k$ is orthogonal to every earlier residual $r_j$ with $j<k$. Combining these two facts shows that the optimality condition above holds, and therefore $x_k$ minimizes $f$ over $x_0+\mathcal{K}_k(A,r_0)$. This completes the proof. <span style="float: right;">$\square$</span>

### Visualizing CG

The animation below shows CG on a 2D quadratic. Notice that it converges in exactly 2 steps (the dimension of the problem). The red arrows indicate the search directions, which are $A$-conjugate: they are orthogonal with respect to the inner product $\langle u, v \rangle_A = u^\top A v$, not with respect to the standard dot product.

![Conjugate Gradient method](figures/conjugate_gradient.gif)

---

## 5. Convergence Comparison

The animation below compares the convergence of all three methods on the same ill-conditioned quadratic, plotting the relative suboptimality $f(x_k)/f(x_0)$ on a logarithmic scale.

- **Gradient descent** (blue): linear convergence with iteration complexity $O(\kappa\,\ln(1/\varepsilon))$.
- **Chebyshev GD** (red): accelerated convergence with iteration complexity $O(\sqrt{\kappa}\,\ln(1/\varepsilon))$, but requires knowledge of $\alpha$ and $\beta$.
- **Conjugate Gradients** (green): matches the Chebyshev rate adaptively and terminates in at most $d$ steps.

![Convergence comparison](figures/convergence_comparison.gif)

---

## 6. The Positive Semidefinite Case

### Motivation

The convergence guarantees of the previous sections all rely on the assumption $\alpha > 0$---that is, $A$ is positive definite. When $\alpha = 0$, so that $A$ is positive semidefinite but singular, the condition number $\kappa = \beta/\alpha$ is infinite and the linear convergence bounds of Theorems 1, 2, and 3 become vacuous. This situation arises naturally in practice: in linear least squares with $A = D^\top D$, the matrix $A$ is singular whenever the data matrix $D \in \mathbb{R}^{m \times d}$ has fewer rows than columns ($m < d$), a common setting in modern high-dimensional statistics.

Despite the infinite condition number, gradient descent still makes progress---the function value continues to decrease, even though the error vector $e_k$ may fail to converge in the Euclidean norm. A more refined analysis reveals two phenomena: gradient descent with a fixed stepsize converges at a **sublinear** $O(1/k)$ rate, while the conjugate gradient method maintains **linear** convergence by adapting automatically to the nonzero spectrum.

### Setup

We consider the same quadratic objective $f(x) = \tfrac{1}{2}x^\top Ax - b^\top x$, but now with $\alpha = 0$. We assume throughout that $b \in \mathrm{range}(A)$, ensuring that the solution set

$$S = \{x \in \mathbb{R}^d : Ax = b\}$$

is nonempty. The eigenvalues of $A$ are ordered as

$$0 = \lambda_1 = \cdots = \lambda_r < \lambda_{r+1} \leq \cdots \leq \lambda_d = \beta,$$

where $r \geq 1$ is the dimension of $\ker(A)$, and we denote the distinct nonzero eigenvalues by $\mu_1 < \mu_2 < \cdots < \mu_m$.

A fundamental difference from the positive definite case is that the solution set $S$ is no longer a singleton---it is an affine subspace of dimension $r$. For a given starting point $x_0$, we write

$$x^\star = \mathrm{proj}_S(x_0)$$

for the closest solution to $x_0$. The initial error $e_0 = x_0 - x^\star$ then lies entirely in $\mathrm{range}(A)$, so its expansion in the eigenbasis has $c_i = 0$ for all $i \leq r$.

### Why linear rates fail

The failure of linear convergence is immediate from the eigenvalue expansion. With any fixed stepsize $\eta \in (0, 2/\beta)$, the contraction factor satisfies

$$\rho(\eta) = \max_{1 \leq i \leq d}\lvert 1 - \eta\lambda_i\rvert \geq \lvert 1 - \eta \cdot 0\rvert = 1.$$

The bound $f(x_k) - f(x^\star) \leq \rho(\eta)^{2k}(f(x_0) - f(x^\star))$ therefore gives only the trivial estimate $f(x_k) \leq f(x_0)$. The underlying issue is that the contraction-rate analysis treats all eigenvalues uniformly, ignoring the crucial fact that the components corresponding to zero eigenvalues do not contribute to the function value.

In the polynomial framework of Section 3, the constraint $p_k(0) = 1$ forces $\max_{\lambda \in [0,\beta]}\lvert p_k(\lambda)\rvert \geq 1$ for every polynomial $p_k$, so the minimax approach on the full interval $[0, \beta]$ yields no convergence guarantee. The path forward is to exploit the eigenvalue weighting in the $A$-norm: the factor $\lambda_i$ in the sum $\sum_i \lambda_i\,p_k(\lambda_i)^2\,c_i^2$ suppresses contributions from eigenvalues near zero. Rather than bounding $(1 - \eta\lambda_i)^{2k}$ alone and pulling it out of the sum, we instead bound the product $\lambda_i(1 - \eta\lambda_i)^{2k}$ directly.

### Sublinear convergence of gradient descent

<div style="background-color: #eef6fc; border-left: 4px solid #2980b9; padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">

**Theorem 5 (Sublinear convergence of gradient descent).** *Let $A$ be positive semidefinite with largest eigenvalue $\beta > 0$, and suppose $b \in \mathrm{range}(A)$. With stepsize $\eta = 1/\beta$, the gradient descent iterates satisfy*

$$f(x_k) - f(x^\star) \leq \frac{\beta}{2(2k+1)}\,\|x_0 - x^\star\|^2, \tag{5}$$

*where $x^\star = \mathrm{proj}_S(x_0)$ is the closest solution to $x_0$.*

</div>

*Proof.* Since $x^\star = \mathrm{proj}_S(x_0)$, the initial error $e_0 = x_0 - x^\star$ lies in $\mathrm{range}(A)$, so $c_i = 0$ for all $i \leq r$. The function value gap therefore satisfies

$$
f(x_k) - f(x^\star) = \frac{1}{2}\sum_{i=r+1}^{d}\lambda_i(1-\lambda_i/\beta)^{2k}\,c_i^2 \leq \frac{1}{2}\max_{\lambda \in (0,\beta]}\lambda(1-\lambda/\beta)^{2k}\cdot\|e_0\|^2.
$$

It remains to compute the maximum. Define $h(t) = t(1-t)^{2k}$ for $t \in [0,1]$ and substitute $t = \lambda/\beta$. Differentiating gives

$$h'(t) = (1-t)^{2k-1}\bigl(1 - (2k+1)t\bigr),$$

so the unique maximizer on $[0,1]$ is $t^\star = 1/(2k+1)$. The maximum value is

$$
h(t^\star) = \frac{1}{2k+1}\left(\frac{2k}{2k+1}\right)^{2k} \leq \frac{1}{2k+1},
$$

where the inequality uses $(1-1/n)^{n-1} \leq 1$ for all integers $n \geq 2$, applied with $n = 2k+1$. Therefore

$$\max_{\lambda \in (0,\beta]}\lambda(1-\lambda/\beta)^{2k} = \beta\,h(t^\star) \leq \frac{\beta}{2k+1}.$$

Combining the preceding two estimates yields the conclusion $(5)$. <span style="float: right;">$\square$</span>

### Iteration complexity

From Theorem 5, the bound $f(x_k) - f(x^\star) \leq \varepsilon$ is guaranteed whenever

$$k \geq \frac{\beta\,\|x_0 - x^\star\|^2}{4\varepsilon}.$$

The iteration complexity is therefore

$$O\!\left(\frac{\beta\,\|x_0 - x^\star\|^2}{\varepsilon}\right).$$

Compared with the $O(\kappa\,\ln(1/\varepsilon))$ complexity of gradient descent in the positive definite case, the dependence on accuracy has changed from **logarithmic** to **polynomial**: achieving an additional digit of accuracy now requires a tenfold increase in iterations, rather than a fixed additive cost. This is the hallmark of sublinear convergence.

### The role of the initial condition

An important feature of Theorem 5 is that the convergence bound involves the squared Euclidean distance $\|x_0 - x^\star\|^2$ rather than the initial function gap $f(x_0) - f(x^\star)$.

This change is unavoidable. Since $f(x_0) - f(x^\star) = \tfrac{1}{2}\sum_{i>r}\lambda_i\,c_i^2$ and $\|e_0\|^2 = \sum_{i>r}c_i^2$, the ratio $(f(x_0) - f(x^\star))/\|e_0\|^2$ can be arbitrarily small when the initial error concentrates along eigenvectors with small nonzero eigenvalues. Consequently, no bound of the form

$$f(x_k) - f(x^\star) \leq g(k)\bigl(f(x_0) - f(x^\star)\bigr) \tag{6}$$

with $g(k) \to 0$ can hold uniformly over all starting points when $\alpha = 0$. The Euclidean distance $\|x_0 - x^\star\|$ is the natural measure of initial error in the positive semidefinite setting.

### Acceleration by Chebyshev stepsizes

As in the positive definite case, the $O(1/k)$ rate of fixed-stepsize gradient descent can be improved by varying the stepsize across iterations. The polynomial viewpoint makes this transparent: gradient descent with stepsizes $\eta_1, \ldots, \eta_k$ produces the polynomial $p_k(\lambda) = \prod_{j=1}^{k}(1-\eta_j\lambda)$, and any degree-$k$ polynomial with $p(0) = 1$ can be realized by choosing the stepsizes as the reciprocals of its roots. Finding the optimal time-varying stepsizes is therefore equivalent to finding the degree-$k$ polynomial $p$ with $p(0) = 1$ that minimizes $\max_{\lambda \in (0,\beta]} \lambda\,p(\lambda)^2$.

The solution involves the **Chebyshev polynomial of the second kind** of degree $j$, denoted $U_j$, which is characterized by the identity

$$U_j(\cos\theta) = \frac{\sin\bigl((j+1)\theta\bigr)}{\sin\theta},$$

with the evaluation $U_j(1) = j+1$. See the figure below.

![Chebyshev polynomials of the second kind](figures/chebyshev_polynomials_2nd.png)

Just as the Chebyshev polynomial $T_k$ of the first kind solves the minimax problem on $[\alpha, \beta]$ in the positive definite case, the polynomial $U_{k-1}$ plays the analogous role in the positive semidefinite setting. Its roots on $[-1,1]$ are $\cos(j\pi/k)$ for $j = 1, \ldots, k-1$, which under the substitution $x = 1 - 2\lambda/\beta$ correspond to $\lambda_j = \beta\sin^2(j\pi/(2k))$.

<div style="background-color: #eef6fc; border-left: 4px solid #2980b9; padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">

**Theorem 6 (Chebyshev stepsizes, PSD case).** *Let $A$ be positive semidefinite with largest eigenvalue $\beta > 0$, and suppose $b \in \mathrm{range}(A)$. Define the stepsizes*

$$\eta_j = \frac{1}{\beta\sin^2(j\pi/(2k))} \qquad \textrm{for}~ j = 1, \ldots, k.$$

*Then the gradient descent iterates satisfy*

$$f(x_k) - f(x^\star) \leq \frac{\beta}{8k^2}\,\|x_0 - x^\star\|^2. \tag{7}$$

</div>

This is a **quadratic improvement** over the $O(1/k)$ rate of Theorem 5: where fixed-stepsize gradient descent requires $O(1/\varepsilon)$ iterations to reach accuracy $\varepsilon$, the Chebyshev stepsizes require only $O(1/\sqrt{\varepsilon})$.

*Proof.* The stepsizes $\eta_1, \ldots, \eta_k$ produce the degree-$k$ polynomial

$$\phi_k(\lambda) = \prod_{j=1}^{k}\bigl(1 - \eta_j\lambda\bigr) = \prod_{j=1}^{k}\left(1 - \frac{\lambda}{\beta\sin^2(j\pi/(2k))}\right).$$

Since $\sin^2(k\pi/(2k)) = 1$, the factor corresponding to $j = k$ is $(1 - \lambda/\beta)$. The remaining roots $\beta\sin^2(j\pi/(2k))$ for $j = 1, \ldots, k-1$ are precisely the images of the roots of $U_{k-1}$ under the substitution $x = 1 - 2\lambda/\beta$. Therefore

$$\phi_k(\lambda) = \left(1 - \frac{\lambda}{\beta}\right)\frac{U_{k-1}(1 - 2\lambda/\beta)}{k},$$

where the factor $1/k$ ensures $\phi_k(0) = 1$ (since $U_{k-1}(1) = k$). We claim that

$$\max_{\lambda \in (0,\beta]}\,\lambda\,\phi_k(\lambda)^2 \leq \frac{\beta}{4k^2}. \tag{8}$$

To verify $(8)$, substitute $\cos\theta = 1 - 2\lambda/\beta$ for $\theta \in [0,\pi]$, so that $\lambda = \beta\sin^2(\theta/2)$ and $1 - \lambda/\beta = \cos^2(\theta/2)$. The Chebyshev identity gives $U_{k-1}(\cos\theta) = \sin(k\theta)/\sin\theta$, and the factorization $\sin\theta = 2\sin(\theta/2)\cos(\theta/2)$ yields

$$
\begin{aligned}
\lambda\,\phi_k(\lambda)^2
&= \beta\sin^2(\theta/2)\cdot\cos^4(\theta/2)\cdot\frac{\sin^2(k\theta)}{k^2\sin^2\theta} \\[4pt]
&= \beta\sin^2(\theta/2)\cdot\cos^4(\theta/2)\cdot\frac{\sin^2(k\theta)}{4k^2\sin^2(\theta/2)\cos^2(\theta/2)} \\[4pt]
&= \frac{\beta\cos^2(\theta/2)\,\sin^2(k\theta)}{4k^2}.
\end{aligned}
$$

Since $\cos^2(\theta/2) \leq 1$ and $\sin^2(k\theta) \leq 1$ for all $\theta$, the claim $(8)$ follows. The error bound from Section 2 then gives

$$f(x_k) - f(x^\star) = \tfrac{1}{2}\|e_k\|_A^2 \leq \frac{1}{2}\cdot\frac{\beta}{4k^2}\cdot\|e_0\|^2 = \frac{\beta}{8k^2}\,\|x_0 - x^\star\|^2.$$

This completes the proof. <span style="float: right;">$\square$</span>

The iteration complexity is $O(\sqrt{\beta\,\|x_0 - x^\star\|^2/\varepsilon})$---a square-root improvement over the $O(\beta\,\|x_0 - x^\star\|^2/\varepsilon)$ complexity of fixed-stepsize gradient descent. The mechanism behind this acceleration is visible in the polynomial $\phi_k$: the factor $(1 - \lambda/\beta)$ suppresses the contribution from the largest eigenvalue, while the Chebyshev factor $U_{k-1}(1 - 2\lambda/\beta)/k$ distributes the remaining approximation power efficiently across the spectrum. The combined effect is the $1/k^2$ rate. By contrast, the fixed-stepsize polynomial $(1 - \lambda/\beta)^k$ must use all $k$ degrees of freedom to suppress the largest eigenvalue, leaving none to exploit the eigenvalue weighting---this is why gradient descent is limited to a $1/k$ rate.

As in the positive definite case, the Chebyshev stepsizes require knowledge of $\beta$ and the total number of iterations $k$ must be fixed in advance.

### Conjugate gradients in the positive semidefinite case

The conjugate gradient method matches the $O(1/k^2)$ rate of the Chebyshev stepsizes *adaptively*, without requiring knowledge of $\beta$ or a preset iteration count. As in the positive definite case, CG achieves this by optimizing over the full Krylov subspace.

The correctness guarantee of Theorem 4 continues to hold: at each step, $x_k$ minimizes $f$ over $x_0 + \mathcal{K}_k(A, r_0)$. The only condition needed beyond those in the positive definite case is that $p_k^\top Ap_k > 0$ at each step. Since $b \in \mathrm{range}(A)$ and $Ax_0 \in \mathrm{range}(A)$, the initial residual $r_0 = b - Ax_0$ lies in $\mathrm{range}(A)$. The Krylov subspace $\mathcal{K}_k(A, r_0) = \mathrm{span}\{r_0, Ar_0, \ldots, A^{k-1}r_0\}$ is therefore contained in $\mathrm{range}(A)$, and so every search direction $p_k$ lies in $\mathrm{range}(A)$. Since $A$ is positive definite on its range, the inequality $p_k^\top Ap_k > 0$ holds whenever $p_k \neq 0$.

<div style="background-color: #eef6fc; border-left: 4px solid #2980b9; padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">

**Theorem 7 (CG convergence, PSD case).** *Let $A$ be positive semidefinite with largest eigenvalue $\beta > 0$, and suppose $b \in \mathrm{range}(A)$. Let $\mu_1 < \mu_2 < \cdots < \mu_m$ denote the distinct nonzero eigenvalues of $A$. The CG iterates satisfy*

$$f(x_k) - f(x^\star) \leq \frac{\beta}{8k^2}\,\|x_0 - x^\star\|^2, \tag{9}$$

*and CG terminates in at most $m$ iterations.*

</div>

*Proof.* By Theorem 4, the CG iterate $x_k$ minimizes $f$ over $x_0 + \mathcal{K}_k(A, r_0)$. Since $e_0 \in \mathrm{range}(A)$, the polynomial representation gives

$$
\|e_k\|_A^2 = \min_{\substack{p \in \mathcal{P}_k \\ p(0)=1}} \|p(A)\,e_0\|_A^2 \leq \min_{\substack{p \in \mathcal{P}_k \\ p(0)=1}}\; \max_{\lambda \in (0,\beta]}\, \lambda\,p(\lambda)^2 \cdot \|e_0\|^2.
$$

The polynomial $\phi_k$ from the proof of Theorem 6 is a feasible choice. Applying the estimate $(8)$ yields

$$f(x_k) - f(x^\star) = \tfrac{1}{2}\|e_k\|_A^2 \leq \frac{\beta}{8k^2}\,\|x_0 - x^\star\|^2.$$

For finite termination, the polynomial $p(\lambda) = \prod_{j=1}^{m}(1 - \lambda/\mu_j)$ has degree $m$, satisfies $p(0) = 1$, and vanishes at every nonzero eigenvalue of $A$. Since $e_0 \in \mathrm{range}(A)$, the identity $p(A)\,e_0 = 0$ follows, and therefore $\|e_m\|_A = 0$. <span style="float: right;">$\square$</span>

*Remark.* CG also enjoys a linear rate of convergence in the positive semidefinite case. Applying the Chebyshev polynomial $T_k$ on $[\mu_1, \beta]$ as in the proof of Theorem 3 and using Lemma 1 with $\kappa' = \beta/\mu_1$ gives

$$f(x_k) - f(x^\star) \leq 4\left(\frac{\sqrt{\kappa'}-1}{\sqrt{\kappa'}+1}\right)^{2k}\bigl(f(x_0) - f(x^\star)\bigr).$$

The sublinear bound $(9)$ is most useful when the smallest nonzero eigenvalue $\mu_1$ is very small or unknown, while the linear bound becomes tighter once $k$ is large relative to $\sqrt{\kappa'}$.

### Discussion

The parallel between the positive definite and positive semidefinite cases is now complete. In both settings, the three methods exhibit the same hierarchy: gradient descent provides a baseline rate, Chebyshev stepsizes achieve a square-root acceleration, and CG matches the accelerated rate adaptively. The table below summarizes the iteration complexities.

| | PD case ($\alpha > 0$) | PSD case ($\alpha = 0$) |
|---|---|---|
| Gradient descent | $O(\kappa\,\ln(1/\varepsilon))$ | $O(\beta\,\|x_0-x^\star\|^2/\varepsilon)$ |
| Chebyshev GD | $O(\sqrt{\kappa}\,\ln(1/\varepsilon))$ | $O(\sqrt{\beta\,\|x_0-x^\star\|^2/\varepsilon})$ |
| Conjugate Gradients | $O(\sqrt{\kappa}\,\ln(1/\varepsilon))$ | $O(\sqrt{\beta\,\|x_0-x^\star\|^2/\varepsilon})$ |

In every row, the Chebyshev and CG methods improve the gradient descent complexity by a square root. CG additionally terminates in at most $m$ iterations---the number of distinct nonzero eigenvalues---which may be much smaller than the ambient dimension $d$.

---

## Summary

**Positive definite case** ($\alpha > 0$):

| Method | Per-step cost | Iteration complexity | Requires $\alpha, \beta$? |
|--------|--------------|---------------------|-------------------|
| Gradient descent | One matvec | $O(\kappa\,\ln(1/\varepsilon))$ | Yes (for optimal step) |
| Chebyshev GD | One matvec | $O(\sqrt{\kappa}\,\ln(1/\varepsilon))$ | Yes |
| Conjugate Gradients | One matvec | $O(\sqrt{\kappa}\,\ln(1/\varepsilon))$, at most $d$ steps | No |

**Positive semidefinite case** ($\alpha = 0$):

| Method | Per-step cost | Iteration complexity | Rate type |
|--------|--------------|---------------------|-----------|
| Gradient descent | One matvec | $O(\beta\,\|x_0 - x^\star\|^2/\varepsilon)$ | Sublinear $O(1/k)$ |
| Chebyshev GD | One matvec | $O(\sqrt{\beta\,\|x_0 - x^\star\|^2/\varepsilon})$ | Sublinear $O(1/k^2)$ |
| Conjugate Gradients | One matvec | $O(\sqrt{\beta\,\|x_0 - x^\star\|^2/\varepsilon})$, at most $m$ steps | Sublinear $O(1/k^2)$ |

The key takeaway: on quadratics, the Chebyshev and CG methods achieve a square-root improvement over gradient descent in *every* regime---whether measured by the condition number $\kappa$ in the positive definite case or by the iteration complexity in the positive semidefinite case. CG accomplishes this *adaptively*, without needing to know the eigenvalues or fixing the iteration count in advance, and terminates (in exact precision) in a number of steps bounded by the number of distinct nonzero eigenvalues of $A$. In practice CG doesn't exactly terminate after finitely many steps due to compounding of numerical errors.

---

[← Back to course page](./)
