---
layout: default
title: Course Overview
---

# DSC 243 – Advanced Optimization

**Instructor:** [Dmitriy Drusvyatskiy](mailto:ddrusvyatskiy@ucsd.edu)

**Lectures:** MWF 10:00–10:50 AM in PCYNH 121

**Office Hour:** Monday 11:00 AM in HDSI 330

## Table of contents

- [Course summary](#course-summary)
- [Pre-requisites](#pre-requisites)
- [Grading policy](#grading-policy)
- [Course outline](#course-outline)
- [Lecture notes](#lecture-notes)
- [Homework](#homework)
- [Textbooks and references](#textbooks-and-references)

## Course summary

This course develops the foundations of the modern optimization methods that form the backbone of machine learning and AI. Building on the prerequisite introductory course in optimization (DSC 211), this course focuses on the design, analysis, and practical behavior of algorithms for large-scale, high-dimensional problems. The emphasis is on gaining both theoretical insight and algorithmic intuition—understanding *why* methods work, how to choose between them, and how they scale to real-world settings. By the end of the course, students will be equipped to think critically about optimization in contemporary applications, from training machine learning models to solving complex decision and inference problems.

## Pre-requisites

- Introductory course on optimization (DSC 211 or equivalent)
- Comfort with advanced linear algebra and multivariate calculus

## Grading policy

All grades will be based on homework sets, which will be due roughly every two weeks.

## Course outline

<table>
<thead>
<tr><th>Week(s)</th><th>Topics</th><th>Lecture Notes</th></tr>
</thead>
<tbody>
<tr><td>1</td><td><ul><li>Gradient descent</li><li>Acceleration by Chebyshev stepsizes</li><li>Conjugate Gradient method</li></ul></td><td></td></tr>
<tr><td>2–3</td><td><ul><li>Proximal gradient descent and accelerated gradient descent for convex optimization</li><li>Lower complexity bounds for convex optimization</li><li>Mirror descent and accelerated mirror descent</li><li>Optimization of relatively smooth functions</li></ul></td><td></td></tr>
<tr><td>4–5</td><td><ul><li>Mirror descent for nonsmooth convex functions</li><li>Frank-Wolfe algorithm as subgradient method in the dual</li></ul></td><td></td></tr>
<tr><td>7</td><td><ul><li>Smoothing algorithms</li><li>Monotone operators and variational inequalities</li><li>Extragradient and Chambolle-Pock algorithms</li></ul></td><td></td></tr>
<tr><td>8</td><td><ul><li>Stochastic gradient method</li><li>Coordinate descent</li><li>Variance reduction: SVRG, SPIDER, STORM</li><li>Adaptive algorithms: ADAGRAD, ADAM, RMSPROP</li></ul></td><td></td></tr>
<tr><td>9–10</td><td><ul><li>SGD and STORM for nonconvex stochastic optimization</li><li>Polyak–Łojasiewicz condition and convergence of gradient systems</li><li>Examples in deep learning, sampling, control, and reinforcement learning</li></ul></td><td></td></tr>
</tbody>
</table>


## Homework

Homework sets will appear here.

## Textbooks and references

This course will largely be self-contained with no required textbook. Relevant sources and follow-up reading material will be cited during the lectures. In particular, I will draw on some material in these sources:

1. A. Beck. [*Introduction to Nonlinear Optimization – Theory, Algorithms and Applications.*](https://epubs.siam.org/doi/book/10.1137/1.9781611977622) SIAM, 2014.
2. S. Bubeck. [*Convex Optimization: Algorithms and Complexity*](https://arxiv.org/pdf/1405.4980.pdf).
3. S. Boyd and L. Vandenberghe. [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/).
4. D. Drusvyatskiy. [*Convex Analysis and Nonsmooth Optimization*](https://sites.math.washington.edu/~ddrusv/crs/Math_516_2021/bookwithindex.pdf).
5. J. Duchi. [*Introductory Lectures on Stochastic Convex Optimization*](https://stanford.edu/~jduchi/PCMIConvex/Duchi16.pdf).
6. Y. T. Lee and S. Vempala. [*Techniques in Optimization and Sampling*](https://github.com/YinTat/optimizationbook/blob/main/main.pdf).
7. G. Lan. [*First-order and Stochastic Optimization Methods for Machine Learning.*](https://link.springer.com/book/10.1007/978-3-030-39568-1) Springer, 2020.
8. Y. Nesterov. [*Introductory Lectures on Convex Optimization: A Basic Course.*](https://link.springer.com/book/10.1007/978-1-4419-8853-9) Springer, 2004.
9. B. Recht and S. J. Wright. [*Optimization for Modern Data Analysis.*](https://people.eecs.berkeley.edu/~brecht/opt4ml_book/) Cambridge University Press, 2022.
10. A. Sidford. [*Optimization Algorithms*](https://drive.google.com/file/d/1BfMkt2glaZpJGwg7gwsJw9T_XxH3o8gx/view).

## Canvas

 I will use Canvas only for communicating by email with the class. All course material will appear on this webpage.
