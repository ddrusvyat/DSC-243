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
- [Notes](#notes)
- [Course outline](#course-outline)
- [Grading policy](#grading-policy)
- [Textbooks and references](#textbooks-and-references)

## Course summary

This course develops the foundations of the modern optimization methods that form the backbone of machine learning and AI. Building on the prerequisite introductory course in optimization (DSC 211), it takes the **convex quadratic optimization problem** as its central object of study. Optimization of quadratics is the most fundamental problem in numerical optimization, yet many of the phenomena for this particular problem class have direct analogues for highly nonlinear and complex models (e.g. deep learning). Since the objective is quadratic, we can develop sharp intuition for the design, analysis, and practical behavior of algorithms using only basic linear-algebraic tools. We will study both worst-case convergence rates—depending only on extremal eigenvalues—and more refined guarantees that account for the interaction between initialization and the shape of the entire spectrum, covering gradient descent, Chebyshev acceleration, conjugate gradients, stochastic gradient methods, lower bounds, and the high-dimensional limits that connect these ideas to large-scale machine learning. The emphasis throughout is on gaining both theoretical insight and algorithmic intuition—understanding *why* methods work, how to choose between them, and how they scale to real-world settings.

## Pre-requisites

- Introductory course on optimization (DSC 211 or equivalent)
- Comfort with advanced linear algebra and multivariate calculus

## Notes

The notes for the course are being developed. Here is the latest draft: [notes](week1.html).

## Course outline

{::nomarkdown}
<table>
<thead>
<tr><th>Week(s)</th><th>Topics</th><th>Notes</th></tr>
</thead>
<tbody>
<tr><td>1–3</td><td><ul><li>Gradient descent</li><li>Acceleration by Chebyshev stepsizes</li><li>Conjugate Gradient method</li></ul></td><td><a href="part1.html">Lecture notes</a></td></tr>
<tr><td>4–6</td><td><ul><li>Improved rates under source and spectral conditions (e.g. Kernel regression)</li><li>Average case analysis (Marchenko Pastur Law)</li><li>Stochastic gradient method</li><li>Acceleration of SGD by averaging</li></ul></td><td><a href="part2.html">Lecture notes</a></td></tr>
<tr><td>7-8</td><td><ul><li>Lower bound for deterministic algorithms</li><li>Lower bounds for deterministic algorithms with a prescribed spectral measure</li><li>Lower bounds for stochastic optimization</li></ul></td><td><a href="part3.html">Lecture notes</a></td></tr>
<tr><td>9-10</td><td><ul><li>High-dimensional limit of SGD and consequences</li></ul></td><td><a href="part4.html">Lecture notes</a></td></tr>
</tbody>
</table>
{:/nomarkdown}


## Grading policy

All grades will be based on homework sets, which will be due roughly every two weeks.


## Canvas

 I will use Canvas only for communicating by email with the class. All course material will appear on this webpage.


## Textbooks and references

This course is largely self-contained with no required textbook. Specific results are cited inline in the [lecture notes](part1.html); the sources below are background and follow-up reading, grouped by the topics they support.

**First-order and stochastic optimization** (gradient descent, acceleration, SGD).

- A. Beck. [*Introduction to Nonlinear Optimization – Theory, Algorithms and Applications.*](https://epubs.siam.org/doi/book/10.1137/1.9781611977622) SIAM, 2014.
- S. Bubeck. [*Convex Optimization: Algorithms and Complexity.*](https://arxiv.org/pdf/1405.4980.pdf) 2015.
- J. Duchi. [*Introductory Lectures on Stochastic Convex Optimization.*](https://stanford.edu/~jduchi/PCMIConvex/Duchi16.pdf) 2016.
- G. Lan. [*First-order and Stochastic Optimization Methods for Machine Learning.*](https://link.springer.com/book/10.1007/978-3-030-39568-1) Springer, 2020.
- Y. Nesterov. [*Introductory Lectures on Convex Optimization: A Basic Course.*](https://link.springer.com/book/10.1007/978-1-4419-8853-9) Springer, 2004.
- B. Recht and S. J. Wright. [*Optimization for Modern Data Analysis.*](https://people.eecs.berkeley.edu/~brecht/opt4ml_book/) Cambridge University Press, 2022.
- A. Sidford. [*Optimization Algorithms.*](https://drive.google.com/file/d/1BfMkt2glaZpJGwg7gwsJw9T_XxH3o8gx/view)

**Krylov subspace methods and conjugate gradients.**

- A. Greenbaum. [*Iterative Methods for Solving Linear Systems.*](https://epubs.siam.org/doi/10.1137/1.9781611970937) SIAM, 1997.
- Y. Saad. [*Iterative Methods for Sparse Linear Systems.*](https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf) 2nd ed., SIAM, 2003.

**Spectral structure, inverse problems, and random matrices.**

- Z. Bai and J. W. Silverstein. [*Spectral Analysis of Large Dimensional Random Matrices.*](https://link.springer.com/book/10.1007/978-1-4419-0661-8) 2nd ed., Springer, 2010.
- H. W. Engl, M. Hanke, and A. Neubauer. [*Regularization of Inverse Problems.*](https://link.springer.com/book/9780792341574) Kluwer, 1996.
- T. Tao. [*Topics in Random Matrix Theory.*](https://terrytao.files.wordpress.com/2011/02/matrix-book.pdf) Graduate Studies in Mathematics, vol. 132, American Mathematical Society, 2012.
- R. Vershynin. [*High-Dimensional Probability: An Introduction with Applications in Data Science.*](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-2.pdf) Cambridge University Press, 2018.

**High-dimensional limits of SGD.**

- E. Paquette. [*High-dimensional limits of stochastic gradient descent.*](https://elliotpaquette.github.io/notes/high-d-limits-sgd-july2023.pdf) Lecture notes, Lehigh University, 2023.

