---
layout: blog
title: "Geometric GNNs"
author_profile: false
---

# Physics Informed Neural Networks (PINN) in 5 Minutes

**TL;DR:** This blog introduces the high level ideas of physics informed neural networks [1] in 5 minutes. A simple implementation to solve a Poission's equation with both PINN and finite difference is also provided. 


- The toy implementation can be found [here](https://github.com/wenhangao21/Tutorials/tree/main/Neural_PDE_Solvers).

## 1. PINN Formulation
Consider the following general form of a PDE for $u(\boldsymbol{x})$ :

$$
\begin{cases}\mathcal{D} u(\boldsymbol{x})=f(\boldsymbol{x}), & \text { in } \Omega, \\ \mathcal{B} u(\boldsymbol{x})=g(\boldsymbol{x}), & \text { on } \partial \Omega,\end{cases}
$$

we wish to approximate $u(\boldsymbol{x})$ with a neural network, denoted by $\phi(\boldsymbol{x} ; \boldsymbol{\theta})$. We can train the neural network with physicsinformed loss. That is, we aim to solve the following optimization problem:

$$
\boldsymbol{\theta}^*=\underset{\boldsymbol{\theta}}{\arg \min } \mathcal{L}(\boldsymbol{\theta}):=\underset{\boldsymbol{\theta}}{\arg \min } \underbrace{\|\mathcal{D} \phi(\boldsymbol{x} ; \boldsymbol{\theta})-f(\boldsymbol{x})\| _ 2^2} _ {\begin{array}{c}
\text { Difference between the L.H.S. and } \\
\text { R.H.S. of the differential equation}
\end{array}}+\lambda\|\mathcal{B} \phi(\boldsymbol{x} ; \boldsymbol{\theta})-g(\boldsymbol{x})\| _ 2^2.
$$

> $\mathcal{D} \phi(\boldsymbol{x} ; \boldsymbol{\theta})$ can be approximated using the Monte Carlo method at collocation points.

> Intuition: We penalize the neural network by the extend to which it violates the PDE/boundary/initial conditions. If the residual is exactly $0$, although nearly impossible, it means that $\phi(\boldsymbol{x} ; \boldsymbol{\theta})$ strictly satisfies the governing equations.

## 2. Implementation


## References

[1] Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, M. Raissi et al.


## Other Useful Resources for Starters

### Lecture Recordings
1. [First Italian School on Geometric Deep Learning](https://www.youtube.com/playlist?list=PLn2-dEmQeTfRQXLKf9Fmlk3HmReGg3YZZ) (Very nice mathematical prerequisites)
2. [Group Equivariant Deep Learning (UvA - 2022)](https://www.youtube.com/playlist?list=PL8FnQMH2k7jzPrxqdYufoiYVHim8PyZWd)
3. [Introduction to Functional Analysis - MIT](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/syllabus/)

### Youtube Channels/Talks
1. [ML for Solving PDEs: Neural Operators on Function Spaces by Prof. Anima Anandkumar](https://www.youtube.com/watch?v=y5EJr4ofGOc)
2. [CRUNCH Seminar](https://www.youtube.com/channel/UC2ZZB80udkRvWQ4N3a8DOKQ)
