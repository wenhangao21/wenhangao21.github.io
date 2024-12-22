---
layout: blog
title: "Operator Learning"
author_profile: false
---

# Operator Learning

**TL;DR:** This blog briefly introduces the concept of operator learning, which maps between infinite-dimensional function spaces.

This tutorial aims to **organize concepts for newcomers**.

## 1. Preliminary

### 1.1. Functions and Operators

A Function is a mapping between finite-dimensional vector spaces.

> Example: $$f(x)=\frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2}\left(\frac{x-\mu k}{\sigma}\right)^2}$, $x\in\mathbf{R}.$$

An operator is a mapping between infinite-dimensional function spaces, $$G(a(x))=u(x)$$.

> Examples: Derivative Operator, Nabla Operator, Differential Operator, etc..

In the operator learning setting, we are interested in training a neural network $G_\theta$ such that $$G_\theta(f)\approx G(f)$$ through a given finite collection of observations of input-output pairs $$\left\{a_i, u_i\right\}_{i=1}^N$$, where each $a_i$ and $u_i$ are functions. In practice, the training data is solved numerically or observed in experiments.

<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/o1_operator_learning/operator.png" style="width: 55%; display: block; margin: 0 auto;" />
</figure>

### 1.2. Graphs and Geometric Graphs

**Graphs** are purely topological objects and **geometric graphs** are a type of graphs where nodes are additionally endowed with <span style="color: red;">geometric information</span>.

<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/2_geometric_GNNs/geometric_graphs.png" style="width: 55%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;"> Comparison of graphs and geometric graphs. Figure adopted from [1]. </figcaption>



## References

[1] A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems, Duvel et al

[2] SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions, Kristof T. Schütt et al.

[3] Directional Message Passing for Molecular Graphs, Johannes Gasteiger et al.

[4] Spherical Message Passing for 3D Graph Networks, Yi Liu et al.

[5] Artificial Intelligence for Science in Quantum, Atomistic, and Continuum Systems (Section 2), Xuan Zhang (Texas A&M) et al.


## Other Useful Resources for Starters

### Lecture Recordings
1. [First Italian School on Geometric Deep Learning](https://www.youtube.com/playlist?list=PLn2-dEmQeTfRQXLKf9Fmlk3HmReGg3YZZ) (Very nice mathematical prerequisites)
2. [Group Equivariant Deep Learning (UvA - 2022)](https://www.youtube.com/playlist?list=PL8FnQMH2k7jzPrxqdYufoiYVHim8PyZWd)

### Youtube Channels/Talks
1. [Graphs and Geometry Reading Group](https://www.youtube.com/playlist?list=PLoVkjhDgBOt2UwOm70DAuxHf1Jc9ijmzl)
2. [Euclidean Neural Networks for Learning from Physical Systems](https://www.youtube.com/watch?v=ANyOgrnCdGk)
3. [A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems](https://www.youtube.com/watch?v=BUe45d5wrfc)

### Architectures
1. [Geometric GNN Dojo](https://github.com/chaitjo/geometric-gnn-dojo/tree/main) provides unified implementations of several popular geometric GNN architectures