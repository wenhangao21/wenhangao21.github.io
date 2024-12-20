---
layout: blog
title: "Unconstrained Methods"
author_profile: false
---

# Unconstrained Methods for Symmetries

**TL;DR:** This blog introduces unconstrained methods for symmetries, taking the $E(3)$ group and point cloud data as an example.

This tutorial aims to **simplify abstract concepts for newcomers**. Coding examples are provided to illustrate PCA on point clouds, frame averaging, and equivariance.

- The toy implementation, along with some slides, can be found [here](https://github.com/wenhangao21/Tutorials/tree/main/Equivariance).
- It is assumed that you are familiar with the [Group CNN](https://wenhangao21.github.io/blogs/files/1_gconv/1_gconv/) and [Geometric GNNs](https://wenhangao21.github.io/blogs/files/2_geometric_GNNs/2_geometric_GNNs/). If not, please read them first.

## 1. Introduction

### 1.1. Motivation

(Constrained) Geometric GNNs enforce symmetries directly into the architecture.
- Restricting its set of possible operations or representations.
- Hindering network capacity to fully express the intricacies of the data.
- Computational inefficient.
- For large pretrained models, such as GPT, we cannot alter their network designs to ensure equivariance!

<div style="display: flex; justify-content: center; align-items: center; border: 2px solid black; padding: 20px; max-width: 600px; margin: 0 auto; text-align: center;">
  <span style="color: red;"><em>Does enforcing equivariance/symmetries as an inductive bias truly offset a potential reduction in optimization diversity within the constrained learning spaces?</em></span>
</div>



<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/3_unconstrained_methods/expressive.png" style="width: 55%; display: block; margin: 0 auto;" />
</figure>

An alternative approach that does not have any of the aforementioned shortcomings is frame/group averaging.

### 1.2. Group Averaging


## 2. Geometric GNNs

### 2.1. GNNs and Geometric Message Passing

<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/2_geometric_GNNs/3d_performance.png" style="width: 40%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;">GNNs that learn with 3D representations outperforms their 2D counterparts by a large margin. </figcaption>


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