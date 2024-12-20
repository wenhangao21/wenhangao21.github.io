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

<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/3_unconstrained_methods/expressive.png" style="width: 45%; display: block; margin: 0 auto;" />
</figure>


<div style="display: flex; justify-content: center; align-items: center; border: 2px solid black; padding: 20px; max-width: 600px; margin: 0 auto; text-align: center;">
  <span style="color: red;"><em>Does enforcing equivariance/symmetries as an inductive bias truly offset a potential reduction in optimization diversity within the constrained learning spaces?</em></span>
</div>


It is desirable to have a pipeline that can learn any equivariant functions.

### 1.2. Group Averaging
Consider an arbitrary $\Phi: X \rightarrow Y$, where $X, Y$ are input and output (vector) spaces, respectively.

The GA operator $\langle\Phi\rangle _ G: X \rightarrow Y$ is defined as:

$$
\langle\Phi\rangle _ G(x)=\mathbb{E} _ {g \sim \nu} \rho _ 2(g) \cdot \Phi\left(\rho _ 1(g)^{-1} \cdot x\right) = \int _ G \rho _ 2(g) \cdot \Phi\left(\rho _ 1(g)^{-1} \cdot x\right) d \nu(x),
$$

or in summation form for discrete groups:

$$
\langle\Phi\rangle _ G(x) = \frac{1}{\vert G \vert} \sum _ {g \in G} \rho _ 2(g) \cdot \Phi\left(\rho _ 1(g)^{-1} \cdot x\right).
$$

- $\rho _ 1(g), \rho _ 2(g)$: Group representations on $X$ and $Y$, respectively.

- $\nu$: Harr measure over $G$ ("uniform" over $G$).



> *Claim: The GA operator is equivariant to $G$.*

*Proof:*

$$
\begin{align*}
\langle\Phi\rangle_G(h \cdot x) &= \mathbb{E}_{g \sim \nu} \rho_2(g) \cdot \Phi\left(\rho_1(g)^{-1} \cdot(\rho_1(h) \cdot x)\right) \\
 &= \mathbb{E}_{g \sim \nu} \rho_2(g) \cdot \Phi\left(\rho_1\left(h^{-1} g\right)^{-1} \cdot x\right) \\
 &= \rho_2(h) \mathbb{E}_{g \sim \nu} \rho_2\left(h^{-1} g\right) \cdot \Phi\left(\rho_1\left(h^{-1} g\right)^{-1} \cdot x\right) \\
 &= \rho_2(h)\langle\Phi\rangle_G(x)
\end{align*}
$$


Intuition: Similar to group convolutions, we have already calculated all the transformed versions of the input, $$\rho_1(g)^{-1}x$$ and $$\rho_2(g)$$ "corrects" the output for equivariance.
- $$\left\{\Phi\left(\rho_1(g)^{-1} \cdot x\right), \forall g\right\}$$ will result in the same set of outputs, but in a different order, for transformed inputs.
  - Why? Because the set of inputs $$\left\{\rho_1(g)^{-1} \cdot x\right\}$$ is the same but in a different order for a transformed $x$.
- Thus, integrating/summing over these outputs will result in invariant outputs.
- $$\rho_2(g)$$ "corrects" the output by applying the transformation back.



> *Claim: GA is as expressive as its backbone $\Phi$ when $\Phi$ is equivariant to $G$.*

*Proof:*

$$
\begin{aligned}
\langle\Phi\rangle _ G(x) & =\mathbb{E} _ {g \sim \nu} g \cdot \Phi\left(g^{-1} \cdot x\right) \\
& =\mathbb{E} _ {g \sim \nu} g \cdot g^{-1} \cdot \Phi(x) \\
& =\Phi(x)
\end{aligned}
$$


> Opinion: When $\Phi$ is a general neural network, it is not equivariant. Therefore, this does not imply universal approximation of GA. In the case of FA (will introduce soon), it might even make the learning tasks more difficult.

> Issue: When $\vert G \vert$ is large (e.g., combinatorial groups such as permutations), infinite (e.g., continuous groups such as rotations), or even non-compact, then the exact averaging is intractable.

## 2. Frame Averaging

### 2.1. Motivating Example: Geometric Pre-processing

Consider image segmentation, in which we want translation equivariance. Assuming we are using a model other than CNN, so that we do not have inherent translation equivariance, what can we do if we want translation equivariance? We can use group averaging, but it will be computationally intractable for the translation group (large in the discrete case or even infinite if we view it in a continuous manner). We should think of another way of achieving equivariance.

One way is geometric pre-processing:

Given an image, we can achieve equivariance by preprocessing the image.

<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/3_unconstrained_methods/cat_pre.png" style="width: 35%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;">For example, given cat images, if we have the location of the left eye of a cat, we can preprocess the image such that all cats will have their left eyes in the same location. Therefore, if an image is translated, it will be preprocessed to the same canonical position, eliminating the need for neural networks to have translation symmetries as these are accounted for in the data preprocessing (offset symmetries to the data). </figcaption>
  
### 2.2. Definition

A frame is defined as a set valued function $\mathscr{F}: X \rightarrow 2^G$ (Taking an input $x \in X$ and mapping it to a subset of $G$).

A frame is equivariant to $G$ if $\mathscr{F}(g \cdot x)=g \mathscr{F}(x), ∀g\in G$ (equality of sets).

Frame Averaging operator is defined as:

$$
\langle\Phi\rangle _ {\mathcal{F}}(x)=\frac{1}{\vert \mathcal{F}(x) \vert} \sum _ {g \in \mathcal{F}(x)} \rho_2(g) \Phi\left(\rho_1(g)^{-1} x\right)
$$

- Similar to group averaging, we can prove that frame averaging operator is equivariant to $G$ if $\mathcal{F}$ is equivariant to $G$, and it is as expressive as its backbone if its backbone is equivariant.

> Example: Consider $X=\mathbb{R}^n, Y=\mathbb{R^n}$, and $G=\mathbb{R}$ with addition as the group action. We choose the group actions in this case to be $\rho_1(t) \boldsymbol{x}=\boldsymbol{x}+t \mathbf{1}$, and $\rho_2(a) y=y+t$, where $t\in G$, $\boldsymbol{x} \in X, y \in Y$ are point clouds of $n$ points, and $\mathbf{1} \in \mathbb{R}^n$ is the vector of all ones.  
We can define the frame in this case using the averaging operator $\mathcal{F}(\boldsymbol{x})=\left\{\frac{1}{n} \mathbf{1}^T \boldsymbol{x}\right\} \subset G=\mathbb{R}$.  
Note that in this case the frame contains only one element from the group, in other cases finding such a small frame is hard or even impossible.  
One can check that this frame is equivariant. The FA: $\langle\Phi\rangle_{\mathcal{F}}(\boldsymbol{x})=\Phi\left(\boldsymbol{x}-\frac{1}{n}\left(\mathbf{1}^T x\right) \mathbf{1}\right)+\frac{1}{n} \mathbf{1}^T x$ in the equivariant case.  
- Intuition: Geometric pre-processing, we subtract the average and then add the average back to obtain equivariance.  

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