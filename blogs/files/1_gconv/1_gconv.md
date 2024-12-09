---
layout: blog
title: "Technical Blogs"
author_profile: false
---
<div style="height: 20px;"></div>

# Group Convolution Neural Networks

## Introduction

### Why Symmetries
Group equivariance in ML models is about enforcing symmetries in the architectures.
- Many learning tasks, oftentimes, have symmetries under some set of transformations acting on the data.
	- For example, in image classification, rotating or flipping an image of a cat should not change its classification as a "cat."
- More importantly, the nature itself is about symmetries.
	- Similar symmetries appear in physical systems, molecular structures, and many other scientific data.
	
<figure style="text-align: center;">
  <img alt="Symmetry Diagram" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/symmetry.png" style="width: 50%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;">Figure 1: Symmetries in ML tasks and in nature.</figcaption>

FYI: Dr. Chen Ning Yang from Stony Brook received the Nobel Prize in physics (1957) for discoveries about symmetries, and his B.S. thesis is “Group Theory and Molecular Spectra”.

### Learning Symmetries
To learn symmetry, a common approach is to do data-augmentation: Feed augmented data and hope the model “learns” the symmetry.

<figure style="text-align: center;">
  <img alt="Data Augmentation" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/data_augmentation.png" style="width: 50%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;">Figure 2: Data augmentaton to learn symmetries.</figcaption>

<span style="color: red;">Issues:</span>
- <span style="color: red;">No guarantee</span> of having symmetries in the model
- <span style="color: red;">Wasting valuable net capacity</span> on learning symmetries from data
- <span style="color: red;">Redundancy</span> in learned feature representation

<span style="color: green;">Solution:</span>
- Building symmetries into the model by design! 

## Mathematical Preliminary
### Definition: Group
A **group** $(G, \cdot)$ is a set of elements $G$ equipped with a group product $\cdot$, a binary operator, that satisfies the following four axioms:
- Closure: Given two elements $g$ and $h$ of $G$, the product $g \cdot h$ is also in $G$.
- Associativity: For $g, h, i \in G$ the product $\cdot$ is associative, i.e., $g \cdot(h \cdot i)=(g \cdot h) \cdot i$.
- Identity element: There exists an identity element $e \in G$ such that $e \cdot g=g \cdot e=g$ for any $g \in G$.
- Inverse element: For each $g \in G$ there exists an inverse element $g^{-1} \in G$ s.t. $g^{-1} \cdot g=g \cdot g^{-1}=e$.

**Example:**

The translation group consists of all possible translations in $\mathbb{R}^2$ and is equipped with the group product and group inverse:

$$
\begin{aligned}
g \cdot g^{\prime} & =\left(t+t^{\prime}\right), \quad t, t^{\prime} \in \mathbb{R}^2 \\
g^{-1} & =(-t),
\end{aligned}
$$

with $g=(t), g^{\prime}=\left(t^{\prime}\right)$, and $e=(0,0)$.

### Definition: Representation and Left-regular Representation
A **representation** $\rho: G \rightarrow G L(V)$ is a group homomorphism from $\mathrm{G}$ to the general linear group $G L(V)$. That is, $\rho(g)$ is a linear transformation parameterized by group elements $g \in G$ that transforms some vector $\mathbf{v} \in V$ (e.g. an image or a tensor) such that
$$
\rho\left(g^{\prime}\right) \circ \rho(g)[\mathbf{v}]=\rho\left(g^{\prime} \cdot g\right)[\mathbf{v}]
$$

**This essentially means that we can transfer group structure to other types of objects now, such as vectors or images.**

Note: 
- A **homomorphism** is a structure-preserving map between two algebraic structures of the same type (such as two groups, two rings, or two vector spaces). 
- A **general linear group** is the group of all invertible $d_V \times d_V$ matrices.