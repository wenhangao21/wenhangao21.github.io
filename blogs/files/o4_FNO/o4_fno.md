---
layout: blog
title: "FNO"
author_profile: false
---

# Operator Learning

**TL;DR:** This blog introduces the Fourier Neural Operator [1] and its implementation. 

This tutorial aims to **organize concepts for newcomers**.

## 1. Introduction

### 1.1. Formulation

FNO (Fourier neural operator) is a neural network designed to approximate operators $\mathcal{G}$, which are mappings from functions to functions:

$$
\mathcal{G}: a(x) \mapsto u(y).
$$

FNO is a type of integral neural operator. Inspired by the kernel method for PDEs, each integral neural operator layer consists of a fixed non-linearity and a kernel integral operator $$\mathcal{K}$$ modeled by network parameters, defined as $$(\mathcal{K} v)(x)=\int \kappa(x, y) v(y) \mathrm{d} y$$. All operations in integral neural operators are defined on functions; thus, integral neural operators are understood as function space architectures. 

As a natural choice inspired by CNNs and the perspective of fundamental solutions, FNO imposes the integral kernel to be translation invariant, $$\kappa(x, y)=\kappa(x-y)$$. Thus, the kernel integral operator becomes a convolution operator, and FNO performs global convolution in the frequency domain. Each Fourier layer in FNO consists of a fixed non-linearity and a convolution operator $$\mathcal{K}$$ modeled by network parameters:


$$
    (\mathcal{K} v)(x)=\int _ { \mathbb{R}^d} \kappa(x-y) v(y) dy.
$$

Convolution can be efficiently carried out as element-wise multiplication in the frequency domain:

$$
(\mathcal{K} v)(x)=\mathcal{F}^{-1}(\mathcal{F} \kappa \cdot \mathcal{F} v)(x),
$$

where $$\mathcal{F}$$ and $$\mathcal{F}^{-1}$$ are the Fourier transform and its inverse, respectively. FNO directly learns $$\mathcal{F} \kappa$$ in the frequency domain instead of learning the kernel $$\kappa$$ in physical space. The Fourier transform captures global information effectively and efficiently, leading to superior performance for FNO. 

### 1.2. FNO Architecture

Let $$\mathcal{A}\left(\Omega, \mathbb{R}^{d _ a}\right)$$ and $$\mathcal{U}\left(\Omega, \mathbb{R}^{d _ u}\right)$$ be two appropriate function spaces defined on bounded domain $$\Omega \subset \mathbb{R}^d$$. The Fourier neural operator $$\mathcal{G}: \mathcal{A} \rightarrow \mathcal{U}$$ is defined as a compositional mapping between functions spaces:
 
$$
\mathcal{G} (a):= Q \circ \mathcal{L} _ {L} \circ \mathcal{L} _ {L-1} \circ \cdots \circ \mathcal{L} _ {1} \circ P(a).
$$

The input function $$a \in \mathcal{A}$$ is lifted to the latent space of $$R^{d _ {v _ 0}}$$-valued (usually $$d _ {v _ 0} > d _ a$$) functions through a lifting layer acting locally:

$$
P: \left\{a: \Omega \rightarrow \mathbb{R}^{d _ a}\right\} \mapsto\left\{v _ 0: \Omega \rightarrow \mathbb{R}^{d _ {v _ 0}}\right\}.
$$
Lifting layer usually is implemented as a linear layer represented by a matrix $$P\in \mathbb{R}^{d _ v \times d _ a}$$ or as a point-wise multi-layer perceptron ($$1$$ by $$1$$ convolution layers) with activation function $$\sigma$$. 

Then the result goes through $$L$$ Fourier layers:

$$
 \mathcal{L} _ \ell(v)(x)=\sigma\bigg(W _ \ell v(x) + b _ \ell(x) + \mathcal{K} _ {\ell}v(x)
\bigg), \quad \ell = 1, 2, \ldots, L,
$$

where, $$\sigma$$ is a non-linear activation function, $$W _ {\ell}$$ acts locally and can be represented by a matrix $$ \in \mathbb{R}^{d _ {v _ {\ell-1}} \times d _ {v _ \ell}}$$, $$b _ {\ell}(x) \in \mathcal{U}\left(\Omega ; \mathbb{R}^{d _ v}\right)$$ is the bias function (usually a constant function for easy implementation), and 

$$
\mathcal{K} _ {\ell}v(x) = \mathcal{F}^{-1} \Big(P _ {\ell}(k) \cdot \mathcal{F}(v)(k)\Big)(x)
$$

is a linear but non-local convolution operator carried out in the Fourier space with $$P _ {\ell}: \mathbb{Z}^d \rightarrow \mathbb{C}^{d _ v \times d _ v}$$ being the Fourier coefficients of the convolution kernels. We will denote the output of the $$\ell$$-th Fourier layer by $$v _ {\ell}$$.

The output function is obtained from the projection layer acting locally, similar to the lifting layer:

$$
Q: \left\{v _ L: \Omega \rightarrow \mathbb{R}^{d _ {v _ {L}}}\right\} \mapsto\left\{u: \Omega \rightarrow \mathbb{R}^{d _ {u}}\right\}.
$$

For simplicity, we will assume that all the inputs and outputs to the Fourier layers have the same channel dimension, i.e., $$d _ {v _ 0} = d _ {v _ 1} = \ldots d _ {v _ L} = d _ c$$. 

In practice, the FNO, as well as all the intermediate layers, takes discretizations of functions as inputs and produces discretizations of functions as outputs. Therefore, this discrete implementation is terms as the Pseudo Fourier Neural Operators, readers are referred to [2] for more details.

## 2. Implementation

1. Define the `DenseNet` class for both branch and trunk networks (you can also other networks too, e.g. CNN-based networks for regular grid data)

```python
class DenseNet(nn.Module):
    """
    A fully connected neural network (MLP) with ReLU activations between layers, except the last one.
    """
    def __init__(self, layers):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.layers = nn.ModuleList()
        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            if j != self.n_layers - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x
```

2. The branch network takes the input function values at fixed sensor locations as input and the trunk network take spatiotemporal locations as input

```python
class DeepONet(nn.Module):
    def __init__(self, branch_layer, trunk_layer):
        super(DeepONet, self).__init__()
        self.branch = DenseNet(branch_layer)
        self.trunk = DenseNet(trunk_layer)

    def forward(self, a, grid):
        b = self.branch(a)
        t = self.trunk(grid)
        return torch.einsum('bp,np->bn', b, t)

branch_layers = [250, 250, 250, 250, 250]
trunk_layers = [250, 250, 250, 250, 250]
model = DeepONet(branch_layer=[a_num_points] + branch_layers,
                   trunk_layer=[d] + trunk_layers).to(device)
				   
# Note: a_num_points is the number of sensor locations for observing the input function a
# and u_dim is the spatiotemporal dimension of the output function u, e.g. 2 for 2D BVP (no time). 
```	
		
> Full implementation can be found [here](https://github.com/wenhangao21/Tutorials/tree/main/Neural_PDE_Solvers).



On universal approximation and error bounds for Fourier Neural Operators

## References

[1] Fourier Neural Operator for Parametric Partial Differential Equations, Zongyi Li et al.

[2] On Universal Approximation and Error Bounds for Fourier Neural Operators, Nikola Kovachki et al.