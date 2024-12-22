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

A function is a mapping between finite-dimensional vector spaces, e.g. $$f(x) = z$$ for vectors $x$ and $z$.

> Example: $$f(x)=\frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$, $x \in \mathbf{R}$.

An operator is a mapping between infinite-dimensional function spaces, e.g. $$G(a(x))=u(x)$$ for functions $a$ and $u$.

> Examples: Derivative Operator, Nabla Operator, Differential Operator, etc.

In the operator learning setting, we are interested in training a neural network $G_\theta$ such that $$G_\theta(a)\approx G(a)$$ through a given finite collection of observations of input-output pairs $$\left\{a_i, u_i\right\}_{i=1}^N$$, where each $a_i$ and $u_i$ are functions. In practice, the training data is solved numerically or observed in experiments.

<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/o1_operator_learning/operator.png" style="width: 35%; display: block; margin: 0 auto;" />
</figure>

### 1.2. Parametric PDEs and the Learning Task

In numerous fields, we seek to study the behavior of physical systems under various parameters. Neural operators approximate the mapping from parameter function space to solution function space. Once trained, obtaining a solution can be several orders of magnitude faster than numerical methods. A particular example of operator learning is learning parametric PDEs.

Consider a parametric PDE of the form:

$$
\mathcal{N}(a, u)=0,
$$

where $a$ is the input function (can also be a constant, a constant is also a function), and $u$ is the PDE solution. The PDE solution operator is defined as

$$
G(a)=u
$$

such that $(a, u)$ satisfies the PDE. 

> Essentials of operator learning:  
- Domain
  - $\Omega \subset \mathbb{R}^d$ be a bounded open set of spatial coordinates
- Input and output function spaces on $\Omega$ (e.g., Banach spaces, Hilbert spaces)
  - $\mathcal{A}$ and $\mathcal{U}$
- Ground truth solution operator
  - $$G: \mathcal{A} \mapsto \mathcal{U}$$  with $$G(a) = u$$
- Training data
  - Observed (possibly noisy) function pairs $$\left(a_i, u_i\right) \in \mathcal{A} \times \mathcal{U}, u_i=G\left(a_i\right)$$ with measures $$a_i \sim \nu_a, u_i \sim \nu_u$$, where $$\nu_u$$ is the pushforward measure of $$\nu_a$$ by $G$
- Task: Learn operators from data
  - $G _ {\theta}(a)\approx u$
  
A **challenge** in operator learning is that DNNs are mappings between *finite* dimensional spaces: $$\phi _ {\text{network}}: \mathbb{R}^{d _ {in} < \infty} \mapsto \mathbb{R}^{d _ {out} < \infty}$$.

<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/o1_operator_learning/finite_infinite.png" style="width: 65%; display: block; margin: 0 auto;" />
</figure>

## 2. Learning Paradiams

### 2.1. Finite-dimensional Learning
A naive workaround to the challenge is to have a simplified setting in which functions are characterized by finite dimensional features.

> Example 1:  
Let $a$ be a function, and we just take the function values on some sensor locations to be its finite dimensional features.

<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/o1_operator_learning/cnn.png" style="width: 75%; display: block; margin: 0 auto;" />
</figure>
<figcaption style="text-align: center;">Assuming a rectangular domain and uniform sampling of the functions, we can treat it as a (finite-dimensional) image-to-image mapping task and use a CNN-based architecture to learn the mapping. </figcaption>

> Example 2:  
Let $a$ be a function in some function space, let say $a \in L^2(D)$.  
We can write $a$ as an infinite sum of some basis functions, $a=\sum_{k=1}^{\infty} c_k \varphi_k$, where $\varphi_k$ is some basis, e.g., fourier basis.  
We can approximate $a$ with a truncated basis, $a=\sum_{k=1}^{d_y} c_k\varphi_k$.  
Now, the function is characterized by finite dimensinal feasures $c_k$.  

<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/o1_operator_learning/sno.png" style="width: 75%; display: block; margin: 0 auto;" />
</figure>
<figcaption style="text-align: center;">Given appropriate fixed function bases, under fairly general assumptions, functions can be projected into a finite-dimensional space with any desired precision: $f=\sum_ {i=0} ^{k} c_i f_i$. We can learn the mapping between the (finite) coefficients of the input and output functions. A particular work that follows this flow is Spectral Neural Operators. </figcaption>

Many numerical schemes can be represented by this diagram as well.

| Method | Encoder | Approximator | Example Reconstructor |
| --- | --- | --- | --- |
| Finite Difference | Point Values | Numerical Scheme | Polynomial Interpolantion |
| Finite Element | Node Values | Numerical Scheme | Galerkin Basis |
| Finite Volume | Cell Averages | Numerical Scheme | Polynomial Interpolantion |
| Spectral Methods | Fourier Coefs. | Numerical Scheme | Fourier Basis |

How we make choices of encoders, reconstructors (decoders), and approximators gives rise to different neural operators with different pros and cons.

| Method | Encoder | Approximator | Example Reconstructor |
| --- | --- | --- | --- |
| CNN-based Networks | Grid Point Values | DNN | Interpolantion |
| SNO [1] | Fourier/Chebyshev Coefs. | DNN | Fourier/Chebyshev Basis |
| DeepOnet [2] | Sensor Point Values | Branch Net (DNN) | Trunk Net (DNN) |
| PCA-Net [3] | PCA | DNN | PCA |
| IAE-Net [4] | Auto-encoder | DNN | Auto-decoder |
| CORAL [5] | Implicit Neural Representation (DNN) | DNN | Implicit Neural Representation (DNN) |

### 2.2. Infinite-dimensional Learning

For some of the methods we previously discussed, such as CNN-based models, the network is highly dependent on the resolution of the data or sensor locations.

<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/o1_operator_learning/cnn_resolution.png" style="width: 25%; display: block; margin: 0 auto;" />
</figure>
<figcaption style="text-align: center;">In CNN-based methods, fixed-size kernels converge to a point-wise operator as the resolution increases. </figcaption>

Another perspective on operator learning is to think in terms of the continuum.

<div style="display: flex; justify-content: center; align-items: center; border: 2px solid black; padding: 20px; max-width: 600px; margin: 0 auto; text-align: center;">
  <span style="color: red;"><em>Since we are learning an operator, the network should be independent of the discretization of the input and output functions, and the learned parameters should be transferable between discretizations.</em></span>
</div>

In this perspective, we parameterize the model in infinite-dimensional spaces, so it learns continuous functions instead of discretized vectors.

In a standard deep neural network, a layer can be written as:

$$
\text { Input: } v _ {t}
\longrightarrow
\text {Linear Transformation: } W^Tv _ {t}+b
\longrightarrow
\text { Non-linearity } \longrightarrow  \text { Output: }v _ {t+1}
$$

where the input, $v_{t}$, and the output, $v_{t+1}$, are both vectors. 

However, we wish to learn continuous functions instead of discretized vectors. We need to adjust the formulation of our linear layers as it has to be able to take functions as input:

$$v_{t}(x)
\longrightarrow
\text {Integral Linear Operator: } \int\kappa(x, y) v_t(y)dy + b(x)\longrightarrow
\text { Non-linearity } \longrightarrow  v_{t+1}(x)$$

Now our vector $v_t$ is replaced by a function $v_t(x)$. We reformulate the linear layers as *kernel integral operators*. We are able to take inputs at different discretizations (e.g., 128x128 or 256x256) representing the same function, hence allowing the neural operator to be *discretization independent*. It learns the continuous functions instead of discretized vectors.

A standard deep neural network can be written as:

**Instantiations of Integral Neural Operator Layers:**

- GNO (Graph): Assuming a uniform distribution of $y$, the integral can be approximated by a sum

$$v(x) = \int \kappa(x, y) v(y) d y \approx \frac{1}{|N(x)|} \sum _ {y _ i \in N(x)} k\left(x, y _ i\right) v\left(y _ i\right) \approx \frac{1}{|B(x, r)|} \sum _ {y _ i \in B(x, r)} k\left(x, y _ i\right) v\left(y _ i\right).$$

- LNO (Laplace):

$$\sum_{j=1}^r\left\langle\psi^{(j)}, \nu\right\rangle \varphi^{(j)}(x)$$

- FNO (Fourier): Imposing $\kappa(x, y)=\kappa(x-y)$ (translation invariance), which is a natural choice from the perspective of fundamental solutions

$$
\begin{array}{rr}
\text { Integral Linear Operator } & \int \kappa(x, y) v(y) d y \\
\begin{array}{r}
\text { Convolution Operator } \\
\end{array} & \int \kappa(x-y) v(y) d y \\
\text { Solving Convolution in Fourier domain } & \mathcal{F}^{-1}(\mathcal{F}(\kappa) \cdot \mathcal{F}(v))
\end{array}
$$

> For more details and complete yet simple implementations of DeepONet and FNO, you can refer to the following blogs: [DeepONet](https://wenhangao21.github.io/blogs/files/o3_DON/o3_don/) and [FNO](https://wenhangao21.github.io/blogs/files/o4_FNO/o4_fno/).


## References

[1] Spectral neural operators, V. Fanaskov et al.

[2] DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators, Lu Lu et al.

[3] Model reduction and neural networks for parametric PDEs, Kaushik Bhattachary et al.

[4] Integral autoencoder network for discretization-invariant learning, Yong Zheng Ong et al.

[5] Operator learning with neural fields: Tackling PDEs on general geometries, Louis Serrano et al.




## Other Useful Resources for Starters

### Lecture Recordings
1. [Introduction to Scientific Machine Learning - Purdue](https://www.youtube.com/playlist?list=PLUwQEimVf25Q-WjXNQT0aQjupfk70hxlx)
2. [Deep Learning in Scientific Computing 2023 - ETH Zürich](https://www.youtube.com/playlist?list=PLJkYEExhe7rYY5HjpIJbgo-tDZ3bIAqAm)

### Youtube Channels/Talks
1. [Graphs and Geometry Reading Group](https://www.youtube.com/playlist?list=PLoVkjhDgBOt2UwOm70DAuxHf1Jc9ijmzl)
2. [Euclidean Neural Networks for Learning from Physical Systems](https://www.youtube.com/watch?v=ANyOgrnCdGk)
3. [A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems](https://www.youtube.com/watch?v=BUe45d5wrfc)

### Architectures
1. [Geometric GNN Dojo](https://github.com/chaitjo/geometric-gnn-dojo/tree/main) provides unified implementations of several popular geometric GNN architectures