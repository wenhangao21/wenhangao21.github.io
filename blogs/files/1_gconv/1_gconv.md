---
layout: blog
title: "Technical Blogs"
author_profile: false
---
<div style="height: 20px;"></div>

# Group Convolution Neural Networks

## 1. Introduction

### 1.1 Why Symmetries

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Group equivariance in ML models is about enforcing symmetries in the architectures.
- Many learning tasks, oftentimes, have symmetries under some set of transformations acting on the data.
	- For example, in image classification, rotating or flipping an image of a cat should not change its classification as a "cat."
- More importantly, the nature itself is about symmetries.
	- Similar symmetries appear in physical systems, molecular structures, and many other scientific data.

<figure style="text-align: center;">
  <img alt="Symmetry Diagram" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/symmetry.png" style="width: 50%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;">Figure 1: Symmetries in ML tasks and in nature.</figcaption>

FYI: Dr. Chen Ning Yang from Stony Brook received the Nobel Prize in physics (1957) for discoveries about symmetries, and his B.S. thesis is “Group Theory and Molecular Spectra”.

### 1.2 Learning Symmetries

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To learn symmetries, a common approach is to do data-augmentation: Feed augmented data and hope the model “learns” the symmetry.

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

## 2. Mathematical Preliminary
### 2.1 Definition: Group
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; A **group** $(G, \cdot)$ is a set of elements $G$ equipped with a group product $\cdot$, a binary operator, that satisfies the following four axioms:
- Closure: Given two elements $g$ and $h$ of $G$, the product $g \cdot h$ is also in $G$.
- Associativity: For $g, h, i \in G$ the product $\cdot$ is associative, i.e., $g \cdot(h \cdot i)=(g \cdot h) \cdot i$.
- Identity element: There exists an identity element $e \in G$ such that $e \cdot g=g \cdot e=g$ for any $g \in G$.
- Inverse element: For each $g \in G$ there exists an inverse element $g^{-1} \in G$ s.t. $g^{-1} \cdot g=g \cdot g^{-1}=e$.
<!-- Line breaker -->
<span style="color: gray;">Example:</span>

The translation group consists of all possible translations in $\mathbb{R}^2$ and is equipped with the group product and group inverse:

$$
\begin{aligned}
g \cdot g^{\prime} & =\left(t+t^{\prime}\right), \quad t, t^{\prime} \in \mathbb{R}^2 \\
g^{-1} & =(-t),
\end{aligned}
$$

with $g=(t), g^{\prime}=\left(t^{\prime}\right)$, and $e=(0,0)$.

### 2.2 Definition: Representation and Left-regular Representation
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; A **representation** $\rho: G \rightarrow G L(V)$ is a group homomorphism from $\mathrm{G}$ to the general linear group $G L(V)$. That is, $\rho(g)$ is a linear transformation parameterized by group elements $g \in G$ that transforms some vector $\mathbf{v} \in V$ (e.g. an image or a tensor) such that

$$
\rho\left(g^{\prime}\right) \circ \rho(g)[\mathbf{v}]=\rho\left(g^{\prime} \cdot g\right)[\mathbf{v}].
$$

**This essentially means that we can transfer group structure to other types of objects now, such as vectors or images.**

Note: 
- A **homomorphism** is a structure-preserving map between two algebraic structures of the same type (such as two groups, two rings, or two vector spaces). 
- A **general linear group** is the group of all invertible $d_V \times d_V$ matrices.

A **left-regular representation** $\mathscr{L}_g$ is a representation that transforms functions $f$ by transforming their domains via the inverse group action

$$
\mathscr{L}_g[f](x):=f\left(g^{-1} \cdot x\right).
$$

<span style="color: gray;">Example I:</span>

- $f \in \mathbb{L}_2\left(\mathbb{R}\right)$: A function defined on a line.
- $G=\mathbb{R}$: The 1D translation group.
- $[\mathscr{L}_{g = t}f]~(x)=f\left(t^{-1}_θ \odot x\right) = f(x - t)$: A translation of the function.

<span style="color: gray;">Example II:</span>

- $f \in \mathbb{L}_2\left(\mathbb{R}^2\right)$: A 2D image.
- $G=S E(2)$: The 2D roto-translation group.
- $[\mathscr{L}_{g = (t, \theta)}f]~(x)=f\left(\mathbb{R}^{-1}_θ (x-t)\right)$: A roto-translation of the image.

<span style="color: gray;">Remark: Now we have group stucture on different objects</span>

1. Group Product (acting on $G$ it self): $g\cdot g'$
2. Left Regular Representation (acting on a vector spaces): $\mathscr{L}_gf$
3. Group Actions (acting on $\mathbb{R}^d$): $g \odot x$

### 2.3 Definition: Equivariance and Invariance

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Equivariance** is a property of an operator $\Phi: X \rightarrow Y$ (such as a neural network layer) by which it commutes with the group action:
$$
\Phi \circ \rho^X(g)=\rho^Y(g) \circ \Phi,
$$

**Invariance** is a property of an operator $\Phi: X \rightarrow Y$ (such as a neural network layer) by which it remains unchanged after the group action:

$$
\Phi \circ \rho^X(g)=\Phi,
$$

- $\rho^X(g)$: group representation action on $X$
- $\rho^Y(g)$: group representation action on $Y$
- Invariance is a special case of equivariance when $\rho^Y(g)$ is the identity.

<figure style="text-align: center;">
  <img alt="Invariance and Equivariance" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/invariance_and_equvariance.png" style="width: 50%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;">Figure 4: Invariant task in the left as the classification label remains unchanged after translating the cat. Equivariant task in the right as the localization operator commutes with translation.</figcaption>
  
## 3. CNNs and Translation Equivariance

### 3.1 Convolution and Cross-Correlation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The **convolution** of $f$ and $g$ is written as $f * g$, denoting the operator with the symbol $*$. It is defined as the integral of the product of the two functions after one is reflected and shifted. As such, it is a particular kind of integral transform:

$$
(k * f)(x):=\int_{\mathbb{R}^d} k(x-x')f(x') d x' .
$$

> An equivalent definition is (commutativity):

$$
(k * f)(x):=\int_{\mathbb{R}^d} k(x')f(x-x') d x' .
$$


The **corss-correlation** of $f$ and $g$ is written $f \star g$, denoting the operator with the symbol $\star$. It is defined as the integral of the product of the two functions after one is shifted. As such, it is a particular kind of integral transform:
$$
(k \star f)(x):=\int_{\mathbb{R}^d} k(x'-x)f(x') d x' .
$$
> An equivalent definition is (not commutativity in this case):
$$
(k \star f)(x):=\int_{\mathbb{R}^d} k(x')f(x'+x) d x' .
$$

Note: Neural networks perform the same whether use convolution or correlation because the learned filters enable adaptability. The filters are learned and if a CNN can learn a task using convolution operation, it can also learn the same task using correlation operation (It would learn the rotated version of each filter).

### 3.2 Translation Equivariance

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Convolution and Cross-Correlation are translation equivariant, so are their discrete counterparts.

<span style="color: gray;">Proof:</span>

1. Translate $f$ by $t$ first, then apply the convolution:

$$
(k \star \mathscr{L}_tf)(x)=\int_{\mathbb{R}^d} k(x'-x)[t^{-1} \odot f(x')] d x' = \int_{\mathbb{R}^d} k(x'-x)f(x'-t) d x'.
$$

2. Apply convolution first, and then translate by $t$:

$$
\begin{aligned}
\mathscr{L}_t(k \star f)(x) &= \mathscr{L}_t \left( \int_{\mathbb{R}^d} k(x' - x) f(x') \, dx' \right) \\
&= \int_{\mathbb{R}^d} k(x' - (x - t)) f(x') \, dx' \\
&= \int_{\mathbb{R}^d} k(x' - x + t) f(x') \, dx' \\
&= \int_{\mathbb{R}^d} k(x' - x) f(x' - t) \, dx'.
\end{aligned}
$$

In the last equality, we just replace $x'$ by $x' -t$. Note that this operation is valid because this substitution is a bijection $\mathbb{R}^d \rightarrow \mathbb{R}^d$ and we integrate over the entire $\mathbb{R}^d$.

By similar arguments, we can prove translation equivariance for convolution and the discrete versions.

### 3.3 Intuition on Translation Equivariance

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Mathematically, it is easy to prove translation equivariance. However, let's look at the definiton of cross-correlation again to gain some intution about how to achieve equivariance.

Cross-Correlation:

$$
(k \star f)(x):=\int_{\mathbb{R}^d} k(x'-x)f(x') d x' .
$$

Replace $x'$ by $x'+x$:

$$
(k \star f)(x):=\int_{\mathbb{R}^d} k(x')f(x'+x) d x' .
$$

<span style="color: red;">Intuition: </span> 
- **$f(x'+x)$ represents a translated version of $f(x)$.** We have created many translated version of $f(x)$ while creating the feature map. If we need to compute the cross-correlation for a transformed $f$, we can just go and look up the relevant outputs, because we have already computed them. Equivalently, $k(x'-x)$ represents a translated version of $k(x)$.
- In CNN, we translate the kernel across the image to "scan" the image.

<figure style="text-align: center;">
  <img alt="Convolution" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/CNNkernel.png" style="width: 30%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;">Figure 5: CNN scans through the input by translating the convolution kernels; this is equivalent  to translating the input.</figcaption>
  
### 3.4 Generalization
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Let's look at the definition of cross-correlation:

<figure style="text-align: center;">
  <img alt="Convolution" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/conv_math.png" style="width: 65%; display: block; margin: 0 auto;" />
</figure>

Here, we explicityly think of the cross-correlation in terms of translations. To generalize, if we want to transform $f$ with other groups, the trick is to make the kernel $k$ to be represented by a group. Group representations on $k$ is reflected on $f$ as well.

To generalize to other groups, we should consider the followings:

- Make the function defined on the group of interest.
- Integrate over the group of interest.
- Make the kernel reflect the actions of the group of interest.

## 4. Regular Group CNN and $SE(2)$ Equivariance

### 4.1 Definition: $SE(2)$ Lifting Correlation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To make the function defined on the group of interest, we define the **lifting operation**. The lifting correlation of $f$ and $g$ is written $f \star_{SE(2)} g$, denoting the operator with the symbol $\star_{SE(2)}$. It is defined as the integral of the product of the two functions after one is shifted and rotated. As such, it is a particular kind of integral transform:

<!-- 
$$
(k \star_{SE(2)} f)(x, \theta):=\int_{\mathbb{R}^2} k\Big(\mathbf{R}^{-1}_{\theta}(x'-x)\Big)f(x') d x'  = \int_{\mathbb{R}^2} [\mathscr{L}_{g=(x, \theta)}k(x')]f(x') d x' = \left\langle \mathscr{L}_{g=(x, \theta)}k, f \right\rangle_{\mathbb{L}_2\left(\mathbb{R}^2\right)} .
$$
-->
<figure style="text-align: center;">
  <img alt="Lifting" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/gconv_math.png" style="width: 65%; display: block; margin: 0 auto;" />
</figure>

Lifting correlation raise the feature map to a higher dimension that represents rotation. Now, planar rotation becomes a rotation in the $xy$-axes and a periodic shift (translation) in the $\theta$-axis.

<figure style="text-align: center;">
  <img alt="Lifting" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/lifting.png" style="width: 55%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;">Figure 6: Lifting operation convolves the input with rotated copies of the kernel to reflect the SE(2) group. An additional dimension is included to reflect the rotation angles.</figcaption>
  

### 4.2 Demonstration: Lifting Correlation with the $p_4$ Rotation Group

The $p_4$ group can be described as a semi-direct product:

$$
P_4=C_4 \ltimes \mathbb{Z}^2,
$$

where:
- $C_4$ : The cyclic group of order 4 representing the rotational symmetries.
- $\mathbb{Z}^2$ : The group of translations in the plane (not $\mathbb{R}^2$ because images are discrete).


The lifting operation will simply convolve the input with the kernels rotated by $0^\circ$, $90^\circ$, $180^\circ$, and $270^\circ$, respectively. The result contains $4$ feature maps that correspond to these angles.

```python
def lift_correlation(image, kernel):
    """
    Apply lifting correlation/convolution on an image.

    Parameters:
    - image (numpy.ndarray): The input image as a 2D array, size (s,s)
    - conv_kernel (numpy.ndarray): The convolution kernel as a 2D array.

    Returns:
    - numpy.ndarray: Resulting feature maps after lifting correlation, size (|G|,s,s)
    """
    results = []
    for i in range(4):  # apply rotations to the kernel and convolve with the input
        rotated_kernel = np.rot90(conv_kernel, i)
        result = convolve2d(image, rotated_kernel, mode='same', boundary='symm')
        results.append(result)
    return np.array(results)
```
<figure style="text-align: center;">
  <img alt="Lifting" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/lifted_features.png" style="width: 50%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;">Figure 6: Lifting operation includes an additional dimension to reflect the rotation angles. Now, a rotation in the input will results in a planar rotation in the spatial dimensions and a periodic shift (translation) in the angular dimension. </figcaption>


