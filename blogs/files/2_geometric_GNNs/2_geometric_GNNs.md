---
layout: blog
title: "Group CNNs"
author_profile: false
---

# Geometric GNNs

**TL;DR:** This blog introduces geometric GNNs, which guarantee Euclidian (E(n)) symmetries in neural networks; for example, when you rotate a molecular, scalar quantities such as potential energy should remain invariant and vector or tensor quantities should be equivariant to the rotation.

This tutorial aims to **simplify abstract concepts for newcomers**. Coding examples are provided to illustrate concepts including tensor decomposition, equivariance, and irreducibility. 

- The toy implementation along with some slides can be found [here](https://github.com/wenhangao21/Tutorials/tree/main/Equivariance).
- It is assumed that you are familiar with the basic concepts of equivariance. If not, please read [Group CNN](https://wenhangao21.github.io/blogs/files/1_gconv/1_gconv/) first.
- [Reference [1]](https://www.chaitjo.com/publication/duval-2023-hitchhikers/) provides great introduction to geometric GNNs, this blog will introduce geometric GNNs in less details and focus on explaining tensor decomposition, equivariance of tensors, and irreducibility.

## 1. Introduction

### 1.1. Geometric Representation of Atomistic Systems

There are different ways of representing molecules; for example:
- SMILE strings (1D)
- Planar graphs (2D)
- Geometric graphs (3D)

<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/2_geometric_GNNs/representations.png" style="width: 55%; display: block; margin: 0 auto;" />
</figure>

3D geometric configuration (coordinates) is crucial in determining properties and so, GNNs that learn with 3D representations outperforms their 2D counterparts by a large margin.

<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/2_geometric_GNNs/3d_performance.png" style="width: 40%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;">GNNs that learn with 3D representations outperforms their 2D counterparts by a large margin. </figcaption>

### 1.2. Graphs and Geometric Graphs

**Graphs** are purely topological objects and **geometric graphs** are a type of graphs where nodes are additionally endowed with <span style="color: red;">geometric information</span>.

<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/2_geometric_GNNs/geometric_graphs.png" style="width: 55%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;"> Comparison of graphs and geometric graphs. Figure adopted from [1]. </figcaption>

| Graphs | Geometric Graphs |
|$G = (A,S)$ |$G = (A,S,X,V)$ |
|$A \in \mathbb{R}^{n \times n}:$ Adjacency matrix |$A \in \mathbb{R}^{n \times n}:$ Adjacency matrix |
|$S \in \mathbb{R}^{n \times f}$ : Scalar node features |$S \in \mathbb{R}^{n \times f}$ : Scalar node features |
||$X \in \mathbb{R}^{n \times 3}$ : $xyz$-coordinates |
||$V \in \mathbb{R}^{n \times b \times 3}:$ Geometric features, e.g., velocity|


Here,
- Scalar loosely refers to features without geometric information.
- $n$ is the number of nodes, $f$ and $b$ are the sizes of the scalar and geometric node features, respectively.


### 1.3. Symmetries

We have two types of features: <span style="color: blue;">scalar features</span> and <span style="color: red;">geometric features</span>. We have the following symmetries:

- <span style="color: blue;">Scalar features</span> remain unchanged (invariance).
- <span style="color: red;">Geometric features</span> transform with Euclidean transformations of the system (equivariance).
- Graphs,including geometric graphs, are permutationally equivariant node-wise and invariant graph-wise; it is still the same graph even if the nodes are given in a different order.


<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/2_geometric_GNNs/symmetries.png" style="width: 75%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;"> Geometric GNNs should account for all physical symmetries. Figure adopted from [1]. </figcaption>



## 2. Geometric GNNs

### 2.1. GNNs and Geometric Message Passing

Graph Neural Networks (GNNs) are a class of deep learning models designed to operate on graph-structured data by learning node or graph representations through message-passing mechanisms to iteratively update node features to obtain useful hidden representations. In each layer, nodes aggregate information from their neighbors to update their features, allowing GNNs to effectively capture the relational and topological structure of graphs. GNNs are naturally permutation equivariant.

<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/2_geometric_GNNs/GNN.png" style="width: 75%; display: block; margin: 0 auto;" />
</figure>


- Readers who are not familiar with GNNs are refered to [Stanford CS224W: Machine Learning with Graphs](https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn).

For geometric message passing, we condition on geometries. Without loss of generality, let $a_{i j}$ contain geometric information for nodes $i,j$, we can have the following message passing schemes:

$$
\mathbf{m}_{i j}=f_1\left(\mathbf{s}_i, \mathbf{s}_j, a_{ij}\right)
$$



To ensure symmetries
- <span style="color: blue;">Scalar features</span> must be updated in an invariant manner.
- <span style="color: red;">Geometric features</span> must be updated in an equivariant manner.

>　For example, let the relative position be the geometries and $f_1$ be an MLP, the messages $\mathbf{m}_{i j}=f_1\left(\mathbf{s}_i, \mathbf{s}_j, x_j-x_i\right)$ are clearly not equivraiant. 


To make it equivariant (invariant) to $E(3)$, there are in general two directions: <span style="color: blue;">Scalarization</span> and <span style="color: red;">Using Steerable Tensor Features</span>. We term them as <span style="color: blue;">invariant GNNs</span> and <span style="color: red;">equivariant GNNs</span> (Tensor Operations). Invariant GNNs constraint the geometric information that can be utilized, while the other constraints the model operations.

## Scalarization GNNs (Invariant GNNs)

### 2.1. Summarization of Scalarization GNNs

Scalarization networks use invariant quantities as geometries that are conditioned. For example:

- Using relative distances (e.g. SchNet [2]):
	- $\mathbf{m} _{i j}=f_1\left(\mathbf{s}_i, \mathbf{s}_j, d _{i j}\right)$, where $d _{i j}=\left\|x_j-x_i\right\|$
	- $1$-hop, body order $2$, $O(nk)$ to compute invariant quantities with $n$ being the total number of nodes and $k$ being the average degree of a node.
	- This is $E(3)$ invariant, but we limit the expressivity of the model as we cannot distinguish different local geometries. 
	- We cannot distinguish two local neighbourhoods apart using the unordered set of distances only.
	
<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/2_geometric_GNNs/distance.png" style="width: 20%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;">The set of distances are the same, but the graphs are different. Image adopted from [1]. </figcaption>
  
- Using relative distances and bond angles (e.g. DimeNet [3]):
	- $\mathbf{m}_ {i j}=f_1\left(s_ i, s_j, d_ {i j}, \sum_{k \in \mathcal{N}_j \backslash\{i\}} f_3\left(s_j, s_k, d _{ij}, d _{j k},\measuredangle i j k\right)\right)$
	- $2$-hop, body order $3$, $O(nk^2)$ to compute invariant quantities
	- This is $E(3)$ invariant, but again we limit the expressivity of the model due to similar reasons.
	
- Using relative distances, bond angles, and torsion angles (e.g. SphereNet [4]):
	- $\boldsymbol{m} _ {i j}=f_1\left(s_i, s_j, d _ {i j}, \sum_{k \in \mathcal{N}_j \backslash\{i\}, l \in \mathcal{N}_k \backslash\{i, j\}} f_3\left(s_k, s_l, d _ {k l}, d _ {i j}, d _ {j k}, \measuredangle i j k, \measuredangle j k l, \measuredangle i j k l\right)\right)$
	- $3$-hop, body order $4$, $O(nk^3)$ to compute invariant quantities
	- This is $SE(3)$ invariant and complete, meaning that it can uniquely determine the 3D configuration of the geometric graph up to $SO(3)$ transformations (Not $E(3)$ because reflections changes the sign of torsians, you can make it $E(3)$ by ignoring the sign). 
	
### 2.2. Pros and Cons

In summary, invariant GNNs update latent representations by scalarizing local geometry information. This is efficient, and we can achieve invariance with simple MLP without specific constraints on the operations or activations we can take. 

Pros:
- Simple usage of network architecture and non-linearities on many-body scalars.
- Great performance on some use-cases (e.g. GemNet on OC20).

Cons:
- Scalability of scalar’s pre-computation. The accounting of higher-order tuples is expensive. 
- Making invariant predictions may still require solving equivariant sub-tasks.
- May lack generalization capabilities (equivariant tasks, multi-domain).

  
## 3. Spherical Tensor GNNs (Equivariant GNNs)

### 3.1. Introduction 

In invariant GNNs, invariants are 'fixed' prior to message passing. In equivariant GNNs, vector/tensor quantities remain available. Equivariant GNNs can also build up invariants 'on the go' during message passing. More layers of message passing can lead to more complex invariants being built up. 

- In invariant GNNs, we work with only scalars $f\left(s_1, s_2, \ldots, s_n\right)$.

- In equivariant GNNs, we work with vectors $f\left(s_1, s_2, \ldots s_n, \boldsymbol{v}_1, \ldots, \boldsymbol{v}_m\right)$.

Instantiation - "Scalar-vector" GNNs:
- Scalar message:

$$
\mathbf{m}_i:=f_1\left(\mathbf{s}_i,\left\|\mathbf{v} _ {\mathbf{i}}\right\|\right) + \sum _ {j \in \mathcal{N}_i} f_2\left(\mathbf{s}_i, \mathbf{s}_j,\left\|\vec{x} _ {i j}\right\|,\left\|\boldsymbol{v}_j\right\|, \vec{x} _ {i j} \cdot \mathbf{v}_j, \vec{x} _ {i j} \cdot \mathbf{v}_i, \mathbf{v}_i \cdot \mathbf{v}_j\right).
$$

- Vector message:

$$
\begin{aligned}
\overrightarrow{\mathbf{m}}_i:=f_3\left(\mathbf{s}_i,\left\|\mathbf{v} _ {\mathbf{i}}\right\|\right) \odot \mathbf{v}_i & +\sum _ {j \in \mathcal{N}_i} f _ 4\left(\mathbf{s}_i, \mathbf{s}_j,\left\|\vec{x} _ {i j}\right\|,\left\|\boldsymbol{v}_j\right\|, \vec{x} _ {i j} \cdot \mathbf{v}_j, \vec{x} _ {i j} \cdot \mathbf{v}_i, \mathbf{v}_i \cdot \mathbf{v}_j\right) \odot \mathbf{v}_j \\
& +\sum _ {j \in \mathcal{N}_i} f_5\left(\mathbf{s}_i, \mathbf{s}_j,\left\|\vec{x} _ {i j}\right\|,\left\|\boldsymbol{v}_j\right\|, \vec{x} _ {i j} \cdot \mathbf{v}_j, \vec{x} _ {i j} \cdot \mathbf{v}_i, \mathbf{v}_i \cdot \mathbf{v}_j\right) \odot \vec{x} _ {i j}.
\end{aligned}
$$
	- where $\vec{x} _ {i j} = \vec{x} _ {j} - \vec{x} _ {i}$ denotes the relative position vector and $\odot$ denotes a scalar-vector multiplication. 

Clearly, we can achieve equivariance while using geometric features $\mathbf{v}_i$-s and $\vec{x} _ {i j}$-s, but we have to constraint the model operations. The high-level idea is to keep track of the "types" of the objects and apply equivariant operations; we treat scalar and vector features separately and ensure that they are maintained the same type through message passing.

As of now, we are constrained to have only scalar or vector features. What about higher order tensors?
 

### 3.2. Catersian Tensors and Tensor Products

A tensor is a multi-dimensional array with directional information. A rank-$n$ *Cartesian tensor* $T$ can be viewed as a multidimensional array with $n$ indices, i.e., $T _ {\mathrm{i} _ 1 \mathrm{i} _ 2 \cdots \mathrm{i} _ n}$ with $i_k \in$ $\{1,2,3\}$ for $\forall k \in\{1, \cdots, n\}$. Furthermore, each index of $T _ {i_1 i_2 \cdots i_n}$ transformsindependently as a vector under rotation.

<figure style="text-align: center;">
  <img alt="Image" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/2_geometric_GNNs/catersian_tensors.png" style="width: 50%; display: block; margin: 0 auto;" />
</figure>

- For a rotation represented by an orthogonal matrix $R$ , the components of $T$ transform as follows:

$$
T_{i_1^{\prime} i_2^{\prime} \cdots i_n^{\prime}}=\sum _ {i_1, i_2, \ldots, i_n} R _ {i_1^{\prime} i_1} R _ {i_2^{\prime} i_2} \cdots R _ {i_n^{\prime} i_n} T _ {i_1 i_2 \cdots i_n}
$$ 


Equivalently, in index notation with Einstein summation convention, this can be written compactly as (refered to this [StackOverflow Post](https://stackoverflow.com/questions/26089893/understanding-numpys-einsum) for einsum operations):

$$
T_{i_1^{\prime} i_2^{\prime} \cdots i_n^{\prime}}=R _ {i_1^{\prime} i_1} R _ {i_2^{\prime} i_2} \cdots R _ {i_n^{\prime} i_n} T _ {i_1 i_2 \cdots i_n}
$$

A vector (rank-$1$ tensor) $v$ in 3D Euclidean space $\mathbb{R}^3$ can be expressed in the familiar Cartesian coordinate system in the standard basis
```math
\mathbf{e} _ x=\left(\begin{array}{l}1 \\ 0 \\ 0\end{array}\right) \mathbf{e} _ y=\left(\begin{array}{l}0 \\ 1 \\ 0\end{array}\right) \mathbf{e} _ z=\left(\begin{array}{l}0 \\ 0 \\ 1\end{array}\right).
```

### 3.3. Intuition on Translation Equivariance


  
### 3.4. Generalization
 
Let's look at the definition of cross-correlation:

<figure style="text-align: center;">
  <img alt="Convolution" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/conv_math.png" style="width: 75%; display: block; margin: 0 auto;" />
</figure>

Here, we explicitly think of the cross-correlation in terms of translations. To generalize, if we want to transform $f$ with other groups, the trick is to make the kernel $k$ be represented by a group. Group representations on $k$ are reflected on $f$ as well.  

To generalize to other groups, we should consider the following:

- Make the function defined on the group of interest.  
- Integrate over the group of interest.  
- Make the kernel reflect the actions of the group of interest.  

## 4. Regular Group CNN and $SE(2)$ Equivariance

### 4.1. Definition: $SE(2)$ Lifting Correlation

To make the function defined on the group of interest, we define the **lifting operation**. The lifting correlation of $f$ and $g$ is written as $f \star_{SE(2)} g$, denoting the operator with the symbol $\star_{SE(2)}$. It is defined as the integral of the product of the two functions after one is shifted and rotated. As such, it is a particular kind of integral transform:  
<!-- 
$$
(k \star_{SE(2)} f)(x, \theta):=\int_{\mathbb{R}^2} k\Big(\mathbf{R}^{-1}_{\theta}(x'-x)\Big)f(x') d x'  = \int_{\mathbb{R}^2} [\mathscr{L}_{g=(x, \theta)}k(x')]f(x') d x' = \left\langle \mathscr{L}_{g=(x, \theta)}k, f \right\rangle_{\mathbb{L}_2\left(\mathbb{R}^2\right)} .
$$
-->
<figure style="text-align: center;">
  <img alt="Lifting" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/gconv_math.png" style="width: 75%; display: block; margin: 0 auto;" />
</figure>

Lifting correlation raises the feature map to a higher dimension that represents rotation. Now, planar rotation becomes a planar rotation in the $xy$-axes and a periodic shift (translation) in the $\theta$-axis.  

<figure style="text-align: center;">
  <img alt="Lifting" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/lifting.png" style="width: 65%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;">Figure 6: Lifting operation convolves the input with rotated copies of the kernel to reflect the SE(2) group. An additional dimension is included to reflect the rotation angles.</figcaption>
  

### 4.2. Demonstration: Lifting Correlation with the $p_4$ Rotation Group

The $p_4$ group can be described as a semi-direct product:

$$
p_4=C_4 \ltimes \mathbb{Z}^2,
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

The resulting feature maps in the group space are equivariant (rotation in the input $\mapsto$ planar rotation + periodic shift in the output features).

<figure style="text-align: center;">
  <img alt="Lifting" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/lifted_features.png" style="width: 80%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;">Figure 6: Lifting correlation includes an additional dimension to reflect the rotation angles. Now, a rotation in the input will results in a planar rotation in the spatial dimensions and a periodic shift (translation) in the angular dimension (this specifies the equivariance of the lifting correlation). </figcaption>

### 4.3. Definition: $SE(2)$ Group Cross Correlations

Now, the function is already defined on the group of interest after lifting. We still need to convolve over the group of interest and make the kernel reflect the actions of the group of interest.  

The group correlation of $f$ and $g$ is written as $f \star_{SE(2)} g$, denoting the operator with the symbol $\star_{SE(2)}$. It is defined as the integral of the product of the two functions after one is shifted and rotated:  

<figure style="text-align: center;">
  <img alt="Cross Correlation" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/gconv_math2.png" style="width: 75%; display: block; margin: 0 auto;" />
</figure>

Although the examples are given for the group $\mathrm{SE}(2)$, the idea can generalize to other affine groups (semi-direct product groups).  

If we look carefully at how rotational equivariance is achieved, we find that it basically adds a rotation dimension represented by an axis $\theta$. Thus, the rotational equivariance problem now becomes a translation equivariance problem, which can be solved easily by convolution/cross-correlation.  

$$\text { translational weight sharing } \Longleftrightarrow \quad \text { translation group equivariance }$$

$$\text { affine weight sharing } \Longleftrightarrow \quad \text { affine group equivariance }$$

Note: Translations and $H$-transformations form so-called affine groups: $\operatorname{Aff}(H) := \left(\mathbb{R}^d, +\right) \rtimes H.$  

### 4.4. Demonstration: Cross Correlation with the $p_4$ Rotation Group
Now, we have to reflect the differences in formulation between the lifting correlation and cross-correlation in the code as well.  

```python
def p4_group_convolution(features, kernel):
    """
    Perform P4 group convolution on a set of feature maps on P4 group.

    Parameters:
    - features (numpy.ndarray): A 3D array of feature maps with shape (|G|, s, s).
    - kernel (numpy.ndarray): A 2D array representing the convolution kernel.

    Returns:
    - numpy.ndarray: feature maps after the P4 group convolution with shape (|G|, s, s).
    """
    output = np.zeros_like(features)
    # Perform convolution for each feature map, convolve over both planar and angular axes
    for i in range(features.shape[0]):
        feature_map = features[i]
        result = np.zeros_like(feature_map)
        # SE(2) group on the kernels
        for j in range(4):
            rotated_kernel = np.rot90(kernel, j)  
            result += convolve2d(feature_map, rotated_kernel, mode='same', boundary='symm')
        output[i] = result
    return output
```

Similar to above, you can check that the resulting feature maps in the group space are equivariant (rotation in the input $\mapsto$ planar rotation + periodic shift in the output features).  

In actual implementation, the group dimension can be added to the channel dimension:  

<figure style="text-align: center;">
  <img alt="Invariance and Equivariance" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/group_conv_channel_implementation.png" style="width: 85%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;">Figure 7: Actual Implementation of Group CNNs: The group dimension is added to the channel dimension. </figcaption>
  
### 4.5. Overall Group CNN Pipeline
Overall, Group CNNs have the following structures:

1. **Lifting Layer (Generate group equivariant feature maps):**  
   - 2D input $\Rightarrow$ 3D feature maps with the third dimension representing rotation.  

2. **Group Conv Layers (Convolve over the group space):**  
   - 3D feature maps $\Rightarrow$ 3D feature maps.  

3. **Projection Layer (Collapse the group dimension):**  
   - **Invariance:** 3D feature map $\Rightarrow$ 2D feature map by (e.g., max/avg) pooling over the $\theta$ dimension. Now, it is invariant in the $\theta$ dimension.  
   - **Equivariance:** The resulting 2D feature map is rotation equivariant with respect to the input.  

<figure style="text-align: center;">
  <img alt="Invariance and Equivariance" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/GCNN.png" style="width: 85%; display: block; margin: 0 auto;" />
</figure>
  <figcaption style="text-align: center;">Figure 8: Overall Structure of Group CNNs: Group Lifting Layer $\Rightarrow$ Group Convolution Layers $\Rightarrow$ Group Projection Layer. Figure Source: [5]. </figcaption>

## 5. High-level Ideas on $SE(2)$ Steerable CNNs

### 5.1 From Group CNNs to Steerable CNNs
Group CNNs typically work with discrete groups of transformations, such as the $p_4$ group we have considered. However, many groups, including the rotation group, are continuous. You may perform very fine-grained discretization to capture the continuous nature of such groups, but the computational hurdle is intractable, and even so, discretizations still lose some of the continuity inherent in the group structure.  

In a single sentence, steerable CNNs **interpolate** discrete (in terms of the rotation dimension) feature maps from group CNNs using Fourier/trigonometric interpolations.  

- After the lifting layer, we have an extra dimension $\theta$ for the rotation angles. If we look at a specific pixel location, we can view all the feature values at this location as a periodic function $f: \theta \in [0, 2\pi) \mapsto \mathbb{R}$.  

<figure style="text-align: center;">
  <img alt="Cross Correlation" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/fiber.png" style="width: 45%; display: block; margin: 0 auto;" />
</figure>

- How do we get continuous functions from discrete values? The answer is interpolation. As this function is periodic and defined on $[0, 2\pi)$, it is very natural to represent this function as a Fourier series. We can get the Fourier coefficients from discrete points, e.g., $0^\circ$, $90^\circ$, $180^\circ$, and $270^\circ$, by performing a discrete Fourier transform.  

- Now, a periodic shift (translation) is a phase shift on these coefficients (Fourier shift theorem), and convolution is a point-wise multiplication with the coefficients.  

- A little caveat: this is an approximation to equivariance if the degrees of rotation are not one of those discrete points.  

For details, the readers are refered to [2]. 


## References

[1] A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems, Duvel et al

[2] SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions, Kristof T. Schütt et al.

[3] Directional Message Passing for Molecular Graphs, Johannes Gasteiger et al.

[4] Spherical Message Passing for 3D Graph Networks, Yi Liu et al.


## Other Useful Resources for Starters

### Lecture Recordings
1. [First Italian School on Geometric Deep Learning](https://www.youtube.com/playlist?list=PLn2-dEmQeTfRQXLKf9Fmlk3HmReGg3YZZ) (Very nice mathematical prerequisites)
2. [Group Equivariant Deep Learning (UvA - 2022)](https://www.youtube.com/playlist?list=PL8FnQMH2k7jzPrxqdYufoiYVHim8PyZWd)

