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

### 1.3. Universal Approximation

The universal approximation property of the Fourier Neural Operator (FNO) is established in [2]. The proof can be summarized in the following steps:

1. **Universal Approximation of Fourier Projection and Conjugate Operators**:
   - FNO can approximate the Fourier projection and Fourier conjugate operators effectively.
   - By treating Fourier coefficients as constant functions, the FNO is capable of mapping input functions to their truncated Fourier coefficients.
   - Similarly, FNO can map the Fourier coefficients (treated as constant functions) back to the functions they represent.
   - For any desired level of precision, suitable Fourier projection and conjugate operators can be selected to ensure the approximation error remains within the specified tolerance.

2. **Universal Approximation of Mapping Fourier Coefficients to Fourier Coefficients**:
   - Given that FNO can approximate both the Fourier projection and conjugate operators, the problem reduces to mapping finite Fourier coefficients to other finite Fourier coefficients.
   - This mapping task is guaranteed by the universal approximation theorem of neural networks, which ensures that neural networks can approximate any continuous mapping with arbitrary precision.

3. **Combining Both Steps**:
   - By combining the above two steps, it follows that for any desired precision, an appropriate Fourier Neural Operator can always be constructed, establishing its universal approximation property.


## 2. Implementation

> Note: The model is adopted from [1], and detailed explanations are added as comments here. The full code can be found [here](https://github.com/wenhangao21/fourier_neural_operator), which is a fork of the original authors' repository. If you are not familiar with einsum (Einstein Summation), this [StackOverflow post](https://stackoverflow.com/questions/26089893/understanding-numpys-einsum) provides a good explanation.

1\. Implement global continuous convolution in the Fourier space with Fast Fourier Transforms (FFTs).

```python
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. This layer performs FFT (Fast Fourier Transform),
        applies a linear transformation in the Fourier domain, and then 
        performs an Inverse FFT to return to the spatial domain.
        """
        # in_channels and out_channels are typically set to the same value.
        # modes1 and modes2 determine the number of Fourier modes to retain in the x and y dimensions, respectively.
        # They are at most floor(N/2) + 1 (see the documentation for np or torch RFFT).

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes in the x direction.
        self.modes2 = modes2  # Number of Fourier modes in the y direction.

        # Scale factor to ensure the weight initialization is appropriately normalized.
        self.scale = (1 / (in_channels * out_channels))
        # Initialize two sets of trainable weights with complex values, scaled by the normalization factor.
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        """
        Perform complex-valued matrix multiplication in the Fourier domain.
        Args:
            input: Fourier coefficients of shape (batch, in_channels, x, y).
            weights: Trainable weights of shape (in_channels, out_channels, x, y).
        Returns:
            Resulting Fourier coefficients of shape (batch, out_channels, x, y).
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        Forward pass for the SpectralConv2d layer.
        Args:
            x: Input tensor of shape (batch_size, resolution_x, resolution_y, in_channels).
        Returns:
            Output tensor of the same shape after applying the Fourier transform,
            linear transformation, and inverse Fourier transform.
        """
        batchsize = x.shape[0]

        # Compute the Fourier coefficients of the input using rFFT (real FFT).
        # Resulting shape: (batch_size, in_channels, resolution_x, resolution_y//2 + 1)
        x_ft = torch.fft.rfft2(x)

        # Initialize a tensor to store the transformed Fourier coefficients.
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)

        # Apply the trainable weights to the retained Fourier modes in the positive range.
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        # Apply the trainable weights to the retained Fourier modes in the negative range.
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Transform back to the spatial domain using the inverse rFFT (real IFFT).
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
```

2\. Implement the FNO model (the 2D time-dependent case, i.e. for Navier-Stokes).

```python
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        """
        2D Fourier Neural Operator model.
        Input: The solution of the previous T_in timesteps.
        Input shape: (batchsize, x=64, y=64, channels=T_in).
        Output: The solution for the next timestep.
        Output shape: (batchsize, x=64, y=64, channels=1).
        """
        self.modes1 = modes1  # Number of Fourier modes in the x-direction.
        self.modes2 = modes2  # Number of Fourier modes in the y-direction.
        self.width = width  # Number of channels in the Fourier layer.
        
        # Linear layer to map channels to `width` channels (lifting).
		# T_in = 10 timesteps + 2 spatial coordinates (x, y).
        self.p = nn.Linear(10 + 2, self.width)
        
        # Four Fourier layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2
		
		# Four 1 by 1 conv layers 
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        
        # Residual connections with 1x1 convolutions (shortcut connections).
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        # Instance normalization
        self.norm = nn.InstanceNorm2d(self.width)
        
        # Fully connected layers to map from `width` channels to the final output.
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        """
        Forward pass for the FNO2d model.
        Args:
            x: Input tensor of shape (batchsize, x=64, y=64, channels=10).
        Returns:
            Output tensor of shape (batchsize, x=64, y=64, channels=1).
        """
        # Add spatial coordinate information to the input (x, y).
        grid = self.get_grid(x.shape, x.device) # shape (batchsize, x=64, y=64, channels=12)
		# Lifitng
		x = self.p(x)  # shape (batchsize, x=64, y=64, channels=width)
		# Permute for convolutions 
        x = x.permute(0, 3, 1, 2)  # shape (batchsize, channels=width, x=64, y=64)
		
		# Convolution in Fourier space
        x1 = self.norm(self.conv0(self.norm(x))) # shape (batchsize, channels=width, x=64, y=64)
        # 1 by 1 convolution to promote interactions between different channels
		x1 = self.mlp0(x1) # shape (batchsize, channels=width, x=64, y=64)
		# Residual connection
        x2 = self.w0(x) # shape (batchsize, channels=width, x=64, y=64)
        x = x1 + x2   # shape (batchsize, channels=width, x=64, y=64)
        x = F.gelu(x)   # shape (batchsize, channels=width, x=64, y=64)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = self.q(x) # shape (batchsize, channels=1, x=64, y=64)
        x = x.permute(0, 2, 3, 1) # shape (batchsize, x=64, y=64, channels=1)
        return x

```	
		
> Full implementation can be found [here](https://github.com/wenhangao21/Tutorials/tree/main/Neural_PDE_Solvers).



On universal approximation and error bounds for Fourier Neural Operators

## References

[1] Fourier Neural Operator for Parametric Partial Differential Equations, Zongyi Li et al.

[2] On Universal Approximation and Error Bounds for Fourier Neural Operators, Nikola Kovachki et al.