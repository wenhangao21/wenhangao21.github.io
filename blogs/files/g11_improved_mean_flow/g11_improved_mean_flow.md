---
layout: blog
title: "Improved Mean Flow"
author_profile: false
---

# Improved Mean Flow (iMF)

**TL;DR**: This write-up contains the minimum essential concepts and simple code to understand [improved mean flow](https://arxiv.org/abs/2512.02012), focusing only on unconditional generation, up to Sec. 4.1.

Readers are assumed to be familiar with flow matching and mean flow.

The code follows the same style as the [flow matching](https://wenhangao21.github.io/blogs/files/g9_flow_matching/g9_flow_matching/) 
and [mean flow](https://wenhangao21.github.io/blogs/files/g10_mean_flow/g10_mean_flow/) tutorials, using simple if–else statements to switch between 
flow matching (FM), mean flow (MF), and improved mean flow (iMF) for easy transition and comparison.

The notebook version with the full code can be accessed [here](https://github.com/wenhangao21/Tutorials/tree/main/Generative_Models/improved_mean_flow). 
If you have any questions or notice any errors, please contact [me](https://wenhangao21.github.io/).

## 1. Problem Formulation
**Given samples from two distributions $\pi_0$ and $\pi_1$, we aim to find a transport map $\mathcal{T}$ such that, when $X_0 \sim \pi_0$, then $X_1 = \mathcal{T}(Z_0) \sim \pi_1$.**
- $\pi_0$ is the source distribution, and $\pi_1$ is the target distribution.
- For image generation, $\pi_1$ can be the image data distribution and $\pi_0$ can be any prior distribution easy to sample from, e.g. Gaussian.

**Code:** We will use the standard Gaussian distribution as the source distribution and the “checkerboard distribution” as the target distribution. Let’s set up these two distributions.

```python
def sample_pi_0(N=1000):
  return np.random.randn(N, 2).astype(np.float32)

def sample_pi_1(N=1000, grid_size=4, scale=2.0):
  return sample_checkerboard(N=N, grid_size=grid_size, scale=scale)

# Generate data
pi_0 = sample_pi_0(N=5_000)
grid_size, scale = 4, 2
pi_1 = sample_pi_1(N=5_000, grid_size=grid_size, scale=scale)
```
<figure id="figure-1" style="display:block; text-align:center;">
  <img
    src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/g9_flow_matching/distributions.png"
    style="display:block; margin:auto; max-width:500px; margin:auto;"
  >
  <figcaption style="display:block; margin-top:0.5em;">
    <a href="#figure-1">Figure 1</a>. Source distribution and target distribution.
  </figcaption>
</figure>

## 2. Background Information
Flow matching is introduced in this [tutorial](https://wenhangao21.github.io/blogs/files/g9_flow_matching/g9_flow_matching/); mean flow is introduced in this [tutorial](https://wenhangao21.github.io/blogs/files/g10_mean_flow/g10_mean_flow/).

**Notation**:
- $\pi_0$: the source distribution
- $\pi_1$: the target distribution
- $X_0 \sim \pi_0$: random variables sampled from the source distribution
- $X_1 \sim \pi_1$: random variables sampled from the target distribution
- $\alpha_t=1-t$ and $\beta_t=t$: linear time schedule
- $X_t = t X_1+(1-t) X_0$: the flow interpolation
- $v_s\left(X_t, t\right) = X_1 - X_0$: the conditional velocity
- $\frac{d}{d t} X_t=v\left(X_t, t\right),t \in[0,1]$: the flow ODE
- $v\left(X_t, t\right)$: the marginal velocity
- $u(X_t, r, t) = \frac{1}{t - r} \int_{r}^{t} v(X_s, s) ds$: the average velocity over an interval $[r, t]$ with $t > r$
- $u(X_t, r, t)= v(X_t, t)-(t - r)\frac{d}{dt} u(X_t, r, t)$: mean flow identity


The flow matching objective is:

$$
\min _\theta \mathbb{E}_{X_0 \sim \pi_0, X_1 \sim \pi_1, t\sim \text{Uniform}(0,1)}\left[\left\|v_s-v_\theta\left(X_t, t\right)\right\|^2\right].
$$

The mean flow objective is:

$$
\min _\theta \mathbb{E}_{X_0 \sim \pi_0, X_1 \sim \pi_1, r, t}\left[\left\| u_\theta\left(X_t, r, t\right) - \left(v_s - (t-r) \left[\frac{\partial u_\theta}{\partial X_t}, \frac{\partial u_\theta}{\partial r}, \frac{\partial u_\theta}{\partial t}\right] \cdot [v_s, 0,1] \right)\right\|^2\right].
$$

## 3. Mean Flow → Improved Mean Flow in One Line of Code

We first implement the improved mean flow by **adding only one line of code** from the original mean flow.

**Summary**: In mean flow, the average velocity $u$ is converted into an expression involving the instantaneous velocity $v$ and $\frac{du}{dt}$ for training with the network approximation $u_\theta$. However, $\frac{du_\theta}{dt}$ must be rewritten in terms of partial derivatives for Autograd, which introduces the term $\frac{dX_t}{dt} = v$, and this is replaced by the conditional velocity $v_s$ during training. This substitution is incorrect and leads to training issues. To mitigate this, the $v$ in $\frac{du_\theta}{dt}$ is replaced by the network’s approximation of the instantaneous velocity $u(X_t, t, t)$ instead.

```python
class MLP(nn.Module):
    def __init__(self, in_dim=2, context_dim=2, h=128, out_dim=2):
        super(MLP, self).__init__()
        self.context_dim = context_dim
        self.network = nn.Sequential(nn.Linear(in_dim + context_dim, h), nn.GELU(),
                                     nn.Linear(h, h), nn.GELU(),
                                     nn.Linear(h, h), nn.GELU(),
                                     nn.Linear(h, h), nn.GELU(),
                                     nn.Linear(h, out_dim))

    def forward(self, x, t, r=None):
      if r == None:
        return self.network(torch.cat((x, t), dim=1))
      else: # mean flow takes an additional time r
        return self.network(torch.cat((x, t, r), dim=1))


def train_flow(flow_model, n_iterations=5_001, lr=3e-3, batch_size=4096, save_freq=1_000, flow="FM", improved=True):
    print(f"Training {flow}")
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=lr)
    losses = []
    progress_bar = tqdm(range(n_iterations), desc="Training Flow Model", ncols=100)
    for iteration in progress_bar:
        x1 = torch.from_numpy(sample_pi_1(N=batch_size)).to(device)
        x0 = torch.from_numpy(sample_pi_0(N=batch_size)).to(device)
        if flow == "FM":
            t = torch.rand((x1.shape[0], 1), device=device) # randomly sample t
            x_t = t * x1 + (1.-t) * x0
            v = x1 - x0
            v_pred = flow_model(x_t, t)
            loss = torch.nn.functional.mse_loss(v_pred, v)
        elif flow == "MF":
            tr = torch.rand((x1.shape[0], 2), device=device)
            t = tr.max(dim=1, keepdim=True).values
            r = tr.min(dim=1, keepdim=True).values
            x_t = t * x1 + (1.-t) * x0
            v = x1 - x0   # conditional velocity given the end points
            dtdt, drdt = torch.ones_like(t), torch.zeros_like(r)
            # only changing the input to JVP for iMF; iMF: network velocity; MF: conditional velocity
            v_input = flow_model(x_t, t, t) if improved else v
            u_pred, dudt = torch.func.jvp(flow_model, (x_t, t, r), (v_input, dtdt, drdt))
            u_target = (v - (t - r) * dudt).detach()
            loss = torch.nn.functional.mse_loss(u_pred, u_target)
        else:
            raise NotImplementedError
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return flow_model, losses
```		

## 4. Mean Flow: Training Objective

Start with the mean flow identity:

$$
\begin{aligned}
\underbrace{u(X_t, r, t)}_{\text{avg. vel.}}
    &= \underbrace{v(X_t, t)}_{\text{instant. vel.}}
       -
       (t - r) \underbrace{\frac{d}{dt} u(X_t, r, t)}_{\text{time derivative}}.
\end{aligned}
$$

**Note that autograd can only directly compute explicit (partial) derivatives, while the time derivative is implicit ($u$ is a functional of a function of $t$).** We apply multivariate chain rule:

$$\frac{d}{d t} u\left(X_t, r, t\right)=\frac{d X_t}{d t} \frac{\partial u}{\partial X_t}+\frac{d r}{d t}  \frac{\partial u}{\partial r}+\frac{d t}{d t}  \frac{\partial u}{\partial t},$$

where $\frac{d X_t}{d t} = v, \frac{d t}{d t} = 1$ and $\frac{d r}{d t} = 0$. This equation shows that the total derivative is given by the Jacobian-vector product (JVP) between $\left[\frac{\partial u}{\partial X_t}, \frac{\partial u}{\partial r}, \frac{\partial u}{\partial t}\right]$ (the Jacobian matrix of the function $u$ ) and the tangent vector $[v, 0,1]$. It can be efficiently computed by `torch.func.jvp` in PyTorch.

The training objective is:

$$
\min _\theta  \mathbb{E}_{X_0 \sim \pi_0, X_1 \sim \pi_1, r, t}\left[\left\| u_\theta\left(X_t, r, t\right) - \left(v(X_t, t) - (t-r) \left[\frac{\partial u}{\partial X_t}, \frac{\partial u}{\partial r}, \frac{\partial u}{\partial t}\right] \cdot [v(X_t, t), 0,1] \right)\right\|^2\right],
$$

where $r=\min \left(s_1, s_2\right), t=\max \left(s_1, s_2\right)$ with $s_1,s_2 \sim \text{Uniform (0,1)}$, $X_t=t X_1+(1-t) X_0$.

**In mean flow, both $v(X_t, t)$ are replaced with the conditional velocity given the end points $v_s=X_1-X_0$:**


$$
\min _\theta \mathbb{E}_{X_0 \sim \pi_0, X_1 \sim \pi_1, r, t}\left[\left\| u_\theta\left(X_t, r, t\right) - \left(v_s - (t-r) \left[\frac{\partial u_\theta}{\partial X_t}, \frac{\partial u_\theta}{\partial r}, \frac{\partial u_\theta}{\partial t}\right] \cdot [v_s, 0,1] \right)\right\|^2\right].
$$

**While replacing the first $v$ with $v_s$ is consistent with flow matching, replacing the second $v$ is not justified and leads to unstable training.**


## 5. Mean Flow: Non-decreasing Loss with Large Variance
When training mean flow models, the loss often tends to be non-decreasing and displays significant fluctuations.


<figure id="figure-2" style="display:block; text-align:center;">
  <img
    src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/g11_improved_mean_flow/loss_curve_MF.png"
    style="display:block; max-width:350px; margin:auto;"
  >
  <figcaption style="display:block; margin-top:0.5em;">
    <a href="#figure-2">Figure 2</a>.
    Training loss curve of the mean-flow model.
  </figcaption>
</figure>

## 6. Improved Mean Flow: Training Objective
Note that the mean flow identity can be rewritten as:

$$
\begin{aligned}
\underbrace{v(X_t, t)}_{\text{instant. vel.}}
    &= \underbrace{u(X_t, r, t)}_{\text{avg. vel.}}
       +
       (t - r) \underbrace{\frac{d}{dt} u(X_t, r, t)}_{\text{time derivative}}.
\end{aligned}
$$

In this view, if we parameterize $u$ with a neural network $u_\theta$, we can interpret the objective similar to a regular flow-matching loss that approximates the instantaneous velocity with a neural network:
$$
\mathcal{L}(\theta)=\mathbb{E}_{t, r, X_t}\left[\left\| v\left(X_t, t\right) - v_\theta\left(X_t, r, t\right)\right\|^2\right],
$$
where $v_\theta\left(X_t, r, t\right) = u_\theta\left(X_t, r, t\right)+(t-r) \frac{d}{d t} u_\theta\left(X_t, r, t\right)$.

Similar to regular flow matching, we replace the marginal velocity $v\left(X_t, t\right)$ with the conditional velocity $v_s = X_1-X_0$. However, in mean flow, $\frac{d u}{d t}$ is converted into explicit derivatives via JVP, **with a term involving the true marginal velocity $v$**, and this velocity is likewise replaced by the conditional velocity $v_s$ (see above) and "use as input to $v_\theta$". This leads to an issue:
- The regression target (the conditional velocity) is leaked as it is "used as input to $v_\theta$".
  - As the regression target is given, the network can simply be an identity function to minimize loss. However, the real regression target is not the conditional velocity, but the marginal velocity $v\left(X_t, t\right)=\mathbb{E}_{X_0, X_1 \mid X_t}\left[X_1 - X_0\right]$.

**To resolve this issue, we can use a neural network to learn the marginal velocity and be used in JVP. Notice that $u\left(X_t, t, t\right)=v\left(X_t, t\right)$. A simple approach is to use $u_\theta\left(Z_t, t, t\right)$ in the JVP instead of $v_s$.**

The improved mean flow objective is:

$$
\min _\theta \mathbb{E}_{X_0 \sim \pi_0, X_1 \sim \pi_1, r, t}\left[\left\| u_\theta\left(X_t, r, t\right) - \left(v_s - (t-r) \left[\frac{\partial u_\theta}{\partial X_t}, \frac{\partial u_\theta}{\partial r}, \frac{\partial u_\theta}{\partial t}\right] \cdot [u_\theta(X_t, t, t), 0,1] \right)\right\|^2\right].
$$

The improved mean flow training is more stable:

<figure id="figure-3" style="display:block; text-align:center;">
  <img
    src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/g11_improved_mean_flow/loss_curve_FM_MF_iMF.png"
    style="display:block; max-width:800px; margin:auto;"
  >
  <figcaption style="display:block; margin-top:0.5em;">
    <a href="#figure-3">Figure 3</a>.
    Training loss curves of the flow matching (FM), mean flow (MF), and improved mean flow (iMF) models, respectively.
  </figcaption>
</figure>


## 7. Additional Visualization

<figure id="figure-4" style="display:block; text-align:center;">
  <img
    src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/g11_improved_mean_flow/multistep_generation_FM_MF_iMF.png"
    style="display:block; max-width:600px; margin:auto;"
  >
  <figcaption style="display:block; margin-top:0.5em;">
    <a href="#figure-4">Figure 4</a>.
    Samples with different sampling steps $T$ using flow matching (FM), mean flow (MF), and improved mean flow (iMF), respectively. 
  </figcaption>
</figure>


<figure id="figure-5" style="display:block; text-align:center;">
  <img
    src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/g11_improved_mean_flow/pi_0_to_pi_1_FM_MF_iMF.png"
    style="display:block; margin:auto; max-width:800px;"
  >
  <figcaption style="display:block; margin-top:0.5em;">
    <a href="#figure-5">Figure 5</a>.
    Visualization of the learned intermediate distributions $\pi_t$ using flow matching (FM), mean flow (MF), and improved mean flow (iMF), respectively. 
  </figcaption>
</figure>


<figure id="figure-6" style="display:block; text-align:center;">
  <img
    src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/g11_improved_mean_flow/samples_different_ite_FM_MF_iMF.png"
    style="display:block; margin:auto; max-width:800px;"
  >
  <figcaption style="display:block; margin-top:0.5em;">
    <a href="#figure-6">Figure 6</a>.
    Visualization of the learned target distribution at different training iterations using flow matching (FM), mean flow (MF), and improved mean flow (iMF), respectively. 
  </figcaption>
</figure>

