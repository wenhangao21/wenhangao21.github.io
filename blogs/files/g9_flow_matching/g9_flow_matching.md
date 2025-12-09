---
layout: blog
title: "Generative Models"
author_profile: false
---

# Flow Matching

**TL;DR:** This write-up contains the minimum essential concepts and simple code to understand flow matching.

The notebook version with the full code can be accessed [here](). If you have any questions or notice any errors, please contact [me](https://wenhangao21.github.io/).

## Problem Formulation

**Given samples from two distributions $\pi_0$ and $\pi_1$, we aim to find a transport map $\mathcal{T}$ such that, when $X_0 \sim \pi_0$, then $X_1 = \mathcal{T}(Z_0) \sim \pi_1$.**
- $\pi_0$ is the source distribution, and $\pi_1$ is the target distribution.
- For image generation, $\pi_1$ can be the image data distribution and $\pi_0$ can be any prior distribution easy to sample from, e.g. Gaussian.

**Code:** We will use the standard Gaussian distribution as the source distribution and the “checkerboard distribution” as the target distribution. Let’s set up these two distributions.

```python
def sample_checkerboard(N=1000, grid_size=4, scale=2.0):
    """
    Generate samples from a 2D checkerboard distribution.
    This function divides the square domain [-scale, scale] × [-scale, scale]
    into a grid of `grid_size × grid_size` equally sized square cells.
    """
    grid_length = 2 * scale / grid_size
    # Randomly choose integer tile coordinates
    gx = np.random.randint(0, grid_size, size=N)
    gy = np.random.randint(0, grid_size, size=N)
    mask = ((gx % 2) ^ (gy % 2)).astype(bool) # Keep only tiles where (even, odd) or (odd, even) — XOR rule
    while not np.all(mask): # Resample indices until all entries satisfy the checkerboard mask
        bad = np.where(~mask)[0]
        gx[bad] = np.random.randint(0, grid_size, size=len(bad))
        gy[bad] = np.random.randint(0, grid_size, size=len(bad))
        mask = ((gx % 2) ^ (gy % 2)).astype(bool)
    # Sample uniformly inside each chosen tile
    offsets = np.random.rand(N, 2) * grid_length
    xs = -scale + gx * grid_length + offsets[:, 0]
    ys = -scale + gy * grid_length + offsets[:, 1]
    return np.stack([xs, ys], axis=1).astype(np.float32)

def sample_pi_0(N=1000):
  return np.random.randn(N, 2).astype(np.float32)

def sample_pi_1(N=1000, grid_size=4, scale=2.0):
  return sample_checkerboard(N=N, grid_size=grid_size, scale=scale)

# Generate data
pi_0 = sample_pi_0(N=5_000)
grid_size, scale = 4, 2
pi_1 = sample_pi_1(N=5_000, grid_size=grid_size, scale=scale)
# Plot
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].scatter(pi_0[:, 0], pi_0[:, 1], c='purple', s=5)
axes[0].set_aspect('equal', 'box')
axes[0].set_title("Source Distribution: $X_0$")
axes[1].scatter(pi_1[:, 0], pi_1[:, 1], c='red', s=5)
axes[1].set_xlim(-scale, scale)
axes[1].set_ylim(-scale, scale)
axes[1].set_aspect('equal', 'box')
axes[1].set_title("Target Distribution: $X_1$")

plt.tight_layout()
plt.show()
```
<figure id="figure-1" style="margin: 0 auto 1em auto; text-align: center;">
  <div style="display: flex; justify-content: center;">
    <img src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/g9_flow_matching/distributions.png" width="500"> 
  </div>
  <figcaption>
    <a href="#figure-1">Figure 1</a>.
    Source distribution and target distribution.
  </figcaption>
</figure>

## Transport Map: ODEs
In flow models, the mapping $\mathcal{T}$ is implicitly defined by an ordinary differential equation (ODE):

$$
\frac{d}{d t} X_t=v\left(X_t, t\right), \quad X_0 \sim \pi_0, \quad \forall t \in[0,1],
$$

where $v\left(X_t, t\right)$ is called the velocity field. Given the source data $X_0 \sim \pi_0$, we can generate the target data $X_1 \sim \pi_1$ by following the ODE.

## Interpolants
There are infinitely many ways to define the vector field $v$:

<figure id="figure-2" style="margin: 0 auto 1em auto; text-align: center;">
  <div style="display: flex; justify-content: center;">
    <img src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/g9_flow_matching/flow_matching_different_velocities.png" width="500"> 
  </div>
  <figcaption>
    <a href="#figure-2">Figure 2</a>.
    Two different velocity fields that lead to the same endpoint distributions. Image from <a href="https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html">Tor Fjelde et al., 2024</a>.
  </figcaption>
</figure>

To define a specific $v$, we leverage an interpolation process with a time schedule $\alpha_t$ and $\beta_t$:

$$
X_t=\alpha_t X_0+\beta_t X_1,
$$

where $t \in[0,1], \alpha_0=\beta_1=1, \alpha_1=\beta_0=0$, and $\{X_t\}_t$ denotes the resulting trajectory. 
A common choice of interpolant is the linear interpolant, given by $\alpha_t=1-t$ and $\beta_t=t$.

## Conditional Velocity
Differentiate both sides with respect to $t$ gives us an "ODE":

$$
\frac{d X_t}{d t}=\alpha_t^{\prime} X_0+\beta_t^{\prime} X_1 \triangleq  v_s\left(X_t, t \mid X_0, X_1 \right), 
$$

where $v_s$ is the conditional velocity of a sample trajectory conditioned on the endpoints $X_0$ and $X_1$. 
**This ODE is not generative because the endpoints must be given.**

## Learning the Velocity
To resolve the non-generative issue, we learn a velocity field $v_\theta\left(X_t, t\right)$ such that the generative ODE $\frac{d}{d t} X_t=v_\theta\left(X_t, t\right)$ can approximate the previous non-generative process as closely as possible.

The simplest way to do this is to minimize the squared error between the two systems' velocity fields:

$$
\min _\theta \int_0^1 \mathbb{E}_{X_0 \sim \pi_0, X_1 \sim \pi_1}\left[\left\|\left(\alpha_t^{\prime} X_0+\beta_t^{\prime} X_1\right)-v_\theta\left(X_t, t\right)\right\|^2\right] d t,
$$

and it is equivalent to

$$
\min _\theta \mathbb{E}_{X_0 \sim \pi_0, X_1 \sim \pi_1, t\sim \text{Uniform}(0,1)}\left[\left\|\left(\alpha_t^{\prime} X_0+\beta_t^{\prime} X_1\right)-v_\theta\left(X_t, t\right)\right\|^2\right].
$$

**Code**: We will use the linear interpolant $\alpha_t=1-t$ and $\beta_t=t$ with $\alpha_t^{\prime} X_0+\beta_t^{\prime} X_1 = X1 - X0$.

1. We setup the velocity network $v_\theta$.
```python
class MLP_FM(nn.Module):
    def __init__(self, in_dim=2, context_dim=1, h=128, out_dim=2): # context is time
        super(MLP_FM, self).__init__()
        self.context_dim = context_dim
        self.network = nn.Sequential(nn.Linear(in_dim + context_dim, h), nn.GELU(),
                                     nn.Linear(h, h), nn.GELU(),
                                     nn.Linear(h, h), nn.GELU(),
                                     nn.Linear(h, h), nn.GELU(),
                                     nn.Linear(h, out_dim))

    def forward(self, x, t):
        return self.network(torch.cat((x, t), dim=1))
```

2. We train the velocity network by randomly sampling $X_0, X_1$, and $t$, and minimizing the squared error above.
```python
def train_flow_matching(flow_model, n_iterations=5_001, lr=3e-3, batch_size=4096, save_freq=2_000):
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=lr)
    losses = []
    progress_bar = tqdm(range(n_iterations), desc="Training Flow Matching", ncols=100)
    for iteration in progress_bar:
        x1 = torch.from_numpy(sample_pi_1(N=batch_size)).to(device)
        x0 = torch.from_numpy(sample_pi_0(N=batch_size)).to(device)
        t = torch.rand((x1.shape[0], 1), device=device) # randomly sample t
        x_t = t * x1 + (1.-t) * x0  # swap x0, x1 in the equations above
        v = x1 - x0  # swap x0, x1 in the equations above
        v_pred = flow_model(x_t, t)
        loss = torch.nn.functional.mse_loss(v_pred, v)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})
    return flow_model, losses
```

```python
flow_model = MLP_FM(in_dim=2, context_dim=1, h=128, out_dim=2).to(device)
trained_model_FM, training_losses_FM = train_flow_matching(flow_model=flow_model)
```

## Sampling with Learned Velocity
Once the velocity field $v_\theta\left(X_t, t\right)$ is learned, we can simulate samples from $\pi_1$ given samples from $\pi_0$ by solving the ODE:$
\frac{d}{d t} X_t=v_\theta\left(X_t, t\right),~X_0 \sim \pi_0,~t \in[0,1]$. One way to solve it is using the [forward Euler method](https://en.wikipedia.org/wiki/Euler_method):

$$X_{t+\Delta t}=X_t+\Delta t \cdot v_\theta\left(X_t, t\right),$$

where $\Delta t = \frac{1}{T}$ with T being the number of steps used in sampling. 
```python
def sample_mean_flow(flow_model, N, T):
    """
    Generate N samples from pi_1 with the mean flow model in T timesteps.
    """
    x = torch.from_numpy(sample_pi_0(N=N)).to(device)
    dt = 1./T
    for i in (range(T)):
        t = torch.ones((x.shape[0], 1), device=x.device) * (i * dt)
        v_pred = flow_model(x.squeeze(0), t)
        x = x + v_pred * 1. / T # reverse integration as we learn the velocity from data to noise
    return x
```

Generated Samples:

<figure id="figure-3" style="margin: 0 auto 1em auto; text-align: center;">
  <div style="display: flex; justify-content: center;">
    <img src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/g9_flow_matching/samples.png" width="300"> 
  </div>
  <figcaption>
    <a href="#figure-3">Figure 3</a>.
    Samples from the learned target distribution by flow matching.
  </figcaption>
</figure>

## Additional Visualization

<figure id="figure-4" style="margin: 0 auto 1em auto; text-align: center;">
  <div style="display: flex; justify-content: center;">
    <img src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/g9_flow_matching/pi_0_to_pi_1.png" width="500"> 
  </div>
  <figcaption>
    <a href="#figure-4">Figure 4</a>.
    Visualization of the learned intermediate distributions $\pi_t$.
  </figcaption>
</figure>

<figure id="figure-5" style="margin: 0 auto 1em auto; text-align: center;">
  <div style="display: flex; justify-content: center;">
    <img src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/g9_flow_matching/samples_different_ite.png" width="600"> 
  </div>
  <figcaption>
    <a href="#figure-5">Figure 5</a>.
    Visualization of the learned target distribution at different training iterations.
  </figcaption>
</figure>

## Marginal Velocity
Although the regression target for each sample is the conditional velocity $\alpha_t^{\prime} X_0+\beta_t^{\prime} X_1$. The **underlying unique regression target** is the marginal velocity $v\left(X_t, t\right)=\mathbb{E}_{X_0, X_1 \mid X_t, t}\left[\alpha_t^{\prime} X_0+\beta_t^{\prime} X_1\right]$. To see this:

$$\begin{aligned} \mathcal{L} & =\mathbb{E}_{X_0, X_1, t}\left[\left\|v_s\left(X_t, t \mid X_0, X_1 \right) - v_\theta(X_t, t)\right\|^2\right] \\ & =\mathbb{E}_{X_t, t}\left[\mathbb{E}_{X_0, X_1 \mid X_t, t}\left[\left\|\alpha_t^{\prime} X_0+\beta_t^{\prime} X_1 - v_\theta(X_t,t) \right\|^2\right]\right] .\end{aligned}$$

Since the expectation operator is linear, the optimal velocity field is:

$$
v^*\left(X_t, t\right)=\mathbb{E}_{X_0, X_1 \mid X_t, t}\left[\alpha_t^{\prime} X_0+\beta_t^{\prime} X_1\right].
$$

**Therefore, even if we use a linear interpolant with straight-line sample trajectories, the sampling cannot be done in one step, since the actual regression target is not the marginal velocity.**