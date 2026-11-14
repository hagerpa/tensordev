# Signatures

This page covers truncated path signatures, blocking and Chen's identity, free developments of
tensor-valued paths, and piecewise log-linear paths. Results are tensors in the format of the
selected core; [Tensor algebra](tensor-algebra.md) describes the cores and the storage layout.

## Path signatures

For a path $X\colon[0,T]\to\mathbb{R}^d$ of bounded variation, the signature $\mathrm{Sig}(X)_{0,T}$
collects the iterated integrals of $X$. Truncated at level $N$ it is

$$
\pi_{\le N}\mathrm{Sig}(X)_{0,T} = \bigl(1, \mathrm{Sig}(X)^{(1)}_{0,T}, \dots, \mathrm{Sig}(X)^{(N)}_{0,T}\bigr), \qquad
\mathrm{Sig}(X)^{(k)}_{0,T} = \int_{0<t_1<\cdots<t_k<T} dX_{t_1}\otimes\cdots\otimes dX_{t_k} \in (\mathbb{R}^d)^{\otimes k},
$$

and the coefficient of a word $w = i_1\cdots i_k$ is
$\langle w, \mathrm{Sig}(X)_{0,T}\rangle = \int_{0<t_1<\cdots<t_k<T} dX^{i_1}_{t_1}\cdots dX^{i_k}_{t_k}$. This is the
notation of the papers cited on this site. An array of $S+1$ points $x_0, \dots, x_S$ is read as
the piecewise linear path with $X_{t_i} = x_i$ on a grid $0 = t_0 < \cdots < t_S = T$, with increments
$\Delta_i = x_i - x_{i-1}$. The signature of one linear piece is the tensor exponential
$\exp_\otimes(\Delta_i)$, so by Chen's identity

$$
\pi_{\le N}\mathrm{Sig}(X)_{0,T} = \exp_\otimes(\Delta_1)\otimes_N\exp_\otimes(\Delta_2)\otimes_N\cdots\otimes_N\exp_\otimes(\Delta_S),
$$

where $\mathbf{a}\otimes_N\mathbf{b} = \pi_{\le N}(\mathbf{a}\otimes\mathbf{b})$ is the product truncated at level $N$.
[`path_signature`][tensordev.path_signature] evaluates this product. The examples on this page
share one session with 64-bit floats and a batch of four random paths in $\mathbb{R}^2$:

```python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import tensordev as td
from tensordev.util import random_trigonometric_polynomial_paths

X = random_trigonometric_polynomial_paths(batch=4, steps=32, dim=2, key=0)
print(X.shape)  # (4, 33, 2)
```

With the default core, the signature is a tuple of levels $0, \dots, N$, and level $k$ has shape
`batch + (d**k,)`. The word layout is described under
[Truncated tensors](tensor-algebra.md#truncated-tensors).

Level 0 is 1 and level 1 is the total increment $X_{0,T} = X_T - X_0$. The generated paths are closed
loops: each starts and ends at 0, so level 1 of `sig3` is zero. The first half of `X` is an
open path, and the checks of level 1 on this page use it:

```python
sig3 = td.path_signature(X, trunc=3)
print([level.shape for level in sig3])  # [(4, 1), (4, 2), (4, 4), (4, 8)]
np.testing.assert_allclose(sig3[0], 1.0)
np.testing.assert_allclose(sig3[1], 0.0, atol=1e-12)  # X is a closed loop

X_open = X[:, :17]
sig_open = td.path_signature(X_open, trunc=3)
np.testing.assert_allclose(sig_open[1], X_open[:, -1] - X_open[:, 0], atol=1e-12)
```

### Arguments

All arguments except `x` are keyword-only.

| Argument | Default | Description |
|---|---|---|
| `x` | | Path of shape `batch + (S+1, d)`, or increments of shape `batch + (S, d)` when `increment_input=True`. |
| `trunc` | `None` | Truncation. The default core needs an integer; a bidegree core accepts `(N, M)` and may supply a default. |
| `increment_input` | `False` | `x` already holds increments; skip differencing. |
| `starting_point` | `None` | Left seed $g$ in the core's native format; the result is $g\otimes_N\pi_{\le N}\mathrm{Sig}(x)$. Defaults to the identity. |
| `axis` | `None` | Step axis of `x`. Defaults to `seq_core.default_time_axis`, which is `-2` for the JAX sequential core. |
| `block_size` | `None` | Steps per emitted block; `None` gives one block over all steps. It must divide the number of steps. |
| `output_starting_point` | `False` | Prepend the seed to the output along `axis`. Without blocking, the output then has an axis of length 2: seed and signature. |
| `accumulate` | `True` | Carry the running product across blocks. With `False`, each block holds the signature of its own steps. |
| `accumulate_in_tree` | `False` | Accumulate across blocks with an associative tree scan. |
| `parallel` | `False` | Compute all per-step exponentials first and combine them with an associative tree scan: more parallelism, more memory. `False` streams the steps through `tensor_fmexp`, which multiplies the running signature by $\exp_\otimes(\Delta_i)$ with a fused Horner scheme: less memory, sequential depth. |
| `core` | `None` | Tensor-algebra core. Defaults to the installed default core. |
| `seq_core` | `None` | Sequential-operations backend. Defaults to the one that matches `core`. |
| `execution` | `"auto"` | `"auto"` keeps the default execution policy. `"jax"` never selects the wordwise GPU kernels; the [native CPU kernels](performance.md#native-cpu-kernels) still apply where eligible. `"wordwise"` requests the alpha GPU kernels. |

The `parallel=True` and `increment_input=True` routes give the same signature:

```python
np.testing.assert_allclose(
    td.tensor_to_flat(td.path_signature(X, trunc=3, parallel=True)),
    td.tensor_to_flat(sig3),
    atol=1e-12,
)
np.testing.assert_allclose(
    td.tensor_to_flat(td.path_signature(jnp.diff(X, axis=1), trunc=3, increment_input=True)),
    td.tensor_to_flat(sig3),
    atol=1e-12,
)
```

[`Signature`][tensordev.Signature] binds `trunc`, `core` and `seq_core` once. Calling it forwards
`axis`, `block_size`, `accumulate`, `starting_point`, `output_starting_point`, `parallel`,
`accumulate_in_tree`, `increment_input` and `execution` to `path_signature`. It resolves the
default core when it is constructed, so a later `set_default_core` does not change an existing
`Signature`.

```python
signature = td.Signature(trunc=3)
np.testing.assert_allclose(td.tensor_to_flat(signature(X)), td.tensor_to_flat(sig3))
```

On one NVIDIA GPU, `execution="wordwise"` requests alpha kernels that parallelize over words and
paths; [Wordwise GPU execution](performance.md#wordwise-gpu-execution) lists its restrictions.

## Blocking and Chen's identity

Chen's identity splits a signature at an intermediate time: for $s \le u \le t$,

$$
\mathrm{Sig}(X)_{s,t} = \mathrm{Sig}(X)_{s,u}\otimes\mathrm{Sig}(X)_{u,t},
$$

and after truncation $\pi_{\le N}\mathrm{Sig}(X)_{s,t} = \pi_{\le N}\mathrm{Sig}(X)_{s,u}\otimes_N\pi_{\le N}\mathrm{Sig}(X)_{u,t}$.

`block_size` splits the steps into blocks and emits one signature per block, along a block axis
at `axis`. With `accumulate=True` (the default), entry $j$ is the signature from the start up to
the end of block $j$. With `accumulate=False`, entry $j$ is the signature of block $j$ alone. The
blocks are computed side by side, so blocking does not lower the peak memory; see
[Long paths and gradients](performance.md#long-paths-and-gradients). The README example below
takes two independent blocks, composes them with
[`tensor_product`][tensordev.core.universal.Universal.tensor_product], and also checks the
shuffle identity of the next section:

--8<-- "README.md:signature_identities"

With `block_size=1` and `output_starting_point=True`, the output holds the signature of every
prefix of the path, starting with the identity:

```python
prefixes = td.path_signature(X, trunc=3, block_size=1, output_starting_point=True)
print([level.shape for level in prefixes])  # [(4, 33, 1), (4, 33, 2), (4, 33, 4), (4, 33, 8)]

first = td.tensor_to_flat(td.tensor_slice(prefixes)[:, 0])
last = td.tensor_to_flat(td.tensor_slice(prefixes)[:, -1])
np.testing.assert_allclose(first[:, 0], 1.0)  # the identity: 1, then zeros
np.testing.assert_allclose(first[:, 1:], 0.0)
np.testing.assert_allclose(last, td.tensor_to_flat(sig3), atol=1e-12)
```

`starting_point` continues a signature that is already known. This is Chen's identity in
streaming form:

```python
first_half = td.path_signature(X[:, :17], trunc=3)
continued = td.path_signature(X[:, 16:], trunc=3, starting_point=first_half)
np.testing.assert_allclose(td.tensor_to_flat(continued), td.tensor_to_flat(sig3), atol=1e-12)
```

`block_size` must divide the number of steps; `block_size=5` with 32 steps raises a
`ValueError`.

## The shuffle identity

Linear functionals of a signature multiply through the shuffle product: for words $u$ and $v$,

$$
\langle u, \mathrm{Sig}(X)_{0,T}\rangle\,\langle v, \mathrm{Sig}(X)_{0,T}\rangle = \langle u \sqcup\!\sqcup v, \mathrm{Sig}(X)_{0,T}\rangle .
$$

The README example above checks this for the letters $e_1$ and $e_2$, where
$e_1 \sqcup\!\sqcup e_2 = e_1e_2 + e_2e_1$. Here $e_1$ and $e_2$ are the basis vectors of $\mathbb{R}^2$, the
letters 0 and 1 of the layout in [Truncated tensors](tensor-algebra.md#truncated-tensors), which numbers
letters from 0 (the papers number them from 1). Full shuffle products need a core with shuffle plans,
so the example builds one with `make_core(..., precompute_shuffle=True)`; the default core
raises a `RuntimeError`. See [Shuffle plans and memory](tensor-algebra.md#shuffle-plans-and-memory).
With that `core`, level 2 of $e_1 \sqcup\!\sqcup e_2$ holds the words $e_1e_2$ and $e_2e_1$, at
flat indices 1 and 2:

```python
print(core.tensor_shuffle_product(e1, e2, trunc=2)[2])  # [0. 1. 1. 0.]
```

On the closed loops `X`, both sides of the README check are zero up to rounding. On the open
path `X_open` they are not, and the identity still holds; the concatenation $e_1e_2$ alone
would give a different value:

```python
lhs = td.tensor_inner_product(sig_open, core.tensor_shuffle_product(e1, e2, trunc=3))
rhs = td.tensor_inner_product(sig_open, e1) * td.tensor_inner_product(sig_open, e2)
print(np.round(rhs, 3))  # [-0.271 -0.025  1.059  0.155]
np.testing.assert_allclose(lhs, rhs, rtol=1e-12)

concatenation = td.tensor_inner_product(sig_open, td.tensor_product(e1, e2, trunc=3))
print(np.allclose(concatenation, rhs))  # False
```

For two letters the identity is integration by parts, $\langle ij, \mathrm{Sig}(X)_{0,T}\rangle + \langle ji, \mathrm{Sig}(X)_{0,T}\rangle = X^i_{0,T}X^j_{0,T}$ with
$X_{0,T} = X_T - X_0$: the symmetric part of level 2 is half the outer product of the increment.
No shuffle plans are needed to check it:

```python
increment = X_open[:, -1] - X_open[:, 0]
level2 = sig_open[2].reshape(4, 2, 2)
np.testing.assert_allclose(
    level2 + level2.transpose(0, 2, 1),
    increment[:, :, None] * increment[:, None, :],
    atol=1e-12,
)
```

The antisymmetric part of level 2 is the Lévy area; [Example paths](#example-paths) computes it
for a circle.

## Gradients

With the default execution policy, `path_signature` is differentiable and can be compiled with
`jax.jit`. Level 1 is $X_T - X_0$, so the gradient of its sum is $-1$ at the first point, $+1$ at
the last point and zero elsewhere:

```python
def level_one_sum(path):
    return td.path_signature(path, trunc=3)[1].sum()

grad = jax.grad(level_one_sum)(X)
expected = np.zeros(X.shape)
expected[:, 0], expected[:, -1] = -1.0, 1.0
np.testing.assert_allclose(grad, expected, atol=1e-12)
```

For a nonlinear function of the signature, compare the gradient with a central difference along
a random direction:

```python
def squared_norm(path):
    s = td.path_signature(path, trunc=3)
    return td.tensor_inner_product(s, s).sum()

grad = jax.jit(jax.grad(squared_norm))(X)
print(grad.shape)  # (4, 33, 2)

direction = random_trigonometric_polynomial_paths(batch=4, steps=32, dim=2, key=1)
h = 1e-6
central = (squared_norm(X + h * direction) - squared_norm(X - h * direction)) / (2 * h)
np.testing.assert_allclose(jnp.vdot(grad, direction), central, rtol=1e-6)
```

`execution="wordwise"` does not support an outer `jit`, `grad` or `vmap`.

## Free developments

[`free_development`][tensordev.free_development] generalizes `path_signature` to tensor-valued
paths $Z = (Z^{(1)}, \dots, Z^{(n)})$, where $Z^{(k)}$ takes values in $(\mathbb{R}^d)^{\otimes k}$. Each
increment $\Delta_i = (\Delta_i^{(1)}, \dots, \Delta_i^{(n)})$ is a tensor with zero scalar part, and the
free development is the truncated product

$$
D_N(Z) = \exp_\otimes(\Delta_1)\otimes_N\exp_\otimes(\Delta_2)\otimes_N\cdots\otimes_N\exp_\otimes(\Delta_S).
$$

This is the free development of P. K. Friz and P. P. Hager,
[*Expected Signature Kernels for Rough Paths*](https://arxiv.org/abs/2509.07893), Section 2.4, written
$\mathcal{S}(\gamma)$ there: the solution of
$\mathcal{S}(\gamma)_{s,t} = 1 + \int_s^t \mathcal{S}(\gamma)_{s,u}\otimes\dot\gamma(u)\,\mathrm{d}u$ for the
piecewise linear interpolation $\gamma$ of $Z$, truncated at level $N$. On a linear piece with
increment $\Delta_i$ it is $\exp_\otimes(\Delta_i)$, and the flow property
$\mathcal{S}(\gamma)_{s,t} = \mathcal{S}(\gamma)_{s,u}\otimes\mathcal{S}(\gamma)_{u,t}$ gives the product.

For $n = 1$ this is the signature. The input is a tuple of levels starting at degree 1. With the
default core, level $k$ has shape `batch + (S+1, d**k)`, or `batch + (S, d**k)` with
`increment_input=True`. Bidegree cores accept one combined first-level path `(X1,)` whose width
is `sum(core.dims)`. The keyword arguments are those of `path_signature` without `execution`.

```python
# level 1 in R^2, level 2 in R^2 ⊗ R^2, flattened to width 4
Z = (X, 0.5 * random_trigonometric_polynomial_paths(batch=4, steps=32, dim=4, key=2))
dev = td.free_development(Z, trunc=3)
print([level.shape for level in dev])  # [(4, 1), (4, 2), (4, 4), (4, 8)]

# the same product, one step at a time
steps = tuple(jnp.diff(level, axis=1) for level in Z)
manual = td.tensor_exponential(tuple(level[:, 0] for level in steps), trunc=3)
for i in range(1, 32):
    factor = td.tensor_exponential(tuple(level[:, i] for level in steps), trunc=3)
    manual = td.tensor_product(manual, factor, trunc=3)
np.testing.assert_allclose(td.tensor_to_flat(manual), td.tensor_to_flat(dev), atol=1e-12)

# increment_input=True takes the increments directly
from_steps = td.free_development(steps, trunc=3, increment_input=True)
np.testing.assert_allclose(td.tensor_to_flat(from_steps), td.tensor_to_flat(dev), atol=1e-12)
```

[`FreeDevelopment`][tensordev.FreeDevelopment] binds `trunc`, `core` and `seq_core`, as
`Signature` does:

```python
develop = td.FreeDevelopment(trunc=3)
np.testing.assert_allclose(td.tensor_to_flat(develop(Z)), td.tensor_to_flat(dev))
```

The inner product of the untruncated free developments,
$\langle D(Z), D(W)\rangle = \lim_{N\to\infty}\langle D_N(Z), D_N(W)\rangle$, is the
[free kernel](kernels.md#free-kernel), which a PDE solver computes without truncation.

## Piecewise log-linear paths

A block of steps can be replaced by one log-linear piece: the tensor logarithm of the block's
signature, used as a single increment of a tensor-valued path.

--8<-- "README.md:free_development"

[`tensor_exponential`][tensordev.core.universal.Universal.tensor_exponential] and
[`tensor_logarithm`][tensordev.core.universal.Universal.tensor_logarithm] both take levels that
start at degree 1. For $X = (X_1, X_2, \dots)$, `tensor_exponential` sets $X_0 = 0$, and
`tensor_logarithm` reads its input as $1 + H$ with $H = (0, X_1, X_2, \dots)$:

$$
\exp(X) = \sum_{n=0}^{N} \frac{X^{\otimes n}}{n!}, \qquad
\log(1 + H) = \sum_{n=1}^{N} \frac{(-1)^{n+1}}{n}\, H^{\otimes n},
$$

truncated at level $N$ = `trunc`. `output_zero_level=False` drops level 0, which is 1 for the
exponential and 0 for the logarithm. This is why the example passes `block_sigs[1:]` and asks for
`output_zero_level=False`: the logarithms are then increments that `free_development` accepts.
On tensors with scalar part 1, such as signatures, the two maps are inverse within one
truncation:

```python
np.testing.assert_allclose(
    td.tensor_to_flat(td.tensor_exponential(log_sigs, trunc=3)),
    td.tensor_to_flat(block_sigs),
    atol=1e-12,
)
```

The log-linear path therefore reproduces the signature exactly when the block logarithms keep
every level up to $N$. With block logarithms truncated at a lower degree $M < N$, the development
agrees with the signature up to level $M$ only:

```python
sig3 = td.path_signature(X, trunc=3)
block_sigs_2 = td.free_development((X,), trunc=2, block_size=8, accumulate=False)
log_sigs_2 = td.tensor_logarithm(block_sigs_2[1:], trunc=2, output_zero_level=False)
approx = td.free_development(log_sigs_2, trunc=3, increment_input=True)

for k in range(3):  # levels 0, 1 and 2 agree
    np.testing.assert_allclose(approx[k], sig3[k], atol=1e-12)
print(np.allclose(approx[3], sig3[3]))  # False
```

[`HigherOrderKernel`](kernels.md#higher-order-kernel) uses this construction: `log_steps` is the
block size and `log_degree` the degree $M$.

## Example paths

`tensordev.util` provides path generators for tests and examples, and helpers that prepare
inputs:

--8<-- "README.md:util"

| Function | Description |
|---|---|
| [`random_trigonometric_polynomial_paths`][tensordev.util.random_trigonometric_polynomial_paths] | Batched random trigonometric-polynomial paths, shape `(batch, steps + 1, dim)`, starting and ending at 0: closed loops, so level 1 of their signatures is zero. `key` is a JAX key or an integer. |
| [`unit_speed_paths`][tensordev.util.unit_speed_paths] | Smooth unit-speed paths on $[0, 1]$, generated from `seed`, as a NumPy array of shape `(n_paths, int(1 / dt) + 1, dim)`, with `dim` 2 or 3. `dt` must divide 1 and `dt_fine` (default `0.001`) must divide `dt`. |
| [`integrated_ou_first_on_path`][tensordev.util.integrated_ou_first_on_path] | Tensor-valued path whose flattened coordinates are independent integrated Ornstein-Uhlenbeck processes: a tuple of `trunc` levels, level $k$ of shape `(batch, steps + 1, dim**k)` and scaled by `level_lambda**k` (and $1/k!$ with `factorial=True`). |
| [`deterministic_trigonometric_path_pair`][tensordev.util.deterministic_trigonometric_path_pair] | Two fixed trigonometric paths `x`, `w` of shape `(length, dim)` on a given grid, meant as base signals for kernel experiments and benchmarks. |
| [`perturb_path_batch`][tensordev.util.perturb_path_batch] | A batch `(batch_size, length, d)` of deterministic trigonometric perturbations of one base path, with amplitudes that grow with the batch index. |
| [`path_to_increments`][tensordev.util.path_to_increments] | Increments of each level of a tensor-valued path along axis `-2`. |
| [`velocity_to_increments`][tensordev.util.velocity_to_increments] | Integrates a velocity callable, `t -> tuple of levels`, over the intervals of a 1-D grid (left, midpoint or trapezoid rule, optional dyadic refinement); one increment array `(S, width)` per level. |
| [`bucket_pad_ragged_paths`][tensordev.util.bucket_pad_ragged_paths] | Groups paths of different lengths into dense chunks, padded by repeating each path's last value. |

Padding by the last value adds zero increments, and $\exp(0)$ is the identity, so padding does
not change a signature. [Batching](kernels.md#batching) uses this for ragged kernel inputs.

`velocity_to_increments` turns a velocity into increments that `increment_input=True` accepts.
The circle of radius $1/(2\pi)$, traversed once with unit speed, encloses the area $1/(4\pi)$,
and its Lévy area $\tfrac12(\langle e_1e_2, \mathrm{Sig}(X)_{0,1}\rangle - \langle e_2e_1, \mathrm{Sig}(X)_{0,1}\rangle)$ approaches that value as the grid is refined:

```python
from tensordev.util import velocity_to_increments

def velocity(t):
    return (jnp.stack([jnp.cos(2 * jnp.pi * t), jnp.sin(2 * jnp.pi * t)], axis=-1),)

grid = jnp.linspace(0.0, 1.0, 65)
(circle,) = velocity_to_increments(velocity, grid, dyadic_order=2)
print(circle.shape)  # (256, 2)

level2 = td.path_signature(circle, trunc=2, increment_input=True)[2]
levy_area = 0.5 * (level2[1] - level2[2])  # words e1 e2 and e2 e1
np.testing.assert_allclose(levy_area, 1 / (4 * np.pi), rtol=1e-3)
```
