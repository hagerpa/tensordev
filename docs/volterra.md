# Volterra signatures

A Volterra signature develops a path into the tensor algebra like the signature does, but
weights every increment by a kernel that looks back in time. This page gives the definition,
the kernels that `tensordev` provides, and the options of [`vsig`][tensordev.vsig]: schemes,
blocking, rectangular truncation and gradients. The examples share one batch of paths and use
64-bit floats:

```python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import tensordev as td
from tensordev.util import random_trigonometric_polynomial_paths

X = random_trigonometric_polynomial_paths(batch=4, steps=32, dim=2, key=0)
print(X.shape)  # (4, 33, 2): 4 paths, 33 nodes, d = 2
```

## The Volterra signature

--8<-- "README.md:volterra_overview"

Let $x\colon[0,T]\to\mathbb{R}^d$ be a Lipschitz path (`vsig` reads `X` as the piecewise linear
path through its nodes) and $K(t,s)\in\mathbb{R}^{m\times d}$, $0\le s\le t\le T$, a kernel with
$\sup_{t}\int_0^t|K(t,s)|\,\mathrm{d}s<\infty$ (all kernels below qualify), and write
$K^{i}(t,s)$ for the $i$-th row of $K(t,s)$.
Following [Hager, Harang, Pelizzari and Tindel (2026)](https://arxiv.org/abs/2603.04525), the
Volterra signature $\mathrm{VSig}(x;K)$ has, for a word $i_1\cdots i_n$ over
$\{1,\dots,m\}$ and times $s\le t\le\tau$, the coordinate

$$
\mathrm{VSig}(x;K)^{i_1\cdots i_n,\tau}_{s,t}=\int_{s<r_1<\cdots<r_n<t}\,\prod_{l=1}^{n}K^{i_l}(r_{l+1},r_l)\,\mathrm{d}x_{r_l},\qquad r_{n+1}=\tau .
$$

Each increment $\mathrm{d}x_{r_l}$ is weighted by the kernel between its own time and the time
of the next letter; the last letter looks back from the readout time $\tau$. Collected over
all words, $\mathbf{z}^{\tau}_{s,t}=\mathrm{VSig}(x;K)^{\tau}_{s,t}\in T((\mathbb{R}^m))$ is
the unique solution of the linear Volterra equation in the tensor algebra

$$
\mathbf{z}^{\tau}_{s,t}=1+\int_s^t\mathbf{z}^{u}_{s,u}\otimes K(\tau,u)\,\mathrm{d}x_u .
$$

[`vsig`][tensordev.vsig] returns $\mathbf{z}^{T}_{0,T}$, truncated at degree $N$, where $T$ is
the last time of the grid. With blocking and the default `accumulate=True` it returns
$\mathbf{z}^{t}_{0,t}$ at the end $t$ of every block; with `accumulate=False` it returns
$\mathbf{z}^{t}_{t',t}$ for every block $[t',t]$, which for the convolution kernels below is the
Volterra signature of the block as a separate path. The letters are the $m$ output coordinates of the kernel, so the result is shaped like
a signature over $\mathbb{R}^m$, not over $\mathbb{R}^d$. For a constant kernel $K\equiv A$
the definition reduces to the signature of the path $Ax$. A readout time $\tau$ later than $t$
is available for finite state-space kernels, as `tau_dt` on
[State-space signatures](state-space.md#volterra-signatures-by-state-recursion).

## Convolution kernels

All kernels in `tensordev.volterra` are convolution kernels

$$
K(t,s)=\sum_{p=1}^{q}k_p(t-s)\,A_p,\qquad A_p\in\mathbb{R}^{m\times d},
$$

with scalar functions $k_p$ and fractional orders $\beta_p>0$ ($\beta_p=1$ for smooth
kernels). The matrices $A_p$ are stacked into one array `A` of shape `(q, m, d)`, and every
kernel reads its properties `q`, `m` and `path_dim` ($d$) from that shape.

--8<-- "README.md:volterra_kernels"

The constructors return the classes below and check their arguments eagerly; shapes or values
outside these ranges raise `ValueError`:

| Class | Constructor | `q` | Arguments |
|---|---|---|---|
| [`FractionalKernel`][tensordev.FractionalKernel] | [`ConvolutionKernel.fractional`][tensordev.ConvolutionKernel.fractional] | any | `beta` of shape `(q,)` with positive entries; `A` of shape `(q, m, d)` |
| [`GammaKernel`][tensordev.GammaKernel] | [`ConvolutionKernel.gamma`][tensordev.ConvolutionKernel.gamma] | 1 | `beta > 0`, `scale > 0` and `rate >= 0`, each a scalar or of shape `(1,)`; `A` of shape `(1, m, d)`; static `quad_order`, the number of quadrature nodes of the fallback for large `rate` times the step. The constructor defaults `scale` and `rate` to 1. |
| [`FSSKConvolutionKernel`][tensordev.FSSKConvolutionKernel] | [`ConvolutionKernel.fssk`][tensordev.ConvolutionKernel.fssk] | any | an [`FSSK`][tensordev.FSSK]; `beta` is fixed to ones |

The output dimension $m$ need not equal the path dimension $d$, and components may have
different orders:

```python
A2 = jnp.stack([jnp.eye(2), jnp.array([[0.0, 1.0], [1.0, 0.0]])])  # (q=2, m=2, d=2)
kernel_2 = td.ConvolutionKernel.fractional(beta=jnp.array([0.7, 1.0]), A=A2)
print(type(kernel_2).__name__, kernel_2.q, kernel_2.m, kernel_2.path_dim)
# FractionalKernel 2 2 2

A3 = jnp.array([[[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]])  # (q=1, m=3, d=2)
kernel_3 = td.ConvolutionKernel.gamma(
    beta=jnp.array([0.8]),
    rate=jnp.array([2.0]),
    scale=jnp.array([1.0]),
    A=A3,
)
print(type(kernel_3).__name__, kernel_3.q, kernel_3.m, kernel_3.path_dim)
# GammaKernel 1 3 2
```

A finite state-space kernel (FSSK) has the scalar kernels
$k_p(t)=\mathbf{1}^\top e^{-\Lambda t}b_p$. Wrap an [`FSSK`][tensordev.FSSK] with
`ConvolutionKernel.fssk` to use it with `vsig`; here
$k(t)=\tfrac12 e^{-t}+\tfrac12 e^{-4t}$:

```python
from tensordev.sss import FSSK

fssk = FSSK.from_jordan(
    real_rates=jnp.array([1.0, 4.0]),
    real_sizes=(1, 1),
    A=jnp.eye(2)[None],
    b=jnp.array([[0.5, 0.5]]),
)
kernel_s = td.ConvolutionKernel.fssk(fssk)
print(type(kernel_s).__name__, kernel_s.q, kernel_s.state_dim, kernel_s.beta)
# FSSKConvolutionKernel 1 2 [1.]
```

[State-space signatures](state-space.md) computes the Volterra signature of an FSSK by a state
recursion that also supports streaming;
[Comparison with vsig](state-space.md#comparison-with-vsig) compares both routes.

### The classical signature as a special case

With $\beta=1$ the fractional kernel is $k\equiv 1$, so $K\equiv A$ and `vsig` returns the
signature of $Ax$. With $A$ the identity this is [`path_signature`][tensordev.path_signature]:

```python
flat = td.ConvolutionKernel.fractional(beta=jnp.array([1.0]), A=jnp.eye(2)[None])
np.testing.assert_allclose(
    td.tensor_to_flat(td.vsig(X, kernel=flat, dt=1.0 / 32, trunc=3)),
    td.tensor_to_flat(td.path_signature(X, trunc=3)),
    atol=1e-12,
)

lifted = td.ConvolutionKernel.fractional(beta=jnp.array([1.0]), A=A3)  # m = 3
np.testing.assert_allclose(
    td.tensor_to_flat(td.vsig(X, kernel=lifted, dt=1.0 / 32, trunc=3)),
    td.tensor_to_flat(td.path_signature(X @ A3[0].T, trunc=3)),
    atol=1e-12,
)
```

### Approximating a fractional kernel by an FSSK

For $0<\beta<1$, the fractional kernel has the Laplace representation

$$
\frac{t^{\beta-1}}{\Gamma(\beta)}=\frac{\sin(\pi\beta)}{\pi}\int_0^\infty e^{-xt}\,x^{-\beta}\,\mathrm{d}x .
$$

`tensordev.sss.rough_approx.fractional_fssk(beta=..., R=..., A=..., T=1.0)` accepts
$\tfrac12<\beta<1$. It replaces the integral by $R$ exponentials with logarithmically spaced rates and fits nonnegative weights
in relative error on a logarithmic grid of $[10^{-3}T,\,T]$. The result is an
[`FSSK`][tensordev.FSSK] with a diagonal state matrix of size $R$. The function is not
re-exported, so import it from its module. The fit needs SciPy, which the `rough` extra
installs: `pip install "tensordev[rough]"`.

```python
from tensordev.sss.rough_approx import fractional_fssk

approx = fractional_fssk(beta=0.75, R=8, A=jnp.eye(2)[None])
t = jnp.geomspace(1e-2, 1.0, 50)
k_approx = approx.Lambda.expm(t).sum(axis=-2) @ approx.b[0]  # 1ᵀ exp(-Λt) b
k_exact = t ** (0.75 - 1.0) / jax.scipy.special.gamma(0.75)
print(f"{float(jnp.max(jnp.abs(k_approx / k_exact - 1.0))):.1e}")  # 7.9e-04
```

## Computing a Volterra signature

--8<-- "README.md:volterra_example"

The function [`vsig`][tensordev.vsig] and the method
[`VolterraSignature.vsig`][tensordev.VolterraSignature.vsig] take the same arguments. The class
binds `kernel`, `trunc`, `core` and `seq_core`; `trunc` is static, so a new value retraces.
[`VolterraSignature.fractional`][tensordev.VolterraSignature.fractional] and
[`VolterraSignature.gamma`][tensordev.VolterraSignature.gamma] forward their remaining keyword
arguments to the kernel classes.

| Argument | Default | Description |
|---|---|---|
| `X` | required | Path nodes, or increments with `increment_input=True`. The trailing axis has size `kernel.path_dim`; `axis` is the step axis. |
| `kernel` | required | A convolution kernel (keyword-only, as all arguments after `X`). |
| `trunc` | `None` | An integer $N$ for a total-degree core, a pair $(N, M)$ for a bidegree core. May be omitted when the core has a default truncation; the default JAX core has none and raises. |
| `dt` | `1.0` | Step size. A scalar gives a uniform grid; a 1-D array of length $S$ gives a non-uniform grid with times at its cumulative sums. |
| `axis` | `-2` | Step axis of `X`. |
| `block_size` | `None` | Steps per emitted block, counted on the original grid; $S$ must be divisible by it. `None` returns the terminal value only. |
| `accumulate` | `True` | With blocks: `True` carries the history across blocks and returns the signature at each block end; `False` treats every block as an independent path. Ignored without `block_size`. |
| `starting_point` | `None` | Tensor prepended to the output when `output_starting_point=True`. Output only: it does not seed the recursion. Defaults to the unit. |
| `output_starting_point` | `False` | Prepend the starting point along `axis`. |
| `increment_input` | `False` | `X` holds increments; skip differencing. |
| `order` | `0` | `0`, `1` or `2`; see [Schemes](#schemes). |
| `dyadic_order` | `0` | Split every increment into $2^k$ equal sub-increments, $k$ = `dyadic_order`. |
| `scheme` | `"auto"` | `"auto"`, `"fft"`, `"quadratic"` or `"adams"`; see [Schemes](#schemes). |
| `lag_tables` | `None` | Precomputed lag tables for the FFT scheme, from `tensordev.volterra.iteration_fft.precompute_lag_tables`. The caller must reuse them only with the same kernel, step size, dtype and basis exponents (set by `order` and `beta`). |
| `core`, `seq_core` | `None` | Algebra and sequential cores. Omitted values resolve to the process-wide default pair; the output uses the format of `core`. |

For a total-degree core the result is a tuple of $N+1$ arrays, one per degree, with shape
`batch + (m**n,)` at degree $n$; for a bidegree core it is a
[`BigradedTensor`][tensordev.BigradedTensor]:

```python
sig = td.vsig(X, kernel=kernel, dt=dt, trunc=3)
print([level.shape for level in sig])  # [(4, 1), (4, 2), (4, 4), (4, 8)]

nonuniform = td.vsig(X, kernel=kernel, dt=jnp.full((32,), dt), trunc=3)
np.testing.assert_allclose(td.tensor_to_flat(nonuniform), td.tensor_to_flat(sig), atol=1e-12)
```

### Blocks and trajectories

`block_size` emits a block axis at `axis`. With `accumulate=True` the full recursion runs and
the output is the signature at every block end, so blocking does not reduce the cost. With
`accumulate=False` every block restarts from the unit, as a separate path:

```python
blocks = td.vsig(X, kernel=kernel, dt=dt, trunc=3, block_size=8)
print([level.shape for level in blocks])  # [(4, 4, 1), (4, 4, 2), (4, 4, 4), (4, 4, 8)]
np.testing.assert_allclose(blocks[3][:, -1], sig[3], atol=1e-12)

independent = td.vsig(X, kernel=kernel, dt=dt, trunc=3, block_size=8, accumulate=False)
second_block = td.vsig(X[:, 8:17], kernel=kernel, dt=dt, trunc=3)
np.testing.assert_allclose(independent[3][:, 1], second_block[3], atol=1e-12)

trajectory = td.vsig(
    X, kernel=kernel, dt=dt, trunc=3, block_size=1, output_starting_point=True
)
print(trajectory[1].shape)  # (4, 33, 2): the unit at t = 0, then one value per node
```

### Compilation

The first call with a new static configuration compiles a solver; later calls reuse it. The
configuration covers the core pair, the truncation, the kernel class and array shapes,
`axis`, `block_size`, `accumulate`, `output_starting_point`, `increment_input`, `order`,
`dyadic_order` and `scheme`. Kernel values stay dynamic, so new values of `beta` or `A` do
not recompile, with one exception: for `order > 0` and a scheme other than `"adams"` the solver
is specialized on the values of `beta`.

`vsig` also runs inside `jax.jit` when the kernel is built outside and closed over. Kernel
constructors check their values eagerly, so building a kernel inside `jax.jit`, or passing one
as a traced argument, raises `TracerBoolConversionError`. Under `jax.grad` alone this is
allowed; see [Gradients](#gradients).

## Schemes

--8<-- "README.md:volterra_schemes"

| `scheme` | Grid | Kernels | Method |
|---|---|---|---|
| `"auto"` (default) | any | any | `"fft"` when `dt` is a scalar and the cost estimate below favors it, otherwise `"quadratic"`. Never selects `"adams"`. |
| `"fft"` | uniform: scalar `dt` | any | FFT convolution in time. An array `dt` raises `ValueError`. |
| `"quadratic"` | uniform or non-uniform: scalar or 1-D `dt` | any | Quadratic recursion over all pairs of steps. |
| `"adams"` | uniform: scalar `dt` | [`FractionalKernel`][tensordev.FractionalKernel] with `q=1` | Adams product integration with FFT convolution: `order=0` Euler, `order>=1` predictor-corrector. Other kernels raise `TypeError`, `q > 1` and an array `dt` raise `ValueError`. |

[`vsig_fft`][tensordev.volterra.vsig_fft] in `tensordev.volterra` is `vsig` with
`scheme="fft"` preset.

For `"auto"`, let $S$ be the number of increments, $S_{\mathrm{eff}}=2^kS$ with $k$ = `dyadic_order`,
$J$ the smallest power of two with $J\ge 2S_{\mathrm{eff}}-1$, and $N$ the largest total
degree of the truncation ($N+M$ for a bidegree $(N,M)$). The FFT scheme is selected when
$S_{\mathrm{eff}}>1$ and

$$
S_{\mathrm{eff}}\,\log_2 J\;N^{q}\;<\;S_{\mathrm{eff}}^{2}\ \text{ for } q=1,\qquad
S_{\mathrm{eff}}\,\log_2 J\;N^{q}\;<\;S_{\mathrm{eff}}^{2}\,N\ \text{ for } q>1 .
$$

For the 32-step paths above with $N=3$ and $q=1$, the left side is $32\cdot 6\cdot 3=576$ and
the right side $1024$, so `"auto"` runs the FFT scheme:

```python
auto = td.vsig(X, kernel=kernel, dt=dt, trunc=3)
fft = td.vsig(X, kernel=kernel, dt=dt, trunc=3, scheme="fft")
quadratic = td.vsig(X, kernel=kernel, dt=dt, trunc=3, scheme="quadratic")
np.testing.assert_allclose(td.tensor_to_flat(auto), td.tensor_to_flat(fft), atol=1e-12)
np.testing.assert_allclose(td.tensor_to_flat(fft), td.tensor_to_flat(quadratic), atol=1e-12)
```

`order` selects the local basis of the quadratic and FFT schemes. On a step $[t_i,t_{i+1}]$
these schemes need the history $\mathbf{z}^{t_i+u}_{0,t_i}$ as a function of the readout offset
$u\in[0,t_{i+1}-t_i]$: `0` approximates it by its left-point value $\mathbf{z}^{t_i}_{0,t_i}$,
`1` interpolates it in the basis $\{1, u^{\beta_p}, u\}$ and `2` in
$\{1, u^{\beta_p}, u, u^{\beta_p+1}, u^2\}$, with the exponents of every component $p$ (an
exponent that occurs twice, such as $\beta_p=1$, is used once). For `"adams"`, `order=0` gives the Euler weights and `order=1` or `2` the
predictor-corrector weights. `dyadic_order` refines the grid before any scheme runs. All four schemes discretize the same equation, so on a coarse grid
they differ by their discretization errors. The quadratic and FFT schemes agree to rounding at
equal `order`; the Adams scheme does not, and refining the grid brings them together:

```python
for dyadic_order in (0, 2):
    pc = td.vsig(X, kernel=kernel, dt=dt, trunc=3, scheme="adams", order=1, dyadic_order=dyadic_order)
    quad = td.vsig(X, kernel=kernel, dt=dt, trunc=3, order=1, dyadic_order=dyadic_order)
    gap = jnp.max(jnp.abs(td.tensor_to_flat(pc) - td.tensor_to_flat(quad)))
    print(dyadic_order, f"{float(gap):.1e}")
# 0 1.4e-02
# 2 1.4e-03
```

The low-level functions [`quadratic_iteration`][tensordev.volterra.quadratic_iteration],
[`fft_iteration`][tensordev.volterra.fft_iteration] and
[`pc_iteration`][tensordev.volterra.pc_iteration] in `tensordev.volterra` run one scheme on
increments that are already on the final grid. `vsig` does the differencing, the dyadic
refinement, the scheme selection and the blocking before it calls them. `pc_iteration`
defaults to `order=1`, `vsig` to `order=0`.

## Rectangular truncation

A bidegree core splits the $m$ output letters into two groups of sizes `dims = (m', m'')`,
the first $m'$ and the last $m''$ coordinates, and a bidegree truncation $(N, M)$ keeps the
words with at most $N$ letters from the first group and at most $M$ from the second. Build the
core with [`make_core`][tensordev.make_core]; `sum(dims)` must equal `kernel.m`, otherwise
`vsig` raises `ValueError`. Bidegree tensors and their layout are described on
[Tensor algebra](tensor-algebra.md).

The result `rectangular` of the example in
[Computing a Volterra signature](#computing-a-volterra-signature) uses the core default
$(N, M) = (2, 1)$. It equals the total-degree result of degree $N+M$ restricted to the kept
words; the core method `tensor_from_total` does that restriction:

```python
print(rectangular.grades)  # ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (2, 1))

total = td.vsig(X, kernel=kernel, dt=dt, trunc=3)  # N + M = 2 + 1
expected = bicore.tensor_from_total(total, trunc=(2, 1))
for grade in rectangular.grades:
    np.testing.assert_allclose(rectangular[grade], expected[grade], atol=1e-12)
```

Kernels with `q = 1` need no shuffle plans. For `q > 1` and a truncation of total degree above
one, construct the core with `precompute_shuffle="generator"` (or `True`); otherwise the
quadratic and FFT schemes raise `RuntimeError`. See
[Shuffle plans and memory](tensor-algebra.md#shuffle-plans-and-memory).

```python
generator_core = td.make_core(dims=(1, 1), max_trunc=(2, 1), precompute_shuffle="generator")
rectangular_2 = td.vsig(X, kernel=kernel_2, dt=dt, trunc=(2, 1), core=generator_core)
print(type(rectangular_2).__name__, rectangular_2[2, 1].shape)  # BigradedTensor (4, 3)
```

Partially symmetrized and shear cores are accepted in the same way, and the result uses their
coordinates.

## Gradients

`vsig` is differentiable with respect to the path, the matrices `A` and the kernel parameters:
`beta` of a `FractionalKernel`, and `beta`, `scale` and `rate` of a `GammaKernel`. This holds for
the quadratic and FFT schemes at `order` 0, 1 and 2, and for `scheme="adams"`, which accepts a
`FractionalKernel`. Build the kernel inside the differentiated function:

```python
def level_two_energy(beta):
    kernel = td.FractionalKernel(beta=beta, A=jnp.eye(2)[None])
    return jnp.sum(td.vsig(X, kernel=kernel, dt=dt, trunc=3)[2] ** 2)

beta = jnp.array([0.8])
grad = jax.grad(level_two_energy)(beta)

eps = 1e-6
central = (level_two_energy(beta + eps) - level_two_energy(beta - eps)) / (2 * eps)
np.testing.assert_allclose(grad[0], central, rtol=1e-6)
```

The same check for `beta` and `rate` of a `GammaKernel`:

```python
def gamma_energy(beta, rate):
    kernel = td.GammaKernel(beta=beta, scale=jnp.array([1.0]), rate=rate, A=jnp.eye(2)[None])
    return jnp.sum(td.vsig(X, kernel=kernel, dt=dt, trunc=3)[2] ** 2)

beta, rate = jnp.array([0.8]), jnp.array([1.0])
grad_beta, grad_rate = jax.grad(gamma_energy, argnums=(0, 1))(beta, rate)
np.testing.assert_allclose(
    grad_beta[0], (gamma_energy(beta + eps, rate) - gamma_energy(beta - eps, rate)) / (2 * eps), rtol=1e-6
)
np.testing.assert_allclose(
    grad_rate[0], (gamma_energy(beta, rate + eps) - gamma_energy(beta, rate - eps)) / (2 * eps), rtol=1e-6
)
```

The derivatives with respect to `beta` come from `tensordev.util.special.betainc`, a regularized
incomplete beta function that is differentiable in all three arguments.

!!! warning "Limits of the kernel-parameter gradients"
    - First derivatives work with `jax.grad`, `jax.jacfwd` and `jax.jacrev`. For second
      derivatives use `jax.jacrev(jax.jacfwd(...))`: `jax.hessian` and `jax.grad(jax.grad(...))`
      raise through `vsig`, because JAX drops the custom derivative rule of `betainc` when it
      linearizes a loop.
    - With `order > 0` and a scheme other than `"adams"`, the interpolation basis depends on the
      value of `beta`. A traced `beta` therefore runs the solver without its cached compilation,
      which is slower.
    - Pass kernel parameters to `jax.jit` as closed-over constants, not as jit arguments: the
      kernel constructors check their arguments eagerly, which fails on traced values.
    - Gradients with respect to the step size `dt` are not supported.
