# Signature kernels

A signature kernel compares two paths through the inner product of their signatures,
$k(X, Y) = \langle \mathrm{Sig}(X), \mathrm{Sig}(Y)\rangle$, computed without truncation by solving a PDE. This page
covers the kernel classes in `tensordev.kernel`, the empirical statistics they share, gradients,
and batching.

--8<-- "README.md:kernels_overview"

## Empirical statistics

The README example sets up the session for this page: two batches `X` and `Y` of four paths in
$\mathbb{R}^2$, and a signature kernel `k`.

--8<-- "README.md:kernels_example"

Every kernel class inherits these methods from [`BaseKernel`][tensordev.kernel.BaseKernel]. All
of them also take the keywords `max_batch=100` and `increment_input=None`, described under
[Batching](#batching).

| Method | Returns | Value |
|---|---|---|
| [`compute_kernel(X, Y)`][tensordev.kernel.BaseKernel.compute_kernel] | `batch` | $k(x_i, y_i)$. `X` and `Y` must have the same batch shape. |
| [`compute_Gram(X, Y=None, *, sym=False)`][tensordev.kernel.BaseKernel.compute_Gram] | `batch_x + batch_y` | $k(x_i, y_j)$. `Y=None` sets `Y = X` and `sym=True`. With `sym=True`, only blocks on and above the diagonal are computed and then mirrored; this needs equal batch shapes. |
| [`compute_mmd(X, Y)`][tensordev.kernel.BaseKernel.compute_mmd] | scalar | Unbiased estimate of the squared MMD, below. |
| [`compute_scoring_rule(X, y)`][tensordev.kernel.BaseKernel.compute_scoring_rule] | scalar | $\frac{1}{n(n-1)}\sum_{i\neq j} k(x_i, x_j) - \frac{2}{n}\sum_i k(x_i, y)$ for one sample `y`. |
| [`compute_expected_scoring_rule(X, Y)`][tensordev.kernel.BaseKernel.compute_expected_scoring_rule] | scalar | The scoring rule with its second term averaged over all $y_j$ in `Y`. |

For samples $x_1, \dots, x_n$ and $y_1, \dots, y_m$, `compute_mmd` returns

$$
\widehat{\mathrm{MMD}}^2 = \frac{1}{n(n-1)}\sum_{i\neq j} k(x_i, x_j) + \frac{1}{m(m-1)}\sum_{i\neq j} k(y_i, y_j) - \frac{2}{nm}\sum_{i,j} k(x_i, y_j).
$$

The within-sample terms leave out the diagonal, so the estimate is unbiased; it can be negative,
and each sample needs at least two paths. The value is the squared MMD, not its square root.

```python
import jax.numpy as jnp

Kyy = k.compute_Gram(Y)
n = m = 4
unbiased = (
    (Kxx.sum() - jnp.trace(Kxx)) / (n * (n - 1))
    + (Kyy.sum() - jnp.trace(Kyy)) / (m * (m - 1))
    - 2 * gram.mean()
)
np.testing.assert_allclose(mmd, unbiased, rtol=1e-12)
np.testing.assert_allclose(jnp.diag(gram), vals, rtol=1e-12)

# the expected scoring rule averages the scoring rule over Y
rules = jnp.stack([k.compute_scoring_rule(X, Y[j]) for j in range(4)])
np.testing.assert_allclose(esr, rules.mean(), rtol=1e-12)
```

Calling a kernel object directly returns more than the terminal value. With `evaluate="grid"`,
`k(X, Y)` returns the PDE solution on the whole grid, $k(X|_{[0,s]}, Y|_{[0,t]})$ at every grid
point $(s, t)$. For paths with $S$ and $T$ intervals and dyadic orders $a$ and $b$ (see the next
section), the refined grid has $2^a S + 1$ by $2^b T + 1$ points. `pairwise=True` gives the Gram
layout.

```python
surface = k(X, Y, evaluate="grid")
print(surface.shape)  # (4, 65, 65)

np.testing.assert_allclose(surface[:, 0, :], 1.0)                 # the kernel starts at 1
np.testing.assert_allclose(surface[:, -1, -1], vals, rtol=1e-12)  # and ends at k(x_i, y_i)
print(k(X, Y, evaluate="grid", pairwise=True).shape)  # (4, 4, 65, 65)
```

The next README example also uses an FSSK kernel, which needs a finite state-space kernel. This
one has a single exponential mode, $K(t, s) = e^{-(t-s)} I_2$;
[State-space signatures](state-space.md) explains the constructors.

```python
from tensordev.sss import StateSpaceSignature

sss = StateSpaceSignature.from_jordan(
    real_rates=jnp.array([1.0]),
    real_sizes=(1,),
    A=jnp.eye(2)[None],  # (q=1, m=2, d=2)
    b=jnp.ones((1, 1)),  # (q=1, R=1)
    trunc=3,
)
```

--8<-- "README.md:kernels_dropin"

## The classical signature kernel

For two paths $X$ and $Y$, the function $u(s, t) = k(X|_{[0,s]}, Y|_{[0,t]})$ solves the
Goursat problem

$$
\frac{\partial^2 u}{\partial s\,\partial t}(s, t) = \langle \dot X_s, \dot Y_t\rangle\, u(s, t), \qquad u(0, \cdot) = u(\cdot, 0) = 1,
$$

as shown by C. Salvi, T. Cass, J. Foster, T. Lyons and W. Yang,
[*The Signature Kernel is the Solution of a Goursat PDE*](https://arxiv.org/abs/2006.14794),
SIAM Journal on Mathematics of Data Science, 2021.
[`SigKernel`][tensordev.SigKernel] solves this problem for piecewise linear paths on the grid of
path points, with a second-order finite-difference stencil inspired by `sigkernel`
([Acknowledgements](background.md#acknowledgements)).

| Field | Default | Description |
|---|---|---|
| `dyadic_order` | `0` | Split every interval into `2**dyadic_order` sub-intervals before solving. An integer, or a pair `(order_x, order_y)`. |
| `static_kernel` | `LinearKernel(scale=1.0)` | Kernel on path values that replaces the increment inner product; see [Static kernels](#static-kernels). |
| `backend` | `"scan"` | `"scan"` sweeps the grid row by row. `"wavefront"` processes anti-diagonals in parallel with `jax.vmap` and keeps two diagonal buffers. |
| `num_devices` | `1` | Number of JAX devices that share the batch; see [Batching](#batching). |
| `increment_input` | `False` | Inputs are increments of shape `(batch, S, dim)` instead of path values. |

Inputs have shape `(batch, length, dim)`; a single path of shape `(length, dim)` becomes a batch of
one. Each path needs at least two points.

`dyadic_order` trades run time for accuracy. Each refinement multiplies the number of grid cells
by four (with the same order on both axes). The truncated inner product at level 10 serves as
reference here:

```python
import tensordev as td

reference = td.tensor_inner_product(
    td.path_signature(X, trunc=10), td.path_signature(Y, trunc=10)
)
for order in range(4):
    values = SigKernel(dyadic_order=order).compute_kernel(X, Y)
    error = jnp.max(jnp.abs(values - reference) / jnp.abs(reference))
    print(order, f"{error:.1e}")
```

```text
0 8.3e-04
1 2.1e-04
2 5.2e-05
3 1.3e-05
```

The relative error falls by about a factor of four per refinement in this example.

`SigKernel`, `FreeKernel` and `HigherOrderKernel` require the installed default core to use
standard total-degree coordinates. Under a bidegree or shear default they raise a
`RuntimeError` before any numerical work; call `td.reset_default_core()` first.

## Static kernels

`static_kernel` replaces the increment inner product $\langle \Delta x_i, \Delta y_j\rangle$ of
grid cell $(i, j)$ by the mixed difference of a kernel $\kappa$ on path values:

$$
G_{ij} = \kappa(x_{i+1}, y_{j+1}) - \kappa(x_i, y_{j+1}) - \kappa(x_{i+1}, y_j) + \kappa(x_i, y_j).
$$

With the default `LinearKernel(scale=1.0)`, $G_{ij} = \langle \Delta x_i, \Delta y_j\rangle$
exactly.

With `dyadic_order` $(a, b)$, every sub-cell of cell $(i, j)$ receives $G_{ij}/2^{a+b}$, as in `sigkernel`. For `LinearKernel` this is the inner product of the sub-increments; for a non-linear $\kappa$ it is not the mixed difference of $\kappa$ at the refined points, so refining the solver grid is not the same as refining the input path.

| Class | Parameters | $\kappa(x, y)$ |
|---|---|---|
| [`LinearKernel`][tensordev.LinearKernel] | `scale=1.0` | $\langle \mathrm{scale}\cdot x, \mathrm{scale}\cdot y\rangle$ |
| [`RBFKernel`][tensordev.RBFKernel] | `sigma=1.0` | $\exp(-\lVert x - y\rVert^2/\sigma)$ |

`LinearKernel` and `RBFKernel` act on points of $\mathbb{R}^d$. To define another kernel, subclass [`StaticKernel`][tensordev.kernel.StaticKernel] and implement `batch_kernel` and `Gram_matrix`; the solvers call them with path values of shape `batch + (length, dim)`.

`tensordev.kernel` also exports [`RBF_CEXP_Kernel`][tensordev.kernel.RBF_CEXP_Kernel] and [`RBF_SQR_Kernel`][tensordev.kernel.RBF_SQR_Kernel], ported from `sigkernel` for function-valued paths whose values have shape `(length_x, dim)`. No kernel class of this release accepts such paths, so do not pass them as `static_kernel`: `SigKernel` and `FreeKernel` return wrong values with them, and `FSSKSigKernel` raises.

Two consequences of the construction:

- `SigKernel` and `FreeKernel` rebuild the points $x_i$ from the increments, starting at the origin. $\kappa$ therefore sees every path translated to start at 0, and these kernels do not depend on the starting points. `FSSKSigKernel` applies $\kappa$ to the projected path values $A_r x_i$ as given, as in *Computational Aspects of the Volterra Signature*, so with a non-linear $\kappa$ such as `RBFKernel` its value depends on the starting points; with `increment_input=True` its points also start at the origin.
- `LinearKernel(scale=c)` gives the kernel of the paths scaled by $c$.

```python
from tensordev.kernel import LinearKernel

k_rbf = SigKernel(static_kernel=RBFKernel(sigma=1.0))
np.testing.assert_allclose(
    k_rbf.compute_Gram(X + 3.0, Y - 1.0), k_rbf.compute_Gram(X, Y), atol=1e-12
)

k_scaled = SigKernel(static_kernel=LinearKernel(scale=2.0))
np.testing.assert_allclose(
    k_scaled.compute_Gram(X, Y), SigKernel().compute_Gram(2.0 * X, 2.0 * Y), rtol=1e-12
)
```

## Free kernel

[`FreeKernel`][tensordev.FreeKernel] is the inner product of the
[free developments](signatures.md#free-developments) of two tensor-valued paths,
$k(Z, W) = \langle D(Z), D(W)\rangle$, again without truncation. An input is one array, read as a
level-1 path, or a tuple or list of levels; level $k$ has shape `batch + (S+1, d**k)`. A missing
sample axis is added. The fields are those of `SigKernel`. `static_kernel` only applies when both
inputs have a single level.

For a level-1 path, `FreeKernel` and `SigKernel` agree. For tensor-valued paths, the PDE couples
the scalar kernel $u$ with tensor-valued components $f$ and $g$. This is the system of
P. K. Friz and P. P. Hager, [*Expected Signature Kernels for Rough Paths*](https://arxiv.org/abs/2509.07893):
`FreeKernel` solves its truncated form, Theorem 3.8 there, in which $u$ is written $w$, $g$ is
written $\widetilde f$, and $D$ is written $\mathcal S$. For piecewise log-linear paths it is the
system of Lemercier, Lyons and Salvi (see [Higher-order kernel](#higher-order-kernel)); with the
integrated characteristic velocities of two continuous inhomogeneous Lévy processes as inputs,
$u$ approximates their expected signature kernel. For tensor-valued paths the solver advances
$(u, f, g)$ cell by cell with a predictor–corrector step adapted from `high-order-sigkernel`.
The check below compares the free kernel with the inner product of free developments truncated
at level 12:

```python
np.testing.assert_allclose(
    FreeKernel(dyadic_order=1).compute_Gram(X, Y),
    SigKernel(dyadic_order=1).compute_Gram(X, Y),
    rtol=1e-12,
)

X2 = 0.1 * random_trigonometric_polynomial_paths(batch=4, steps=32, dim=4, key=2)
Y2 = 0.1 * random_trigonometric_polynomial_paths(batch=4, steps=32, dim=4, key=3)

gram_tensor = FreeKernel(dyadic_order=2).compute_Gram((X, X2), (Y, Y2))

dev_x = td.free_development((X, X2), trunc=12)
dev_y = td.free_development((Y, Y2), trunc=12)
reference = td.tensor_inner_product(
    tuple(level[:, None] for level in dev_x),  # pair every x_i
    tuple(level[None, :] for level in dev_y),  # with every y_j
)
np.testing.assert_allclose(gram_tensor, reference, rtol=1e-3)
```

The function [`free_kernel`][tensordev.free_kernel] is the solver behind `FreeKernel` and
`SigKernel`. It takes a tuple or list of levels, or a pair `(velocity, grid)` of a velocity
callable and a grid, integrated with the rule that `quadrature` selects (`"left"`, `"midpoint"`
or `"trapezoid"`; default `"left"`). Pass `(X,)`, not `X`: a bare array is read as a sequence of
levels along its first axis. It does not add a sample axis or split the batch.
`pairwise=True` gives the Gram layout, `evaluate="grid"` the whole grid, and `return_fg=True`
also returns $f$ and $g$. Its increment flag is called `increment_in`.

```python
from tensordev.kernel import free_kernel

gram_fn = free_kernel((X,), (Y,), pairwise=True, dyadic_order=1)
np.testing.assert_allclose(gram_fn, gram_free, rtol=1e-12)  # gram_free from the README example
```

## Higher-order kernel

[`HigherOrderKernel`][tensordev.HigherOrderKernel] replaces each path by a
[piecewise log-linear path](signatures.md#piecewise-log-linear-paths) and evaluates the free
kernel of the two approximations. For each input separately, it splits the intervals into
blocks of `log_steps` steps, computes the block signatures truncated at `log_degree`, and takes
their tensor logarithms as the increments of the new path. The PDE grid then has one row per
block of `X` and one column per block of `Y`, and each cell carries tensor-valued data. For the log-PDE method, see M. Lemercier,
T. Lyons and C. Salvi,
[*Log-PDE Methods for Rough Signature Kernels*](https://arxiv.org/abs/2404.02926), SIAM Journal
on Numerical Analysis, 2026.

| Field | Default | Description |
|---|---|---|
| `log_steps` | required | `(log_steps_x, log_steps_y)`, the block sizes for `X` and `Y`. Each must divide the number of intervals of its input. |
| `log_degree` | required | `(log_degree_x, log_degree_y)`, the truncation of the block signatures, at least 1. |
| `backend`, `dyadic_order`, `increment_input`, `num_devices` | as `SigKernel` | Passed on to the free-kernel solver. |

Inputs are level-1 paths of shape `(batch, length, dim)`. A block size that does not divide the
number of intervals raises a `ValueError`. With one step per block and degree 1, every log-linear
piece is the original linear piece, and the kernel is the signature kernel:

```python
k_steps = HigherOrderKernel(log_steps=(1, 1), log_degree=(1, 1))
np.testing.assert_allclose(
    k_steps.compute_Gram(X, Y), SigKernel().compute_Gram(X, Y), rtol=1e-12
)
```

The function [`higher_order_kernel`][tensordev.higher_order_kernel] takes the fields of
`HigherOrderKernel` as keywords except `num_devices` (`log_steps` and `log_degree` are required),
together with `evaluate`, `return_fg` and `pairwise`. Its increment flag is `increment_input`,
unlike `increment_in` of `free_kernel` and `fssk_sigkernel`. It accepts an array or a tuple of
levels:

```python
from tensordev.kernel import higher_order_kernel

gram_ho = higher_order_kernel(X, Y, log_steps=(2, 2), log_degree=(3, 3), pairwise=True)
print(gram_ho.shape)  # (4, 4)
```

## FSSK kernel

[`FSSKSigKernel`][tensordev.FSSKSigKernel] is the kernel induced by the Volterra signature of a
finite state-space kernel ([State-space signatures](state-space.md)). Like the other kernels, it
solves a PDE on the grid of path points: the matrix Goursat system of P. P. Hager, F. N. Harang,
L. Pelizzari and S. Tindel, *Computational Aspects of the Volterra Signature* (Part II below),
section on the Volterra signature kernel.

| Field | Default | Description |
|---|---|---|
| `kernel` | required | The [`FSSK`][tensordev.FSSK], for example `sss.kernel` of a `StateSpaceSignature`. |
| `dt_x`, `dt_y` | required | Time steps of `X` and `Y`: a scalar for a uniform grid, or a 1-D array with one width per interval. |
| `scheme` | `"heun"` | `"heun"` is the predictor–corrector scheme of Part II, second order. `"etd1"` is the first-order scheme with exponential integration of the auxiliary equations that Part II uses for comparison. |
| `precompute_propagators` | `True` | Precompute the matrix exponentials and $\varphi$-functions once per step size and reuse them in every cell: faster for uniform steps, more memory. |
| `static_kernel` | `LinearKernel(scale=1.0)` | Kernel $\kappa$ applied to the projected path values $A_r x$, $r = 1, \dots, q$. The cell coefficient is $\gamma_{ij} = b^\top G_{ij} b$, where $G_{ij}^{r,p}$ is the mixed difference of $\kappa(A_r x, A_p y)$ over cell $(i, j)$, as in Part II. |
| `backend`, `dyadic_order`, `increment_input`, `num_devices` | as `SigKernel` | |

The path dimension must equal `kernel.path_dim`; otherwise the input check raises a
`ValueError`. The check below compares the FSSK kernel with the inner product of Volterra
signatures truncated at level 8:

```python
deep = StateSpaceSignature(kernel=sss.kernel, trunc=8)
vsig_x = deep.vsig(X, dt=1.0 / 32)
vsig_y = deep.vsig(Y, dt=1.0 / 32)
reference = td.tensor_inner_product(
    tuple(level[:, None] for level in vsig_x), tuple(level[None, :] for level in vsig_y)
)

k_fssk_2 = FSSKSigKernel(kernel=sss.kernel, dt_x=1.0 / 32, dt_y=1.0 / 32, dyadic_order=2)
np.testing.assert_allclose(k_fssk_2.compute_Gram(X, Y), reference, rtol=1e-4)
```

With $\Lambda = 0$, $A = I$ and $b = 1$, the kernel $K(t, s)$ is the identity, and the FSSK kernel
is the classical signature kernel:

```python
from tensordev import FSSK

flat = FSSK.from_matrix(Lambda=jnp.zeros((1, 1)), A=jnp.eye(2)[None], b=jnp.ones((1, 1)))
k_flat = FSSKSigKernel(kernel=flat, dt_x=1.0 / 32, dt_y=1.0 / 32, dyadic_order=3)
np.testing.assert_allclose(
    k_flat.compute_Gram(X, Y), SigKernel(dyadic_order=3).compute_Gram(X, Y), rtol=1e-5
)
```

The function [`fssk_sigkernel`][tensordev.fssk_sigkernel] takes the same arguments as keywords,
with `increment_in` for the increment flag. Its `precompute_propagators` defaults to `False`.
With `return_fg=True` it returns the tuple `(η, K, A, B)` of the matrix Goursat system of
Part II: `K` is the $R \times R$ matrix there, with
$\eta = 1 + \mathbf 1^\top K \mathbf 1$, and `A` and `B` are the auxiliary matrices written
$\Psi$ and $\Phi$ there, transported along $s$ and $t$. They are not the matrices $A_r$ of the
FSSK.

```python
from tensordev.kernel import fssk_sigkernel

gram_fssk = fssk_sigkernel(X, Y, kernel=sss.kernel, dt_x=1.0 / 32, dt_y=1.0 / 32, pairwise=True)
np.testing.assert_allclose(gram_fssk, k_fssk.compute_Gram(X, Y), rtol=1e-8)
```

## Gradients

With the default execution policy, kernel values are differentiable in the paths, for all four
kernel classes, and in the parameters of the static kernel. The static kernels are JAX pytrees,
so a bandwidth can be a traced value. The MMD below is differentiated with respect to the RBF
bandwidth and checked against a central difference:

```python
def mmd_of_sigma(sigma):
    return SigKernel(static_kernel=RBFKernel(sigma=sigma)).compute_mmd(X, Y)

value, slope = jax.value_and_grad(mmd_of_sigma)(1.0)
h = 1e-5
central = (mmd_of_sigma(1.0 + h) - mmd_of_sigma(1.0 - h)) / (2 * h)
np.testing.assert_allclose(slope, central, rtol=1e-6)
```

The gradient with respect to the sample `X` has the shape of `X`:

```python
grad_X = jax.grad(lambda Z: k.compute_mmd(Z, Y))(X)
print(grad_X.shape)  # (4, 33, 2)

direction = random_trigonometric_polynomial_paths(batch=4, steps=32, dim=2, key=2)
h = 1e-6
central = (k.compute_mmd(X + h * direction, Y) - k.compute_mmd(X - h * direction, Y)) / (2 * h)
np.testing.assert_allclose(jnp.vdot(grad_X, direction), central, rtol=1e-6)
```

Gradients of `FSSKSigKernel` with respect to the state-space operator $\Lambda$ work in forward
and reverse mode: the rates, decays and frequencies of an `FSSK` built with `from_jordan`, and the
matrix of one built with `from_matrix`.

## Batching

`max_batch` bounds the number of samples per kernel evaluation. The split happens in a Python
loop before any JAX work, so it limits peak memory independently of the number of devices:

- `compute_kernel` evaluates chunks of at most `max_batch` pairs.
- `compute_Gram` evaluates blocks of `max_batch` by `max_batch` pairs; with `sym=True` only the
  blocks on and above the diagonal.
- `compute_mmd` and the scoring rules use it for their Gram matrices.

The default is `100`. `None` or a value of at most 0 turns the split off. The result does not
depend on it:

```python
X8 = random_trigonometric_polynomial_paths(batch=8, steps=32, dim=2, key=5)
np.testing.assert_allclose(
    k.compute_Gram(X8, max_batch=4), k.compute_Gram(X8, max_batch=None), rtol=1e-12, atol=1e-12
)
```

`increment_input` reads the inputs as increments, `(batch, S, dim)`, instead of path values. Set
it as a field of the kernel, or override it for one call:

```python
dX, dY = jnp.diff(X, axis=1), jnp.diff(Y, axis=1)
np.testing.assert_allclose(k.compute_Gram(dX, dY, increment_input=True), gram, rtol=1e-12)
np.testing.assert_allclose(
    SigKernel(dyadic_order=1, increment_input=True).compute_Gram(dX, dY), gram, rtol=1e-12
)
```

### Ragged and continuous inputs

[`bucket_pad_ragged_paths`][tensordev.util.bucket_pad_ragged_paths] takes a list of paths of
different lengths, optionally sorts them by length, groups them into chunks of at most
`chunk_size` paths, and pads each path by repeating its last value. Padding adds zero
increments, which leave the kernel unchanged. `return_indices=True` also returns the original
position of every path in each chunk, and `target_length` pads all chunks to one fixed length
to keep batch shapes stable, for example to avoid recompilation.

```python
from tensordev.util import bucket_pad_ragged_paths

paths = [random_trigonometric_polynomial_paths(batch=1, steps=n, dim=2, key=n)[0] for n in (8, 16, 32)]

chunks, index = bucket_pad_ragged_paths(paths, chunk_size=2, return_indices=True)
print([chunk.shape for chunk in chunks], index)  # [(2, 17, 2), (1, 33, 2)] [(0, 1), (2,)]

(padded,) = bucket_pad_ragged_paths(paths)
direct = jnp.stack([k.compute_Gram(path, Y)[0] for path in paths])
np.testing.assert_allclose(k.compute_Gram(padded, Y), direct, atol=1e-12)
```

[`velocity_to_increments`][tensordev.util.velocity_to_increments] integrates a velocity
`t -> tuple of levels` over the intervals of a grid and returns one increment array per level,
ready for `increment_input=True`. With two levels, the result is a tensor-valued path for
`FreeKernel`:

```python
from tensordev.util import velocity_to_increments

def velocity(t):
    v = jnp.stack([jnp.cos(2 * jnp.pi * t), jnp.sin(2 * jnp.pi * t)], axis=-1)
    return v, 0.1 * (v[:, :, None] * v[:, None, :]).reshape(-1, 4)

circle = velocity_to_increments(velocity, jnp.linspace(0.0, 1.0, 33))
print([level.shape for level in circle])  # [(32, 2), (32, 4)]

k_circle = FreeKernel(dyadic_order=1, increment_input=True)
print(k_circle.compute_kernel(circle, circle).shape)  # (1,)
```

### Several devices

`num_devices` splits the batch across JAX devices with `jax.pmap`, after the `max_batch` split.
For the Gram matrix the rows of `X` are split and `Y` is replicated. JAX on a CPU shows one device
by default; set `XLA_FLAGS` before JAX is imported to create more:

```bash
XLA_FLAGS="--xla_force_host_platform_device_count=8" python my_script.py
```

Then pass `num_devices` to any kernel class. If it exceeds the visible device count,
`tensordev` warns and uses the devices it sees. A batch smaller than `num_devices` runs on one
device, and other batches are padded to a multiple of `num_devices`. The virtual CPU devices
divide the CPU thread pool among themselves, so more devices give more batch parallelism but
fewer threads for the matrix operations on each device. The module documentation advises more
devices when the batch is large relative to the path size, and fewer for a small batch of large
problems.

```python
k_devices = SigKernel(dyadic_order=1, num_devices=jax.device_count())
np.testing.assert_allclose(k_devices.compute_Gram(X, Y), gram, rtol=1e-12)
```

The module [`tensordev.kernel.parallel`](https://github.com/hagerpa/tensordev/blob/main/src/tensordev/kernel/parallel.py)
holds the helpers: `pmap_batch(fn, *pytrees, num_devices=...)` shards every argument along
axis 0, `cpu_device_count()` returns the number of visible JAX devices, and
`recommend_parallelism()` returns a short text with the matching `XLA_FLAGS` setting.
