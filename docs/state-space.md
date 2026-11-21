# State-space signatures

`tensordev.sss` computes Volterra signatures for finite state-space kernels (FSSK) by a
recursion on a hidden state. The state makes streaming possible: feed increments as they
arrive and read out the signature at any time. This page covers the kernels and their
constructors, the [`StateSpaceSignature`][tensordev.StateSpaceSignature] class, the functional
forms and streaming. The examples share one batch of paths and use 64-bit floats:

```python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import tensordev as td
from tensordev.util import random_trigonometric_polynomial_paths

X = random_trigonometric_polynomial_paths(batch=4, steps=32, dim=2, key=0)
dt = 1.0 / 32
```

## Finite state-space kernels

--8<-- "README.md:sss_overview"

Here $\Lambda\in\mathbb{R}^{R\times R}$ is the state-space operator, $b_r\in\mathbb{R}^R$ are
the state vectors, $A_r\in\mathbb{R}^{m\times d}$ are the kernel matrices, $R$ is the state
dimension and $\mathbf{1}\in\mathbb{R}^R$ is the vector of ones. An FSSK is a convolution
kernel as on [Volterra signatures](volterra.md#convolution-kernels), with the smooth scalar
kernels $k_r(t)=\mathbf{1}^\top e^{-\Lambda t}b_r$ (with `beta` fixed to ones). In real Jordan form,
$\Lambda$ produces sums of exponentials and damped oscillations, multiplied by powers of $t$
for blocks larger than one.

The matrix exponential factorizes: for $u\le t\le\tau$,
$e^{-\Lambda(\tau-u)}=e^{-\Lambda(t-u)}\,e^{-\Lambda(\tau-t)}$. Everything the path up to time
$t$ contributes at a later readout time $\tau$ therefore passes through a hidden state
$Z^r_t$, one tensor over $\mathbb{R}^m$ with $R$ state coordinates for each component $r$. The
Volterra signature is the linear readout

$$
\mathrm{VSig}(x;K)^{\tau}_{0,t}=1+\sum_{r=1}^{q}Z^{r}_{t}\,e^{-\Lambda(\tau-t)}\,b_r .
$$

The recursion advances $Z$ over each step of a piecewise-linear path. There is no time
discretization of the history, unlike the generic schemes of [`vsig`][tensordev.vsig] (see
[Comparison with vsig](#comparison-with-vsig)). The per-step coefficients are contour
integrals, evaluated by the trapezoidal rule on the parabolic contour of *Computational Aspects
of the Volterra Signature*. `quad_order` (default 32) is the number of nodes in the upper
half-plane ($m$ in that paper's algorithm EvalPhiPsi), so the rule uses $2\cdot$`quad_order`
nodes in all. It is accurate only while the eigenvalues of $-\Lambda\,\mathrm{d}t$ lie well
inside the contour. With the default `quad_order`, decaying modes of any stiffness are exact to
rounding; weakly damped oscillations need $\omega\,\mathrm{d}t\lesssim 4$ and growing modes
(a negative real rate $\lambda$) need $|\lambda|\,\mathrm{d}t\lesssim 1$. These bounds scale with
`quad_order`.

[`fssk_state`][tensordev.fssk_state], [`fssk_vsig`][tensordev.fssk_vsig] and
[`StateSpaceSignature.vsig`][tensordev.StateSpaceSignature.vsig] split longer steps into equal
sub-steps. Inserting nodes on the segments does not change a piecewise-linear path, so the result
is exact; only the cost grows with the number of sub-steps. The split needs concrete values of
$\Lambda$ and `dt`. When they are traced, for example under `jax.jit` or `jax.grad` with respect to
$\Lambda$, steps are not split: oscillations up to $\omega\,\mathrm{d}t\approx 12$ and growing
modes up to $|\lambda|\,\mathrm{d}t\approx 5$ then lose up to about five digits, and beyond that
the coefficients of a `JordanLambda` are NaN instead of silently wrong. Refine the grid yourself in
that case. Raising `quad_order` widens the contour but costs accuracy to rounding, because the
weights grow like `exp(0.26 * quad_order)` (about $10^{-10}$ at 64 and $10^{-3}$ at 128).

## Constructing a kernel

### State-space operators

An FSSK holds its operator $\Lambda$ as a [`Lambda`][tensordev.sss.Lambda] object, which
provides $e^{-\Lambda\,\mathrm{d}t}$ (`expm`), shifted solves and the other operations the
recursion needs. Two realizations exist:

- [`DenseLambda`][tensordev.sss.DenseLambda] takes any dense `(R, R)` matrix. It caches an
  eigendecomposition at construction for the operator actions and falls back to Padé-based
  matrix functions and direct solves when the eigenvector matrix is ill-conditioned.
- [`JordanLambda`][tensordev.sss.JordanLambda] is block diagonal in real Jordan form. A real
  block with rate $\lambda$ and size $n$ is $\lambda I_n-N_n$, with $N_n$ the ones on the
  superdiagonal. An oscillatory pair with decay $a$, frequency $\omega>0$ and size $n$ is a
  $2n\times 2n$ block with
  $\bigl(\begin{smallmatrix}a&-\omega\\ \omega&a\end{smallmatrix}\bigr)$ on the diagonal
  and $-I_2$ on the block superdiagonal; for $n=1$,
  $e^{-\Lambda t}=e^{-at}\bigl(\begin{smallmatrix}\cos\omega t&\sin\omega t\\ -\sin\omega t&\cos\omega t\end{smallmatrix}\bigr)$.
  *Computational Aspects of the Volterra Signature* prints the transposed $2\times2$ block
  ($-\Lambda=J_n(-a,\omega)$ with
  $C(a,\omega)=\bigl(\begin{smallmatrix}a&-\omega\\ \omega&a\end{smallmatrix}\bigr)$). Both have
  eigenvalues $a\pm i\omega$, but only the orientation used here turns the vector $b$ of that
  paper's Prony lemma, which `from_prony` builds, into $\beta\cos\omega t+\delta\sin\omega t$;
  the printed one gives $\beta\cos\omega t-\delta\sin\omega t$. Real blocks come first, then
  oscillatory pairs, so $R$ is `sum(real_sizes) + 2 * sum(osc_sizes)`. Rates, decays and
  frequencies are JAX arrays, so new values do not retrace; block sizes are static.

```python
from tensordev.sss import DenseLambda, JordanLambda

lam = JordanLambda(
    real_rates=jnp.array([1.0]),
    real_sizes=(1,),
    osc_decays=jnp.array([0.5]),
    osc_freqs=jnp.array([2 * jnp.pi]),
    osc_sizes=(1,),
)
print(lam.state_dim)  # 3
print(np.round(lam.matrix(), 3))
# [[ 1.     0.     0.   ]
#  [ 0.     0.5   -6.283]
#  [ 0.     6.283  0.5  ]]

t = jnp.array([0.1, 0.5, 1.0])
np.testing.assert_allclose(DenseLambda(lam.matrix()).expm(t), lam.expm(t), atol=1e-12)
```

### Constructors

All constructors take `A` of shape `(q, m, d)` and `quad_order` (default 32), the number of
contour quadrature nodes in the upper half-plane for the per-step coefficients (see
[Finite state-space kernels](#finite-state-space-kernels)). They differ in how $\Lambda$ and $b$
are given:

| Constructor | $\Lambda$ | `b` |
|---|---|---|
| [`FSSK.from_matrix`][tensordev.FSSK.from_matrix] | `Lambda`: dense `(R, R)` array, stored as a `DenseLambda` | `(q, R)` |
| [`FSSK.from_jordan`][tensordev.FSSK.from_jordan] | `real_rates`, `real_sizes`, `osc_decays`, `osc_freqs`, `osc_sizes`: a `JordanLambda` | `(q, R)`, in the Jordan basis of those blocks |
| [`FSSK.from_prony`][tensordev.FSSK.from_prony] | the same Jordan data | built from Prony coefficients `alpha`, `beta`, `delta` |

[`FSSK`][tensordev.FSSK]`(Lambda=..., A=..., b=...)` takes any `Lambda` object directly. A
kernel exposes `q`, `m`, `path_dim` ($d$) and `state_dim` ($R$).
[`StateSpaceSignature.from_matrix`][tensordev.StateSpaceSignature.from_matrix],
[`from_jordan`][tensordev.StateSpaceSignature.from_jordan] and
[`from_prony`][tensordev.StateSpaceSignature.from_prony] take `trunc` (required), `state`,
`core` and `seq_core`, and forward all other keyword arguments to the matching `FSSK`
constructor. `StateSpaceSignature(kernel, trunc)` wraps an existing kernel.

### Prony coefficients

`from_prony` takes one row of coefficients per component $r$: `alpha` of shape
`(q, sum(real_sizes))` for the real blocks, and `beta`, `delta` of shape
`(q, sum(osc_sizes))` for the oscillatory pairs. It builds the vectors $b_r$ such that
$k_r(t)=\mathbf{1}^\top e^{-\Lambda t}b_r$ has the Prony form of *Computational Aspects of the
Volterra Signature*:

$$
k_r(t)=\sum_{i=1}^{Q}\sum_{\ell=1}^{m_i}e^{-\lambda_i t}\frac{t^{\ell-1}}{(\ell-1)!}\,\alpha_{i,\ell}+\sum_{i=1}^{P}\sum_{\ell=1}^{n_i}e^{-a_i t}\frac{t^{\ell-1}}{(\ell-1)!}\bigl(\beta_{i,\ell}\cos\omega_i t+\delta_{i,\ell}\sin\omega_i t\bigr).
$$

Here the $Q$ real blocks have rates $\lambda_i$ (`real_rates`) and sizes $m_i$ (`real_sizes`), and
the $P$ oscillatory pairs have decays $a_i$ (`osc_decays`), frequencies $\omega_i$ (`osc_freqs`)
and sizes $n_i$ (`osc_sizes`). The coefficients $\alpha_{i,1},\dots,\alpha_{i,m_i}$ are, in
order, the entries of row $r$ of `alpha` in the slice of block $i$, and likewise for
$\beta_{i,\ell}$ and $\delta_{i,\ell}$ in `beta` and `delta`; the first entry of a slice is the
constant term. The paper indexes the blocks by $r$; here $r$ is the component. A check against
$\mathbf{1}^\top e^{-\Lambda t}b$:

```python
from tensordev.sss import FSSK

prony = FSSK.from_prony(
    A=jnp.eye(2)[None],
    real_rates=jnp.array([1.0]),
    real_sizes=(2,),
    osc_decays=jnp.array([0.5]),
    osc_freqs=jnp.array([3.0]),
    osc_sizes=(1,),
    alpha=jnp.array([[1.0, -0.5]]),
    beta=jnp.array([[0.3]]),
    delta=jnp.array([[0.7]]),
)
print(prony.q, prony.m, prony.path_dim, prony.state_dim)  # 1 2 2 4

t = jnp.linspace(0.0, 2.0, 9)
k = prony.Lambda.expm(t).sum(axis=-2) @ prony.b[0]  # 1ᵀ exp(-Λt) b
expected = jnp.exp(-t) * (1.0 - 0.5 * t) + jnp.exp(-0.5 * t) * (
    0.3 * jnp.cos(3.0 * t) + 0.7 * jnp.sin(3.0 * t)
)
np.testing.assert_allclose(k, expected, atol=1e-12)
```

### Per-step coefficients

[`FSSK.coef`][tensordev.FSSK.coef]`(dt, trunc=N)` returns the
[`FSSKCoefficients`][tensordev.sss.FSSKCoefficients] for a batch of step sizes, with `dt.shape`
as leading axes: the propagators `E` $=e^{-\Lambda\,\mathrm{d}t}$ and the normalized
coefficients of algorithm EvalPhiPsi in *Computational Aspects of the Volterra Signature*,

$$
\texttt{psi}[i]=\mathrm{d}t^{-|\ell|-1}\,\mathbf{1}^\top\Psi^{w(\ell)}(\mathrm{d}t)\quad(|\ell|\le N-1),\qquad
\texttt{phi}[r-1,i]=\mathrm{d}t^{-|\ell|-1}\,\Phi^{r\,w(\ell)}(\mathrm{d}t)\quad(|\ell|\le N-2),
$$

for $r=1,\dots,q$. Here $\ell\in\mathbb{N}^q$ is the multi-index `coef.layout.ell[i]`,
$w(\ell)=1^{\ell_1}\cdots q^{\ell_q}$ is the word with $\ell_p$ letters $p$,
$\Phi^{\varnothing}(\delta)=e^{-\Lambda\delta}$,
$\Phi^{p_1\cdots p_n}(\delta)=\int_0^\delta\Phi^{p_1\cdots p_{n-1}}(u)\,b_{p_n}\mathbf{1}^\top e^{-\Lambda(\delta-u)}\,\mathrm{d}u$
and $\Psi^{w}(\delta)=\int_0^\delta\Phi^{w}(u)\,\mathrm{d}u$. The recursion builds them
internally; call `coef` to inspect them.

```python
coef = prony.coef(jnp.array([dt, 2 * dt]), trunc=3)
print(coef.E.shape, coef.psi.shape, coef.phi.shape)  # (2, 4, 4) (2, 3, 4) (2, 1, 2, 4, 4)
np.testing.assert_allclose(coef.E, prony.Lambda.expm(jnp.array([dt, 2 * dt])), atol=1e-12)
```

## Volterra signatures by state recursion

--8<-- "README.md:sss_example"

--8<-- "README.md:sss_gpu"

In this example $k(t)=e^{-t}+2e^{-t/2}\cos(2\pi t)$, $R=3$ and the output has the shapes of a
truncated signature over $\mathbb{R}^2$. A
[`StateSpaceSignature`][tensordev.StateSpaceSignature] binds the kernel, the truncation, the
cores and a hidden `state` (zero by default). It is an immutable JAX PyTree, so it can be
passed through `jax.jit`; methods that change the state return a new object.

| Method | Returns | Stored `state` |
|---|---|---|
| [`vsig`][tensordev.StateSpaceSignature.vsig]`(X, *, dt, ...)` | Volterra signature at the terminal time, or at each block end | seeds the recursion, unchanged |
| [`states`][tensordev.StateSpaceSignature.states]`(X, *, dt, ...)` | hidden-state trajectory, seed included by default | seeds the recursion, unchanged |
| [`update_with_path`][tensordev.StateSpaceSignature.update_with_path]`(X, *, dt, ...)` | new instance holding the terminal state | seeds the recursion |
| [`update_with_increment`][tensordev.StateSpaceSignature.update_with_increment]`(dx, *, dt, ...)` | new instance advanced by one increment `dx` of shape `(..., d)` | seeds the recursion |
| [`readout`][tensordev.StateSpaceSignature.readout]`(*, tau_dt=0.0)` | Volterra signature from the stored state | read |
| [`reset`][tensordev.StateSpaceSignature.reset]`(new_state=None)` | copy with a zero state, or with `new_state` | replaced |

`vsig` and `states` take the following keyword arguments; `update_with_path` takes `dt`,
`axis`, `increment_input` and `execution`:

| Argument | Default | Description |
|---|---|---|
| `dt` | required | Step size: a scalar, an `(S,)` array, or an array matching the batch and step axes of `X` without the trailing axis. |
| `axis` | `-2` | Step axis of `X`; the trailing axis has size `kernel.path_dim`. |
| `block_size` | `None` | Steps per emitted block; `None` covers the full sequence. |
| `accumulate` | `True` | Carry the hidden state across blocks. |
| `initial_state` | `None` | Seed state; defaults to the stored `state`. |
| `output_starting_state` | `False` for `vsig`, `True` for `states` | Prepend the readout of the seed (`vsig`) or the seed (`states`). |
| `tau_dt` | `0.0` | `vsig` only: readout lag $\tau-t\ge 0$; broadcasts against the batch axes. |
| `increment_input` | `False` | `X` holds increments; skip differencing. |
| `execution` | `"auto"` | `"auto"`, `"jax"` or `"wordwise"`. |

With `block_size=1` and `output_starting_state=True`, `vsig` returns the signature at every
node. `tau_dt` sets the readout time $\tau$ to $t$ plus `tau_dt`, for example one lag per path:

```python
trajectory = sss.vsig(X, dt=dt, block_size=1, output_starting_state=True)
print(trajectory[1].shape)  # (4, 33, 2): one value per node, starting with the unit

lagged = sss.vsig(X, dt=dt, tau_dt=jnp.array([0.0, 0.1, 0.2, 0.3]))
np.testing.assert_allclose(lagged[2][0], result[2][0], atol=1e-12)  # lag 0 for path 0
```

For `q = 1`, `execution="wordwise"` requests the alpha GPU implementation; the default is
unchanged (see [Wordwise GPU execution](performance.md#wordwise-gpu-execution)).

`trunc` is a positive integer for a total-degree core or a bidegree $(N, M)$ for a bidegree
core, whose `dims` must sum to $m$. Kernels with `q = 1` accept total-degree and bidegree
cores in standard or shear coordinates, including partially symmetrized ones; `q > 1` requires
the standard total-degree core, and other cores raise `RuntimeError`. A bidegree result equals
the total-degree result restricted to the kept words:

```python
bicore = td.make_core(dims=(1, 1), max_trunc=(2, 1))
rectangular = StateSpaceSignature(sss.kernel, trunc=(2, 1), core=bicore).vsig(X, dt=dt)
expected = bicore.tensor_from_total(result, trunc=(2, 1))
for grade in rectangular.grades:
    np.testing.assert_allclose(rectangular[grade], expected[grade], atol=1e-12)
```

### Functional forms

[`fssk_vsig`][tensordev.fssk_vsig] and [`fssk_state`][tensordev.fssk_state] take the kernel,
`trunc`, `core` and `seq_core` as arguments and otherwise the same keywords as the methods.
`trunc` defaults to the core's default truncation and `initial_state` to zero. Unlike the method
`states`, `fssk_state` does not prepend the seed by default (`output_starting_state=False`). The state is in
first-on format: one array per degree $n=1,\dots,N$, without degree zero, of shape
`batch + (q, 1, R, m**n)` for a total-degree core.

```python
functional = td.fssk_vsig(X, kernel=sss.kernel, dt=dt, trunc=3)
np.testing.assert_allclose(td.tensor_to_flat(functional), td.tensor_to_flat(result), atol=1e-12)

state = td.fssk_state(X, kernel=sss.kernel, dt=dt, trunc=3)
print([level.shape for level in state])  # [(4, 1, 1, 3, 2), (4, 1, 1, 3, 4), (4, 1, 1, 3, 8)]
```

### Comparison with vsig

`ConvolutionKernel.fssk` passes the same kernel to the generic [`vsig`][tensordev.vsig].
Its schemes approximate the history inside each step, so they agree with the state recursion
up to a discretization error that shrinks with `order` and `dyadic_order`. The quadratic and
FFT schemes give the same values here; the block uses the quadratic one:

```python
generic_kernel = td.ConvolutionKernel.fssk(sss.kernel)
for order, dyadic_order in ((0, 0), (2, 0), (2, 1), (2, 2)):
    generic = td.vsig(
        X, kernel=generic_kernel, dt=dt, trunc=3,
        scheme="quadratic", order=order, dyadic_order=dyadic_order,
    )
    gap = max(float(jnp.max(jnp.abs(g - r))) for g, r in zip(generic, result))
    print(order, dyadic_order, f"{gap:.1e}")
# 0 0 1.2e+00
# 2 0 1.2e-05
# 2 1 7.7e-07
# 2 2 4.8e-08

for g, r in zip(generic, result):
    np.testing.assert_allclose(g, r, atol=1e-6, rtol=1e-6)
```

The default `order=0` leaves a large gap on this grid. For FSSK kernels the state recursion
avoids this error; the generic schemes serve kernels without a finite state, such as the
fractional and gamma kernels.

## Streaming evaluation

--8<-- "README.md:sss_streaming"

`update_with_path` returns a new instance and leaves `sss` unchanged. `vsig` and `states` on
an updated instance continue from its state, and `readout(tau_dt=...)` evaluates the stored
state at a later readout time without new increments:

```python
continued = sss_mid.vsig(X[:, 16:], dt=dt)
np.testing.assert_allclose(td.tensor_to_flat(continued), td.tensor_to_flat(vsig_end), atol=1e-12)

np.testing.assert_allclose(
    td.tensor_to_flat(sss_end.readout(tau_dt=0.25)),
    td.tensor_to_flat(sss.vsig(X, dt=dt, tau_dt=0.25)),
    atol=1e-12,
)
```

For increments that arrive one at a time, use `update_with_increment`. A jitted step compiles
once per state shape:

```python
step = jax.jit(lambda model, dx: model.update_with_increment(dx, dt=dt))

online = sss.reset()
for k in range(X.shape[-2] - 1):
    online = step(online, X[:, k + 1] - X[:, k])

np.testing.assert_allclose(
    td.tensor_to_flat(online.readout()), td.tensor_to_flat(vsig_end), atol=1e-12
)
print([level.shape for level in online.state])
# [(4, 1, 1, 3, 2), (4, 1, 1, 3, 4), (4, 1, 1, 3, 8)]
```

`reset()` returns a copy with a zero state; `reset(saved)` restores a stored state, for
example `sss.reset(sss_mid.state)`. The recursion is differentiable in forward and reverse mode with
respect to the path, `A`, `b` and $\Lambda$: the rates, decays and frequencies of a
`JordanLambda`, and the matrix of a `DenseLambda`.
