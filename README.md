# tensordev

JAX-based tensor algebra library for signatures, free developments, Volterra signatures and inner product-kernels thereof.

## Status

`tensordev` provides tensor algebra, signature development, state-space kernels,
Volterra signatures, and free and higher-order signature kernels.

JAX is the supported backend. PyTorch, TensorFlow, and Numba backends are not
available.

The implemented JAX components are end-to-end differentiable — from elementary tensor operations and path signatures through to signature-kernel evaluations and Volterra kernel parameters.

## Requirements

`tensordev` requires Python 3.11+ and JAX 0.10.0–0.11.x.

The package is developed and tested primarily with the JAX backend.

## Installation

```bash
pip install tensordev
```

For the latest development version:

```bash
pip install git+https://github.com/hagerpa/tensordev.git
```

## License

`tensordev` is released under the Apache License 2.0.
See the [license](https://github.com/hagerpa/tensordev/blob/main/LICENSE) for
details.

## Quick start

```python
import jax
jax.config.update("jax_enable_x64", True)

import tensordev as td
from tensordev.util import random_trigonometric_polynomial_paths

X = random_trigonometric_polynomial_paths(batch=4, steps=32, dim=2, key=0)

sig = td.path_signature(X, trunc=4)
ip = td.tensor_inner_product(sig, sig)

print(td.tensor_to_flat(sig).shape)
print(ip.shape)
```

### Device placement

With a CUDA-enabled JAX installation, device placement follows JAX. To compute
only a signature on a GPU, move the path and leave the rest of the application
on the CPU:

```python
gpu = jax.devices("gpu")[0]
X_gpu = jax.device_put(X, gpu)

sig_gpu = td.path_signature(X_gpu, trunc=4)
```

Public calls use the portable JAX implementation on the selected device.
Specialized wordwise executors for ordinary and exact scalar-FSSK signatures
are included, but automatic selection remains closed until exact NVIDIA
targets and profitable workload regions pass the bundled real-GPU benchmark.
No device or execution-method option is part of the core API.

## Package structure

```text
tensordev/
├── core/           # Tensor algebra backends
├── development/    # Signature development, free and classical
├── sss/            # State-space signatures, aka Volterra signatures for finite state-space kernels
├── volterra/       # Volterra signatures: fractional, gamma and FSSK kernels
├── kernel/         # Signature kernels: classical, free, FSSK, higher-order
└── util/           # Path generators and combinatorics
```

### `tensordev.core` — tensor algebra operations

A core provides the tensor-algebra operations exposed on the `tensordev`
module. The default JAX core is dimension-free and accepts any total-degree
truncation. The same module-level operations work with bounded total-degree,
bidegree, partially symmetrized, and shear cores, including products, series,
inner products, and coordinate conversions.

Use `make_core` to construct a core without changing the background default.
`set_default_core` accepts the same construction arguments and installs the
result, or it can install an already constructed core:

| `dims` | `max_trunc` | `partially_symmetrized` | `coordinates` | core |
|---|---|---|---|---|
| integer | integer | `False` | `"standard"` | dense total degree |
| pair | pair | `False` / `True` | `"standard"` | bidegree |
| pair | integer | `False` | `"shear"` | dense total-degree shear |
| pair | pair | `False` / `True` | `"shear"` | bidegree shear |

`max_trunc` fixes the capacity, while `default_trunc` may select a smaller
degree or bidegree rectangle. Module-level operations use the installed core;
pass `core=...` to a development to override it locally.

```python
import jax.numpy as jnp
import tensordev as td

core = td.set_default_core(
    dims=(2, 1),
    max_trunc=(4, 3),
    default_trunc=(2, 2),
    partially_symmetrized=True,
    coordinates="shear",
    precompute_shuffle="generator",
)
X_split = jnp.zeros((4, 33, 3))
bisig = td.path_signature(X_split)
level_12 = bisig[1, 2]                 # bidegree (1, 2)
low_bidegrees = bisig[:2, :2]          # structural truncation (1, 1)
first_two = td.tensor_slice(bisig)[:2]  # first two paths in every block
td.reset_default_core()
```

A `BigradedTensor` stores one unpadded array per bidegree. `A[n, m]` selects a
block and `A[:N, :M]` selects an upper-exclusive bidegree prefix. Use
`td.tensor_slice(A)[key]` instead to apply `key` to the batch or time axes of
every block. The core methods `tensor_from_total` and `tensor_to_total`
convert ordered tensors between bidegree and dense total layouts;
`tensor_to_ordered` expands partially symmetrized words into ordered words. It
is the adjoint of partial symmetrization, not an inverse reconstruction.

Set `precompute_shuffle` to `False`, `"generator"` (the first-level shuffles
needed by multi-component Volterra methods), or `True` (all shuffles within
capacity). Use `td.core_expected_memory(...)` before constructing large
capacities; it reports retained plan memory in MiB by default.

`shear_core(core, ...)` and `symmetrize_core(core, ...)` derive a matching core
from an existing one and inherit its compatible configuration and plans.

Use `tensor_shear_pairing` to pair words in shear coordinates with a tensor in
standard coordinates. Partially symmetrized cores accept either an ordered or
a partially symmetrized standard tensor; the compact case is contracted
directly without expansion:

```python
import jax.numpy as jnp
import tensordev as td

X3 = jnp.array([[0., 0., 0.], [0.2, -0.1, 0.3]])
standard = td.make_core(dims=3, max_trunc=2)
standard_sig = td.path_signature(X3, core=standard)
shear = td.shear_core(standard, dims=(2, 1), precompute_shuffle=True)

alpha = (jnp.zeros((1,)), jnp.array([1., 0., 0.]), jnp.zeros((9,)))
beta = (jnp.zeros((1,)), jnp.array([0., 0., 1.]), jnp.zeros((9,)))
words = shear.tensor_shuffle_product(alpha, beta, trunc=2)
values = shear.tensor_shear_pairing(words, standard_sig)
```

`tensor_inner_product` is the Euclidean pairing for tensors in the same
coordinates. When tuple operands omit the scalar level,
`tensor_shear_pairing` accepts `words_first_on=True` or
`standard_first_on=True` as appropriate. For homogeneous partially
symmetrized standard blocks, pass
`standard_partially_symmetrized=True` explicitly.

Set defaults before tracing JAX functions. Signatures, free developments, and
Volterra signatures support configured cores. `reset_default_core()` restores
the environment-selected default.

The repository includes an optional
[native CPU companion](https://github.com/hagerpa/tensordev/tree/main/native)
for sufficiently large CPU Horner steps with standard-coordinate, partially
symmetrized bidegree cores and `dims=(1, q)`, `q > 1`. It is not included in
the TensorDev wheel and is not separately published for 0.1.0. After cloning
the repository, install it from source with `python -m pip install ./native`.

The construction is described in Hager and Pelizzari,
[*Expected signatures via partial integration, coordinate change and
symmetrization*](https://arxiv.org/abs/2607.29534).  Precomputation and storage
details are collected in the
[*technical implementation note*](https://github.com/hagerpa/tensordev/tree/main/academia/bidegree).

### `tensordev.development` — signature development

Compute truncated signatures with optional blocking. `block_size` splits the path into chunks and chains them via Chen's identity internally. This is useful for long sequences where the full path does not fit in memory at once.

```python
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import tensordev as td
from tensordev.util import random_trigonometric_polynomial_paths

X = random_trigonometric_polynomial_paths(batch=4, steps=32, dim=2, key=0)

sig = td.path_signature(X, trunc=4)  # one shot

# signature on two consecutive intervals
sig_blocked = td.path_signature(X, trunc=4, block_size=16, accumulate=False)

# Chen composition via tensor_product
sig_a = td.tensor_slice(sig_blocked)[:, 0]
sig_b = td.tensor_slice(sig_blocked)[:, 1]
sig_c = td.tensor_product(sig_a, sig_b, trunc=4)

np.testing.assert_allclose(
    td.tensor_to_flat(sig),
    td.tensor_to_flat(sig_c),
    rtol=1e-5,
    atol=1e-6,
)  # ✓

# shuffle identity via tensor_shuffle_product and tensor_inner_product
# a, b are fixed basis vectors — broadcast over the batch dimension
e1 = td.tensor_densify((None, jnp.array([1., 0.])))
e2 = td.tensor_densify((None, jnp.array([0., 1.])))
core = td.make_core(
    dims=2,
    max_trunc=4,
    precompute_shuffle=True,
)

np.testing.assert_allclose(
    td.tensor_inner_product(sig, core.tensor_shuffle_product(e1, e2, trunc=4)),
    td.tensor_inner_product(sig, e1) * td.tensor_inner_product(sig, e2),
    rtol=1e-5,
    atol=1e-6,
)  # ✓
```

`free_development` generalises this to tensor-valued paths and adds block-level control. The example below computes per-block signatures and their tensor logarithms — the piecewise log-linear approximation that `HigherOrderKernel` uses internally:

```python
from tensordev.development import free_development

# per-block signatures: (batch=4, n_blocks=4, dim^k) for block_size=8, steps=32
block_sigs = free_development((X,), trunc=3, block_size=8, accumulate=False)

# tensor log of each block signature → log-linear increments
log_sigs = td.tensor_logarithm(block_sigs[1:], trunc=3, output_zero_level=False)

# free development of the piecewise log-linear path
higher_order_sig = free_development(log_sigs, trunc=3, increment_input=True)

# piecewise log-linear sig recovers the original sig at the same truncation
sig = td.path_signature(X, trunc=3)

np.testing.assert_allclose(
    td.tensor_to_flat(sig),
    td.tensor_to_flat(higher_order_sig),
    rtol=1e-5,
    atol=1e-6,
)  # ✓
```

### `tensordev.sss` — state-space signatures

State-space signatures are Volterra signatures whose convolution kernel is a *finite state-space kernel* (FSSK), i.e. a matrix-exponential kernel of the form

$$K_{A,b}^\Lambda(t,s) = \sum_{r=1}^q \bigl(\mathbf{1}^\top e^{-\Lambda(t-s)} b_r\bigr) A_r,$$

with dense or Jordan state-space operators $\Lambda$. This package propagates
and reads out the ODE hidden state, making online evaluation of these
Volterra signatures exact and efficient.

```python
import jax
import jax.numpy as jnp
from tensordev.sss import StateSpaceSignature

# Jordan kernel:
# one real exponential with rate 1.0
# plus one oscillatory pair with decay 0.5 and frequency 2π
# acting on R^2 paths via A = I_2 → state dim R = 1 + 2 = 3
sss = StateSpaceSignature.from_jordan(
    real_rates=jnp.array([1.0]),
    real_sizes=(1,),
    osc_decays=jnp.array([0.5]),
    osc_freqs=jnp.array([2 * jnp.pi]),
    osc_sizes=(1,),
    A=jnp.eye(2)[None],   # (q=1, m=2, d=2)
    b=jnp.ones((1, 3)),   # (q=1, R=3)
    trunc=3,
)

result = sss.vsig(X, dt=1.0 / 32)

# Moving only X selects GPU execution while the surrounding workflow stays
# on the CPU.
gpu = jax.devices("gpu")[0]
result_gpu = sss.vsig(jax.device_put(X, gpu), dt=1.0 / 32)
```

`StateSpaceSignature` carries an optional persistent hidden state for streaming/online evaluation:

```python
dt = 1.0 / 32

# consume the first half of the path — state is updated, not lost
sss_mid = sss.update_with_path(X[:, :17], dt=dt)
vsig_mid = sss_mid.readout()          # Volterra signature at t = 0.5

# continue with the second half
sss_end = sss_mid.update_with_path(X[:, 16:], dt=dt)
vsig_end = sss_end.readout()          # Volterra signature at t = 1.0

# equivalent to the one-shot call
np.testing.assert_allclose(
    td.tensor_to_flat(vsig_end),
    td.tensor_to_flat(sss.vsig(X, dt=dt)),
    atol=0,
)  # ✓
```

### `tensordev.volterra` — Volterra signature

Volterra signatures for fractional, gamma, and finite state-space kernels,
computed by a quadratic recursion or, on uniform grids, by FFT acceleration.

```python
import jax.numpy as jnp
import tensordev as td
from tensordev.volterra import VolterraSignature, vsig

A = jnp.eye(2)[None]  # (q=1, m=2, d=2)
dt = 1.0 / 32

# functional API — fractional kernel k(t,s) = (t-s)^{β-1} / Γ(β)
kernel = td.ConvolutionKernel.fractional(beta=jnp.array([0.8]), A=A)
result = vsig(X, kernel=kernel, dt=dt, trunc=3)

# class-based — Gamma kernel, adding exponential damping to the fractional kernel
kernel_g = td.ConvolutionKernel.gamma(
    beta=jnp.array([0.8]),
    rate=jnp.array([1.0]),
    scale=jnp.array([1.0]),
    A=A,
)
vsig_obj = VolterraSignature(kernel=kernel_g, trunc=3)
result = vsig_obj.vsig(X, dt=dt)

# Native rectangular truncation.  The split dimensions must sum to kernel.m.
bicore = td.make_core(
    dims=(1, 1),
    max_trunc=(3, 2),
    default_trunc=(2, 1),
)
rectangular = vsig(X, kernel=kernel, core=bicore, dt=dt)

# Use the core's default truncation.
rectangular_vsig = VolterraSignature(kernel=kernel, core=bicore)
```

The quadratic, FFT, and fractional Adams schemes all return tensors in the
format selected by `core`.  Scalar-component kernels (`q=1`) do not
need shuffle plans.  For multi-component kernels above total depth one,
construct the bidegree core with `precompute_shuffle="generator"`; full
shuffle-product plans are unnecessary for Volterra evaluation.

Available kernel constructors:

| Constructor | Formula | Parameters |
|---|---|---|
| `ConvolutionKernel.fractional` | $k_p(t,s) = \Gamma(\beta_p)^{-1}(t-s)^{\beta_p-1}$ | `beta`, `A` |
| `ConvolutionKernel.gamma` | $k(t,s) = \mathrm{scale}\cdot e^{-\mathrm{rate}(t-s)}\cdot\Gamma(\beta)^{-1}(t-s)^{\beta-1}$ | `beta`, `rate`, `scale`, `A` |
| `ConvolutionKernel.fssk` | finite state-space kernel | `fssk` |

Setting `beta=1` with `ConvolutionKernel.fractional` recovers the classical iterated-integral signature.

### `tensordev.kernel` — signature kernels

Kernel objects for empirical statistics: batchwise values, Gram matrices, MMD, and scoring rules. All inherit from `BaseKernel`.

| Class | Description |
|---|---|
| `SigKernel` | Classical signature kernel for Euclidean paths |
| `FreeKernel` | Free signature kernel for tensor-valued paths |
| `FSSKSigKernel` | Kernel induced by the FSSK Volterra signature |
| `HigherOrderKernel` | Higher-order kernel via piecewise log-linear approximation |
| `LinearKernel`, `RBFKernel`, `RBF_CEXP_Kernel`, `RBF_SQR_Kernel` | Static pointwise kernels used as increment kernels |

```python
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from tensordev.util import random_trigonometric_polynomial_paths
from tensordev.kernel import SigKernel, RBFKernel

X = random_trigonometric_polynomial_paths(batch=4, steps=32, dim=2, key=0)
Y = random_trigonometric_polynomial_paths(batch=4, steps=32, dim=2, key=1)

k = SigKernel(dyadic_order=1)

vals = k.compute_kernel(X, Y)                  # (4,)   — batchwise k(X_i, Y_i)
gram = k.compute_Gram(X, Y)                    # (4, 4) — full cross Gram matrix
Kxx  = k.compute_Gram(X)                       # (4, 4) — symmetric Y=None shortcut
mmd  = k.compute_mmd(X, Y)                     # scalar — empirical MMD²
esr  = k.compute_expected_scoring_rule(X, Y)   # scalar — E_Y[S(X, y)]
sr   = k.compute_scoring_rule(X, Y[0])         # scalar — S(X, y) for a single y

# RBF increment kernel — replaces the default ⟨dx, dy⟩ inner product
k_rbf = SigKernel(dyadic_order=0, static_kernel=RBFKernel(sigma=1.0))
gram_rbf = k_rbf.compute_Gram(X, Y)            # (4, 4)
```

`FreeKernel`, `HigherOrderKernel`, and `FSSKSigKernel` share the same empirical API and are drop-in replacements for `SigKernel`:

```python
from tensordev.kernel import FreeKernel, HigherOrderKernel, FSSKSigKernel

# free kernel — accepts tensor-valued paths; level-1 path reduces to SigKernel
k_free = FreeKernel(dyadic_order=1)
gram_free = k_free.compute_Gram(X, Y)          # (4, 4)

# higher-order kernel — log_steps must divide the number of intervals, 32 here
k_ho = HigherOrderKernel(log_steps=(2, 2), log_degree=(3, 3))
mmd_ho = k_ho.compute_mmd(X, Y)                # scalar

# FSSK kernel — wraps the sss.kernel of a StateSpaceSignature
k_fssk = FSSKSigKernel(kernel=sss.kernel, dt_x=1.0 / 32, dt_y=1.0 / 32)
mmd_fssk = k_fssk.compute_mmd(X, Y)            # scalar
```

### `tensordev.util` — utilities

```python
from tensordev.util import (
    path_to_increments,
    integrated_ou_first_on_path,
    random_trigonometric_polynomial_paths,
    unit_speed_paths,
    perturb_path_batch,
    deterministic_trigonometric_path_pair,
    bucket_pad_ragged_paths,
    velocity_to_increments,
)
```

## Tests

```bash
pytest tests/
```

Tests are organized by subpackage:

```text
tests/core/
tests/development/
tests/sss/
tests/volterra/
tests/kernel/
```

## Backends

The core abstraction supports multiple array frameworks; JAX is the only fully
implemented backend.

| Class | Backend | Status |
|---|---|---|
| `Jax` | JAX, JIT-compiled; optional bounded shuffle plans | stable |
| `JaxSequentialCore` | JAX scan / `lax.associative_scan` | stable |
| `Universal` | any array-API namespace | stable |
| `Einsum` | einsum-based base class | stable |
| `Numba` | Numba | stub |
| `Torch` / `TensorFlow` | PyTorch / TensorFlow | stub |

`TENSORDEV_BACKEND` selects the initial backend and defaults to `"jax"`.
`set_default_core` may then replace the active core, while
`reset_default_core` restores that initial backend selection.

```bash
TENSORDEV_BACKEND=jax python my_script.py
```

```python
import tensordev as td

core = td.get_default_core()
seq_core = td.get_default_seq_core()
```

## Acknowledgements and theoretical background

`tensordev` is an independent implementation, but it was influenced by several excellent open-source projects in the signature-computation ecosystem:

- [`iisignature`](https://github.com/bottler/iisignature): a gold-standard reference for efficient signature and logsignature computation.
- [`signatory`](https://github.com/patrick-kidger/signatory): inspired the fused Horner-style evaluation used for efficient tensor exponential / signature development routines.
- [`signax`](https://github.com/anh-tong/signax): provided the initial motivation for building a JAX-native tensor algebra and signature package.
- [`pySigLib`](https://github.com/daniil-shmelev/pySigLib): a high-performance CPU/GPU library for signatures and signature kernels, whose CUDA and JAX support provides an important contemporary reference point for accelerator-aware signature computation.
- [`sigkernel`](https://github.com/crispitagorico/sigkernel): inspired parts of the signature-kernel API and the second-order finite-difference stencil used for the standard signature kernel.
- [`high-order-sigkernel`](https://github.com/maudl3116/high-order-sigkernel): inspired the predictor-corrector schemes for higher-order signature-kernel PDE systems, which are adapted and further developed in this package.

The main theoretical background for the algorithms implemented here is:

- J. Reizenstein and B. Graham,
  [*Algorithm 1004: The iisignature Library: Efficient Calculation of Iterated-Integral Signatures and Log Signatures*](https://arxiv.org/abs/1802.08252),
  ACM Transactions on Mathematical Software, 2020.

- P. Kidger and T. Lyons,
  [*Signatory: differentiable computations of the signature and logsignature transforms, on both CPU and GPU*](https://arxiv.org/abs/2001.00706),
  ICLR 2021.

- T. Nygaard,
  [*pathsig: A GPU-Accelerated Library for Truncated and Projected Path Signatures*](https://arxiv.org/abs/2602.24066),
  arXiv preprint, 2026.

- C. Salvi, T. Cass, J. Foster, T. Lyons and W. Yang,
  [*The Signature Kernel is the Solution of a Goursat PDE*](https://arxiv.org/abs/2006.14794),
  SIAM Journal on Mathematics of Data Science, 2021.

- M. Lemercier, T. Lyons and C. Salvi,
  [*Log-PDE Methods for Rough Signature Kernels*](https://arxiv.org/abs/2404.02926),
  arXiv preprint, 2024.

- P. K. Friz and P. P. Hager,
  [*Expected Signature Kernels for Lévy Rough Paths*](https://arxiv.org/abs/2509.07893),
  arXiv preprint, 2025.

- P. P. Hager, F. N. Harang, L. Pelizzari and S. Tindel,
  [*The Volterra Signature*](https://arxiv.org/abs/2603.04525),
  arXiv preprint, 2026.

- P. P. Hager, F. N. Harang, L. Pelizzari and S. Tindel,
  *Computational Aspects of the Volterra Signature*,
  manuscript.

- P. P. Hager and L. Pelizzari,
  [*Expected signatures via partial integration, coordinate change and symmetrization*](https://arxiv.org/abs/2607.29534),
  arXiv preprint, 2026.

- P. P. Hager and L. Pelizzari,
  [*A Technical Note on Signature Computations with Bidegree Truncation, Coordinate Changes, and Symmetrization*](https://github.com/hagerpa/tensordev/tree/main/academia/bidegree),
  technical note, 2026.
