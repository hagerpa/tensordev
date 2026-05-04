# tensordev

JAX-based tensor algebra library for signatures, free developments, Volterra signatures and inner product-kernels thereof.

## Status

`tensordev` is under active development. The core tensor algebra, signature development, state-space kernels, Volterra signatures, and free/higher-order kernels are implemented and tested.

The current stable backend is JAX. Multi-backend support for PyTorch, TensorFlow, and Numba is planned, but not yet implemented.

The implemented JAX components are end-to-end differentiable — from elementary tensor operations and path signatures through to signature-kernel evaluations and Volterra kernel parameters.

## Requirements

`tensordev` currently requires Python 3.10+ and JAX 0.10.0+.

The package is developed and tested primarily with the JAX backend.

## Installation

Development version from GitHub:

```bash
pip install git+https://github.com/hagerpa/tensordev.git
```

Editable development install:

```bash
git clone https://github.com/hagerpa/tensordev.git
cd tensordev
pip install -e ".[test]"
```

After the first successful PyPI release, the stable installation command will be:

```bash
pip install tensordev
```

## License

`tensordev` is released under the Apache License 2.0.
See [`LICENSE`](LICENSE) for details.

## Quick start

```python
import tensordev as td
from tensordev.util import random_trigonometric_polynomial_paths

X = random_trigonometric_polynomial_paths(batch=4, steps=32, dim=2, key=0)

sig = td.path_signature(X, trunc=4)
ip = td.tensor_inner_product(sig, sig)

print(td.tensor_to_flat(sig).shape)
print(ip.shape)
```

## Package structure

```text
tensordev/
├── core/           # Tensor algebra backends
├── development/    # Signature development, free and classical
├── sss/            # State-space signatures, aka Volterra signatures for finite state-space kernels
├── volterra/       # Volterra signatures: fractional, gamma, piecewise-constant kernels
├── kernel/         # Signature kernels: classical, free, FSSK, higher-order
└── util/           # Path generators and combinatorics
```

### `tensordev.core` — tensor algebra operations

A *core object* exposes all tensor algebra operations over a chosen array backend. The default is `Jax()`. Operations are also available directly on the `tensordev` module:

```python
# If desired, enable float64 — must be set before importing JAX or tensordev.
import jax
jax.config.update("jax_enable_x64", True)

import tensordev as td
import jax.numpy as jnp

# two elements of the truncated tensor algebra, levels 0..3
A = (jnp.ones((1,)), jnp.ones((2,)), jnp.ones((4,)), jnp.ones((8,)))
B = (jnp.ones((1,)), jnp.ones((2,)), jnp.ones((4,)), jnp.ones((8,)))

# Chen/tensor/concatenation product
C = td.tensor_product(A, B, trunc=3)

# level-wise sum
S = td.tensor_summation(A, B)

# truncated exponential and logarithm, inputs start at level 1
E = td.tensor_exponential(A[1:], trunc=3)
L = td.tensor_logarithm(A[1:], trunc=3)

# inner product
ip = td.tensor_inner_product(A, B)
```

For the shuffle, i.e. commutative, product, create a `ShuffleCore` for a fixed base dimension and truncation. All shuffle operators are precomputed at construction time as sparse arrays, so repeated calls are pure compute with no reallocation.

```python
sc = td.shuffle_core(d=2, trunc=3)
C = sc.tensor_shuffle_product(A, B, trunc=3)
```

Use `shuffle_core_expected_memory` to check the memory budget before constructing at large `d` or `trunc`. It returns an upper bound; actual usage is typically 20–40% less.

```python
td.shuffle_core_expected_memory(d=2, trunc=8)   # →   1.63 MB  (actual  0.52 MB)
td.shuffle_core_expected_memory(d=4, trunc=6)   # →   5.85 MB  (actual  4.07 MB)
td.shuffle_core_expected_memory(d=4, trunc=8)   # → 363.85 MB  (actual 218.93 MB)
td.shuffle_core_expected_memory(d=8, trunc=6)   # → 353.44 MB  (actual 297.57 MB)
```

### `tensordev.development` — signature development

Compute truncated signatures with optional blocking. `block_size` splits the path into chunks and chains them via Chen's identity internally. This is useful for long sequences where the full path does not fit in memory at once.

```python
import numpy as np
import jax.numpy as jnp
import tensordev as td
from tensordev.util import random_trigonometric_polynomial_paths

X = random_trigonometric_polynomial_paths(batch=4, steps=32, dim=2, key=0)

sig = td.path_signature(X, trunc=4)  # one shot

# signature on two consecutive intervals
sig_blocked = td.path_signature(X, trunc=4, block_size=16, accumulate=False)

# Chen's identity: sig = sig_a ⊗ sig_b
sig_a = td.tensor_slice(sig_blocked)[:, 0]
sig_b = td.tensor_slice(sig_blocked)[:, 1]
sig_c = td.tensor_product(sig_a, sig_b, trunc=4)

np.testing.assert_allclose(
    td.tensor_to_flat(sig),
    td.tensor_to_flat(sig_c),
    atol=1e-12,
)  # ✓

# shuffle identity: <sig, a ⊔ b> = <sig, a> · <sig, b>
# a, b are fixed basis vectors — broadcast over the batch dimension
e1 = td.tensor_densify((None, jnp.array([1., 0.])))
e2 = td.tensor_densify((None, jnp.array([0., 1.])))
sc = td.shuffle_core(d=2, trunc=4)

np.testing.assert_allclose(
    td.tensor_inner_product(sig, sc.tensor_shuffle_product(e1, e2, trunc=4)),
    td.tensor_inner_product(sig, e1) * td.tensor_inner_product(sig, e2),
    atol=1e-10,
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
    atol=1e-12,
)  # ✓
```

### `tensordev.sss` — state-space signatures

State-space signatures are Volterra signatures whose convolution kernel is a *finite state-space kernel* (FSSK), i.e. a matrix-exponential kernel of the form

$$K_{A,b}^\Lambda(t,s) = \sum_{r=1}^q \bigl(\mathbf{1}^\top e^{-\Lambda(t-s)} b_r\bigr) A_r .$$

with dense or Jordan state-space operators $\Lambda$. This package provides functionality for propagating and reading out the hidden state that evolves via an ODE, making online/streaming evaluation of such Volterra signatures exact and efficient.

```python
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

Volterra signatures for fractional, Gamma, and piecewise-constant kernel families, computed via the quadratic triangular recursion with JAX-vectorised evaluation of the inner loop.

```python
import jax.numpy as jnp
from tensordev.volterra import VolterraKernel, VolterraSignature, vsig

A = jnp.eye(2)[None]   # (q=1, m=2, d=2)
dt = 1.0 / 32

# functional API — fractional kernel k(t,s) = (t-s)^{β-1} / Γ(β)
kernel = VolterraKernel.fractional(beta=jnp.array([0.8]), A=A)
result = vsig(X, kernel=kernel, dt=dt, trunc=3)

# class-based — Gamma kernel, adding exponential damping to the fractional kernel
kernel_g = VolterraKernel.gamma(
    beta=jnp.array([0.8]),
    rate=jnp.array([1.0]),
    scale=jnp.array([1.0]),
    A=A,
)
vsig_obj = VolterraSignature(kernel=kernel_g, trunc=3)
result = vsig_obj.vsig(X, dt=dt)
```

Available kernel constructors:

| Constructor | Formula | Parameters |
|---|---|---|
| `VolterraKernel.fractional` | $k_p(t,s) = \Gamma(\beta_p)^{-1}(t-s)^{\beta_p-1}$ | `beta`, `A` |
| `VolterraKernel.gamma` | $k(t,s) = \mathrm{scale}\cdot e^{-\mathrm{rate}(t-s)}\cdot\Gamma(\beta)^{-1}(t-s)^{\beta-1}$ | `beta`, `rate`, `scale`, `A` |
| `VolterraKernel.piecewise_constant` | $k(i,j) = B_{p,i,j}$ | `B`, `A` |

Setting `beta=1` with `VolterraKernel.fractional` recovers the classical iterated-integral signature.

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

The core abstraction supports multiple array frameworks. Currently only the JAX backend is fully implemented.

| Class | Backend | Status |
|---|---|---|
| `Jax` | JAX, JIT-compiled | stable |
| `JaxSequentialCore` | JAX scan / `lax.associative_scan` | stable |
| `JaxShuffleCore` | JAX sparse shuffle operators | stable |
| `Universal` | any array-API namespace | stable |
| `Einsum` | einsum-based base class | stable |
| `Numba` | Numba | stub |
| `Torch` / `TensorFlow` | PyTorch / TensorFlow | stub |

The active backend is selected via the `TENSORDEV_BACKEND` environment variable. The default is `"jax"`.

```bash
TENSORDEV_BACKEND=jax python my_script.py
```

```python
from tensordev._backend import get_default_core, get_default_seq_core

core = get_default_core()
seq_core = get_default_seq_core()
```

## Acknowledgements and theoretical background

`tensordev` is an independent implementation, but it was influenced by several excellent open-source projects in the signature-computation ecosystem:

- [`signatory`](https://github.com/patrick-kidger/signatory): inspired the fused Horner-style evaluation used for efficient tensor exponential / signature development routines.
- [`signax`](https://github.com/anh-tong/signax): provided the initial motivation for building a JAX-native tensor algebra and signature package.
- [`sigkernel`](https://github.com/crispitagorico/sigkernel): inspired parts of the signature-kernel API and the second-order finite-difference stencil used for the standard signature kernel.
- [`high-order-sigkernel`](https://github.com/maudl3116/high-order-sigkernel): inspired the predictor-corrector schemes for higher-order signature-kernel PDE systems, which are adapted and further developed in this package.

The main theoretical background for the algorithms implemented here is:

- P. Kidger and T. Lyons,
  [*Signatory: differentiable computations of the signature and logsignature transforms, on both CPU and GPU*](https://arxiv.org/abs/2001.00706),
  ICLR 2021.

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
