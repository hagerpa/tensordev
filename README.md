# tensordev

JAX-based tensor algebra library for signature kernels and Volterra signature methods.

## Status

Active development. The core tensor algebra, signature development, state-space kernels, Volterra signatures, and free/higher-order kernels are implemented and tested. Multi-backend support (PyTorch, TensorFlow, Numba) is planned but not yet implemented.

## Installation

```bash
pip install -e .
```

## Package structure

```
tensordev/
├── core/           # Tensor algebra backends
├── development/    # Signature development (free and classical)
├── sss/            # Finite-state-space Volterra kernels (FSSK)
├── volterra/       # Volterra signature (non-Markovian, quadratic recursion)
├── kernel/         # Signature kernels (Gram matrices, MMD, scoring rules)
└── util/           # Path generators and combinatorics
```

### `tensordev.core` — tensor algebra backends

The central abstraction is a *core object* that provides all tensor algebra operations (products, exponentials, logarithms, shuffles, path developments) over a chosen array framework.

| Class | Backend | Status |
|---|---|---|
| `Jax` | JAX (JIT-compiled) | stable |
| `Universal` | any array-API namespace | stable |
| `Einsum` | base class for einsum-based cores | stable |
| `JaxSequentialCore` | JAX scan/lax.associative_scan | stable |
| `Numba` | Numba | stub |
| `Torch` / `TensorFlow` | PyTorch / TensorFlow | stub |

```python
from tensordev import Jax, Universal
import numpy as np

core = Jax()
# or: core = Universal(np)
```

All cores expose the same API: `tensor_product`, `tensor_summation`, `tensor_exponential`, `tensor_logarithm`, `tensor_development`, `tensor_path_signature`, `shuffle_product`, etc.

### `tensordev.development` — signature development

High-level functions that compute the (free or classical) signature of a path, with optional blocking, parallelism, and batching.

```python
from tensordev import path_signature, Signature

# functional
sig = path_signature(X, trunc=4, core=core)

# class-based (binds core + trunc)
sig_fn = Signature(trunc=4)
sig = sig_fn(X)
```

For tensor-valued paths (packed positive-level form):

```python
from tensordev.development import free_development, FreeDevelopment

sig = free_development((X1, X2), trunc=3, core=core, seq_core=seq_core)
```

### `tensordev.sss` — finite-state-space Volterra kernels

Kernels of the form

$$K_{A,b}^\Lambda(t,s) = \sum_{r=1}^q \bigl(\mathbf{1}^\top e^{-\Lambda(t-s)} b_r\bigr) A_r$$

with dense or Jordan state-space operators $\Lambda$.

```python
from tensordev.sss import FSSK, StateSpaceSignature, Lambda, DenseLambda, JordanLambda

kernel = FSSK(Lambda=DenseLambda(L), A=A, b=b)

# compute the Volterra signature (FSSK path)
vsig_fn = StateSpaceSignature(kernel=kernel, trunc=4)
result = vsig_fn.vsig(X, times=times)
```

`StateSpaceSignature` carries an optional persistent hidden state for streaming / online evaluation.

### `tensordev.volterra` — Volterra signature

Volterra signature for non-Markovian convolution kernels via the quadratic triangular recursion. Supports fractional, Gamma, and piecewise-constant kernel families.

```python
from tensordev.volterra import VolterraKernel, VolterraSignature, vsig

# fractional kernel: k_p(t,s) = (t-s)^{beta_p - 1} / Gamma(beta_p)
kernel = VolterraKernel.fractional(beta=jnp.array([0.8]), A=A)

# functional API
result = vsig(X, kernel=kernel, times=times, trunc=3)

# class-based (binds kernel + trunc)
vsig_fn = VolterraSignature(kernel=kernel, trunc=3)
result = vsig_fn.vsig(X, times=times)
```

Available kernel constructors:

| Constructor | Formula | Parameters |
|---|---|---|
| `VolterraKernel.fractional` | $k_p(t,s) = \Gamma(\beta_p)^{-1}(t-s)^{\beta_p-1}$ | `beta`, `A` |
| `VolterraKernel.gamma` | $k(t,s) = \mathrm{scale}\cdot e^{-\mathrm{rate}(t-s)}\cdot\Gamma(\beta)^{-1}(t-s)^{\beta-1}$ | `beta`, `rate`, `scale`, `A` |
| `VolterraKernel.piecewise_constant` | $k(i,j) = B_{p,i,j}$ | `B`, `A` |

Setting `beta=1` with `VolterraKernel.fractional` recovers the classical iterated-integral signature.

### `tensordev.kernel` — signature kernels

Kernel objects for empirical statistics (batchwise values, Gram matrices, MMD, energy score). All inherit from `BaseKernel`.

| Class | Description |
|---|---|
| `SigKernel` | Classical signature kernel for Euclidean paths |
| `FreeKernel` | Free signature kernel for tensor-valued paths |
| `FSSKSigKernel` | Kernel induced by the FSSK Volterra signature |
| `HigherOrderKernel` | Higher-order kernel via piecewise log-linear approximation |
| `LinearKernel`, `RBFKernel`, `RBF_CEXP_Kernel`, `RBF_SQR_Kernel` | Static (pointwise) kernels used as increment kernels |

```python
from tensordev.kernel import SigKernel, FreeKernel, HigherOrderKernel

k = SigKernel(dyadic_order=1)
gram = k.gram(X, Y)           # (N, M) Gram matrix
mmd  = k.mmd(X, Y)            # scalar MMD^2
score = k.energy_score(X, Y)  # energy score

k_ho = HigherOrderKernel(log_steps=(2, 2), log_degree=(3, 3))
```

### `tensordev.util` — utilities

```python
from tensordev.util import (
    path_to_increments,
    integrated_ou_first_on_path,
    random_trigonometric_polynomial_paths,
    unit_speed_paths,
)
```

## Quick start

```python
import jax.numpy as jnp
from tensordev import Jax, path_signature
from tensordev.volterra import VolterraKernel, vsig
from tensordev.kernel import SigKernel

# Classical signature
X = jnp.array([[0.0, 0.0], [0.2, -0.1], [0.4, 0.3], [0.1, 0.5]])
sig = path_signature(X, trunc=4)

# Volterra signature (fractional kernel, beta=0.7)
A = jnp.eye(2)[None]
kernel = VolterraKernel.fractional(beta=jnp.array([0.7]), A=A)
result = vsig(X, kernel=kernel, dt=1.0, trunc=3)

# Signature kernel MMD
k = SigKernel()
mmd = k.mmd(X[None], X[None])   # trivially 0 for identical samples
```

## Backend selection

The default backend is JAX. Override via environment variable:

```bash
TENSORDEV_BACKEND=jax python my_script.py
```

Programmatic access:

```python
from tensordev._backend import get_default_core, get_default_seq_core
core = get_default_core()
seq_core = get_default_seq_core()
```

## Tests

```bash
pytest tests/
```

Tests are organized by subpackage: `tests/core/`, `tests/development/`, `tests/sss/`, `tests/volterra/`, `tests/kernel/`.
