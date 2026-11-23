# Performance

This page covers what decides the speed and memory use of `tensordev`: the native CPU kernels,
device placement, the `execution` argument, JIT compilation, and the memory of plans, long paths
and gradients. The last section shows benchmark tables and how to reproduce them.

The examples use this setup:

```python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import tensordev as td
from tensordev.util import random_trigonometric_polynomial_paths

X = random_trigonometric_polynomial_paths(batch=4, steps=32, dim=2, key=0)
```

## Native CPU kernels

The wheels for Linux (x86_64 and aarch64) and macOS on Apple Silicon contain compiled CPU kernels,
which JAX calls through the XLA FFI. The Linux wheels are built and tested in `manylinux_2_28`
containers, so glibc 2.28 is the tested baseline. The kernels compute one operation, the
signature step $g \mapsto g\otimes\exp_\otimes(\Delta)$ of
[Path signatures](signatures.md#arguments), truncated at the bidegree $(N, M)$ of a partially
symmetrized core. A step uses them when

- the core is partially symmetrized (`partially_symmetrized=True`), in standard coordinates, with
  `dims=(1, q)` and `q > 1`;
- the data are `float32` or `float64`;
- the work per path $W=\sum_{(n,m)\neq(0,0)}(N+M-n-m+1)\,w_{n,m}$, with $w_{n,m}$ the number of
  coordinates of block $(n, m)$, is at least 3000 for $q = 2$ and 1500 for $q \ge 3$, and $W$
  times the number of paths is at least 32768;
- JAX runs the computation on the CPU.

Otherwise the same call runs the JAX implementation, without an error or a warning.

The kernels are part of the JAX program. They work inside `jax.jit`, and both `execution="auto"`
and `execution="jax"` use them where they apply. Derivatives are computed by the JAX
implementation, so `jax.grad` works the same way with and without them:

```python
core = td.make_core(dims=(1, 2), max_trunc=(4, 4), partially_symmetrized=True)
X3 = random_trigonometric_polynomial_paths(batch=8, steps=64, dim=3, key=1)

sig = td.path_signature(X3, core=core)
print(sig[2, 1].shape)  # (8, 6): the bidegree (2, 1) block of each path

grad = jax.grad(lambda x: core.tensor_to_flat(td.path_signature(x, core=core)).sum())(X3)
print(grad.shape)  # (8, 65, 3)
```

The kernels are loaded from the private module `tensordev._native_cpu`, which a wheel with native
kernels bundles, or from the standalone companion module `tensordev_native_cpu`, which
`python -m pip install ./native` installs from a clone. This prints whether either is installed:

```python
import importlib.util

print(any(importlib.util.find_spec(name) is not None
          for name in ("tensordev._native_cpu", "tensordev_native_cpu")))
```

To build the kernels when installing from a clone of the repository, set `TENSORDEV_BUILD_NATIVE`:

```bash
TENSORDEV_BUILD_NATIVE=1 python -m pip install .
```

This needs a C++17 compiler; the build environment fetches CMake and the JAX headers. A failed
native build is reported, not replaced by a wheel without kernels. The
[native README](https://github.com/hagerpa/tensordev/tree/main/native) explains platform wheels,
the macOS deployment target and the binary interface. For the plain installation see
[Installation](index.md#installation).

## Device placement

--8<-- "README.md:device_placement"

`jax.devices()` lists the devices that JAX sees, and `JAX_PLATFORMS=cpu` keeps a process on the
CPU.

On NVIDIA GPUs of the Ampere generation and newer, XLA runs `float32` matrix products in TF32
by default, which keeps about three significant digits. tensordev requests full `float32`
precision for its own matrix products, so its results do not depend on this setting. Your own
JAX code around tensordev still follows it; to change it for the whole process:

```python
import jax

jax.config.update("jax_default_matmul_precision", "highest")
```

## Wordwise GPU execution

--8<-- "README.md:wordwise"

!!! warning "Alpha"

    Check the results of `execution="wordwise"` against `execution="jax"`.

The `execution` argument is accepted by [`path_signature`][tensordev.path_signature], calls of a
[`Signature`][tensordev.Signature], [`fssk_vsig`][tensordev.fssk_vsig],
[`fssk_state`][tensordev.fssk_state], and the methods `vsig`, `states`, `update_with_path` and
`update_with_increment` of [`StateSpaceSignature`][tensordev.StateSpaceSignature].

| Value | Implementation |
|---|---|
| `"auto"` (default) | The JAX implementation on every device, as with `"jax"`. |
| `"jax"` | The JAX implementation; never selects the wordwise kernels. |
| `"wordwise"` | The alpha NVIDIA kernels. A call that does not meet the requirements below raises `ValueError` instead of falling back. |

Requirements of `execution="wordwise"`:

| Requirement | Detail |
|---|---|
| Workload | Ordinary signatures (`path_signature`, `Signature`), or FSSK signatures and states with `q = 1`. |
| Core | A JAX core of any type: total degree (including the default core), bidegree, shear, partially symmetrized. |
| Call | Eager: not inside `jax.jit`, `jax.grad` or `jax.vmap`. |
| Options | For signatures, `parallel=False` and `accumulate_in_tree=False`, the defaults. |
| Input | `float32` or `float64`, on one NVIDIA GPU with CUDA compute capability 8.0 or newer. |
| Derivatives | Not supported. For gradients use `"auto"` or `"jax"`, which differentiate the JAX implementation. |

On a CPU, a wordwise request raises, and `"auto"` agrees with `"jax"`:

```python
X_cpu = jax.device_put(X, jax.devices("cpu")[0])
try:
    td.path_signature(X_cpu, trunc=4, execution="wordwise")
except ValueError as error:
    print(error)
# execution='wordwise' requires input on a single NVIDIA GPU with CUDA compute capability
# 8.0 or newer. Use execution='jax' or 'auto' for portable execution.

sig_auto = td.path_signature(X_cpu, trunc=4)
sig_jax = td.path_signature(X_cpu, trunc=4, execution="jax")
np.testing.assert_allclose(td.tensor_to_flat(sig_auto), td.tensor_to_flat(sig_jax))
```

On a supported GPU the request is the same call with the path on that device, for example
`td.path_signature(X_gpu, trunc=4, execution="wordwise")` with `X_gpu` as in
[Device placement](#device-placement).

For a piecewise linear path with increments $\Delta_k$ and $S_k=\pi_{\le N}\mathrm{Sig}(X)_{0,t_k}$,
Chen's identity gives the coefficient of a word $w=i_1\cdots i_n$ after step $k$ as

$$
\langle w, S_k\rangle=\sum_{j=0}^{n}\langle i_1\cdots i_j, S_{k-1}\rangle\,
\frac{\Delta_k^{i_{j+1}}\cdots\Delta_k^{i_n}}{(n-j)!} .
$$

A word needs only its own prefixes, so words and paths are processed in parallel. One kernel
serves ordinary and `q = 1` FSSK signatures. With a state $P_v\in\mathbb{R}^R$ for every word $v$
and $(A_\Delta X)_{u\ell}=X_u\,\Delta^{\ell}$ for a word $u$ and a letter $\ell$, a step is

$$
P\leftarrow PE+\sum_{n=1}^{N}X_nA_\Delta^{\,n},\qquad
X_n=P\varphi_{n-1},\quad X_n=\psi_{n-1}\ \text{at the empty word,}
$$

with the [coefficients](state-space.md#per-step-coefficients) `E`, `psi`, `phi` of the FSSK recursion; the
signature is $R=1$, $E=1$, $\psi_{n-1}=\varphi_{n-1}=1/n!$, the formula above. For a partially
symmetrized core the words are replaced by their classes, and $A_\Delta$ sums over the letters
that a class can end with.

One launch computes all levels. A lane of the GPU owns a set of words that contains their
prefixes:

| Core | Words of one lane |
|---|---|
| Total degree | A word of length $N-L$, its prefixes and its extensions by at most $L$ letters. |
| Ordered bidegree | A word of length $N-1$, its prefixes and its extensions by one prime letter, or by one double-prime letter. |
| Partially symmetrized | A class of the top bidegree and the classes of its prefixes. |

A word owned by several lanes is computed by each of them and stored by one. The approach is from
`pathsig` ([Acknowledgements](background.md#acknowledgements)); the
[technical note](background.md#technical-note) adapts it to the bidegree layouts.

## JIT and compilation

`tensordev` functions are JAX code. Wrap a computation in `jax.jit` to compile it as one
program: the first call traces and compiles it, and later calls with inputs of the same shape and
dtype reuse the compiled program. A new shape or dtype compiles again.

```python
signature = jax.jit(lambda x: td.path_signature(x, trunc=4))

sig = signature(X)            # traces and compiles
sig = signature(2.0 * X)      # same shape and dtype: reuses the program
sig = signature(X[:, :17])    # new shape: compiles again
print(td.tensor_to_flat(sig).shape)  # (4, 31)
```

Without `jax.jit`, the first call also compiles, because the implementation compiles its inner
loops; later calls are faster. The [benchmark tables](#bidegree-versus-total-degree) show both
phases.

A traced function keeps the core it was traced with. The module-level operations, the default
core and its default truncation are read during tracing, so a later `td.set_default_core` does not
reach a function that is already compiled:

```python
signature = jax.jit(lambda x: td.path_signature(x))

td.set_default_core(dims=2, max_trunc=3)
print(td.tensor_to_flat(signature(X)).shape)        # (4, 15): truncation 3

td.set_default_core(dims=2, max_trunc=2)
print(td.tensor_to_flat(signature(X)).shape)        # (4, 15): still truncation 3
print(td.tensor_to_flat(td.path_signature(X)).shape)  # (4, 7): an eager call sees the new core
td.reset_default_core()
```

Set the default core before you trace, or pass `core=` explicitly. See
[Tensor algebra](tensor-algebra.md) for `set_default_core` and `reset_default_core`.

## Memory

### Plans of bounded cores

[`core_expected_memory`][tensordev.core_expected_memory] gives the plan memory of a core
configuration without building the core; see
[Shuffle plans and memory](tensor-algebra.md#shuffle-plans-and-memory).

### Long paths and gradients

For a path with $S$ increments and a truncation with $P$ coordinates, the options of
`path_signature` behave as follows.

| Option | Memory |
|---|---|
| `parallel=False` (default) | Streams the increments through a scan and keeps one running tensor of $P$ coordinates per path. Work memory is of the order of the input plus these running tensors; it does not grow with $S \cdot P$. |
| `parallel=True` | Forms the exponential of every increment first, then combines them with an associative scan. More parallelism; memory grows with $S \cdot P$. |
| `block_size=B` | $B$ must divide $S$. Computes the signatures of the $S/B$ blocks side by side and, with `accumulate=True`, chains them with Chen's identity. The output has a block axis of length $S/B$. It does not lower the peak memory. |
| `jax.grad` | Reverse-mode differentiation keeps intermediate tensors of every step: memory grows with $S \cdot P$. |

XLA reports the temporary memory of a compiled function. With $S = 1024$ and $P = 31$ (two
dimensions, truncation 4), on the CPU with JAX 0.10.0:

```python
def work_kib(fn, x):
    """Temporary memory of the compiled function, as estimated by XLA, in KiB."""
    return jax.jit(fn).lower(x).compile().memory_analysis().temp_size_in_bytes / 2**10

X_long = random_trigonometric_polynomial_paths(batch=4, steps=1024, dim=2, key=0)
loss = lambda x: td.tensor_to_flat(td.path_signature(x, trunc=4)).sum()

print(round(work_kib(lambda x: td.path_signature(x, trunc=4), X_long)))                 # 128
print(round(work_kib(lambda x: td.path_signature(x, trunc=4, parallel=True), X_long)))  # 2486
print(round(work_kib(lambda x: td.path_signature(x, trunc=4, block_size=64), X_long)))  # 145
print(round(work_kib(jax.grad(loss), X_long)))                                          # 803

blocks = td.path_signature(X_long, trunc=4, block_size=64)
print(td.tensor_to_flat(blocks).shape)  # (4, 16, 31): the running signature after each block
```

For long paths, keep the default `parallel=False`. If a gradient does not fit in memory, process
the batch in smaller pieces.

## Benchmarks

The scripts in [`benchmarks/`](https://github.com/hagerpa/tensordev/tree/main/benchmarks) are
diagnostics, not tests: timings depend on the machine, the JAX version and the backend. They write
JSON files, and `docs/scripts/benchmarks.py` turns these into the tables below. Each table states
where it was measured.

### Bidegree versus total degree

`benchmarks/bidegree_efficiency.py` compares a bidegree core with the total-degree core on a
four-dimensional path, split as $(2, 2)$. It times ordinary signatures (`td.path_signature`) and
Volterra signatures (`td.vsig` with a `FractionalKernel` with $\beta = 0.7$ and $A = I_4$,
`scheme="quadratic"`, `order=0`). The bidegree core is ordered and in standard coordinates
(`make_core(dims=(2, 2), max_trunc=(N, N))`), so the native CPU kernels play no part.

A matched comparison sets total degree $2N$ against bidegree $(N, N)$: both keep words up to
length $2N$, but the bidegree core drops the words with more than $N$ letters from either half of
the alphabet. The payload column of the first table shows the coordinate counts of
[Bidegree tensors](tensor-algebra.md#bidegree-tensors) for $d' = d'' = 2$, counting the scalar.
With `--include-mismatched` the script adds total degree 6 against bidegree
$(4, 4)$, labelled non-equivalent: $(4, 4)$ also keeps words of lengths 7 and 8, so that row is no
speed comparison.

The script times these phases separately:

- setup of the core and the kernel;
- in `outer-jit` mode, lowering and compilation of the whole call, before it runs;
- the first execution; in `direct` mode, which calls the public function without an outer
  `jax.jit`, it includes the compilation done inside the call;
- warm executions after a few untimed warm-up calls.

Every output array is synchronized before a timer stops. The JSON file also records XLA's cost and
memory analysis of the `outer-jit` rows; these are diagnostic estimates, not timings, and are not
in the tables.

--8<-- "docs/generated/bidegree.md"

To reproduce, run from a clone of the repository (the script imports helpers from `notebooks/`):

```bash
# quick check
PYTHONPATH=src python benchmarks/bidegree_efficiency.py \
    --preset smoke --cases matched6 --modes outer-jit

# the tables above
PYTHONPATH=src python benchmarks/bidegree_efficiency.py \
    --preset standard --output benchmarks/results/bidegree_efficiency.json
python docs/scripts/benchmarks.py
```

The script also writes a CSV file next to the JSON file. `--dtype float32`,
`--volterra-scheme fft` and `--clear-caches` (clear the JAX caches before every row, for cold
starts) change the setup; `--help` lists all options.

### Wordwise versus JAX

`benchmarks/wordwise_gpu.py` checks the alpha wordwise kernels against the JAX implementation and
times both on one NVIDIA GPU with compute capability 8.0 or newer. It can cover ordinary
signatures and `q = 1` FSSK recursions (hidden states and the readout), total-degree, bidegree and
partially symmetrized cores in standard and shear coordinates, `float32` and `float64`, and
terminal, accumulated and independent block outputs. Differentiation is not benchmarked, because
transformed calls use the JAX implementation.

--8<-- "docs/generated/wordwise_gpu.md"

To reproduce, on a machine with a supported GPU:

```bash
PYTHONPATH=src python benchmarks/wordwise_gpu.py \
    --workloads ordinary fssk \
    --families total-standard bidegree-partial-standard \
    --fssk-outputs state readout \
    --warmups 5 --repeats 30 \
    --output benchmarks/results/wordwise_gpu.json
python docs/scripts/benchmarks.py
```

`--families`, `--dtypes`, `--block-modes`, `--coefficient-modes` and `--initial-states` select
the rest of the matrix; `--help` lists all options.
