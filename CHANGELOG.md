# Changelog

## Unreleased

### Fixed

- `RBF_SQR_Kernel` and `RBF_CEXP_Kernel` act on the node values
  `batch + (length, dim)` like every static kernel. They flattened the time
  axis before, so `SigKernel` and `FreeKernel` returned one value for every
  pair of paths and `FSSKSigKernel` raised. Function-valued paths are passed
  flattened; the new `RBF_CEXP_Kernel(channels=...)` recovers their grid.
- FSSK coefficients were silently wrong for steps beyond the reach of their
  contour quadrature: weakly damped oscillations with `omega * dt` above
  about 10 and growing modes with `|rate| * dt` above about 3 (up to 40%
  relative error in `fssk_vsig`). `fssk_state`, `fssk_vsig` and
  `StateSpaceSignature.vsig` now split such steps into equal sub-steps,
  which is exact for piecewise-linear paths. When `Lambda` or `dt` is
  traced, the coefficients of a `JordanLambda` are NaN beyond the contour
  instead of wrong.

## 0.1.1

### Fixed

- `GammaKernel` coefficients use the unnormalised basis weight
  `(u - s)**rho`, as `FractionalKernel` and the interpolation basis do.
  Order-1 and order-2 `vsig` and `vsig_fft` results for Gamma kernels were
  wrong before.
- `GammaKernel` coefficients are exact. A positive series in the fractional
  closed form replaces Gauss-Legendre quadrature, which lost accuracy at the
  endpoint singularity (up to 8% relative error for small `beta`). Beyond
  the series range (`rate * (t - s) > 13.8` in float64) a tanh-sinh rule
  with `quad_order` nodes is used; its default is now 64. Gradients on
  masked time triples are zero instead of NaN.
- Derivatives with respect to `beta`: the new
  `tensordev.util.special.betainc` is differentiable in all arguments, and
  the higher-order basis exponents follow `beta`. `jax.grad` now works
  through `FractionalKernel`, `GammaKernel`, `vsig` and `vsig_fft` for
  `order` 0, 1 and 2.
- The alpha wordwise kernels (`execution="wordwise"`) failed to compile on
  NVIDIA GPUs with JAX 0.10, because the Triton backend rejects value-level
  scatter and slicing. The kernels now lower on JAX 0.10 and 0.11 with
  unchanged results, and a CPU-runnable test exports every kernel family for
  CUDA.
- Keep dense state-space matrix functions and shifted solves connected to the
  current matrix parameter, including under JAX transformations.
- Support reverse-mode and higher derivatives of state-space phi functions,
  with stable values near zero and correct Jordan exponential derivatives at
  zero time.
- Preserve explicit working precision for state-space time increments and
  allow dense and oscillatory Jordan construction under JIT.
- `float32` results on NVIDIA Ampere and newer GPUs lost about three digits
  (state-space signatures up to 2% relative error), because XLA runs
  `float32` matrix products in TF32 by default. All matrix products in
  tensordev now request full `float32` precision. CPU results are unchanged.

### Documentation

- A documentation site, built with MkDocs from `docs/`: guides for tensor
  algebra, signatures, state-space and Volterra signatures, signature kernels
  and performance, and an API reference.

## 0.1.0

### Added and changed

- Add `make_core` and `set_default_core` as the common construction and
  background-configuration API for bounded total-degree and bidegree cores.
- Add ordered and partially symmetrized bidegree tensors through
  `symmetrize_core`, rectangular truncation, and conversion back to ordered
  tensors.
- Add total-degree and bidegree shear coordinates, including shuffle products
  and `tensor_shear_pairing` with ordered or partially symmetrized tensors in
  standard coordinates.
- Make signatures, free developments, Volterra signatures, and tensor
  operations follow the selected default core.
- Replace the separate shuffle core with optional shuffle precomputation on
  each bounded core and provide one public memory estimator,
  `core_expected_memory`.
- Bundle native CPU kernels for partially symmetrized bidegree signature
  evaluation in Linux x86_64/aarch64 and macOS Apple Silicon wheels. Retain
  compiler-free pure-Python wheels and source installations, with automatic
  JAX fallback for unsupported workloads and CPU-excluded configurations.
- Add alpha NVIDIA wordwise execution for ordinary and scalar-FSSK signatures
  through `execution="wordwise"`. The default `"auto"` policy retains portable
  JAX on GPUs; `"jax"` explicitly selects portable execution. Explicit wordwise
  requests report unsupported calls instead of falling back.

### Migration from 0.0.3

- Python 3.10 is no longer supported; TensorDev now requires Python 3.11 or
  newer.
- The supported dependency range is `jax>=0.10.0,<0.12` and
  `jaxlib>=0.10.0,<0.12`.
- Replace `JaxShuffleCore(d, trunc)` or `shuffle_core(d, trunc)` with
  `make_core(dims=d, max_trunc=trunc, precompute_shuffle=True)`. Use
  `set_default_core` with the same arguments when the core should become the
  background default.
- Replace `shuffle_core_expected_memory` with `core_expected_memory`, using
  `dims=d`, `max_trunc=trunc`, and `precompute_shuffle=True`. The result is
  reported in MiB by default.

`JaxSequentialCore` remains the public sequential core. Development functions
resolve it automatically for built-in cores, including cores returned by
`make_core`; `set_default_core` installs the matching core pair.

The mathematical construction and implementation details are documented in
`academia/bidegree/` and the papers cited from the README.
