# Changelog

## Unreleased

### Fixed

- `RBF_SQR_Kernel` and `RBF_CEXP_Kernel` act on the node values
  `batch + (length, dim)` like every static kernel. They flattened the time
  axis before, so `SigKernel` and `FreeKernel` returned one value for every
  pair of paths and `FSSKSigKernel` raised. A path of functions with shape
  `batch + (length_t, L, channels)` is passed as
  `X.reshape(X.shape[:-2] + (-1,))`, with the new argument
  `RBF_CEXP_Kernel(channels=...)`.
- FSSK coefficients were wrong for long steps: up to 40% relative error in
  `fssk_vsig` for weakly damped oscillations with `omega * dt` above about 10
  and for growing modes with `|rate| * dt` above about 3. `fssk_state`,
  `fssk_vsig` and `StateSpaceSignature.vsig` now split every step into equal
  sub-steps with `omega * dt <= 4` and `|rate| * dt <= 1` (default
  `quad_order`), which leaves a piecewise-linear path unchanged. Under
  `jax.jit`, or `jax.grad` with respect to `Lambda`, steps are not split, and
  a `JordanLambda` gives NaN for `omega * dt > 12` or `|rate| * dt > 5`.
- Reverse-mode gradients of `GammaKernel` coefficients were NaN for
  `rate * (t - s)` above about 37.4 in float64 and above 18.93, the whole
  quadrature range, in float32.

## 0.1.1

### Fixed

- `GammaKernel` coefficients use the unnormalised basis weight
  `(u - s)**rho`, as `FractionalKernel` and the interpolation basis do.
  Order-1 and order-2 `vsig` and `vsig_fft` results for Gamma kernels were
  wrong before.
- `GammaKernel` coefficients are summed from a series of `FractionalKernel`
  coefficients, exact to rounding for `rate * (t - s) <= 13.8` in float64
  (18.9 in float32). They were computed by Gauss-Legendre quadrature, with
  up to 8% relative error for small `beta`. Larger `rate * (t - s)` use a
  quadrature with `quad_order` nodes, whose default is now 64 (was 32).
  `jax.grad` is zero instead of NaN for time triples `(s, t, tau)` outside
  `s < t <= tau`, whose coefficients are zero.
- `jax.grad` with respect to `beta` works through `FractionalKernel`,
  `GammaKernel`, `vsig` and `vsig_fft` for `order` 0, 1 and 2, including
  the dependence of the order-1 and order-2 interpolation bases on `beta`.
  The new `tensordev.util.special.betainc`, the regularized incomplete beta
  function `I_x(a, b)`, is differentiable in `a`, `b` and `x`.
- The alpha wordwise kernels (`execution="wordwise"`) failed to compile on
  NVIDIA GPUs with JAX 0.10. They lower for CUDA with JAX 0.10 and 0.11 and
  run on an NVIDIA A40 with JAX 0.10, with unchanged results.
- `DenseLambda` computes `expm`, `phi1`, `phi2` and the shifted solves from
  `matrix` in every call instead of from an eigendecomposition stored at
  construction, so values and gradients follow the current `matrix`, also
  under JAX transformations.
- `phi1` and `phi2` support reverse-mode and higher derivatives and stay
  accurate for `dt * Lambda` near zero; the derivative of the
  `JordanLambda` exponential at `dt = 0` is correct.
- `dt` keeps the precision of the requested `dtype`, also when the `Lambda`
  parameters are `float32`. A `DenseLambda` and a `JordanLambda` with
  oscillatory blocks can be built inside `jax.jit`.
- `float32` results on NVIDIA Ampere and newer GPUs kept only about three
  significant digits (state-space signatures up to 2% relative error),
  because XLA runs `float32` matrix products in TF32 by default. All matrix
  products in tensordev now request full `float32` precision. CPU results are
  unchanged.

### Documentation

- Documentation at <https://hagerpa.github.io/tensordev/>: guides for tensor
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
