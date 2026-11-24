# Changelog

## Unreleased

### Changed

- The alpha wordwise execution (`execution="wordwise"`) is one kernel for
  ordinary and `q = 1` FSSK signatures on every core type, in place of one
  kernel for each combination of signature or FSSK state and ordered or
  partially symmetrized coordinates. It computes all levels in one launch,
  two for an ordered bidegree core, and its results agree with the previous
  kernels to rounding. An FSSK step of more than `2**16` tile operations is
  refused: for two letters a state dimension above 76 at truncation 4 and
  above 28 at truncation 12, where the previous kernels accepted state
  dimensions up to 128. On an NVIDIA A40 (200 steps, batch 1 and 32, 141
  cases) the median speedup over the previous kernels is 5.9: partially
  symmetrized FSSK states with dims `(4, 2)`, truncation `(4, 2)` and state
  dimension 3 take 6.0 ms instead of 3.0 s in float32 at batch 32, float64
  signatures of two letters at truncation 12 take 2.4 ms instead of 98 ms.
  Compile times are at most 51 s instead of 163 s. Calls below 0.6 ms take
  up to 0.14 ms longer, and float64 FSSK states of dimension 8 at truncation
  4 take 6.0 ms instead of 4.2 ms.

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
- With `dyadic_order > 0` and tensor-valued inputs, `free_kernel`,
  `FreeKernel` and `HigherOrderKernel` built the boundary developments from
  the unrefined paths, so the refined solution converged only at first order
  and at low orders could be less accurate than no refinement. The boundary
  developments now follow the refined paths, and `dyadic_order` gives the
  result of splitting every interval into `2**dyadic_order` equal parts.
- `jax.grad` of `vsig` with respect to `dt` was NaN for kernels with
  `beta < 1`; it now matches finite differences for every scheme.
- `FractionalKernel` coefficient gradients at `tau == t` were NaN or
  infinite. As for `GammaKernel`, they are now finite there, and their sum in
  `t` and `tau` is the exact derivative along `tau == t`.
- Kernel parameters can be traced under `jax.jit`: the constructors of
  `FractionalKernel`, `GammaKernel` and `FSSKConvolutionKernel` check only
  concrete values. For `order > 0` with the `"fft"` or `"quadratic"` scheme,
  a traced `beta` raises a `ValueError` that says it must be concrete.
- `path_signature` and the free developments raised a `ValueError` for NumPy
  increments (`increment_input=True`) with NumPy 2, whose arrays carry a
  `device` attribute that is not a JAX device.

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
