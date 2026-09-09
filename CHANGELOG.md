# Changelog

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
