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
- Add source-installable native CPU companion code for partially symmetrized
  bidegree signature evaluation. The companion is not separately published as
  a package or wheel in this release.
- Add candidate NVIDIA wordwise executors and release benchmarks for ordinary
  and exact scalar-FSSK signatures. Public calls retain portable JAX execution
  until exact target and workload regions pass the release gates.

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
