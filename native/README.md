# TensorDev native CPU kernels

`tensordev-native-cpu` is an optional platform wheel containing TensorDev's
CPU XLA FFI kernels. It is separate from the pure Python `tensordev` package;
eligible partially symmetrized signature steps use it automatically when it
is installed, while unsupported workloads retain the JAX implementation.
The current fused Horner path targets sufficiently large `float32`/`float64`
workloads in standard coordinates with `partially_symmetrized=True`
and `dims=(1, q)`, `q > 1`.

The package supports Linux and macOS. Its C++17 library uses JAX's typed XLA
FFI headers and the XLA CPU thread pool. It does not use OpenMP, fast-math, or
host-specific instruction flags.

The ragged Horner handler derives its grade count and layout from metadata, so
its compiled interface is independent of the truncation.

Install the companion directly from the repository root with:

```bash
python -m pip install ./native
```

Build the wheel from the repository root with:

```bash
python -m pip wheel ./native --wheel-dir dist --no-deps
```

For a distributable macOS wheel, set the platform tag to the same deployment
target used by CMake:

```bash
MACOSX_DEPLOYMENT_TARGET=11.0 \
  python -m pip wheel ./native --wheel-dir dist --no-deps
```

The Python package retains the loaded shared-library handle and exposes the
handler capsules without registering them:

```python
import tensordev_native_cpu

capsules = tensordev_native_cpu.registrations()
```

The companion must be built against a JAX installation compatible with the
JAX version used at runtime. Source, binary, and distribution metadata are
licensed under the repository's Apache-2.0 license.
