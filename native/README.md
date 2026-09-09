# TensorDev native CPU kernels

TensorDev's Linux x86_64/aarch64 (glibc 2.28+) and macOS Apple Silicon wheels
include CPU XLA FFI kernels. Eligible partially symmetrized signature steps
use them automatically, while unsupported workloads retain the JAX implementation.
The current fused Horner path targets sufficiently large `float32`/`float64`
workloads in standard coordinates with `partially_symmetrized=True`
and `dims=(1, q)`, `q > 1`.

The package supports Linux and macOS. Its C++17 library uses JAX's typed XLA
FFI headers and the XLA CPU thread pool. It does not use OpenMP, fast-math, or
host-specific instruction flags.

The ragged Horner handler derives its grade count and layout from metadata, so
its compiled interface is independent of the truncation.

## Source builds

Source installations and the universal wheel use portable JAX without a C++
compiler. To include native kernels when installing from the repository root,
use:

```bash
TENSORDEV_BUILD_NATIVE=1 python -m pip install .
```

This requires a C++17 compiler. The isolated build environment obtains CMake
and the JAX 0.10.0 headers used for the supported JAX 0.10--0.11 runtime range.
Native build failures are reported rather than silently producing a pure wheel.

Build a platform wheel from the repository root with:

```bash
TENSORDEV_BUILD_NATIVE=1 python -m pip wheel . --wheel-dir dist --no-deps
```

For a distributable macOS wheel, set the platform tag to the same deployment
target used by CMake:

```bash
TENSORDEV_BUILD_NATIVE=1 MACOSX_DEPLOYMENT_TARGET=11.0 \
  python -m pip wheel . --wheel-dir dist --no-deps
```

Without `TENSORDEV_BUILD_NATIVE`, `python -m build` produces the source
distribution and the compiler-free `py3-none-any` wheel. The source distribution
contains the native sources, so either wheel can be built from it.

The standalone companion remains source-installable with
`python -m pip install ./native`. It is unnecessary with a native-enabled
TensorDev wheel and is not published separately. Bundled kernels take precedence;
the two installations use separate module paths and do not overwrite each other.

## Binary interface

The bundled loader lives in the private `tensordev._native_cpu` module. It shares
its source with the standalone `tensordev_native_cpu` loader and retains the
shared-library handle for the lifetime of its FFI registrations. The library
does not use the CPython extension ABI, so one platform wheel serves all
supported Python versions.

The binary must be built against a JAX installation compatible with the JAX
version used at runtime. Release wheels are tested with both supported JAX
minor versions. Source, binary, and distribution metadata are licensed under
the repository's Apache-2.0 license.
