---
hide:
  - navigation
  - toc
---

<div class="td-hero" markdown>

<div class="td-title" markdown>

![tensordev logo](assets/logo.svg){ .td-logo }

# tensordev

</div>

--8<-- "README.md:tagline"

[Get started](#installation){ .md-button .md-button--primary }
[API reference](api.md){ .md-button }
[GitHub](https://github.com/hagerpa/tensordev){ .md-button }

</div>

<div class="grid cards" markdown>

-   :material-family-tree:{ .lg .middle } __Configurable cores__

    ---

    One set of tensor operations for dense total-degree, bidegree, partially symmetrized and
    shear cores. Build a core with `make_core`, or install it as the default with
    `set_default_core`.

-   :material-vector-polyline:{ .lg .middle } __Signatures and free developments__

    ---

    Truncated path signatures, with optional blocking through Chen's identity, and free
    developments of tensor-valued paths.

-   :material-math-integral-box:{ .lg .middle } __Volterra and state-space signatures__

    ---

    Volterra signatures for fractional, gamma and finite state-space kernels. State-space
    signatures carry a hidden state for streaming evaluation.

-   :material-chart-bell-curve:{ .lg .middle } __Signature kernels__

    ---

    Classical, free, higher-order and FSSK signature kernels share one API for kernel values,
    Gram matrices, MMD and scoring rules.

-   :material-function-variant:{ .lg .middle } __Built on JAX__

    ---

    JIT-compiled cores, and device placement follows JAX. With the default execution policy,
    every development and kernel is differentiable with respect to the paths and the kernel
    matrices `A` and `b`. Other kernel parameters have limits: see
    [Volterra signatures](volterra.md#gradients) and [Signature kernels](kernels.md#gradients).

-   :material-chip:{ .lg .middle } __Native CPU kernels, alpha GPU path__

    ---

    Linux (x86_64, aarch64) and macOS Apple Silicon wheels include native CPU kernels for
    eligible partially symmetrized signatures. Wordwise execution on NVIDIA GPUs is alpha and not
    yet validated on GPU.

</div>

--8<-- "README.md:install"

--8<-- "README.md:quickstart"

Continue with [Tensor algebra](tensor-algebra.md), [Signatures](signatures.md),
[Volterra signatures](volterra.md), [State-space signatures](state-space.md),
[Signature kernels](kernels.md), [Performance](performance.md) and [Background](background.md).
