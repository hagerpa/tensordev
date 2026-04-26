"""
CPU parallelism utilities for JAX-based kernel evaluation.

Overview
--------
JAX on CPU exposes a single device by default.  To use all physical cores for
**batch-level** parallelism (i.e. computing different (X_i, Y_j) pairs on
different cores simultaneously) you need two things:

1. **Before importing JAX**, tell XLA to create N virtual CPU devices::

       import os
       os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
       import jax  # now jax.device_count() == 8

   Alternatively set the variable in your shell before running your script::

       XLA_FLAGS="--xla_force_host_platform_device_count=$(sysctl -n hw.logicalcpu)" python train.py

2. **Pass ``num_devices=N``** to ``fssk_sigkernel`` / ``FSSKSigKernel``.  The
   batch dimension of gamma (or dx/dy) will be split across those N virtual
   devices and evaluated in parallel via ``jax.pmap``.

Two axes of parallelism
-----------------------
``jax.pmap`` (device-level)
    Each virtual device handles ``batch / num_devices`` pairs.  Different pairs
    are fully independent so this scales perfectly.  This is controlled by the
    ``num_devices`` argument.

XLA BLAS threading (intra-op)
    Within each device, matrix multiplications already use all CPU threads via
    Eigen/OpenBLAS.  Setting ``XLA_FLAGS`` with
    ``--xla_force_host_platform_device_count=N`` divides the thread pool across
    N virtual devices, so there is a trade-off: more devices = more batch
    parallelism but fewer BLAS threads per device.

    Rule of thumb: if your batch is large relative to the path/matrix size, use
    more devices.  If you have a small batch but large matrices, fewer devices
    (or 1) may be faster.

Quick start
-----------
::

    import os
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

    import jax
    from tensordev.kernel.sss import FSSKSigKernel

    kernel = FSSKSigKernel(..., num_devices=8)
    gram = kernel.compute_Gram(X, Y)

"""

from __future__ import annotations

import os
import warnings
import jax
import jax.numpy as jnp


def cpu_device_count() -> int:
    """Return the number of JAX CPU devices currently visible.

    If this returns 1, set ``XLA_FLAGS`` *before* importing JAX to expose more
    virtual CPU devices (see module docstring).
    """
    return jax.device_count()


def physical_cpu_count() -> int:
    """Return the number of *physical* logical CPU cores on this machine."""
    return os.cpu_count() or 1


def recommend_parallelism() -> str:
    """Return a human-readable recommendation string for CPU parallelism setup."""
    n_phys = physical_cpu_count()
    n_jax = cpu_device_count()
    lines = [
        f"Physical CPUs : {n_phys}",
        f"JAX devices   : {n_jax}",
    ]
    if n_jax < n_phys:
        lines += [
            "",
            "To use all CPU cores, set this environment variable BEFORE importing JAX:",
            f'    export XLA_FLAGS="--xla_force_host_platform_device_count={n_phys}"',
            "",
            "Or in Python (must come before any jax import):",
            "    import os",
            f'    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={n_phys}"',
            "    import jax",
            "",
            f"Then pass num_devices={n_phys} to fssk_sigkernel / FSSKSigKernel.",
        ]
    else:
        lines += [
            "",
            f"All {n_phys} CPUs are already exposed as JAX devices.",
            f"Pass num_devices={n_jax} to fssk_sigkernel / FSSKSigKernel.",
        ]
    return "\n".join(lines)


def pmap_batch(fn, *sharded_pytrees, num_devices: int):
    """Evaluate *fn* with all positional pytree arguments sharded along axis 0.

    This is the core building block for device-level batch parallelism.  All
    leaves in every ``sharded_pytrees`` argument must share the same size on
    axis 0 (the batch dimension).  That axis is split across ``num_devices``
    virtual JAX devices and the results are reassembled transparently.

    Parameters
    ----------
    fn : callable
        A function whose positional arguments match ``sharded_pytrees`` in
        number and structure.  Any additional state (e.g. non-batched arrays,
        Python objects) should be captured via a closure — pmap replicates
        closed-over JAX arrays automatically.
    *sharded_pytrees :
        One or more JAX pytrees (arrays, tuples/lists of arrays, …) whose
        **axis 0** carries the batch dimension to shard.  Tuples of arrays
        (as used by ``free_kernel`` for multi-level inputs) are fully
        supported.
    num_devices : int
        Number of virtual JAX devices to distribute across.  Clamped to
        ``jax.device_count()`` with a ``RuntimeWarning`` if too large.

    Returns
    -------
    Same pytree structure as ``fn(*sharded_pytrees)``.  Axis 0 is the
    reassembled batch dimension.

    Examples
    --------
    Single-array shard (sss style)::

        result = pmap_batch(lambda g: solver(g, dt_x, dt_y, ...), gamma, num_devices=4)

    Multi-level tuple shard (free_kernel style)::

        result = pmap_batch(solve_fn, dx_tuple, dy_tuple, num_devices=4)
    """
    devices = jax.devices()[:num_devices]
    if len(devices) < num_devices:
        warnings.warn(
            f"num_devices={num_devices} requested but JAX only sees {len(devices)} device(s). "
            f"Falling back to num_devices={len(devices)}. "
            "To expose all CPU cores set XLA_FLAGS before importing JAX:\n"
            f"    export XLA_FLAGS=\"--xla_force_host_platform_device_count={num_devices}\"\n"
            "See tensordev.kernel.parallel for details.",
            RuntimeWarning,
            stacklevel=2,
        )
        num_devices = len(devices)

    if num_devices <= 1:
        return fn(*sharded_pytrees)

    # Infer batch size from the first leaf of the first pytree argument.
    first_leaf = jax.tree_util.tree_leaves(sharded_pytrees[0])[0]
    batch = int(first_leaf.shape[0])

    pad = (-batch) % num_devices
    local_batch = (batch + pad) // num_devices

    def _pad(x):
        return jnp.concatenate([x, x[:pad]], axis=0) if pad else x

    def _shard(x):
        return x.reshape(num_devices, local_batch, *x.shape[1:])

    sharded = tuple(
        jax.tree_util.tree_map(lambda x: _shard(_pad(x)), pt)
        for pt in sharded_pytrees
    )

    result = jax.pmap(fn, devices=devices)(*sharded)

    def _flatten(x):
        # x.shape == (num_devices, local_batch, ...)
        return x.reshape(-1, *x.shape[2:])[:batch]

    return jax.tree_util.tree_map(_flatten, result)


def pmap_solver_call(solver, gamma, dt_x, dt_y, *, lambda_op, transport_params, num_devices: int):
    """Evaluate *solver* with the batch dimension distributed across devices.

    Thin wrapper around :func:`pmap_batch` specialised for the FSSK solver
    interface ``(gamma, dt_x, dt_y, *, lambda_op, transport_params) -> tuple``.

    Parameters
    ----------
    solver : callable
        Compiled solver returned by ``_get_solver``.
    gamma : Array, shape ``(batch, ...)``
        The combined gamma grid.  Axis 0 is the batch dimension to shard.
    dt_x, dt_y : Array
        Time-step arrays (replicated to all devices via closure).
    lambda_op :
        Lambda operator (replicated via closure).
    transport_params :
        Pre-computed propagators (replicated via closure).
    num_devices : int
        Number of virtual CPU (or GPU) devices to distribute across.

    Returns
    -------
    tuple
        Same structure as ``solver(...)``.  Batch axis 0 is reassembled.
    """
    def _fn(g_shard):
        return solver(
            g_shard, dt_x, dt_y,
            lambda_op=lambda_op,
            transport_params=transport_params,
        )

    return pmap_batch(_fn, gamma, num_devices=num_devices)


__all__ = [
    "cpu_device_count",
    "physical_cpu_count",
    "recommend_parallelism",
    "pmap_batch",
    "pmap_solver_call",
]







