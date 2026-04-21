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
    from tensordev.kernel.fssk import FSSKSigKernel

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


def pmap_solver_call(solver, gamma, dt_x, dt_y, *, lambda_op, transport_params, num_devices: int):
    """Evaluate *solver* with the batch dimension distributed across devices.

    Parameters
    ----------
    solver : callable
        A compiled solver function with signature
        ``(gamma, dt_x, dt_y, *, lambda_op, transport_params) -> tuple``.
    gamma : Array, shape ``(batch, ...)``
        The combined gamma grid.  Axis 0 is the batch dimension to shard.
    dt_x, dt_y : Array
        Time-step arrays (replicated to all devices).
    lambda_op :
        Lambda operator (replicated via closure — must be a JAX pytree or have
        only JAX-traced operations called on it during compilation).
    transport_params :
        Pre-computed propagators (replicated via closure).
    num_devices : int
        Number of virtual CPU (or GPU) devices to distribute across.
        Must be ≤ ``jax.device_count()``.

    Returns
    -------
    tuple
        Same structure as ``solver(...)``.  Batch axis 0 is reassembled.
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
        # Nothing to shard — call the solver directly.
        return solver(gamma, dt_x, dt_y, lambda_op=lambda_op, transport_params=transport_params)

    batch = gamma.shape[0]
    # Pad so batch is divisible by num_devices
    pad = (-batch) % num_devices
    if pad:
        gamma = jnp.concatenate([gamma, gamma[:pad]], axis=0)

    local_batch = gamma.shape[0] // num_devices
    gamma_sharded_arr = gamma.reshape(num_devices, local_batch, *gamma.shape[1:])

    # Scatter shards to individual devices.
    # In JAX >= 0.4: pmap distributes the leading axis across devices
    # automatically — no explicit device_put_sharded needed.
    gamma_per_device = gamma_sharded_arr

    # dt_x, dt_y, lambda_op and transport_params are captured as constants in
    # the closure and automatically broadcast/replicated by pmap.
    def _local(g_shard):
        return solver(
            g_shard, dt_x, dt_y,
            lambda_op=lambda_op,
            transport_params=transport_params,
        )

    result = jax.pmap(_local, devices=devices)(gamma_per_device)

    # Merge device axis back into batch and strip padding
    def _flatten_device_axis(x):
        # x.shape == (num_devices, local_batch, ...)
        return x.reshape(-1, *x.shape[2:])[:batch]

    return jax.tree_util.tree_map(_flatten_device_axis, result)


__all__ = [
    "cpu_device_count",
    "physical_cpu_count",
    "recommend_parallelism",
    "pmap_solver_call",
]





