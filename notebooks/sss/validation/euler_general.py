r"""Reference Euler scheme for finite-state-space Volterra signatures.

This module implements the explicit Euler discretisation of the tensor ODE
from Part I, Proposition ``mean_reverting_prop``:

    dZ^ell_t = - sum_k Lambda[ell, k] Z^k_t dt
               + (1 + sum_k Z^k_t) \otimes d(Bx)^ell_t,

where

    d(Bx)^ell_t = sum_{p=1}^n b[p, ell] A[p] dx_t.

It is intended as a transparent numerical-validation reference for the exact
Part II algorithms. It is general in n, R, m and truncation. The optional
``dyadic_order`` parameter linearly refines each input interval into
``2**dyadic_order`` Euler substeps by splitting both increments and time steps
uniformly.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from tensordev.core.universal import DenseElem, DenseElemFirstOn
from tensordev.sss.kernel import FSSK
from tensordev.sss.lambdas import Lambda

Array = jax.Array


@partial(
    jax.jit,
    static_argnames=(
        "trunc",
        "axis",
        "increment_input",
        "output_starting_state",
        "dyadic_order",
    ),
)
def fssk_euler_state(
    X: Array,
    *,
    kernel: FSSK,
    dt: Array | float,
    trunc: int,
    axis: int = -2,
    increment_input: bool = False,
    initial_state: DenseElemFirstOn | None = None,
    output_starting_state: bool = False,
    dyadic_order: int = 0,
) -> DenseElemFirstOn:
    """Compute the explicit-Euler hidden tensor state for an FSSK kernel.

    ``dyadic_order`` splits every original increment and time step into
    ``2**dyadic_order`` equal Euler subincrements/substeps.
    """
    if trunc <= 0:
        raise ValueError(f"trunc must be positive, got {trunc}.")
    if dyadic_order < 0:
        raise ValueError(f"dyadic_order must be nonnegative, got {dyadic_order}.")

    X = jnp.asarray(X)
    if X.ndim < 2:
        raise ValueError("X must have at least a time axis and a trailing path dimension.")

    axis_norm = axis % X.ndim
    if axis_norm == X.ndim - 1:
        raise ValueError("axis must identify the time axis, not the trailing path dimension.")
    if X.shape[-1] != kernel.path_dim:
        raise ValueError(
            f"X trailing dimension must be {kernel.path_dim}, got {X.shape[-1]}."
        )

    dtype = X.dtype
    dX = (X if increment_input else jnp.diff(X, axis=axis_norm)).astype(dtype)
    S = dX.shape[axis_norm]
    if S == 0:
        raise ValueError("fssk_euler_state requires at least one increment.")

    dt_time = _normalize_dt(dt, increment_shape=dX.shape, S=S, axis_norm=axis_norm)
    dt_time = dt_time.astype(dtype)
    dX_time = jnp.moveaxis(dX, axis_norm, 0)
    dX_time, dt_time = _dyadically_refine_increments(dX_time, dt_time, dyadic_order)

    dY_time = jnp.einsum(
        "qr,qmd,...d->...rm",
        kernel.b.astype(dtype),
        kernel.A.astype(dtype),
        dX_time,
    )

    return euler_state_from_latent_increments(
        dY_time,
        Lambda=kernel.Lambda,
        dt=dt_time,
        trunc=trunc,
        axis=0,
        initial_state=initial_state,
        output_starting_state=output_starting_state,
        dyadic_order=0,
    )


@partial(
    jax.jit,
    static_argnames=("trunc", "axis", "output_starting_state", "dyadic_order"),
)
def euler_state_from_latent_increments(
    dY: Array,
    *,
    Lambda: Lambda,
    dt: Array | float,
    trunc: int,
    axis: int = 0,
    initial_state: DenseElemFirstOn | None = None,
    output_starting_state: bool = False,
    dyadic_order: int = 0,
) -> DenseElemFirstOn:
    r"""Explicit Euler scheme from already projected latent increments.

    ``dY`` must have trailing shape ``(R, m)`` and represents

        dY[..., ell, :] = sum_p b[p, ell] A[p] dX.
    """
    if trunc <= 0:
        raise ValueError(f"trunc must be positive, got {trunc}.")
    if dyadic_order < 0:
        raise ValueError(f"dyadic_order must be nonnegative, got {dyadic_order}.")

    dY = jnp.asarray(dY)
    if dY.ndim < 3:
        raise ValueError("dY must have a time axis and trailing shape (R, m).")

    axis_norm = axis % dY.ndim
    if axis_norm >= dY.ndim - 2:
        raise ValueError("axis must identify the time axis, not one of the trailing (R, m) axes.")

    dtype = dY.dtype
    R = int(dY.shape[-2])
    m = int(dY.shape[-1])
    if R != Lambda.state_dim:
        raise ValueError(
            f"dY has R={R}, but Lambda.state_dim={Lambda.state_dim}."
        )

    dY_time = jnp.moveaxis(dY, axis_norm, 0).astype(dtype)
    S = dY_time.shape[0]
    if S == 0:
        raise ValueError("euler_state_from_latent_increments requires at least one increment.")

    dt_time = _normalize_dt_for_latent(dt, dY_shape=dY.shape, S=S, axis_norm=axis_norm)
    dt_time = dt_time.astype(dtype)
    dY_time, dt_time = _dyadically_refine_increments(dY_time, dt_time, dyadic_order)
    S_refined = dY_time.shape[0]

    init_batch = () if initial_state is None else tuple(initial_state)[0].shape[:-2]
    batch_shape = jnp.broadcast_shapes(dY_time.shape[1:-2], dt_time.shape[1:], init_batch)

    seed = _make_seed(
        initial_state,
        batch_shape=batch_shape,
        R=R,
        m=m,
        trunc=trunc,
        dtype=dtype,
    )

    dY_time = jnp.broadcast_to(dY_time, (S_refined,) + batch_shape + (R, m))
    dt_time = jnp.broadcast_to(dt_time, (S_refined,) + batch_shape)

    def step(Z: DenseElemFirstOn, xs) -> tuple[DenseElemFirstOn, DenseElemFirstOn]:
        dY_t, dt_t = xs
        Z_next = _euler_step(Z, dY_t, dt_t, Lambda=Lambda)
        return Z_next, Z_next

    final, states = lax.scan(step, seed, (dY_time, dt_time))

    if output_starting_state:
        return tuple(jnp.concatenate((z0[None], z), axis=0) for z0, z in zip(seed, states))
    return final


@jax.jit
def euler_readout(state: DenseElemFirstOn) -> DenseElem:
    r"""Read out ``VSig^t = 1 + sum_ell Z^ell_t`` from the Euler state."""
    Z = tuple(state)
    if not Z:
        raise ValueError("state must not be empty.")
    batch_shape = Z[0].shape[:-2]
    one = jnp.ones(batch_shape + (1,), dtype=Z[0].dtype)
    return (one,) + tuple(jnp.sum(z, axis=-2) for z in Z)


@partial(
    jax.jit,
    static_argnames=("trunc", "axis", "increment_input", "dyadic_order"),
)
def fssk_euler_vsig(
    X: Array,
    *,
    kernel: FSSK,
    dt: Array | float,
    trunc: int,
    axis: int = -2,
    increment_input: bool = False,
    initial_state: DenseElemFirstOn | None = None,
    dyadic_order: int = 0,
) -> DenseElem:
    """Convenience wrapper returning the terminal Euler Volterra signature."""
    state = fssk_euler_state(
        X,
        kernel=kernel,
        dt=dt,
        trunc=trunc,
        axis=axis,
        increment_input=increment_input,
        initial_state=initial_state,
        output_starting_state=False,
        dyadic_order=dyadic_order,
    )
    return euler_readout(state)


def _euler_step(
    Z: DenseElemFirstOn,
    dY: Array,
    dt: Array,
    *,
    Lambda: Lambda,
) -> DenseElemFirstOn:
    """One explicit Euler step for the mean-reverting tensor ODE."""
    Z = tuple(Z)
    batch_shape = Z[0].shape[:-2]
    dtype = Z[0].dtype

    W: list[Array] = [jnp.ones(batch_shape + (1,), dtype=dtype)]
    W.extend(jnp.sum(z, axis=-2) for z in Z)

    dt_fac = dt[..., None, None]
    out = []
    for k, z in enumerate(Z):
        drift = -Lambda.lambda_multiply_left(z, dtype=dtype)
        force = _append_latent_increment(W[k], dY)
        out.append(z + dt_fac * drift + force)
    return tuple(out)


def _append_latent_increment(prefix: Array, dY: Array) -> Array:
    r"""Return ``prefix \otimes dY^ell`` for all latent states ell."""
    prod = prefix[..., None, :, None] * dY[..., :, None, :]
    return prod.reshape(prod.shape[:-2] + (prod.shape[-2] * prod.shape[-1],))


def _dyadically_refine_increments(
    increments_time: Array,
    dt_time: Array,
    dyadic_order: int,
) -> tuple[Array, Array]:
    """Split each time-first increment/time-step into dyadic substeps."""
    if dyadic_order == 0:
        return increments_time, dt_time

    factor = 1 << dyadic_order
    inc_factor = jnp.asarray(factor, dtype=increments_time.dtype)
    dt_factor = jnp.asarray(factor, dtype=dt_time.dtype)
    increments_refined = jnp.repeat(increments_time / inc_factor, factor, axis=0)
    dt_refined = jnp.repeat(dt_time / dt_factor, factor, axis=0)
    return increments_refined, dt_refined


def _make_seed(
    initial_state: DenseElemFirstOn | None,
    *,
    batch_shape: tuple[int, ...],
    R: int,
    m: int,
    trunc: int,
    dtype: jnp.dtype,
) -> DenseElemFirstOn:
    tails = tuple((R, m ** k) for k in range(1, trunc + 1))
    if initial_state is None:
        return tuple(jnp.zeros(batch_shape + tail, dtype=dtype) for tail in tails)

    Z = tuple(initial_state)
    if len(Z) != trunc:
        raise ValueError(f"initial_state must have {trunc} levels, got {len(Z)}.")

    out = []
    for k, (z, tail) in enumerate(zip(Z, tails), start=1):
        z = jnp.asarray(z, dtype=dtype)
        if z.shape[-2:] != tail:
            raise ValueError(
                f"initial_state level {k} must have trailing shape {tail}, "
                f"got {z.shape[-2:]}"
            )
        out.append(jnp.broadcast_to(z, batch_shape + tail))
    return tuple(out)


def _normalize_dt(
    dt: Array | float,
    *,
    increment_shape: tuple[int, ...],
    S: int,
    axis_norm: int,
) -> Array:
    """Normalize dt to time-first shape matching path increment leading axes."""
    dt = jnp.asarray(dt)
    step_batch_shape = tuple(increment_shape[:-1])
    time_batch_shape = (
        (step_batch_shape[axis_norm],)
        + step_batch_shape[:axis_norm]
        + step_batch_shape[axis_norm + 1 :]
    )

    if dt.ndim == 0:
        return jnp.full(time_batch_shape, dt, dtype=dt.dtype)

    if dt.ndim == 1:
        if dt.shape[0] not in (1, S):
            raise ValueError(f"1D dt must have length 1 or S={S}, got {dt.shape[0]}.")
        dt_time = jnp.broadcast_to(dt, (S,))
        return jnp.broadcast_to(
            dt_time.reshape((S,) + (1,) * (len(time_batch_shape) - 1)),
            time_batch_shape,
        )

    if dt.ndim == len(increment_shape) - 1:
        dt_time = jnp.moveaxis(dt, axis_norm, 0)
        if dt_time.shape[0] not in (1, S):
            raise ValueError(
                f"dt time length must be 1 or S={S}, got {dt_time.shape[0]}."
            )
        return jnp.broadcast_to(dt_time, time_batch_shape)

    raise ValueError(
        "dt must be scalar, shape (1,), shape (S,), or match the batch/time "
        "axes of X without the trailing path-coordinate dimension."
    )


def _normalize_dt_for_latent(
    dt: Array | float,
    *,
    dY_shape: tuple[int, ...],
    S: int,
    axis_norm: int,
) -> Array:
    """Normalize dt for an array with trailing shape ``(R, m)``."""
    dt = jnp.asarray(dt)
    step_batch_shape = tuple(dY_shape[:-2])
    time_batch_shape = (
        (step_batch_shape[axis_norm],)
        + step_batch_shape[:axis_norm]
        + step_batch_shape[axis_norm + 1 :]
    )

    if dt.ndim == 0:
        return jnp.full(time_batch_shape, dt, dtype=dt.dtype)

    if dt.ndim == 1:
        if dt.shape[0] not in (1, S):
            raise ValueError(f"1D dt must have length 1 or S={S}, got {dt.shape[0]}.")
        dt_time = jnp.broadcast_to(dt, (S,))
        return jnp.broadcast_to(
            dt_time.reshape((S,) + (1,) * (len(time_batch_shape) - 1)),
            time_batch_shape,
        )

    if dt.ndim == len(dY_shape) - 2:
        dt_time = jnp.moveaxis(dt, axis_norm, 0)
        if dt_time.shape[0] not in (1, S):
            raise ValueError(
                f"dt time length must be 1 or S={S}, got {dt_time.shape[0]}."
            )
        return jnp.broadcast_to(dt_time, time_batch_shape)

    raise ValueError(
        "dt must be scalar, shape (1,), shape (S,), or match the batch/time "
        "axes of dY without the trailing (R, m) axes."
    )


__all__ = [
    "euler_readout",
    "euler_state_from_latent_increments",
    "fssk_euler_state",
    "fssk_euler_vsig",
]
