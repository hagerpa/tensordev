r"""Cellwise Euler reference for the general fractional Volterra scheme.

This module implements a numerical reference for the *approximative* general
Volterra-signature algorithm.

The important point is that dyadic refinement is only used inside each coarse
cell. Across coarse cells, the past Volterra signature is frozen at the
left endpoint of the cell, matching the structural approximation used by the
general quadratic algorithm.

For q=1 and fractional kernel

    K_beta(t, s) = (t - s) ** (beta - 1) / Gamma(beta),

the implemented scheme is

    V_j = 1 + sum_{i=0}^j V_i_left tensor E_{i,j}^{cell},

where ``V_i_left`` is the already computed coarse-grid history value at the
left endpoint of cell i, and ``E_{i,j}^{cell}`` is approximated by an Euler
scheme on a dyadic refinement of the single cell [t_i, t_{i+1}].

As ``dyadic_order`` increases, this should converge to the exact local
cell contribution used by ``volterra_vsig`` on the same coarse grid.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammaln

from tensordev.core.jax import Jax
from tensordev.core.universal import DenseElem

Array = jax.Array

_CORE = Jax()


@partial(
    jax.jit,
    static_argnames=(
        "beta",
        "trunc",
        "axis",
        "increment_input",
        "output_starting_point",
        "dyadic_order",
    ),
)
def fractional_cell_euler_vsig(
    X: Array,
    *,
    beta: float,
    A: Array,
    dt: Array | float,
    trunc: int,
    axis: int = -2,
    increment_input: bool = False,
    output_starting_point: bool = False,
    dyadic_order: int = 0,
) -> DenseElem:
    """Compute the cellwise Euler approximation of the fractional VSig scheme.

    Parameters
    ----------
    X:
        Path nodes or increments. The time axis is selected by ``axis`` and the
        trailing axis is the path dimension ``d``.
    beta:
        Fractional parameter in

            K_beta(t, s) = (t - s) ** (beta - 1) / Gamma(beta).

    A:
        Projection matrix with shape ``(1, m, d)`` or ``(m, d)``. Only q=1 is
        supported.
    dt:
        Scalar step size or one-dimensional array of coarse step sizes.
    trunc:
        Tensor truncation level.
    axis:
        Time axis of ``X``.
    increment_input:
        Set to ``True`` when ``X`` already contains increments.
    output_starting_point:
        If ``False``, return only the terminal tensor element. If ``True``,
        return the full coarse-grid trajectory including the initial unit.
    dyadic_order:
        Each coarse cell is split into ``2**dyadic_order`` local Euler substeps.

    Returns
    -------
    DenseElem
        Terminal or full-trajectory dense tensor element.
    """
    _validate_static_args(beta=beta, trunc=trunc, dyadic_order=dyadic_order)

    X = jnp.asarray(X)
    A = _normalize_A(A)

    if X.ndim < 2:
        raise ValueError("X must have at least a time axis and a trailing path dimension.")

    axis_norm = axis % X.ndim
    if axis_norm == X.ndim - 1:
        raise ValueError("axis must identify the time axis, not the trailing path dimension.")

    if X.shape[-1] != A.shape[-1]:
        raise ValueError(
            f"X trailing dimension must match A.shape[-1]={A.shape[-1]}, "
            f"got {X.shape[-1]}."
        )

    dX = X if increment_input else jnp.diff(X, axis=axis_norm)
    dX_time = jnp.moveaxis(dX, axis_norm, 0)

    # q=1 projection: dY[..., a] = sum_d A[0, a, d] dX[..., d].
    dY_time = jnp.einsum("md,...d->...m", A[0].astype(dX_time.dtype), dX_time)

    out = _fractional_cell_euler_time_first(
        dY_time,
        beta=beta,
        dt=dt,
        trunc=trunc,
        output_starting_point=output_starting_point,
        dyadic_order=dyadic_order,
    )

    if output_starting_point:
        return tuple(jnp.moveaxis(level, 0, axis_norm) for level in out)

    return out


@partial(
    jax.jit,
    static_argnames=(
        "beta",
        "trunc",
        "axis",
        "output_starting_point",
        "dyadic_order",
    ),
)
def fractional_cell_euler_vsig_from_increments(
    dY: Array,
    *,
    beta: float,
    dt: Array | float,
    trunc: int,
    axis: int = 0,
    output_starting_point: bool = False,
    dyadic_order: int = 0,
) -> DenseElem:
    """Cellwise Euler scheme from already projected q=1 increments.

    Parameters
    ----------
    dY:
        Projected increments with trailing shape ``(m,)``. The time axis is
        selected by ``axis``.
    beta:
        Fractional parameter.
    dt:
        Scalar step size or one-dimensional array of coarse step sizes.
    trunc:
        Tensor truncation level.
    axis:
        Time axis of ``dY``.
    output_starting_point:
        If ``False``, return only the terminal tensor element. If ``True``,
        return the full coarse-grid trajectory including the initial unit.
    dyadic_order:
        Each coarse cell is split into ``2**dyadic_order`` local Euler substeps.

    Returns
    -------
    DenseElem
        Terminal or full-trajectory dense tensor element.
    """
    _validate_static_args(beta=beta, trunc=trunc, dyadic_order=dyadic_order)

    dY = jnp.asarray(dY)
    if dY.ndim < 2:
        raise ValueError("dY must have a time axis and a trailing tensor dimension.")

    axis_norm = axis % dY.ndim
    if axis_norm == dY.ndim - 1:
        raise ValueError("axis must identify the time axis, not the trailing tensor dimension.")

    dY_time = jnp.moveaxis(dY, axis_norm, 0)

    out = _fractional_cell_euler_time_first(
        dY_time,
        beta=beta,
        dt=dt,
        trunc=trunc,
        output_starting_point=output_starting_point,
        dyadic_order=dyadic_order,
    )

    if output_starting_point:
        return tuple(jnp.moveaxis(level, 0, axis_norm) for level in out)

    return out


def _fractional_cell_euler_time_first(
    dY_time: Array,
    *,
    beta: float,
    dt: Array | float,
    trunc: int,
    output_starting_point: bool,
    dyadic_order: int,
) -> DenseElem:
    """Core implementation.

    ``dY_time`` has shape ``(J, batch..., m)``.
    """
    if dY_time.ndim < 2:
        raise ValueError("dY_time must have shape (J, ..., m).")

    J = int(dY_time.shape[0])
    if J == 0:
        raise ValueError("at least one increment is required.")

    dtype = dY_time.dtype
    batch_shape = dY_time.shape[1:-1]
    m = int(dY_time.shape[-1])

    dt_time = _normalize_dt(dt, J=J, dtype=dtype)
    times = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=dtype),
            jnp.cumsum(dt_time, axis=0),
        ],
        axis=0,
    )

    unit = _unit(batch_shape=batch_shape, m=m, trunc=trunc, dtype=dtype)

    history0 = tuple(
        jnp.zeros((J + 1,) + batch_shape + (m**level,), dtype=dtype)
        for level in range(trunc + 1)
    )
    history0 = tuple(
        level_arr.at[0].set(unit[level])
        for level, level_arr in enumerate(history0)
    )

    # Precompute the local diagonal histories for each coarse cell once.
    # Each level has shape (J, L + 1, batch..., m**level).
    cell_histories = jax.vmap(
        lambda dy, a, b: _local_cell_history(
            dy,
            a=a,
            b=b,
            beta=beta,
            trunc=trunc,
            dyadic_order=dyadic_order,
        ),
        in_axes=(0, 0, 0),
        out_axes=0,
    )(dY_time, times[:-1], times[1:])

    source_idx = jnp.arange(J)

    def outer_step(history, j):
        tau = times[j + 1]

        # history[i] is the left-point coarse history for cell i.
        v_left = tuple(level[:-1] for level in history)

        def source_contribution(i):
            valid = i <= j
            a = times[i]
            b = times[i + 1]
            tau_safe = jnp.where(valid, tau, b)

            cell_history_i = tuple(level[i] for level in cell_histories)

            E_i = _local_cell_readout(
                cell_history_i,
                dY_time[i],
                a=a,
                b=b,
                tau=tau_safe,
                beta=beta,
                trunc=trunc,
                dyadic_order=dyadic_order,
            )
            return _mask(E_i, valid)

        # Tuple of arrays, each with leading source axis J.
        E_all = jax.vmap(source_contribution, in_axes=0, out_axes=0)(source_idx)

        terms = _CORE.tensor_product(v_left, E_all, trunc=trunc)
        summed = _sum_axis0(terms)
        new_v = _add(unit, summed)

        history = tuple(
            level_arr.at[j + 1].set(new_v[level])
            for level, level_arr in enumerate(history)
        )
        return history, new_v

    history_final, path = lax.scan(outer_step, history0, jnp.arange(J))

    if output_starting_point:
        return history_final

    return tuple(level[-1] for level in history_final)


def _local_cell_history(
    dY: Array,
    *,
    a: Array,
    b: Array,
    beta: float,
    trunc: int,
    dyadic_order: int,
) -> DenseElem:
    """Local diagonal Euler history inside one coarse cell.

    Returns local values on the dyadic subgrid of a single coarse cell. The
    output levels have shape ``(L + 1, batch..., m**level)``.
    """
    L = 1 << int(dyadic_order)

    dtype = dY.dtype
    batch_shape = dY.shape[:-1]
    m = int(dY.shape[-1])

    unit = _unit(batch_shape=batch_shape, m=m, trunc=trunc, dtype=dtype)

    history0 = tuple(
        jnp.zeros((L + 1,) + batch_shape + (m**level,), dtype=dtype)
        for level in range(trunc + 1)
    )
    history0 = tuple(
        level_arr.at[0].set(unit[level])
        for level, level_arr in enumerate(history0)
    )

    h = (b - a) / jnp.asarray(L, dtype=dtype)
    dy = dY / jnp.asarray(L, dtype=dtype)
    edges = a + h * jnp.arange(L + 1, dtype=dtype)

    sub_idx = jnp.arange(L)

    def target_step(history, k):
        target = edges[k + 1]

        def source_step(acc, ell):
            valid = ell <= k

            z = _weighted_increment(
                dy,
                beta=beta,
                left=edges[ell],
                right=edges[ell + 1],
                target=target,
            )
            v_left = tuple(level[ell] for level in history)
            term = _right_multiply_level1(v_left, z, trunc=trunc)
            term = _mask(term, valid)
            return _add(acc, term), None

        zero = _zero(batch_shape=batch_shape, m=m, trunc=trunc, dtype=dtype)
        summed, _ = lax.scan(source_step, zero, sub_idx)
        new_v = _add(unit, summed)

        history = tuple(
            level_arr.at[k + 1].set(new_v[level])
            for level, level_arr in enumerate(history)
        )
        return history, new_v

    history_final, _ = lax.scan(target_step, history0, sub_idx)
    return history_final


def _local_cell_readout(
    history: DenseElem,
    dY: Array,
    *,
    a: Array,
    b: Array,
    tau: Array,
    beta: float,
    trunc: int,
    dyadic_order: int,
) -> DenseElem:
    """Read out the local positive-degree cell contribution at time ``tau``."""
    L = 1 << int(dyadic_order)

    dtype = dY.dtype
    batch_shape = dY.shape[:-1]
    m = int(dY.shape[-1])

    h = (b - a) / jnp.asarray(L, dtype=dtype)
    dy = dY / jnp.asarray(L, dtype=dtype)
    edges = a + h * jnp.arange(L + 1, dtype=dtype)

    zero = _zero(batch_shape=batch_shape, m=m, trunc=trunc, dtype=dtype)
    sub_idx = jnp.arange(L)

    def source_step(acc, ell):
        z = _weighted_increment(
            dy,
            beta=beta,
            left=edges[ell],
            right=edges[ell + 1],
            target=tau,
        )
        v_left = tuple(level[ell] for level in history)
        term = _right_multiply_level1(v_left, z, trunc=trunc)
        return _add(acc, term), None

    out, _ = lax.scan(source_step, zero, sub_idx)
    return out


def _weighted_increment(
    dy: Array,
    *,
    beta: float,
    left: Array,
    right: Array,
    target: Array,
) -> Array:
    """Return dy times the average fractional kernel over [left, right]."""
    dtype = dy.dtype

    beta_arr = jnp.asarray(beta, dtype=dtype)
    gamma_beta = jnp.exp(gammaln(beta_arr))

    h = right - left
    A = target - left
    B = jnp.maximum(target - right, jnp.asarray(0.0, dtype=dtype))

    integral = (A**beta_arr - B**beta_arr) / (beta_arr * gamma_beta)
    avg_kernel = integral / h
    return avg_kernel * dy


def _right_multiply_level1(
    v: DenseElem,
    z: Array,
    *,
    trunc: int,
) -> DenseElem:
    """Return ``v tensor z`` where ``z`` is degree-one, with zero level 0."""
    out = [jnp.zeros_like(v[0])]

    for level in range(1, trunc + 1):
        out.append(_CORE.tensor_product_homogeneous(v[level - 1], z))

    return tuple(out)


def _unit(
    *,
    batch_shape: tuple[int, ...],
    m: int,
    trunc: int,
    dtype: jnp.dtype,
) -> DenseElem:
    return tuple(
        jnp.ones(batch_shape + (1,), dtype=dtype)
        if level == 0
        else jnp.zeros(batch_shape + (m**level,), dtype=dtype)
        for level in range(trunc + 1)
    )


def _zero(
    *,
    batch_shape: tuple[int, ...],
    m: int,
    trunc: int,
    dtype: jnp.dtype,
) -> DenseElem:
    return tuple(
        jnp.zeros(batch_shape + (m**level,), dtype=dtype)
        for level in range(trunc + 1)
    )


def _add(a: DenseElem, b: DenseElem) -> DenseElem:
    return tuple(x + y for x, y in zip(a, b))


def _sum_axis0(a: DenseElem) -> DenseElem:
    return tuple(jnp.sum(level, axis=0) for level in a)


def _mask(a: DenseElem, valid: Array) -> DenseElem:
    return tuple(jnp.where(valid, level, jnp.zeros_like(level)) for level in a)


def _normalize_dt(dt: Array | float, *, J: int, dtype: jnp.dtype) -> Array:
    dt_arr = jnp.asarray(dt, dtype=dtype)

    if dt_arr.ndim == 0:
        return jnp.full((J,), dt_arr, dtype=dtype)

    if dt_arr.ndim == 1:
        if dt_arr.shape[0] not in (1, J):
            raise ValueError(f"1D dt must have length 1 or J={J}, got {dt_arr.shape[0]}.")
        return jnp.broadcast_to(dt_arr, (J,)).astype(dtype)

    raise ValueError("dt must be a scalar or a one-dimensional array of step sizes.")


def _normalize_A(A: Array) -> Array:
    A = jnp.asarray(A)

    if A.ndim == 2:
        return A[None, :, :]

    if A.ndim == 3 and A.shape[0] == 1:
        return A

    raise ValueError(
        "Only q=1 projections are supported. "
        "A must have shape (m, d) or (1, m, d)."
    )


def _validate_static_args(
    *,
    beta: float,
    trunc: int,
    dyadic_order: int,
) -> None:
    if float(beta) <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}.")
    if int(trunc) <= 0:
        raise ValueError(f"trunc must be positive, got {trunc}.")
    if int(dyadic_order) < 0:
        raise ValueError(f"dyadic_order must be nonnegative, got {dyadic_order}.")


__all__ = [
    "fractional_cell_euler_vsig",
    "fractional_cell_euler_vsig_from_increments",
]