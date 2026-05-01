from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
from jax import lax

from tensordev.core.jax import Jax
from tensordev.core.universal import DenseElem
from tensordev.volterra.coeffs import VolterraCoefficients
from tensordev.volterra.kernel import VolterraKernel
from tensordev.volterra.eval_scalar import eval_vte as eval_vte_scalar
from tensordev.volterra.eval_general import eval_vte as eval_vte_general


Array = jax.Array

_CORE = Jax()


def vsig(
    X: Array,
    *,
    kernel: VolterraKernel,
    trunc: int,
    times: Array | None = None,
    dt: Array | float | None = None,
    axis: int = -2,
    output_starting_point: bool = False,
    increment_input: bool = False,
) -> DenseElem:
    r"""
    Compute a truncated Volterra signature from path nodes or increments.

    This is the general quadratic Volterra-Chen recursion under the coefficient
    symmetry hypothesis implemented by :class:`VolterraKernel`.  Unlike the SSS
    recursion there is no fixed-size hidden Markov state: internally we carry a
    padded history buffer ``[1, V_0, ..., V_{j-1}, 0, ...]`` so that the outer
    recursion is a :func:`jax.lax.scan` and each inner source-interval sum is a
    batched local ``eval_vte`` call.

    Parameters
    ----------
    X:
        Path nodes or increments.  The trailing axis is the path dimension
        ``kernel.path_dim``; ``axis`` is the step/node axis.  Set
        ``increment_input=True`` to skip :func:`jnp.diff`.
    kernel:
        Volterra kernel supplying projections and coefficient builders.
    trunc:
        Tensor truncation level (positive integer).
    times:
        Optional one-dimensional node times of shape ``(S + 1,)``.  Mutually
        exclusive with ``dt``.
    dt:
        Optional scalar uniform step size.  If both ``times`` and ``dt`` are
        omitted, ``dt=1`` is used.
    axis:
        Step/node axis of ``X``.
    output_starting_point:
        If ``False`` (default), return the terminal Volterra signature.  If
        ``True``, return the whole trajectory with the tensor unit prepended,
        i.e. ``[1, V_0, ..., V_{S-1}]`` with the trajectory axis at ``axis``.
    increment_input:
        Treat ``X`` as increments rather than path nodes.

    Returns
    -------
    DenseElem
        Terminal signature by default.  With ``output_starting_point=True``,
        each level carries an additional trajectory axis at ``axis``.
    """
    if trunc <= 0:
        raise ValueError(f"trunc must be positive, got {trunc}.")

    X = jnp.asarray(X)
    if X.ndim < 2:
        raise ValueError("X must have at least a step axis and a trailing path dimension.")

    axis_norm = axis % X.ndim
    if axis_norm == X.ndim - 1:
        raise ValueError("axis must identify the step axis, not the trailing path dimension.")
    if X.shape[-1] != kernel.path_dim:
        raise ValueError(
            f"X trailing dimension must be {kernel.path_dim}, got {X.shape[-1]}."
        )

    dtype = X.dtype
    dX = (X if increment_input else jnp.diff(X, axis=axis_norm)).astype(dtype)
    S = dX.shape[axis_norm]
    if S == 0:
        raise ValueError("volterra_vsig requires at least one increment.")

    projected = jnp.einsum("qmd,...d->...qm", kernel.A.astype(dtype), dX)
    y = projected[..., 0, :] if kernel.q == 1 else projected

    y_time = jnp.moveaxis(y, axis_norm, 0)
    y_time = _normalize_projected_y_time(y_time, kernel)
    times_arr = _normalize_times(times, dt, S=S, dtype=dtype)

    batch_shape = _projected_batch_shape(y_time, kernel)
    unit = _make_unit(batch_shape=batch_shape, m=kernel.m, trunc=trunc, dtype=dtype)
    history0 = _make_history_seed(S=S, unit=unit, m=kernel.m, trunc=trunc, dtype=dtype)
    source = jnp.arange(S, dtype=jnp.int32)

    def step(history: DenseElem, j: Array) -> tuple[DenseElem, DenseElem]:
        v_prev = tuple(level[:S] for level in history)
        coef_j = _coef_row(
            kernel,
            source=source,
            readout=j,
            times=times_arr,
            trunc=trunc,
            dtype=dtype,
        )
        coef_j = _insert_singleton_batch_axes(coef_j, batch_ndim=len(batch_shape))
        terms = _eval_vte(v_prev, y_time, coef_j)
        contribution = tuple(jnp.sum(level, axis=0) for level in terms)
        V_j = _CORE.tensor_summation(unit, contribution, trunc=trunc)
        history_next = tuple(level.at[j + 1].set(V_j[n]) for n, level in enumerate(history))
        return history_next, V_j

    history_final, _ = lax.scan(step, history0, source)

    if output_starting_point:
        out = history_final
        # The public axis refers to the user's input layout.  Since the
        # trajectory axis has length S+1 while the input had S increments, move
        # it back to the same relative position among non-tail axes.
        if axis_norm != 0:
            out = tuple(jnp.moveaxis(level, 0, axis_norm) for level in out)
        return out

    return tuple(level[-1] for level in history_final)


def _eval_vte(v: DenseElem, y: Array, coef: VolterraCoefficients) -> DenseElem:
    """Local recursion dispatcher matching the scalar/general split."""
    if coef.q == 1:
        return eval_vte_scalar(v, y, coef)
    return eval_vte_general(v, y, coef)


def _coef_row(
    kernel: VolterraKernel,
    *,
    source: Array,
    readout: Array,
    times: Array,
    trunc: int,
    dtype: jnp.dtype,
) -> VolterraCoefficients:
    """Build coefficients for all source intervals at one readout index."""
    if kernel.kind == "piecewise_constant":
        return kernel.coef_from_indices(source, readout, trunc=trunc, dtype=dtype)

    return kernel.coef(
        times[:-1],
        times[1:],
        times[readout + 1],
        trunc=trunc,
        dtype=dtype,
    )


def _normalize_projected_y_time(y_time: Array, kernel: VolterraKernel) -> Array:
    """Validate projected increments after moving the source axis to zero."""
    if kernel.q == 1:
        if y_time.shape[-1] == kernel.m:
            if y_time.ndim >= 3 and y_time.shape[-2:] == (1, kernel.m):
                return y_time[..., 0, :]
            return y_time
        raise ValueError(
            f"For q=1, y must have trailing shape ({kernel.m},) or (1, {kernel.m}); "
            f"got shape {y_time.shape}."
        )

    if y_time.ndim < 3 or y_time.shape[-2:] != (kernel.q, kernel.m):
        raise ValueError(
            f"For q>1, y must have trailing shape ({kernel.q}, {kernel.m}); "
            f"got shape {y_time.shape}."
        )
    return y_time


def _projected_batch_shape(y_time: Array, kernel: VolterraKernel) -> tuple[int, ...]:
    """Batch shape of time-first projected increments."""
    if kernel.q == 1:
        return tuple(y_time.shape[1:-1])
    return tuple(y_time.shape[1:-2])


def _normalize_times(
    times: Array | None,
    dt: Array | float | None,
    *,
    S: int,
    dtype: jnp.dtype,
) -> Array:
    """Return one-dimensional node times of shape ``(S + 1,)``."""
    if times is not None and dt is not None:
        raise ValueError("Provide either times or dt, not both.")

    if times is None:
        dt_arr = jnp.asarray(1.0 if dt is None else dt, dtype=dtype)
        if dt_arr.ndim != 0:
            raise ValueError(f"dt must be scalar for volterra_vsig, got shape {dt_arr.shape}.")
        return jnp.arange(S + 1, dtype=dtype) * dt_arr

    times_arr = jnp.asarray(times, dtype=dtype)
    if times_arr.ndim != 1:
        raise ValueError(f"times must be one-dimensional, got shape {times_arr.shape}.")
    if times_arr.shape[0] != S + 1:
        raise ValueError(
            f"times must have length S + 1 = {S + 1}, got {times_arr.shape[0]}."
        )
    return times_arr


def _make_unit(
    *,
    batch_shape: tuple[int, ...],
    m: int,
    trunc: int,
    dtype: jnp.dtype,
) -> DenseElem:
    """Tensor unit with zero positive levels."""
    return (
        jnp.ones(batch_shape + (1,), dtype=dtype),
        *(
            jnp.zeros(batch_shape + (m ** n,), dtype=dtype)
            for n in range(1, trunc + 1)
        ),
    )


def _make_history_seed(
    *,
    S: int,
    unit: DenseElem,
    m: int,
    trunc: int,
    dtype: jnp.dtype,
) -> DenseElem:
    """Padded history ``[unit, 0, ..., 0]`` with length ``S + 1``."""
    batch_shape = unit[0].shape[:-1]
    levels = []
    for n in range(trunc + 1):
        hist = jnp.zeros((S + 1,) + batch_shape + (m ** n,), dtype=dtype)
        hist = hist.at[0].set(unit[n])
        levels.append(hist)
    return tuple(levels)


def _insert_singleton_batch_axes(
    coef: VolterraCoefficients,
    *,
    batch_ndim: int,
) -> VolterraCoefficients:
    """Make a source-row coefficient broadcast over path batch axes."""
    if batch_ndim <= 0:
        return coef
    leading = coef.leading_shape + (1,) * int(batch_ndim)
    return replace(
        coef,
        alpha=coef.alpha.reshape(leading + coef.alpha.shape[-2:]),
        valid=coef.valid.reshape(leading),
    )


__all__ = ["vsig"]
