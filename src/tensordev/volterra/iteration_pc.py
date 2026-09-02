"""Adams predictor-corrector iteration for fractional Volterra signatures.

Implements product-integration PC (Adams-Moulton type) and Euler schemes via
FFT convolution on uniform grids.  The scheme supports only
``FractionalKernel`` with a single component (``q=1``).

The ``order`` parameter maps as follows:
    0  →  Euler (left-point product-integration weights)
    ≥1 →  Predictor-corrector (trapezoidal product-integration weights)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from tensordev.core.utils.pytrees import tree_index, tree_map
from tensordev.volterra._convolution import causal_fft_from_raw_weights
from tensordev.volterra.algebra import (
    ResolvedVolterraAlgebra,
    resolve_volterra_algebra,
    resolve_volterra_core_pair,
)
from tensordev.volterra.kernel import ConvolutionKernel, FractionalKernel

Array = jax.Array


def pc_iteration(
        dX: Array,
        *,
        kernel: ConvolutionKernel,
        trunc=None,
        dt: Array | float = 1.0,
        axis: int = -2,
        return_trajectory: bool = False,
        order: int = 1,
        core=None,
        seq_core=None,
):
    """Volterra signature via Adams predictor-corrector on a uniform grid.

    Restricted to :class:`~tensordev.volterra.kernel.FractionalKernel` with a
    single component (``q=1``).  The time convolution is accelerated with FFT,
    so ``dt`` must be a scalar (uniform grid).

    Parameters
    ----------
    dX:
        Increments with step axis at ``axis`` and trailing path dimension
        ``d = kernel.path_dim``.  Dyadic refinement is applied by the caller
        (:func:`~tensordev.volterra.signature.vsig`) before this function is
        invoked.
    kernel:
        Must be a :class:`~tensordev.volterra.kernel.FractionalKernel` with
        ``q=1``.
    trunc:
        Active integer or bidegree truncation.  May be omitted for a bounded
        core with a configured default.
    dt:
        Uniform step size scalar (default ``1.0``).
    axis:
        Step axis of ``dX`` (default ``-2``).
    return_trajectory:
        If ``True``, return ``[V_1, ..., V_S]`` with the step axis at
        ``axis``.  If ``False`` (default), return only the terminal ``V_S``.
    order:
        ``0`` → Euler (left-point) quadrature weights.
        ``≥1`` → predictor-corrector (trapezoidal) quadrature weights.
    core, seq_core:
        Algebra and sequential JAX cores.  Omitted values resolve from the
        coherent process default.

    Returns
    -------
    object
        Native tensor element, or a native full trajectory when
        ``return_trajectory=True``.

    Raises
    ------
    TypeError
        If ``kernel`` is not a :class:`~tensordev.volterra.kernel.FractionalKernel`.
    ValueError
        If ``kernel.q != 1`` or other arguments are invalid.
    """
    if not isinstance(kernel, FractionalKernel):
        raise TypeError(
            f"scheme='adams' requires a FractionalKernel; got {type(kernel).__name__}."
        )
    if kernel.q != 1:
        raise ValueError(
            f"scheme='adams' supports only single-component kernels "
            f"(q=1); got q={kernel.q}."
        )
    core, seq_core = resolve_volterra_core_pair(core, seq_core)
    del seq_core  # Pair validation is shared even though Adams does not scan.
    algebra = resolve_volterra_algebra(core, trunc, kernel.m)

    xp = core.xp
    dX = xp.asarray(dX)
    if dX.ndim < 2:
        raise ValueError("dX must have at least a step axis and a trailing path dimension.")

    axis_norm = axis % dX.ndim
    if axis_norm == dX.ndim - 1:
        raise ValueError(
            "axis must identify the step axis, not the trailing path dimension."
        )
    if dX.shape[-1] != kernel.path_dim:
        raise ValueError(
            f"dX trailing dimension must be {kernel.path_dim}, got {dX.shape[-1]}."
        )

    # Project: dY[..., a] = sum_d A[0, a, d] * dX[..., d]
    A_mat = kernel.A[0].astype(dX.dtype)           # (m, d)
    dY = xp.einsum("md,...d->...m", A_mat, dX)     # (..., S, m)
    dY_time = xp.moveaxis(dY, axis_norm, 0)         # (S, *batch, m)

    S = int(dY_time.shape[0])
    if S == 0:
        raise ValueError("pc_iteration requires at least one increment.")

    dtype = dY_time.dtype
    h = xp.asarray(dt, dtype=dtype)
    if h.ndim != 0:
        raise ValueError(
            "pc_iteration requires a scalar dt because the Adams scheme "
            "assumes a uniform grid."
        )

    beta = kernel.beta[0].astype(dtype)
    scheme = "euler" if order == 0 else "pc"

    history = _solve_all_grades(
        dY_time,
        beta=beta,
        h=h,
        algebra=algebra,
        scheme=scheme,
    )

    if return_trajectory:
        # Every block has shape (S+1, *batch, width); drop the t=0 entry.
        return tree_map(lambda block: xp.moveaxis(block[1:], 0, axis_norm), history)

    return tree_index(history, -1)


# ---------------------------------------------------------------------------
# Core level solver
# ---------------------------------------------------------------------------

def _solve_all_grades(
        dY_time: Array,
        *,
        beta: Array,
        h: Array,
        algebra: ResolvedVolterraAlgebra,
        scheme: str,
) -> object:
    S = int(dY_time.shape[0])
    batch_shape = dY_time.shape[1:-1]
    dtype = dY_time.dtype
    xp = algebra.core.xp

    w_left, w_right, w_euler = _lag_weights(beta=beta, h=h, S=S, dtype=dtype)

    histories: dict[object, Array] = {
        algebra.zero_grade: xp.ones(
            (S + 1,) + tuple(batch_shape) + (1,), dtype=dtype
        )
    }

    for degree in range(1, algebra.max_order + 1):
        predecessor = algebra.diagonal(degree - 1)
        target = algebra.diagonal(degree)
        previous = tuple(histories[grade] for grade in predecessor.grades)

        previous_left = tuple(block[:-1] for block in previous)
        left_blocks = tuple(
            algebra.right_generator_output_block(
                predecessor,
                previous_left,
                dY_time,
                output_grade,
            )
            for output_grade in target.grades
        )
        packed_left = target.pack(xp, left_blocks)

        if scheme == "euler":
            packed_values = causal_fft_from_raw_weights(packed_left, w_euler)
        else:
            previous_right = tuple(block[1:] for block in previous)
            right_blocks = tuple(
                algebra.right_generator_output_block(
                    predecessor,
                    previous_right,
                    dY_time,
                    output_grade,
                )
                for output_grade in target.grades
            )
            packed_right = target.pack(xp, right_blocks)
            packed_values = (
                causal_fft_from_raw_weights(packed_left, w_left)
                + causal_fft_from_raw_weights(packed_right, w_right)
            )

        for grade, values in zip(target.grades, target.split(packed_values)):
            zero = algebra.zero_block(
                grade,
                batch_shape=tuple(batch_shape),
                dtype=dtype,
            )
            histories[grade] = xp.concat((zero[None], values), axis=0)

    return algebra.assemble(tuple(histories[grade] for grade in algebra.grades))


# ---------------------------------------------------------------------------
# Lag weights
# ---------------------------------------------------------------------------

def _lag_weights(
        *,
        beta: Array,
        h: Array,
        S: int,
        dtype: jnp.dtype,
) -> tuple[Array, Array, Array]:
    """Uniform-grid PC/Euler lag weights.  Entry j corresponds to lag k = j+1."""
    beta = beta.astype(dtype)
    gamma_beta = jnp.exp(gammaln(beta))

    k   = jnp.arange(1, S + 1, dtype=dtype)
    km1 = k - jnp.asarray(1.0, dtype=dtype)

    delta_beta   = k ** beta - km1 ** beta
    delta_beta_1 = k ** (beta + 1.0) - km1 ** (beta + 1.0)

    scale   = h ** (beta - 1.0) / gamma_beta
    w_euler = scale * delta_beta / beta
    w_right = scale * (k * delta_beta / beta - delta_beta_1 / (beta + 1.0))
    w_left  = w_euler - w_right

    return w_left, w_right, w_euler


__all__ = ["pc_iteration"]
