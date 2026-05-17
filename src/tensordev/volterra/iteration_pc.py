"""Adams predictor-corrector iteration for fractional Volterra signatures.

Implements product-integration PC (Adams-Moulton type) and Euler schemes via
FFT convolution on uniform grids.  Only FractionalKernel with a single
component (q=1) is currently supported.

The ``order`` parameter maps as follows:
    0  →  Euler (left-point product-integration weights)
    ≥1 →  Predictor-corrector (trapezoidal product-integration weights)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from tensordev.core.jax import Jax
from tensordev.core.universal import DenseElem
from tensordev.volterra.kernel import ConvolutionKernel, FractionalKernel

Array = jax.Array
_CORE = Jax()


def pc_iteration(
        dX: Array,
        *,
        kernel: ConvolutionKernel,
        trunc: int,
        dt: Array | float = 1.0,
        axis: int = -2,
        return_trajectory: bool = False,
        order: int = 1,
) -> DenseElem:
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
        Tensor truncation level (positive integer).
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

    Returns
    -------
    DenseElem
        Terminal signature, or full trajectory when ``return_trajectory=True``.

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
            f"scheme='adams' currently supports only single-component kernels "
            f"(q=1); got q={kernel.q}."
        )
    if trunc <= 0:
        raise ValueError(f"trunc must be positive, got {trunc}.")

    dX = jnp.asarray(dX)
    if dX.ndim < 2:
        raise ValueError("dX must have at least a step axis and a trailing path dimension.")

    axis_norm = axis % dX.ndim

    # Project: dY[..., a] = sum_d A[0, a, d] * dX[..., d]
    A_mat = kernel.A[0].astype(dX.dtype)           # (m, d)
    dY = jnp.einsum("md,...d->...m", A_mat, dX)    # (..., S, m)
    dY_time = jnp.moveaxis(dY, axis_norm, 0)       # (S, *batch, m)

    S = int(dY_time.shape[0])
    if S == 0:
        raise ValueError("pc_iteration requires at least one increment.")

    dtype = dY_time.dtype
    h = jnp.asarray(dt, dtype=dtype)
    if h.ndim > 0:
        h = h[0]

    beta = kernel.beta[0].astype(dtype)
    scheme = "euler" if order == 0 else "pc"

    levels = _solve_all_levels(dY_time, beta=beta, h=h, trunc=trunc, scheme=scheme)

    if return_trajectory:
        # levels[n] has shape (S+1, *batch, m^n); drop the t=0 entry.
        return tuple(jnp.moveaxis(lvl[1:], 0, axis_norm) for lvl in levels)

    return tuple(lvl[-1] for lvl in levels)


# ---------------------------------------------------------------------------
# Core level solver
# ---------------------------------------------------------------------------

def _solve_all_levels(
        dY_time: Array,
        *,
        beta: Array,
        h: Array,
        trunc: int,
        scheme: str,
) -> tuple[Array, ...]:
    S = int(dY_time.shape[0])
    m = int(dY_time.shape[-1])
    batch_shape = dY_time.shape[1:-1]
    dtype = dY_time.dtype

    w_left, w_right, w_euler = _lag_weights(beta=beta, h=h, S=S, dtype=dtype)

    level0 = jnp.ones((S + 1,) + batch_shape + (1,), dtype=dtype)
    levels: list[Array] = [level0]

    for degree in range(1, trunc + 1):
        prev = levels[degree - 1]
        width = m ** degree
        zero = jnp.zeros(batch_shape + (width,), dtype=dtype)

        contrib_left  = jax.vmap(_tensor_product)(prev[:-1], dY_time)
        contrib_right = jax.vmap(_tensor_product)(prev[1:],  dY_time)

        if scheme == "euler":
            values = _causal_fft(contrib_left, w_euler)
        else:
            values = _causal_fft(contrib_left, w_left) + _causal_fft(contrib_right, w_right)

        levels.append(jnp.concatenate([zero[None], values], axis=0))

    return tuple(levels)


# ---------------------------------------------------------------------------
# Weight computation  (identical formulas to predictor_corrector.py)
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


# ---------------------------------------------------------------------------
# FFT convolution helper
# ---------------------------------------------------------------------------

def _causal_fft(contrib: Array, weights: Array) -> Array:
    """First S terms of the causal convolution weights * contrib along axis 0."""
    S = int(contrib.shape[0])
    n_fft = _next_pow2(2 * S - 1)

    fw = jnp.fft.rfft(weights, n=n_fft, axis=0)
    fw = fw.reshape(fw.shape + (1,) * (contrib.ndim - 1))
    fc = jnp.fft.rfft(contrib, n=n_fft, axis=0)
    return jnp.fft.irfft(fw * fc, n=n_fft, axis=0)[:S].astype(contrib.dtype)


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


def _tensor_product(prefix: Array, dy: Array) -> Array:
    return _CORE.tensor_product_homogeneous(prefix, dy)


__all__ = ["pc_iteration"]