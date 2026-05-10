r"""Predictor-corrector schemes for fractional Volterra signatures.

This module implements product-integration reference schemes for the
Tensor-valued fractional Volterra equation

    V_t^0 = 1,

    V_t^n = integral_0^t K_beta(t-s) V_s^{n-1} tensor dY_s,

where

    K_beta(u) = u**(beta - 1) / Gamma(beta),

and ``Y`` is the projected path.  The scheme is intended as a transparent
numerical-validation reference for exact/quadratic Volterra algorithms.

Only the q=1 fractional case is implemented here.  This is the case needed for
checking ``VolterraKernel.fractional(beta=..., A=...)`` against a direct
tensor-algebra-valued Volterra equation solver.

The optional ``dyadic_order`` parameter linearly refines each input interval
into ``2**dyadic_order`` equal substeps by splitting both increments and time
steps uniformly.

New in this version
-------------------
The time summation can now be computed with FFT convolution on uniform grids:

    time_method="auto"   # default: FFT for scalar/length-1 dt, direct otherwise
    time_method="fft"    # require scalar/length-1 dt and use FFT convolution
    time_method="direct" # always use the triangular product-integration sum

For non-uniform time steps, the direct method is used because the weight matrix
is no longer Toeplitz.
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
        "output_starting_value",
        "dyadic_order",
        "scheme",
        "time_method",
    ),
)
def fractional_pc_vsig(
    X: Array,
    *,
    beta: float,
    A: Array,
    dt: Array | float,
    trunc: int,
    axis: int = -2,
    increment_input: bool = False,
    output_starting_value: bool = False,
    dyadic_order: int = 0,
    scheme: str = "pc",
    time_method: str = "auto",
) -> DenseElem:
    """Compute a fractional Volterra-signature reference.

    Parameters
    ----------
    X:
        Path nodes or increments.  The time axis is selected by ``axis`` and the
        trailing axis is the path dimension ``d``.
    beta:
        Fractional parameter in ``K_beta(u) = u**(beta - 1) / Gamma(beta)``.
        This is static for JIT compilation.
    A:
        Projection matrix with shape ``(1, m, d)`` or ``(m, d)``.  Only q=1 is
        supported.
    dt:
        Scalar step size or a one-dimensional array of step sizes of length
        ``S``.  FFT acceleration is used only when this is scalar or length 1.
    trunc:
        Tensor truncation level.
    axis:
        Time axis of ``X``.  The trailing axis must be the path-coordinate axis.
    increment_input:
        Set to ``True`` when ``X`` already contains increments.
    output_starting_value:
        If ``False`` return the terminal signature.  If ``True`` return the full
        node trajectory including the initial unit value at time zero.
    dyadic_order:
        Split every input increment into ``2**dyadic_order`` equal substeps.
    scheme:
        ``"pc"`` for product-integration predictor-corrector weights, or
        ``"euler"`` for left-point product Euler weights.
    time_method:
        ``"auto"`` uses FFT convolution for scalar/length-1 ``dt`` and direct
        summation otherwise.  ``"fft"`` requires scalar/length-1 ``dt``.
        ``"direct"`` always uses the triangular product-integration sum.

    Returns
    -------
    DenseElem
        If ``output_starting_value=False``, a terminal dense tensor element with
        levels ``0..trunc``.  If ``output_starting_value=True``, each level has a
        leading time axis of size ``S_refined + 1``.
    """
    _validate_static_args(
        beta=beta,
        trunc=trunc,
        dyadic_order=dyadic_order,
        scheme=scheme,
        time_method=time_method,
    )

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

    return fractional_pc_vsig_from_increments(
        dY_time,
        beta=beta,
        dt=dt,
        trunc=trunc,
        axis=0,
        output_starting_value=output_starting_value,
        dyadic_order=dyadic_order,
        scheme=scheme,
        time_method=time_method,
    )


@partial(
    jax.jit,
    static_argnames=(
        "beta",
        "trunc",
        "axis",
        "output_starting_value",
        "dyadic_order",
        "scheme",
        "time_method",
    ),
)
def fractional_pc_vsig_from_increments(
    dY: Array,
    *,
    beta: float,
    dt: Array | float,
    trunc: int,
    axis: int = 0,
    output_starting_value: bool = False,
    dyadic_order: int = 0,
    scheme: str = "pc",
    time_method: str = "auto",
) -> DenseElem:
    """Predictor-corrector reference from already projected increments.

    Parameters
    ----------
    dY:
        Projected q=1 increments with trailing shape ``(m,)``.  The time axis is
        selected by ``axis``.
    beta:
        Fractional parameter in ``K_beta(u) = u**(beta - 1) / Gamma(beta)``.
    dt:
        Scalar step size or one-dimensional array of step sizes.
        FFT acceleration is used only when this is scalar or length 1.
    trunc:
        Tensor truncation level.
    axis:
        Time axis of ``dY``.
    output_starting_value:
        If ``False`` return only the terminal tensor element.  If ``True``,
        return the full node trajectory.
    dyadic_order:
        Split every increment and time step into ``2**dyadic_order`` substeps.
    scheme:
        ``"pc"`` or ``"euler"``.
    time_method:
        ``"auto"`` uses FFT convolution for scalar/length-1 ``dt`` and direct
        summation otherwise.  ``"fft"`` requires scalar/length-1 ``dt``.
        ``"direct"`` always uses the triangular product-integration sum.

    Returns
    -------
    DenseElem
        Terminal or full-trajectory dense tensor element.
    """
    _validate_static_args(
        beta=beta,
        trunc=trunc,
        dyadic_order=dyadic_order,
        scheme=scheme,
        time_method=time_method,
    )

    dY = jnp.asarray(dY)
    if dY.ndim < 2:
        raise ValueError("dY must have a time axis and a trailing tensor dimension.")

    axis_norm = axis % dY.ndim
    if axis_norm == dY.ndim - 1:
        raise ValueError("axis must identify the time axis, not the trailing tensor dimension.")

    dY_time = jnp.moveaxis(dY, axis_norm, 0)
    S = int(dY_time.shape[0])
    if S == 0:
        raise ValueError("fractional_pc_vsig_from_increments requires at least one increment.")

    dtype = dY_time.dtype
    dt_time, uniform_grid = _normalize_dt(dt, S=S, dtype=dtype)

    dY_time, dt_time = _dyadically_refine_increments(
        dY_time,
        dt_time,
        dyadic_order,
    )

    return _fractional_pc_vsig_time_first(
        dY_time,
        beta=beta,
        dt_time=dt_time,
        trunc=trunc,
        output_starting_value=output_starting_value,
        scheme=scheme,
        time_method=time_method,
        uniform_grid=uniform_grid,
    )


def _fractional_pc_vsig_time_first(
    dY_time: Array,
    *,
    beta: float,
    dt_time: Array,
    trunc: int,
    output_starting_value: bool,
    scheme: str,
    time_method: str,
    uniform_grid: bool,
) -> DenseElem:
    """Core time-first solver.

    ``dY_time`` has shape ``(S, batch..., m)`` and ``dt_time`` has shape
    ``(S,)``.
    """
    S = int(dY_time.shape[0])
    m = int(dY_time.shape[-1])
    batch_shape = dY_time.shape[1:-1]
    dtype = dY_time.dtype

    if time_method == "fft" and not uniform_grid:
        raise ValueError(
            "time_method='fft' requires dt to be a scalar or a length-1 array. "
            "For a uniform length-S dt array, pass dt=dt[0] to enable FFT."
        )

    use_fft = time_method == "fft" or (time_method == "auto" and uniform_grid)

    if use_fft:
        h = dt_time[0].astype(dtype)
        times = None
    else:
        h = None
        times = jnp.concatenate(
            [
                jnp.zeros((1,), dtype=dtype),
                jnp.cumsum(dt_time.astype(dtype), axis=0),
            ],
            axis=0,
        )

    level0 = jnp.ones((S + 1,) + batch_shape + (1,), dtype=dtype)
    levels: list[Array] = [level0]

    for degree in range(1, trunc + 1):
        if use_fft:
            level = _solve_fractional_level_fft(
                prev=levels[degree - 1],
                dY_time=dY_time,
                h=h,
                beta=beta,
                degree=degree,
                m=m,
                scheme=scheme,
            )
        else:
            level = _solve_fractional_level_direct(
                prev=levels[degree - 1],
                dY_time=dY_time,
                times=times,
                beta=beta,
                degree=degree,
                m=m,
                scheme=scheme,
            )
        levels.append(level)

    if output_starting_value:
        return tuple(levels)

    return tuple(level[-1] for level in levels)


def _solve_fractional_level_fft(
    *,
    prev: Array,
    dY_time: Array,
    h: Array,
    beta: float,
    degree: int,
    m: int,
    scheme: str,
) -> Array:
    """Solve one homogeneous tensor level by FFT convolution on a uniform grid.

    For target node ``n = 1, ..., S`` and source interval ``i < n``, the
    product-integration weights depend only on the lag ``k = n - i`` when
    ``dt_i = h``.  Thus each tensor channel is a causal convolution along the
    time axis.
    """
    S = int(dY_time.shape[0])
    batch_shape = dY_time.shape[1:-1]
    dtype = dY_time.dtype
    width = m**degree

    zero = jnp.zeros(batch_shape + (width,), dtype=dtype)

    # Source-wise tensor contributions, shape (S, batch..., width).
    contrib_left = jax.vmap(_append_increment)(prev[:-1], dY_time)
    contrib_right = jax.vmap(_append_increment)(prev[1:], dY_time)

    w_left, w_right, w_euler = _fractional_uniform_lag_weights(
        beta=beta,
        h=h,
        S=S,
        dtype=dtype,
    )

    if scheme == "euler":
        values = _causal_fft_convolve_time(contrib_left, w_euler)
    else:
        values = _causal_fft_convolve_time(contrib_left, w_left)
        values = values + _causal_fft_convolve_time(contrib_right, w_right)

    return jnp.concatenate([zero[None], values], axis=0)


def _solve_fractional_level_direct(
    *,
    prev: Array,
    dY_time: Array,
    times: Array,
    beta: float,
    degree: int,
    m: int,
    scheme: str,
) -> Array:
    """Solve one homogeneous tensor level by direct triangular summation.

    ``prev`` is the already computed previous level with shape

        (S + 1, batch..., m**(degree - 1)).

    The returned level has shape

        (S + 1, batch..., m**degree).

    Memory layout: source contributions are precomputed once as
    ``(S, batch..., width)`` arrays (O(S·batch·width)).  The outer loop over the
    S target nodes is a ``lax.scan`` (carry = None, so XLA can pipeline/fuse it),
    and the inner weighted sum over sources is fully vectorised for each target
    step.
    """
    S = int(dY_time.shape[0])
    batch_shape = dY_time.shape[1:-1]
    dtype = dY_time.dtype
    width = m**degree

    zero = jnp.zeros(batch_shape + (width,), dtype=dtype)
    i_idx = jnp.arange(S)  # (S,)
    n_idx = jnp.arange(1, S + 1)  # (S,)

    # vmap over source axis: (S, batch..., width)
    contrib_left = jax.vmap(_append_increment)(prev[i_idx], dY_time)
    contrib_right = jax.vmap(_append_increment)(prev[i_idx + 1], dY_time)

    # Precompute source time-interval endpoints.
    a_all = times[i_idx]  # (S,)
    b_all = times[i_idx + 1]  # (S,)

    extra = (1,) * (len(batch_shape) + 1)  # broadcast over (batch..., width)

    def target_step(_, n):
        tn = times[n]
        valid = i_idx < n  # (S,)

        # Safe values keep fractional powers well-defined for masked entries.
        a_s = jnp.where(valid, a_all, jnp.zeros_like(a_all))
        b_s = jnp.where(valid, b_all, jnp.ones_like(b_all))
        tn_s = jnp.where(valid, tn, jnp.ones_like(a_all))

        w_left, w_right, w_euler = _fractional_interval_weights(
            beta=beta,
            tn=tn_s,
            a=a_s,
            b=b_s,
            dtype=dtype,
        )
        # w_*: (S,) — one scalar weight per source interval.

        # Broadcast weights to (S, 1..., 1) for (S, batch..., width).
        w_l = w_left.reshape(w_left.shape + extra)
        w_r = w_right.reshape(w_right.shape + extra)
        w_e = w_euler.reshape(w_euler.shape + extra)

        if scheme == "euler":
            weighted = w_e * contrib_left
        else:
            weighted = w_l * contrib_left + w_r * contrib_right

        mask = valid.reshape(valid.shape + extra)
        out = jnp.where(mask, weighted, jnp.zeros_like(weighted)).sum(axis=0)
        return None, out

    _, values = lax.scan(target_step, None, n_idx)
    return jnp.concatenate([zero[None], values], axis=0)


def _fractional_uniform_lag_weights(
    *,
    beta: float,
    h: Array,
    S: int,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array]:
    r"""Uniform-grid PC/Euler weights as lag kernels.

    The returned arrays have length ``S``.  Entry ``j`` corresponds to lag
    ``k = j + 1``.  For the output node index ``n = 1, ..., S``:

        V_n = sum_{i=0}^{n-1} w[n - i - 1] * contribution_i.

    These are algebraically the same weights as ``_fractional_interval_weights``
    with ``a = i h``, ``b = (i + 1) h`` and ``tn = n h``.
    """
    beta_arr = jnp.asarray(beta, dtype=dtype)
    gamma_beta = jnp.exp(gammaln(beta_arr))

    k = jnp.arange(1, S + 1, dtype=dtype)
    km1 = k - jnp.asarray(1.0, dtype=dtype)

    delta_beta = k**beta_arr - km1**beta_arr
    delta_beta_plus_1 = k ** (beta_arr + 1.0) - km1 ** (beta_arr + 1.0)

    scale = h ** (beta_arr - 1.0) / gamma_beta

    w_euler = scale * delta_beta / beta_arr
    w_right = scale * (
        k * delta_beta / beta_arr
        - delta_beta_plus_1 / (beta_arr + jnp.asarray(1.0, dtype=dtype))
    )
    w_left = w_euler - w_right

    return w_left, w_right, w_euler


def _causal_fft_convolve_time(contrib: Array, weights: Array) -> Array:
    """Return the first S terms of the causal convolution ``weights * contrib``.

    Parameters
    ----------
    contrib:
        Array with shape ``(S, ...)``.
    weights:
        One-dimensional lag kernel with shape ``(S,)``.

    Returns
    -------
    Array
        Array with shape ``(S, ...)`` whose first element is
        ``weights[0] * contrib[0]``.
    """
    S = int(contrib.shape[0])
    conv_len = 2 * S - 1
    fft_len = _next_power_of_two(conv_len)

    fw = jnp.fft.rfft(weights, n=fft_len, axis=0)
    fw = fw.reshape(fw.shape + (1,) * (contrib.ndim - 1))

    fc = jnp.fft.rfft(contrib, n=fft_len, axis=0)
    out = jnp.fft.irfft(fw * fc, n=fft_len, axis=0)
    return out[:S].astype(contrib.dtype)


def _next_power_of_two(n: int) -> int:
    """Small static helper for FFT zero-padding."""
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def _fractional_interval_weights(
    *,
    beta: float,
    tn: Array,
    a: Array,
    b: Array,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array]:
    r"""Product-integration weights on one interval.

    For ``h = b - a`` and ``K_beta(u) = u**(beta - 1) / Gamma(beta)``,

        w_euler = (1/h) int_a^b K_beta(tn - s) ds,

        w_left  = (1/h) int_a^b ((b-s)/h) K_beta(tn - s) ds,

        w_right = (1/h) int_a^b ((s-a)/h) K_beta(tn - s) ds.

    These are the coefficients multiplying ``V_a tensor dY`` and
    ``V_b tensor dY`` when ``Y`` is linearly interpolated on the interval.
    """
    beta_arr = jnp.asarray(beta, dtype=dtype)
    gamma_beta = jnp.exp(gammaln(beta_arr))

    h = b - a
    A = tn - a
    B = jnp.maximum(tn - b, jnp.asarray(0.0, dtype=dtype))

    I0 = (A**beta_arr - B**beta_arr) / (beta_arr * gamma_beta)

    Iright = (
        A * (A**beta_arr - B**beta_arr) / beta_arr
        - (A ** (beta_arr + 1.0) - B ** (beta_arr + 1.0)) / (beta_arr + 1.0)
    ) / (h * gamma_beta)

    Ileft = I0 - Iright

    # Divide by h because dY_s = (Delta Y / h) ds on a linear interval.
    return (
        Ileft / h,
        Iright / h,
        I0 / h,
    )


def _append_increment(prefix: Array, dy: Array) -> Array:
    """Return ``prefix tensor dy`` using the tensor core implementation."""
    return _CORE.tensor_product_homogeneous(prefix, dy)


def _dyadically_refine_increments(
    increments_time: Array,
    dt_time: Array,
    dyadic_order: int,
) -> tuple[Array, Array]:
    """Split each time-first increment/time-step into dyadic substeps."""
    if dyadic_order == 0:
        return increments_time, dt_time

    factor = 1 << int(dyadic_order)
    inc_factor = jnp.asarray(factor, dtype=increments_time.dtype)
    dt_factor = jnp.asarray(factor, dtype=dt_time.dtype)

    increments_refined = jnp.repeat(increments_time / inc_factor, factor, axis=0)
    dt_refined = jnp.repeat(dt_time / dt_factor, factor, axis=0)
    return increments_refined, dt_refined


def _normalize_dt(dt: Array | float, *, S: int, dtype: jnp.dtype) -> tuple[Array, bool]:
    """Normalize ``dt`` to shape ``(S,)`` and flag static uniform grids.

    The returned boolean is intentionally based on the *shape* of ``dt`` rather
    than a value comparison, because this function is used under ``jax.jit``.
    A scalar or length-1 array is considered uniform.  A length-``S`` array is
    treated as non-uniform, even if all entries happen to be equal; pass
    ``dt=dt[0]`` to enable FFT in that case.
    """
    dt_arr = jnp.asarray(dt, dtype=dtype)

    if dt_arr.ndim == 0:
        return jnp.full((S,), dt_arr, dtype=dtype), True

    if dt_arr.ndim == 1:
        if dt_arr.shape[0] not in (1, S):
            raise ValueError(f"1D dt must have length 1 or S={S}, got {dt_arr.shape[0]}.")
        uniform_grid = dt_arr.shape[0] == 1
        return jnp.broadcast_to(dt_arr, (S,)).astype(dtype), uniform_grid

    raise ValueError("dt must be a scalar or a one-dimensional array of step sizes.")


def _normalize_A(A: Array) -> Array:
    """Normalize ``A`` to shape ``(1, m, d)``."""
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
    scheme: str,
    time_method: str,
) -> None:
    beta = float(beta)

    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}.")
    if trunc <= 0:
        raise ValueError(f"trunc must be positive, got {trunc}.")
    if dyadic_order < 0:
        raise ValueError(f"dyadic_order must be nonnegative, got {dyadic_order}.")
    if scheme not in {"pc", "euler"}:
        raise ValueError("scheme must be either 'pc' or 'euler'.")
    if time_method not in {"auto", "fft", "direct"}:
        raise ValueError("time_method must be one of 'auto', 'fft', or 'direct'.")


__all__ = [
    "fractional_pc_vsig",
    "fractional_pc_vsig_from_increments",
]
