from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
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
    dyadic_order: int = 0,
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
    dyadic_order:
        Non-negative integer.  Each original increment is split into
        ``2**dyadic_order`` equal sub-increments (each multiplied by
        ``1 / 2**dyadic_order``), and ``dt`` / ``times`` are refined
        accordingly.  ``dyadic_order=0`` (default) leaves the path unchanged.

    Returns
    -------
    DenseElem
        Terminal signature by default.  With ``output_starting_point=True``,
        each level carries an additional trajectory axis at ``axis``.
    """
    if trunc <= 0:
        raise ValueError(f"trunc must be positive, got {trunc}.")
    if dyadic_order < 0:
        raise ValueError(f"dyadic_order must be non-negative, got {dyadic_order}.")

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

    # Dyadic refinement: split each increment into 2**dyadic_order equal sub-increments.
    if dyadic_order > 0:
        factor = 1 << int(dyadic_order)
        dX = jnp.repeat(dX / factor, factor, axis=axis_norm)
        S = dX.shape[axis_norm]
        times, dt = _refine_times_or_dt(times, dt, factor=factor, dtype=dtype)

    projected = jnp.einsum("qmd,...d->...qm", kernel.A.astype(dtype), dX)
    y = projected[..., 0, :] if kernel.q == 1 else projected

    y_time = jnp.moveaxis(y, axis_norm, 0)
    y_time = _normalize_projected_y_time(y_time, kernel)
    times_arr = _normalize_times(times, dt, S=S, dtype=dtype)

    batch_shape = _projected_batch_shape(y_time, kernel)
    unit = _make_unit(batch_shape=batch_shape, m=kernel.m, trunc=trunc, dtype=dtype)
    history0 = _make_history_seed(S=S, unit=unit, m=kernel.m, trunc=trunc, dtype=dtype)
    source = jnp.arange(S, dtype=jnp.int32)

    # For continuous kernels (fractional, gamma) the coefficient computation
    # involves expensive special functions (betainc).  Precomputing the full
    # (S × S) grid in one batched call before the scan is much more efficient
    # than evaluating one column at a time inside the scan body.
    # Piecewise-constant kernels are already a cheap gather and are left as-is.
    if kernel.kind != "piecewise_constant":
        _precomp = kernel.coef_grid(times_arr, trunc=trunc, dtype=dtype)
    else:
        _precomp = None

    def step(history: DenseElem, j: Array) -> tuple[DenseElem, DenseElem]:
        v_prev = tuple(level[:S] for level in history)
        if _precomp is not None:
            coef_j = _precomp[:, j]
        else:
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


def _refine_times_or_dt(
    times: Array | None,
    dt: Array | float | None,
    *,
    factor: int,
    dtype: jnp.dtype,
) -> tuple[Array | None, Array | float | None]:
    """Refine ``times`` or ``dt`` for a dyadic subdivision by ``factor``."""
    if times is not None:
        times_arr = jnp.asarray(times, dtype=dtype)
        # Fine grid: for each coarse interval subdivide into `factor` equal parts.
        fine_dts = jnp.repeat(jnp.diff(times_arr) / factor, factor)
        fine_times = jnp.concatenate([times_arr[:1], times_arr[:1] + jnp.cumsum(fine_dts)])
        return fine_times, None

    # scalar dt (or None → defaults to 1)
    dt_val = 1.0 if dt is None else dt
    dt_arr = jnp.asarray(dt_val, dtype=dtype)
    if dt_arr.ndim != 0:
        raise ValueError(f"dt must be scalar when using dyadic_order, got shape {dt_arr.shape}.")
    return None, dt_arr / factor


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


def vsig_fft(
    X: Array,
    *,
    kernel: VolterraKernel,
    trunc: int,
    times: Array | None = None,
    dt: Array | float | None = None,
    axis: int = -2,
    output_starting_point: bool = False,
    increment_input: bool = False,
    dyadic_order: int = 0,
    order: int = 0,
) -> DenseElem:
    r"""Level-by-level Volterra signature using FFT convolution (q=1 only).

    Restructures the sequential readout scan of :func:`vsig` into a
    per-level pass that evaluates all readout times simultaneously.  The
    key identity is the Horner expansion

    .. math::
        (V_i \otimes_N E_{ij})^{(n)}
        = \sum_{k=0}^{n-1} \alpha[i,j,n{-}k{-}1]\,
          \bigl(V_i^{(k)} \otimes y_i^{\otimes(n-k)}\bigr),

    so :math:`V_j^{(n)}` for **all** :math:`j` simultaneously is a causal
    discrete convolution computable via FFT in :math:`O(S \log S)`.

    This path is only taken for **convolution kernels** (``fractional``,
    ``gamma``) on a **uniform grid** (``times=None``), where the coefficient
    matrix is Toeplitz and only the :math:`O(S \cdot M)` impulse-response
    row needs to be stored — not an :math:`O(S^2 \cdot M)` grid.

    For all other cases (non-uniform grids, ``piecewise_constant``) the call
    is forwarded to :func:`vsig`, which retains its :math:`O(S \cdot M)`
    per-step memory footprint via the scan.

    Parameters
    ----------
    X, kernel, trunc, times, dt, axis, output_starting_point,
    increment_input, dyadic_order:
        Same semantics as :func:`vsig`.
    order:
        Quadrature order for the local kernel expansion.  ``0`` (default)
        uses the standard Horner scheme; ``1`` uses a three-point
        :math:`\theta \in \{0, \tfrac{1}{2}, 1\}` interpolation that fits
        the singular basis :math:`\{1, s^\beta, s\}` explicitly, giving
        higher accuracy per step for fractional kernels with
        :math:`\beta \in (\tfrac{1}{2}, 1)`.  ``order=1`` requires
        ``kernel.kind == "fractional"`` and a uniform grid (``times=None``).

    Returns
    -------
    DenseElem
        Terminal signature (or full trajectory with ``output_starting_point``),
        numerically equivalent to the output of :func:`vsig`.

    Raises
    ------
    NotImplementedError
        If ``kernel.q != 1``, or if ``order=1`` with a non-fractional kernel.
    """
    if trunc <= 0:
        raise ValueError(f"trunc must be positive, got {trunc}.")
    if dyadic_order < 0:
        raise ValueError(f"dyadic_order must be non-negative, got {dyadic_order}.")
    if order not in (0, 1):
        raise ValueError(f"order must be 0 or 1, got {order}.")
    if kernel.q != 1:
        raise NotImplementedError(
            f"vsig_fft only supports scalar kernels (q=1); got q={kernel.q}."
        )
    if order == 1:
        if kernel.kind != "fractional":
            raise NotImplementedError(
                "order=1 only supports fractional kernels; "
                f"got kernel.kind={kernel.kind!r}. Use order=0 for gamma kernels."
            )
        if times is not None:
            raise NotImplementedError(
                "order=1 requires a uniform grid; pass times=None and use dt instead."
            )

    # Non-Toeplitz fallback (order=0 only): piecewise_constant or non-uniform grid.
    # Forward to vsig which holds only one column at a time (O(S·M) memory).
    if order == 0 and not (kernel.kind in ("fractional", "gamma") and times is None):
        return vsig(
            X, kernel=kernel, trunc=trunc, times=times, dt=dt, axis=axis,
            output_starting_point=output_starting_point,
            increment_input=increment_input, dyadic_order=dyadic_order,
        )

    # --- FFT path (convolution kernel, uniform grid) ---
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
        raise ValueError("vsig_fft requires at least one increment.")

    if dyadic_order > 0:
        factor = 1 << int(dyadic_order)
        dX = jnp.repeat(dX / factor, factor, axis=axis_norm)
        S = dX.shape[axis_norm]
        times, dt = _refine_times_or_dt(times, dt, factor=factor, dtype=dtype)

    projected = jnp.einsum("qmd,...d->...qm", kernel.A.astype(dtype), dX)
    y_time = jnp.moveaxis(projected[..., 0, :], axis_norm, 0)  # (S, *batch, m)
    times_arr = _normalize_times(times, dt, S=S, dtype=dtype)
    batch_shape = tuple(y_time.shape[1:-1])
    m = kernel.m

    unit = _make_unit(batch_shape=batch_shape, m=m, trunc=trunc, dtype=dtype)

    if order == 0:
        return _vsig_fft_order0(
            y_time=y_time,
            times_arr=times_arr,
            kernel=kernel,
            trunc=trunc,
            S=S,
            m=m,
            batch_shape=batch_shape,
            unit=unit,
            axis_norm=axis_norm,
            output_starting_point=output_starting_point,
            dtype=dtype,
        )
    else:
        return _vsig_fft_order1(
            y_time=y_time,
            times_arr=times_arr,
            kernel=kernel,
            trunc=trunc,
            S=S,
            m=m,
            batch_shape=batch_shape,
            axis_norm=axis_norm,
            output_starting_point=output_starting_point,
            dtype=dtype,
        )


def _vsig_fft_order0(
    *,
    y_time: Array,
    times_arr: Array,
    kernel: VolterraKernel,
    trunc: int,
    S: int,
    m: int,
    batch_shape: tuple[int, ...],
    unit: DenseElem,
    axis_norm: int,
    output_starting_point: bool,
    dtype: jnp.dtype,
) -> DenseElem:
    """Standard Horner-expansion FFT scheme (order=0)."""
    # Toeplitz impulse response — only O(S·M) memory.
    g_coef = kernel.coef(times_arr[0], times_arr[1], times_arr[1:], trunc=trunc, dtype=dtype)
    g = g_coef.alpha[..., 0, :]  # shape (S, M)
    fft_n = _next_pow2(2 * S)
    G = jnp.fft.rfft(g, n=fft_n, axis=0)  # (fft_n//2+1, M)

    y_pow: list[Array] = [
        jnp.ones((S,) + batch_shape + (1,), dtype=dtype),
        y_time,
    ]
    for r in range(2, trunc + 1):
        y_pow.append(_CORE.tensor_product_homogeneous(y_pow[r - 1], y_time))

    # V_all[n] shape: (S+1, *batch, m^n).  Slot 0 = unit[n]; slot j+1 = V_j^(n).
    V_all: list[Array] = [jnp.ones((S + 1,) + batch_shape + (1,), dtype=dtype)]
    extra_bc = (1,) * (len(batch_shape) + 1)  # broadcasts G over batch+tensor dims

    for n in range(1, trunc + 1):
        m_n = m ** n
        acc = jnp.zeros((S,) + batch_shape + (m_n,), dtype=dtype)
        for k in range(n):
            ell_idx = n - k - 1
            signal = _CORE.tensor_product_homogeneous(V_all[k][:S], y_pow[n - k])
            g_ell = G[:, ell_idx].reshape((-1,) + extra_bc)
            SIG = jnp.fft.rfft(signal, n=fft_n, axis=0)
            acc = acc + jnp.fft.irfft(g_ell * SIG, n=fft_n, axis=0)[:S]
        V_all.append(jnp.concatenate([unit[n][None], unit[n] + acc], axis=0))

    if output_starting_point:
        out = tuple(V_all)
        if axis_norm != 0:
            out = tuple(jnp.moveaxis(lev, 0, axis_norm) for lev in out)
        return out
    return tuple(lev[-1] for lev in V_all)


def _vsig_fft_order1(
    *,
    y_time: Array,
    times_arr: Array,
    kernel: VolterraKernel,
    trunc: int,
    S: int,
    m: int,
    batch_shape: tuple[int, ...],
    axis_norm: int,
    output_starting_point: bool,
    dtype: jnp.dtype,
) -> DenseElem:
    """Higher-order 3-point θ interpolation FFT scheme for fractional kernels (order=1).

    Fits a local basis {1, s^β, s} at each interval to resolve the
    singularity of the fractional kernel explicitly.
    """
    beta = kernel.beta[0].astype(dtype)  # scalar β, shape ()
    h0 = times_arr[1] - times_arr[0]

    theta_half = jnp.asarray(0.5, dtype=dtype)
    theta_beta = theta_half ** beta

    y_powers: list[Array] = [
        jnp.ones((S,) + batch_shape + (1,), dtype=dtype),
        y_time,
    ]
    for q in range(2, trunc + 1):
        y_powers.append(_CORE.tensor_product_homogeneous(y_powers[-1], y_time))

    # C0/Cb/C1 histories: C_levels[ell] has shape (S, *batch, m^ell).
    C0_levels: list[Array] = [jnp.ones((S,) + batch_shape + (1,), dtype=dtype)]
    Cb_levels: list[Array] = [jnp.zeros((S,) + batch_shape + (1,), dtype=dtype)]
    C1_levels: list[Array] = [jnp.zeros((S,) + batch_shape + (1,), dtype=dtype)]

    for ell in range(1, trunc + 1):
        F0 = _compute_level_higher(ell, 0.0, S, C0_levels, Cb_levels, C1_levels, y_powers, h0, beta, m, batch_shape, dtype)
        Fh = _compute_level_higher(ell, 0.5, S, C0_levels, Cb_levels, C1_levels, y_powers, h0, beta, m, batch_shape, dtype)
        F1 = _compute_level_higher(ell, 1.0, S, C0_levels, Cb_levels, C1_levels, y_powers, h0, beta, m, batch_shape, dtype)

        d_half = Fh - F0
        d_full = F1 - F0
        denom_beta = (h0 ** beta) * (theta_beta - theta_half)
        denom_one = h0 * (theta_beta - theta_half)

        C0_levels.append(F0)
        Cb_levels.append((d_half - theta_half * d_full) / denom_beta)
        C1_levels.append((theta_beta * d_full - d_half) / denom_one)

    # Trajectory at all S+1 nodes: F^ell(θ=0) evaluated with out_len=S+1.
    out_levels: list[Array] = [jnp.ones((S + 1,) + batch_shape + (1,), dtype=dtype)]
    for ell in range(1, trunc + 1):
        out_levels.append(
            _compute_level_higher(ell, 0.0, S + 1, C0_levels, Cb_levels, C1_levels, y_powers, h0, beta, m, batch_shape, dtype)
        )

    if output_starting_point:
        out = tuple(out_levels)
        if axis_norm != 0:
            out = tuple(jnp.moveaxis(lev, 0, axis_norm) for lev in out)
        return out
    return tuple(lev[-1] for lev in out_levels)


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _causal_conv_fft(src: Array, w: Array, *, out_len: int) -> Array:
    """Causal discrete convolution out[t] = sum_{s: lag=t-s >= 1} src[s]*w[lag] via FFT.

    src shape: (S, *trailing)
    w shape:   (out_len,)
    output:    (out_len, *trailing)
    """
    S = src.shape[0]
    trailing = src.shape[1:]
    src_flat = src.reshape((S, -1))
    nfft = _next_pow2(S + w.shape[0] - 1)
    src_pad = jnp.pad(src_flat, ((0, nfft - S), (0, 0)))
    w_pad = jnp.pad(w, (0, nfft - w.shape[0]))
    out_flat = jnp.fft.irfft(
        jnp.fft.rfft(src_pad, axis=0) * jnp.fft.rfft(w_pad)[:, None],
        n=nfft, axis=0,
    )
    return out_flat[:out_len].reshape((out_len,) + trailing)


def _outer_tensor(C: Array, Yq: Array, *, m: int, out_level: int) -> Array:
    """Pointwise outer product C_i ⊗ Yq_i along the leading source axis.

    C shape:   (S, *batch, m^a)
    Yq shape:  (S, *batch, m^b)  where a + b == out_level
    output:    (S, *batch, m^out_level)
    """
    prod = C[..., :, None] * Yq[..., None, :]
    return prod.reshape(C.shape[:-1] + (m ** out_level,))


def _frac_lag_weights(
    *,
    out_len: int,
    h: Array,
    theta: Array,
    q: int,
    rho: Array,
    beta: Array,
    dtype: jnp.dtype,
) -> Array:
    r"""Lag-k weights w_{q,ρ}(k+θ) for the fractional order=1 higher-order scheme.

    For lag k >= 1, D = (k + θ) * h and

        w = Γ(ρ+1)/Γ(ρ+qβ+1) * D^{ρ+qβ} * I_{h/D}(ρ+(q-1)β+1, β) / h^q.

    Lag 0 is always set to zero (strict causal constraint i < j).
    """
    lag = jnp.arange(out_len, dtype=dtype)
    valid = lag >= 1.0
    D = (lag + theta) * h
    safe_D = jnp.where(valid, D, jnp.ones_like(D))
    z = jnp.clip(h / safe_D, 0.0, 1.0)
    alpha_b = rho + (q - 1) * beta + 1.0
    log_pf = jsp.gammaln(rho + 1.0) - jsp.gammaln(rho + q * beta + 1.0)
    w = (
        jnp.exp(log_pf)
        * (safe_D ** (rho + q * beta))
        * jsp.betainc(alpha_b, beta, z)
        / (h ** q)
    )
    return jnp.where(valid, w, jnp.zeros_like(w)).astype(dtype)


def _compute_level_higher(
    ell: int,
    theta: float,
    out_len: int,
    C0_levels: list[Array],
    Cb_levels: list[Array],
    C1_levels: list[Array],
    y_powers: list[Array],
    h: Array,
    beta: Array,
    m: int,
    batch_shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    r"""F_j^ell(θ) for the order=1 higher-order fractional scheme.

    Computes:
        F_j^ell(θ) = sum_{i<j} sum_{q=1}^{ell} sum_{ρ∈{0,β,1}}
                       w_{q,ρ}(j-i+θ) * C_{i,ell-q,ρ} ⊗ y_i^⊗q

    via FFT causal convolution.  Output shape: (out_len, *batch, m^ell).
    """
    theta_arr = jnp.asarray(theta, dtype=dtype)
    rho_zero = jnp.asarray(0.0, dtype=dtype)
    rho_one = jnp.asarray(1.0, dtype=dtype)
    rho_values = (rho_zero, beta, rho_one)
    C_lists = (C0_levels, Cb_levels, C1_levels)

    acc = jnp.zeros((out_len,) + batch_shape + (m ** ell,), dtype=dtype)

    for q in range(1, ell + 1):
        Yq = y_powers[q]
        for rho, C_levels in zip(rho_values, C_lists):
            C_src = C_levels[ell - q]
            src = _outer_tensor(C_src, Yq, m=m, out_level=ell)
            w = _frac_lag_weights(
                out_len=out_len, h=h, theta=theta_arr,
                q=q, rho=rho, beta=beta, dtype=dtype,
            )
            acc = acc + _causal_conv_fft(src, w, out_len=out_len)

    return acc


__all__ = ["vsig", "vsig_fft"]
