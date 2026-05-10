#TODO: Implement higher order scheme for general non-uniform grid branch.
#TODO: Implement q>1 in FFT branch.
from __future__ import annotations

from dataclasses import replace, dataclass

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
        dt: Array | float = 1.0,
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
    dt:
        Step size(s).  A scalar gives a uniform grid; a 1-D array of length
        ``S`` gives a non-uniform grid via cumulative sums (default ``1.0``).
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
        ``1 / 2**dyadic_order``) and ``dt`` is refined accordingly.
        ``dyadic_order=0`` (default) leaves the path unchanged.

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
        dt = _refine_dt(dt, factor=factor, dtype=dtype)

    projected = jnp.einsum("qmd,...d->...qm", kernel.A.astype(dtype), dX)
    y = projected[..., 0, :] if kernel.q == 1 else projected

    y_time = jnp.moveaxis(y, axis_norm, 0)
    y_time = _normalize_projected_y_time(y_time, kernel)
    times_arr = _normalize_times(dt, S=S, dtype=dtype)

    batch_shape = tuple(y_time.shape[1:-1]) if kernel.q == 1 else tuple(y_time.shape[1:-2])
    unit = _make_unit(batch_shape=batch_shape, m=kernel.m, trunc=trunc, dtype=dtype)
    history0 = _make_history_seed(S=S, unit=unit, m=kernel.m, trunc=trunc, dtype=dtype)
    source = jnp.arange(S, dtype=jnp.int32)

    # Precomputing the full (S × S) coefficient grid in one batched call before
    # the scan is much more efficient than evaluating one column at a time inside
    # the scan body (avoids repeated betainc / quadrature calls).
    _precomp = kernel.coef_grid(times_arr, trunc=trunc, dtype=dtype)

    def step(history: DenseElem, j: Array) -> tuple[DenseElem, DenseElem]:
        v_prev = tuple(level[:S] for level in history)
        coef_j = _precomp[:, j]
        coef_j = _insert_singleton_batch_axes(coef_j, batch_ndim=len(batch_shape))
        terms = eval_vte_scalar(v_prev, y_time, coef_j) if coef_j.q == 1 else eval_vte_general(v_prev, y_time, coef_j)
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


def _refine_dt(
        dt: Array | float,
        *,
        factor: int,
        dtype: jnp.dtype,
) -> Array | float:
    """Subdivide ``dt`` for dyadic refinement by ``factor``.

    Scalar ``dt`` divides by ``factor``.  A per-step array produces a refined
    array repeated ``factor`` times per original step.
    """
    dt_arr = jnp.asarray(dt, dtype=dtype)
    if dt_arr.ndim == 0:
        return dt_arr / factor
    return jnp.repeat(dt_arr / factor, factor)


def _normalize_times(
        dt: Array | float,
        *,
        S: int,
        dtype: jnp.dtype,
) -> Array:
    """Return node times of shape ``(S + 1,)`` from a scalar or per-step ``dt``.

    - Scalar ``dt``: uniform grid ``[0, dt, 2·dt, …, S·dt]``.
    - 1-D array of length ``S``: ``[0, dt[0], dt[0]+dt[1], …]``.
    """
    dt_arr = jnp.asarray(dt, dtype=dtype)
    if dt_arr.ndim == 0:
        return jnp.arange(S + 1, dtype=dtype) * dt_arr
    if dt_arr.ndim == 1 and dt_arr.shape[0] == S:
        return jnp.concatenate([jnp.zeros((1,), dtype=dtype), jnp.cumsum(dt_arr)])
    raise ValueError(
        f"dt must be a scalar or a 1-D array of length S={S}, got shape {dt_arr.shape}."
    )


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


@dataclass(frozen=True, slots=True)
class FFTContext:
    """Shared prepared data for all FFT Volterra-signature schemes.

    This deliberately does not contain ``kernel``: kernels usually contain
    Python methods and configuration, so we pass them explicitly instead of
    making them part of the JAX pytree state.
    """

    y_time: Array
    y_powers: DenseElem
    times: Array
    h: Array
    unit: DenseElem

    S: int
    m: int
    trunc: int
    batch_shape: tuple[int, ...]
    dtype: jnp.dtype


@dataclass(frozen=True, slots=True)
class BasisExpansionSpec:
    """Basis/interpolation specification for higher-order FFT schemes.

    ``rhos`` and ``thetas`` have the same length ``B = (order+1)(order+2)//2``.
    The constant basis element ``s^0 = 1`` is always ``rhos[0] = 0``.
    """

    rhos: tuple[Array, ...]
    thetas: tuple[Array, ...]
    interpolation_inverse: Array


@dataclass(frozen=True, slots=True)
class BasisExpansionFFTState:
    """State for basis-expansion FFT schemes.

    components[b][ell] stores the level-ell history for basis component b.

    For the current order=1 scheme:

        components[0] == C0
        components[1] == Cb
        components[2] == C1
    """

    components: tuple[DenseElem, ...]
    spec: BasisExpansionSpec


@dataclass(frozen=True, slots=True)
class LagFFTTable:
    """Precomputed FFTs of lag weights.

    weights[q - 1][b] is the FFT of the lag-weight sequence for tensor
    increment order q and basis component b.
    """

    weights: tuple[tuple[Array, ...], ...]
    nfft: int
    out_len: int


@dataclass(frozen=True, slots=True)
class PrecomputedLagTables:
    """Lag FFT tables for one (kernel, grid, order, trunc) configuration.

    Build once with :func:`precompute_lag_tables` and pass to
    :func:`vsig_fft` via ``lag_tables=`` to skip the (potentially
    expensive) ``betainc`` / quadrature + rfft work on every call.

    Only useful for the basis-expansion FFT schemes (``order >= 1``).
    """

    theta_tables: tuple[LagFFTTable, ...]
    output_table: LagFFTTable
    S: int
    trunc: int
    order: int


def precompute_lag_tables(
    kernel: VolterraKernel,
    *,
    S: int,
    h: float | Array,
    order: int,
    trunc: int,
    dtype: jnp.dtype,
) -> PrecomputedLagTables:
    """Precompute lag FFT tables for :func:`vsig_fft`.

    Call once per (kernel, grid, order, trunc, dtype) configuration, then
    pass the result as ``lag_tables=`` to :func:`vsig_fft`.  This avoids
    recomputing the kernel-dependent weights on every call when only the
    path data changes.

    This approach is fully JAX JIT-friendly: the tables are plain JAX
    arrays computed eagerly outside of any JIT boundary.  Under
    ``jax.jit`` the arrays are captured as XLA constants, so there is no
    runtime overhead.

    Parameters
    ----------
    kernel:
        Volterra kernel.
    S:
        Number of increments (after any dyadic refinement).
    h:
        Uniform step size.
    order:
        FFT scheme order (1 or 2).  Order 0 (Horner/Toeplitz) does not use
        lag tables.
    trunc:
        Tensor truncation level.
    dtype:
        Floating-point dtype.

    Returns
    -------
    PrecomputedLagTables
        Pass directly to ``vsig_fft(..., lag_tables=...)``.
    """
    if order == 0:
        raise ValueError(
            "order=0 (Horner/Toeplitz) does not use lag tables; "
            "precompute_lag_tables is only needed for order >= 1."
        )
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}.")

    dtype_ = jnp.dtype(dtype)
    h_arr = jnp.asarray(h, dtype=dtype_)
    beta = jnp.atleast_1d(kernel.beta)[0].astype(dtype_)
    rhos = _basis_rhos(order, beta=beta, dtype=dtype_)
    thetas = _chebyshev_lobatto_thetas(n=len(rhos), dtype=dtype_)

    theta_tables = tuple(
        _make_lag_fft_table(
            kernel=kernel,
            S=S,
            h=h_arr,
            trunc=trunc,
            dtype=dtype_,
            out_len=S,
            theta=theta,
            rhos=rhos,
        )
        for theta in thetas
    )
    output_table = _make_lag_fft_table(
        kernel=kernel,
        S=S,
        h=h_arr,
        trunc=trunc,
        dtype=dtype_,
        out_len=S + 1,
        theta=jnp.asarray(0.0, dtype=dtype_),
        rhos=rhos,
    )
    return PrecomputedLagTables(
        theta_tables=theta_tables,
        output_table=output_table,
        S=S,
        trunc=trunc,
        order=order,
    )


def vsig_fft(
        X: Array,
        *,
        kernel: VolterraKernel,
        trunc: int,
        dt: Array | float = 1.0,
        axis: int = -2,
        output_starting_point: bool = False,
        increment_input: bool = False,
        dyadic_order: int = 0,
        order: int = 0,
        lag_tables: PrecomputedLagTables | None = None,
) -> DenseElem:
    r"""Level-by-level Volterra signature using FFT convolution.

    The implementation is organized around FFT schemes rather than hard-coded
    ``order == 0`` / ``order == 1`` internals.

    ``order=0`` uses the Horner/Toeplitz scheme.

    ``order>=1`` uses a basis-expansion scheme.  ``order=1`` uses the
    fractional three-point basis ``{1, s^beta, s}``; ``order=2`` uses the
    five-point basis ``{1, s^beta, s, s^(beta+1), s^2}``.

    Parameters
    ----------
    X, kernel, trunc, times, dt, axis, output_starting_point,
    increment_input, dyadic_order:
        Same semantics as :func:`vsig`.

    order:
        FFT quadrature/scheme order.

        ``0``:
            Standard Horner/Toeplitz FFT scheme.  Supports convolution kernels
            ``fractional`` and ``gamma`` on a uniform grid.

        ``1``:
            Fractional basis-expansion scheme using the basis
            ``{1, s^beta, s}``.

        ``2``:
            Fractional basis-expansion scheme using the basis
            ``{1, s^beta, s, s^(beta+1), s^2}``.

        Higher-order schemes require ``kernel.kind == "fractional"`` and a
        uniform grid.

    lag_tables:
        Optional precomputed lag FFT tables returned by
        :func:`precompute_lag_tables`.  When supplied the kernel-dependent
        weight computation is skipped entirely, which is the main cost for
        higher-order schemes.  The tables must have been built for the same
        ``S`` (after dyadic refinement), ``trunc``, and ``order`` as this
        call; a :exc:`ValueError` is raised otherwise.  Ignored when
        ``order=0``.

    Returns
    -------
    DenseElem
        Terminal signature, or full trajectory if ``output_starting_point`` is
        true.

    Raises
    ------
    ValueError
        For invalid truncation, dyadic order, axis, or unsupported scheme order.

    NotImplementedError
        If the kernel is unsupported by the requested FFT scheme.
    """
    if trunc <= 0:
        raise ValueError(f"trunc must be positive, got {trunc}.")
    if dyadic_order < 0:
        raise ValueError(f"dyadic_order must be non-negative, got {dyadic_order}.")
    if order not in (0, 1, 2):
        raise ValueError(f"order must currently be 0, 1, or 2, got {order}.")
    if kernel.q != 1:
        raise NotImplementedError(
            f"vsig_fft only supports scalar kernels (q=1); got q={kernel.q}."
        )

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

    dX = X if increment_input else jnp.diff(X, axis=axis_norm)
    dX = dX.astype(dtype)

    S = dX.shape[axis_norm]
    if S == 0:
        raise ValueError("vsig_fft requires at least one increment.")

    if dyadic_order > 0:
        factor = 1 << int(dyadic_order)
        dX = jnp.repeat(dX / factor, factor, axis=axis_norm)
        S = dX.shape[axis_norm]
        dt = _refine_dt(dt, factor=factor, dtype=dtype)

    projected = jnp.einsum("qmd,...d->...qm", kernel.A.astype(dtype), dX)
    y_time = jnp.moveaxis(projected[..., 0, :], axis_norm, 0)

    times_arr = _normalize_times(dt, S=S, dtype=dtype)
    h = times_arr[1] - times_arr[0]

    batch_shape = tuple(y_time.shape[1:-1])
    m = kernel.m

    unit = _make_unit(
        batch_shape=batch_shape,
        m=m,
        trunc=trunc,
        dtype=dtype,
    )

    y_powers = _tensor_powers(
        y_time,
        trunc=trunc,
        dtype=dtype,
    )

    ctx = FFTContext(
        y_time=y_time,
        y_powers=y_powers,
        times=times_arr,
        h=h,
        unit=unit,
        S=S,
        m=m,
        trunc=trunc,
        batch_shape=batch_shape,
        dtype=dtype,
    )

    # Validate precomputed tables if provided.
    if lag_tables is not None and order != 0:
        if lag_tables.S != S:
            raise ValueError(
                f"lag_tables.S={lag_tables.S} does not match the effective S={S} "
                f"(after dyadic refinement).  Rebuild with precompute_lag_tables."
            )
        if lag_tables.trunc != trunc:
            raise ValueError(
                f"lag_tables.trunc={lag_tables.trunc} != trunc={trunc}."
            )
        if lag_tables.order != order:
            raise ValueError(
                f"lag_tables.order={lag_tables.order} != order={order}."
            )

    if order == 0:
        out_levels = _run_horner_fft(ctx=ctx, kernel=kernel)
    else:
        out_levels = _run_basis_fft(ctx=ctx, kernel=kernel, order=order, lag_tables=lag_tables)

    if output_starting_point:
        if axis_norm == 0:
            return tuple(out_levels)
        return tuple(jnp.moveaxis(level, 0, axis_norm) for level in out_levels)
    return tuple(level[-1] for level in out_levels)


def _tensor_powers(
        y_time: Array,
        *,
        trunc: int,
        dtype: jnp.dtype,
) -> DenseElem:
    """Compute ``y_time^{⊗r}`` for ``r = 0, ..., trunc``."""

    S = y_time.shape[0]
    batch_shape = tuple(y_time.shape[1:-1])

    powers: list[Array] = [
        jnp.ones((S,) + batch_shape + (1,), dtype=dtype),
        y_time,
    ]

    for _ in range(2, trunc + 1):
        powers.append(_CORE.tensor_product_homogeneous(powers[-1], y_time))

    return tuple(powers[: trunc + 1])


def _finish_fft_output(
        out_levels: DenseElem,
        *,
        axis_norm: int,
        output_starting_point: bool,
) -> DenseElem:
    """Return either the full trajectory or only the terminal signature."""

    if output_starting_point:
        if axis_norm == 0:
            return tuple(out_levels)
        return tuple(jnp.moveaxis(level, 0, axis_norm) for level in out_levels)

    return tuple(level[-1] for level in out_levels)


def _run_horner_fft(
        *,
        ctx: FFTContext,
        kernel: VolterraKernel,
) -> DenseElem:
    """Order-0 Horner/Toeplitz FFT scheme."""

    g_coef = kernel.coef(
        ctx.times[0],
        ctx.times[1],
        ctx.times[1:],
        trunc=ctx.trunc,
        dtype=ctx.dtype,
    )
    g = g_coef.alpha[..., 0, :]

    fft_n = _next_pow2(2 * ctx.S)
    G = jnp.fft.rfft(g, n=fft_n, axis=0)
    # Pre-expand G with trailing singletons for broadcasting against (batch, m^n).
    G_bcast = G.reshape(G.shape + (1,) * (len(ctx.batch_shape) + 1))

    out_levels: list[Array] = [
        jnp.ones((ctx.S + 1,) + ctx.batch_shape + (1,), dtype=ctx.dtype)
    ]

    for n in range(1, ctx.trunc + 1):
        acc = jnp.zeros(
            (ctx.S,) + ctx.batch_shape + (ctx.m ** n,),
            dtype=ctx.dtype,
        )

        for k in range(n):
            ell_idx = n - k - 1

            signal = _CORE.tensor_product_homogeneous(
                out_levels[k][: ctx.S],
                ctx.y_powers[n - k],
            )

            SIG = jnp.fft.rfft(signal, n=fft_n, axis=0)
            acc = acc + jnp.fft.irfft(
                G_bcast[:, ell_idx] * SIG,
                n=fft_n,
                axis=0,
            )[: ctx.S]

        out_levels.append(
            jnp.concatenate(
                [ctx.unit[n][None], ctx.unit[n] + acc],
                axis=0,
            )
        )

    return tuple(out_levels)


def _run_basis_fft(
        *,
        ctx: FFTContext,
        kernel: VolterraKernel,
        order: int,
        lag_tables: PrecomputedLagTables | None = None,
) -> DenseElem:
    """Higher-order basis-expansion FFT scheme.

    The structure is basis-generic: adding another order mostly means adding
    another ``_basis_spec_order*`` function.
    """

    spec = _basis_spec(
        ctx=ctx,
        kernel=kernel,
        order=order,
    )
    state = _init_basis_state(ctx=ctx, spec=spec)

    if lag_tables is not None:
        theta_tables = lag_tables.theta_tables
        output_table = lag_tables.output_table
    else:
        theta_tables = tuple(
            _make_lag_fft_table(
                kernel=kernel,
                S=ctx.S,
                h=ctx.h,
                trunc=ctx.trunc,
                dtype=ctx.dtype,
                out_len=ctx.S,
                theta=theta,
                rhos=spec.rhos,
            )
            for theta in spec.thetas
        )
        output_table = _make_lag_fft_table(
            kernel=kernel,
            S=ctx.S,
            h=ctx.h,
            trunc=ctx.trunc,
            dtype=ctx.dtype,
            out_len=ctx.S + 1,
            theta=jnp.asarray(0.0, dtype=ctx.dtype),
            rhos=spec.rhos,
        )

    for ell in range(1, ctx.trunc + 1):
        evaluations = tuple(
            _compute_basis_level(
                ell,
                ctx=ctx,
                state=state,
                table=table,
            )
            for table in theta_tables
        )

        state = _append_interpolated_basis_level(
            state=state,
            evaluations=evaluations,
        )

    return _basis_output_levels(
        ctx=ctx,
        state=state,
        table=output_table,
    )


def _basis_spec(
        *,
        ctx: FFTContext,
        kernel: VolterraKernel,
        order: int,
) -> BasisExpansionSpec:
    """Return the basis/interpolation specification for a higher-order scheme.

    The only thing that varies across orders is the exponent tuple ``rhos``.
    Chebyshev-Lobatto nodes are used for all orders; for order=1 (n=3) they
    coincide with ``{0, 1/2, 1}``.
    """
    beta = jnp.atleast_1d(kernel.beta)[0].astype(ctx.dtype)
    rhos = _basis_rhos(order, beta=beta, dtype=ctx.dtype)
    thetas = _chebyshev_lobatto_thetas(n=len(rhos), dtype=ctx.dtype)
    interpolation = _basis_interpolation_matrix(
        h=ctx.h,
        thetas=thetas,
        rhos=rhos,
        dtype=ctx.dtype,
    )
    return BasisExpansionSpec(
        rhos=rhos,
        thetas=thetas,
        interpolation_inverse=jnp.linalg.inv(interpolation),
    )


def _basis_rhos(
        order: int,
        *,
        beta: Array,
        dtype: jnp.dtype,
) -> tuple[Array, ...]:
    """Basis exponents for the fractional FFT scheme of the given order.

    order=1: ``{1, s^beta, s}``              → ``(0, beta, 1)``
    order=2: ``{1, s^beta, s, s^(beta+1), s^2}`` → ``(0, beta, 1, beta+1, 2)``
    """
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    if order == 1:
        return (zero, beta, one)
    if order == 2:
        return (zero, beta, one, beta + one, jnp.asarray(2.0, dtype=dtype))
    raise NotImplementedError(
        f"Basis-expansion FFT scheme order={order} is not implemented yet."
    )


def _chebyshev_lobatto_thetas(
        *,
        n: int,
        dtype: jnp.dtype,
) -> tuple[Array, ...]:
    """Chebyshev-Lobatto nodes on ``[0, 1]``.

    The nodes include both endpoints.  For ``n=5`` this gives

        0, 1/2 - sqrt(2)/4, 1/2, 1/2 + sqrt(2)/4, 1.
    """

    if n < 2:
        raise ValueError(f"Need at least two interpolation nodes, got n={n}.")

    k = jnp.arange(n, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    two = jnp.asarray(2.0, dtype=dtype)
    nodes = (one - jnp.cos(jnp.pi * k / jnp.asarray(n - 1, dtype=dtype))) / two

    return tuple(nodes[i] for i in range(n))


def _basis_interpolation_matrix(
        *,
        h: Array,
        thetas: tuple[Array, ...],
        rhos: tuple[Array, ...],
        dtype: jnp.dtype,
) -> Array:
    """Matrix mapping basis coefficients to point evaluations.

    Entry ``V[a, b]`` is the value of basis component ``b`` at interpolation
    point ``theta[a]``.
    """

    rows = []

    for theta in thetas:
        row = []

        for rho in rhos:
            is_const = rho == jnp.asarray(0.0, dtype=dtype)

            value = jnp.where(
                is_const,
                jnp.asarray(1.0, dtype=dtype),
                (theta * h) ** rho,
            )
            row.append(value)

        rows.append(jnp.stack(row, axis=0))

    return jnp.stack(rows, axis=0)


def _init_basis_state(
        *,
        ctx: FFTContext,
        spec: BasisExpansionSpec,
) -> BasisExpansionFFTState:
    """Initial level-0 basis histories."""

    components: list[DenseElem] = []

    for b in range(len(spec.rhos)):
        if b == 0:
            level0 = jnp.ones(
                (ctx.S,) + ctx.batch_shape + (1,),
                dtype=ctx.dtype,
            )
        else:
            level0 = jnp.zeros(
                (ctx.S,) + ctx.batch_shape + (1,),
                dtype=ctx.dtype,
            )

        components.append((level0,))

    return BasisExpansionFFTState(
        components=tuple(components),
        spec=spec,
    )


def _append_interpolated_basis_level(
        *,
        state: BasisExpansionFFTState,
        evaluations: DenseElem,
) -> BasisExpansionFFTState:
    """Append one level of basis coefficients from point evaluations."""

    F = jnp.stack(evaluations, axis=0)
    # One GEMM for all basis components instead of B separate dot products.
    all_coeffs = jnp.tensordot(state.spec.interpolation_inverse, F, axes=1)

    components = tuple(
        levels + (all_coeffs[b],)
        for b, levels in enumerate(state.components)
    )

    return BasisExpansionFFTState(
        components=components,
        spec=state.spec,
    )


def _basis_output_levels(
        *,
        ctx: FFTContext,
        state: BasisExpansionFFTState,
        table: LagFFTTable,
) -> DenseElem:
    """Build trajectory levels from the final basis-expansion state."""

    out_levels: list[Array] = [
        jnp.ones((ctx.S + 1,) + ctx.batch_shape + (1,), dtype=ctx.dtype)
    ]

    for ell in range(1, ctx.trunc + 1):
        out_levels.append(
            _compute_basis_level(
                ell,
                ctx=ctx,
                state=state,
                table=table,
            )
        )

    return tuple(out_levels)


def _make_lag_fft_table(
        *,
        kernel: VolterraKernel,
        S: int,
        h: Array,
        trunc: int,
        dtype: jnp.dtype,
        out_len: int,
        theta: Array,
        rhos: tuple[Array, ...],
) -> LagFFTTable:
    """Precompute FFTs of all lag weights needed for one interpolation point."""

    nfft = _next_pow2(S + out_len - 1)
    rows: list[tuple[Array, ...]] = []

    for q_ord in range(1, trunc + 1):
        cols = []

        for rho in rhos:
            w = kernel.lag_weights(
                out_len=out_len,
                h=h,
                theta=theta,
                q=q_ord,
                rho=rho,
                dtype=dtype,
            )[..., 0, 0]
            cols.append(jnp.fft.rfft(w, n=nfft))

        rows.append(tuple(cols))

    return LagFFTTable(
        weights=tuple(rows),
        nfft=nfft,
        out_len=out_len,
    )


def _compute_basis_level(
    ell: int,
    *,
    ctx: FFTContext,
    state: BasisExpansionFFTState,
    table: LagFFTTable,
) -> Array:
    r"""Compute one level for a basis-expansion FFT scheme.

    Computes

    .. math::

        F_j^\ell(\theta) = \sum_{i < j}
            \sum_{q=1}^{\ell} \sum_b w_{q,b}(j-i+\theta) C_{i,\ell-q,b} \otimes y_i^{\otimes q}

    using causal FFT convolutions.
    """

    acc = jnp.zeros(
        (table.out_len,) + ctx.batch_shape + (ctx.m ** ell,),
        dtype=ctx.dtype,
    )

    for q_ord in range(1, ell + 1):
        Yq = ctx.y_powers[q_ord]
        # Stack all B source arrays and their precomputed weight FFTs so XLA
        # can execute one batched FFT instead of B sequential ones.
        srcs = jnp.stack(
            [_CORE.tensor_product_homogeneous(levels[ell - q_ord], Yq)
             for levels in state.components],
            axis=0,
        )  # (B, S, ..., m^ell)
        Ws = jnp.stack(
            [table.weights[q_ord - 1][b] for b in range(len(state.components))],
            axis=0,
        )  # (B, nfft//2+1)
        acc = acc + jnp.sum(
            _causal_conv_fft_batched(srcs, Ws, nfft=table.nfft, out_len=table.out_len),
            axis=0,
        )

    return acc



def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _causal_conv_fft_batched(
        srcs: Array,
        Ws: Array,
        *,
        nfft: int,
        out_len: int,
) -> Array:
    """Batched causal FFT convolution over B source/weight pairs.

    Args:
        srcs: ``(B, S, ..., m^ell)``
        Ws:   ``(B, nfft//2+1)`` — precomputed rfft of lag weights.

    Returns:
        ``(B, out_len, ..., m^ell)``
    """
    B, S = srcs.shape[:2]
    trailing = srcs.shape[2:]

    srcs_flat = srcs.reshape((B, S, -1))  # (B, S, prod(trailing))
    SRC = jnp.fft.rfft(srcs_flat, n=nfft, axis=1)  # (B, nfft//2+1, prod(trailing))
    out_flat = jnp.fft.irfft(
        SRC * Ws[:, :, None],
        n=nfft,
        axis=1,
    )  # (B, nfft, prod(trailing))
    return out_flat[:, :out_len].reshape((B, out_len) + trailing)


__all__ = ["vsig", "vsig_fft", "precompute_lag_tables", "PrecomputedLagTables"]
