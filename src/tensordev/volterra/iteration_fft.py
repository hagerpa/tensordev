# TODO: Implement q>1 in FFT branch.
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from tensordev.core.jax import Jax
from tensordev.core.universal import DenseElem
from tensordev.volterra.kernel import ConvolutionKernel
from tensordev.volterra.iteration import (
    _refine_dt,
    _normalize_times,
    _make_unit,
    BasisExpansionSpec,
    _basis_rhos,
    _chebyshev_lobatto_thetas,
    _basis_interpolation_matrix,
)

Array = jax.Array

_CORE = Jax()


@dataclass(frozen=True, slots=True)
class FFTContext:
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
class LagFFTTable:
    """
    weights[n - 1][b] is the FFT of the lag-weight sequence for tensor
    increment order n and basis component b.
    """

    weights: tuple[tuple[Array, ...], ...]
    nfft: int
    out_len: int


@dataclass(frozen=True, slots=True)
class PrecomputedLagTables:
    """Build once with :func:`precompute_lag_tables` and pass to
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
        kernel: ConvolutionKernel,
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
        FFT scheme order (0, 1, or 2).
    trunc:
        Tensor truncation level.
    dtype:
        Floating-point dtype.

    Returns
    -------
    PrecomputedLagTables
        Pass directly to ``vsig_fft(..., lag_tables=...)``.
    """
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1, or 2, got {order}.")

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
        kernel: ConvolutionKernel,
        trunc: int,
        dt: Array | float = 1.0,
        axis: int = -2,
        output_starting_point: bool = False,
        increment_input: bool = False,
        dyadic_order: int = 0,
        order: int = 0,
        lag_tables: PrecomputedLagTables | None = None,
) -> DenseElem:
    r"""Volterra signature via FFT convolution on a uniform grid.

    Requires a scalar kernel (``n == 1``) and a **uniform** time grid.  For
    non-uniform grids or ``n > 1`` use :func:`vsig` instead.

    Parameters
    ----------
    X:
        Path nodes or increments.  The trailing axis is the path dimension
        ``kernel.path_dim``; ``axis`` is the step/node axis.  Set
        ``increment_input=True`` to skip :func:`jnp.diff`.
    kernel:
        Volterra kernel.  Must satisfy ``n == 1``.
    trunc:
        Tensor truncation level (positive integer).
    dt:
        Uniform step size (scalar) or a 1-D array of per-step sizes.  The
        FFT scheme derives a single step ``h = times[1] - times[0]`` and
        treats the grid as uniform; passing a non-uniform array gives
        incorrect results without a runtime error.
    axis:
        Step/node axis of ``X``.
    output_starting_point:
        If ``False`` (default), return the terminal signature.  If ``True``,
        return the full trajectory ``[1, V_0, ..., V_{S-1}]`` with the
        trajectory axis at ``axis``.
    increment_input:
        Treat ``X`` as increments rather than path nodes.
    dyadic_order:
        Non-negative integer.  Each increment is split into
        ``2**dyadic_order`` equal sub-increments.  ``0`` (default) leaves
        the path unchanged.
    order:
        ``0``:
            Order-0 scheme (constant basis ``{1}``).

        ``1``:
            Basis-expansion scheme with fractional basis ``{1, s^beta, s}``.

        ``2``:
            Basis-expansion scheme with fractional basis
            ``{1, s^beta, s, s^(beta+1), s^2}``.

    lag_tables:
        Optional precomputed lag FFT tables from :func:`precompute_lag_tables`.
        When supplied the kernel-dependent weight computation is skipped.
        The tables must match the effective ``S`` (after dyadic refinement),
        ``trunc``, and ``order``; a :exc:`ValueError` is raised otherwise.

    Returns
    -------
    DenseElem
        Terminal signature by default.  With ``output_starting_point=True``,
        each level carries an additional trajectory axis at ``axis``.

    Raises
    ------
    ValueError
        For invalid truncation, dyadic order, axis, or scheme order.
    NotImplementedError
        For ``n > 1`` kernels.
    """
    if trunc <= 0:
        raise ValueError(f"trunc must be positive, got {trunc}.")
    if dyadic_order < 0:
        raise ValueError(f"dyadic_order must be non-negative, got {dyadic_order}.")
    if order not in (0, 1, 2):
        raise ValueError(f"order must currently be 0, 1, or 2, got {order}.")
    if kernel.q != 1:
        raise NotImplementedError(
            f"vsig_fft only supports scalar kernels (n=1); got n={kernel.q}."
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

    if lag_tables is not None:
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


def _run_basis_fft(
        *,
        ctx: FFTContext,
        kernel: ConvolutionKernel,
        order: int,
        lag_tables: PrecomputedLagTables | None = None,
) -> DenseElem:
    """Basis-expansion FFT scheme (order 0, 1, or 2)."""

    spec = _basis_spec(ctx=ctx, kernel=kernel, order=order)
    B = len(spec.rhos)

    # Initialise level-0 histories: component 0 is all-ones, the rest zero.
    components: tuple[DenseElem, ...] = tuple(
        (jnp.ones((ctx.S,) + ctx.batch_shape + (1,), dtype=ctx.dtype),)
        if b == 0
        else (jnp.zeros((ctx.S,) + ctx.batch_shape + (1,), dtype=ctx.dtype),)
        for b in range(B)
    )

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
            _compute_basis_level(ell, ctx=ctx, components=components, table=table)
            for table in theta_tables
        )
        components = _append_interpolated_basis_level(
            components=components,
            evaluations=evaluations,
            interpolation_inverse=spec.interpolation_inverse,
        )

    return _basis_output_levels(ctx=ctx, components=components, table=output_table)


def _basis_spec(
        *,
        ctx: FFTContext,
        kernel: ConvolutionKernel,
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


def _append_interpolated_basis_level(
        *,
        components: tuple,
        evaluations: DenseElem,
        interpolation_inverse: Array,
) -> tuple:
    """Append one level of basis coefficients from point evaluations."""

    F = jnp.stack(evaluations, axis=0)
    all_coeffs = jnp.tensordot(interpolation_inverse, F, axes=1)

    return tuple(
        levels + (all_coeffs[b],)
        for b, levels in enumerate(components)
    )


def _basis_output_levels(
        *,
        ctx: FFTContext,
        components: tuple,
        table: LagFFTTable,
) -> DenseElem:
    """Build trajectory levels from the final basis-expansion state."""

    out_levels: list[Array] = [
        jnp.ones((ctx.S + 1,) + ctx.batch_shape + (1,), dtype=ctx.dtype)
    ]

    for ell in range(1, ctx.trunc + 1):
        out_levels.append(
            _compute_basis_level(ell, ctx=ctx, components=components, table=table)
        )

    return tuple(out_levels)


def _make_lag_fft_table(
        *,
        kernel: ConvolutionKernel,
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

    for n in range(1, trunc + 1):
        cols = []

        for rho in rhos:
            w = kernel.lag_weights(
                out_len=out_len,
                h=h,
                theta=theta,
                n=n,
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
        components: tuple,
        table: LagFFTTable,
) -> Array:
    r"""Compute one level of the basis-expansion FFT scheme.

    .. math::

        F_j^\ell(\theta) = \sum_{i < j}
            \sum_{n=1}^{\ell} \sum_b w_{n,b}(j-i+\theta) C_{i,\ell-n,b} \otimes y_i^{\otimes n}
    """
    B = len(components)

    srcs = jnp.stack(
        [
            _CORE.tensor_product_homogeneous(components[b][ell - q_ord], ctx.y_powers[q_ord])
            for q_ord in range(1, ell + 1)
            for b in range(B)
        ],
        axis=0,
    )
    Ws = jnp.stack(
        [
            table.weights[q_ord - 1][b]
            for q_ord in range(1, ell + 1)
            for b in range(B)
        ],
        axis=0,
    )

    return jnp.sum(
        _causal_conv_fft_batched(srcs, Ws, nfft=table.nfft, out_len=table.out_len),
        axis=0,
    )


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

    srcs_flat = srcs.reshape((B, S, -1))
    SRC = jnp.fft.rfft(srcs_flat, n=nfft, axis=1)
    out_flat = jnp.fft.irfft(
        SRC * Ws[:, :, None],
        n=nfft,
        axis=1,
    )
    return out_flat[:, :out_len].reshape((B, out_len) + trailing)


__all__ = ["vsig_fft", "precompute_lag_tables", "PrecomputedLagTables"]
