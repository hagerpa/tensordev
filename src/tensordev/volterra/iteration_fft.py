# FFT-based Volterra signature for uniform grids.
# Supports both q = 1 (scalar fast path, ordinary tensor powers) and
# q > 1 (multi-component path, normalized shuffle monomials and (p, ell) channels).
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tensordev.core.jax import Jax
from tensordev.core.universal import DenseElem
from tensordev.util.combinatorics import build_multiindex_layout, multiindex_batched_navigation
from tensordev.volterra.kernel import ConvolutionKernel
from tensordev.volterra.iteration import (
    _normalize_times,
    _make_unit,
    BasisExpansionSpec,
    _basis_rhos_multicomp,
    _chebyshev_lobatto_thetas,
    _basis_interpolation_matrix,
)

Array = jax.Array

_CORE = Jax()


@dataclass(frozen=True, slots=True)
class FFTContext:
    """Preprocessing context shared by all FFT scheme variants.

    Attributes
    ----------
    y:
        Projected increments with shape ``(S, *batch_shape, q, m)``.
        The q-axis is always present; for ``kernel.q == 1`` it has size 1.
    y_powers:
        Tuple of tensors ``y_scalar^{⊗r}`` for ``r = 0, ..., trunc``,
        where ``y_scalar = y[..., 0, :]``.  Used exclusively by the scalar
        (q = 1) fast path, which avoids the overhead of full multi-index
        enumeration.  Set to ``None`` for q > 1, where the source channels
        are built via :func:`_shuffle_monomials_by_degree` and
        :func:`_local_multicomp_channels` instead.
    """

    y: Array
    y_powers: DenseElem | None
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
    rhos = _basis_rhos_multicomp(order, betas=kernel.beta, dtype=dtype_)
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


def fft_iteration(
        dX: Array,
        *,
        kernel: ConvolutionKernel,
        trunc: int,
        dt: Array | float = 1.0,
        axis: int = -2,
        return_trajectory: bool = False,
        order: int = 0,
        lag_tables: PrecomputedLagTables | None = None,
) -> DenseElem:
    r"""Volterra signature via FFT convolution on a uniform grid.

    Expects ``dX`` already in increment form and on the final time grid.
    All preprocessing is the responsibility of the caller — typically the
    high-level :func:`~tensordev.volterra.signature.vsig`.

    Requires a **uniform** time grid; for non-uniform grids use
    :func:`quadratic_iteration` instead.  Supports both scalar
    (``kernel.q == 1``) and multi-component (``kernel.q > 1``) kernels.

    Parameters
    ----------
    dX:
        Increments.  Shape ``(*batch, S, d)`` with step axis at ``axis``
        and trailing path dimension ``d = kernel.path_dim``.
    kernel:
        Volterra kernel.
    trunc:
        Tensor truncation level (positive integer).
    dt:
        Uniform step size scalar (default ``1.0``).  Passing a 1-D array
        silently uses only ``times[1] - times[0]`` as the step size.
    axis:
        Step axis of ``dX`` (default ``-2``).
    return_trajectory:
        If ``True``, return ``[V_1, ..., V_S]`` with the step axis at
        ``axis``.  If ``False`` (default), return the terminal ``V_S``.
    order:
        ``0`` (default) constant basis; ``1`` fractional basis
        ``{1, s^beta, s}``; ``2`` extended basis
        ``{1, s^beta, s, s^(beta+1), s^2}``.
    lag_tables:
        Optional precomputed lag FFT tables from :func:`precompute_lag_tables`.
        Must match ``S``, ``trunc``, and ``order``.

    Returns
    -------
    DenseElem
        Terminal signature, or full trajectory when ``return_trajectory=True``.

    Raises
    ------
    ValueError
        For invalid truncation, axis, or scheme order.
    """
    if trunc <= 0:
        raise ValueError(f"trunc must be positive, got {trunc}.")
    if order not in (0, 1, 2):
        raise ValueError(f"order must currently be 0, 1, or 2, got {order}.")

    dX = jnp.asarray(dX)
    if dX.ndim < 2:
        raise ValueError("dX must have at least a step axis and a trailing path dimension.")

    axis_norm = axis % dX.ndim
    if axis_norm == dX.ndim - 1:
        raise ValueError("axis must identify the step axis, not the trailing path dimension.")
    if dX.shape[-1] != kernel.path_dim:
        raise ValueError(
            f"dX trailing dimension must be {kernel.path_dim}, got {dX.shape[-1]}."
        )

    dtype = dX.dtype
    dX = dX.astype(dtype)

    S = dX.shape[axis_norm]
    if S == 0:
        raise ValueError("fft_iteration requires at least one increment.")

    projected = jnp.einsum("qmd,...d->...qm", kernel.A.astype(dtype), dX)
    # y has shape (S, *batch_shape, q, m) — q-axis always present.
    y = jnp.moveaxis(projected, axis_norm, 0)

    times_arr = _normalize_times(dt, S=S, dtype=dtype)
    h = times_arr[1] - times_arr[0]

    # batch_shape strips the leading S and the trailing (q, m) axes.
    batch_shape = tuple(y.shape[1:-2])
    m = kernel.m

    unit = _make_unit(
        batch_shape=batch_shape,
        m=m,
        trunc=trunc,
        dtype=dtype,
    )

    # q = 1 scalar fast path: build tensor powers from the single q = 0 slice.
    # This avoids all multi-index overhead.
    # q > 1: y_powers is not used; the multi-component path builds normalized
    # shuffle monomials on demand inside _run_basis_fft.
    if kernel.q == 1:
        y_scalar = y[..., 0, :]  # shape: (S, *batch_shape, m)
        y_powers: DenseElem | None = _tensor_powers(
            y_scalar,
            trunc=trunc,
            dtype=dtype,
        )
    else:
        y_powers = None

    ctx = FFTContext(
        y=y,
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

    if return_trajectory:
        # Drop the leading unit (V_0) so the trajectory is [V_1, ..., V_S],
        # matching the S-entry convention of quadratic_iteration.
        traj = tuple(level[1:] for level in out_levels)
        if axis_norm == 0:
            return traj
        return tuple(jnp.moveaxis(level, 0, axis_norm) for level in traj)

    return tuple(level[-1] for level in out_levels)


def _tensor_powers(
        y_scalar: Array,
        *,
        trunc: int,
        dtype: jnp.dtype,
) -> DenseElem:
    """Compute ``y_scalar^{⊗r}`` for ``r = 0, ..., trunc``.

    ``y_scalar`` has shape ``(S, *batch_shape, m)`` — the q=0 slice of
    the projected increments.
    """

    S = y_scalar.shape[0]
    batch_shape = tuple(y_scalar.shape[1:-1])

    powers: list[Array] = [
        jnp.ones((S,) + batch_shape + (1,), dtype=dtype),
        y_scalar,
    ]

    for _ in range(2, trunc + 1):
        powers.append(_CORE.tensor_product_homogeneous(powers[-1], y_scalar))

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
    """Basis-expansion FFT scheme (order 0, 1, or 2).

    Dispatches internally between two source-construction strategies:

    q = 1 (scalar fast path)
        Source channels are ordinary tensor powers ``y^{⊗r}`` stored in
        ``ctx.y_powers``.  No shuffle-monomial overhead.

    q > 1 (multi-component path)
        Normalized shuffle monomials are built once via
        :func:`_shuffle_monomials_by_degree`, then individual output levels
        are computed by :func:`_compute_basis_level_multicomp`, which uses
        ``(p, ell)`` source channels with channel order
        ``p * M_{r-1} + ell_index``.

    In both cases the persistent state is ``components[b][level]``; the
    ``(p, ell)`` multi-index appears only in the temporary FFT source
    channels and never in the state.
    """

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

    if kernel.q == 1:
        # Scalar fast path: tensor powers already in ctx.y_powers; no monomials.
        monomials = None
    else:
        # Multi-component path: build normalized shuffle monomials once.
        # For output level n, local order r ranges 1..n, and we need monomials[r-1].
        # The maximum needed degree is therefore ctx.trunc - 1.
        monomials = _shuffle_monomials_by_degree(
            ctx.y, trunc=ctx.trunc - 1, dtype=ctx.dtype
        )

    for ell in range(1, ctx.trunc + 1):
        if monomials is None:
            evaluations = tuple(
                _compute_basis_level_scalar(ell, ctx=ctx, components=components, table=table)
                for table in theta_tables
            )
        else:
            evaluations = tuple(
                _compute_basis_level_multicomp(
                    ell, ctx=ctx, components=components, table=table, monomials=monomials
                )
                for table in theta_tables
            )
        components = _append_interpolated_basis_level(
            components=components,
            evaluations=evaluations,
            interpolation_inverse=spec.interpolation_inverse,
        )

    return _basis_output_levels(
        ctx=ctx, components=components, table=output_table, monomials=monomials
    )


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
    rhos = _basis_rhos_multicomp(order, betas=kernel.beta, dtype=ctx.dtype)
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
        monomials: tuple[Array, ...] | None,
) -> DenseElem:
    """Build trajectory levels from the final basis-expansion state.

    Dispatches to :func:`_compute_basis_level_scalar` (q = 1) or
    :func:`_compute_basis_level_multicomp` (q > 1) based on whether
    ``monomials`` is ``None``.
    """

    out_levels: list[Array] = [
        jnp.ones((ctx.S + 1,) + ctx.batch_shape + (1,), dtype=ctx.dtype)
    ]

    for ell in range(1, ctx.trunc + 1):
        if monomials is None:
            out_levels.append(
                _compute_basis_level_scalar(ell, ctx=ctx, components=components, table=table)
            )
        else:
            out_levels.append(
                _compute_basis_level_multicomp(
                    ell, ctx=ctx, components=components, table=table, monomials=monomials
                )
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
    """Dispatch to the scalar or multi-component lag FFT table constructor.

    q = 1  →  :func:`_make_lag_fft_table_scalar`
    q > 1  →  :func:`_make_lag_fft_table_multicomp`
    """
    if kernel.q == 1:
        return _make_lag_fft_table_scalar(
            kernel=kernel,
            S=S,
            h=h,
            trunc=trunc,
            dtype=dtype,
            out_len=out_len,
            theta=theta,
            rhos=rhos,
        )
    return _make_lag_fft_table_multicomp(
        kernel=kernel,
        S=S,
        h=h,
        trunc=trunc,
        dtype=dtype,
        out_len=out_len,
        theta=theta,
        rhos=rhos,
    )


def _make_lag_fft_table_scalar(
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
    """Precompute FFTs of all lag weights needed for one interpolation point.

    Scalar fast path (``kernel.q == 1``): exploits the fact that the kernel
    matrix is 1×1 so only the ``[0, 0]`` entry is extracted.

    ``weights[n-1][b]`` has shape ``(nfft//2+1,)`` — one frequency vector per
    (local order n, basis component b).
    """

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


def _make_lag_fft_table_multicomp(
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
    """Precompute FFTs of all lag weights for one interpolation point (q > 1).

    ``kernel.lag_weights(...)`` returns shape ``(out_len, q, M_{n-1})``,
    where ``M_{n-1}`` is the number of packed multi-indices of degree ``n-1``
    for ``q`` components.  The ``(q, M_{n-1})`` trailing axes are flattened
    into a single channel axis using the **same ordering as the source
    channels** built by :func:`_local_multicomp_channels`:

    .. code-block:: text

        channel = p * M_{n-1} + ell_index

    so that weight and source channels are always aligned.

    ``weights[n-1][b]`` has shape ``(q * M_{n-1}, nfft//2+1)`` — one
    frequency vector per (lag channel, basis component b).
    """
    q = kernel.q
    nfft = _next_pow2(S + out_len - 1)
    rows: list[tuple[Array, ...]] = []

    for n in range(1, trunc + 1):
        cols = []

        for rho in rhos:
            # w: (out_len, q, M_{n-1})
            w = kernel.lag_weights(
                out_len=out_len,
                h=h,
                theta=theta,
                n=n,
                rho=rho,
                dtype=dtype,
            )
            M = w.shape[-1]
            assert w.shape == (out_len, q, M), (
                f"lag_weights returned unexpected shape {w.shape}; "
                f"expected (out_len={out_len}, q={q}, M_{n-1}={M})."
            )
            # Flatten (q, M) into a single channel axis: (out_len, q*M)
            w_flat = w.reshape(out_len, q * M)
            # rfft over the lag axis → (nfft//2+1, q*M)
            W_freq = jnp.fft.rfft(w_flat, n=nfft, axis=0)
            # Transpose to channel-first: (q*M, nfft//2+1)
            cols.append(W_freq.T)

        rows.append(tuple(cols))

    return LagFFTTable(
        weights=tuple(rows),
        nfft=nfft,
        out_len=out_len,
    )


def _compute_basis_level_scalar(
        ell: int,
        *,
        ctx: FFTContext,
        components: tuple,
        table: LagFFTTable,
) -> Array:
    r"""Compute one level of the basis-expansion FFT scheme (scalar fast path).

    Scalar fast path (``kernel.q == 1``): uses ordinary tensor powers
    ``y^{⊗r}`` rather than a generic multi-index construction.

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


def _local_multicomp_channels(
        monomials: tuple[Array, ...],
        y: Array,
        r: int,
) -> Array:
    r"""Build batched local source channels for all ``(p, ell)`` at local order ``r``.

    For local order ``r``, the source assigned to channel
    ``p * M_{r-1} + ell_index`` is

    .. math::

        M_\ell(y_i) \otimes y_{i,p}

    where ``|\ell| = r-1`` and ``p = 0, \ldots, q-1``.

    This channel ordering matches :func:`_make_lag_fft_table_multicomp`
    exactly, so weight channel ``p * M_{r-1} + ell_index`` and source
    channel ``p * M_{r-1} + ell_index`` always correspond to the same
    ``(p, ell)`` pair.

    Parameters
    ----------
    monomials:
        Output of :func:`_shuffle_monomials_by_degree`; ``monomials[k]`` has
        shape ``(S, *batch, M_k, m**k)``.
    y:
        Projected increments, shape ``(S, *batch, q, m)``.
    r:
        Local order (positive integer).

    Returns
    -------
    Array, shape ``(q * M_{r-1}, S, *batch, m**r)``
    """
    # prefix: (S, *batch, M_{r-1}, m**(r-1))
    prefix = monomials[r - 1]
    q = y.shape[-2]

    channels_per_p: list[Array] = []
    for p in range(q):
        # tail: (S, *batch, 1, m) — broadcast over M_{r-1} rows of prefix.
        tail = y[..., p, :][..., None, :]
        # tensor_product_homogeneous sees batch=(S,*batch,M_{r-1}), returns
        # (S, *batch, M_{r-1}, m**r).
        local_p = _CORE.tensor_product_homogeneous(prefix, tail)
        # Move M_{r-1} axis to front: (M_{r-1}, S, *batch, m**r).
        channels_per_p.append(jnp.moveaxis(local_p, -2, 0))

    # Concatenate over p → (q * M_{r-1}, S, *batch, m**r).
    return jnp.concatenate(channels_per_p, axis=0)


def _compute_basis_level_multicomp(
        n: int,
        *,
        ctx: FFTContext,
        components: tuple,
        table: LagFFTTable,
        monomials: tuple[Array, ...],
) -> Array:
    r"""Compute one output tensor level for q > 1 via batched FFT convolution.

    For output level ``n``, local order ``r``, and basis component ``b``,
    the source for lag channel ``(p, ell)`` (with ``|\ell| = r-1``) is

    .. math::

        C^{(b)}_{i,n-r} \otimes M_\ell(y_i) \otimes y_{i,p}

    convolved against lag-weight channel ``a_{p,\ell}`` from
    ``table.weights[r - 1][b]``.

    All ``(r, b)`` source blocks are concatenated along the channel axis and
    processed in a single call to :func:`_causal_conv_fft_batched`.

    Parameters
    ----------
    n:
        Output tensor level (positive integer, ``1 <= n <= ctx.trunc``).
    ctx:
        FFT context; ``ctx.y`` has shape ``(S, *batch, q, m)``.
    components:
        Persistent basis-coefficient histories; ``components[b][level]`` has
        shape ``(S, *batch, m**level)``.
    table:
        Precomputed lag FFT table built by :func:`_make_lag_fft_table_multicomp`.
    monomials:
        Shuffle monomials from :func:`_shuffle_monomials_by_degree`;
        ``monomials[k]`` has shape ``(S, *batch, M_k, m**k)``.

    Returns
    -------
    Array, shape ``(table.out_len, *batch, m**n)``
    """
    B = len(components)
    all_srcs: list[Array] = []
    all_Ws: list[Array] = []

    for r in range(1, n + 1):
        # Build all (p, ell) source channels for this local order.
        # Shape: (q * M_{r-1}, S, *batch, m**r)
        local_r = _local_multicomp_channels(monomials, ctx.y, r)
        C_r = local_r.shape[0]  # = q * M_{r-1}

        for b in range(B):
            # History component for this (r, b) pair.
            # hist: (S, *batch, m**(n-r))
            hist = components[b][n - r]

            # Tensor-product history with every channel of local_r.
            # Expand hist to (1, S, *batch, m**(n-r)) so it broadcasts over C_r.
            hist_exp = hist[None, ...]  # (1, S, *batch, m**(n-r))
            # tensor_product_homogeneous sees batch=(C_r, S, *batch), returns
            # (C_r, S, *batch, m**n).
            srcs_rb = _CORE.tensor_product_homogeneous(hist_exp, local_r)

            Ws_rb = table.weights[r - 1][b]  # (C_r, nfreq)

            assert srcs_rb.shape[0] == C_r, (
                f"Source channel count {srcs_rb.shape[0]} != C_r={C_r} "
                f"at n={n}, r={r}, b={b}."
            )
            assert Ws_rb.shape[0] == C_r, (
                f"Weight channel count {Ws_rb.shape[0]} != C_r={C_r} "
                f"at n={n}, r={r}, b={b}."
            )

            all_srcs.append(srcs_rb)
            all_Ws.append(Ws_rb)

    # Stack all (r, b) blocks into a single batch for the FFT convolution.
    srcs = jnp.concatenate(all_srcs, axis=0)  # (C_total, S, *batch, m**n)
    Ws = jnp.concatenate(all_Ws, axis=0)      # (C_total, nfreq)

    result = jnp.sum(
        _causal_conv_fft_batched(srcs, Ws, nfft=table.nfft, out_len=table.out_len),
        axis=0,
    )  # (out_len, *batch, m**n)

    assert result.shape == (table.out_len,) + ctx.batch_shape + (ctx.m ** n,), (
        f"Result shape {result.shape} != expected "
        f"{(table.out_len,) + ctx.batch_shape + (ctx.m ** n,)} at n={n}."
    )
    return result


def _shuffle_monomials_by_degree(
        y: Array,
        *,
        trunc: int,
        dtype: jnp.dtype,
) -> tuple[Array, ...]:
    r"""Build normalized shuffle monomials by total degree.

    For projected increments ``y`` with shape ``(S, *batch_shape, q, m)`` returns
    a tuple ``monomials`` where

    ``monomials[k]`` has shape ``(S, *batch_shape, M_k, m**k)``

    and entry ``ell_index`` (along the ``M_k`` axis) stores

    .. math::

        M_\ell(y_i) = \frac{1}{\ell !}\,
            y_1^{\sqcup\,\ell_1} \sqcup \cdots \sqcup y_q^{\sqcup\,\ell_q}

    for the ``ell_index``-th multi-index ``\ell`` of total degree ``k``.

    The multi-index ordering within each degree block is the same
    ``_compositions_desc`` order used by
    :func:`~tensordev.util.combinatorics.build_multiindex_layout` and by
    :meth:`~tensordev.volterra.kernel.ConvolutionKernel.lag_weights`, so that
    the ``ell_index`` axis here aligns directly with the last axis of
    ``kernel.lag_weights(..., n=k+1)``.

    Recursion (forward, by degree):

    .. math::

        M_{\ell + e_a}(y) \mathrel{+}=
            \frac{1}{k+1}\,(M_\ell(y) \sqcup y_a)

    summed over all predecessors ``\ell`` with ``|\ell| = k``.

    Parameters
    ----------
    y:
        Projected increments, shape ``(S, *batch_shape, q, m)``.
    trunc:
        Maximum total degree.
    dtype:
        Floating-point dtype.

    Returns
    -------
    tuple of ``trunc + 1`` arrays, ``monomials[k]`` with shape
    ``(S, *batch_shape, M_k, m**k)``.

    Notes
    -----
    This helper is used only by the q > 1 FFT path.  The q = 1 scalar path
    continues to use ordinary tensor powers via :func:`_tensor_powers`.
    """
    if trunc < 0:
        raise ValueError(f"trunc must be non-negative, got {trunc}.")
    if y.ndim < 3:
        raise ValueError(
            f"y must have at least 3 dimensions (S, ..., q, m), got ndim={y.ndim}."
        )

    y = jnp.asarray(y, dtype=dtype)
    S = y.shape[0]
    q = y.shape[-2]
    m = y.shape[-1]
    batch_shape = tuple(y.shape[1:-2])

    # Host-side layout & successor tables — computed once, never traced by JAX.
    layout = build_multiindex_layout(q=q, trunc=trunc)
    # Convert offsets to a plain NumPy array once so all degree-block size
    # arithmetic stays on the host and is never accidentally traced by JAX.
    offsets = np.asarray(layout.offsets)
    _, succ_local_by_deg = multiindex_batched_navigation(q=q, trunc=trunc)

    # Degree 0: single scalar 1, shape (S, *batch, 1, 1).
    monomials: list[Array] = [
        jnp.ones((S,) + batch_shape + (1, 1), dtype=dtype)
    ]

    for k in range(trunc):
        cur = monomials[k]  # (S, *batch, M_k, m**k)
        M_k = int(offsets[k + 1] - offsets[k])
        M_k1 = int(offsets[k + 2] - offsets[k + 1])
        inv = jnp.asarray(1.0 / (k + 1), dtype=dtype)

        # succ_local_by_deg[k]: tuple of q numpy int arrays, each shape (M_k,).
        # succ_local_by_deg[k][a][i] = local index (in degree-(k+1) block) of
        # the successor of degree-k entry i via component a.
        succ_r = succ_local_by_deg[k]

        # Build predecessor tables (host-side, static numpy) by inverting succ_r.
        # pred_locals[a][j] = local index (in degree-k block) of the predecessor
        # of degree-(k+1) entry j via component a.
        # Sentinel value M_k (beyond cur's last row) marks "no predecessor via a".
        pred_locals: list[np.ndarray] = []
        for a in range(q):
            pred_a = np.full(M_k1, M_k, dtype=np.intp)  # default = sentinel
            for i_src, j_dst in enumerate(succ_r[a]):
                pred_a[j_dst] = i_src
            pred_locals.append(pred_a)

        # Append a zero row to cur so the sentinel index M_k maps to zero.
        # Shuffling zero with any vector gives zero, so no masking is needed.
        cur_ext = jnp.concatenate(
            [cur, jnp.zeros((S,) + batch_shape + (1, m ** k), dtype=dtype)],
            axis=-2,
        )  # (S, *batch, M_k+1, m**k)

        nxt = jnp.zeros((S,) + batch_shape + (M_k1, m ** (k + 1)), dtype=dtype)

        for a in range(q):
            # Gather: pred_locals[a] is a static numpy int array of shape (M_{k+1},).
            # JAX compiles this as a gather (no scatter / .at[].add overhead).
            # Entries where pred_locals[a][j] == M_k pick up the zero sentinel row.
            preds = cur_ext[..., pred_locals[a], :]  # (S, *batch, M_{k+1}, m**k)

            # Shuffle all predecessors with y_a simultaneously — (S, *batch, M_{k+1})
            # is the effective batch for tensor_shuffle_vector_homogeneous.
            y_a_exp = y[..., a, :][..., None, :]  # (S, *batch, 1, m)
            shuffled = _CORE.tensor_shuffle_vector_homogeneous(preds, y_a_exp, k)
            # (S, *batch, M_{k+1}, m**(k+1))

            nxt = nxt + inv * shuffled

        monomials.append(nxt)

    return tuple(monomials)


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


__all__ = ["fft_iteration", "precompute_lag_tables", "PrecomputedLagTables"]
