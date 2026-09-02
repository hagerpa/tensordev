"""FFT-based Volterra signatures on native graded tensor cores.

The lag tables and logical FFT channels depend only on total order. Algebra
grades are therefore combined inside each logical channel, and all output
grades on one total-order diagonal are coordinate-packed into the same causal
FFT. For the total-degree core every diagonal contains one block, so this
schedule reduces to the dense total-degree case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tensordev._backend import _get as _get_backend_pair
from tensordev.core.utils.pytrees import tree_map
from tensordev.util.combinatorics import (
    build_multiindex_layout,
    multiindex_batched_navigation,
)
from tensordev.volterra._convolution import (
    apply_transformed_causal_fft,
    next_power_of_two,
)
from tensordev.volterra.algebra import (
    GradeWorkset,
    ResolvedVolterraAlgebra,
    require_volterra_shuffle,
    resolve_volterra_algebra,
    resolve_volterra_core_pair,
)
from tensordev.volterra.iteration_quad import (
    BasisExpansionSpec,
    _basis_interpolation_matrix,
    _basis_rhos_multicomp,
    _chebyshev_lobatto_thetas,
    _normalize_times,
)
from tensordev.volterra.kernel import ConvolutionKernel


Array = jax.Array
Grade = Any
GradeBlocks = tuple[Array, ...]
ComponentState = dict[Grade, Array]


def _total_degree_algebra(trunc: int, alphabet_dim: int) -> ResolvedVolterraAlgebra:
    """Resolve the cached total-degree JAX algebra independently of defaults."""
    total_core, _ = _get_backend_pair("jax")
    return resolve_volterra_algebra(total_core, trunc, alphabet_dim)


@dataclass(frozen=True, slots=True)
class FFTContext:
    """Static algebra metadata and dynamic data shared by FFT variants."""

    y: Array
    y_powers: tuple[GradeBlocks, ...] | None
    times: Array
    h: Array
    unit: Any
    algebra: ResolvedVolterraAlgebra
    seq_core: Any

    S: int
    m: int
    max_order: int
    batch_shape: tuple[int, ...]
    dtype: jnp.dtype

    @property
    def trunc(self) -> Any:
        """Resolved algebra truncation."""
        return self.algebra.truncation


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class LagFFTTable:
    """Frequency-domain lag weights for one interpolation point.

    ``weights[n - 1][b]`` is the transformed lag sequence for local total
    order ``n`` and basis component ``b``. In the multi-component case its
    leading axis follows the packed ``(kernel component, multi-index)`` order.
    """

    weights: tuple[tuple[Array, ...], ...]
    nfft: int = field(metadata={"static": True})
    out_len: int = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class PrecomputedLagTables:
    """Lag tables reusable across paths and compatible algebra gradings.

    The table contains no algebra-coordinate data. Compatibility of the
    kernel, step size, dtype, and basis exponents remains the caller's
    responsibility; ``fft_iteration`` validates the effective step count,
    scheme order, and sufficient total depth.
    """

    theta_tables: tuple[LagFFTTable, ...]
    output_table: LagFFTTable
    S: int = field(metadata={"static": True})
    max_order: int = field(metadata={"static": True})
    order: int = field(metadata={"static": True})

    @property
    def trunc(self) -> int:
        """Maximum total order represented by the lag tables."""
        return self.max_order


def _lag_table_max_order(
        kernel: ConvolutionKernel,
        trunc: Any,
        core: Any,
) -> int:
    """Resolve public lag-table truncation without grading ambiguity."""
    if isinstance(trunc, Integral) and not isinstance(trunc, bool):
        max_order = int(trunc)
        if max_order <= 0:
            raise ValueError(f"trunc must be positive, got {max_order}.")
        # Integer precomputation is deliberately algebra-independent, even
        # when a non-total background or explicit core is present.
        return max_order

    resolved_core, _ = resolve_volterra_core_pair(core, None)
    algebra = resolve_volterra_algebra(resolved_core, trunc, kernel.m)
    return algebra.max_order


def precompute_lag_tables(
        kernel: ConvolutionKernel,
        *,
        S: int,
        h: float | Array,
        order: int,
        trunc: int | tuple[int, int] | None,
        dtype: jnp.dtype,
        core: Any = None,
) -> PrecomputedLagTables:
    """Precompute grading-independent lag FFT tables for ``fft_iteration``.

    An integer ``trunc`` is interpreted directly as total depth and never
    consults the selected background core. A pair, or ``None``, resolves the
    explicit/background core and uses the resulting algebra's ``max_order``.
    A table built to a larger depth can serve a smaller active truncation.

    ``PrecomputedLagTables`` validates ``S``, ``order``, and total depth only.
    Callers are responsible for reusing a table with the same kernel, step
    size, dtype, and derived basis exponents.
    """
    if isinstance(S, bool) or not isinstance(S, Integral):
        raise TypeError(f"S must be a positive integer, got {S!r}.")
    S = int(S)
    if S <= 0:
        raise ValueError(f"S must be positive, got {S}.")
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1, or 2, got {order}.")

    max_order = _lag_table_max_order(kernel, trunc, core)
    dtype_ = jnp.dtype(dtype)
    h_arr = jnp.asarray(h, dtype=dtype_)
    rhos = _basis_rhos_multicomp(order, betas=kernel.beta, dtype=dtype_)
    thetas = _chebyshev_lobatto_thetas(n=len(rhos), dtype=dtype_)

    theta_tables = tuple(
        _make_lag_fft_table(
            kernel=kernel,
            S=S,
            h=h_arr,
            trunc=max_order,
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
        trunc=max_order,
        dtype=dtype_,
        out_len=S + 1,
        theta=jnp.asarray(0.0, dtype=dtype_),
        rhos=rhos,
    )
    return PrecomputedLagTables(
        theta_tables=theta_tables,
        output_table=output_table,
        S=S,
        max_order=max_order,
        order=order,
    )


def fft_iteration(
        dX: Array,
        *,
        kernel: ConvolutionKernel,
        trunc=None,
        dt: Array | float = 1.0,
        axis: int = -2,
        return_trajectory: bool = False,
        order: int = 0,
        lag_tables: PrecomputedLagTables | None = None,
        core=None,
        seq_core=None,
):
    r"""Volterra signature via FFT convolution on a uniform grid.

    ``dX`` must already contain increments on the final grid. The returned
    tensor uses the representation native to the resolved algebra core.
    ``trunc`` may therefore be an integer for total degree or a bidegree pair
    for a standard bidegree core; a bounded core may supply its default when
    ``trunc`` is omitted.
    """
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1, or 2, got {order}.")

    core, seq_core = resolve_volterra_core_pair(core, seq_core)
    algebra = resolve_volterra_algebra(core, trunc, kernel.m)
    if kernel.q > 1:
        require_volterra_shuffle(algebra, feature="The q > 1 FFT scheme")
    xp = core.xp

    dX = xp.asarray(dX)
    if dX.ndim < 2:
        raise ValueError(
            "dX must have at least a step axis and a trailing path dimension."
        )
    axis_norm = axis % dX.ndim
    if axis_norm == dX.ndim - 1:
        raise ValueError(
            "axis must identify the step axis, not the trailing path dimension."
        )
    if dX.shape[-1] != kernel.path_dim:
        raise ValueError(
            f"dX trailing dimension must be {kernel.path_dim}, got {dX.shape[-1]}."
        )

    dtype = dX.dtype
    S = int(dX.shape[axis_norm])
    if S == 0:
        raise ValueError("fft_iteration requires at least one increment.")
    dt_array = xp.asarray(dt, dtype=dtype)
    if dt_array.ndim != 0:
        raise ValueError(
            "fft_iteration requires a scalar dt because FFT convolution "
            "assumes a uniform grid."
        )

    projected = xp.einsum("qmd,...d->...qm", kernel.A.astype(dtype), dX)
    # (S, *batch, q, m), with the kernel-component axis always present.
    y = xp.moveaxis(projected, axis_norm, 0)
    times_arr = _normalize_times(dt_array, S=S, dtype=dtype)
    h = times_arr[1] - times_arr[0]
    batch_shape = tuple(y.shape[1:-2])

    if kernel.q == 1:
        y_powers = _native_tensor_powers(
            y[..., 0, :], algebra=algebra, dtype=dtype
        )
    else:
        y_powers = None

    ctx = FFTContext(
        y=y,
        y_powers=y_powers,
        times=times_arr,
        h=h,
        unit=algebra.unit(batch_shape=batch_shape, dtype=dtype),
        algebra=algebra,
        seq_core=seq_core,
        S=S,
        m=kernel.m,
        max_order=algebra.max_order,
        batch_shape=batch_shape,
        dtype=dtype,
    )
    if lag_tables is not None:
        _validate_lag_tables(lag_tables, ctx=ctx, order=order)

    output = _run_basis_fft(
        ctx=ctx, kernel=kernel, order=order, lag_tables=lag_tables
    )
    if return_trajectory:
        trajectory = tree_map(lambda block: block[1:], output)
        if axis_norm != 0:
            trajectory = tree_map(
                lambda block: xp.moveaxis(block, 0, axis_norm), trajectory
            )
        return trajectory
    return tree_map(lambda block: block[-1], output)


def _validate_lag_tables(
        lag_tables: PrecomputedLagTables,
        *,
        ctx: FFTContext,
        order: int,
) -> None:
    if lag_tables.S != ctx.S:
        raise ValueError(
            f"lag_tables.S={lag_tables.S} does not match the effective S={ctx.S} "
            "(after dyadic refinement). Rebuild with precompute_lag_tables."
        )
    if lag_tables.max_order < ctx.max_order:
        raise ValueError(
            f"lag_tables.max_order={lag_tables.max_order} is smaller than the "
            f"required total depth {ctx.max_order}."
        )
    if lag_tables.order != order:
        raise ValueError(
            f"lag_tables.order={lag_tables.order} != order={order}."
        )


def _native_tensor_powers(
        y_scalar: Array,
        *,
        algebra: ResolvedVolterraAlgebra,
        dtype: jnp.dtype,
) -> tuple[GradeBlocks, ...]:
    """Build exact-order generator powers as native grade diagonals."""
    S = y_scalar.shape[0]
    batch_shape = tuple(y_scalar.shape[1:-1])
    powers: list[GradeBlocks] = [
        (
            algebra.core.xp.ones(
                (S,) + batch_shape + (algebra.block_width(algebra.zero_grade),),
                dtype=dtype,
            ),
        )
    ]
    if algebra.max_order == 0:
        return tuple(powers)

    generator = dict(algebra.generator_blocks(y_scalar))
    powers.append(
        tuple(generator[grade] for grade in algebra.diagonal(1).grades)
    )
    for total_order in range(2, algebra.max_order + 1):
        powers.append(
            algebra.right_generator_action(
                algebra.diagonal(total_order - 1),
                powers[-1],
                y_scalar,
                algebra.diagonal(total_order),
            )
        )
    return tuple(powers)


def _tensor_powers(
        y_scalar: Array,
        *,
        trunc: int,
        dtype: jnp.dtype,
):
    """Dense total-degree view of the native power recurrence."""
    algebra = _total_degree_algebra(trunc, int(y_scalar.shape[-1]))
    return tuple(
        algebra.diagonal(total_order).pack(algebra.core.xp, values)
        for total_order, values in enumerate(
            _native_tensor_powers(y_scalar, algebra=algebra, dtype=dtype)
        )
    )


def _run_basis_fft(
        *,
        ctx: FFTContext,
        kernel: ConvolutionKernel,
        order: int,
        lag_tables: PrecomputedLagTables | None = None,
):
    """Run the shared basis expansion with native grade-block state."""
    spec = _basis_spec(ctx=ctx, kernel=kernel, order=order)
    B = len(spec.rhos)
    zero = ctx.algebra.zero_grade

    components: tuple[ComponentState, ...] = tuple(
        {
            zero: (
                ctx.algebra.core.xp.ones(
                    (ctx.S,) + ctx.batch_shape + (1,), dtype=ctx.dtype
                )
                if b == 0
                else ctx.algebra.core.xp.zeros(
                    (ctx.S,) + ctx.batch_shape + (1,), dtype=ctx.dtype
                )
            )
        }
        for b in range(B)
    )

    if lag_tables is None:
        theta_tables = tuple(
            _make_lag_fft_table(
                kernel=kernel,
                S=ctx.S,
                h=ctx.h,
                trunc=ctx.max_order,
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
            trunc=ctx.max_order,
            dtype=ctx.dtype,
            out_len=ctx.S + 1,
            theta=jnp.asarray(0.0, dtype=ctx.dtype),
            rhos=spec.rhos,
        )
    else:
        theta_tables = lag_tables.theta_tables
        output_table = lag_tables.output_table

    if kernel.q == 1:
        monomials = None
    else:
        monomials = _shuffle_monomials_by_grade(
            ctx.y,
            max_order=ctx.max_order - 1,
            dtype=ctx.dtype,
            algebra=ctx.algebra,
        )

    for total_order in range(1, ctx.max_order + 1):
        evaluations = tuple(
            _compute_basis_diagonal(
                total_order,
                ctx=ctx,
                components=components,
                table=table,
                monomials=monomials,
            )
            for table in theta_tables
        )
        components = _append_interpolated_basis_diagonal(
            total_order=total_order,
            components=components,
            evaluations=evaluations,
            interpolation_inverse=spec.interpolation_inverse,
            algebra=ctx.algebra,
        )

    return _basis_output(
        ctx=ctx,
        components=components,
        table=output_table,
        monomials=monomials,
    )


def _basis_spec(
        *,
        ctx: FFTContext,
        kernel: ConvolutionKernel,
        order: int,
) -> BasisExpansionSpec:
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


def _append_interpolated_basis_diagonal(
        *,
        total_order: int,
        components: tuple[ComponentState, ...],
        evaluations: tuple[Array, ...],
        interpolation_inverse: Array,
        algebra: ResolvedVolterraAlgebra,
) -> tuple[ComponentState, ...]:
    """Interpolate once across a packed native output diagonal."""
    stacked = algebra.core.xp.stack(evaluations, axis=0)
    coefficients = algebra.core.xp.tensordot(
        interpolation_inverse, stacked, axes=1
    )
    diagonal = algebra.diagonal(total_order)
    updated: list[ComponentState] = []
    for component, packed in zip(components, coefficients):
        values = diagonal.split(packed)
        state = dict(component)
        state.update(zip(diagonal.grades, values))
        updated.append(state)
    return tuple(updated)


def _basis_output(
        *,
        ctx: FFTContext,
        components: tuple[ComponentState, ...],
        table: LagFFTTable,
        monomials: tuple[GradeBlocks, ...] | None,
):
    blocks: dict[Grade, Array] = {
        ctx.algebra.zero_grade: ctx.algebra.core.xp.ones(
            (ctx.S + 1,) + ctx.batch_shape + (1,), dtype=ctx.dtype
        )
    }
    for total_order in range(1, ctx.max_order + 1):
        packed = _compute_basis_diagonal(
            total_order,
            ctx=ctx,
            components=components,
            table=table,
            monomials=monomials,
        )
        diagonal = ctx.algebra.diagonal(total_order)
        blocks.update(zip(diagonal.grades, diagonal.split(packed)))
    return ctx.algebra.assemble(
        tuple(blocks[grade] for grade in ctx.algebra.grades)
    )


def _state_diagonal(
        component: ComponentState,
        workset: GradeWorkset,
) -> GradeBlocks:
    return tuple(component[grade] for grade in workset.grades)


def _sum_product_splits(
        *,
        algebra: ResolvedVolterraAlgebra,
        output_grade: Grade,
        history_workset: GradeWorkset,
        history_values: GradeBlocks,
        local_workset: GradeWorkset,
        local_values: GradeBlocks,
) -> Array:
    """Group all algebra-grade splits into one logical FFT source block."""
    contributions: list[tuple[Array, Array, Grade, Grade]] = []
    for history_grade, local_grade in algebra.layout.product_splits(output_grade):
        if not (
            history_workset.contains(history_grade)
            and local_workset.contains(local_grade)
        ):
            continue
        history = history_workset.block(history_values, history_grade)[None, ...]
        local = local_workset.block(local_values, local_grade)
        contributions.append(
            (history, local, history_grade, local_grade)
        )
    if not contributions:
        raise ValueError(
            f"no admissible source split for output grade {output_grade!r}."
        )
    return algebra.product_output_block(contributions, output_grade)


def _compute_basis_diagonal(
        total_order: int,
        *,
        ctx: FFTContext,
        components: tuple[ComponentState, ...],
        table: LagFFTTable,
        monomials: tuple[GradeBlocks, ...] | None,
) -> Array:
    """Compute one coordinate-packed output diagonal in one causal FFT."""
    algebra = ctx.algebra
    target = algebra.diagonal(total_order)
    basis_count = len(components)
    source_groups: list[Array] = []
    weight_groups: list[Array] = []

    for local_order in range(1, total_order + 1):
        local_workset = algebra.diagonal(local_order)
        if monomials is None:
            assert ctx.y_powers is not None
            # Scalar path has one logical local channel.
            local_values = tuple(
                block[None, ...] for block in ctx.y_powers[local_order]
            )
        else:
            local_values = _local_multicomp_channels_by_grade(
                monomials,
                ctx.y,
                local_order,
                algebra=algebra,
            )

        history_workset = algebra.diagonal(total_order - local_order)
        for basis_index in range(basis_count):
            history_values = _state_diagonal(
                components[basis_index], history_workset
            )
            output_blocks = tuple(
                _sum_product_splits(
                    algebra=algebra,
                    output_grade=output_grade,
                    history_workset=history_workset,
                    history_values=history_values,
                    local_workset=local_workset,
                    local_values=local_values,
                )
                for output_grade in target.grades
            )
            source_groups.append(target.pack(algebra.core.xp, output_blocks))

            weights = table.weights[local_order - 1][basis_index]
            if monomials is None:
                weights = weights[None, :]
            weight_groups.append(weights)

    sources = algebra.core.xp.concat(source_groups, axis=0)
    transformed_weights = algebra.core.xp.concat(weight_groups, axis=0)
    convolved = apply_transformed_causal_fft(
        sources,
        transformed_weights,
        nfft=table.nfft,
        out_len=table.out_len,
    )
    return algebra.core.xp.sum(convolved, axis=0)


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
    nfft = next_power_of_two(S + out_len - 1)
    rows: list[tuple[Array, ...]] = []
    for local_order in range(1, trunc + 1):
        cols = []
        for rho in rhos:
            weights = kernel.lag_weights(
                out_len=out_len,
                h=h,
                theta=theta,
                n=local_order,
                rho=rho,
                dtype=dtype,
            )[..., 0, 0]
            cols.append(jnp.fft.rfft(weights, n=nfft))
        rows.append(tuple(cols))
    return LagFFTTable(weights=tuple(rows), nfft=nfft, out_len=out_len)


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
    q = kernel.q
    nfft = next_power_of_two(S + out_len - 1)
    rows: list[tuple[Array, ...]] = []
    for local_order in range(1, trunc + 1):
        cols = []
        for rho in rhos:
            weights = kernel.lag_weights(
                out_len=out_len,
                h=h,
                theta=theta,
                n=local_order,
                rho=rho,
                dtype=dtype,
            )
            multiindices = weights.shape[-1]
            if weights.shape != (out_len, q, multiindices):
                raise ValueError(
                    f"lag_weights returned unexpected shape {weights.shape}; "
                    f"expected ({out_len}, {q}, {multiindices})."
                )
            flattened = weights.reshape(out_len, q * multiindices)
            cols.append(jnp.fft.rfft(flattened, n=nfft, axis=0).T)
        rows.append(tuple(cols))
    return LagFFTTable(weights=tuple(rows), nfft=nfft, out_len=out_len)


def _local_multicomp_channels_by_grade(
        monomials: tuple[GradeBlocks, ...],
        y: Array,
        local_order: int,
        *,
        algebra: ResolvedVolterraAlgebra,
) -> GradeBlocks:
    """Build native local blocks in logical ``(component, multi-index)`` order."""
    prefix_workset = algebra.diagonal(local_order - 1)
    target_workset = algebra.diagonal(local_order)
    prefix = monomials[local_order - 1]
    q = int(y.shape[-2])
    by_component: list[GradeBlocks] = []
    for component in range(q):
        generator = y[..., component, :][..., None, :]
        values = algebra.right_generator_action(
            prefix_workset,
            prefix,
            generator,
            target_workset,
        )
        # The multi-index row is the final batch axis; make it the logical
        # channel axis while preserving time and all path batch axes.
        by_component.append(
            tuple(jnp.moveaxis(block, -2, 0) for block in values)
        )
    return tuple(
        jnp.concatenate(
            tuple(values[index] for values in by_component), axis=0
        )
        for index in range(target_workset.size)
    )


def _local_multicomp_channels(
        monomials: tuple[Array, ...],
        y: Array,
        r: int,
) -> Array:
    """Dense total-degree view of native local multi-component channels."""
    algebra = _total_degree_algebra(r, int(y.shape[-1]))
    graded_monomials = tuple((monomial,) for monomial in monomials[:r])
    return _local_multicomp_channels_by_grade(
        graded_monomials, y, r, algebra=algebra
    )[0]


def _shuffle_monomials_by_grade(
        y: Array,
        *,
        max_order: int,
        dtype: jnp.dtype,
        algebra: ResolvedVolterraAlgebra,
) -> tuple[GradeBlocks, ...]:
    """Build normalized shuffle monomials as sparse native grade diagonals."""
    if max_order < 0:
        raise ValueError(f"max_order must be non-negative, got {max_order}.")
    if y.ndim < 3:
        raise ValueError(
            f"y must have at least 3 dimensions (S, ..., q, m), got ndim={y.ndim}."
        )

    xp = algebra.core.xp
    y = xp.asarray(y, dtype=dtype)
    S = int(y.shape[0])
    q = int(y.shape[-2])
    batch_shape = tuple(y.shape[1:-2])
    layout = build_multiindex_layout(q=q, trunc=max_order)
    offsets = np.asarray(layout.offsets)
    _, successors = multiindex_batched_navigation(q=q, trunc=max_order)

    monomials: list[GradeBlocks] = [
        (
            xp.ones(
                (S,) + batch_shape + (1, algebra.block_width(algebra.zero_grade)),
                dtype=dtype,
            ),
        )
    ]
    for degree in range(max_order):
        source_workset = algebra.diagonal(degree)
        target_workset = algebra.diagonal(degree + 1)
        current = monomials[degree]
        current_rows = int(offsets[degree + 1] - offsets[degree])
        next_rows = int(offsets[degree + 2] - offsets[degree + 1])
        inverse_degree = xp.asarray(1.0 / (degree + 1), dtype=dtype)

        predecessor_tables: list[np.ndarray] = []
        for component in range(q):
            predecessor = np.full(next_rows, current_rows, dtype=np.intp)
            for source_index, target_index in enumerate(successors[degree][component]):
                predecessor[target_index] = source_index
            predecessor_tables.append(predecessor)

        extended = tuple(
            xp.concat(
                (
                    block,
                    xp.zeros(
                        block.shape[:-2] + (1, block.shape[-1]), dtype=dtype
                    ),
                ),
                axis=-2,
            )
            for block in current
        )
        accumulated: GradeBlocks | None = None
        for component in range(q):
            gathered = tuple(
                block[..., predecessor_tables[component], :]
                for block in extended
            )
            generator = y[..., component, :][..., None, :]
            contribution = algebra.shuffle_generator_action(
                source_workset,
                gathered,
                generator,
                target_workset,
            )
            if accumulated is None:
                accumulated = contribution
            else:
                accumulated = tuple(
                    left + right for left, right in zip(accumulated, contribution)
                )
        assert accumulated is not None
        monomials.append(
            tuple(inverse_degree * block for block in accumulated)
        )
    return tuple(monomials)


def _shuffle_monomials_by_degree(
        y: Array,
        *,
        trunc: int,
        dtype: jnp.dtype,
) -> tuple[Array, ...]:
    """Dense total-degree view of the native monomial recurrence."""
    if trunc < 0:
        raise ValueError(f"trunc must be non-negative, got {trunc}.")
    if y.ndim < 3:
        raise ValueError(
            f"y must have at least 3 dimensions (S, ..., q, m), got ndim={y.ndim}."
        )
    if trunc == 0:
        return (
            jnp.ones(
                tuple(y.shape[0:-2]) + (1, 1),
                dtype=dtype,
            ),
        )
    algebra = _total_degree_algebra(trunc, int(y.shape[-1]))
    graded = _shuffle_monomials_by_grade(
        y, max_order=trunc, dtype=dtype, algebra=algebra
    )
    return tuple(values[0] for values in graded)


def _next_pow2(n: int) -> int:
    """Return the causal-FFT transform size for ``n`` entries."""
    return next_power_of_two(n)


def _causal_conv_fft_batched(
        srcs: Array,
        Ws: Array,
        *,
        nfft: int,
        out_len: int,
) -> Array:
    """Apply transformed batched causal convolution."""
    return apply_transformed_causal_fft(
        srcs, Ws, nfft=nfft, out_len=out_len
    )


__all__ = ["fft_iteration", "precompute_lag_tables", "PrecomputedLagTables"]
