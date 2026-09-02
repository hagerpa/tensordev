"""Volterra signature — high-level entry point and VolterraSignature wrapper."""
from __future__ import annotations

import math
import threading
from collections import OrderedDict, namedtuple
from dataclasses import dataclass, field, fields as dataclass_instance_fields
from typing import Any, Literal, Optional

import jax
import numpy as np

from tensordev.core.utils.pytrees import tree_prepend, tree_stack, tree_take
from tensordev.volterra.algebra import (
    ResolvedVolterraAlgebra,
    resolve_volterra_algebra,
    resolve_volterra_core_pair,
)
from tensordev.volterra.kernel import ConvolutionKernel, FractionalKernel, GammaKernel
from tensordev.volterra.iteration_quad import quadratic_iteration as _vsig_quadratic
from tensordev.volterra.iteration_fft import fft_iteration as _vsig_fft, PrecomputedLagTables
from tensordev.volterra.iteration_pc import pc_iteration as _vsig_pc

Array = jax.Array


# Whole-solver JIT boundaries avoid dispatching each compiled Volterra
# subroutine separately.  Numerical values, including kernel leaves, remain
# dynamic; only the finite algebra and Python control-flow choices are
# captured.  Cache keys never depend on kernel object identity.
_SOLVER_CACHE_MAXSIZE = 32
_SolverCacheInfo = namedtuple("SolverCacheInfo", "hits misses maxsize currsize")
_solver_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_solver_cache_lock = threading.RLock()
_solver_cache_hits = 0
_solver_cache_misses = 0


@dataclass(frozen=True, slots=True)
class _KernelRebuilder:
    """Rebuild a kernel with dynamic leaves without running validation.

    JAX's default dataclass unflattening invokes the dataclass constructor.
    Kernel constructors deliberately perform eager value validation, which is
    not legal for tracer-valued leaves.  Public ``vsig`` has already received
    and validated a concrete immutable kernel, so a compiled solver can safely
    rebuild the same class by assigning its dynamic fields directly.

    Only kernels whose PyTree leaves are direct dataclass attributes use this
    fast path.  More involved third-party or nested kernel PyTrees use eager
    orchestration to preserve their semantics.
    """

    kernel_type: type
    dynamic_fields: tuple[str, ...]
    static_fields: tuple[tuple[str, Any], ...]
    static_key: tuple[Any, ...]

    def rebuild(self, leaves: tuple[Any, ...]) -> Any:
        if len(leaves) != len(self.dynamic_fields):
            raise ValueError(
                f"kernel expects {len(self.dynamic_fields)} dynamic leaves, "
                f"got {len(leaves)}."
            )
        kernel = object.__new__(self.kernel_type)
        for name, value in self.static_fields:
            object.__setattr__(kernel, name, value)
        for name, value in zip(self.dynamic_fields, leaves):
            object.__setattr__(kernel, name, value)
        return kernel


def _static_value_key(value: Any) -> tuple[Any, ...]:
    """Content key for small static kernel metadata, never array identity."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array = np.asarray(jax.device_get(value))
        return ("array", array.dtype.str, tuple(array.shape), array.tobytes())
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return ("value", type(value), value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_static_value_key(item) for item in value))
    if isinstance(value, frozenset):
        return ("frozenset", frozenset(_static_value_key(item) for item in value))
    # An arbitrary object's hash commonly encodes its identity and can also
    # hide mutable semantics.  Such custom metadata stays on the eager path.
    raise TypeError(
        f"unsupported static kernel metadata type {type(value).__name__}"
    )


def _kernel_rebuilder(
        kernel: ConvolutionKernel,
        *,
        static_leaf_fields: tuple[str, ...] = (),
):
    """Return ``(rebuilder, leaves, treedef)`` or ``None`` for eager fallback."""
    try:
        path_leaves, treedef = jax.tree_util.tree_flatten_with_path(kernel)
        all_leaf_fields = tuple(
            path[0].name
            for path, _ in path_leaves
            if len(path) == 1 and isinstance(path[0], jax.tree_util.GetAttrKey)
        )
        if len(all_leaf_fields) != len(path_leaves):
            return None
        # A malformed/custom registration with duplicate attribute paths is
        # not suitable for direct reconstruction.
        if len(set(all_leaf_fields)) != len(all_leaf_fields):
            return None
        promoted_static = frozenset(static_leaf_fields)
        if not promoted_static.issubset(all_leaf_fields):
            return None
        dynamic_fields = tuple(
            name for name in all_leaf_fields if name not in promoted_static
        )
        dynamic_set = set(dynamic_fields)
    except (AttributeError, TypeError):
        return None

    try:
        dataclass_fields = dataclass_instance_fields(kernel)
    except TypeError:
        return None
    try:
        static_fields = tuple(
            (item.name, getattr(kernel, item.name))
            for item in dataclass_fields
            if item.name not in dynamic_set
        )
        static_key = tuple(
            (name, _static_value_key(value))
            for name, value in static_fields
        )
    except (AttributeError, TypeError, ValueError):
        return None
    leaf_by_name = dict(zip(all_leaf_fields, (leaf for _, leaf in path_leaves)))
    return (
        _KernelRebuilder(type(kernel), dynamic_fields, static_fields, static_key),
        tuple(leaf_by_name[name] for name in dynamic_fields),
        treedef,
    )


def _solver_cache_info():
    """Return a snapshot of the bounded whole-solver cache statistics."""
    with _solver_cache_lock:
        return _SolverCacheInfo(
            _solver_cache_hits,
            _solver_cache_misses,
            _SOLVER_CACHE_MAXSIZE,
            len(_solver_cache),
        )


def _clear_solver_cache() -> None:
    """Clear cached public Volterra solver boundaries."""
    global _solver_cache_hits, _solver_cache_misses
    with _solver_cache_lock:
        _solver_cache.clear()
        _solver_cache_hits = 0
        _solver_cache_misses = 0


def _validate_vsig_static_options(
        *,
        scheme: str,
        dyadic_order: int,
        order: int,
        block_size: Optional[int],
) -> None:
    if scheme not in ("auto", "fft", "quadratic", "adams"):
        raise ValueError(f"scheme must be 'auto', 'fft', 'quadratic', or 'adams', got {scheme!r}.")
    if dyadic_order < 0:
        raise ValueError(f"dyadic_order must be non-negative, got {dyadic_order}.")
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1, or 2, got {order}.")
    if block_size is not None and block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size}.")


def _make_compiled_solver(
        *,
        rebuilder: _KernelRebuilder,
        trunc: Any,
        axis: int,
        block_size: Optional[int],
        accumulate: bool,
        output_starting_point: bool,
        increment_input: bool,
        order: int,
        dyadic_order: int,
        scheme: str,
        core: Any,
        seq_core: Any,
):
    """Bind static orchestration once while keeping all arrays dynamic."""

    @jax.jit
    def solve(X, dt, kernel_leaves, starting_point, lag_tables):
        kernel = rebuilder.rebuild(kernel_leaves)
        return _vsig_impl(
            X,
            kernel=kernel,
            trunc=trunc,
            dt=dt,
            axis=axis,
            block_size=block_size,
            accumulate=accumulate,
            starting_point=starting_point,
            output_starting_point=output_starting_point,
            increment_input=increment_input,
            order=order,
            dyadic_order=dyadic_order,
            scheme=scheme,
            lag_tables=lag_tables,
            core=core,
            seq_core=seq_core,
        )

    return solve


def _cached_compiled_solver(key: tuple[Any, ...], **kwargs):
    """Return one solver from a bounded, thread-safe LRU."""
    global _solver_cache_hits, _solver_cache_misses
    # Validate third-party static PyTree descriptors before cache lookup so an
    # unsupported descriptor cannot affect public behavior.
    hash(key)
    with _solver_cache_lock:
        cached = _solver_cache.get(key)
        if cached is not None:
            _solver_cache.move_to_end(key)
            _solver_cache_hits += 1
            return cached

        solver = _make_compiled_solver(**kwargs)
        _solver_cache[key] = solver
        _solver_cache_misses += 1
        if len(_solver_cache) > _SOLVER_CACHE_MAXSIZE:
            _solver_cache.popitem(last=False)
        return solver


def _vsig_impl(
        X: Array,
        *,
        kernel: ConvolutionKernel,
        trunc=None,
        dt: Array | float = 1.0,
        axis: int = -2,
        block_size: Optional[int] = None,
        accumulate: bool = True,
        starting_point: Any = None,
        output_starting_point: bool = False,
        increment_input: bool = False,
        order: int = 0,
        dyadic_order: int = 0,
        scheme: Literal["auto", "fft", "quadratic", "adams"] = "auto",
        lag_tables: Optional[PrecomputedLagTables] = None,
        core=None,
        seq_core=None,
):
    """Compute the truncated Volterra signature of ``X``.

    High-level entry point that handles scheme selection, blocking, and
    accumulation before delegating to the low-level iteration kernels.

    Parameters
    ----------
    X:
        Path nodes or increments.  The trailing axis is the path dimension
        ``kernel.path_dim``; ``axis`` is the step/node axis.
    kernel:
        Volterra kernel supplying projections and coefficient builders.
    trunc:
        Active tensor truncation.  The total-degree core requires a positive
        integer; a bounded bidegree core accepts ``(N, M)`` and may provide a
        default when this is omitted.
    dt:
        Step size(s).  A scalar gives a uniform grid; a 1-D array of length
        ``S`` gives a non-uniform grid via cumulative sums (default ``1.0``).
    axis:
        Step/node axis of ``X`` (default ``-2``).
    block_size:
        Number of steps per emitted block.  ``None`` (default) → single
        terminal result.  With ``accumulate=True`` the signature at the end
        of each block is returned (block axis inserted at ``axis``).  With
        ``accumulate=False`` each block is treated as an independent path
        via rebatch (block axis inserted at ``axis``).
    accumulate:
        If ``True`` (default) history carries across blocks (standard
        Volterra semantics).  If ``False`` each block is processed
        independently by reshaping the batch dimension.  Ignored when
        ``block_size`` is ``None``.
    starting_point:
        Optional native tensor element prepended when
        ``output_starting_point=True``.  This is output-only and does not seed
        the recurrence.  Defaults to the selected core's tensor unit.
    output_starting_point:
        If ``True``, prepend the seed (unit or ``starting_point``) to the
        output along ``axis``.
    increment_input:
        If ``True``, treat ``X`` as increments and skip :func:`jnp.diff`.
    order:
        Quadrature order for the higher-order basis-expansion scheme.
        ``0`` (default) left-point approximation.
    dyadic_order:
        Non-negative integer.  Each increment is split into
        ``2**dyadic_order`` equal sub-increments.  ``0`` (default) leaves
        the path unchanged.
    scheme:
        Which iteration scheme to use.

        ``"auto"``
            Use ``"fft"`` when ``dt`` is a scalar (uniform grid) **and**
            the expected FFT op-count is lower than the quadratic op-count
            (comparing ``S·log₂J_eff·N^q`` vs ``S²`` or ``S²·N``, with
            ``q = kernel.q``);
            otherwise falls back to ``"quadratic"``.
        ``"fft"``
            FFT-based convolution — uniform grids only.
        ``"quadratic"``
            General quadratic recursion — supports non-uniform grids.
        ``"adams"``
            Adams predictor-corrector (product-integration PC/Euler).
            Requires a :class:`FractionalKernel` with ``q=1`` and a
            uniform grid.  ``order=0`` → Euler; ``order≥1`` → PC.

    Returns
    -------
    object
        Native tensor element for the selected core.  With blocking, each
        native block carries an extra block axis at ``axis``.
        With ``output_starting_point=True``, the seed is prepended along
        that axis.

    Raises
    ------
    ValueError
        For invalid truncation, scheme, block size, or non-divisible ``S``.
    """
    core, seq_core = resolve_volterra_core_pair(core, seq_core)
    algebra = resolve_volterra_algebra(core, trunc, kernel.m)
    trunc = algebra.truncation
    max_order = algebra.max_order
    xp = core.xp

    # preprocessing
    X = xp.asarray(X)
    if X.ndim < 2:
        raise ValueError("X must have at least a step axis and a trailing path dimension.")

    axis_norm = axis % X.ndim
    if axis_norm == X.ndim - 1:
        raise ValueError("axis must identify the step axis, not the trailing path dimension.")
    if X.shape[-1] != kernel.path_dim:
        raise ValueError(
            f"X trailing dimension must be {kernel.path_dim}, got {X.shape[-1]}."
        )

    # scheme selection
    _use_adams = scheme == "adams"
    if scheme == "auto":
        if np.ndim(dt) == 0:
            _x_shape = X.shape
            _s_eff = (_x_shape[axis % len(_x_shape)] - (0 if increment_input else 1)) * (1 << dyadic_order)
            _j_eff = 1 << math.ceil(math.log2(max(2 * _s_eff - 1, 2))) if _s_eff > 1 else 1
            _fft_cost = _s_eff * math.log2(_j_eff) * (max_order ** kernel.q)
            _quad_cost = _s_eff ** 2 if kernel.q <= 1 else _s_eff ** 2 * max_order
            _use_fft = _s_eff > 1 and _fft_cost < _quad_cost
        else:
            _use_fft = False
    elif scheme == "fft":
        _use_fft = True
    else:
        _use_fft = False
    if _use_adams and np.ndim(dt) != 0:
        raise ValueError(
            "scheme='adams' requires a scalar dt because it assumes a "
            "uniform grid."
        )
    if _use_fft and np.ndim(dt) != 0:
        raise ValueError(
            "scheme='fft' requires a scalar dt because FFT convolution "
            "assumes a uniform grid."
        )

    dX = X if increment_input else xp.diff(X, axis=axis_norm)
    S_orig = dX.shape[axis_norm]
    if S_orig == 0:
        raise ValueError("vsig requires at least one increment.")

    # dyadic refinement is internal only; block_size always refers to original-grid steps
    factor = 1
    if dyadic_order > 0:
        factor = 1 << int(dyadic_order)
        dX = xp.repeat(dX / factor, factor, axis=axis_norm)
        dt_arr = xp.asarray(dt, dtype=dX.dtype)
        dt = dt_arr / factor if dt_arr.ndim == 0 else xp.repeat(dt_arr / factor, factor)
    if block_size is not None and S_orig % block_size != 0:
        raise ValueError(
            f"S={S_orig} must be divisible by block_size={block_size}."
        )
    output_seed = (
        _resolve_starting_point(
            starting_point,
            algebra=algebra,
            batch_shape=_path_batch_shape(dX.shape, axis_norm),
            dtype=dX.dtype,
        )
        if output_starting_point
        else None
    )

    # --- routing ---
    if _use_adams:
        _iter_fn = _vsig_pc
        _extra_kwargs: dict = {}
    elif _use_fft:
        _iter_fn = _vsig_fft
        _extra_kwargs = {"lag_tables": lag_tables} if lag_tables is not None else {}
    else:
        _iter_fn = _vsig_quadratic
        _extra_kwargs = {}

    _iteration_kwargs = dict(
        kernel=kernel,
        trunc=trunc,
        core=core,
        seq_core=seq_core,
        order=order,
    )

    if block_size is None:
        result = _iter_fn(
            dX,
            dt=dt,
            axis=axis_norm,
            **_iteration_kwargs,
            **_extra_kwargs,
        )
        if output_starting_point:
            result = tree_stack(xp, (output_seed, result), axis=axis_norm)
        return result  # type: ignore[return-value]

    num_blocks = S_orig // block_size
    block_size_ref = block_size * factor  # refined steps per original block

    if accumulate:
        # Output convenience only: run the full scan, subsample at block boundaries.
        # accumulate=True blocking carries no efficiency gain — O(S²) either way.
        full_traj = _iter_fn(
            dX,
            dt=dt,
            axis=axis_norm,
            return_trajectory=True,
            **_iteration_kwargs,
            **_extra_kwargs,
        )
        # Every native block contains [V_1, ..., V_{S_ref}] along axis_norm.
        # Block boundary b maps to refined index b*block_size_ref - 1.
        block_indices = xp.arange(1, num_blocks + 1) * block_size_ref - 1
        result = tree_take(
            xp,
            full_traj,
            block_indices,
            axis=axis_norm,
        )

    else:
        # Independent blocks: split step axis, vmap over blocks.
        # O(S · block_size) — genuine compute saving vs O(S²).
        if not seq_core.supports("map"):
            raise RuntimeError(
                f"{type(seq_core).__name__} does not provide the map "
                "capability required for independent Volterra blocks."
            )
        pre_batch = dX.shape[:axis_norm]
        post_batch = dX.shape[axis_norm + 1:-1]
        dX_blocked = dX.reshape(pre_batch + (num_blocks, block_size_ref) + post_batch + (dX.shape[-1],))

        if np.ndim(dt) == 0:
            def _call_block(dX_b):
                return _iter_fn(
                    dX_b,
                    dt=dt,
                    axis=axis_norm,
                    **_iteration_kwargs,
                    **_extra_kwargs,
                )

            result = seq_core.tensor_map(
                (dX_blocked,),
                map_op=_call_block,
                in_axes=(axis_norm,),
                out_axis=axis_norm,
            )
        else:
            dt_per_block = xp.reshape(dt, (num_blocks, block_size_ref))

            def _call_block(dX_b, dt_b):
                return _iter_fn(
                    dX_b,
                    dt=dt_b,
                    axis=axis_norm,
                    **_iteration_kwargs,
                    **_extra_kwargs,
                )

            result = seq_core.tensor_map(
                (dX_blocked, dt_per_block),
                map_op=_call_block,
                in_axes=(axis_norm, 0),
                out_axis=axis_norm,
            )

    if output_starting_point:
        result = tree_prepend(xp, output_seed, result, axis=axis_norm)
    return result  # type: ignore[return-value]


def vsig(
        X: Array,
        *,
        kernel: ConvolutionKernel,
        trunc=None,
        dt: Array | float = 1.0,
        axis: int = -2,
        block_size: Optional[int] = None,
        accumulate: bool = True,
        starting_point: Any = None,
        output_starting_point: bool = False,
        increment_input: bool = False,
        order: int = 0,
        dyadic_order: int = 0,
        scheme: Literal["auto", "fft", "quadratic", "adams"] = "auto",
        lag_tables: Optional[PrecomputedLagTables] = None,
        core=None,
        seq_core=None,
):
    """Compute the truncated Volterra signature of ``X``."""
    _validate_vsig_static_options(
        scheme=scheme,
        dyadic_order=dyadic_order,
        order=order,
        block_size=block_size,
    )
    core, seq_core = resolve_volterra_core_pair(core, seq_core)
    algebra = resolve_volterra_algebra(core, trunc, kernel.m)
    trunc = algebra.truncation

    # Higher-order basis exponents and their deduplication are a static
    # interpolation schedule.  Specialize on beta *content* in that case;
    # all other kernel arrays (and beta for order zero/Adams) remain dynamic.
    static_leaf_fields = (
        ("beta",)
        if order > 0 and scheme != "adams"
        else ()
    )
    rebuilt = _kernel_rebuilder(
        kernel,
        static_leaf_fields=static_leaf_fields,
    )
    if rebuilt is not None:
        rebuilder, kernel_leaves, kernel_treedef = rebuilt
        key = (
            id(core),
            id(seq_core),
            type(kernel),
            kernel_treedef,
            rebuilder.static_key,
            trunc,
            axis,
            block_size,
            accumulate,
            output_starting_point,
            increment_input,
            order,
            dyadic_order,
            scheme,
        )
        try:
            solver = _cached_compiled_solver(
                key,
                rebuilder=rebuilder,
                trunc=trunc,
                axis=axis,
                block_size=block_size,
                accumulate=accumulate,
                output_starting_point=output_starting_point,
                increment_input=increment_input,
                order=order,
                dyadic_order=dyadic_order,
                scheme=scheme,
                core=core,
                seq_core=seq_core,
            )
        except TypeError:
            # An unhashable static PyTree descriptor cannot key the LRU and
            # therefore uses eager orchestration.
            pass
        else:
            X_arg = core.xp.asarray(X)
            dt_arg = core.xp.asarray(dt)
            starting_arg = starting_point if output_starting_point else None
            return solver(
                X_arg,
                dt_arg,
                kernel_leaves,
                starting_arg,
                lag_tables,
            )

    return _vsig_impl(
        X,
        kernel=kernel,
        trunc=trunc,
        dt=dt,
        axis=axis,
        block_size=block_size,
        accumulate=accumulate,
        starting_point=starting_point,
        output_starting_point=output_starting_point,
        increment_input=increment_input,
        order=order,
        dyadic_order=dyadic_order,
        scheme=scheme,
        lag_tables=lag_tables,
        core=core,
        seq_core=seq_core,
    )

vsig.__doc__ = _vsig_impl.__doc__


def _path_batch_shape(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    return tuple(shape[:axis]) + tuple(shape[axis + 1:-1])


def _resolve_starting_point(
        starting_point,
        *,
        algebra: ResolvedVolterraAlgebra,
        batch_shape: tuple[int, ...],
        dtype,
):
    if starting_point is None:
        return algebra.unit(batch_shape=batch_shape, dtype=dtype)
    for grade in algebra.grades:
        block = algebra.block(starting_point, grade)
        if tuple(block.shape[:-1]) != batch_shape:
            raise ValueError(
                f"starting_point block {grade!r} has batch shape "
                f"{tuple(block.shape[:-1])}, expected {batch_shape}."
            )
    return starting_point


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class VolterraSignature:
    """
    Thin wrapper binding a kernel, truncation, and coherent core pair.

    Parameters
    ----------
    kernel:
        The underlying Volterra kernel.
    trunc:
        Active integer or bidegree truncation.  May be omitted when the bound
        core supplies a default.  Static (changes cause retracing).
    core, seq_core:
        Algebra and sequential backends.  Omitted values resolve from the
        process-wide coherent default pair.
    """

    kernel: ConvolutionKernel
    trunc: Any = field(default=None, metadata={"static": True})
    core: Any = field(
        default=None,
        repr=False,
        compare=False,
        metadata={"static": True},
    )
    seq_core: Any = field(
        default=None,
        repr=False,
        compare=False,
        metadata={"static": True},
    )

    def __post_init__(self) -> None:
        core, seq_core = resolve_volterra_core_pair(self.core, self.seq_core)
        algebra = resolve_volterra_algebra(core, self.trunc, self.kernel.m)
        object.__setattr__(self, "core", core)
        object.__setattr__(self, "seq_core", seq_core)
        object.__setattr__(self, "trunc", algebra.truncation)

    # ------------------------------------------------------------------
    # Convenience constructors — thin wrappers around kernel constructors.
    # ------------------------------------------------------------------

    @classmethod
    def fractional(
            cls,
            *,
            trunc=None,
            core=None,
            seq_core=None,
            **kwargs,
    ) -> "VolterraSignature":
        """Construct from a fractional kernel. Forwards all kwargs to :class:`FractionalKernel`."""
        return cls(
            kernel=FractionalKernel(**kwargs),
            trunc=trunc,
            core=core,
            seq_core=seq_core,
        )

    @classmethod
    def gamma(
            cls,
            *,
            trunc=None,
            core=None,
            seq_core=None,
            **kwargs,
    ) -> "VolterraSignature":
        """Construct from a Gamma kernel. Forwards all kwargs to :class:`GammaKernel`."""
        return cls(
            kernel=GammaKernel(**kwargs),
            trunc=trunc,
            core=core,
            seq_core=seq_core,
        )

    # ------------------------------------------------------------------
    # Forwarded properties
    # ------------------------------------------------------------------

    @property
    def q(self) -> int:
        """Number of scalar kernel components."""
        return self.kernel.q

    @property
    def m(self) -> int:
        """Latent output dimension of each ``A_p``."""
        return self.kernel.m

    @property
    def path_dim(self) -> int:
        """Input path dimension ``d``."""
        return self.kernel.path_dim

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def vsig(
            self,
            X: Array,
            *,
            dt: Array | float = 1.0,
            axis: int = -2,
            block_size: Optional[int] = None,
            accumulate: bool = True,
            starting_point: Any = None,
            output_starting_point: bool = False,
            increment_input: bool = False,
            dyadic_order: int = 0,
            order: int = 0,
            scheme: Literal["auto", "fft", "quadratic", "adams"] = "auto",
            lag_tables: Optional[PrecomputedLagTables] = None,
    ):
        """Compute the truncated Volterra signature of ``X``.

        Thin wrapper around the module-level :func:`vsig`.  ``self.kernel``
        and ``self.trunc`` are forwarded automatically; all other arguments
        are passed through unchanged.

        Parameters
        ----------
        X:
            Path nodes or increments ``(..., S, d)``; step axis ``axis``.
        dt:
            Step size(s).  Scalar → uniform grid; 1-D array → non-uniform.
            Default ``1.0``.
        axis:
            Step axis of ``X`` (default ``-2``).
        block_size:
            Steps per emitted block.  ``None`` → single terminal result.
        accumulate:
            Carry history across blocks (``True``) or treat blocks
            independently via rebatch (``False``).
        starting_point:
            Optional seed prepended when ``output_starting_point=True``.
        output_starting_point:
            If ``True``, prepend the seed to the output along ``axis``.
        increment_input:
            Treat ``X`` as increments; skip :func:`jnp.diff`.
        dyadic_order:
            Dyadic refinement order (default ``0``).
        order:
            Quadrature order for the basis-expansion scheme (default ``0``).
        scheme:
            ``"auto"`` (default), ``"fft"``, or ``"quadratic"``.

        Returns
        -------
        object
            Native Volterra tensor; see module-level :func:`vsig` for details.
        """
        return vsig(
            X,
            kernel=self.kernel,
            trunc=self.trunc,
            dt=dt,
            axis=axis,
            block_size=block_size,
            accumulate=accumulate,
            starting_point=starting_point,
            output_starting_point=output_starting_point,
            increment_input=increment_input,
            dyadic_order=dyadic_order,
            order=order,
            scheme=scheme,
            lag_tables=lag_tables,
            core=self.core,
            seq_core=self.seq_core,
        )


__all__ = ["vsig", "VolterraSignature"]
