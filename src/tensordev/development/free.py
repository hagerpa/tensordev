from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Any, Optional

from tensordev.core.bigraded.types import BigradedTensor
from tensordev.core.sequential import DenseElem, SequentialCore
from tensordev.core.universal import DenseElemFirstOn
from tensordev.core.utils.pytrees import (
    tree_first_leaf,
    tree_map,
    tree_prepend,
    tree_stack,
)
from tensordev._backend import (
    _resolve_seq_core,
    get_default_core,
    get_default_core_pair,
    get_default_seq_core,
)


TensorElement = DenseElem | BigradedTensor


@dataclass(frozen=True, slots=True)
class _FreeDevelopmentSeedPolicy:
    """Seed actions shared by portable and specialized development runners."""

    canonical_start: TensorElement
    has_starting_point: bool
    portable_seed: Optional[TensorElement]
    portable_output_starting_point: bool
    output_starting_point: bool


@dataclass(frozen=True, slots=True)
class _PreparedFreeDevelopmentCall:
    """Normalized inputs and static execution policy for a free development."""

    increments: DenseElemFirstOn
    core: Any
    seq_core: SequentialCore
    trunc: Any
    axis: int
    block_size: Optional[int]
    accumulate: bool
    accumulate_in_tree: bool
    parallel: bool
    neutral: TensorElement
    reduce_op: Any
    acc_op: Any
    seed_policy: _FreeDevelopmentSeedPolicy


@lru_cache(maxsize=None)
def _development_ops(core: Any, trunc: Any):
    """The reduce and accumulate operations, built once per `(core, trunc)`.

    Cached deliberately: these are static arguments to `tensor_abra`, and
    partials compare by identity, so rebuilding them per call would recompile.
    """
    return (partial(core.tensor_fmexp, trunc=trunc, output_zero_level=True),
            partial(core.tensor_product, trunc=trunc))


def _prepare_free_development_call(
        X: DenseElemFirstOn,
        *,
        trunc: Any = None,
        increment_input: bool = False,
        starting_point: Optional[TensorElement] = None,
        axis: Optional[int] = None,
        block_size: Optional[int] = None,
        output_starting_point: bool = False,
        accumulate: bool = True,
        accumulate_in_tree: bool = False,
        parallel: bool = False,
        core: Any = None,
        seq_core: SequentialCore = None,
        colocate_with_input: bool = False,
) -> _PreparedFreeDevelopmentCall:
    """Resolve and normalize one free-development call without executing it."""
    if core is None:
        default_core, default_seq_core = get_default_core_pair()
        core = default_core
        if seq_core is None:
            seq_core = default_seq_core
    elif seq_core is None:
        if core is get_default_core():
            seq_core = get_default_seq_core()
        else:
            seq_core = _resolve_seq_core(core, None)

    trunc = core.normalize_truncation(trunc)
    axis = seq_core.default_time_axis if axis is None else axis
    increments = core.prepare_development_input(
        X,
        trunc=trunc,
        increment_input=increment_input,
        axis=axis,
    )
    neutral = core.development_neutral(increments, trunc=trunc, axis=axis)
    reduce_op, acc_op = _development_ops(core, trunc)

    if colocate_with_input:
        from tensordev._wordwise.dispatch import (
            _colocate_pytree,
            _concrete_single_device,
        )

        reference = tree_first_leaf(increments)
        target_device = _concrete_single_device(reference)
        neutral_device = _concrete_single_device(tree_first_leaf(neutral))
        if starting_point is not None or neutral_device != target_device:
            neutral = _colocate_pytree(neutral, reference)
            starting_point = _colocate_pytree(starting_point, reference)

    portable_defers_seed = starting_point is not None and not accumulate
    canonical_start = (
        acc_op(starting_point, neutral)
        if starting_point is not None
        else neutral
    )
    seed_policy = _FreeDevelopmentSeedPolicy(
        canonical_start=canonical_start,
        has_starting_point=starting_point is not None,
        portable_seed=(
            canonical_start
            if starting_point is not None and accumulate
            else None
        ),
        portable_output_starting_point=(
            output_starting_point and not portable_defers_seed
        ),
        output_starting_point=output_starting_point,
    )
    return _PreparedFreeDevelopmentCall(
        increments=increments,
        core=core,
        seq_core=seq_core,
        trunc=trunc,
        axis=axis,
        block_size=block_size,
        accumulate=accumulate,
        accumulate_in_tree=accumulate_in_tree,
        parallel=parallel,
        neutral=neutral,
        reduce_op=reduce_op,
        acc_op=acc_op,
        seed_policy=seed_policy,
    )


def _run_portable_free_development(
        call: _PreparedFreeDevelopmentCall,
) -> TensorElement:
    """Execute a prepared call through its sequential core."""
    return call.seq_core.tensor_abra(
        call.increments,
        reduce_op=call.reduce_op,
        acc_op=call.acc_op,
        neutral=call.neutral,
        axis=call.axis,
        block_size=call.block_size,
        accumulate=call.accumulate,
        seed=call.seed_policy.portable_seed,
        # In the non-accumulating case each independent block is seeded in the
        # shared finalizer. Emitting it here would seed that entry twice.
        output_starting_point=(
            call.seed_policy.portable_output_starting_point
        ),
        first_apply_all=call.parallel,
        reduce_in_tree=call.parallel,
        accumulate_in_tree=call.accumulate_in_tree,
    )


def _finalize_free_development_call(
        call: _PreparedFreeDevelopmentCall,
        result: TensorElement,
        *,
        runner_applied_seed: bool,
        runner_emitted_starting_point: bool,
) -> TensorElement:
    """Complete seed actions not performed by a development runner.

    A runner returns either one terminal element or a sequence of block
    elements in the call's native container and requested axis.  The flags are
    static runner guarantees, allowing the portable and wordwise paths to use
    the same seed and prepend logic without changing their internal arithmetic.
    """
    policy = call.seed_policy
    apply_seed = policy.has_starting_point and not runner_applied_seed
    prepend_start = (
        policy.output_starting_point
        and not runner_emitted_starting_point
    )
    if not apply_seed and not prepend_start:
        return result

    first_increment = tree_first_leaf(call.increments)
    step_axis = (
        call.axis if call.axis >= 0 else first_increment.ndim + call.axis
    )
    steps = int(first_increment.shape[step_axis])
    block_count = (
        1
        if call.block_size in (None, -1)
        else steps // int(call.block_size)
    )

    if block_count == 1:
        if apply_seed:
            result = call.acc_op(policy.canonical_start, result)
        if prepend_start:
            result = tree_stack(
                call.core.xp,
                [policy.canonical_start, result],
                axis=call.axis,
            )
        return result

    if apply_seed:
        broadcast_seed = tree_map(
            lambda leaf: call.core.xp.expand_dims(leaf, axis=call.axis),
            policy.canonical_start,
        )
        result = call.acc_op(broadcast_seed, result)
    if prepend_start:
        result = tree_prepend(
            call.core.xp,
            policy.canonical_start,
            result,
            axis=call.axis,
        )
    return result


def _execute_portable_free_development_call(
        call: _PreparedFreeDevelopmentCall,
) -> TensorElement:
    """Run and finalize a prepared call through the established executor."""
    result = _run_portable_free_development(call)
    return _finalize_free_development_call(
        call,
        result,
        runner_applied_seed=(call.seed_policy.portable_seed is not None),
        runner_emitted_starting_point=(
            call.seed_policy.portable_output_starting_point
        ),
    )


def free_development(
        X: DenseElemFirstOn,
        *,
        trunc: Any = None,
        increment_input: bool = False,
        starting_point: Optional[TensorElement] = None,
        axis: Optional[int] = None,
        block_size: Optional[int] = None,
        output_starting_point: bool = False,
        # backend
        accumulate: bool = True,
        accumulate_in_tree: bool = False,
        parallel: bool = False,
        core: Any = None,
        seq_core: SequentialCore = None
) -> TensorElement:
    """
    Truncated free development of a tensor-valued path.

    Computes the running Chen product
        S = exp(dX_1) ⊗ exp(dX_2) ⊗ ... ⊗ exp(dX_S)
    truncated at degree ``trunc``, optionally left-seeded by ``starting_point``.

    Parameters
    ----------
    X : DenseElemFirstOn
        Tensor-valued path levels starting at degree 1.  For a total-degree
        core, level k has shape ``batch + (S+1, d**k)`` when
        ``increment_input=False``, or ``batch + (S, d**k)`` when
        ``increment_input=True``.  Bidegree cores accept one combined
        first-level path array, supplied as ``(X1,)``, whose width is
        ``sum(core.dims)``.
    trunc : int or pair of int, optional
        Active truncation. The unbounded total-degree core requires an integer;
        a bidegree core accepts ``(N, M)`` and may supply a default.
    increment_input : bool, default False
        If True, ``X`` is already in increment form; skip differencing.
    starting_point : DenseElem or BigradedTensor, optional
        Left seed ``g`` in the selected core's native tensor format; the
        output is ``g ⊗ S``.  Defaults to the identity of the selected core.
    axis : int, optional
        Step axis of ``X``.  Defaults to ``seq_core.default_time_axis``.
    block_size : int, optional
        Steps per emitted block.  ``None`` → one block covering all steps.
    output_starting_point : bool, default False
        If True, prepend the seed to the output.
    accumulate : bool, default True
        Carry the running product across blocks.
    accumulate_in_tree : bool, default False
        Use an associative tree scan for across-block accumulation.
    parallel : bool, default False
        If True, pre-compute all per-step exponentials via
        ``tensor_fmexp(neutral, step)`` and combine them with
        ``tensor_product`` using an associative tree scan (higher parallelism,
        higher memory).  If False, stream via sequential ``tensor_fmexp``
        (lower memory, sequential depth).
    core :
        Tensor algebra backend.  Must provide ``tensor_fmexp``,
        ``tensor_product``, ``tensor_exponential``, and ``xp``.
    seq_core : SequentialCore
        Sequential operations backend.

    Returns
    -------
    DenseElem or BigradedTensor
        Tensor element in the selected core's native format.  With blocking,
        its arrays carry a block axis at ``axis``.
    """
    call = _prepare_free_development_call(
        X,
        trunc=trunc,
        increment_input=increment_input,
        starting_point=starting_point,
        axis=axis,
        block_size=block_size,
        output_starting_point=output_starting_point,
        accumulate=accumulate,
        accumulate_in_tree=accumulate_in_tree,
        parallel=parallel,
        core=core,
        seq_core=seq_core,
    )
    return _execute_portable_free_development_call(call)


@dataclass(frozen=True)
class FreeDevelopment:
    """
    Truncated free development of a tensor-valued path.

    Thin wrapper around :func:`free_development` that binds ``core``,
    ``seq_core``, and ``trunc`` so they do not have to be repeated at every
    call site.

    Parameters
    ----------
    trunc : int or pair of int, optional
        Active truncation. May be omitted when the bound core supplies a default.
    core : optional
        Tensor algebra core. Defaults to the active core configured by
        :func:`tensordev.set_default_core` or, initially, by the
        ``TENSORDEV_BACKEND`` environment variable.
    seq_core : SequentialCore, optional
        Sequential operations backend. Defaults to the same backend as ``core``.
    """

    trunc: Any = None
    core: Any = field(default=None, repr=False, compare=False)
    seq_core: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.core is None:
            object.__setattr__(self, "core", get_default_core())
        object.__setattr__(
            self, "trunc", self.core.normalize_truncation(self.trunc)
        )
        if self.seq_core is None:
            default_core, default_seq_core = get_default_core_pair()
            object.__setattr__(
                self,
                "seq_core",
                default_seq_core
                if self.core is default_core
                else _resolve_seq_core(self.core, None),
            )

    def __call__(
            self,
            X: DenseElemFirstOn,
            *,
            axis: Optional[int] = None,
            block_size: Optional[int] = None,
            accumulate: bool = True,
            starting_point: Optional[TensorElement] = None,
            output_starting_point: bool = False,
            parallel: bool = False,
            accumulate_in_tree: bool = False,
            increment_input: bool = False,
    ) -> TensorElement:
        """Compute the free development of ``X``.

        Forwards all arguments to :func:`free_development` with the bound
        ``core``, ``seq_core``, and ``trunc``.
        """
        return free_development(
            X,
            increment_input=increment_input,
            seq_core=self.seq_core,
            trunc=self.trunc,
            axis=axis,
            block_size=block_size,
            accumulate=accumulate,
            starting_point=starting_point,
            output_starting_point=output_starting_point,
            parallel=parallel,
            accumulate_in_tree=accumulate_in_tree,
            core=self.core,
        )


__all__ = ["free_development", "FreeDevelopment"]
