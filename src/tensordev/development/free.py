from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Any, Optional

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


@lru_cache(maxsize=None)
def _development_ops(core: Any, trunc: Any):
    """The reduce and accumulate operations, built once per `(core, trunc)`.

    Cached deliberately: these are static arguments to `tensor_abra`, and
    partials compare by identity, so rebuilding them per call would recompile.
    """
    return (partial(core.tensor_fmexp, trunc=trunc, output_zero_level=True),
            partial(core.tensor_product, trunc=trunc))


def free_development(
        X: DenseElemFirstOn,
        *,
        trunc: Any = None,
        increment_input: bool = False,
        starting_point: Optional[DenseElem] = None,
        axis: Optional[int] = None,
        block_size: Optional[int] = None,
        output_starting_point: bool = False,
        # backend
        accumulate: bool = True,
        accumulate_in_tree: bool = False,
        parallel: bool = False,
        core: Any = None,
        seq_core: SequentialCore = None
) -> DenseElem:
    """
    Truncated free development of a tensor-valued path.

    Computes the running Chen product
        S = exp(dX_1) ⊗ exp(dX_2) ⊗ ... ⊗ exp(dX_S)
    truncated at degree ``trunc``, optionally left-seeded by ``starting_point``.

    Parameters
    ----------
    X : DenseElemFirstOn
        Tensor-valued path levels starting at degree 1.  Level k has shape
        ``batch + (S+1, d**k)`` when ``increment_input=False``, or
        ``batch + (S, d**k)`` when ``increment_input=True``.
    trunc : int or pair of int, optional
        Active truncation. The unbounded total-degree core requires an integer;
        a bidegree core accepts ``(N, M)`` and may supply a default.
    increment_input : bool, default False
        If True, ``X`` is already in increment form; skip differencing.
    starting_point : DenseElem, optional
        Left seed ``g``; the output is ``g ⊗ S``.  Defaults to the identity.
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
    DenseElem
        Terminal signature when no blocking is requested, or packed levels with
        a block axis at ``axis`` otherwise.
    """
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

    axis_ = seq_core.default_time_axis if axis is None else axis
    dX = core.prepare_development_input(
        X,
        trunc=trunc,
        increment_input=increment_input,
        axis=axis_,
    )
    neutral = core.development_neutral(dX, trunc=trunc, axis=axis_)

    reduce_op, acc_op = _development_ops(core, trunc)

    post_seed_blocks = starting_point is not None and not accumulate
    canonical_start = (
        acc_op(starting_point, neutral)
        if starting_point is not None
        else neutral
    )
    seed = canonical_start if starting_point is not None and accumulate else None
    result = seq_core.tensor_abra(
        dX,
        reduce_op=reduce_op,
        acc_op=acc_op,
        neutral=neutral,
        axis=axis_,
        block_size=block_size,
        accumulate=accumulate,
        seed=seed,
        # In the non-accumulating case each independent block is seeded below.
        # Letting tensor_abra emit the seed here would seed that entry twice.
        output_starting_point=(output_starting_point and not post_seed_blocks),
        first_apply_all=parallel,
        reduce_in_tree=parallel,
        accumulate_in_tree=accumulate_in_tree,
    )

    # tensor_abra deliberately ignores seed when accumulate=False: the blocks
    # are independent.  Left-multiply every block by the same starting point,
    # introducing a singleton block axis only when several blocks were emitted.
    if post_seed_blocks:
        first_increment = tree_first_leaf(dX)
        step_axis = axis_ if axis_ >= 0 else first_increment.ndim + axis_
        steps = int(first_increment.shape[step_axis])
        block_count = (
            1 if block_size in (None, -1) else steps // int(block_size)
        )

        if block_count == 1:
            result = acc_op(canonical_start, result)
            if output_starting_point:
                result = tree_stack(
                    core.xp,
                    [canonical_start, result],
                    axis=axis_,
                )
        else:
            broadcast_seed = tree_map(
                lambda leaf: core.xp.expand_dims(leaf, axis=axis_),
                canonical_start,
            )
            result = acc_op(broadcast_seed, result)
            if output_starting_point:
                result = tree_prepend(
                    core.xp,
                    canonical_start,
                    result,
                    axis=axis_,
                )
    return result


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
        Tensor algebra backend. Defaults to the backend selected by the
        ``TENSORDEV_BACKEND`` environment variable (default: ``"jax"``).
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
            starting_point: Optional[DenseElem] = None,
            output_starting_point: bool = False,
            parallel: bool = False,
            accumulate_in_tree: bool = False,
            increment_input: bool = False,
    ) -> DenseElem:
        """Compute the free development of ``X``.

        Forwards all arguments to :func:`free_development` with the bound
        ``core``, ``seq_core``, and ``trunc``.
        """
        return free_development(X, increment_input=increment_input, seq_core=self.seq_core, trunc=self.trunc, axis=axis,
                                block_size=block_size, accumulate=accumulate, starting_point=starting_point,
                                output_starting_point=output_starting_point, parallel=parallel,
                                accumulate_in_tree=accumulate_in_tree, core=self.core)


__all__ = ["free_development", "FreeDevelopment"]
