from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

from tensordev.core.sequential import DenseElem, SequentialCore
from tensordev.core.universal import DenseElemFirstOn
from tensordev._backend import get_default_core, get_default_seq_core


def free_development(
        X: DenseElemFirstOn,
        *,
        trunc: int,
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
    trunc : int
        Maximum output degree (inclusive).
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
        core = get_default_core()
    if seq_core is None:
        seq_core = get_default_seq_core()

    X = tuple(X)
    if not X:
        raise ValueError("free_development: X must contain at least one level.")

    axis_ = seq_core.default_time_axis if axis is None else axis

    dX = X[:trunc] if increment_input else tuple(
        core.xp.diff(L, axis=axis_) for L in X[:trunc]
    )

    t = axis_ if axis_ >= 0 else dX[0].ndim + axis_
    idx = tuple(0 if i == t else slice(None) for i in range(dX[0].ndim))
    zero1 = core.xp.zeros_like(dX[0][idx])
    neutral = core.tensor_exponential((zero1,), trunc=trunc, output_zero_level=True)

    reduce_op = partial(core.tensor_fmexp, trunc=trunc, output_zero_level=True)
    acc_op = partial(core.tensor_product, trunc=trunc)

    seed = acc_op(starting_point, neutral) if starting_point is not None else neutral

    result = seq_core.tensor_abra(
        dX,
        reduce_op=reduce_op,
        acc_op=acc_op,
        neutral=neutral,
        axis=axis_,
        block_size=block_size,
        accumulate=accumulate,
        seed=seed,
        output_starting_point=output_starting_point,
        first_apply_all=parallel,
        reduce_in_tree=parallel,
        accumulate_in_tree=accumulate_in_tree,
    )

    # tensor_abra ignores seed when accumulate=False; apply starting_point here.
    if starting_point is not None and not accumulate:
        result = acc_op(starting_point, result)
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
    trunc : int
        Truncation level.
    core : optional
        Tensor algebra backend. Defaults to the backend selected by the
        ``TENSORDEV_BACKEND`` environment variable (default: ``"jax"``).
    seq_core : SequentialCore, optional
        Sequential operations backend. Defaults to the same backend as ``core``.
    """

    trunc: int
    core: Any = field(default=None, repr=False, compare=False)
    seq_core: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.trunc < 0:
            raise ValueError(f"trunc must be non-negative, got {self.trunc}.")
        if self.core is None:
            object.__setattr__(self, "core", get_default_core())
        if self.seq_core is None:
            object.__setattr__(self, "seq_core", get_default_seq_core())

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
