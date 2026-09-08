from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from tensordev.core.bigraded.types import BigradedTensor
from tensordev.core.sequential import DenseElem, SequentialCore
from tensordev.core.universal import _Array
from tensordev._backend import (
    _resolve_seq_core,
    get_default_core,
    get_default_core_pair,
)
from .free import (
    _execute_portable_free_development_call,
    _finalize_free_development_call,
    _prepare_free_development_call,
    free_development,
)


def path_signature(
        x: _Array,
        *,
        trunc: Any = None,
        increment_input: bool = False,
        starting_point: Optional[DenseElem | BigradedTensor] = None,
        axis: Optional[int] = None,
        block_size: Optional[int] = None,
        output_starting_point: bool = False,
        # backend
        accumulate: bool = True,
        accumulate_in_tree: bool = False,
        parallel: bool = False,
        core: Any = None,
        seq_core: SequentialCore = None
) -> DenseElem | BigradedTensor:
    """
    Truncated signature of a scalar path.

    A thin wrapper around :func:`free_development` that accepts a single
    level-1 path array ``x`` instead of a full ``DenseElemFirstOn`` tuple.

    Parameters
    ----------
    x : Array
        Path with shape ``batch + (S+1, d)`` when ``increment_input=False``,
        or ``batch + (S, d)`` when ``increment_input=True``.
    trunc : int or pair of int, optional
        Active truncation. The unbounded total-degree core requires an integer;
        a bidegree core accepts ``(N, M)`` and may supply a default.
    increment_input : bool, default False
        If True, ``x`` is already in increment form; skip differencing.
    starting_point : DenseElem or BigradedTensor, optional
        Left seed ``g`` in the selected core's native tensor format; the
        output is ``g ⊗ Sig(x)``.  Defaults to the identity of the selected
        core.
    axis : int, optional
        Step axis of ``x``.  Defaults to ``seq_core.default_time_axis``.
    block_size : int, optional
        Steps per emitted block.  ``None`` → one block covering all steps.
    output_starting_point : bool, default False
        If True, prepend the seed to the output.
    accumulate : bool, default True
        Carry the running product across blocks.
    accumulate_in_tree : bool, default False
        Use an associative tree scan for across-block accumulation.
    parallel : bool, default False
        If True, pre-compute all per-step exponentials and combine with an
        associative tree scan (higher parallelism, higher memory).
        If False, stream via sequential ``tensor_fmexp`` (lower memory,
        sequential depth).
    core :
        Tensor algebra backend.
    seq_core : SequentialCore
        Sequential operations backend.

    Returns
    -------
    DenseElem or BigradedTensor
        Tensor element in the selected core's native format.  With blocking,
        its arrays carry a block axis at ``axis``.
    """
    call = _prepare_free_development_call(
        (x,),
        increment_input=increment_input,
        seq_core=seq_core,
        trunc=trunc,
        axis=axis,
        block_size=block_size,
        accumulate=accumulate,
        starting_point=starting_point,
        output_starting_point=output_starting_point,
        parallel=parallel,
        accumulate_in_tree=accumulate_in_tree,
        core=core,
        colocate_with_input=True,
    )

    # Check concrete device placement before importing an executor or
    # constructing a wordwise plan.
    from tensordev._wordwise.dispatch import ordinary_wordwise_device_eligible

    if not ordinary_wordwise_device_eligible(call):
        return _execute_portable_free_development_call(call)
    from tensordev._wordwise.ordinary import try_ordinary_wordwise

    result = try_ordinary_wordwise(call)
    if result is None:
        return _execute_portable_free_development_call(call)
    return _finalize_free_development_call(
        call,
        result,
        runner_applied_seed=False,
        runner_emitted_starting_point=False,
    )


@dataclass(frozen=True)
class Signature:
    """
    Truncated signature of a scalar path.

    Thin wrapper around :func:`path_signature` that binds ``core``,
    ``seq_core``, and ``trunc`` so they do not need to be repeated at every
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
        Sequential operations backend.  Defaults to the same backend as ``core``.
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
            x: _Array,
            *,
            axis: Optional[int] = None,
            block_size: Optional[int] = None,
            accumulate: bool = True,
            starting_point: Optional[DenseElem | BigradedTensor] = None,
            output_starting_point: bool = False,
            parallel: bool = False,
            accumulate_in_tree: bool = False,
            increment_input: bool = False,
    ) -> DenseElem | BigradedTensor:
        """Compute the signature of ``x``.

        Forwards all arguments to :func:`path_signature` with the bound
        ``core``, ``seq_core``, and ``trunc``.
        """
        return path_signature(
            x,
            increment_input=increment_input,
            accumulate=accumulate,
            trunc=self.trunc,
            axis=axis,
            block_size=block_size,
            accumulate_in_tree=accumulate_in_tree,
            starting_point=starting_point,
            output_starting_point=output_starting_point,
            parallel=parallel,
            core=self.core,
            seq_core=self.seq_core,
        )


__all__ = ["path_signature", "Signature"]
