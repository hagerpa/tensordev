from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from tensordev.core.sequential import DenseElem, SequentialCore
from tensordev.core.universal import _Array
from tensordev._backend import get_default_core, get_default_seq_core
from .free import free_development


def path_signature(
        x: _Array,
        *,
        increment_input: bool = False,
        accumulate: bool = True,
        trunc: int,
        axis: Optional[int] = None,
        block_size: Optional[int] = None,
        accumulate_in_tree: bool = False,
        starting_point: Optional[DenseElem] = None,
        output_starting_point: bool = False,
        parallel: bool = False,
        core: Any = None,
        seq_core: SequentialCore = None
) -> DenseElem:
    """
    Truncated signature of a scalar path.

    A thin wrapper around :func:`free_development` that accepts a single
    level-1 path array ``x`` instead of a full ``DenseElemFirstOn`` tuple.

    Parameters
    ----------
    x : Array
        Path with shape ``batch + (S+1, d)`` when ``increment_input=False``,
        or ``batch + (S, d)`` when ``increment_input=True``.
    core :
        Tensor algebra backend.
    seq_core : SequentialCore
        Sequential operations backend.
    trunc : int
        Maximum output degree (inclusive).
    axis : int, optional
        Step axis of ``x``.  Defaults to ``seq_core.default_time_axis``.
    block_size : int, optional
        Steps per emitted block.  ``None`` → one block covering all steps.
    accumulate : bool, default True
        Carry the running product across blocks.
    starting_point : DenseElem, optional
        Left seed ``g``; the output is ``g ⊗ Sig(x)``.
    output_starting_point : bool, default False
        If True, prepend the seed to the output.
    parallel : bool, default False
        If True, pre-compute all per-step exponentials and combine with an
        associative tree scan (higher parallelism, higher memory).
        If False, stream via sequential ``tensor_fmexp`` (lower memory).
    accumulate_in_tree : bool, default False
        Use an associative tree scan for across-block accumulation.
    increment_input : bool, default False
        If True, ``x`` is already in increment form; skip differencing.

    Returns
    -------
    DenseElem
        Terminal signature when no blocking is requested, or packed levels
        with a block axis at ``axis`` otherwise.
    """
    return free_development((x,), increment_input=increment_input, seq_core=seq_core, trunc=trunc, axis=axis,
                            block_size=block_size, accumulate=accumulate, starting_point=starting_point,
                            output_starting_point=output_starting_point, parallel=parallel,
                            accumulate_in_tree=accumulate_in_tree, core=core)


@dataclass(frozen=True)
class Signature:
    """
    Truncated signature of a scalar path.

    Thin wrapper around :func:`path_signature` that binds ``core``,
    ``seq_core``, and ``trunc`` so they do not need to be repeated at every
    call site.

    Parameters
    ----------
    trunc : int
        Truncation level.
    core : optional
        Tensor algebra backend.  Defaults to the backend selected by the
        ``TENSORDEV_BACKEND`` environment variable (default: ``"jax"``).
    seq_core : SequentialCore, optional
        Sequential operations backend.  Defaults to the same backend as ``core``.
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
            x: _Array,
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
        """Compute the signature of ``x``.

        Forwards all arguments to :func:`path_signature` with the bound
        ``core``, ``seq_core``, and ``trunc``.
        """
        return path_signature(x, increment_input=increment_input, accumulate=accumulate, trunc=self.trunc, axis=axis,
                              block_size=block_size, accumulate_in_tree=accumulate_in_tree,
                              starting_point=starting_point, output_starting_point=output_starting_point,
                              parallel=parallel, core=self.core, seq_core=self.seq_core)


__all__ = ["path_signature", "Signature"]
