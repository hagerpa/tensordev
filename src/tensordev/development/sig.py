from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import jax

from tensordev.core.bigraded.types import BigradedTensor
from tensordev.core.sequential import DenseElem, SequentialCore
from tensordev.core.universal import _Array
from tensordev._backend import _resolve_core_pair
from .free import (
    _execute_portable_free_development_call,
    _prepare_free_development_call,
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
        seq_core: SequentialCore = None,
        execution: str = "auto",
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
    execution : {"auto", "jax", "wordwise"}, default "auto"
        ``"auto"`` retains the default execution policy; ``"jax"`` forces
        portable JAX. ``"wordwise"`` explicitly requests the alpha NVIDIA
        GPU kernels for float32/float64 input on one device with CUDA compute
        capability 8.0 or newer. Wordwise execution requires eager calls with
        ``parallel=False`` and ``accumulate_in_tree=False``; it does not
        support outer ``jit``, ``grad``, or ``vmap`` transformations.
        Unsupported explicit requests raise instead of falling back.

    Returns
    -------
    DenseElem or BigradedTensor
        Tensor element in the selected core's native format.  With blocking,
        its arrays carry a block axis at ``axis``.
    """
    from tensordev._wordwise.dispatch import (
        _colocate_pytree,
        _concrete_single_device,
        _validate_execution,
        ordinary_wordwise_device_eligible,
    )

    _validate_execution(execution)
    core, seq_core = _resolve_core_pair(core, seq_core)
    trunc = core.normalize_truncation(trunc)
    # Decided from the inputs, before an executor is imported or a plan is built.
    if ordinary_wordwise_device_eligible(
        execution=execution,
        parallel=parallel,
        accumulate_in_tree=accumulate_in_tree,
        core=core,
        seq_core=seq_core,
        reference=x,
        differentiable_inputs=(x, starting_point),
    ):
        from tensordev._wordwise.ordinary import wordwise_path_signature

        # Committed to the checked device, which therefore runs the program.
        return wordwise_path_signature(
            jax.device_put(x, _concrete_single_device(x)),
            _colocate_pytree(starting_point, x),
            trunc=trunc,
            increment_input=increment_input,
            axis=axis,
            block_size=block_size,
            output_starting_point=output_starting_point,
            accumulate=accumulate,
            core=core,
            seq_core=seq_core,
            execution=execution,
        )
    return _execute_portable_free_development_call(
        _prepare_free_development_call(
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
        core, seq_core = _resolve_core_pair(self.core, self.seq_core)
        object.__setattr__(self, "core", core)
        object.__setattr__(self, "seq_core", seq_core)
        object.__setattr__(self, "trunc", core.normalize_truncation(self.trunc))

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
            execution: str = "auto",
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
            execution=execution,
        )


__all__ = ["path_signature", "Signature"]
