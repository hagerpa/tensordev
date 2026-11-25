"""Wordwise execution of prepared signature calls."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from tensordev._wordwise.kernel import WordwiseError, wordwise_blocks
from tensordev._wordwise.layout import PlanResourceError, WordwiseLayoutPlan, build_layout_plan
from tensordev.development.free import (
    _execute_portable_free_development_call,
    _finalize_free_development_call,
    _prepare_free_development_call,
)


def run_ordinary_wordwise(
        call: Any,
        *,
        plan: WordwiseLayoutPlan | None = None,
        **options: Any,
) -> Any:
    """Signature of one prepared call from the tensor unit, in the coordinates of its core.

    A user seed and a prepended starting point are applied by the shared development finalizer;
    ``options`` are those of :func:`~tensordev._wordwise.kernel.wordwise_blocks`.
    """
    increments = jnp.moveaxis(call.increments[0], call.axis, -2)
    batch_shape = increments.shape[:-2]
    plan = plan or build_layout_plan(call.core, call.trunc, alphabet_dim=int(increments.shape[-1]))
    blocks = wordwise_blocks(
        increments.reshape((-1,) + increments.shape[-2:]),
        plan=plan, block_size=call.block_size, accumulate=call.accumulate, **options,
    )

    def restore(block):
        """Batch axes of the call, and the axis of emitted blocks in place of the path axis."""
        block = block.reshape(batch_shape + block.shape[1:])
        return block if block.ndim == len(batch_shape) + 1 else jnp.moveaxis(block, -2, call.axis)

    standard = plan.assemble_signature([restore(block) for block in blocks])
    if getattr(call.core, "coordinates", "standard") == "standard":
        return standard
    return call.core.tensor_from_standard_coordinates(standard, trunc=call.trunc, first_on=False)


def try_ordinary_wordwise(call: Any) -> Any | None:
    """The wordwise signature, or ``None`` if the plan or the kernel does not admit the call."""
    try:
        return run_ordinary_wordwise(call)
    except (PlanResourceError, WordwiseError):
        return None


@partial(
    jax.jit,
    static_argnames=(
        "trunc", "increment_input", "axis", "block_size", "output_starting_point", "accumulate", "core",
        "seq_core", "execution",
    ),
)
def wordwise_path_signature(
        x: Any,
        starting_point: Any,
        *,
        trunc: Any,
        increment_input: bool,
        axis: int | None,
        block_size: int | None,
        output_starting_point: bool,
        accumulate: bool,
        core: Any,
        seq_core: Any,
        execution: str,
) -> Any:
    """:func:`~tensordev.path_signature` through the wordwise kernel, as one compiled program.

    Under ``execution="auto"`` a call that the plan or the kernel does not admit takes the
    portable recursion.
    """
    call = _prepare_free_development_call(
        (x,), trunc=trunc, increment_input=increment_input, starting_point=starting_point, axis=axis,
        block_size=block_size, output_starting_point=output_starting_point, accumulate=accumulate,
        core=core, seq_core=seq_core,
    )
    result = run_ordinary_wordwise(call) if execution == "wordwise" else try_ordinary_wordwise(call)
    if result is None:
        if execution == "wordwise":
            raise RuntimeError("execution='wordwise' did not produce a signature.")
        return _execute_portable_free_development_call(call)
    return _finalize_free_development_call(
        call, result, runner_applied_seed=False, runner_emitted_starting_point=False
    )


__all__ = ["run_ordinary_wordwise", "try_ordinary_wordwise", "wordwise_path_signature"]
