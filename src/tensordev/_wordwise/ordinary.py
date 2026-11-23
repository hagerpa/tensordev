"""Wordwise execution of prepared signature calls."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from tensordev._wordwise.dispatch import _concrete_single_device, ordinary_wordwise_device_eligible
from tensordev._wordwise.kernel import WordwiseError, wordwise_blocks
from tensordev._wordwise.layout import PlanResourceError, WordwiseLayoutPlan, build_layout_plan


def run_ordinary_wordwise(
        call: Any,
        *,
        plan: WordwiseLayoutPlan | None = None,
        interpret: bool = False,
        tile: int = 32,
) -> Any:
    """Signature of one prepared call from the tensor unit, in the coordinates of its core.

    A user seed and a prepended starting point are applied by the shared development finalizer.
    """
    increments = jnp.moveaxis(call.increments[0], call.axis, -2)
    batch_shape = increments.shape[:-2]
    plan = plan or build_layout_plan(call.core, call.trunc, alphabet_dim=int(increments.shape[-1]))
    blocks = wordwise_blocks(
        increments.reshape((-1,) + increments.shape[-2:]),
        plan=plan, block_size=call.block_size, accumulate=call.accumulate, tile=tile, interpret=interpret,
    )
    unit = jnp.ones(
        blocks[0].shape[:-1] + (1,), dtype=increments.dtype, device=_concrete_single_device(increments)
    )

    def restore(block):
        """Batch axes of the call, and the axis of emitted blocks in place of the path axis."""
        block = block.reshape(batch_shape + block.shape[1:])
        return block if block.ndim == len(batch_shape) + 1 else jnp.moveaxis(block, -2, call.axis)

    standard = plan.assemble_signature([restore(block) for block in (unit, *blocks)])
    if getattr(call.core, "coordinates", "standard") == "standard":
        return standard
    return call.core.tensor_from_standard_coordinates(standard, trunc=call.trunc, first_on=False)


def try_ordinary_wordwise(call: Any) -> Any | None:
    """The wordwise signature, or ``None`` if the device, the plan or the kernel does not admit it."""
    if not ordinary_wordwise_device_eligible(call):
        return None
    try:
        return run_ordinary_wordwise(call)
    except (PlanResourceError, WordwiseError):
        return None


__all__ = ["run_ordinary_wordwise", "try_ordinary_wordwise"]
