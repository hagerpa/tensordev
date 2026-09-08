"""Assembly layer for ordinary wordwise signature execution."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import jax.numpy as jnp

from tensordev._wordwise.dispatch import ordinary_wordwise_device_eligible
from tensordev._wordwise.layout import (
    PlanResourceError,
    WordwiseLayoutPlan,
    build_layout_plan,
)
from tensordev._wordwise.pallas_ordinary import (
    PallasOrdinaryError,
    ordered_ordinary_pallas,
)
from tensordev._wordwise.pallas_quotient import quotient_ordinary_pallas


@dataclass(frozen=True, slots=True)
class _CanonicalOrdinaryInput:
    increments: Any
    batch_shape: tuple[int, ...]
    output_axis: int

    def restore_block(self, block: Any) -> Any:
        """Restore flattened batch axes and the requested emitted-block axis."""
        if block.ndim == 2:
            return block.reshape(self.batch_shape + (block.shape[-1],))
        if block.ndim != 3:
            raise ValueError(
                "a wordwise block must have rank two or three, got "
                f"shape {block.shape}."
            )
        restored = block.reshape(
            self.batch_shape + (block.shape[-2], block.shape[-1])
        )
        return jnp.moveaxis(restored, -2, self.output_axis)


def _canonicalize_ordinary_input(call: Any) -> _CanonicalOrdinaryInput:
    increment = call.increments[0]
    axis = call.axis % increment.ndim
    moved = jnp.moveaxis(increment, axis, -2)
    batch_shape = tuple(map(int, moved.shape[:-2]))
    canonical = moved.reshape(
        (prod(batch_shape) if batch_shape else 1,)
        + tuple(map(int, moved.shape[-2:]))
    )
    return _CanonicalOrdinaryInput(
        increments=canonical,
        batch_shape=batch_shape,
        output_axis=call.axis,
    )


def _ordered_standard_signature(
        call: Any,
        plan: WordwiseLayoutPlan,
        *,
        interpret: bool,
        tile_words: int,
) -> Any:
    if plan.partially_symmetrized:
        raise ValueError("an ordered executor requires an ordered layout plan.")

    canonical = _canonicalize_ordinary_input(call)
    output_blocks: list[Any | None] = [None] * len(plan.blocks)
    for group in plan.execution_groups:
        values = ordered_ordinary_pallas(
            canonical.increments,
            degree=group.total_degree,
            word_codes=group.decoder_codes,
            block_size=call.block_size,
            accumulate=call.accumulate,
            tile_words=tile_words,
            interpret=interpret,
        )
        values = canonical.restore_block(values)
        for block_index, (start, stop) in zip(
            group.block_indices,
            group.block_slices,
        ):
            output_blocks[block_index] = values[..., start:stop]

    if any(block is None for block in output_blocks):
        raise AssertionError("wordwise execution did not produce every block.")
    return plan.assemble_signature(output_blocks)


def _quotient_standard_signature(
        call: Any,
        plan: WordwiseLayoutPlan,
        *,
        interpret: bool,
        tile_prime_words: int,
) -> Any:
    if not plan.partially_symmetrized or plan.grading != "bidegree":
        raise ValueError(
            "a quotient executor requires a partially symmetrized "
            "bidegree layout plan."
        )

    canonical = _canonicalize_ordinary_input(call)
    output_blocks = []
    for block in plan.blocks:
        values = quotient_ordinary_pallas(
            canonical.increments,
            block_plan=block,
            d_prime=plan.dims[0],
            block_size=call.block_size,
            accumulate=call.accumulate,
            tile_prime_words=tile_prime_words,
            interpret=interpret,
        )
        output_blocks.append(canonical.restore_block(values))
    return plan.assemble_signature(output_blocks)


def run_ordinary_wordwise(
        call: Any,
        *,
        plan: WordwiseLayoutPlan | None = None,
        interpret: bool = False,
        tile_words: int = 128,
        tile_prime_words: int = 16,
) -> Any:
    """Run one prepared ordered or quotient signature from the tensor unit.

    The returned tensor is in the selected core's native coordinates, but an
    arbitrary user seed and an optional prepended starting point have not yet
    been applied.  Those operations remain in the shared development
    finalizer.
    """
    increment = call.increments[0]
    plan = (
        build_layout_plan(
            call.core,
            call.trunc,
            alphabet_dim=int(increment.shape[-1]),
        )
        if plan is None
        else plan
    )
    if plan.partially_symmetrized:
        standard = _quotient_standard_signature(
            call,
            plan,
            interpret=interpret,
            tile_prime_words=tile_prime_words,
        )
    else:
        standard = _ordered_standard_signature(
            call,
            plan,
            interpret=interpret,
            tile_words=tile_words,
        )
    if getattr(call.core, "coordinates", "standard") == "standard":
        return standard
    return call.core.tensor_from_standard_coordinates(
        standard,
        trunc=call.trunc,
        first_on=False,
    )


def try_ordinary_wordwise(call: Any) -> Any | None:
    """Try the conservative automatic CUDA path, returning ``None`` on fallback."""
    if not ordinary_wordwise_device_eligible(call):
        return None
    try:
        plan = build_layout_plan(
            call.core,
            call.trunc,
            alphabet_dim=int(call.increments[0].shape[-1]),
        )
        return run_ordinary_wordwise(call, plan=plan)
    except (PlanResourceError, PallasOrdinaryError):
        return None


__all__ = ["run_ordinary_wordwise", "try_ordinary_wordwise"]
