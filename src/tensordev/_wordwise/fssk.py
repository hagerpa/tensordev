"""Wordwise execution of prepared scalar-FSSK calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from tensordev._wordwise.kernel import WordwiseError, WordwiseUnsupportedError, wordwise_blocks
from tensordev._wordwise.layout import PlanResourceError, WordwiseLayoutPlan, build_layout_plan


@dataclass(frozen=True, slots=True)
class PreparedFSSKQ1Call:
    """Canonical scalar-FSSK data prepared by the shared SSS front end.

    ``y_time`` has shape ``(S, B, m)``.  The coefficient arrays have shapes
    ``(T, C, R, R)``, ``(T, C, N, R)``, and
    ``(T, C, N - 1, R, R)``, respectively, where ``T`` is one or ``S`` and
    ``C`` is one or ``B``. Their q-axis has already been removed. ``B`` is the
    product of ``batch_shape``; singleton coefficient batches stay compact.

    ``initial_standard`` is either ``None`` or a standard-coordinate,
    first-on tensor whose block-leading axis is one or ``B``. ``batch_shape``,
    ``axis``, and ``output_starting_state`` are restoration metadata for the
    shared SSS finalizer; this executor does not apply them.
    """

    y_time: Any
    E_time: Any
    psi_time: Any
    phi_time: Any
    initial_standard: Any
    core: Any
    trunc: Any
    batch_shape: tuple[int, ...]
    axis: int
    block_size: int | None
    accumulate: bool
    output_starting_state: bool


def _kernel_blocks(
        call: PreparedFSSKQ1Call,
        *,
        plan: WordwiseLayoutPlan | None,
        weights: Any = None,
        **options: Any,
) -> tuple[WordwiseLayoutPlan, tuple[Any, ...]]:
    """Plan of a prepared call and its kernel blocks, batch first.

    ``options`` are those of :func:`~tensordev._wordwise.kernel.wordwise_blocks`.
    """
    y, E, psi, phi = (
        jnp.moveaxis(value, 0, 1) for value in (call.y_time, call.E_time, call.psi_time, call.phi_time)
    )
    plan = plan or build_layout_plan(call.core, call.trunc, alphabet_dim=int(y.shape[-1]))
    seed = call.initial_standard
    if seed is not None:
        seed = jnp.concatenate(
            [jnp.moveaxis(block[:, 0, 0], -1, -2) for block in getattr(seed, "blocks", seed)], axis=1
        )
    return plan, wordwise_blocks(
        y, plan=plan, coefficients=(E, psi, phi), initial=seed, weights=weights,
        block_size=call.block_size, accumulate=call.accumulate, **options,
    )


def run_fssk_q1_wordwise(
        call: PreparedFSSKQ1Call,
        *,
        plan: WordwiseLayoutPlan | None = None,
        **options: Any,
) -> Any:
    """Standard-coordinate states of one prepared call.

    Every first-on block has shape ``(n_blocks, B, 1, 1, R, width)``.  The initial state enters
    the recursion; prepending it to the output, squeezing a sole terminal block, restoring public
    axes and converting coordinates belong to the shared SSS finalizer.
    """
    plan, blocks = _kernel_blocks(call, plan=plan, **options)
    return plan.assemble_first_on([
        jnp.moveaxis(block.reshape((block.shape[0], -1) + block.shape[-2:]), 1, 0)[:, :, None, None]
        for block in blocks
    ])


def run_fssk_q1_wordwise_readout(
        call: PreparedFSSKQ1Call,
        weights: Any,
        *,
        plan: WordwiseLayoutPlan | None = None,
        **options: Any,
) -> Any:
    """Terminal standard-coordinate signature of one prepared call, the scalar unit included.

    ``weights`` has shape ``(1 or B, R)``; the blocks keep the flattened batch axis and have no
    block axis.  Calls that emit blocks or their starting state take the state-then-readout path.
    """
    if call.block_size is not None:
        raise WordwiseUnsupportedError("direct scalar-FSSK readout requires block_size=None.")
    if call.output_starting_state:
        raise WordwiseUnsupportedError("direct scalar-FSSK readout cannot emit the starting state.")
    plan, blocks = _kernel_blocks(call, plan=plan, weights=weights, **options)
    return plan.assemble_signature(blocks)


def try_fssk_q1_wordwise(call: PreparedFSSKQ1Call, **options: Any) -> Any | None:
    """The wordwise states, or ``None`` if the plan or the kernel does not admit the call."""
    try:
        return run_fssk_q1_wordwise(call, **options)
    except (PlanResourceError, WordwiseError):
        return None


def try_fssk_q1_wordwise_readout(call: PreparedFSSKQ1Call, weights: Any, **options: Any) -> Any | None:
    """The wordwise readout, or ``None`` if the plan or the kernel does not admit the call."""
    try:
        return run_fssk_q1_wordwise_readout(call, weights, **options)
    except (PlanResourceError, WordwiseError):
        return None


__all__ = [
    "PreparedFSSKQ1Call",
    "run_fssk_q1_wordwise",
    "run_fssk_q1_wordwise_readout",
    "try_fssk_q1_wordwise",
    "try_fssk_q1_wordwise_readout",
]
