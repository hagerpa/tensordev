"""Assembly of prepared scalar-FSSK calls for wordwise execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from tensordev._wordwise.layout import (
    PlanResourceError,
    WordwiseLayoutPlan,
    build_layout_plan,
)
from tensordev._wordwise.pallas_fssk_q1 import (
    PallasFSSKPlanError,
    ordered_fssk_q1_pallas,
    ordered_fssk_q1_readout_pallas,
)
from tensordev._wordwise.pallas_fssk_quotient import (
    quotient_fssk_q1_pallas,
    quotient_fssk_q1_readout_pallas,
)
from tensordev._wordwise.pallas_ordinary import (
    PallasOrdinaryResourceError,
    PallasOrdinaryUnavailableError,
    PallasOrdinaryUnsupportedError,
)
from tensordev._wordwise.pallas_quotient import PallasQuotientPlanError


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


def _canonical_executor_arrays(
    call: PreparedFSSKQ1Call,
) -> tuple[Any, Any, Any, Any]:
    """Move the prepared time axes behind the flattened batch axis."""
    return (
        jnp.moveaxis(call.y_time, 0, 1),
        jnp.moveaxis(call.E_time, 0, 1),
        jnp.moveaxis(call.psi_time, 0, 1),
        jnp.moveaxis(call.phi_time, 0, 1),
    )


def _tensor_blocks(tensor: Any) -> tuple[Any, ...]:
    return tuple(tensor.blocks) if hasattr(tensor, "blocks") else tuple(tensor)


def _retain_block_axis(standard: Any, *, plan: WordwiseLayoutPlan) -> Any:
    """Return canonical blocks with shape ``(n_blocks, B, ..., width)``."""
    output = []
    for block in _tensor_blocks(standard):
        if block.ndim == 5:
            output.append(block[None])
        elif block.ndim == 6:
            output.append(jnp.moveaxis(block, 1, 0))
        else:
            raise ValueError(
                "a scalar-FSSK executor block must have rank five or six, "
                f"got shape {block.shape}."
            )
    return plan.assemble_first_on(output)


def _execution_route(
    call: PreparedFSSKQ1Call,
    y: Any,
    *,
    plan: WordwiseLayoutPlan | None,
    readout: bool,
    interpret: bool,
    tile_words: int,
    tile_prime_words: int,
):
    plan = (
        build_layout_plan(
            call.core,
            call.trunc,
            alphabet_dim=int(y.shape[-1]),
        )
        if plan is None
        else plan
    )
    if plan.partially_symmetrized:
        executor = (
            quotient_fssk_q1_readout_pallas
            if readout
            else quotient_fssk_q1_pallas
        )
        tile_kwargs = {"tile_prime_words": tile_prime_words}
    else:
        executor = (
            ordered_fssk_q1_readout_pallas
            if readout
            else ordered_fssk_q1_pallas
        )
        tile_kwargs = {"tile_words": tile_words}
    kwargs = {
        "plan": plan,
        "initial_state": call.initial_standard,
        "interpret": interpret,
        **tile_kwargs,
    }
    if not readout:
        kwargs.update(
            block_size=call.block_size,
            accumulate=call.accumulate,
        )
    return plan, executor, kwargs


def run_fssk_q1_wordwise(
    call: PreparedFSSKQ1Call,
    *,
    plan: WordwiseLayoutPlan | None = None,
    interpret: bool = False,
    tile_words: int = 32,
    tile_prime_words: int = 8,
) -> Any:
    """Return canonical standard-coordinate states for one prepared call.

    Every returned first-on block has shape
    ``(n_blocks, B, 1, 1, R, width)``.  The initial state participates in the
    recursion, but prepending it to the output, squeezing a sole terminal
    block, restoring public axes, and converting coordinates belong to the
    shared SSS finalizer.
    """
    y, E, psi, phi = _canonical_executor_arrays(call)
    plan, executor, executor_kwargs = _execution_route(
        call,
        y,
        plan=plan,
        readout=False,
        interpret=interpret,
        tile_words=tile_words,
        tile_prime_words=tile_prime_words,
    )
    standard = executor(y, E, psi, phi, **executor_kwargs)
    return _retain_block_axis(standard, plan=plan)


def run_fssk_q1_wordwise_readout(
    call: PreparedFSSKQ1Call,
    weights: Any,
    *,
    plan: WordwiseLayoutPlan | None = None,
    interpret: bool = False,
    tile_words: int = 32,
    tile_prime_words: int = 8,
) -> Any:
    """Emit a complete terminal standard-coordinate q=1 signature.

    ``weights`` has canonical shape ``(1 or B, R)``.  The result includes
    the scalar unit and retains the flattened batch axis, but has no block
    axis.  Calls requesting block emission or their starting state are left
    to the portable state-then-readout path.
    """
    if call.block_size is not None:
        raise PallasOrdinaryUnsupportedError(
            "direct scalar-FSSK readout requires block_size=None."
        )
    if call.output_starting_state:
        raise PallasOrdinaryUnsupportedError(
            "direct scalar-FSSK readout cannot emit the starting state."
        )
    y, E, psi, phi = _canonical_executor_arrays(call)
    _, executor, executor_kwargs = _execution_route(
        call,
        y,
        plan=plan,
        readout=True,
        interpret=interpret,
        tile_words=tile_words,
        tile_prime_words=tile_prime_words,
    )
    return executor(y, E, psi, phi, weights, **executor_kwargs)


_FALLBACK_ERRORS = (
    PlanResourceError,
    PallasFSSKPlanError,
    PallasQuotientPlanError,
    PallasOrdinaryUnavailableError,
    PallasOrdinaryUnsupportedError,
    PallasOrdinaryResourceError,
)


def try_fssk_q1_wordwise(
    call: PreparedFSSKQ1Call,
    **kwargs: Any,
) -> Any | None:
    """Return ``None`` only for bounded wordwise plan or backend failures."""
    try:
        return run_fssk_q1_wordwise(call, **kwargs)
    except _FALLBACK_ERRORS:
        return None


def try_fssk_q1_wordwise_readout(
    call: PreparedFSSKQ1Call,
    weights: Any,
    **kwargs: Any,
) -> Any | None:
    """Return ``None`` when direct terminal readout cannot be executed."""
    try:
        return run_fssk_q1_wordwise_readout(call, weights, **kwargs)
    except _FALLBACK_ERRORS:
        return None


__all__ = [
    "PreparedFSSKQ1Call",
    "run_fssk_q1_wordwise",
    "run_fssk_q1_wordwise_readout",
    "try_fssk_q1_wordwise",
    "try_fssk_q1_wordwise_readout",
]
