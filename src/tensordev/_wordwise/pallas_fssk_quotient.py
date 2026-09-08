"""Quotient-native scalar-FSSK Pallas execution in standard coordinates."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tensordev._wordwise.dispatch import _colocate_array
from tensordev._wordwise.layout import WordwiseLayoutPlan
from tensordev._wordwise.pallas_fssk_q1 import (
    PallasFSSKPlanError,
    _assemble_readout_signature,
    _canonicalize_inputs,
    _canonicalize_readout_weights,
    _fssk_state_extent,
    _load_matrix,
    _pad_fssk_state_operands,
    _pack_initial_state,
    _right_matrix,
)
from tensordev._wordwise.pallas_ordinary import (
    PallasOrdinaryResourceError,
    PallasOrdinaryUnsupportedError,
    _MAX_DEGREE,
    _MAX_LOCAL_BYTES,
    _MAX_OUTPUT_ELEMENTS,
    _MAX_TILE_WORDS,
    _compiler_warps,
    _effective_tile_words,
    _load_pallas,
    _normalize_block_size,
    _padded_word_count,
    _require_integer,
)
from tensordev._wordwise.pallas_quotient import (
    _prime_letters,
    _resolve_letter,
    _validate_packed_plan,
)
from tensordev.core.utils.precompute import _readonly


Array = jax.Array

_INT32_MAX = int(np.iinfo(np.int32).max)
_MAX_STATE_DIM = 128
_EXECUTION_PLAN_CACHE_SIZE = 32
_EXECUTION_PLAN_CACHE_BYTES = 64 * 1024**2


@dataclass(frozen=True, slots=True)
class _BlockResources:
    block_index: int
    total_degree: int
    output_width: int
    prime_degree: int
    graph_count: int
    dense_prime_width: int
    node_capacity: int
    maximum_in_degree: int


@dataclass(frozen=True, slots=True, eq=False)
class _QuotientFSSKExecutionPlan:
    fingerprint: tuple[object, ...]
    grade_offsets: np.ndarray
    blocks: tuple[_BlockResources, ...]

    def memory_bytes(self) -> int:
        return int(self.grade_offsets.nbytes)


_EXECUTION_PLAN_CACHE: OrderedDict[
    tuple[object, ...], _QuotientFSSKExecutionPlan
] = OrderedDict()
_EXECUTION_PLAN_CACHE_LOCK = RLock()
_EXECUTION_PLAN_CACHE_BYTES_USED = 0


def _check_layout(plan: WordwiseLayoutPlan) -> None:
    if plan.grading != "bidegree" or not plan.partially_symmetrized:
        raise PallasFSSKPlanError(
            "the quotient scalar-FSSK executor requires a partially "
            "symmetrized bidegree plan."
        )
    if not plan.blocks or plan.blocks[0].grade != (0, 0):
        raise PallasFSSKPlanError(
            "the quotient layout must begin with its scalar block."
        )
    if plan.blocks[0].flat_offset != 0 or plan.blocks[0].width != 1:
        raise PallasFSSKPlanError(
            "the scalar block must occupy flat coordinate zero."
        )
    if len(plan.blocks) == 1:
        raise PallasFSSKPlanError(
            "scalar-FSSK execution requires a positive truncation."
        )


def _positive_grade_offsets(plan: WordwiseLayoutPlan) -> np.ndarray:
    """Return compact first-on offsets indexed by the bidegree rectangle."""
    n_max, m_max = plan.truncation
    offsets = np.full((n_max + 1, m_max + 1), -1, dtype=np.int32)
    for block in plan.blocks:
        if block.total_degree == 0:
            continue
        offset = block.flat_offset - 1
        if offset > _INT32_MAX:
            raise PallasOrdinaryResourceError(
                "quotient scalar-FSSK state exceeds signed int32 indexing."
            )
        offsets[block.grade] = offset
    if np.any(offsets.ravel()[1:] < 0):
        # ``ravel()[1:]`` is not a statement about grade order; it is only a
        # cheap completeness check after the scalar entry was set to -1.
        missing = [
            grade
            for grade in (
                (n, m)
                for n in range(n_max + 1)
                for m in range(m_max + 1)
            )
            if grade != (0, 0) and offsets[grade] < 0
        ]
        if missing:
            raise PallasFSSKPlanError(
                f"quotient layout is missing active grades {missing!r}."
            )
    return offsets


def _build_execution_plan(
        plan: WordwiseLayoutPlan,
) -> _QuotientFSSKExecutionPlan:
    """Validate packed graphs once and retain only compact derived metadata."""
    global _EXECUTION_PLAN_CACHE_BYTES_USED

    _check_layout(plan)
    key = plan.fingerprint
    with _EXECUTION_PLAN_CACHE_LOCK:
        cached = _EXECUTION_PLAN_CACHE.get(key)
        if cached is not None:
            _EXECUTION_PLAN_CACHE.move_to_end(key)
            return cached

    resources = []
    for block_index, block in enumerate(plan.blocks[1:], start=1):
        (
            prime_degree,
            _doubleprime_degree,
            graph_count,
            dense_prime_width,
            node_capacity,
            maximum_in_degree,
        ) = _validate_packed_plan(
            block,
            d_prime=plan.dims[0],
            alphabet_dim=plan.alphabet_dim,
        )
        resources.append(
            _BlockResources(
                block_index=block_index,
                total_degree=block.total_degree,
                output_width=block.width,
                prime_degree=prime_degree,
                graph_count=graph_count,
                dense_prime_width=dense_prime_width,
                node_capacity=node_capacity,
                maximum_in_degree=maximum_in_degree,
            )
        )

    result = _QuotientFSSKExecutionPlan(
        fingerprint=key,
        grade_offsets=_readonly(_positive_grade_offsets(plan).reshape(-1)),
        blocks=tuple(resources),
    )
    with _EXECUTION_PLAN_CACHE_LOCK:
        existing = _EXECUTION_PLAN_CACHE.get(key)
        if existing is not None:
            _EXECUTION_PLAN_CACHE.move_to_end(key)
            return existing
        _EXECUTION_PLAN_CACHE[key] = result
        _EXECUTION_PLAN_CACHE.move_to_end(key)
        _EXECUTION_PLAN_CACHE_BYTES_USED += result.memory_bytes()
        while (
            len(_EXECUTION_PLAN_CACHE) > _EXECUTION_PLAN_CACHE_SIZE
            or _EXECUTION_PLAN_CACHE_BYTES_USED > _EXECUTION_PLAN_CACHE_BYTES
        ):
            _, evicted = _EXECUTION_PLAN_CACHE.popitem(last=False)
            _EXECUTION_PLAN_CACHE_BYTES_USED -= evicted.memory_bytes()
    return result


def _check_kernel_resources(
        *,
        flat_batch: int,
        steps: int,
        block_count: int,
        coefficient_steps: int,
        coefficient_order: int,
        state_dim: int,
        total_degree: int,
        graph_count: int,
        node_capacity: int,
        maximum_in_degree: int,
        prime_degree: int,
        tile_prime_words: int,
        padded_prime_width: int,
        itemsize: int,
        emit_readout: bool,
) -> None:
    for name, value in (
        ("flat batch size", flat_batch),
        ("step count", steps),
        ("block count", block_count),
        ("coefficient step count", coefficient_steps),
        ("coefficient order", coefficient_order),
        ("state dimension", state_dim),
        ("graph count", graph_count),
        ("node capacity", node_capacity),
        ("padded prime-word count", padded_prime_width),
    ):
        if value > _INT32_MAX:
            raise PallasOrdinaryResourceError(
                f"{name} {value} exceeds signed int32 indexing."
            )
    if state_dim > _MAX_STATE_DIM:
        raise PallasOrdinaryResourceError(
            f"state dimension {state_dim} exceeds the "
            f"{_MAX_STATE_DIM}-dimensional kernel guard."
        )
    if total_degree > _MAX_DEGREE:
        raise PallasOrdinaryResourceError(
            f"degree {total_degree} exceeds the "
            f"{_MAX_DEGREE}-degree kernel guard."
        )

    vector_elements = node_capacity * tile_prime_words * state_dim
    # E and the active phi/psi coefficients are loaded as tuples before the
    # degree loop.  Include their complete footprint in the pre-lowering
    # guard rather than assuming only two matrices remain live.
    local_bytes = (
        6 * vector_elements * itemsize
        + total_degree * state_dim * state_dim * itemsize
        + total_degree * state_dim * itemsize
        + (state_dim * itemsize if emit_readout else 0)
        + prime_degree * tile_prime_words * np.dtype(np.int32).itemsize
        + node_capacity
        * (5 + 3 * maximum_in_degree)
        * np.dtype(np.int32).itemsize
    )
    if local_bytes > _MAX_LOCAL_BYTES:
        raise PallasOrdinaryResourceError(
            "quotient scalar-FSSK tile requires an estimated "
            f"{local_bytes} local bytes, exceeding the "
            f"{_MAX_LOCAL_BYTES}-byte guard; reduce tile_prime_words."
        )
    output_state_width = 1 if emit_readout else state_dim
    output_elements = (
        flat_batch
        * block_count
        * graph_count
        * padded_prime_width
        * output_state_width
    )
    if output_elements > _MAX_OUTPUT_ELEMENTS:
        raise PallasOrdinaryResourceError(
            "quotient scalar-FSSK padded output contains "
            f"{output_elements} elements, exceeding the int32 execution guard."
        )


@lru_cache(maxsize=128)
def _quotient_fssk_call(
        shape: tuple[int, int, int],
        dtype: np.dtype,
        coefficient_steps: int,
        coefficient_singletons: tuple[bool, bool, bool],
        state_dim: int,
        total_degree: int,
        prime_degree: int,
        doubleprime_truncation: int,
        graph_count: int,
        node_capacity: int,
        maximum_in_degree: int,
        dense_prime_width: int,
        padded_prime_width: int,
        tile_prime_words: int,
        d_prime: int,
        emission_block_size: int | None,
        accumulate: bool,
        seed_singleton: bool,
        readout_batch: int,
        interpret: bool,
):
    pl, pltriton = _load_pallas()
    flat_batch, steps, _alphabet_dim = shape
    block_count = (
        1 if emission_block_size is None else steps // emission_block_size
    )
    if readout_batch and emission_block_size is not None:
        raise AssertionError("readout emission is terminal-only.")

    def run_kernel(
            y_ref,
            E_ref,
            psi_ref,
            phi_ref,
            initial_ref,
            weights_ref,
            grade_offsets_ref,
            graph_node_offsets_ref,
            graph_edge_offsets_ref,
            terminal_indices_ref,
            node_edge_offsets_ref,
            node_prime_degrees_ref,
            node_doubleprime_degrees_ref,
            node_ranks_ref,
            predecessor_indices_ref,
            letter_codes_ref,
            output_ref,
    ):
        batch_index = pl.program_id(0)
        E_batch_index = 0 if coefficient_singletons[0] else batch_index
        psi_batch_index = 0 if coefficient_singletons[1] else batch_index
        phi_batch_index = 0 if coefficient_singletons[2] else batch_index
        initial_batch_index = 0 if seed_singleton else batch_index
        graph_rank = pl.program_id(1)
        word_positions = (
            pl.program_id(2) * tile_prime_words
            + jnp.arange(tile_prime_words, dtype=jnp.int32)
        )
        valid_words = word_positions < dense_prime_width
        prime_letters = _prime_letters(
            word_positions,
            d_prime=d_prime,
            prime_degree=prime_degree,
        )
        node_start = pltriton.load(graph_node_offsets_ref.at[graph_rank])
        node_stop = pltriton.load(graph_node_offsets_ref.at[graph_rank + 1])
        node_count = node_stop - node_start
        graph_edge_start = pltriton.load(
            graph_edge_offsets_ref.at[graph_rank]
        )
        csr_start = node_start + graph_rank
        terminal_index = pltriton.load(terminal_indices_ref.at[graph_rank])

        node_positions = jnp.arange(node_capacity, dtype=jnp.int32)
        valid_nodes = node_positions < node_count
        packed_nodes = node_start + node_positions
        node_prime_degrees = pltriton.load(
            node_prime_degrees_ref.at[packed_nodes],
            mask=valid_nodes,
            other=0,
        )
        node_doubleprime_degrees = pltriton.load(
            node_doubleprime_degrees_ref.at[packed_nodes],
            mask=valid_nodes,
            other=0,
        )
        node_total_degrees = node_prime_degrees + node_doubleprime_degrees
        node_ranks = pltriton.load(
            node_ranks_ref.at[packed_nodes],
            mask=valid_nodes,
            other=0,
        )
        grade_indices = (
            node_prime_degrees * (doubleprime_truncation + 1)
            + node_doubleprime_degrees
        )
        grade_offsets = pltriton.load(
            grade_offsets_ref.at[grade_indices],
            mask=valid_nodes,
            other=-1,
        )

        prefix_codes = jnp.zeros(
            (node_capacity, tile_prime_words), dtype=jnp.int32
        )
        prime_widths = jnp.ones((node_capacity,), dtype=jnp.int32)
        prefix_code = jnp.zeros(
            (tile_prime_words,), dtype=jnp.int32
        )
        prime_width = 1
        for prefix_degree, letter in enumerate(prime_letters, start=1):
            prefix_code = d_prime * prefix_code + letter
            prime_width *= d_prime
            is_degree = node_prime_degrees == prefix_degree
            prefix_codes = jnp.where(
                is_degree[:, None], prefix_code[None, :], prefix_codes
            )
            prime_widths = jnp.where(
                is_degree, prime_width, prime_widths
            )
        compact_indices = (
            grade_offsets[:, None]
            + node_ranks[:, None] * prime_widths[:, None]
            + prefix_codes
        )
        state_positions = jnp.arange(state_dim, dtype=jnp.int32)
        initial_mask = (
            valid_nodes[:, None, None]
            & valid_words[None, :, None]
            & (node_total_degrees[:, None, None] > 0)
        )
        prefixes = pltriton.load(
            initial_ref.at[
                initial_batch_index,
                compact_indices[:, :, None],
                state_positions[None, None, :],
            ],
            mask=initial_mask,
            other=0,
        )
        prefixes = prefixes.at[0].set(0)

        local_edge_starts = pltriton.load(
            node_edge_offsets_ref.at[csr_start + node_positions],
            mask=valid_nodes,
            other=0,
        )
        local_edge_stops = pltriton.load(
            node_edge_offsets_ref.at[csr_start + node_positions + 1],
            mask=valid_nodes,
            other=0,
        )
        local_edge_counts = local_edge_stops - local_edge_starts
        predecessor_slots = []
        resolved_letter_slots = []
        valid_edge_slots = []
        for edge_slot in range(maximum_in_degree):
            valid_edges = valid_nodes & (edge_slot < local_edge_counts)
            edge_indices = graph_edge_start + local_edge_starts + edge_slot
            predecessor_slots.append(
                pltriton.load(
                    predecessor_indices_ref.at[edge_indices],
                    mask=valid_edges,
                    other=0,
                )
            )
            codes = pltriton.load(
                letter_codes_ref.at[edge_indices],
                mask=valid_edges,
                other=0,
            )
            resolved_letter_slots.append(
                _resolve_letter(
                    codes,
                    prime_letters,
                    tile_prime_words=tile_prime_words,
                )
            )
            valid_edge_slots.append(valid_edges)

        def right_action(values, time_index):
            action = jnp.zeros_like(values)
            for predecessor, letters, valid_edges in zip(
                predecessor_slots,
                resolved_letter_slots,
                valid_edge_slots,
            ):
                edge_mask = valid_edges[:, None] & valid_words[None, :]
                increments = pltriton.load(
                    y_ref.at[batch_index, time_index, letters],
                    mask=edge_mask,
                    other=0,
                )
                contribution = values[predecessor, :, :] * increments[..., None]
                action = action + jnp.where(
                    valid_edges[:, None, None], contribution, 0
                )
            return action

        def time_step(time_index, old):
            coefficient_time = 0 if coefficient_steps == 1 else time_index
            E_step = _load_matrix(
                E_ref,
                pltriton,
                batch_index=E_batch_index,
                time_index=coefficient_time,
                matrix_index=None,
                state_dim=state_dim,
            )
            psi_step = tuple(
                pltriton.load(
                    psi_ref.at[
                        psi_batch_index,
                        coefficient_time,
                        degree,
                        state_positions,
                    ]
                )
                for degree in range(total_degree)
            )
            phi_step = tuple(
                _load_matrix(
                    phi_ref,
                    pltriton,
                    batch_index=phi_batch_index,
                    time_index=coefficient_time,
                    matrix_index=order,
                    state_dim=state_dim,
                )
                for order in range(max(total_degree - 1, 0))
            )
            old_E = _right_matrix(old, E_step)
            new = old
            for target_degree in range(total_degree, 0, -1):
                horner = jnp.zeros_like(old)
                root_value = jnp.broadcast_to(
                    psi_step[target_degree - 1],
                    (tile_prime_words, state_dim),
                )
                horner = horner.at[0].set(root_value)
                for degree in range(1, target_degree):
                    incoming = right_action(horner, time_index)
                    memory = _right_matrix(
                        old,
                        phi_step[target_degree - 1 - degree],
                    )
                    horner = jnp.where(
                        (node_total_degrees == degree)[:, None, None],
                        incoming + memory,
                        0,
                    )
                incoming = right_action(horner, time_index)
                updated = old_E + incoming
                new = jnp.where(
                    (node_total_degrees == target_degree)[:, None, None],
                    updated,
                    new,
                )
            return new

        def store(block_index, state):
            terminal = state[terminal_index]
            if readout_batch:
                weights_batch_index = 0 if readout_batch == 1 else batch_index
                weights = pltriton.load(
                    weights_ref.at[weights_batch_index, state_positions]
                )
                values = jnp.sum(terminal * weights[None, :], axis=-1)
            else:
                values = jnp.moveaxis(terminal, -1, 0)
            if emission_block_size is None:
                if readout_batch:
                    pltriton.store(
                        output_ref.at[
                            batch_index,
                            graph_rank,
                            word_positions,
                        ],
                        values,
                        mask=valid_words,
                    )
                else:
                    pltriton.store(
                        output_ref.at[
                            batch_index,
                            graph_rank,
                            state_positions[:, None],
                            word_positions[None, :],
                        ],
                        values,
                        mask=valid_words[None, :],
                    )
            else:
                pltriton.store(
                    output_ref.at[
                        batch_index,
                        block_index,
                        graph_rank,
                        state_positions[:, None],
                        word_positions[None, :],
                    ],
                    values,
                    mask=valid_words[None, :],
                )

        if emission_block_size is None:
            terminal = jax.lax.fori_loop(0, steps, time_step, prefixes)
            store(0, terminal)
            return

        def block_step(block_index, current):
            start = block_index * emission_block_size
            block_initial = current if accumulate else prefixes

            def local_time_step(local_index, state):
                return time_step(start + local_index, state)

            terminal = jax.lax.fori_loop(
                0,
                emission_block_size,
                local_time_step,
                block_initial,
            )
            store(block_index, terminal)
            return terminal if accumulate else current

        jax.lax.fori_loop(0, block_count, block_step, prefixes)

    if readout_batch:
        kernel = run_kernel
        output_shape = (flat_batch, graph_count, padded_prime_width)
    else:
        def kernel(
                y_ref,
                E_ref,
                psi_ref,
                phi_ref,
                initial_ref,
                grade_offsets_ref,
                graph_node_offsets_ref,
                graph_edge_offsets_ref,
                terminal_indices_ref,
                node_edge_offsets_ref,
                node_prime_degrees_ref,
                node_doubleprime_degrees_ref,
                node_ranks_ref,
                predecessor_indices_ref,
                letter_codes_ref,
                output_ref,
        ):
            return run_kernel(
                y_ref,
                E_ref,
                psi_ref,
                phi_ref,
                initial_ref,
                None,
                grade_offsets_ref,
                graph_node_offsets_ref,
                graph_edge_offsets_ref,
                terminal_indices_ref,
                node_edge_offsets_ref,
                node_prime_degrees_ref,
                node_doubleprime_degrees_ref,
                node_ranks_ref,
                predecessor_indices_ref,
                letter_codes_ref,
                output_ref,
            )

        output_shape = (
            (flat_batch, graph_count, state_dim, padded_prime_width)
            if emission_block_size is None
            else (
                flat_batch,
                block_count,
                graph_count,
                state_dim,
                padded_prime_width,
            )
        )
    mode = (
        "readout"
        if readout_batch
        else "terminal"
        if emission_block_size is None
        else "accumulated" if accumulate else "independent"
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(output_shape, dtype),
        grid=(
            flat_batch,
            graph_count,
            padded_prime_width // tile_prime_words,
        ),
        compiler_params=pltriton.CompilerParams(
            num_warps=_compiler_warps(tile_prime_words),
            num_stages=1,
        ),
        interpret=interpret,
        name=f"tensordev_quotient_fssk_q1_{mode}",
    )


def _restore_block(
        values: Array,
        *,
        resource: _BlockResources,
        state_dim: int,
        block_count: int,
        emit_readout: bool,
) -> Array:
    values = values[..., :resource.dense_prime_width]
    if emit_readout:
        return values.reshape(values.shape[0], resource.output_width)
    if block_count == 1:
        values = values[:, :, :state_dim, :]
        values = jnp.transpose(values, (0, 2, 1, 3)).reshape(
            values.shape[0],
            state_dim,
            resource.output_width,
        )
        return values[:, None, None, :, :]
    values = values[:, :, :, :state_dim, :]
    values = jnp.transpose(values, (0, 1, 3, 2, 4)).reshape(
        values.shape[0],
        block_count,
        state_dim,
        resource.output_width,
    )
    return values[:, :, None, None, :, :]


def _quotient_fssk_q1_execute(
        y: Any,
        E: Any,
        psi: Any,
        phi: Any,
        *,
        plan: WordwiseLayoutPlan,
        initial_state: Any = None,
        block_size: int | None = None,
        accumulate: bool = True,
        tile_prime_words: int = 8,
        interpret: bool = False,
        readout_weights: Any = None,
):
    """Execute the shared quotient scalar-FSSK recurrence with Pallas.

    Numerical arrays use the canonical ordered-FSSK contract: ``y`` is
    ``(flat_batch, steps, alphabet)`` and coefficient arrays begin with
    ``(batch, coefficient_steps)``.  The optional initial state is the native
    standard-coordinate first-on tensor; it is packed once into a compact
    ``(1 or flat_batch, positive_coordinates, R)`` operand.  Coefficient
    batches likewise remain one or ``flat_batch``.  No ordered tensor is
    constructed.
    """
    _check_layout(plan)
    canonical = _canonicalize_inputs(y, E, psi, phi)
    emit_readout = readout_weights is not None
    if canonical.alphabet_dim != plan.alphabet_dim:
        raise ValueError(
            f"y alphabet width {canonical.alphabet_dim} disagrees with plan "
            f"dimension {plan.alphabet_dim}."
        )
    maximum_degree = max(block.total_degree for block in plan.blocks)
    if maximum_degree > canonical.coefficient_order:
        raise ValueError(
            f"coefficients cover order {canonical.coefficient_order}, but the "
            f"layout requires order {maximum_degree}."
        )
    if not isinstance(accumulate, bool):
        raise TypeError("accumulate must be a bool.")
    if not isinstance(interpret, bool):
        raise TypeError("interpret must be a bool.")
    tile_prime_words = _require_integer(
        tile_prime_words,
        name="tile_prime_words",
        minimum=1,
    )
    if (
        tile_prime_words > _MAX_TILE_WORDS
        or tile_prime_words & (tile_prime_words - 1)
    ):
        raise PallasOrdinaryResourceError(
            "tile_prime_words must be a power of two no greater than "
            f"{_MAX_TILE_WORDS}, got {tile_prime_words}."
        )
    normalized_block_size, block_count = _normalize_block_size(
        block_size,
        steps=canonical.steps,
    )
    if emit_readout and block_count != 1:
        raise PallasOrdinaryUnsupportedError(
            "scalar-FSSK readout emission is terminal-only."
        )
    state_extent = _fssk_state_extent(canonical.state_dim)
    execution = _build_execution_plan(plan)

    resources = []
    for resource in execution.blocks:
        effective_tile = _effective_tile_words(
            resource.dense_prime_width,
            tile_prime_words,
        )
        padded_width = _padded_word_count(
            resource.dense_prime_width,
            effective_tile,
        )
        _check_kernel_resources(
            flat_batch=canonical.flat_batch,
            steps=canonical.steps,
            block_count=block_count,
            coefficient_steps=canonical.coefficient_steps,
            coefficient_order=canonical.coefficient_order,
            state_dim=state_extent,
            total_degree=resource.total_degree,
            graph_count=resource.graph_count,
            node_capacity=resource.node_capacity,
            maximum_in_degree=resource.maximum_in_degree,
            prime_degree=resource.prime_degree,
            tile_prime_words=effective_tile,
            padded_prime_width=padded_width,
            itemsize=np.dtype(canonical.y.dtype).itemsize,
            emit_readout=emit_readout,
        )
        resources.append((resource, effective_tile, padded_width))

    initial = _pack_initial_state(
        initial_state,
        plan=plan,
        flat_batch=canonical.flat_batch,
        state_dim=canonical.state_dim,
        dtype=canonical.y.dtype,
        reference=canonical.y,
        state_extent=state_extent,
    )
    E_operand, psi_operand, phi_operand, initial = _pad_fssk_state_operands(
        canonical,
        initial,
        state_extent=state_extent,
    )
    if emit_readout:
        weights_operand, readout_batch = _canonicalize_readout_weights(
            readout_weights,
            canonical,
            state_extent=state_extent,
        )
    else:
        weights_operand, readout_batch = None, 0
    state_operands = (
        canonical.y,
        E_operand,
        psi_operand,
        phi_operand,
        initial,
    )
    grade_offsets = _colocate_array(execution.grade_offsets, canonical.y)
    common_operands = (
        (*state_operands, weights_operand, grade_offsets)
        if emit_readout
        else (*state_operands, grade_offsets)
    )

    output_blocks = []
    for resource, effective_tile, padded_width in resources:
        block = plan.blocks[resource.block_index]
        graph = block.prefix_plan
        graph_operands = tuple(
            _colocate_array(value, canonical.y)
            for value in (
                graph.graph_node_offsets,
                graph.graph_edge_offsets,
                graph.terminal_indices,
                graph.node_edge_offsets,
                graph.node_prime_degrees,
                graph.node_doubleprime_degrees,
                graph.node_ranks,
                graph.predecessor_indices,
                graph.letter_codes,
            )
        )
        call = _quotient_fssk_call(
            (
                canonical.flat_batch,
                canonical.steps,
                canonical.alphabet_dim,
            ),
            np.dtype(canonical.y.dtype),
            canonical.coefficient_steps,
            tuple(batch == 1 for batch in canonical.coefficient_batches),
            state_extent,
            resource.total_degree,
            resource.prime_degree,
            plan.truncation[1],
            resource.graph_count,
            resource.node_capacity,
            resource.maximum_in_degree,
            resource.dense_prime_width,
            padded_width,
            effective_tile,
            plan.dims[0],
            None if block_count == 1 else normalized_block_size,
            True if block_count == 1 else accumulate,
            initial.shape[0] == 1,
            readout_batch,
            interpret,
        )
        values = call(*common_operands, *graph_operands)
        output_blocks.append(
            _restore_block(
                values,
                resource=resource,
                state_dim=canonical.state_dim,
                block_count=block_count,
                emit_readout=emit_readout,
            )
        )
    if emit_readout:
        return _assemble_readout_signature(plan, output_blocks, canonical)
    return plan.assemble_first_on(output_blocks)


def quotient_fssk_q1_pallas(
        y: Any,
        E: Any,
        psi: Any,
        phi: Any,
        *,
        plan: WordwiseLayoutPlan,
        initial_state: Any = None,
        block_size: int | None = None,
        accumulate: bool = True,
        tile_prime_words: int = 8,
        interpret: bool = False,
):
    """Evolve a complete partially symmetrized scalar-FSSK state."""
    return _quotient_fssk_q1_execute(
        y,
        E,
        psi,
        phi,
        plan=plan,
        initial_state=initial_state,
        block_size=block_size,
        accumulate=accumulate,
        tile_prime_words=tile_prime_words,
        interpret=interpret,
    )


def quotient_fssk_q1_readout_pallas(
        y: Any,
        E: Any,
        psi: Any,
        phi: Any,
        weights: Any,
        *,
        plan: WordwiseLayoutPlan,
        initial_state: Any = None,
        tile_prime_words: int = 8,
        interpret: bool = False,
):
    """Emit the terminal q=1 readout directly from the quotient recurrence."""
    return _quotient_fssk_q1_execute(
        y,
        E,
        psi,
        phi,
        plan=plan,
        initial_state=initial_state,
        tile_prime_words=tile_prime_words,
        interpret=interpret,
        readout_weights=weights,
    )


def clear_pallas_fssk_quotient_cache() -> None:
    """Clear the bounded host-plan and Pallas-call caches."""
    global _EXECUTION_PLAN_CACHE_BYTES_USED

    _quotient_fssk_call.cache_clear()
    with _EXECUTION_PLAN_CACHE_LOCK:
        _EXECUTION_PLAN_CACHE.clear()
        _EXECUTION_PLAN_CACHE_BYTES_USED = 0


__all__ = [
    "clear_pallas_fssk_quotient_cache",
    "quotient_fssk_q1_pallas",
    "quotient_fssk_q1_readout_pallas",
]
