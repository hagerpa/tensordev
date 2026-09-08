"""Quotient-native Pallas executor for one partially symmetrized block.

The kernel consumes the packed destination-oriented prefix graph owned by a
``WordwiseBlockPlan``.  One program owns one batch item, one multiset rank,
and a tile of dense prime words.  Prefix states are private to that program,
so predecessor actions need neither output scatter nor atomics.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tensordev._wordwise.layout import WordwiseBlockPlan
from tensordev._wordwise.pallas_ordinary import (
    PallasOrdinaryError,
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
from tensordev._wordwise.dispatch import _colocate_array, _concrete_single_device


Array = jax.Array

_INT32_MAX = int(np.iinfo(np.int32).max)


class PallasQuotientPlanError(PallasOrdinaryError):
    """Raised when packed quotient metadata violates the kernel contract."""


def _validate_packed_plan(
        block_plan: WordwiseBlockPlan,
        *,
        d_prime: int,
        alphabet_dim: int,
) -> tuple[int, int, int, int, int, int]:
    """Return static graph sizes after allocation-free consistency checks."""
    if not isinstance(block_plan, WordwiseBlockPlan):
        raise TypeError("block_plan must be a WordwiseBlockPlan.")
    graph = block_plan.prefix_plan
    if graph is None:
        raise PallasQuotientPlanError(
            "block_plan must contain a packed prefix graph."
        )
    if not isinstance(block_plan.grade, tuple) or len(block_plan.grade) != 2:
        raise PallasQuotientPlanError(
            "a quotient block must have a bidegree grade."
        )
    n, m = map(int, block_plan.grade)
    if graph.grade != (n, m):
        raise PallasQuotientPlanError(
            "block and prefix-graph grades disagree."
        )
    if block_plan.total_degree != n + m:
        raise PallasQuotientPlanError(
            "block total degree and bidegree disagree."
        )

    d_prime = _require_integer(d_prime, name="d_prime", minimum=1)
    if d_prime >= alphabet_dim:
        raise ValueError(
            "d_prime must be smaller than the full bidegree alphabet."
        )
    dense_prime_width = d_prime**n
    if block_plan.dense_prime_width != dense_prime_width:
        raise PallasQuotientPlanError(
            "block dense-prime width is inconsistent with d_prime."
        )
    graph_count = graph.graph_count
    if graph_count <= 0:
        raise PallasQuotientPlanError(
            "a quotient block must contain at least one graph."
        )
    if block_plan.width != graph_count * dense_prime_width:
        raise PallasQuotientPlanError(
            "block width is inconsistent with its packed graphs."
        )

    arrays = (
        graph.graph_node_offsets,
        graph.graph_edge_offsets,
        graph.terminal_indices,
        graph.node_edge_offsets,
        graph.node_prime_degrees,
        graph.node_doubleprime_degrees,
        graph.node_ranks,
        graph.predecessor_indices,
        graph.destination_indices,
        graph.letter_codes,
        graph.coefficients,
    )
    if any(array.dtype != np.dtype(np.int32) for array in arrays):
        raise PallasOrdinaryUnsupportedError(
            "packed prefix metadata must use signed int32 arrays."
        )
    if any(array.ndim != 1 for array in arrays):
        raise PallasQuotientPlanError(
            "packed prefix metadata arrays must be one-dimensional."
        )
    if graph.graph_node_offsets.shape != (graph_count + 1,):
        raise PallasQuotientPlanError(
            "graph_node_offsets has an inconsistent shape."
        )
    if graph.graph_edge_offsets.shape != (graph_count + 1,):
        raise PallasQuotientPlanError(
            "graph_edge_offsets has an inconsistent shape."
        )
    if graph.node_edge_offsets.shape != (graph.node_count + graph_count,):
        raise PallasQuotientPlanError(
            "node_edge_offsets has an inconsistent shape."
        )
    if graph.graph_node_offsets[0] != 0 or graph.graph_edge_offsets[0] != 0:
        raise PallasQuotientPlanError(
            "packed graph offsets must start at zero."
        )
    if graph.graph_node_offsets[-1] != graph.node_count:
        raise PallasQuotientPlanError(
            "packed node offsets have an inconsistent endpoint."
        )
    if graph.graph_edge_offsets[-1] != graph.edge_count:
        raise PallasQuotientPlanError(
            "packed edge offsets have an inconsistent endpoint."
        )
    if graph.node_count > _INT32_MAX or graph.edge_count > _INT32_MAX:
        raise PallasOrdinaryResourceError(
            "packed graph metadata exceeds signed int32 indexing."
        )

    node_counts = np.diff(graph.graph_node_offsets)
    if np.any(node_counts <= 0):
        raise PallasQuotientPlanError(
            "every packed prefix graph must contain a node."
        )
    maximum_node_count = int(node_counts.max(initial=0))
    # Triton vector lanes require power-of-two extents.  The extra lanes are
    # masked by ``valid_nodes`` and carry no additional plan metadata.
    node_capacity = 1 << (maximum_node_count - 1).bit_length()
    incoming_counts = np.diff(graph.node_edge_offsets)
    if np.any(incoming_counts < 0):
        # Negative differences occur only at graph boundaries, where each
        # local CSR row restarts at zero.
        boundary_indices = graph.graph_node_offsets[1:-1] + np.arange(
            1, graph_count, dtype=np.int32
        )
        invalid = incoming_counts < 0
        allowed = np.zeros(incoming_counts.shape, dtype=bool)
        allowed[boundary_indices - 1] = True
        if np.any(invalid & ~allowed):
            raise PallasQuotientPlanError(
                "node_edge_offsets is not a valid local CSR table."
            )
    maximum_in_degree = int(incoming_counts.max(initial=0))
    if maximum_in_degree > alphabet_dim - d_prime + 1:
        raise PallasQuotientPlanError(
            "a prefix node has too many incoming predecessor edges."
        )
    if not np.all(graph.coefficients == 1):
        raise PallasOrdinaryUnsupportedError(
            "the quotient kernel requires unit predecessor weights."
        )
    return (
        n,
        m,
        graph_count,
        dense_prime_width,
        node_capacity,
        maximum_in_degree,
    )


def _check_resources(
        *,
        flat_batch: int,
        steps: int,
        block_count: int,
        graph_count: int,
        node_capacity: int,
        maximum_in_degree: int,
        prime_degree: int,
        tile_prime_words: int,
        padded_prime_width: int,
        itemsize: int,
) -> None:
    for name, value in (
        ("flat batch size", flat_batch),
        ("step count", steps),
        ("block count", block_count),
        ("graph count", graph_count),
        ("node capacity", node_capacity),
        ("padded prime-word count", padded_prime_width),
    ):
        if value > _INT32_MAX:
            raise PallasOrdinaryResourceError(
                f"{name} {value} exceeds signed int32 indexing."
            )

    # ``old``, the Horner-stage state, and its destination update can be live
    # together.  Prime letters and one edge action are the remaining dominant
    # vectors.  This deliberately overestimates rather than risking a backend
    # allocation failure.
    local_bytes = (
        4 * node_capacity * tile_prime_words * itemsize
        + prime_degree * tile_prime_words * np.dtype(np.int32).itemsize
        + 2 * tile_prime_words * itemsize
        + node_capacity
        * (3 + 3 * maximum_in_degree)
        * np.dtype(np.int32).itemsize
    )
    if local_bytes > _MAX_LOCAL_BYTES:
        raise PallasOrdinaryResourceError(
            "quotient prime-word tile requires an estimated "
            f"{local_bytes} local bytes, exceeding the "
            f"{_MAX_LOCAL_BYTES}-byte guard; reduce tile_prime_words."
        )

    output_elements = (
        flat_batch * block_count * graph_count * padded_prime_width
    )
    if output_elements > _MAX_OUTPUT_ELEMENTS:
        raise PallasOrdinaryResourceError(
            "padded quotient output contains "
            f"{output_elements} elements, exceeding the int32 execution guard."
        )


def _prime_letters(
        word_positions,
        *,
        d_prime: int,
        prime_degree: int,
) -> tuple[Array, ...]:
    return tuple(
        (word_positions // (d_prime**power)) % d_prime
        for power in range(prime_degree - 1, -1, -1)
    )


def _resolve_letter(
        code,
        prime_letters: tuple[Array, ...],
        *,
        tile_prime_words: int,
) -> Array:
    letter = jnp.broadcast_to(
        code[..., None], code.shape + (tile_prime_words,)
    )
    for prime_index, prime_letter in enumerate(prime_letters):
        letter = jnp.where(
            code[..., None] == -(prime_index + 1),
            prime_letter,
            letter,
        )
    return letter


@lru_cache(maxsize=128)
def _quotient_call(
        shape: tuple[int, int, int],
        dtype: np.dtype,
        total_degree: int,
        prime_degree: int,
        graph_count: int,
        node_capacity: int,
        maximum_in_degree: int,
        dense_prime_width: int,
        padded_prime_width: int,
        tile_prime_words: int,
        d_prime: int,
        emission_block_size: int | None,
        interpret: bool,
):
    pl, pltriton = _load_pallas()
    flat_batch, steps, _alphabet_dim = shape
    block_count = (
        1 if emission_block_size is None else steps // emission_block_size
    )

    def kernel(
            increments_ref,
            graph_node_offsets_ref,
            graph_edge_offsets_ref,
            terminal_indices_ref,
            node_edge_offsets_ref,
            node_prime_degrees_ref,
            node_doubleprime_degrees_ref,
            predecessor_indices_ref,
            letter_codes_ref,
            output_ref,
    ):
        batch_index = pl.program_id(0)
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

        # Graph metadata is invariant across time and Horner stages.  Keep a
        # destination-oriented local view so the hot loop loads only path
        # increments and gathers the previous prefix stage.
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
        letter_code_slots = []
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
            letter_code_slots.append(
                pltriton.load(
                    letter_codes_ref.at[edge_indices],
                    mask=valid_edges,
                    other=0,
                )
            )
            valid_edge_slots.append(valid_edges)

        prefixes = jnp.zeros(
            (node_capacity, tile_prime_words), dtype=dtype
        )
        prefixes = prefixes.at[0].set(1)

        def time_step(time_index, old):
            previous = jnp.zeros_like(old).at[0].set(old[0])
            for denominator in range(total_degree, 0, -1):
                active_degree = total_degree - denominator + 1
                active_nodes = valid_nodes & (
                    node_total_degrees <= active_degree
                )
                action = jnp.zeros_like(previous)
                for predecessor, code, valid_edges in zip(
                    predecessor_slots,
                    letter_code_slots,
                    valid_edge_slots,
                ):
                    letters = _resolve_letter(
                        code,
                        prime_letters,
                        tile_prime_words=tile_prime_words,
                    )
                    edge_mask = valid_edges[:, None] & valid_words[None, :]
                    increments = pltriton.load(
                        increments_ref.at[
                            batch_index,
                            time_index,
                            letters,
                        ],
                        mask=edge_mask,
                        other=0,
                    )
                    contribution = previous[predecessor, :] * increments
                    action = action + jnp.where(
                        valid_edges[:, None], contribution, 0
                    )
                previous = jnp.where(
                    active_nodes[:, None],
                    old + action / jnp.asarray(denominator, dtype=dtype),
                    0,
                )
            return previous

        if emission_block_size is None:
            prefixes = jax.lax.fori_loop(0, steps, time_step, prefixes)
            pltriton.store(
                output_ref.at[batch_index, graph_rank, word_positions],
                prefixes[terminal_index],
                mask=valid_words,
            )
            return

        def block_step(block_index, current):
            start = block_index * emission_block_size

            def local_time_step(local_index, block_current):
                return time_step(start + local_index, block_current)

            current = jax.lax.fori_loop(
                0,
                emission_block_size,
                local_time_step,
                current,
            )
            pltriton.store(
                output_ref.at[
                    batch_index,
                    block_index,
                    graph_rank,
                    word_positions,
                ],
                current[terminal_index],
                mask=valid_words,
            )
            return current

        jax.lax.fori_loop(0, block_count, block_step, prefixes)

    output_shape = (
        (flat_batch, graph_count, padded_prime_width)
        if emission_block_size is None
        else (flat_batch, block_count, graph_count, padded_prime_width)
    )
    mode = "terminal" if emission_block_size is None else "accumulated"
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
        name=f"tensordev_quotient_ordinary_{mode}",
    )


def quotient_ordinary_pallas(
        increments: Array,
        *,
        block_plan: WordwiseBlockPlan,
        d_prime: int,
        block_size: int | None = None,
        accumulate: bool = True,
        tile_prime_words: int = 16,
        interpret: bool = False,
) -> Array:
    """Compute one partially symmetrized bidegree block with Pallas.

    ``increments`` must have canonical shape ``(flat_batch, steps,
    alphabet)``.  The output is rank-major, with the dense prime word as the
    inner coordinate, exactly matching ``BigradedSpec`` block order.  A single
    emitted block has shape ``(flat_batch, block_plan.width)``; several blocks
    add the block axis between batch and coordinates.
    """
    increments = jnp.asarray(increments)
    if increments.ndim != 3:
        raise ValueError(
            "increments must have canonical shape "
            f"(flat_batch, steps, alphabet), got {increments.shape}."
        )
    flat_batch, steps, alphabet_dim = map(int, increments.shape)
    if flat_batch <= 0:
        raise PallasOrdinaryUnsupportedError(
            "the Pallas executor requires a positive flat batch size."
        )
    if steps <= 0:
        raise PallasOrdinaryUnsupportedError(
            "the Pallas executor requires at least one increment."
        )
    if alphabet_dim <= 1:
        raise PallasOrdinaryUnsupportedError(
            "a bidegree alphabet must contain both nonempty parts."
        )
    dtype = np.dtype(increments.dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise PallasOrdinaryUnsupportedError(
            "increments must have dtype float32 or float64, got "
            f"{dtype}."
        )
    if not isinstance(accumulate, bool):
        raise TypeError("accumulate must be a bool.")
    if not isinstance(interpret, bool):
        raise TypeError("interpret must be a bool.")
    tile_prime_words = _require_integer(
        tile_prime_words, name="tile_prime_words", minimum=1
    )
    if (
        tile_prime_words > _MAX_TILE_WORDS
        or tile_prime_words & (tile_prime_words - 1)
    ):
        raise PallasOrdinaryResourceError(
            "tile_prime_words must be a power of two no greater than "
            f"{_MAX_TILE_WORDS}, got {tile_prime_words}."
        )

    (
        n,
        m,
        graph_count,
        dense_prime_width,
        node_capacity,
        maximum_in_degree,
    ) = (
        _validate_packed_plan(
            block_plan,
            d_prime=d_prime,
            alphabet_dim=alphabet_dim,
        )
    )
    total_degree = n + m
    if total_degree > _MAX_DEGREE:
        raise PallasOrdinaryResourceError(
            f"degree {total_degree} exceeds the {_MAX_DEGREE}-degree kernel guard."
        )
    graph = block_plan.prefix_plan
    tile_prime_words = _effective_tile_words(
        dense_prime_width, tile_prime_words
    )
    padded_prime_width = _padded_word_count(
        dense_prime_width, tile_prime_words
    )
    normalized_block_size, block_count = _normalize_block_size(
        block_size, steps=steps
    )
    _check_resources(
        flat_batch=flat_batch,
        steps=steps,
        block_count=block_count,
        graph_count=graph_count,
        node_capacity=node_capacity,
        maximum_in_degree=maximum_in_degree,
        prime_degree=n,
        tile_prime_words=tile_prime_words,
        padded_prime_width=padded_prime_width,
        itemsize=dtype.itemsize,
    )

    if total_degree == 0:
        shape = (
            (flat_batch, 1)
            if block_count == 1
            else (flat_batch, block_count, 1)
        )
        return jnp.ones(
            shape,
            dtype=increments.dtype,
            device=_concrete_single_device(increments),
        )

    operands = (
        _colocate_array(graph.graph_node_offsets, increments),
        _colocate_array(graph.graph_edge_offsets, increments),
        _colocate_array(graph.terminal_indices, increments),
        _colocate_array(graph.node_edge_offsets, increments),
        _colocate_array(graph.node_prime_degrees, increments),
        _colocate_array(graph.node_doubleprime_degrees, increments),
        _colocate_array(graph.predecessor_indices, increments),
        _colocate_array(graph.letter_codes, increments),
    )

    shape = (flat_batch, steps, alphabet_dim)
    if block_count == 1:
        call = _quotient_call(
            shape,
            dtype,
            total_degree,
            n,
            graph_count,
            node_capacity,
            maximum_in_degree,
            dense_prime_width,
            padded_prime_width,
            tile_prime_words,
            int(d_prime),
            None,
            interpret,
        )
        result = call(increments, *operands)
        return result[..., :dense_prime_width].reshape(
            flat_batch, block_plan.width
        )

    if not accumulate:
        blocks = increments.reshape(
            flat_batch * block_count,
            normalized_block_size,
            alphabet_dim,
        )
        independent_shape = (
            flat_batch * block_count,
            normalized_block_size,
            alphabet_dim,
        )
        call = _quotient_call(
            independent_shape,
            dtype,
            total_degree,
            n,
            graph_count,
            node_capacity,
            maximum_in_degree,
            dense_prime_width,
            padded_prime_width,
            tile_prime_words,
            int(d_prime),
            None,
            interpret,
        )
        result = call(blocks, *operands)
        return result[..., :dense_prime_width].reshape(
            flat_batch,
            block_count,
            block_plan.width,
        )

    call = _quotient_call(
        shape,
        dtype,
        total_degree,
        n,
        graph_count,
        node_capacity,
        maximum_in_degree,
        dense_prime_width,
        padded_prime_width,
        tile_prime_words,
        int(d_prime),
        normalized_block_size,
        interpret,
    )
    result = call(increments, *operands)
    return result[..., :dense_prime_width].reshape(
        flat_batch,
        block_count,
        block_plan.width,
    )


def clear_pallas_quotient_cache() -> None:
    """Clear the bounded Pallas-call factory cache."""
    _quotient_call.cache_clear()


__all__ = [
    "PallasQuotientPlanError",
    "clear_pallas_quotient_cache",
    "quotient_ordinary_pallas",
]
