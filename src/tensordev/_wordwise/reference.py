"""Small pure-JAX reference recurrences for wordwise implementations.

These routines provide independent numerical checks and favour directness
over throughput.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tensordev._wordwise.layout import WordwiseBlockPlan, WordwiseLayoutPlan
from tensordev._wordwise.plans import PrefixGraphView
from tensordev.core.bigraded.types import BigradedTensor


Array = jax.Array


def _decoded_words(codes, *, base: int, length: int) -> Array:
    codes = jnp.asarray(codes, dtype=jnp.int32)
    if length == 0:
        return jnp.empty(codes.shape + (0,), dtype=jnp.int32)
    divisors = jnp.asarray(
        [base**power for power in range(length - 1, -1, -1)],
        dtype=jnp.int32,
    )
    return (codes[..., None] // divisors) % base


def ordinary_word_reference(increments: Array, word: Array | Sequence[int]) -> Array:
    """Compute one ordered signature coordinate by its prefix recurrence.

    ``increments`` has shape ``batch + (steps, alphabet)``.  The word length
    is static under JAX transformations, while its letters may be dynamic.
    """

    increments = jnp.asarray(increments)
    word = jnp.asarray(word, dtype=jnp.int32)
    if increments.ndim < 2:
        raise ValueError("increments must have a step and alphabet axis.")
    if word.ndim != 1:
        raise ValueError(f"word must be one-dimensional, got shape {word.shape}.")
    degree = word.shape[0]
    batch_shape = increments.shape[:-2]
    initial = jnp.zeros(batch_shape + (degree + 1,), dtype=increments.dtype)
    initial = initial.at[..., 0].set(1)
    time_major = jnp.moveaxis(increments, -2, 0)

    def step(prefixes, increment):
        for p in range(degree, 0, -1):
            h = jnp.zeros(batch_shape, dtype=increments.dtype)
            for q in range(p):
                h = (
                    increment[..., word[q]]
                    * (prefixes[..., q] + h)
                    / float(p - q)
                )
            prefixes = prefixes.at[..., p].add(h)
        return prefixes, None

    terminal, _ = jax.lax.scan(step, initial, time_major)
    return terminal[..., degree]


def _resolved_graph_letters(graph: PrefixGraphView, prime_word: Array) -> Array:
    codes = jnp.asarray(graph.letter_codes, dtype=jnp.int32)
    if not np.any(graph.letter_codes < 0):
        return codes
    prime_slots = jnp.maximum(-codes - 1, 0)
    return jnp.where(codes < 0, prime_word[prime_slots], codes)


def quotient_word_reference(
    increments: Array,
    graph: PrefixGraphView,
    prime_word: Array | Sequence[int],
) -> Array:
    """Compute one partially symmetrized signature coordinate on its graph."""

    increments = jnp.asarray(increments)
    prime_word = jnp.asarray(prime_word, dtype=jnp.int32)
    if increments.ndim < 2:
        raise ValueError("increments must have a step and alphabet axis.")
    prime_degree = int(graph.node_prime_degrees[graph.terminal_index])
    if prime_word.shape != (prime_degree,):
        raise ValueError(
            f"prime_word must have shape {(prime_degree,)}, got {prime_word.shape}."
        )

    batch_shape = increments.shape[:-2]
    initial = jnp.zeros(batch_shape + (graph.node_count,), dtype=increments.dtype)
    initial = initial.at[..., graph.scalar_index].set(1)
    time_major = jnp.moveaxis(increments, -2, 0)
    predecessors = jnp.asarray(graph.predecessor_indices, dtype=jnp.int32)
    destinations = jnp.asarray(graph.destination_indices, dtype=jnp.int32)
    coefficients = jnp.asarray(graph.coefficients, dtype=increments.dtype)
    letters = _resolved_graph_letters(graph, prime_word)
    degrees = jnp.asarray(
        graph.node_prime_degrees + graph.node_doubleprime_degrees,
        dtype=jnp.int32,
    )
    maximum_degree = graph.total_degree

    def step(old, increment):
        previous = jnp.zeros_like(old)
        previous = previous.at[..., graph.scalar_index].set(
            old[..., graph.scalar_index]
        )
        for denominator in range(maximum_degree, 0, -1):
            action = jnp.zeros_like(old)
            edge_values = (
                previous[..., predecessors]
                * increment[..., letters]
                * coefficients
            )
            action = action.at[..., destinations].add(edge_values)
            active_degree = maximum_degree - denominator + 1
            active = degrees <= active_degree
            previous = jnp.where(
                active,
                old + action / float(denominator),
                jnp.zeros_like(old),
            )
        return previous, None

    terminal, _ = jax.lax.scan(step, initial, time_major)
    return terminal[..., graph.terminal_index]


def _ordered_block_reference(
    increments: Array,
    block: WordwiseBlockPlan,
    *,
    alphabet_dim: int,
) -> Array:
    if block.total_degree == 0:
        return jnp.ones(increments.shape[:-2] + (1,), dtype=increments.dtype)
    codes = (
        np.arange(block.width, dtype=np.int32)
        if block.word_codes is None
        else block.word_codes
    )
    words = _decoded_words(
        codes,
        base=alphabet_dim,
        length=block.total_degree,
    )
    values = jax.vmap(
        lambda word: ordinary_word_reference(increments, word)
    )(words)
    return jnp.moveaxis(values, 0, -1)


def _quotient_block_reference(
    increments: Array,
    block: WordwiseBlockPlan,
    *,
    d_prime: int,
) -> Array:
    if block.total_degree == 0:
        return jnp.ones(increments.shape[:-2] + (1,), dtype=increments.dtype)
    n = int(block.grade[0])
    prime_words = _decoded_words(
        np.arange(block.dense_prime_width, dtype=np.int32),
        base=d_prime,
        length=n,
    )
    rank_values = []
    if block.prefix_plan is None:
        raise AssertionError("partially symmetrized block lacks a prefix plan")
    for rank in range(block.prefix_plan.graph_count):
        graph = block.prefix_plan.graph(rank)
        values = jax.vmap(
            lambda word: quotient_word_reference(increments, graph, word)
        )(prime_words)
        rank_values.append(jnp.moveaxis(values, 0, -1))
    return jnp.concatenate(rank_values, axis=-1)


def ordinary_signature_reference(
    increments: Array,
    plan: WordwiseLayoutPlan,
):
    """Compute a complete small signature in the plan's standard layout."""

    increments = jnp.asarray(increments)
    if increments.ndim < 2:
        raise ValueError("increments must have a step and alphabet axis.")
    if increments.shape[-1] != plan.alphabet_dim:
        raise ValueError(
            f"increment width {increments.shape[-1]} does not match plan "
            f"alphabet dimension {plan.alphabet_dim}."
        )
    blocks = []
    for block in plan.blocks:
        if plan.partially_symmetrized:
            blocks.append(
                _quotient_block_reference(
                    increments,
                    block,
                    d_prime=plan.dims[0],
                )
            )
        else:
            blocks.append(
                _ordered_block_reference(
                    increments,
                    block,
                    alphabet_dim=plan.alphabet_dim,
                )
            )
    return plan.assemble_signature(blocks)


def fssk_q1_word_state_reference(
    y: Array,
    E: Array,
    psi: Array,
    phi: Array,
    word: Array | Sequence[int],
    *,
    initial_prefixes: Array | None = None,
) -> Array:
    """Evolve the ordered scalar-FSSK prefix states for one target word.

    Inputs are deliberately canonical and unbatched: ``y`` is ``(steps, m)``,
    ``E`` is ``(steps, R, R)``, ``psi`` is ``(steps, N, R)``, and ``phi`` is
    ``(steps, N - 1, R, R)``.  Batch behaviour is tested independently with
    :func:`jax.vmap`, keeping this oracle free of production broadcasting code.
    The return shape is ``(word_length, R)``.
    """

    y = jnp.asarray(y)
    E = jnp.asarray(E)
    psi = jnp.asarray(psi)
    phi = jnp.asarray(phi)
    word = jnp.asarray(word, dtype=jnp.int32)
    if y.ndim != 2:
        raise ValueError(f"y must have shape (steps, m), got {y.shape}.")
    if E.ndim != 3 or E.shape[0] != y.shape[0] or E.shape[-1] != E.shape[-2]:
        raise ValueError("E must have shape (steps, R, R).")
    if psi.ndim != 3 or psi.shape[0] != y.shape[0] or psi.shape[-1] != E.shape[-1]:
        raise ValueError("psi must have shape (steps, N, R).")
    if phi.shape != (y.shape[0], max(psi.shape[1] - 1, 0), E.shape[-1], E.shape[-1]):
        raise ValueError(
            "phi must have shape (steps, N - 1, R, R), got "
            f"{phi.shape}."
        )
    if word.ndim != 1 or word.shape[0] == 0:
        raise ValueError("word must be a nonempty one-dimensional array.")
    degree = word.shape[0]
    if degree > psi.shape[1]:
        raise ValueError(
            f"word length {degree} exceeds coefficient truncation {psi.shape[1]}."
        )
    if initial_prefixes is None:
        initial = jnp.zeros(
            (degree, E.shape[-1]),
            dtype=jnp.result_type(y, E, psi, phi),
        )
    else:
        initial = jnp.asarray(initial_prefixes)
        if initial.shape != (degree, E.shape[-1]):
            raise ValueError(
                f"initial_prefixes must have shape {(degree, E.shape[-1])}, "
                f"got {initial.shape}."
            )

    def step(prefixes, inputs):
        y_step, E_step, psi_step, phi_step = inputs
        for p in range(degree, 0, -1):
            h = psi_step[p - 1]
            for a in range(1, p):
                h = (
                    h * y_step[word[a - 1]]
                    + prefixes[a - 1] @ phi_step[p - 1 - a]
                )
            updated = prefixes[p - 1] @ E_step + h * y_step[word[p - 1]]
            prefixes = prefixes.at[p - 1].set(updated)
        return prefixes, None

    terminal, _ = jax.lax.scan(step, initial, (y, E, psi, phi))
    return terminal


def fssk_q1_quotient_graph_state_reference(
        y: Array,
        E: Array,
        psi: Array,
        phi: Array,
        graph: PrefixGraphView,
        prime_word: Array | Sequence[int],
        *,
        initial_states: Array | None = None,
) -> Array:
    """Evolve every scalar-FSSK state in one quotient prefix graph.

    Inputs are canonical and unbatched, as in
    :func:`fssk_q1_word_state_reference`.  ``initial_states`` has shape
    ``(graph.node_count, R)`` and contains the core-native state coordinate
    represented by every local prefix node; the scalar root is zero.  The
    return value has the same shape.

    For each active target degree ``p``, the auxiliary Horner field starts at
    the root with ``psi[p - 1]``.  It is propagated degree by degree through
    the packed right-generator action, adding the old node state multiplied
    by ``phi[p - 1 - a]`` at each nonterminal degree ``a``.  Nodes of degree
    ``p`` receive their old state multiplied by ``E`` plus the final incoming
    action.  Repeating this for every ``p`` is essential: local prefix states
    advance according to their own target degree rather than remaining stale.
    """
    y = jnp.asarray(y)
    E = jnp.asarray(E)
    psi = jnp.asarray(psi)
    phi = jnp.asarray(phi)
    prime_word = jnp.asarray(prime_word, dtype=jnp.int32)
    if y.ndim != 2:
        raise ValueError(f"y must have shape (steps, m), got {y.shape}.")
    if E.ndim != 3 or E.shape[0] != y.shape[0] or E.shape[-1] != E.shape[-2]:
        raise ValueError("E must have shape (steps, R, R).")
    if psi.ndim != 3 or psi.shape[0] != y.shape[0] or psi.shape[-1] != E.shape[-1]:
        raise ValueError("psi must have shape (steps, N, R).")
    if phi.shape != (
        y.shape[0],
        max(psi.shape[1] - 1, 0),
        E.shape[-1],
        E.shape[-1],
    ):
        raise ValueError(
            "phi must have shape (steps, N - 1, R, R), got "
            f"{phi.shape}."
        )
    prime_degree = int(graph.node_prime_degrees[graph.terminal_index])
    if prime_word.shape != (prime_degree,):
        raise ValueError(
            f"prime_word must have shape {(prime_degree,)}, got "
            f"{prime_word.shape}."
        )
    maximum_degree = graph.total_degree
    if maximum_degree > psi.shape[1]:
        raise ValueError(
            f"graph degree {maximum_degree} exceeds coefficient truncation "
            f"{psi.shape[1]}."
        )

    state_shape = (graph.node_count, E.shape[-1])
    if initial_states is None:
        initial = jnp.zeros(state_shape, dtype=jnp.result_type(y, E, psi, phi))
    else:
        initial = jnp.asarray(initial_states)
        if initial.shape != state_shape:
            raise ValueError(
                f"initial_states must have shape {state_shape}, got "
                f"{initial.shape}."
            )
        initial = initial.at[graph.scalar_index].set(0)

    predecessors = jnp.asarray(graph.predecessor_indices, dtype=jnp.int32)
    destinations = jnp.asarray(graph.destination_indices, dtype=jnp.int32)
    coefficients = jnp.asarray(graph.coefficients, dtype=initial.dtype)
    letters = _resolved_graph_letters(graph, prime_word)
    degrees = jnp.asarray(
        graph.node_prime_degrees + graph.node_doubleprime_degrees,
        dtype=jnp.int32,
    )

    def right_action(values, increment):
        edge_values = (
            values[predecessors]
            * increment[letters, None]
            * coefficients[:, None]
        )
        return jnp.zeros_like(values).at[destinations].add(edge_values)

    def step(old, inputs):
        increment, E_step, psi_step, phi_step = inputs
        new = old
        for target_degree in range(maximum_degree, 0, -1):
            horner = jnp.zeros_like(old)
            horner = horner.at[graph.scalar_index].set(
                psi_step[target_degree - 1]
            )
            for degree in range(1, target_degree):
                incoming = right_action(horner, increment)
                memory = old @ phi_step[target_degree - 1 - degree]
                horner = jnp.where(
                    (degrees == degree)[:, None],
                    incoming + memory,
                    0,
                )
            incoming = right_action(horner, increment)
            updated = old @ E_step + incoming
            new = jnp.where(
                (degrees == target_degree)[:, None],
                updated,
                new,
            )
        return new, None

    terminal, _ = jax.lax.scan(step, initial, (y, E, psi, phi))
    return terminal


def fssk_q1_state_reference(
    y: Array,
    E: Array,
    psi: Array,
    phi: Array,
    plan: WordwiseLayoutPlan,
):
    """Compute a small ordered scalar-FSSK hidden state in standard coordinates."""

    if plan.partially_symmetrized:
        raise ValueError(
            "fssk_q1_state_reference requires an ordered plan; "
            "use fssk_q1_quotient_graph_state_reference for partially "
            "symmetrized states."
        )
    y = jnp.asarray(y)
    if len(plan.blocks) <= 1:
        raise ValueError("scalar-FSSK state requires positive truncation.")
    if y.ndim != 2 or y.shape[-1] != plan.alphabet_dim:
        raise ValueError(
            f"y must have shape (steps, {plan.alphabet_dim}), got {y.shape}."
        )
    blocks = []
    for block in plan.blocks:
        if block.total_degree == 0:
            continue
        codes = (
            np.arange(block.width, dtype=np.int32)
            if block.word_codes is None
            else block.word_codes
        )
        words = _decoded_words(
            codes,
            base=plan.alphabet_dim,
            length=block.total_degree,
        )
        states = jax.vmap(
            lambda word: fssk_q1_word_state_reference(
                y,
                E,
                psi,
                phi,
                word,
            )[-1]
        )(words)
        blocks.append(jnp.moveaxis(states, 0, -1))
    return plan.assemble_first_on(blocks)


def fssk_q1_readout_reference(
    state,
    readout: Array,
    plan: WordwiseLayoutPlan,
):
    """Contract ordered q=1 hidden states and restore the scalar unit."""

    if isinstance(state, BigradedTensor):
        state_blocks = state.blocks
    else:
        state_blocks = tuple(state)
    if not state_blocks:
        raise ValueError("scalar-FSSK state must contain a non-scalar block.")
    if len(state_blocks) != len(plan.blocks) - 1:
        raise ValueError("state block count is inconsistent with the plan.")
    readout = jnp.asarray(readout)
    if readout.ndim != 1:
        raise ValueError("readout must be a one-dimensional state-space vector.")
    batch_shape = state_blocks[0].shape[:-2]
    blocks = [jnp.ones(batch_shape + (1,), dtype=state_blocks[0].dtype)]
    for block in state_blocks:
        if block.shape[-2] != readout.shape[0]:
            raise ValueError("state-space dimension and readout length disagree.")
        blocks.append(jnp.einsum("...rw,r->...w", block, readout))
    return plan.assemble_signature(blocks)


__all__ = [
    "fssk_q1_readout_reference",
    "fssk_q1_quotient_graph_state_reference",
    "fssk_q1_state_reference",
    "fssk_q1_word_state_reference",
    "ordinary_signature_reference",
    "ordinary_word_reference",
    "quotient_word_reference",
]
