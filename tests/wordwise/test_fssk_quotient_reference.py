from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import tensordev as td
from tensordev._wordwise.layout import build_layout_plan
from tensordev._wordwise.reference import (
    fssk_q1_quotient_graph_state_reference,
)
from tensordev.core.bigraded.types import BigradedTensor
from tensordev.sss import DenseLambda, FSSK
from tensordev.sss.state_update import fssk_state_from_coef


def _kernel(*, latent_dim=4, state_dim=2):
    return FSSK(
        Lambda=DenseLambda(
            jnp.diag(
                jnp.linspace(0.35, 0.8, state_dim, dtype=jnp.float64)
            )
        ),
        A=jnp.arange(
            1,
            1 + latent_dim * 3,
            dtype=jnp.float64,
        ).reshape(1, latent_dim, 3) / 9,
        b=jnp.linspace(0.7, -0.2, state_dim, dtype=jnp.float64)[None],
    )


def _prime_word(code, *, degree, base):
    return np.asarray(
        [
            (code // (base**power)) % base
            for power in range(degree - 1, -1, -1)
        ],
        dtype=np.int32,
    )


def _random_initial_state(core, *, truncation, state_dim):
    layout = core.resolve_layout(truncation, include_scalar=False)
    keys = jr.split(jr.PRNGKey(735_001), len(layout.grades))
    return BigradedTensor(
        tuple(
            0.03
            * jr.normal(
                key,
                (1, 1, state_dim, layout.block_width(grade)),
                dtype=jnp.float64,
            )
            for key, grade in zip(keys, layout.grades)
        ),
        layout.spec,
    )


def _local_initial(initial_state, graph, prime_word, *, d_prime):
    local = np.zeros(
        (graph.node_count, initial_state.blocks[0].shape[-2]),
        dtype=np.float64,
    )
    for node in range(1, graph.node_count):
        n = int(graph.node_prime_degrees[node])
        m = int(graph.node_doubleprime_degrees[node])
        rank = int(graph.node_ranks[node])
        prime_code = 0
        for letter in prime_word[:n]:
            prime_code = d_prime * prime_code + int(letter)
        coordinate = rank * d_prime**n + prime_code
        local[node] = np.asarray(
            initial_state[n, m][0, 0, :, coordinate]
        )
    return jnp.asarray(local)


def _selected_coordinates(plan, grade):
    block = plan.block(grade)
    ranks = {0, block.prefix_plan.graph_count - 1}
    prime_codes = {0, block.dense_prime_width - 1}
    for rank in sorted(ranks):
        for prime_code in sorted(prime_codes):
            yield block, rank, prime_code


def test_graph_recurrence_matches_portable_quotient_state_over_three_steps():
    truncation = (2, 2)
    core = td.make_core(
        dims=(2, 2),
        max_trunc=truncation,
        partially_symmetrized=True,
    )
    plan = build_layout_plan(core, truncation)
    kernel = _kernel()
    y = 0.05 * jr.normal(jr.PRNGKey(735_002), (3, 4), dtype=jnp.float64)
    coef = kernel.coef(
        jnp.asarray([0.08, 0.11, 0.07]),
        trunc=sum(truncation),
        dtype=y.dtype,
    )
    portable = fssk_state_from_coef(
        y,
        coef=coef,
        trunc=truncation,
        core=core,
    )

    for grade in (
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
        (2, 1),
        (1, 2),
        (2, 2),
    ):
        for block, rank, prime_code in _selected_coordinates(plan, grade):
            graph = block.prefix_plan.graph(rank)
            prime_word = _prime_word(
                prime_code,
                degree=grade[0],
                base=core.dims[0],
            )
            local = fssk_q1_quotient_graph_state_reference(
                y,
                coef.E,
                coef.psi,
                coef.phi[:, 0],
                graph,
                prime_word,
            )
            coordinate = rank * block.dense_prime_width + prime_code
            expected = portable[grade][0, 0, :, coordinate]
            np.testing.assert_allclose(
                local[graph.terminal_index],
                expected,
                atol=3e-12,
                rtol=3e-12,
            )


def test_graph_recurrence_gathers_a_compact_nonzero_initial_state():
    truncation = (2, 2)
    core = td.make_core(
        dims=(2, 2),
        max_trunc=truncation,
        partially_symmetrized=True,
    )
    plan = build_layout_plan(core, truncation)
    kernel = _kernel()
    y = 0.04 * jr.normal(jr.PRNGKey(735_003), (3, 4), dtype=jnp.float64)
    coef = kernel.coef(
        jnp.asarray([0.09, 0.06, 0.12]),
        trunc=sum(truncation),
        dtype=y.dtype,
    )
    initial_state = _random_initial_state(
        core,
        truncation=truncation,
        state_dim=coef.R,
    )
    portable = fssk_state_from_coef(
        y,
        coef=coef,
        trunc=truncation,
        initial_state=initial_state,
        core=core,
    )

    for grade in ((1, 1), (2, 1), (1, 2), (2, 2)):
        for block, rank, prime_code in _selected_coordinates(plan, grade):
            graph = block.prefix_plan.graph(rank)
            prime_word = _prime_word(
                prime_code,
                degree=grade[0],
                base=core.dims[0],
            )
            local_initial = _local_initial(
                initial_state,
                graph,
                prime_word,
                d_prime=core.dims[0],
            )
            local = fssk_q1_quotient_graph_state_reference(
                y,
                coef.E,
                coef.psi,
                coef.phi[:, 0],
                graph,
                prime_word,
                initial_states=local_initial,
            )
            coordinate = rank * block.dense_prime_width + prime_code
            expected = portable[grade][0, 0, :, coordinate]
            np.testing.assert_allclose(
                local[graph.terminal_index],
                expected,
                atol=3e-12,
                rtol=3e-12,
            )
