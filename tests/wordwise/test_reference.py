from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import tensordev as td
from tensordev._wordwise import (
    build_layout_plan,
    fssk_q1_readout_reference,
    fssk_q1_state_reference,
    fssk_q1_word_state_reference,
    ordinary_signature_reference,
    ordinary_word_reference,
    quotient_word_reference,
)
from tensordev.development import path_signature
from tensordev.sss import DenseLambda, FSSK
from tensordev.sss.state_update import fssk_state, fssk_vsig


def _assert_blocks_close(actual, expected, *, atol=2e-10, rtol=2e-10):
    actual_blocks = actual.blocks if hasattr(actual, "blocks") else tuple(actual)
    expected_blocks = (
        expected.blocks if hasattr(expected, "blocks") else tuple(expected)
    )
    assert len(actual_blocks) == len(expected_blocks)
    for actual_block, expected_block in zip(actual_blocks, expected_blocks):
        np.testing.assert_allclose(
            actual_block,
            expected_block,
            atol=atol,
            rtol=rtol,
        )


def _kernel(d=3, m=2, R=2):
    diagonal = jnp.linspace(0.4, 1.0, R, dtype=jnp.float64)
    return FSSK(
        Lambda=DenseLambda(jnp.diag(diagonal)),
        A=jnp.arange(1, 1 + m * d, dtype=jnp.float64).reshape(1, m, d) / 5,
        b=jnp.linspace(0.7, -0.2, R, dtype=jnp.float64)[None],
    )


def test_ordered_word_reference_has_jit_and_gradient_rules():
    increments = jnp.asarray([[0.2, -0.1], [0.4, 0.3]], dtype=jnp.float64)
    word = jnp.asarray([0, 1, 0], dtype=jnp.int32)

    eager = ordinary_word_reference(increments, word)
    compiled = jax.jit(ordinary_word_reference)(increments, word)
    gradient = jax.grad(lambda value: ordinary_word_reference(value, word))(increments)

    np.testing.assert_allclose(compiled, eager)
    assert gradient.shape == increments.shape
    assert np.all(np.isfinite(gradient))


def test_total_reference_matches_portable_signature_with_batch_axes():
    increments = 0.08 * jr.normal(jr.PRNGKey(801), (3, 5, 2), dtype=jnp.float64)
    plan = build_layout_plan(td.Jax(), 4, alphabet_dim=2)

    reference = ordinary_signature_reference(increments, plan)
    portable = path_signature(
        increments,
        trunc=4,
        increment_input=True,
        accumulate=False,
        core=td.Jax(),
    )

    _assert_blocks_close(reference, portable)


def test_ordered_bidegree_reference_matches_portable_selected_words():
    core = td.make_core(dims=(2, 1), max_trunc=(2, 2))
    plan = build_layout_plan(core, (2, 1))
    increments = 0.06 * jr.normal(jr.PRNGKey(802), (2, 4, 3), dtype=jnp.float64)

    reference = ordinary_signature_reference(increments, plan)
    portable = path_signature(
        increments,
        trunc=(2, 1),
        increment_input=True,
        accumulate=False,
        core=core,
    )

    _assert_blocks_close(reference, portable)


def test_quotient_graph_reference_matches_both_existing_routes_at_all_low_grades():
    ordered = td.make_core(dims=(2, 2), max_trunc=(2, 2))
    quotient = td.make_core(
        dims=(2, 2),
        max_trunc=(2, 2),
        partially_symmetrized=True,
    )
    plan = build_layout_plan(quotient, (2, 2))
    increments = 0.05 * jr.normal(jr.PRNGKey(803), (2, 4, 4), dtype=jnp.float64)

    reference = ordinary_signature_reference(increments, plan)
    direct = path_signature(
        increments,
        trunc=(2, 2),
        increment_input=True,
        accumulate=False,
        core=quotient,
    )
    ordered_signature = path_signature(
        increments,
        trunc=(2, 2),
        increment_input=True,
        accumulate=False,
        core=ordered,
    )
    reduced = quotient.tensor_partially_symmetrize(ordered_signature)

    _assert_blocks_close(reference, direct)
    _assert_blocks_close(reference, reduced)

    # Exercise the scalar graph entry point itself at pure double-prime,
    # pure prime, and genuinely mixed grades/ranks.
    for grade in ((0, 2), (2, 0), (1, 1), (2, 2)):
        block = plan.block(grade)
        prefix = block.prefix_plan
        for rank in {0, prefix.graph_count - 1}:
            graph = prefix.graph(rank)
            prime_word = jnp.zeros((grade[0],), dtype=jnp.int32)
            value = quotient_word_reference(increments, graph, prime_word)
            coordinate = rank * block.dense_prime_width
            np.testing.assert_allclose(
                value,
                direct[grade][..., coordinate],
                atol=2e-10,
                rtol=2e-10,
            )


def test_quotient_reference_is_jittable_as_one_static_plan():
    core = td.make_core(
        dims=(1, 2),
        max_trunc=(1, 1),
        partially_symmetrized=True,
    )
    plan = build_layout_plan(core, (1, 1))
    increments = 0.1 * jr.normal(jr.PRNGKey(804), (3, 3), dtype=jnp.float64)

    eager = ordinary_signature_reference(increments, plan)
    compiled = jax.jit(lambda value: ordinary_signature_reference(value, plan))(
        increments
    )

    _assert_blocks_close(compiled, eager)


def test_fssk_word_recurrence_matches_portable_total_state_and_readout():
    kernel = _kernel()
    steps = 5
    truncation = 3
    increments = 0.04 * jr.normal(
        jr.PRNGKey(805),
        (steps, kernel.path_dim),
        dtype=jnp.float64,
    )
    y = jnp.einsum("qmd,td->tqm", kernel.A, increments)[..., 0, :]
    coef = kernel.coef(
        jnp.linspace(0.07, 0.13, steps),
        trunc=truncation,
        dtype=increments.dtype,
    )
    plan = build_layout_plan(td.Jax(), truncation, alphabet_dim=kernel.m)

    reference_state = fssk_q1_state_reference(
        y,
        coef.E,
        coef.psi,
        coef.phi[:, 0],
        plan,
    )
    portable_state = fssk_state(
        increments,
        kernel=kernel,
        dt=jnp.linspace(0.07, 0.13, steps),
        trunc=truncation,
        increment_input=True,
        core=td.Jax(d=kernel.m, max_trunc=truncation),
    )
    for reference, portable in zip(reference_state, portable_state):
        np.testing.assert_allclose(reference, portable[0, 0], atol=2e-10, rtol=2e-10)

    reference_signature = fssk_q1_readout_reference(
        reference_state,
        kernel.b[0],
        plan,
    )
    portable_signature = fssk_vsig(
        increments,
        kernel=kernel,
        dt=jnp.linspace(0.07, 0.13, steps),
        trunc=truncation,
        increment_input=True,
        core=td.Jax(d=kernel.m, max_trunc=truncation),
    )
    _assert_blocks_close(reference_signature, portable_signature)


def test_fssk_bidegree_reference_is_exact_projection_of_total_words():
    kernel = _kernel(d=2, m=3, R=2)
    steps = 4
    truncation = (2, 1)
    y = 0.05 * jr.normal(jr.PRNGKey(806), (steps, kernel.m), dtype=jnp.float64)
    coef = kernel.coef(
        jnp.full((steps,), 0.1),
        trunc=sum(truncation),
        dtype=y.dtype,
    )
    total_plan = build_layout_plan(td.Jax(), sum(truncation), alphabet_dim=kernel.m)
    bidegree_core = td.make_core(dims=(1, 2), max_trunc=truncation)
    bidegree_plan = build_layout_plan(bidegree_core, truncation)

    total = fssk_q1_state_reference(
        y,
        coef.E,
        coef.psi,
        coef.phi[:, 0],
        total_plan,
    )
    selected = fssk_q1_state_reference(
        y,
        coef.E,
        coef.psi,
        coef.phi[:, 0],
        bidegree_plan,
    )

    for grade in selected.grades:
        codes = bidegree_plan.block(grade).word_codes
        np.testing.assert_allclose(
            selected[grade],
            total[sum(grade) - 1][..., codes],
            atol=2e-10,
            rtol=2e-10,
        )


def test_fssk_single_word_reference_vmaps_without_batch_logic():
    kernel = _kernel(d=2, m=2, R=2)
    steps, truncation = 3, 2
    y = jnp.asarray([[0.1, -0.2], [0.3, 0.1], [-0.1, 0.2]])
    coef = kernel.coef(jnp.full((steps,), 0.1), trunc=truncation, dtype=y.dtype)
    words = jnp.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.int32)

    values = jax.jit(
        jax.vmap(
            lambda word: fssk_q1_word_state_reference(
                y,
                coef.E,
                coef.psi,
                coef.phi[:, 0],
                word,
            )[-1]
        )
    )(words)

    assert values.shape == (4, kernel.state_dim)
    assert np.all(np.isfinite(values))
