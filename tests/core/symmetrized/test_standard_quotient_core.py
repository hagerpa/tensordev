from __future__ import annotations

import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from tensordev.core.bigraded import BigradedTensor, JaxBigraded
from tensordev.core.bigraded.symmetrized.jax import (
    JaxPartiallySymmetrizedBigraded,
)


config.update("jax_enable_x64", True)


def _random_tensor(core, key, *, trunc, include_scalar=True, batch=()):
    layout = core.resolve_layout(trunc, include_scalar=include_scalar)
    keys = jr.split(key, len(layout.grades))
    return BigradedTensor(
        tuple(
            jr.normal(
                block_key,
                batch + (layout.block_width(grade),),
                dtype=jnp.float64,
            )
            for block_key, grade in zip(keys, layout.grades)
        ),
        layout.spec,
    )


def _assert_blocks_close(actual, expected, *, atol=1e-11, rtol=1e-11):
    assert actual.spec == expected.spec
    for grade in actual.grades:
        np.testing.assert_allclose(
            actual[grade], expected[grade], atol=atol, rtol=rtol
        )


@pytest.fixture(scope="module")
def cores():
    ordered = JaxBigraded(dims=(2, 2), max_trunc=(2, 2))
    quotient = JaxPartiallySymmetrizedBigraded(
        dims=(2, 2), max_trunc=(2, 2)
    )
    return ordered, quotient


def test_standard_quotient_product_is_q_of_ordered_product(cores):
    ordered, quotient = cores
    A = _random_tensor(ordered, jr.PRNGKey(301), trunc=(2, 2), batch=(2,))
    B = _random_tensor(ordered, jr.PRNGKey(302), trunc=(2, 2), batch=(2,))
    qA = quotient.tensor_partially_symmetrize(A)
    qB = quotient.tensor_partially_symmetrize(B)

    got = quotient.tensor_product(qA, qB, trunc=(2, 2))
    expected = quotient.tensor_partially_symmetrize(
        ordered.tensor_product(A, B, trunc=(2, 2))
    )

    _assert_blocks_close(got, expected)


@pytest.mark.parametrize("side", ("left", "right"))
def test_standard_quotient_adjoint_has_the_defining_pairing(cores, side):
    _, core = cores
    W = _random_tensor(core, jr.PRNGKey(310), trunc=(1, 1), batch=(2, 1))
    X = _random_tensor(core, jr.PRNGKey(311), trunc=(1, 1), batch=(2, 3))
    Y = _random_tensor(core, jr.PRNGKey(312), trunc=(1, 1), batch=(1, 3))

    product = (
        core.tensor_product(W, X, trunc=(1, 1))
        if side == "left"
        else core.tensor_product(X, W, trunc=(1, 1))
    )
    adjoint = core.tensor_adjoint_product(W, Y, trunc=(1, 1), side=side)

    np.testing.assert_allclose(
        core.tensor_inner_product(product, Y),
        core.tensor_inner_product(X, adjoint),
        atol=1e-10,
        rtol=1e-10,
    )


def test_standard_quotient_horner_signature_is_q_of_ordered(cores):
    ordered, quotient = cores
    generator = jnp.asarray([0.2, -0.3, 0.4, 0.7], dtype=jnp.float64)

    got = quotient.tensor_exponential((generator,), trunc=(2, 2))
    expected = quotient.tensor_partially_symmetrize(
        ordered.tensor_exponential((generator,), trunc=(2, 2))
    )

    _assert_blocks_close(got, expected)


def test_standard_quotient_product_is_jittable_and_differentiable(cores):
    _, core = cores
    left_grade = (1, 1)
    right_grade = (1, 0)
    left_width = core.resolve_layout().block_width(left_grade)
    right_width = core.resolve_layout().block_width(right_grade)
    right = jr.normal(jr.PRNGKey(320), (right_width,), dtype=jnp.float64)

    def objective(left):
        product = core.tensor_product_homogeneous(
            left,
            right,
            left_grade=left_grade,
            right_grade=right_grade,
        )
        return jnp.square(product).sum()

    left = jr.normal(jr.PRNGKey(321), (left_width,), dtype=jnp.float64)
    np.testing.assert_allclose(
        jax.jit(jax.grad(objective))(left),
        jax.grad(objective)(left),
        atol=1e-11,
        rtol=1e-11,
    )


def test_standard_quotient_matrix_product_uses_same_rank_law(cores):
    ordered, quotient = cores
    left_grade = (1, 1)
    right_grade = (0, 1)
    ordered_left_width = ordered.resolve_layout().block_width(left_grade)
    ordered_right_width = ordered.resolve_layout().block_width(right_grade)
    left = jr.normal(
        jr.PRNGKey(330), (2, 3, ordered_left_width), dtype=jnp.float64
    )
    right = jr.normal(
        jr.PRNGKey(331), (3, 4, ordered_right_width), dtype=jnp.float64
    )
    qleft = quotient.tensor_partially_symmetrize_homogeneous(
        left, grade=left_grade
    )
    qright = quotient.tensor_partially_symmetrize_homogeneous(
        right, grade=right_grade
    )

    got = quotient.tensor_matrix_product_homogeneous(
        qleft,
        qright,
        left_grade=left_grade,
        right_grade=right_grade,
    )
    expected = quotient.tensor_partially_symmetrize_homogeneous(
        ordered.tensor_matrix_product_homogeneous(
            left,
            right,
            left_grade=left_grade,
            right_grade=right_grade,
        ),
        grade=(1, 2),
    )

    np.testing.assert_allclose(got, expected, atol=1e-11, rtol=1e-11)


def test_q_transpose_and_fused_pairing_are_exact_transposes(cores):
    _, core = cores
    grade = (1, 2)
    quotient_width = core.resolve_layout().block_width(grade)
    ordered_width = core.bridge_plan_store.grade_plan(grade).ordered_block_width
    words = jr.normal(jr.PRNGKey(340), (2, quotient_width), dtype=jnp.float64)
    signature = jr.normal(jr.PRNGKey(341), (1, ordered_width), dtype=jnp.float64)

    fused = core._pair_representation_standard_block_with_ordered_signature(
        words, signature, grade=grade
    )
    lifted = core._lift_partially_symmetrized_block(words, grade)

    np.testing.assert_allclose(
        fused,
        (lifted * signature).sum(axis=-1),
        atol=1e-11,
        rtol=1e-11,
    )


def test_canonical_signature_pairing_fuses_q_transpose(cores):
    ordered, core = cores
    quotient_words = _random_tensor(
        core, jr.PRNGKey(345), trunc=(1, 2), batch=(2, 1)
    )
    ordered_signature = _random_tensor(
        ordered, jr.PRNGKey(346), trunc=(2, 2), batch=(1, 3)
    )

    got = core.tensor_signature_inner_product(
        quotient_words,
        ordered_signature,
    )
    lifted = core._lift_partially_symmetrized(quotient_words)
    expected = ordered.tensor_inner_product(
        lifted,
        BigradedTensor(
            tuple(ordered_signature[grade] for grade in lifted.grades),
            lifted.spec,
        ),
    )

    np.testing.assert_allclose(got, expected, atol=1e-11, rtol=1e-11)


def test_signature_pairing_accepts_an_ordered_signature_beyond_word_capacity():
    core = JaxPartiallySymmetrizedBigraded(
        dims=(1, 2), max_trunc=(1, 1)
    )
    ordered = JaxBigraded(dims=(1, 2), max_trunc=(2, 2))
    words = _random_tensor(core, jr.PRNGKey(347), trunc=(1, 1))
    signature = _random_tensor(ordered, jr.PRNGKey(348), trunc=(2, 2))

    got = core.tensor_signature_inner_product(words, signature)
    lifted = core._lift_partially_symmetrized(words)
    restricted_signature = BigradedTensor(
        tuple(signature[grade] for grade in lifted.grades), lifted.spec
    )
    expected = ordered.at_truncation((1, 1)).tensor_inner_product(
        lifted, restricted_signature
    )

    np.testing.assert_allclose(got, expected, atol=1e-11, rtol=1e-11)


def test_total_layout_conversion_is_explicitly_rejected(cores):
    _, core = cores
    A = _random_tensor(core, jr.PRNGKey(350), trunc=(1, 1))
    with pytest.raises(RuntimeError, match="partially symmetrized"):
        core.tensor_to_total(A)
    with pytest.raises(RuntimeError, match="partially symmetrized"):
        core.tensor_from_total((jnp.ones((1,)),), trunc=(1, 1))


def test_standard_quotient_shuffle_is_dual_transport_of_gamma():
    ordered = JaxBigraded(
        dims=(1, 2), max_trunc=(2, 2), precompute_shuffle=True
    )
    core = JaxPartiallySymmetrizedBigraded(
        dims=(1, 2), max_trunc=(2, 2), precompute_shuffle=True
    )
    left_grade = (1, 1)
    right_grade = (1, 0)
    output_grade = (2, 1)
    left = jr.normal(
        jr.PRNGKey(360),
        (core.resolve_layout().block_width(left_grade),),
        dtype=jnp.float64,
    )
    right = jr.normal(
        jr.PRNGKey(361),
        (core.resolve_layout().block_width(right_grade),),
        dtype=jnp.float64,
    )

    quotient_shuffle = core.tensor_shuffle_product_homogeneous(
        left,
        right,
        left_grade=left_grade,
        right_grade=right_grade,
    )
    lifted = core._lift_partially_symmetrized_block(
        quotient_shuffle, output_grade
    )
    ordered_shuffle = ordered.tensor_shuffle_product_homogeneous(
        core._lift_partially_symmetrized_block(left, left_grade),
        core._lift_partially_symmetrized_block(right, right_grade),
        left_grade=left_grade,
        right_grade=right_grade,
    )

    np.testing.assert_allclose(lifted, ordered_shuffle, atol=1e-11, rtol=1e-11)


def test_standard_quotient_generator_shuffle_scope_is_exact():
    core = JaxPartiallySymmetrizedBigraded(
        dims=(1, 2),
        max_trunc=(2, 2),
        precompute_shuffle="generator",
    )
    grade = (1, 1)
    block = jnp.ones((core.resolve_layout().block_width(grade),))
    vector = jnp.asarray([0.2, -0.1, 0.4])

    result = core.tensor_shuffle_vector_homogeneous(
        block,
        vector,
        input_grade=grade,
        generator_part="prime",
    )
    assert result.shape == (core.resolve_layout().block_width((2, 1)),)
    with pytest.raises(RuntimeError, match="precompute_shuffle=True"):
        core.tensor_shuffle_product_homogeneous(
            block,
            block,
            left_grade=grade,
            right_grade=grade,
        )
