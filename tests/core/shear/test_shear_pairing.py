"""Mixed-coordinate pairing contracts for ordered shear words."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from tensordev.core.bigraded.jax import JaxBigraded
from tensordev.core.bigraded.symmetrized.jax import (
    JaxPartiallySymmetrizedBigraded,
)
from tensordev.core.bigraded.types import BigradedTensor
from tensordev.core.jax import Jax
from tensordev.core.shear.bigraded import JaxShearBigraded
from tensordev.core.shear.jax import JaxShearTotal
from tensordev.core.shear.symmetrized import (
    JaxPartiallySymmetrizedShearBigraded,
)
from tensordev.core.universal import Universal
from tensordev.core.utils.annotations import is_jittable
from tensordev.development import path_signature


ATOL = 5e-5
RTOL = 5e-5


def _random_total(key, *, trunc, batch_shape=(), scale=0.08):
    keys = jr.split(key, trunc + 1)
    return tuple(
        scale
        * jr.normal(
            block_key,
            batch_shape + (2**degree,),
            dtype=jnp.float32,
        )
        for degree, block_key in enumerate(keys)
    )


def _random_bigraded(core, key, *, trunc, batch_shape=(), coordinates):
    layout = core.plan_store.resolve(
        trunc,
        include_scalar=True,
        coordinates=coordinates,
    )
    keys = jr.split(key, len(layout.grades))
    return BigradedTensor(
        tuple(
            0.08
            * jr.normal(
                block_key,
                batch_shape + (layout.block_width(grade),),
                dtype=jnp.float32,
            )
            for grade, block_key in zip(layout.grades, keys)
        ),
        layout.spec,
    )


def _assert_all_finite(tree):
    for leaf in jax.tree.leaves(tree):
        assert jnp.all(jnp.isfinite(leaf))


@pytest.mark.parametrize(
    "core_type",
    (
        Universal,
        Jax,
        JaxBigraded,
        JaxPartiallySymmetrizedBigraded,
        JaxShearTotal,
        JaxShearBigraded,
        JaxPartiallySymmetrizedShearBigraded,
    ),
)
def test_shear_pairing_names_are_exact_canonical_aliases(core_type):
    assert (
        core_type.tensor_shear_inner_product
        is core_type.tensor_signature_inner_product
    )
    assert (
        core_type.tensor_shear_inner_product_homogeneous
        is core_type.tensor_signature_inner_product_homogeneous
    )
    assert is_jittable(core_type.tensor_signature_inner_product)
    assert is_jittable(core_type.tensor_signature_inner_product_homogeneous)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: Jax(d=2, max_trunc=1),
        lambda: JaxBigraded(dims=(1, 1), max_trunc=(1, 1)),
        lambda: JaxPartiallySymmetrizedBigraded(
            dims=(1, 1), max_trunc=(1, 1)
        ),
        lambda: JaxShearTotal(dims=(1, 1), max_trunc=1),
        lambda: JaxShearBigraded(dims=(1, 1), max_trunc=(1, 1)),
        lambda: JaxPartiallySymmetrizedShearBigraded(
            dims=(1, 1), max_trunc=(1, 1)
        ),
    ),
)
def test_jax_pairing_aliases_share_one_compiled_wrapper(factory):
    core = factory()

    assert (
        core.tensor_shear_inner_product.__func__
        is core.tensor_signature_inner_product.__func__
    )
    assert (
        core.tensor_shear_inner_product_homogeneous.__func__
        is core.tensor_signature_inner_product_homogeneous.__func__
    )


@pytest.mark.parametrize("family", ("total", "bigraded"))
def test_standard_coordinate_pairing_matches_native_inner_product(family):
    if family == "total":
        core = Jax(d=2, max_trunc=2)
        words = _random_total(jr.PRNGKey(3001), trunc=2)
        ordered_standard_signature = _random_total(
            jr.PRNGKey(3002), trunc=2
        )
        grade = 2
        words_block = words[grade]
        signature_block = ordered_standard_signature[grade]
    else:
        core = JaxBigraded(dims=(1, 1), max_trunc=(1, 1))
        words = _random_bigraded(
            core,
            jr.PRNGKey(3003),
            trunc=(1, 1),
            coordinates="standard",
        )
        ordered_standard_signature = _random_bigraded(
            core,
            jr.PRNGKey(3004),
            trunc=(1, 1),
            coordinates="standard",
        )
        grade = (1, 1)
        words_block = words[grade]
        signature_block = ordered_standard_signature[grade]

    canonical = core.tensor_signature_inner_product(
        words,
        ordered_standard_signature,
    )
    canonical_keyword = core.tensor_signature_inner_product(
        words,
        standard_tensor=ordered_standard_signature,
    )
    homogeneous = core.tensor_signature_inner_product_homogeneous(
        words_block,
        signature_block,
        grade=grade,
    )

    np.testing.assert_allclose(
        canonical,
        core.tensor_inner_product(words, ordered_standard_signature),
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        canonical_keyword,
        canonical,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        homogeneous,
        core.tensor_inner_product_homogeneous(words_block, signature_block),
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        core.tensor_shear_inner_product(
            words,
            standard_tensor=ordered_standard_signature,
        ),
        canonical,
        atol=ATOL,
        rtol=RTOL,
    )


def test_signature_pairing_is_bilinear_without_complex_conjugation():
    core = Jax(d=1, max_trunc=1)
    words = jnp.asarray([1.0 + 2.0j], dtype=jnp.complex64)
    standard_tensor = jnp.asarray([3.0 + 4.0j], dtype=jnp.complex64)
    expected = jnp.sum(words * standard_tensor)

    homogeneous = core.tensor_signature_inner_product_homogeneous(
        words,
        standard_tensor,
        grade=1,
    )
    full = core.tensor_signature_inner_product(
        (words,),
        (standard_tensor,),
        words_first_on=True,
        standard_first_on=True,
    )

    np.testing.assert_allclose(homogeneous, expected)
    np.testing.assert_allclose(full, expected)
    assert not np.allclose(expected, jnp.vdot(words, standard_tensor))


def test_total_pairing_matches_forward_transpose_and_converted_signature():
    trunc = 2
    standard_core = Jax(d=2, max_trunc=trunc)
    shear_core = JaxShearTotal(
        dims=(1, 1),
        max_trunc=trunc,
        precompute_shuffle=False,
    )
    words = _random_total(jr.PRNGKey(3101), trunc=trunc)
    path = jnp.asarray(
        [[0.0, 0.0], [0.2, -0.1], [-0.1, 0.25]],
        dtype=jnp.float32,
    )
    standard_signature = path_signature(
        path, trunc=trunc, core=standard_core
    )

    got = shear_core.tensor_shear_inner_product(words, standard_signature)
    canonical = shear_core.tensor_signature_inner_product(
        words, standard_signature
    )
    forward_transpose = shear_core._coordinate_forward_transpose(
        words, trunc=trunc, first_on=False
    )
    converted_signature = shear_core.tensor_from_standard_coordinates(
        standard_signature, trunc=trunc
    )
    expected_from_transpose = standard_core.tensor_inner_product(
        forward_transpose, standard_signature
    )
    expected_from_conversion = shear_core.tensor_inner_product(
        words, converted_signature
    )

    np.testing.assert_allclose(got, expected_from_transpose, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(got, expected_from_conversion, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(got, canonical, atol=ATOL, rtol=RTOL)
    assert shear_core.shear_plan_store.shuffle_scope == "none"

    positive = shear_core.tensor_shear_inner_product(
        words[1:], standard_signature, words_first_on=True
    )
    positive_transpose = shear_core._coordinate_forward_transpose(
        words[1:], trunc=trunc, first_on=True
    )
    np.testing.assert_allclose(
        positive,
        standard_core.tensor_inner_product(
            positive_transpose, standard_signature[1:]
        ),
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        positive,
        shear_core.tensor_shear_inner_product(
            words[1:],
            standard_signature[1:],
            words_first_on=True,
            standard_first_on=True,
        ),
        atol=ATOL,
        rtol=RTOL,
    )

    wrong_dimension = tuple(
        jnp.ones((1,), dtype=jnp.float32) for _ in range(trunc + 1)
    )
    with pytest.raises(ValueError, match="standard_tensor.*width"):
        shear_core.tensor_shear_inner_product(words, wrong_dimension)
    with pytest.raises(TypeError, match="words_first_on.*boolean"):
        shear_core.tensor_shear_inner_product(
            words, standard_signature, words_first_on=0
        )
    with pytest.raises(TypeError, match="standard_first_on.*boolean"):
        shear_core.tensor_shear_inner_product(
            words, standard_signature, standard_first_on=0
        )

    grade = 2
    homogeneous = shear_core.tensor_shear_inner_product_homogeneous(
        words[grade], standard_signature[grade], grade=grade
    )
    canonical_homogeneous = (
        shear_core.tensor_signature_inner_product_homogeneous(
            words[grade], standard_signature[grade], grade=grade
        )
    )
    expected_homogeneous = standard_core.tensor_inner_product_homogeneous(
        shear_core._coordinate_forward_transpose_block(words[grade], grade),
        standard_signature[grade],
    )
    np.testing.assert_allclose(
        homogeneous, expected_homogeneous, atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(
        homogeneous, canonical_homogeneous, atol=ATOL, rtol=RTOL
    )
    with pytest.raises(TypeError):
        shear_core.tensor_shear_inner_product_homogeneous(
            words[grade], standard_signature[grade]
        )
    with pytest.raises(TypeError, match="requires grade"):
        shear_core.tensor_shear_inner_product_homogeneous(
            words[grade], standard_signature[grade], grade=None
        )
    with pytest.raises(ValueError, match="standard_tensor.*width"):
        shear_core.tensor_shear_inner_product_homogeneous(
            words[grade], jnp.ones((1,), dtype=jnp.float32), grade=grade
        )


def test_total_empty_grade_overlap_preserves_broadcast_batch():
    core = JaxShearTotal(dims=(1, 1), max_trunc=1)
    positive_words = (jnp.ones((3, 1, 2), dtype=jnp.float32),)
    scalar_signature = (jnp.ones((1, 4, 1), dtype=jnp.float32),)

    result = core.tensor_shear_inner_product(
        positive_words,
        scalar_signature,
        words_first_on=True,
    )

    assert result.shape == (3, 4)
    assert result.dtype == jnp.result_type(
        positive_words[0].dtype, scalar_signature[0].dtype
    )
    np.testing.assert_array_equal(result, jnp.zeros((3, 4)))


@pytest.mark.parametrize("empty_side", ("words", "standard"))
def test_total_empty_operand_pairing_uses_available_batch(empty_side):
    core = JaxShearTotal(dims=(1, 1), max_trunc=0)
    batched_scalar = (jnp.ones((3, 1), dtype=jnp.int8),)
    words = tuple() if empty_side == "words" else batched_scalar
    standard_tensor = tuple() if empty_side == "standard" else batched_scalar

    result = core.tensor_shear_inner_product(words, standard_tensor)
    reference = core.tensor_inner_product_homogeneous(
        jnp.zeros((1,), dtype=jnp.int8),
        jnp.zeros((1,), dtype=jnp.int8),
    )

    assert result.shape == (3,)
    assert result.dtype == reference.dtype
    np.testing.assert_array_equal(result, jnp.zeros((3,), dtype=reference.dtype))


def test_bigraded_pairing_matches_forward_transpose_and_converted_signature():
    trunc = (1, 1)
    standard_core = JaxBigraded(dims=(1, 1), max_trunc=trunc)
    shear_core = JaxShearBigraded(
        dims=(1, 1),
        max_trunc=trunc,
        precompute_shuffle=False,
    )
    words = _random_bigraded(
        shear_core,
        jr.PRNGKey(3102),
        trunc=trunc,
        coordinates="shear",
    )
    path = jnp.asarray(
        [[0.0, 0.0], [0.15, -0.1], [-0.05, 0.2]],
        dtype=jnp.float32,
    )
    standard_signature = path_signature(
        path, trunc=trunc, core=standard_core
    )

    got = shear_core.tensor_shear_inner_product(words, standard_signature)
    canonical = shear_core.tensor_signature_inner_product(
        words, standard_signature
    )
    forward_transpose = shear_core._coordinate_forward_transpose(
        words, trunc=trunc, first_on=False
    )
    converted_signature = shear_core.tensor_from_standard_coordinates(
        standard_signature, trunc=trunc
    )

    np.testing.assert_allclose(
        got,
        standard_core.tensor_inner_product(
            forward_transpose, standard_signature
        ),
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        got,
        shear_core.tensor_inner_product(words, converted_signature),
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(got, canonical, atol=ATOL, rtol=RTOL)
    assert shear_core.shuffle_plan_store is None

    positive_word_spec = words.spec.with_scalar(False)
    positive_words = BigradedTensor(
        tuple(words[grade] for grade in positive_word_spec.grades),
        positive_word_spec,
    )
    positive_standard_spec = standard_signature.spec.with_scalar(False)
    positive_standard = BigradedTensor(
        tuple(
            standard_signature[grade]
            for grade in positive_standard_spec.grades
        ),
        positive_standard_spec,
    )
    positive = shear_core.tensor_shear_inner_product(
        positive_words,
        standard_signature,
        words_first_on=True,
    )
    np.testing.assert_allclose(
        positive,
        shear_core.tensor_shear_inner_product(
            positive_words,
            positive_standard,
            words_first_on=True,
            standard_first_on=True,
        ),
        atol=ATOL,
        rtol=RTOL,
    )
    with pytest.raises(ValueError, match="standard_first_on"):
        shear_core.tensor_shear_inner_product(
            positive_words,
            positive_standard,
            words_first_on=True,
        )

    grade = (1, 1)
    homogeneous = shear_core.tensor_shear_inner_product_homogeneous(
        words[grade], standard_signature[grade], grade=grade
    )
    canonical_homogeneous = (
        shear_core.tensor_signature_inner_product_homogeneous(
            words[grade], standard_signature[grade], grade=grade
        )
    )
    expected_homogeneous = standard_core.tensor_inner_product_homogeneous(
        shear_core._coordinate_forward_transpose_block(words[grade], grade),
        standard_signature[grade],
    )
    np.testing.assert_allclose(
        homogeneous, expected_homogeneous, atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(
        homogeneous, canonical_homogeneous, atol=ATOL, rtol=RTOL
    )
    with pytest.raises(TypeError):
        shear_core.tensor_shear_inner_product_homogeneous(
            words[grade], standard_signature[grade]
        )
    with pytest.raises(TypeError, match="requires grade"):
        shear_core.tensor_shear_inner_product_homogeneous(
            words[grade], standard_signature[grade], grade=None
        )
    with pytest.raises(ValueError, match="standard_tensor.*width"):
        shear_core.tensor_shear_inner_product_homogeneous(
            words[grade], jnp.ones((1,), dtype=jnp.float32), grade=grade
        )

    with pytest.raises(ValueError, match="expected 'shear'"):
        shear_core.tensor_shear_inner_product(
            standard_signature, standard_signature
        )
    with pytest.raises(ValueError, match="expected 'standard'"):
        shear_core.tensor_shear_inner_product(words, converted_signature)


def test_bigraded_empty_grade_overlap_preserves_broadcast_batch():
    shear_core = JaxShearBigraded(dims=(1, 1), max_trunc=(1, 1))
    standard_core = JaxBigraded(dims=(1, 1), max_trunc=(1, 1))
    positive_words = shear_core.tensor_densify(
        {(1, 0): jnp.ones((3, 1, 1), dtype=jnp.float32)},
        trunc=(1, 0),
        include_scalar=False,
    )
    positive_standard = standard_core.tensor_densify(
        {(0, 1): jnp.ones((1, 4, 1), dtype=jnp.float32)},
        trunc=(0, 1),
        include_scalar=False,
    )

    result = shear_core.tensor_shear_inner_product(
        positive_words,
        positive_standard,
        words_first_on=True,
        standard_first_on=True,
    )

    assert result.shape == (3, 4)
    assert result.dtype == jnp.result_type(
        positive_words.blocks[0].dtype,
        positive_standard.blocks[0].dtype,
    )
    np.testing.assert_array_equal(result, jnp.zeros((3, 4)))


@pytest.mark.parametrize("family", ("total", "bigraded"))
def test_empty_grade_overlap_uses_homogeneous_reduction_dtype(family):
    if family == "total":
        core = JaxShearTotal(dims=(1, 1), max_trunc=1)
        words = (jnp.ones((3, 2), dtype=jnp.int8),)
        standard_tensor = (jnp.ones((3, 1), dtype=jnp.int8),)
    else:
        core = JaxShearBigraded(dims=(1, 1), max_trunc=(1, 1))
        words = core.tensor_densify(
            {(1, 0): jnp.ones((3, 1), dtype=jnp.int8)},
            trunc=(1, 0),
            include_scalar=False,
        )
        standard_core = JaxBigraded(dims=(1, 1), max_trunc=(1, 1))
        standard_tensor = standard_core.tensor_densify(
            {(0, 1): jnp.ones((3, 1), dtype=jnp.int8)},
            trunc=(0, 1),
            include_scalar=False,
        )

    result = core.tensor_shear_inner_product(
        words,
        standard_tensor,
        words_first_on=True,
        standard_first_on=family == "bigraded",
    )
    reference = core.tensor_inner_product_homogeneous(
        jnp.zeros((1,), dtype=jnp.int8),
        jnp.zeros((1,), dtype=jnp.int8),
    )

    assert result.shape == (3,)
    assert result.dtype == reference.dtype
    np.testing.assert_array_equal(result, jnp.zeros((3,), dtype=reference.dtype))


@pytest.mark.parametrize("family", ("total", "bigraded"))
def test_pairing_is_jittable_vmappable_and_differentiable(family):
    batch_shape = (3,)
    if family == "total":
        core = JaxShearTotal(dims=(1, 1), max_trunc=2)
        words = _random_total(
            jr.PRNGKey(3201), trunc=2, batch_shape=batch_shape
        )
        standard_signature = _random_total(
            jr.PRNGKey(3202), trunc=2, batch_shape=batch_shape
        )
    else:
        core = JaxShearBigraded(dims=(1, 1), max_trunc=(1, 1))
        words = _random_bigraded(
            core,
            jr.PRNGKey(3203),
            trunc=(1, 1),
            batch_shape=batch_shape,
            coordinates="shear",
        )
        standard_signature = _random_bigraded(
            core,
            jr.PRNGKey(3204),
            trunc=(1, 1),
            batch_shape=batch_shape,
            coordinates="standard",
        )

    eager = core.tensor_signature_inner_product(words, standard_signature)
    alias_result = core.tensor_shear_inner_product(words, standard_signature)
    compiled = jax.jit(
        lambda left, right: core.tensor_signature_inner_product(left, right)
    )(words, standard_signature)
    mapped = jax.vmap(
        lambda left, right: core.tensor_signature_inner_product(left, right)
    )(words, standard_signature)
    gradient = jax.grad(
        lambda left: jnp.sum(
            core.tensor_signature_inner_product(left, standard_signature)
        )
    )(words)

    np.testing.assert_allclose(alias_result, eager, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(compiled, eager, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(mapped, eager, atol=ATOL, rtol=RTOL)
    assert jnp.shape(eager) == batch_shape
    _assert_all_finite(gradient)


def test_total_gamma_character_identity_pairs_with_standard_signature():
    trunc = 2
    core = JaxShearTotal(
        dims=(1, 1), max_trunc=trunc, precompute_shuffle=True
    )
    standard_core = Jax(d=2, max_trunc=trunc)
    left = (
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.asarray([0.4, -0.2], dtype=jnp.float32),
        jnp.zeros((4,), dtype=jnp.float32),
    )
    right = (
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.asarray([-0.3, 0.5], dtype=jnp.float32),
        jnp.zeros((4,), dtype=jnp.float32),
    )
    path = jnp.asarray(
        [[0.0, 0.0], [0.2, -0.1], [-0.1, 0.25]],
        dtype=jnp.float32,
    )
    standard_signature = path_signature(
        path, trunc=trunc, core=standard_core
    )
    product = core.tensor_shuffle_product(left, right, trunc=trunc)

    lhs = core.tensor_shear_inner_product(
        left, standard_signature
    ) * core.tensor_shear_inner_product(right, standard_signature)
    rhs = core.tensor_shear_inner_product(product, standard_signature)
    np.testing.assert_allclose(lhs, rhs, atol=ATOL, rtol=RTOL)


def test_bigraded_gamma_character_identity_pairs_with_standard_signature():
    trunc = (1, 1)
    core = JaxShearBigraded(
        dims=(1, 1), max_trunc=trunc, precompute_shuffle=True
    )
    standard_core = JaxBigraded(dims=(1, 1), max_trunc=trunc)
    left = core.tensor_densify(
        {(1, 0): jnp.asarray([0.4], dtype=jnp.float32)},
        trunc=trunc,
        include_scalar=True,
    )
    right = core.tensor_densify(
        {(0, 1): jnp.asarray([-0.3], dtype=jnp.float32)},
        trunc=trunc,
        include_scalar=True,
    )
    path = jnp.asarray(
        [[0.0, 0.0], [0.2, -0.1], [-0.1, 0.25]],
        dtype=jnp.float32,
    )
    standard_signature = path_signature(
        path, trunc=trunc, core=standard_core
    )
    product = core.tensor_shuffle_product(left, right, trunc=trunc)

    lhs = core.tensor_shear_inner_product(
        left, standard_signature
    ) * core.tensor_shear_inner_product(right, standard_signature)
    rhs = core.tensor_shear_inner_product(product, standard_signature)
    np.testing.assert_allclose(lhs, rhs, atol=ATOL, rtol=RTOL)
