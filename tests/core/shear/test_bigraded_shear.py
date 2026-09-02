"""Correctness tests for placement-factored ordered-bidegree shear."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from tensordev.core.bigraded.jax import JaxBigraded
from tensordev.core.bigraded.precompute import BigradedPlanStore
from tensordev.core.bigraded.types import BigradedSpec, BigradedTensor
from tensordev.core.shear.bigraded import JaxShearBigraded
from tensordev.core.shear.jax import JaxShearTotal


def _random_tensor(core, key, trunc, *, include_scalar=True, batch=()):
    layout = core.plan_store.resolve(
        trunc,
        include_scalar=include_scalar,
        coordinates="standard",
    )
    keys = jr.split(key, len(layout.grades))
    return BigradedTensor(
        tuple(
            jr.normal(block_key, batch + (layout.block_width(grade),))
            for grade, block_key in zip(layout.grades, keys)
        ),
        layout.spec,
    )


def _assert_tensors_close(left, right, *, rtol=2e-5, atol=2e-5):
    assert left.spec == right.spec
    for grade in left.grades:
        np.testing.assert_allclose(left[grade], right[grade], rtol=rtol, atol=atol)


def test_shear_spec_and_layout_metadata_are_distinct_but_share_plans():
    standard = BigradedSpec(1, 2, (2, 1), coordinates="standard")
    shear = BigradedSpec(1, 2, (2, 1), coordinates="shear")
    assert standard != shear
    with pytest.raises(ValueError, match="standard.*shear"):
        BigradedSpec(1, 2, (2, 1), coordinates="unknown")

    store = BigradedPlanStore((1, 2), (2, 1))
    standard_layout = store.resolve((1, 1), coordinates="standard")
    shear_layout = store.resolve((1, 1), coordinates="shear")
    assert standard_layout is not shear_layout
    assert standard_layout.grade_plans == shear_layout.grade_plans
    assert shear_layout.with_scalar(False).coordinates == "shear"


def test_bidegree_shear_constructor_rejects_invalid_specialized_stores():
    plan_store = BigradedPlanStore((1, 1), (1, 1))

    with pytest.raises(TypeError, match="shear_plan_store must be"):
        JaxShearBigraded(
            plan_store=plan_store,
            shear_plan_store=object(),
        )
    with pytest.raises(TypeError, match="shuffle_plan_store must be"):
        JaxShearBigraded(
            plan_store=plan_store,
            shuffle_plan_store=object(),
        )


@pytest.mark.parametrize("grade", ((0, 0), (1, 1), (2, 1), (2, 2), (1, 3)))
def test_block_coordinate_maps_are_inverse_and_transpose_correct(grade):
    core = JaxShearBigraded(
        dims=(2, 2),
        max_trunc=(max(2, grade[0]), max(2, grade[1])),
    )
    width = core.plan_store.grade_plan(grade).block_width
    x = jr.normal(jr.PRNGKey(10 + sum(grade)), (3, width))
    y = jr.normal(jr.PRNGKey(20 + sum(grade)), (3, width))

    forward = core._coordinate_forward_block(x, grade)
    inverse = core._coordinate_inverse_block(forward, grade)
    np.testing.assert_allclose(inverse, x, rtol=2e-5, atol=2e-5)

    lhs = jnp.sum(forward * y, axis=-1)
    rhs = jnp.sum(x * core._coordinate_forward_transpose_block(y, grade), axis=-1)
    np.testing.assert_allclose(lhs, rhs, rtol=2e-5, atol=2e-5)

    gx = core._coordinate_inverse_block(x, grade)
    lhs = jnp.sum(gx * y, axis=-1)
    rhs = jnp.sum(x * core._coordinate_inverse_transpose_block(y, grade), axis=-1)
    np.testing.assert_allclose(lhs, rhs, rtol=2e-5, atol=2e-5)


def test_coordinate_conversion_is_jittable_vmappable_and_differentiable():
    core = JaxShearBigraded(dims=(1, 2), max_trunc=(2, 2))
    standard = _random_tensor(core, jr.PRNGKey(30), (2, 2), batch=(4,))
    shear = core.tensor_from_standard_coordinates(standard, trunc=(2, 2))
    assert shear.spec.coordinates == "shear"
    restored = core.tensor_to_standard_coordinates(shear, trunc=(2, 2))
    _assert_tensors_close(restored, standard)

    grade = (2, 2)
    block = standard[grade]
    vmapped = jax.vmap(lambda row: core._coordinate_forward_block(row, grade))(block)
    np.testing.assert_allclose(vmapped, shear[grade], rtol=2e-5, atol=2e-5)
    gradient = jax.grad(
        lambda value: jnp.sum(core._coordinate_forward_block(value, grade) ** 2)
    )(block[0])
    assert gradient.shape == block[0].shape
    assert jnp.all(jnp.isfinite(gradient))


def test_bidegree_transform_matches_dense_total_transform_without_runtime_densification():
    standard_core = JaxBigraded(dims=(1, 2), max_trunc=(2, 2))
    shear_core = JaxShearBigraded(dims=(1, 2), max_trunc=(2, 2))
    total_core = JaxShearTotal(dims=(1, 2), max_trunc=4)
    standard = _random_tensor(standard_core, jr.PRNGKey(35), (2, 2), batch=(2,))
    dense_standard = standard_core.tensor_to_total(standard)

    shear = shear_core.tensor_from_standard_coordinates(standard, trunc=(2, 2))
    dense_from_bidegrees = shear_core.tensor_to_total(shear)
    dense_direct = total_core.tensor_from_standard_coordinates(
        dense_standard, trunc=4
    )
    for via_bidegrees, direct in zip(dense_from_bidegrees, dense_direct):
        np.testing.assert_allclose(via_bidegrees, direct, rtol=3e-5, atol=3e-5)


def test_transported_product_and_adjoint_match_standard_oracles():
    standard_core = JaxBigraded(dims=(1, 2), max_trunc=(2, 2))
    shear_core = JaxShearBigraded(dims=(1, 2), max_trunc=(2, 2))
    assert hasattr(shear_core.tensor_adjoint_product.__func__, "lower")
    standard_a = _random_tensor(standard_core, jr.PRNGKey(40), (1, 1))
    standard_b = _random_tensor(standard_core, jr.PRNGKey(41), (1, 1))
    shear_a = shear_core.tensor_from_standard_coordinates(standard_a, trunc=(1, 1))
    shear_b = shear_core.tensor_from_standard_coordinates(standard_b, trunc=(1, 1))

    got = shear_core.tensor_product(shear_a, shear_b, trunc=(2, 2))
    expected = shear_core.tensor_from_standard_coordinates(
        standard_core.tensor_product(standard_a, standard_b, trunc=(2, 2)),
        trunc=(2, 2),
    )
    _assert_tensors_close(got, expected)

    target_standard = _random_tensor(standard_core, jr.PRNGKey(42), (2, 2))
    target = shear_core.tensor_from_standard_coordinates(
        target_standard, trunc=(2, 2)
    )
    for side in ("left", "right"):
        adjoint = shear_core.tensor_adjoint_product(
            shear_a, target, trunc=(1, 1), side=side
        )
        multiplier = shear_core._coordinate_inverse(shear_a, trunc=None)
        transformed_target = shear_core._coordinate_forward_transpose(
            target, trunc=None
        )
        oracle = standard_core.tensor_adjoint_product(
            multiplier,
            transformed_target,
            trunc=(1, 1),
            side=side,
        )
        oracle = shear_core._coordinate_inverse_transpose(
            oracle, trunc=(1, 1)
        )
        _assert_tensors_close(adjoint, oracle)

    standard_positive = _random_tensor(
        standard_core, jr.PRNGKey(43), (1, 1), include_scalar=False
    )
    shear_positive = shear_core.tensor_from_standard_coordinates(
        standard_positive, trunc=(1, 1), first_on=True
    )
    got_positive = shear_core.tensor_product(
        shear_positive,
        shear_b,
        trunc=(2, 2),
        a_first_on=True,
    )
    expected_positive = shear_core.tensor_from_standard_coordinates(
        standard_core.tensor_product(
            standard_positive,
            standard_b,
            trunc=(2, 2),
            a_first_on=True,
        ),
        trunc=(2, 2),
        first_on=True,
    )
    _assert_tensors_close(got_positive, expected_positive)

    multiplier_grade = (1, 1)
    output_grade = (1, 0)
    target_grade = (2, 1)
    multiplier = shear_a[multiplier_grade]
    homogeneous_target = target[target_grade]
    homogeneous = shear_core.tensor_adjoint_left_homogeneous(
        multiplier,
        homogeneous_target,
        multiplier_grade=multiplier_grade,
        output_grade=output_grade,
    )
    homogeneous_oracle = shear_core._coordinate_inverse_transpose_block(
        standard_core.tensor_adjoint_left_homogeneous(
            shear_core._coordinate_inverse_block(multiplier, multiplier_grade),
            shear_core._coordinate_forward_transpose_block(
                homogeneous_target, target_grade
            ),
            multiplier_grade=multiplier_grade,
            output_grade=output_grade,
        ),
        output_grade,
    )
    np.testing.assert_allclose(
        homogeneous, homogeneous_oracle, rtol=3e-5, atol=3e-5
    )


@pytest.mark.parametrize(
    "output_grade", ((1, 0), (0, 1), (1, 2), (2, 1), (3, 2), (1, 3))
)
def test_native_generator_action_matches_transported_homogeneous_product(output_grade):
    core = JaxShearBigraded(
        dims=(2, 2),
        max_trunc=(max(3, output_grade[0]), max(2, output_grade[1])),
    )
    z = jnp.asarray([0.3, -0.2, 0.7, -0.4])
    n, m = output_grade
    predecessor_blocks = []
    generator_blocks = []
    predecessor_grades = []
    generator_grades = []
    if m:
        grade = (n, m - 1)
        width = core.plan_store.grade_plan(grade).block_width
        predecessor_blocks.append(jr.normal(jr.PRNGKey(50 + n + m), (width,)))
        generator_blocks.append(z[2:])
        predecessor_grades.append(grade)
        generator_grades.append((0, 1))
    if n:
        grade = (n - 1, m)
        width = core.plan_store.grade_plan(grade).block_width
        predecessor_blocks.append(jr.normal(jr.PRNGKey(60 + n + m), (width,)))
        generator_blocks.append(z[:2])
        predecessor_grades.append(grade)
        generator_grades.append((1, 0))

    got = core._right_multiply_generator_output_block(
        tuple(predecessor_blocks),
        tuple(generator_blocks),
        predecessor_grades=tuple(predecessor_grades),
        generator_grades=tuple(generator_grades),
        output_grade=output_grade,
    )
    terms = tuple(
        core.tensor_product_homogeneous(
            source,
            generator,
            left_grade=source_grade,
            right_grade=generator_grade,
        )
        for source, generator, source_grade, generator_grade in zip(
            predecessor_blocks,
            generator_blocks,
            predecessor_grades,
            generator_grades,
        )
    )
    expected = sum(terms[1:], terms[0])
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)


def test_first_level_fmexp_uses_native_action_and_matches_transport():
    standard_core = JaxBigraded(dims=(2, 1), max_trunc=(3, 2))
    shear_core = JaxShearBigraded(dims=(2, 1), max_trunc=(3, 2))
    standard_g = _random_tensor(standard_core, jr.PRNGKey(70), (1, 1))
    shear_g = shear_core.tensor_from_standard_coordinates(
        standard_g, trunc=(1, 1)
    )
    z = jnp.asarray([0.2, -0.1, 0.4])
    got = shear_core.tensor_fmexp(shear_g, (z,), trunc=(3, 2))
    expected = shear_core.tensor_from_standard_coordinates(
        standard_core.tensor_fmexp(standard_g, (z,), trunc=(3, 2)),
        trunc=(3, 2),
    )
    _assert_tensors_close(got, expected, rtol=4e-5, atol=4e-5)


def test_general_formal_series_and_matrix_product_match_transport():
    standard_core = JaxBigraded(dims=(1, 1), max_trunc=(2, 2))
    shear_core = JaxShearBigraded(dims=(1, 1), max_trunc=(2, 2))
    standard_g = _random_tensor(standard_core, jr.PRNGKey(75), (1, 1))
    standard_x = _random_tensor(
        standard_core, jr.PRNGKey(76), (1, 1), include_scalar=False
    )
    shear_g = shear_core.tensor_from_standard_coordinates(
        standard_g, trunc=(1, 1)
    )
    shear_x = shear_core.tensor_from_standard_coordinates(
        standard_x, trunc=(1, 1), first_on=True
    )
    got_exp = shear_core.tensor_fmexp(shear_g, shear_x, trunc=(2, 2))
    expected_exp = shear_core.tensor_from_standard_coordinates(
        standard_core.tensor_fmexp(standard_g, standard_x, trunc=(2, 2)),
        trunc=(2, 2),
    )
    _assert_tensors_close(got_exp, expected_exp, rtol=4e-5, atol=4e-5)
    got_log = shear_core.tensor_logarithm(shear_x, trunc=(2, 2))
    expected_log = shear_core.tensor_from_standard_coordinates(
        standard_core.tensor_logarithm(standard_x, trunc=(2, 2)),
        trunc=(2, 2),
    )
    _assert_tensors_close(got_log, expected_log, rtol=4e-5, atol=4e-5)

    layout = standard_core.resolve_layout((1, 1))
    matrix_a = BigradedTensor(
        tuple(
            jr.normal(jr.PRNGKey(100 + index), (2, 3, layout.block_width(grade)))
            for index, grade in enumerate(layout.grades)
        ),
        layout.spec,
    )
    matrix_b = BigradedTensor(
        tuple(
            jr.normal(jr.PRNGKey(110 + index), (3, 2, layout.block_width(grade)))
            for index, grade in enumerate(layout.grades)
        ),
        layout.spec,
    )
    shear_a = shear_core.tensor_from_standard_coordinates(matrix_a, trunc=(1, 1))
    shear_b = shear_core.tensor_from_standard_coordinates(matrix_b, trunc=(1, 1))
    got_matrix = shear_core.tensor_matrix_product(
        shear_a, shear_b, trunc=(2, 2)
    )
    expected_matrix = shear_core.tensor_from_standard_coordinates(
        standard_core.tensor_matrix_product(matrix_a, matrix_b, trunc=(2, 2)),
        trunc=(2, 2),
    )
    _assert_tensors_close(got_matrix, expected_matrix, rtol=4e-5, atol=4e-5)

    with pytest.raises(ValueError, match="coordinates 'standard'.*'shear'"):
        shear_core.tensor_fmexp(standard_g, (jnp.asarray([0.1, 0.2]),), trunc=(1, 1))


@pytest.mark.parametrize("active", ((0, 0), (1, 1)))
def test_logarithm_empty_standard_transport_preserves_coordinate_tags(active):
    standard_core = JaxBigraded(dims=(1, 1), max_trunc=(1, 1))
    shear_core = JaxShearBigraded(dims=(1, 1), max_trunc=(1, 1))
    standard_x = _random_tensor(
        standard_core,
        jr.PRNGKey(77),
        (1, 1),
        include_scalar=False,
    )
    shear_x = shear_core.tensor_from_standard_coordinates(
        standard_x,
        trunc=(1, 1),
        first_on=True,
    )

    standard = standard_core.tensor_logarithm(standard_x, trunc=active)
    actual = shear_core.tensor_logarithm(shear_x, trunc=active)
    expected = shear_core.tensor_from_standard_coordinates(
        standard,
        trunc=active,
    )

    assert standard.spec.coordinates == "standard"
    assert actual.spec.coordinates == "shear"
    _assert_tensors_close(actual, expected, rtol=4e-5, atol=4e-5)


def test_empty_exponential_arguments_obey_native_and_raw_coordinate_boundaries():
    standard_core = JaxBigraded(dims=(1, 1), max_trunc=(1, 1))
    shear_core = JaxShearBigraded(dims=(1, 1), max_trunc=(1, 1))
    standard_empty = _random_tensor(
        standard_core,
        jr.PRNGKey(78),
        (0, 0),
        include_scalar=False,
    )
    shear_empty = shear_core.tensor_from_standard_coordinates(
        standard_empty,
        trunc=(0, 0),
        first_on=True,
    )
    standard_g = _random_tensor(
        standard_core,
        jr.PRNGKey(79),
        (0, 0),
    )
    shear_g = shear_core.tensor_from_standard_coordinates(
        standard_g,
        trunc=(0, 0),
    )

    native = shear_core.tensor_fmexp(
        shear_g,
        shear_empty,
        trunc=(1, 1),
    )
    raw_standard = shear_core._standard_tensor_fmexp(
        standard_g,
        standard_empty,
        trunc=(1, 1),
        output_zero_level=True,
    )
    assert native.spec.coordinates == "shear"
    assert raw_standard.spec.coordinates == "standard"

    with pytest.raises(ValueError, match="coordinates 'standard'.*'shear'"):
        shear_core.tensor_exponential(standard_empty, trunc=(1, 1))
    with pytest.raises(ValueError, match="coordinates 'standard'.*'shear'"):
        shear_core.tensor_fmexp(shear_g, standard_empty, trunc=(1, 1))
    with pytest.raises(ValueError, match="coordinates 'shear'.*'standard'"):
        standard_core.tensor_exponential(shear_empty, trunc=(1, 1))


def test_coordinatewise_tree_utilities_validate_and_preserve_shear_tags():
    standard_core = JaxBigraded(dims=(1, 1), max_trunc=(1, 1))
    shear_core = JaxShearBigraded(dims=(1, 1), max_trunc=(1, 1))
    standard = _random_tensor(
        standard_core,
        jr.PRNGKey(82),
        (1, 1),
        batch=(2,),
    )
    shear = shear_core.tensor_from_standard_coordinates(
        standard,
        trunc=(1, 1),
    )

    stacked = shear_core.tensor_stack([shear, shear], axis=0)
    moved = shear_core.tensor_moveaxis(stacked, source=0, destination=1)
    sliced = shear_core.tensor_slice(stacked)[0]
    assert stacked.spec.coordinates == "shear"
    assert moved.spec.coordinates == "shear"
    assert sliced.spec.coordinates == "shear"

    with pytest.raises(ValueError, match="coordinates 'standard'.*'shear'"):
        shear_core.tensor_stack([shear, standard], axis=0)
    with pytest.raises(ValueError, match="coordinates 'standard'.*'shear'"):
        shear_core.tensor_moveaxis(standard, source=0, destination=0)
    with pytest.raises(ValueError, match="coordinates 'standard'.*'shear'"):
        shear_core.tensor_slice(standard)


@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_scalar_only_flatten_preserves_batch_dtype_and_coordinate_validation(
    batch_shape,
):
    standard_core = JaxBigraded(dims=(1, 1), max_trunc=(0, 0))
    shear_core = JaxShearBigraded(dims=(1, 1), max_trunc=(0, 0))
    standard_layout = standard_core.resolve_layout((0, 0))
    scalar = jnp.ones(batch_shape + (1,), dtype=jnp.float32)
    standard = BigradedTensor((scalar,), standard_layout.spec)
    shear = shear_core.tensor_from_standard_coordinates(standard, trunc=(0, 0))

    standard_flat = standard_core.tensor_to_flat(
        standard,
        start_at_level_one=True,
    )
    shear_flat = shear_core.tensor_to_flat(
        shear,
        start_at_level_one=True,
    )
    assert standard_flat.shape == shear_flat.shape == batch_shape + (0,)
    assert standard_flat.dtype == shear_flat.dtype == scalar.dtype

    with pytest.raises(ValueError, match="coordinates 'standard'.*'shear'"):
        shear_core.tensor_to_flat(standard, start_at_level_one=True)


@pytest.mark.parametrize(
    ("left_grade", "right_grade"),
    (
        ((1, 0), (0, 1)),
        ((1, 1), (1, 0)),
        ((1, 1), (1, 1)),
        ((1, 2), (0, 1)),
    ),
)
def test_native_gamma_shuffle_matches_transpose_transport(left_grade, right_grade):
    output_grade = (
        left_grade[0] + right_grade[0],
        left_grade[1] + right_grade[1],
    )
    standard_core = JaxBigraded(
        dims=(2, 2), max_trunc=output_grade, precompute_shuffle=True
    )
    shear_core = JaxShearBigraded(
        dims=(2, 2), max_trunc=output_grade, precompute_shuffle=True
    )
    left_width = shear_core.plan_store.grade_plan(left_grade).block_width
    right_width = shear_core.plan_store.grade_plan(right_grade).block_width
    left = jr.normal(jr.PRNGKey(80), (2, left_width))
    right = jr.normal(jr.PRNGKey(81), (1, right_width))
    got = shear_core._gamma_shuffle_block(
        left, right, left_grade, right_grade, output_grade
    )
    standard_left = shear_core._coordinate_forward_transpose_block(left, left_grade)
    standard_right = shear_core._coordinate_forward_transpose_block(right, right_grade)
    standard = standard_core.tensor_shuffle_product_homogeneous(
        standard_left,
        standard_right,
        left_grade=left_grade,
        right_grade=right_grade,
    )
    expected = shear_core._coordinate_inverse_transpose_block(
        standard, output_grade
    )
    np.testing.assert_allclose(got, expected, rtol=3e-5, atol=3e-5)


def test_gamma_scope_active_views_and_layout_conversion_preserve_semantics():
    disabled = JaxShearBigraded(dims=(1, 1), max_trunc=(1, 1))
    with pytest.raises(RuntimeError, match="no shuffle plans"):
        disabled._require_shuffle()

    core = JaxShearBigraded(
        dims=(1, 1),
        max_trunc=(3, 2),
        default_trunc=(2, 2),
        precompute_shuffle="generator",
    )
    assert core.supports("shuffle")
    assert not core.supports("shuffle_product")
    with pytest.raises(RuntimeError, match="precompute_shuffle=True"):
        core._require_full_shuffle()

    view = core.at_truncation((1, 2))
    assert view.plan_store is core.plan_store
    assert view.shear_plan_store is core.shear_plan_store
    assert view.shuffle_plan_store is core.shuffle_plan_store
    for grade in core.plan_store.grade_plans:
        assert (
            core.shear_plan_store.forward_plans[grade].placement_parities
            is core.shear_plan_store.inverse_plans[grade].placement_parities
        )
    categories = core.memory_bytes_by_category()
    assert core.memory_bytes() == sum(categories.values())

    standard = _random_tensor(core, jr.PRNGKey(90), (1, 2))
    shear = core.tensor_from_standard_coordinates(standard, trunc=(1, 2))
    levels = core.tensor_to_total(shear)
    restored = core.tensor_from_total(levels, trunc=(1, 2))
    assert restored.spec.coordinates == "shear"
    _assert_tensors_close(restored, shear)


@pytest.mark.parametrize(
    ("precompute_shuffle", "expected"),
    ((False, "False"), ("generator", "'generator'"), (True, "True")),
)
def test_repr_includes_public_shuffle_constructor_value(
    precompute_shuffle, expected
):
    core = JaxShearBigraded(
        dims=(1, 1),
        max_trunc=(1, 1),
        precompute_shuffle=precompute_shuffle,
    )

    assert f"precompute_shuffle={expected}" in repr(core)


def test_native_bidegree_shear_path_does_not_use_total_layout_conversion(monkeypatch):
    core = JaxShearBigraded(
        dims=(1, 1), max_trunc=(2, 2), precompute_shuffle=True
    )
    standard = _random_tensor(core, jr.PRNGKey(100), (1, 1))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native bidegree shear densified through a total layout")

    monkeypatch.setattr(core, "tensor_to_total", forbidden)
    monkeypatch.setattr(core, "tensor_from_total", forbidden)
    shear = core.tensor_from_standard_coordinates(standard, trunc=(1, 1))
    core.tensor_product(shear, shear, trunc=(2, 2))
    core.tensor_shuffle_product(shear, shear, trunc=(2, 2))
