from __future__ import annotations

from math import comb

import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from tensordev.core.bigraded.jax import JaxBigraded
from tensordev.core.bigraded.types import BigradedTensor
from tensordev.core.bigraded.symmetrized.bridge import (
    SymmetrizationBridgePlanStore,
)
from tensordev.core.bigraded.symmetrized.gamma import (
    PartiallySymmetrizedShearShufflePlanStore,
)
from tensordev.core.bigraded.symmetrized.jax import (
    JaxPartiallySymmetrizedBigraded,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
)
from tensordev.core.bigraded.symmetrized.transforms import (
    PartiallySymmetrizedShearPlanStore,
)
from tensordev.core.shear.symmetrized import (
    JaxPartiallySymmetrizedShearBigraded,
    PartiallySymmetrizedShearGeneratorPlanStore,
    apply_partially_symmetrized_prime_generator_block,
)
from tensordev.core.shear.bigraded import JaxShearBigraded
from tensordev.development import path_signature


config.update("jax_enable_x64", True)


def _placement(row):
    return tuple(tuple(map(int, block)) for block in row)


def _assert_blocks_close(actual, expected, *, atol=1e-11, rtol=1e-11):
    assert actual.spec == expected.spec
    for grade in actual.grades:
        np.testing.assert_allclose(
            actual[grade],
            expected[grade],
            atol=atol,
            rtol=rtol,
        )


def _random_tensor(core, key, *, trunc, batch=(), include_scalar=True):
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


def _quotient_doubleprime_generator(core, generator):
    plan = core.plan_store.doubleprime_generator_plan((0, 1))
    targets = jnp.asarray(plan.target_ranks[:, 0])
    output = jnp.zeros(
        generator.shape[:-1] + (plan.output_rank_count,),
        dtype=generator.dtype,
    )
    return output.at[..., targets].set(generator)


def test_prime_generator_plans_are_exact_L_and_Lambda_arrays():
    base = PartiallySymmetrizedPlanStore((2, 2), (2, 3))
    store = PartiallySymmetrizedShearGeneratorPlanStore(base)

    for output_grade, plan in store.generator_plans.items():
        source_placements = {
            _placement(row): rank
            for rank, row in enumerate(
                base.grade_plan(plan.source_grade).placements
            )
        }
        expected_ranks = []
        expected_coefficients = []
        for row in base.grade_plan(output_grade).placements:
            placement = _placement(row)
            merged = tuple(
                left + right
                for left, right in zip(placement[-2], placement[-1])
            )
            expected_ranks.append(
                source_placements[placement[:-2] + (merged,)]
            )
            coefficient = 1
            for left, right in zip(placement[-2], placement[-1]):
                coefficient *= comb(left + right, left)
            expected_coefficients.append(coefficient)
        np.testing.assert_array_equal(plan.source_ranks, expected_ranks)
        np.testing.assert_array_equal(
            plan.coefficients,
            expected_coefficients,
        )
        assert not plan.source_ranks.flags.writeable
        assert not plan.coefficients.flags.writeable

        source_width = (
            plan.source_rank_count * plan.source_dense_prime_width
        )
        source = np.arange(2 * source_width, dtype=float).reshape(
            2, source_width
        )
        generator = np.asarray([0.3, -0.7])
        actual = apply_partially_symmetrized_prime_generator_block(
            np,
            source,
            generator,
            plan,
        ).reshape(
            2,
            plan.output_rank_count,
            plan.output_dense_prime_width,
        )
        selected = source.reshape(
            2,
            plan.source_rank_count,
            plan.source_dense_prime_width,
        )[:, plan.source_ranks]
        expected = (
            selected[..., None]
            * generator[None, None, None, :]
            * plan.coefficients[None, :, None, None]
        ).reshape(actual.shape)
        np.testing.assert_array_equal(actual, expected)


def test_quotient_shear_core_shares_every_capacity_store_and_active_view():
    base = PartiallySymmetrizedPlanStore((2, 2), (2, 2))
    bridge = SymmetrizationBridgePlanStore((2, 2), (2, 2))
    transforms = PartiallySymmetrizedShearPlanStore(base)
    generators = PartiallySymmetrizedShearGeneratorPlanStore(base)
    shuffle = PartiallySymmetrizedShearShufflePlanStore(base, "full")
    core = JaxPartiallySymmetrizedShearBigraded(
        plan_store=base,
        bridge_plan_store=bridge,
        shear_plan_store=transforms,
        generator_plan_store=generators,
        shuffle_plan_store=shuffle,
    )

    assert core.plan_store is base
    assert core.bridge_plan_store is bridge
    assert core.shear_plan_store is transforms
    assert core.generator_plan_store is generators
    assert core.shuffle_plan_store is shuffle
    assert core.coordinates == "shear"
    assert core.partially_symmetrized is True
    assert {"coordinate_conversion", "generator_action", "shuffle"} <= (
        core.capabilities
    )
    assert "coordinates='shear'" in repr(core)

    view = core.at_truncation((1, 1))
    assert view.plan_store is base
    assert view.bridge_plan_store is bridge
    assert view.shear_plan_store is transforms
    assert view.generator_plan_store is generators
    assert view.shuffle_plan_store is shuffle
    assert view.default_truncation == (1, 1)
    assert core.at_truncation((1, 1)) is view

    memory = core.memory_bytes_by_category()
    assert memory["shear_generator_source_ranks"] == sum(
        plan.source_ranks.nbytes for plan in generators.generator_plans.values()
    )
    assert memory["shear_generator_coefficients"] == sum(
        plan.coefficients.nbytes for plan in generators.generator_plans.values()
    )
    assert core.plan_statistics()["memory_bytes"] == sum(memory.values())

    other = PartiallySymmetrizedPlanStore((2, 2), (2, 2))
    with pytest.raises(ValueError, match="generator_plan_store must share"):
        JaxPartiallySymmetrizedShearBigraded(
            plan_store=base,
            generator_plan_store=(
                PartiallySymmetrizedShearGeneratorPlanStore(other)
            ),
        )


def test_native_generator_equals_transported_homogeneous_product():
    core = JaxPartiallySymmetrizedShearBigraded(
        dims=(2, 2),
        max_trunc=(2, 2),
    )
    prime_generator = jnp.asarray([[0.3, -0.2]], dtype=jnp.float64)
    doubleprime_generator = jnp.asarray(
        [[0.4, 0.7], [-0.1, 0.2], [0.6, -0.3]],
        dtype=jnp.float64,
    )
    quotient_doubleprime = _quotient_doubleprime_generator(
        core,
        doubleprime_generator,
    )
    keys = iter(jr.split(jr.PRNGKey(881), 18))

    for n in range(3):
        for m in range(3):
            if (n, m) == (0, 0):
                continue
            predecessors = []
            generators = []
            predecessor_grades = []
            generator_grades = []
            transported_terms = []
            if m > 0:
                source_grade = n, m - 1
                width = core.plan_store.grade_plan(source_grade).block_width
                source = jr.normal(
                    next(keys),
                    (2, 1, width),
                    dtype=jnp.float64,
                )
                predecessors.append(source)
                generators.append(doubleprime_generator)
                predecessor_grades.append(source_grade)
                generator_grades.append((0, 1))
                transported_terms.append(
                    core.tensor_product_homogeneous(
                        source,
                        quotient_doubleprime,
                        left_grade=source_grade,
                        right_grade=(0, 1),
                    )
                )
            if n > 0:
                source_grade = n - 1, m
                width = core.plan_store.grade_plan(source_grade).block_width
                source = jr.normal(
                    next(keys),
                    (2, 1, width),
                    dtype=jnp.float64,
                )
                predecessors.append(source)
                generators.append(prime_generator)
                predecessor_grades.append(source_grade)
                generator_grades.append((1, 0))
                transported_terms.append(
                    core.tensor_product_homogeneous(
                        source,
                        prime_generator,
                        left_grade=source_grade,
                        right_grade=(1, 0),
                    )
                )

            native = core._right_multiply_generator_output_block(
                tuple(predecessors),
                tuple(generators),
                predecessor_grades=tuple(predecessor_grades),
                generator_grades=tuple(generator_grades),
                output_grade=(n, m),
            )
            transported = transported_terms[0]
            for term in transported_terms[1:]:
                transported = transported + term
            np.testing.assert_allclose(
                native,
                transported,
                atol=2e-11,
                rtol=2e-11,
            )


def test_core_coordinate_hooks_are_inverses_and_exact_transposes():
    core = JaxPartiallySymmetrizedShearBigraded(
        dims=(2, 2),
        max_trunc=(2, 2),
    )
    grade = (2, 1)
    width = core.plan_store.grade_plan(grade).block_width
    values = jr.normal(jr.PRNGKey(890), (3, width), dtype=jnp.float64)
    probe = jr.normal(jr.PRNGKey(891), (3, width), dtype=jnp.float64)

    forward = core._coordinate_forward_block(values, grade)
    inverse = core._coordinate_inverse_block(forward, grade)
    np.testing.assert_allclose(inverse, values, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(
        jnp.sum(forward * probe),
        jnp.sum(
            values * core._coordinate_forward_transpose_block(probe, grade)
        ),
        atol=2e-11,
        rtol=2e-11,
    )

    inverse_values = core._coordinate_inverse_block(values, grade)
    np.testing.assert_allclose(
        jnp.sum(inverse_values * probe),
        jnp.sum(
            values * core._coordinate_inverse_transpose_block(probe, grade)
        ),
        atol=2e-11,
        rtol=2e-11,
    )


def test_direct_shear_horner_is_hat_psi_of_standard_quotient_horner():
    base = PartiallySymmetrizedPlanStore((2, 2), (2, 2))
    bridge = SymmetrizationBridgePlanStore((2, 2), (2, 2))
    standard = JaxPartiallySymmetrizedBigraded(
        plan_store=base,
        bridge_plan_store=bridge,
    )
    shear = JaxPartiallySymmetrizedShearBigraded(
        plan_store=base,
        bridge_plan_store=bridge,
    )
    generator = jnp.asarray(
        [[0.2, -0.3, 0.4, 0.7], [-0.1, 0.5, 0.6, -0.2]],
        dtype=jnp.float64,
    )

    direct = shear.tensor_exponential((generator,), trunc=(2, 2))
    expected = shear._coordinate_forward(
        standard.tensor_exponential((generator,), trunc=(2, 2)),
        trunc=(2, 2),
        first_on=False,
    )
    _assert_blocks_close(direct, expected, atol=2e-11, rtol=2e-11)

    def loss(value):
        result = shear.tensor_exponential((value,), trunc=(2, 2))
        return sum(jnp.square(block).sum() for block in result.blocks)

    eager_gradient = jax.grad(loss)(generator[0])
    compiled_gradient = jax.jit(jax.grad(loss))(generator[0])
    np.testing.assert_allclose(
        compiled_gradient,
        eager_gradient,
        atol=2e-11,
        rtol=2e-11,
    )


def test_general_quotient_series_are_shared_and_coordinate_covariant():
    capacity = (2, 1)
    standard = JaxPartiallySymmetrizedBigraded(
        dims=(1, 2),
        max_trunc=capacity,
    )
    shear = JaxPartiallySymmetrizedShearBigraded(
        plan_store=standard.plan_store,
        bridge_plan_store=standard.bridge_plan_store,
    )
    layout = standard.resolve_layout(capacity, include_scalar=False)
    keys = jr.split(jr.PRNGKey(899), len(layout.grades))
    standard_x = BigradedTensor(
        tuple(
            0.03
            * jr.normal(
                key,
                (layout.block_width(grade),),
                dtype=jnp.float64,
            )
            for key, grade in zip(keys, layout.grades)
        ),
        layout.spec,
    )
    shear_x = shear.tensor_from_standard_coordinates(
        standard_x,
        trunc=capacity,
        first_on=True,
    )

    standard_exp = standard.tensor_exponential(
        standard_x,
        trunc=capacity,
        output_zero_level=True,
    )
    shear_exp = shear.tensor_exponential(
        shear_x,
        trunc=capacity,
        output_zero_level=True,
    )
    _assert_blocks_close(
        shear_exp,
        shear.tensor_from_standard_coordinates(standard_exp),
        atol=2e-11,
        rtol=2e-11,
    )

    standard_positive_exp = standard.tensor_densify(
        {
            grade: standard_exp[grade]
            for grade in standard_exp.grades
            if grade != (0, 0)
        },
        trunc=capacity,
        include_scalar=False,
    )
    shear_positive_exp = shear.tensor_from_standard_coordinates(
        standard_positive_exp,
        first_on=True,
    )
    recovered_standard = standard.tensor_logarithm(
        standard_positive_exp,
        trunc=capacity,
        output_zero_level=False,
    )
    recovered_shear = shear.tensor_logarithm(
        shear_positive_exp,
        trunc=capacity,
        output_zero_level=False,
    )
    _assert_blocks_close(
        recovered_standard,
        standard_x,
        atol=2e-11,
        rtol=2e-11,
    )
    _assert_blocks_close(
        recovered_shear,
        shear_x,
        atol=2e-11,
        rtol=2e-11,
    )


def test_gamma_character_law_and_homogeneous_jax_paths():
    core = JaxPartiallySymmetrizedShearBigraded(
        dims=(1, 1),
        max_trunc=(1, 1),
        precompute_shuffle=True,
    )
    signature = core.tensor_exponential(
        (jnp.asarray([0.2, -0.3], dtype=jnp.float64),),
        trunc=(1, 1),
    )
    left = jnp.asarray([0.7], dtype=jnp.float64)
    right = jnp.asarray([-0.4], dtype=jnp.float64)

    def gamma(left_value):
        return core.tensor_shuffle_product_homogeneous(
            left_value,
            right,
            left_grade=(1, 0),
            right_grade=(0, 1),
        )

    product = gamma(left)
    lhs = (
        jnp.sum(left * signature[(1, 0)])
        * jnp.sum(right * signature[(0, 1)])
    )
    rhs = jnp.sum(product * signature[(1, 1)])
    np.testing.assert_allclose(rhs, lhs, atol=1e-12, rtol=1e-12)

    left_series = core.tensor_densify(
        {(1, 0): left},
        trunc=(1, 1),
        include_scalar=False,
    )
    right_series = core.tensor_densify(
        {(0, 1): right},
        trunc=(1, 1),
        include_scalar=False,
    )
    gamma_series = core.tensor_shuffle_product(
        left_series,
        right_series,
        trunc=(1, 1),
    )
    np.testing.assert_allclose(
        core.tensor_inner_product(gamma_series, signature),
        core.tensor_inner_product(left_series, signature)
        * core.tensor_inner_product(right_series, signature),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(jax.jit(gamma)(left), product)

    batched = jnp.asarray([[0.7], [-0.2], [0.5]], dtype=jnp.float64)
    np.testing.assert_allclose(
        gamma(batched),
        jax.vmap(gamma)(batched),
    )
    gradient = jax.jit(
        jax.grad(lambda value: jnp.square(gamma(value)).sum())
    )(left)
    assert gradient.shape == left.shape
    assert np.all(np.isfinite(np.asarray(gradient)))


@pytest.mark.parametrize(
    "core_type",
    (
        JaxBigraded,
        JaxPartiallySymmetrizedBigraded,
        JaxShearBigraded,
        JaxPartiallySymmetrizedShearBigraded,
    ),
)
@pytest.mark.parametrize("operation", ("tensor_product", "tensor_shuffle_product"))
def test_products_do_not_pad_beyond_the_operands_natural_rectangle(
    core_type,
    operation,
):
    core = core_type(
        dims=(1, 1),
        max_trunc=(2, 2),
        precompute_shuffle=True,
    )
    first_level = jnp.asarray([[2.0], [3.0]], dtype=jnp.float64)
    short = core.tensor_densify(
        {
            (0, 0): jnp.ones((2, 1), dtype=jnp.float64),
            (1, 0): first_level,
        },
        trunc=(1, 0),
    )

    result = getattr(core, operation)(short, short, trunc=(2, 2))

    assert result.truncation == (2, 0)
    assert result.batch_shape == (2,)
    np.testing.assert_allclose(result[(1, 0)], 2.0 * first_level)
    degree_two_factor = 1.0 if operation == "tensor_product" else 2.0
    np.testing.assert_allclose(
        result[(2, 0)], degree_two_factor * first_level**2
    )


@pytest.mark.parametrize(
    "core_type",
    (
        JaxPartiallySymmetrizedBigraded,
        JaxPartiallySymmetrizedShearBigraded,
    ),
)
def test_doubleprime_shuffle_vector_is_embedded_in_quotient_rank_order(
    core_type,
):
    core = core_type(
        dims=(2, 2),
        max_trunc=(2, 2),
        precompute_shuffle=True,
    )
    input_grade = (1, 0)
    width = core.plan_store.grade_plan(input_grade).block_width
    source = jnp.arange(1, 3 * width + 1, dtype=jnp.float64).reshape(
        3, width
    )
    raw = jnp.asarray(
        [[0.3, -0.7], [0.2, 0.5], [-0.4, 0.6]],
        dtype=jnp.float64,
    )
    quotient = _quotient_doubleprime_generator(core, raw)
    expected = core.tensor_shuffle_product_homogeneous(
        source,
        quotient,
        left_grade=input_grade,
        right_grade=(0, 1),
    )

    homogeneous = core.tensor_shuffle_vector_homogeneous(
        source,
        raw,
        input_grade=input_grade,
        generator_part="doubleprime",
    )
    action = core._shuffle_generator_output_block(
        (
            source,
            jnp.zeros(
                (3, core.plan_store.grade_plan((0, 1)).block_width),
                dtype=source.dtype,
            ),
        ),
        (raw, jnp.zeros((3, core.dims[0]), dtype=source.dtype)),
        predecessor_grades=(input_grade, (0, 1)),
        generator_grades=((0, 1), (1, 0)),
        output_grade=(1, 1),
    )
    words = core.tensor_densify(
        {input_grade: source},
        trunc=input_grade,
        include_scalar=False,
    )
    full = core.tensor_shuffle_vector(
        words,
        jnp.concatenate((jnp.zeros_like(raw), raw), axis=-1),
        trunc=(1, 1),
        a_first_on=True,
    )

    for actual in (homogeneous, action, full[(1, 1)]):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_hat_psi_commutes_with_q_in_both_directions():
    dims = (2, 2)
    capacity = (2, 2)
    grade = (2, 2)
    ordered = JaxShearBigraded(dims=dims, max_trunc=capacity)
    quotient = JaxPartiallySymmetrizedShearBigraded(
        dims=dims,
        max_trunc=capacity,
    )
    width = ordered.plan_store.grade_plan(grade).block_width
    block = jr.normal(jr.PRNGKey(910), (2, width), dtype=jnp.float64)

    q_block = quotient.tensor_partially_symmetrize_homogeneous(
        block,
        grade=grade,
    )
    forward_left = quotient._coordinate_forward_block(q_block, grade)
    forward_right = quotient.tensor_partially_symmetrize_homogeneous(
        ordered._coordinate_forward_block(block, grade),
        grade=grade,
    )
    inverse_left = quotient._coordinate_inverse_block(q_block, grade)
    inverse_right = quotient.tensor_partially_symmetrize_homogeneous(
        ordered._coordinate_inverse_block(block, grade),
        grade=grade,
    )

    np.testing.assert_allclose(
        forward_left, forward_right, atol=2e-11, rtol=2e-11
    )
    np.testing.assert_allclose(
        inverse_left, inverse_right, atol=2e-11, rtol=2e-11
    )


def test_directional_shear_pairing_matches_ordered_standard_oracle():
    dims = (1, 2)
    capacity = (2, 2)
    base = PartiallySymmetrizedPlanStore(dims, capacity)
    bridge = SymmetrizationBridgePlanStore(dims, capacity)
    ordered = JaxBigraded(dims=dims, max_trunc=(3, 3))
    quotient_standard = JaxPartiallySymmetrizedBigraded(
        plan_store=base,
        bridge_plan_store=bridge,
    )
    core = JaxPartiallySymmetrizedShearBigraded(
        plan_store=base,
        bridge_plan_store=bridge,
    )
    path = 0.1 * jr.normal(jr.PRNGKey(920), (7, sum(dims)))
    signature = path_signature(path, trunc=(3, 3), core=ordered)
    layout = core.resolve_layout(capacity, include_scalar=True)
    keys = jr.split(jr.PRNGKey(921), len(layout.grades))
    words = BigradedTensor(
        tuple(
            jr.normal(
                key,
                (2, layout.block_width(grade)),
                dtype=jnp.float64,
            )
            for key, grade in zip(keys, layout.grades)
        ),
        layout.spec,
    )

    actual = core.tensor_shear_pairing(words, signature)
    standard_words = core._coordinate_forward_transpose(words)
    lifted_words = quotient_standard._lift_partially_symmetrized(
        standard_words
    )
    expected = ordered.tensor_inner_product(lifted_words, signature)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(
        jax.jit(core.tensor_shear_pairing)(words, signature),
        expected,
        atol=2e-11,
        rtol=2e-11,
    )

    grade = (2, 1)
    homogeneous = core.tensor_shear_pairing_homogeneous(
        words[grade],
        signature[grade],
        grade=grade,
    )
    lifted_block = core._lift_partially_symmetrized_block(
        core._coordinate_forward_transpose_block(words[grade], grade),
        grade,
    )
    np.testing.assert_allclose(
        homogeneous,
        jnp.sum(lifted_block * signature[grade], axis=-1),
        atol=2e-11,
        rtol=2e-11,
    )


@pytest.mark.parametrize("coordinates", ("standard", "shear"))
def test_pairing_accepts_compact_standard_tensor_and_preserves_ordered_path(
    coordinates,
):
    dims = (1, 2)
    capacity = (2, 2)
    ordered = JaxBigraded(dims=dims, max_trunc=capacity)
    standard = JaxPartiallySymmetrizedBigraded(
        dims=dims,
        max_trunc=capacity,
    )
    core = (
        standard
        if coordinates == "standard"
        else JaxPartiallySymmetrizedShearBigraded(
            plan_store=standard.plan_store,
            bridge_plan_store=standard.bridge_plan_store,
        )
    )
    words = _random_tensor(
        core,
        jr.PRNGKey(930),
        trunc=capacity,
        batch=(2, 1),
    )
    ordered_tensor = _random_tensor(
        ordered,
        jr.PRNGKey(931),
        trunc=capacity,
        batch=(1, 3),
    )
    compact_tensor = standard.tensor_partially_symmetrize(ordered_tensor)
    standard_words = (
        words
        if coordinates == "standard"
        else core._coordinate_forward_transpose(words)
    )

    compact = core.tensor_shear_pairing(words, compact_tensor)
    ordered_result = core.tensor_shear_pairing(words, ordered_tensor)
    oracle = standard.tensor_inner_product(standard_words, compact_tensor)

    assert compact.shape == (2, 3)
    np.testing.assert_allclose(compact, oracle, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(
        compact,
        ordered_result,
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        jax.jit(core.tensor_shear_pairing)(words, compact_tensor),
        compact,
        atol=2e-11,
        rtol=2e-11,
    )


def test_homogeneous_pairing_flag_resolves_equal_width_permuted_layout():
    core = JaxPartiallySymmetrizedBigraded(
        dims=(1, 2),
        max_trunc=(0, 1),
    )
    grade = (0, 1)
    plan = core.bridge_plan_store.grade_plan(grade)
    assert plan.ordered_block_width == plan.quotient_block_width == 2
    assert not np.array_equal(plan.target_ranks, np.arange(plan.source_count))

    words = jnp.asarray([2.0, 5.0], dtype=jnp.float64)
    ordered_tensor = jnp.asarray([7.0, 11.0], dtype=jnp.float64)
    compact_tensor = core.tensor_partially_symmetrize_homogeneous(
        ordered_tensor,
        grade=grade,
    )

    ordered_result = core.tensor_shear_pairing_homogeneous(
        words,
        ordered_tensor,
        grade=grade,
        standard_partially_symmetrized=False,
    )
    compact_result = core.tensor_shear_pairing_homogeneous(
        words,
        compact_tensor,
        grade=grade,
        standard_partially_symmetrized=True,
    )
    np.testing.assert_allclose(ordered_result, compact_result)
    np.testing.assert_allclose(
        core.tensor_shear_pairing_homogeneous(
            words,
            compact_tensor,
            grade=grade,
            standard_partially_symmetrized=False,
        ),
        jnp.sum(words * ordered_tensor),
    )
    assert not np.allclose(
        core.tensor_shear_pairing_homogeneous(
            words,
            compact_tensor,
            grade=grade,
            standard_partially_symmetrized=False,
        ),
        compact_result,
    )


def test_compact_pairing_is_vmappable_and_differentiable():
    capacity = (1, 2)
    standard = JaxPartiallySymmetrizedBigraded(
        dims=(1, 2),
        max_trunc=capacity,
    )
    core = JaxPartiallySymmetrizedShearBigraded(
        plan_store=standard.plan_store,
        bridge_plan_store=standard.bridge_plan_store,
    )
    words = _random_tensor(
        core,
        jr.PRNGKey(932),
        trunc=capacity,
        batch=(3,),
    )
    standard_tensor = _random_tensor(
        standard,
        jr.PRNGKey(933),
        trunc=capacity,
        batch=(3,),
    )

    eager = core.tensor_shear_pairing(words, standard_tensor)
    mapped = jax.vmap(
        lambda left, right: core.tensor_shear_pairing(left, right)
    )(words, standard_tensor)
    gradient = jax.jit(
        jax.grad(
            lambda left: jnp.sum(
                core.tensor_shear_pairing(left, standard_tensor)
            )
        )
    )(words)

    np.testing.assert_allclose(mapped, eager, atol=2e-11, rtol=2e-11)
    for block in gradient.blocks:
        assert jnp.all(jnp.isfinite(block))


def test_compact_pairing_capacity_and_validation_contracts():
    dims = (1, 2)
    core = JaxPartiallySymmetrizedShearBigraded(
        dims=dims,
        max_trunc=(1, 2),
    )
    words = _random_tensor(core, jr.PRNGKey(934), trunc=(1, 2))
    large_standard = JaxPartiallySymmetrizedBigraded(
        dims=dims,
        max_trunc=(2, 3),
    )
    large_tensor = _random_tensor(
        large_standard,
        jr.PRNGKey(935),
        trunc=(2, 3),
    )
    standard_words = core._coordinate_forward_transpose(words)
    expected = sum(
        jnp.sum(standard_words[grade] * large_tensor[grade], axis=-1)
        for grade in standard_words.grades
    )
    np.testing.assert_allclose(
        core.tensor_shear_pairing(words, large_tensor),
        expected,
        atol=2e-11,
        rtol=2e-11,
    )

    wrong_coordinates = core.tensor_from_standard_coordinates(
        large_tensor[:2, :3]
    )
    with pytest.raises(ValueError, match="expected 'standard'"):
        core.tensor_shear_pairing(words, wrong_coordinates)

    wrong_dims_core = JaxPartiallySymmetrizedBigraded(
        dims=(1, 1),
        max_trunc=(1, 1),
    )
    wrong_dims = _random_tensor(
        wrong_dims_core,
        jr.PRNGKey(936),
        trunc=(1, 1),
    )
    with pytest.raises(ValueError, match="alphabet dimensions"):
        core.tensor_shear_pairing(words, wrong_dims)
    with pytest.raises(TypeError, match="BigradedTensor"):
        core.tensor_shear_pairing(words, tuple(large_tensor.blocks))

    positive_spec = large_tensor.spec.with_scalar(False)
    positive_tensor = BigradedTensor(
        tuple(large_tensor[grade] for grade in positive_spec.grades),
        positive_spec,
    )
    with pytest.raises(ValueError, match="standard_first_on"):
        core.tensor_shear_pairing(words, positive_tensor)

    grade = (1, 2)
    with pytest.raises(
        TypeError,
        match="standard_partially_symmetrized must be a boolean",
    ):
        core.tensor_shear_pairing_homogeneous(
            words[grade],
            large_tensor[grade],
            grade=grade,
            standard_partially_symmetrized="compact",
        )
    with pytest.raises(ValueError, match="partially symmetrized standard block"):
        core.tensor_shear_pairing_homogeneous(
            words[grade],
            jnp.zeros((1,), dtype=jnp.float64),
            grade=grade,
            standard_partially_symmetrized=True,
        )

    ordered_core = JaxShearBigraded(dims=dims, max_trunc=(1, 2))
    ordered_words = _random_tensor(
        ordered_core,
        jr.PRNGKey(937),
        trunc=(1, 2),
    )
    with pytest.raises(
        ValueError,
        match="partially_symmetrized.*expected False",
    ):
        ordered_core.tensor_shear_pairing(ordered_words, large_tensor)
