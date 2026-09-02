from __future__ import annotations

from collections import Counter
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quotient_word_oracle import NormalForm, hat_gamma
from tensordev.core.bigraded.symmetrized.gamma import (
    PartiallySymmetrizedShearShufflePlanStore,
    apply_partially_symmetrized_shear_shuffle_block,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
)


def _numpy_scatter_add(output, targets, values):
    np.add.at(output, (..., np.asarray(targets), slice(None)), values)
    return output


def _jax_scatter_add(output, targets, values):
    return output.at[..., targets, :].add(values)


def _basis(base, grade, prime_alphabet):
    prime_words = tuple(product(prime_alphabet, repeat=grade[0]))
    return tuple(
        NormalForm(
            tuple(tuple(map(int, block)) for block in placement),
            prime_word,
        )
        for placement in base.grade_plan(grade).placements
        for prime_word in prime_words
    )


def _apply(store, xp, left, right, left_grade, right_grade, scatter_add):
    plan, swap = store.resolve_block_plan(left_grade, right_grade)
    if swap:
        left, right = right, left
    return apply_partially_symmetrized_shear_shuffle_block(
        xp,
        left,
        right,
        plan,
        scatter_add=scatter_add,
    )


def _assert_basis_pair_oracle(base, store, left_grade, right_grade, primes):
    plan, swap = store.resolve_block_plan(left_grade, right_grade)
    if swap:
        left_grade, right_grade = right_grade, left_grade
    left_basis = _basis(base, left_grade, primes)
    right_basis = _basis(base, right_grade, primes)
    output_basis = _basis(base, plan.output_grade, primes)
    for left_index, left_form in enumerate(left_basis):
        for right_index, right_form in enumerate(right_basis):
            left = np.zeros(len(left_basis))
            right = np.zeros(len(right_basis))
            left[left_index] = 1
            right[right_index] = 1
            actual = apply_partially_symmetrized_shear_shuffle_block(
                np,
                left,
                right,
                plan,
                scatter_add=_numpy_scatter_add,
            )
            expected_polynomial = hat_gamma(
                Counter({left_form: 1}),
                Counter({right_form: 1}),
            )
            expected = np.asarray(
                [expected_polynomial.get(form, 0) for form in output_basis]
            )
            np.testing.assert_array_equal(actual, expected)


def test_gamma_matches_independent_oracle_exhaustively_at_small_grades():
    base = PartiallySymmetrizedPlanStore((1, 2), (2, 2))
    store = PartiallySymmetrizedShearShufflePlanStore(base, True)
    for left_grade, right_grade in store.block_plans:
        _assert_basis_pair_oracle(
            base,
            store,
            left_grade,
            right_grade,
            ("p",),
        )


def test_gamma_prime_axis_interleavings_match_two_letter_oracle():
    base = PartiallySymmetrizedPlanStore((2, 2), (2, 2))
    store = PartiallySymmetrizedShearShufflePlanStore(base, True)
    _assert_basis_pair_oracle(
        base,
        store,
        (1, 1),
        (1, 0),
        ("p0", "p1"),
    )


def test_gamma_scopes_are_exact_subsets_and_fail_without_plan_growth():
    base = PartiallySymmetrizedPlanStore((1, 2), (2, 2))
    disabled = PartiallySymmetrizedShearShufflePlanStore(base, False)
    generator = PartiallySymmetrizedShearShufflePlanStore(base, "generator")
    full = PartiallySymmetrizedShearShufflePlanStore(base, "full")

    assert disabled.scope == "none"
    assert not disabled.block_plans
    expected_generator = {
        pair
        for pair in full.block_plans
        if sum(pair[0]) == 1 or sum(pair[1]) == 1
    }
    assert set(generator.block_plans) == expected_generator
    before_disabled = disabled.block_plans
    before_generator = generator.block_plans
    with pytest.raises(KeyError, match="scope='none'"):
        disabled.resolve_block_plan((1, 0), (1, 0))
    with pytest.raises(KeyError, match="scope='generator'"):
        generator.resolve_block_plan((2, 0), (0, 0))
    assert disabled.block_plans is before_disabled
    assert generator.block_plans is before_generator
    assert not disabled.block_plans
    assert set(generator.block_plans) == expected_generator

    plan, swapped = full.resolve_block_plan((1, 0), (0, 1))
    reverse, reverse_swapped = full.resolve_block_plan((0, 1), (1, 0))
    assert reverse is plan
    assert reverse_swapped is not swapped
    with pytest.raises(KeyError, match="output exceeds capacity"):
        full.resolve_block_plan((2, 2), (1, 0))


def test_gamma_is_commutative_associative_and_has_the_scalar_unit():
    base = PartiallySymmetrizedPlanStore((2, 2), (2, 2))
    store = PartiallySymmetrizedShearShufflePlanStore(base, True)
    rng = np.random.default_rng(412)
    grade_a, grade_b, grade_c = (1, 0), (0, 1), (0, 1)
    a = rng.normal(size=base.grade_plan(grade_a).block_width)
    b = rng.normal(size=base.grade_plan(grade_b).block_width)
    c = rng.normal(size=base.grade_plan(grade_c).block_width)

    ab = _apply(
        store, np, a, b, grade_a, grade_b, _numpy_scatter_add
    )
    ba = _apply(
        store, np, b, a, grade_b, grade_a, _numpy_scatter_add
    )
    np.testing.assert_allclose(ab, ba)

    left_associated = _apply(
        store,
        np,
        ab,
        c,
        (1, 1),
        grade_c,
        _numpy_scatter_add,
    )
    bc = _apply(
        store, np, b, c, grade_b, grade_c, _numpy_scatter_add
    )
    right_associated = _apply(
        store,
        np,
        a,
        bc,
        grade_a,
        (0, 2),
        _numpy_scatter_add,
    )
    np.testing.assert_allclose(left_associated, right_associated)

    unit = np.ones(1)
    np.testing.assert_allclose(
        _apply(
            store,
            np,
            unit,
            a,
            (0, 0),
            grade_a,
            _numpy_scatter_add,
        ),
        a,
    )


def test_gamma_batches_jits_vmaps_and_differentiates():
    base = PartiallySymmetrizedPlanStore((2, 2), (2, 2))
    store = PartiallySymmetrizedShearShufflePlanStore(base, True)
    left_grade, right_grade = (1, 1), (1, 0)
    left_width = base.grade_plan(left_grade).block_width
    right_width = base.grade_plan(right_grade).block_width
    left = jnp.arange(3 * left_width, dtype=jnp.float32).reshape(
        3, left_width
    ) / 17
    right = jnp.arange(right_width, dtype=jnp.float32) / 13

    def operation(left_value, right_value):
        return _apply(
            store,
            jnp,
            left_value,
            right_value,
            left_grade,
            right_grade,
            _jax_scatter_add,
        )

    eager = operation(left, right)
    np.testing.assert_allclose(jax.jit(operation)(left, right), eager)
    np.testing.assert_allclose(
        jax.vmap(operation, in_axes=(0, None))(left, right),
        eager,
    )

    def objective(left_value):
        return jnp.square(operation(left_value, right)).sum()

    gradient = jax.jit(jax.grad(objective))(left[0])
    assert gradient.shape == left[0].shape
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_gamma_memory_accounts_for_shared_coefficients_once_per_pair():
    base = PartiallySymmetrizedPlanStore((1, 2), (2, 2))
    store = PartiallySymmetrizedShearShufflePlanStore(base, True)
    memory = store.memory_bytes_by_category()
    plans = tuple(store.block_plans.values())
    keys = tuple(key for plan in plans for key in plan.key_plans)

    assert memory["shuffle_target_maps"] == sum(
        key.rank_plan.target_ranks.nbytes for key in keys
    )
    assert memory["shuffle_coefficients"] == sum(
        plan.coefficients.nbytes for plan in plans
    )
    assert memory["shuffle_dense_permutations"] == sum(
        len(key.dense_axis_permutation) * np.dtype(np.intp).itemsize
        for key in keys
    )
    assert memory["shuffle_segment_metadata"] == 0
    assert store.memory_bytes() == sum(memory.values())
    assert store.plan_statistics()["memory_bytes"] == store.memory_bytes()
    assert all(not plan.coefficients.flags.writeable for plan in plans)
    assert all(
        not key.rank_plan.target_ranks.flags.writeable for key in keys
    )
