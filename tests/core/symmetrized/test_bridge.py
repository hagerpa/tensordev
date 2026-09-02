from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quotient_word_oracle import normal_form, sym_rank
from tensordev.core.bigraded.precompute import (
    BigradedPlanStore,
    colex_placements,
)
from tensordev.core.bigraded.symmetrized.bridge import (
    SymmetrizationBridgePlanStore,
    _expected_bridge_memory_bytes_by_category,
    _lift_partially_symmetrized_block,
    pair_partially_symmetrized_with_ordered_block,
    partially_symmetrize_block,
)


def _scatter_add(output, targets, values):
    return output.at[..., targets, :].add(values)


def _ordered_word(placement, prime_word, doubleprime_word, prime, doubleprime):
    prime_positions = set(placement)
    word = []
    prime_index = 0
    doubleprime_index = 0
    for position in range(len(prime_word) + len(doubleprime_word)):
        if position in prime_positions:
            word.append(prime[prime_word[prime_index]])
            prime_index += 1
        else:
            word.append(doubleprime[doubleprime_word[doubleprime_index]])
            doubleprime_index += 1
    return tuple(word)


def _oracle_flat_target_indices(d_prime, d_doubleprime, grade):
    n, m = grade
    prime = tuple(f"p{index}" for index in range(d_prime))
    doubleprime = tuple(f"x{index}" for index in range(d_doubleprime))
    targets = []
    for placement in colex_placements(n + m, n):
        for prime_index, prime_word in enumerate(product(range(d_prime), repeat=n)):
            for doubleprime_word in product(range(d_doubleprime), repeat=m):
                word = _ordered_word(
                    placement,
                    prime_word,
                    doubleprime_word,
                    prime,
                    doubleprime,
                )
                form = normal_form(word, prime, doubleprime)
                targets.append(sym_rank(form) * d_prime**n + prime_index)
    return tuple(targets)


def _oracle_q(values, targets, output_width):
    output = np.zeros(values.shape[:-1] + (output_width,), dtype=values.dtype)
    for source, target in enumerate(targets):
        output[..., target] += values[..., source]
    return output


def test_bridge_target_map_matches_independent_word_enumeration():
    dims = (2, 2)
    store = SymmetrizationBridgePlanStore(dims, (2, 3))
    for grade, plan in store.grade_plans.items():
        n, m = grade
        targets = _oracle_flat_target_indices(*dims, grade)
        # Remove the unchanged prime coordinate from the full ordered map.
        expected_factored = np.asarray(targets).reshape(
            plan.ordered_placement_count,
            plan.dense_prime_width,
            plan.ordered_doubleprime_width,
        )[:, 0, :]
        expected_factored //= plan.dense_prime_width

        np.testing.assert_array_equal(plan.target_rank_grid(), expected_factored)
        assert plan.source_count == (
            plan.ordered_placement_count * plan.ordered_doubleprime_width
        )
        assert plan.source_count == (
            len(colex_placements(n + m, n)) * dims[1] ** m
        )
        assert not plan.target_ranks.flags.writeable


def test_forward_q_and_dual_lift_match_explicit_oracle_under_jit():
    dims = (2, 2)
    grade = (1, 2)
    plan = SymmetrizationBridgePlanStore(dims, grade).grade_plan(grade)
    targets = _oracle_flat_target_indices(*dims, grade)
    ordered = jnp.arange(
        2 * plan.ordered_block_width,
        dtype=jnp.float32,
    ).reshape(2, plan.ordered_block_width)
    quotient_words = (
        jnp.arange(plan.quotient_block_width, dtype=jnp.float32) - 3.0
    )

    def apply_q(block):
        return partially_symmetrize_block(
            jnp,
            block,
            plan,
            scatter_add=_scatter_add,
        )

    def apply_q_transpose(block):
        return _lift_partially_symmetrized_block(jnp, block, plan)

    expected_q = _oracle_q(
        np.asarray(ordered),
        targets,
        plan.quotient_block_width,
    )
    expected_q_transpose = np.asarray(quotient_words)[np.asarray(targets)]

    np.testing.assert_allclose(jax.jit(apply_q)(ordered), expected_q)
    np.testing.assert_allclose(
        jax.jit(apply_q_transpose)(quotient_words),
        expected_q_transpose,
    )

    lhs = jnp.sum(quotient_words * apply_q(ordered), axis=-1)
    rhs = jnp.sum(apply_q_transpose(quotient_words) * ordered, axis=-1)
    np.testing.assert_allclose(lhs, rhs)


def test_fused_pairing_equals_protected_lift_without_conjugation():
    plan = SymmetrizationBridgePlanStore((2, 2), (1, 2)).grade_plan((1, 2))
    words = (
        jnp.arange(2 * plan.quotient_block_width, dtype=jnp.float32).reshape(
            2,
            plan.quotient_block_width,
        )
        + 1j
    ).astype(jnp.complex64)
    signature = (
        jnp.arange(plan.ordered_block_width, dtype=jnp.float32)[None, :] - 2j
    ).astype(jnp.complex64)

    def fused(word_block, signature_block):
        return pair_partially_symmetrized_with_ordered_block(
            jnp,
            word_block,
            signature_block,
            plan,
        )

    expected = jnp.sum(
        _lift_partially_symmetrized_block(jnp, words, plan) * signature,
        axis=-1,
    )
    result = jax.jit(fused)(words, signature)

    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda value: jnp.real(fused(value, signature).sum())))(
            words
        ),
        jax.grad(lambda value: jnp.real(fused(value, signature).sum()))(words),
    )


def test_bridge_memory_is_exact_factored_and_store_is_self_contained():
    capacity = (2, 2)
    small_prime = SymmetrizationBridgePlanStore((2, 2), capacity)
    large_prime = SymmetrizationBridgePlanStore((7, 2), capacity)

    assert small_prime.memory_bytes_by_category() == (
        _expected_bridge_memory_bytes_by_category((2, 2), capacity)
    )
    assert large_prime.memory_bytes_by_category() == (
        _expected_bridge_memory_bytes_by_category((7, 2), capacity)
    )
    assert small_prime.memory_bytes() == large_prime.memory_bytes()
    assert small_prime.memory_bytes_by_category().keys() == {
        "bridge_target_maps"
    }
    assert not hasattr(small_prime, "plan_store")
    assert not any(
        isinstance(value, BigradedPlanStore)
        for value in small_prime.__dict__.values()
    )
    with pytest.raises(TypeError):
        small_prime.grade_plans[(0, 0)] = small_prime.grade_plan((0, 0))


def test_bridge_block_width_validation_is_eager():
    plan = SymmetrizationBridgePlanStore((2, 2), (1, 1)).grade_plan((1, 1))
    with pytest.raises(ValueError, match="ordered block"):
        partially_symmetrize_block(
            jnp,
            jnp.zeros((plan.ordered_block_width + 1,)),
            plan,
            scatter_add=_scatter_add,
        )
    with pytest.raises(ValueError, match="partially symmetrized block"):
        _lift_partially_symmetrized_block(
            jnp,
            jnp.zeros((plan.quotient_block_width + 1,)),
            plan,
        )
