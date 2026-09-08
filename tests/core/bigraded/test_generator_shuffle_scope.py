from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tensordev import make_core
from tensordev.core.bigraded import (
    BigradedPlanStore,
    BigradedShufflePlanStore,
)


def _is_generator_pair(pair):
    return sum(pair[0]) == 1 or sum(pair[1]) == 1


def test_generator_store_is_the_exact_subset_of_full_shuffle_pairs():
    plan_store = BigradedPlanStore((1, 2), (3, 2))
    full = BigradedShufflePlanStore(plan_store)
    generator = BigradedShufflePlanStore(plan_store, scope="generator")

    expected_pairs = {
        pair for pair in full.block_plans if _is_generator_pair(pair)
    }
    assert full.scope == "full"
    assert generator.scope == "generator"
    assert set(generator.block_plans) == expected_pairs
    assert generator.memory_bytes() < full.memory_bytes()


@pytest.mark.parametrize("generator_part", ["prime", "doubleprime"])
def test_generator_core_matches_full_core_vector_action(generator_part):
    dims = (1, 2)
    capacity = (2, 2)
    input_grade = (1, 1)
    generator = make_core(
        dims=dims,
        max_trunc=capacity,
        precompute_shuffle="generator",
    )
    full = make_core(
        dims=dims,
        max_trunc=capacity,
        precompute_shuffle=True,
    )
    width = generator.plan_store.grade_plan(input_grade).block_width
    block = jnp.arange(width, dtype=jnp.float32) / 7
    vector = jnp.array([0.25, -0.5, 0.75], dtype=jnp.float32)

    actual = generator.tensor_shuffle_vector_homogeneous(
        block,
        vector,
        input_grade=input_grade,
        generator_part=generator_part,
    )
    expected = full.tensor_shuffle_vector_homogeneous(
        block,
        vector,
        input_grade=input_grade,
        generator_part=generator_part,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_generator_core_exposes_only_generator_shuffle_capability():
    core = make_core(
        dims=(1, 1),
        max_trunc=(2, 2),
        precompute_shuffle="generator",
    )
    block = jnp.ones(
        (core.plan_store.grade_plan((1, 1)).block_width,),
        dtype=jnp.float32,
    )

    assert core.supports("shuffle")
    assert not core.supports("shuffle_product")
    assert (
        core.at_truncation((1, 1)).shuffle_plan_store
        is core.shuffle_plan_store
    )
    statistics = core.plan_statistics()
    assert statistics["shuffle_scope"] == "generator"
    assert statistics["bytes_by_category"] == core.memory_bytes_by_category()
    assert (
        sum(statistics["bytes_by_category"].values())
        == statistics["memory_bytes"]
    )
    with pytest.raises(RuntimeError, match="generator-only"):
        core.tensor_shuffle_product_homogeneous(
            block,
            block,
            left_grade=(1, 1),
            right_grade=(1, 1),
        )


@pytest.mark.parametrize("scope", [None, "generators", 1])
def test_shuffle_store_rejects_invalid_scope(scope):
    plan_store = BigradedPlanStore((1, 1), (1, 1))

    with pytest.raises((TypeError, ValueError), match="scope"):
        BigradedShufflePlanStore(plan_store, scope=scope)
