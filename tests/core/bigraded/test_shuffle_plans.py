from itertools import combinations
from math import comb

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tensordev.core.bigraded.shuffle as shuffle_module
from tensordev.core.bigraded import (
    BigradedPlanStore,
    BigradedShufflePlanStore,
    standard_shuffle_block,
)
from tensordev.core.bigraded.shuffle import (
    _expected_shuffle_memory_bytes_by_category,
    _placement_pair_index_dtype,
)


@pytest.fixture(scope="module")
def stores():
    plan_store = BigradedPlanStore((2, 1), (2, 2))
    return plan_store, BigradedShufflePlanStore(plan_store)


def _decode_word(index, dimension, length):
    word = [0] * length
    for position in range(length - 1, -1, -1):
        word[position] = index % dimension
        index //= dimension
    return tuple(word)


def _word_index(word, dimension):
    index = 0
    for letter in word:
        index = index * dimension + letter
    return index


def _ordinary_shuffle(left, right, dimension, left_degree, right_degree):
    """Small exact ordinary-coordinate oracle used only in these tests."""
    batch_shape = np.broadcast_shapes(left.shape[:-1], right.shape[:-1])
    left = np.broadcast_to(left, batch_shape + (dimension**left_degree,))
    right = np.broadcast_to(right, batch_shape + (dimension**right_degree,))
    output_degree = left_degree + right_degree
    output = np.zeros(
        batch_shape + (dimension**output_degree,),
        dtype=np.result_type(left, right),
    )
    for output_index in range(dimension**output_degree):
        word = _decode_word(output_index, dimension, output_degree)
        for first_positions in combinations(range(output_degree), left_degree):
            first_set = set(first_positions)
            first_word = tuple(word[position] for position in first_positions)
            second_word = tuple(
                word[position]
                for position in range(output_degree)
                if position not in first_set
            )
            output[..., output_index] += (
                left[..., _word_index(first_word, dimension)]
                * right[..., _word_index(second_word, dimension)]
            )
    return output


def _embed_block(store, block, grade):
    plan = store.grade_plan(grade)
    output = np.zeros(
        block.shape[:-1] + ((sum(store.dims)) ** sum(grade),),
        dtype=block.dtype,
    )
    output[..., plan.block_to_total_indices] = block
    return output


@pytest.mark.parametrize(
    "left_grade,right_grade",
    [
        ((0, 0), (1, 1)),
        ((1, 0), (1, 0)),
        ((1, 0), (0, 1)),
        ((1, 1), (1, 0)),
        ((1, 1), (1, 1)),
        ((0, 2), (2, 0)),
    ],
)
def test_homogeneous_shuffle_matches_projected_ordinary_shuffle(
    stores,
    left_grade,
    right_grade,
):
    plan_store, shuffle_store = stores
    rng = np.random.default_rng(401 + 10 * sum(left_grade) + sum(right_grade))
    left = rng.normal(size=plan_store.grade_plan(left_grade).block_width)
    right = rng.normal(size=plan_store.grade_plan(right_grade).block_width)

    result = shuffle_store.tensor_shuffle_product_homogeneous(
        np,
        left,
        right,
        left_grade,
        right_grade,
    )
    output_grade = (
        left_grade[0] + right_grade[0],
        left_grade[1] + right_grade[1],
    )
    embedded_result = _embed_block(plan_store, result, output_grade)
    expected = _ordinary_shuffle(
        _embed_block(plan_store, left, left_grade),
        _embed_block(plan_store, right, right_grade),
        sum(plan_store.dims),
        sum(left_grade),
        sum(right_grade),
    )
    np.testing.assert_allclose(embedded_result, expected, rtol=1e-13, atol=1e-13)


def test_keyed_pair_indices_are_complete_contiguous_and_factored(stores):
    plan_store, shuffle_store = stores
    plan, _ = shuffle_store.resolve_block_plan((1, 1), (1, 0))
    n1, m1 = plan.left_grade
    n2, m2 = plan.right_grade

    assert plan.key_count == comb(n1 + n2, n1) * comb(m1 + m2, m1)
    assert plan.output_placement_count == plan_store.grade_plan(
        plan.output_grade
    ).placement_count
    for key in plan.key_plans:
        assert key.row_count == plan.output_placement_count
        assert key.placement_pair_indices.shape == (
            plan.output_placement_count,
        )
        assert key.placement_pair_indices.dtype == np.dtype(
            _placement_pair_index_dtype(
                plan.left_placement_count * plan.right_placement_count - 1
            )
        )
        assert key.placement_pair_indices.dtype.kind == "u"
        assert not key.placement_pair_indices.flags.writeable
        assert np.all(
            key.placement_pair_indices
            < plan.left_placement_count * plan.right_placement_count
        )
        assert sorted(key.dense_axis_permutation) == list(
            range(sum(plan.output_grade))
        )


@pytest.mark.parametrize("compiled", (False, True))
def test_shared_plan_builder_preserves_exact_plan_arrays(
    stores,
    monkeypatch,
    compiled,
):
    plan_store, reference = stores
    threshold = 0 if compiled else 10**100
    monkeypatch.setattr(
        shuffle_module,
        "_COMPILED_SHUFFLE_ENTRY_THRESHOLD",
        threshold,
    )
    if not compiled:
        emitter = shuffle_module._fill_placement_pair_indices
        monkeypatch.setattr(
            shuffle_module,
            "_fill_placement_pair_indices",
            getattr(emitter, "py_func", emitter),
        )
    actual = BigradedShufflePlanStore(plan_store)

    assert actual._use_compiled_plan_builder is compiled
    assert tuple(actual.block_plans) == tuple(reference.block_plans)
    assert (
        actual.memory_bytes_by_category()
        == reference.memory_bytes_by_category()
    )
    for pair, expected_plan in reference.block_plans.items():
        actual_plan = actual.block_plans[pair]
        assert actual_plan.left_grade == expected_plan.left_grade
        assert actual_plan.right_grade == expected_plan.right_grade
        assert actual_plan.output_grade == expected_plan.output_grade
        assert len(actual_plan.key_plans) == len(expected_plan.key_plans)
        for actual_key, expected_key in zip(
            actual_plan.key_plans,
            expected_plan.key_plans,
        ):
            assert actual_key.prime_key_rank == expected_key.prime_key_rank
            assert (
                actual_key.doubleprime_key_rank
                == expected_key.doubleprime_key_rank
            )
            assert actual_key.prime_placement == expected_key.prime_placement
            assert (
                actual_key.doubleprime_placement
                == expected_key.doubleprime_placement
            )
            assert (
                actual_key.dense_axis_permutation
                == expected_key.dense_axis_permutation
            )
            assert (
                actual_key.placement_pair_indices.dtype
                == expected_key.placement_pair_indices.dtype
            )
            assert not actual_key.placement_pair_indices.flags.writeable
            np.testing.assert_array_equal(
                actual_key.placement_pair_indices,
                expected_key.placement_pair_indices,
            )


@pytest.mark.parametrize(
    "maximum,expected",
    [
        (0, np.uint8),
        (np.iinfo(np.uint8).max, np.uint8),
        (np.iinfo(np.uint8).max + 1, np.uint16),
        (np.iinfo(np.uint16).max + 1, np.uint32),
        (np.iinfo(np.uint32).max + 1, np.uint64),
    ],
)
def test_placement_pair_index_dtype_is_smallest_unsigned(maximum, expected):
    assert _placement_pair_index_dtype(maximum) is expected


def test_reverse_grade_pair_reuses_one_canonical_plan(stores):
    _, shuffle_store = stores
    forward, forward_swap = shuffle_store.resolve_block_plan((1, 1), (0, 1))
    reverse, reverse_swap = shuffle_store.resolve_block_plan((0, 1), (1, 1))
    assert forward is reverse
    assert forward_swap is not reverse_swap


def test_shuffle_supports_broadcast_batches(stores):
    plan_store, shuffle_store = stores
    left_grade = (1, 1)
    right_grade = (1, 0)
    rng = np.random.default_rng(51)
    left = rng.normal(
        size=(2, 1, plan_store.grade_plan(left_grade).block_width)
    )
    right = rng.normal(size=(3, plan_store.grade_plan(right_grade).block_width))

    result = shuffle_store.tensor_shuffle_product_homogeneous(
        np,
        left,
        right,
        left_grade,
        right_grade,
    )
    output_grade = (2, 1)
    expected = _ordinary_shuffle(
        _embed_block(plan_store, left, left_grade),
        _embed_block(plan_store, right, right_grade),
        sum(plan_store.dims),
        sum(left_grade),
        sum(right_grade),
    )
    embedded_result = _embed_block(plan_store, result, output_grade)
    assert result.shape[:2] == (2, 3)
    np.testing.assert_allclose(embedded_result, expected, rtol=1e-13, atol=1e-13)


def test_standalone_block_kernel_is_jittable(stores):
    plan_store, shuffle_store = stores
    requested_left_grade = (1, 0)
    requested_right_grade = (1, 1)
    plan, swap_inputs = shuffle_store.resolve_block_plan(
        requested_left_grade,
        requested_right_grade,
    )
    left = jnp.arange(
        plan_store.grade_plan(requested_left_grade).block_width,
        dtype=jnp.float32,
    )
    right = jnp.arange(
        plan_store.grade_plan(requested_right_grade).block_width,
        dtype=jnp.float32,
    )
    if swap_inputs:
        canonical_left, canonical_right = right, left
    else:
        canonical_left, canonical_right = left, right

    apply = jax.jit(
        lambda first, second: standard_shuffle_block(
            jnp,
            first,
            second,
            plan,
        )
    )
    actual = apply(canonical_left, canonical_right)
    expected = standard_shuffle_block(
        np,
        np.asarray(canonical_left),
        np.asarray(canonical_right),
        plan,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "left_grade,right_grade",
    [((1, 1), (1, 0)), ((1, 1), (1, 1)), ((2, 0), (0, 2))],
)
def test_homogeneous_shuffle_is_commutative(
    stores,
    left_grade,
    right_grade,
):
    plan_store, shuffle_store = stores
    rng = np.random.default_rng(63)
    left = rng.normal(size=plan_store.grade_plan(left_grade).block_width)
    right = rng.normal(size=plan_store.grade_plan(right_grade).block_width)
    forward = shuffle_store.tensor_shuffle_product_homogeneous(
        np,
        left,
        right,
        left_grade,
        right_grade,
    )
    reverse = shuffle_store.tensor_shuffle_product_homogeneous(
        np,
        right,
        left,
        right_grade,
        left_grade,
    )
    np.testing.assert_allclose(forward, reverse, rtol=1e-13, atol=1e-13)


def test_shuffle_plan_memory_reporting_excludes_but_references_shared_store(stores):
    plan_store, shuffle_store = stores
    categories = shuffle_store.memory_bytes_by_category()
    assert set(categories) == {"rank_lists", "dense_permutations"}
    assert categories["rank_lists"] > 0
    assert categories["dense_permutations"] > 0
    assert shuffle_store.memory_bytes() == sum(categories.values())
    assert shuffle_store.memory_mb() == shuffle_store.memory_bytes() / 1024**2

    stats = shuffle_store.plan_statistics()
    assert stats["memory_bytes"] == shuffle_store.memory_bytes()
    assert stats["shared_plan_store_memory_bytes"] == plan_store.memory_bytes()
    assert stats["total_referenced_memory_bytes"] == (
        shuffle_store.memory_bytes() + plan_store.memory_bytes()
    )
    assert stats["block_plan_count"] == len(shuffle_store.block_plans)
    assert stats["key_plan_count"] == sum(
        plan.key_count for plan in shuffle_store.block_plans.values()
    )
    assert shuffle_store.block_plans is shuffle_store.block_plans
    assert categories == _expected_shuffle_memory_bytes_by_category(
        plan_store.dims,
        plan_store.max_truncation,
    )
    with pytest.raises(TypeError):
        shuffle_store.block_plans[((0, 0), (0, 0))] = (
            shuffle_store.resolve_block_plan((0, 0), (0, 0))[0]
        )


def test_generator_scope_filters_pairs_and_memory_estimate():
    plan_store = BigradedPlanStore((2, 1), (2, 2))
    shuffle_store = BigradedShufflePlanStore(plan_store, scope="generator")

    assert shuffle_store.scope == "generator"
    assert all(
        sum(left) == 1 or sum(right) == 1
        for left, right in shuffle_store.block_plans
    )
    assert shuffle_store.memory_bytes_by_category() == (
        _expected_shuffle_memory_bytes_by_category(
            plan_store.dims,
            plan_store.max_truncation,
            scope="generator",
        )
    )
    assert shuffle_store.plan_statistics()["scope"] == "generator"
    with pytest.raises(KeyError, match="scope='generator'"):
        shuffle_store.resolve_block_plan((1, 1), (1, 1))
