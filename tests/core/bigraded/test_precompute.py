from itertools import combinations
from math import comb

import numpy as np
import pytest

from tensordev.core.bigraded import (
    BigradedPlanStore,
    colex_placements,
    colex_rank,
    colex_unrank,
)


@pytest.mark.parametrize("length", range(7))
def test_colex_rank_unrank_exhaustively(length):
    for subset_size in range(length + 1):
        placements = colex_placements(length, subset_size)
        assert len(placements) == comb(length, subset_size)
        for rank, placement in enumerate(placements):
            assert colex_rank(placement, length=length) == rank
            assert colex_unrank(rank, subset_size, length) == placement
        assert set(placements) == set(combinations(range(length), subset_size))


def test_colex_validation():
    with pytest.raises(ValueError, match="strictly increasing"):
        colex_rank((0, 0), length=2)
    with pytest.raises(ValueError, match="outside"):
        colex_rank((2,), length=2)
    with pytest.raises(ValueError, match="outside"):
        colex_unrank(3, 1, 3)
    with pytest.raises(ValueError, match="exceeds"):
        colex_unrank(0, 4, 3)


def test_grade_plans_have_correct_widths_and_readonly_maps():
    store = BigradedPlanStore((2, 3), (3, 2))
    for (n, m), plan in store.grade_plans.items():
        assert plan.placement_count == comb(n + m, n)
        assert plan.block_width == comb(n + m, n) * 2**n * 3**m
        assert plan.dense_shape == (comb(n + m, n), 2**n, 3**m)
        assert plan.placements.shape == (comb(n + m, n), n)
        assert not plan.placements.flags.writeable
        assert not plan.block_to_total_indices.flags.writeable

    assert store.grade_plans is store.grade_plans
    assert store.concat_plans is store.concat_plans
    with pytest.raises(TypeError):
        store.grade_plans[(0, 0)] = store.grade_plan((0, 0))


def test_total_word_conversion_indices_are_unique_for_each_bidegree():
    store = BigradedPlanStore((2, 2), (3, 3))
    for plan in store.grade_plans.values():
        ordinary = plan.block_to_total_indices
        assert ordinary.shape == (plan.block_width,)
        assert len(np.unique(ordinary)) == plan.block_width
        assert np.all(ordinary >= 0)
        assert np.all(ordinary < sum(store.dims) ** plan.total_degree)


def test_conversion_maps_partition_complete_ordinary_total_levels():
    store = BigradedPlanStore((2, 1), (4, 4))
    alphabet_dimension = 3
    for total in range(5):
        indices = np.concatenate(
            [
                store.grade_plan((n, total - n)).block_to_total_indices
                for n in range(total + 1)
            ]
        )
        np.testing.assert_array_equal(np.sort(indices), np.arange(alphabet_dimension**total))


def test_concatenation_offsets_match_direct_placement_concatenation():
    store = BigradedPlanStore((2, 2), (3, 3))
    for plan in store.concat_plans.values():
        left = store.grade_plan(plan.left_grade)
        right = store.grade_plan(plan.right_grade)
        length_left = sum(plan.left_grade)
        expected = np.empty(
            (left.placement_count, right.placement_count),
            dtype=plan.placement_offsets.dtype,
        )
        for j1, p1 in enumerate(left.placements):
            for j2, p2 in enumerate(right.placements):
                placement = tuple(int(v) for v in p1) + tuple(
                    length_left + int(v) for v in p2
                )
                expected[j1, j2] = colex_rank(placement)
        np.testing.assert_array_equal(plan.target_ranks(), expected)
        assert len(np.unique(expected)) == expected.size


def test_concatenation_dense_axis_metadata():
    plan = BigradedPlanStore((2, 3), (3, 3)).concat_plan((1, 2), (2, 1))
    # raw letters: p1, d1, d1, p2, p2, d2
    # output:      p1, p2, p2, d1, d1, d2
    assert plan.dense_axis_permutation == (0, 3, 4, 1, 2, 5)
    # raw full outer axes: P1, p1, d1, d1, P2, p2, p2, d2
    assert plan.outer_axis_permutation == (0, 4, 1, 5, 6, 2, 3, 7)
    assert plan.dense_input_shape == (2, 3, 3, 2, 2, 3)
    assert plan.dense_output_shape == (2, 2, 2, 3, 3, 3)


def test_plan_store_memory_statistics_are_consistent_and_cache_aware():
    store = BigradedPlanStore((2, 1), (2, 2))
    categories = store.memory_bytes_by_category()
    assert set(categories) == {"placements", "conversion", "concatenation"}
    assert all(value >= 0 for value in categories.values())
    assert store.memory_bytes() == sum(categories.values())
    assert store.memory_mb() == store.memory_bytes() / 1024**2

    stats = store.plan_statistics()
    assert stats["memory_bytes"] == store.memory_bytes()
    assert stats["grade_count"] == 9
    assert stats["concat_plan_count"] == len(store.concat_plans)
    assert stats["active_layout_count"] == 0
    store.resolve((1, 1))
    assert store.plan_statistics()["active_layout_count"] == 1
