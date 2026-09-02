from __future__ import annotations

from itertools import product

import numpy as np
import pytest

import tensordev.core.bigraded.precompute as precompute_module
from tensordev.core.bigraded.precompute import (
    BigradedPlanStore,
    colex_placements,
)


def _ordinary_index(word, base):
    value = 0
    for letter in word:
        value = value * base + letter
    return value


def _conversion_oracle(dims, grade):
    d_prime, d_doubleprime = dims
    n, m = grade
    targets = []
    for placement in colex_placements(n + m, n):
        prime_positions = frozenset(placement)
        for prime_word in product(range(d_prime), repeat=n):
            for doubleprime_word in product(range(d_doubleprime), repeat=m):
                prime = 0
                doubleprime = 0
                word = []
                for position in range(n + m):
                    if position in prime_positions:
                        word.append(prime_word[prime])
                        prime += 1
                    else:
                        word.append(d_prime + doubleprime_word[doubleprime])
                        doubleprime += 1
                targets.append(_ordinary_index(word, sum(dims)))
    return targets


@pytest.mark.parametrize(
    ("dims", "grade"),
    (
        ((1, 1), (0, 0)),
        ((2, 3), (0, 3)),
        ((3, 2), (3, 0)),
        ((2, 3), (2, 2)),
        ((1, 2), (4, 3)),
    ),
)
def test_vectorized_conversion_map_is_byte_exact(dims, grade):
    plan = BigradedPlanStore(dims, grade).grade_plan(grade)
    expected = np.asarray(
        _conversion_oracle(dims, grade),
        dtype=plan.block_to_total_indices.dtype,
    )

    assert plan.block_to_total_indices.shape == expected.shape
    assert plan.block_to_total_indices.dtype == expected.dtype
    assert plan.block_to_total_indices.tobytes() == expected.tobytes()
    assert not plan.block_to_total_indices.flags.writeable
    assert not plan.placements.flags.writeable


def test_large_capacity_preserves_exact_payload_and_boundary_map():
    store = BigradedPlanStore((1, 2), (10, 4))
    plan = store.grade_plan((10, 4))

    assert store.memory_bytes_by_category() == {
        "placements": 142_692,
        "conversion": 229_372,
        "concatenation": 73_964,
    }
    assert plan.block_to_total_indices.shape == (16_016,)
    assert plan.block_to_total_indices.dtype == np.int32
    assert np.unique(plan.block_to_total_indices).size == 16_016
    assert not plan.block_to_total_indices.flags.writeable


def test_digit_tables_are_reused_only_within_one_store(monkeypatch):
    calls = []
    original = precompute_module._base_digit_matrix

    def counted(base, length, dtype):
        calls.append((base, length, np.dtype(dtype)))
        return original(base, length, dtype)

    monkeypatch.setattr(precompute_module, "_base_digit_matrix", counted)
    store = BigradedPlanStore((1, 2), (10, 4))

    assert store.grade_plans
    assert len(calls) == 16
    assert len(set(calls)) == len(calls)
    assert not hasattr(store, "_digit_tables")
