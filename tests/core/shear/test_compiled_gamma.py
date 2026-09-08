"""Exactness tests for compiled ordered shear-shuffle support."""

from __future__ import annotations

import os
from math import comb
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import tensordev.core.shear.bigraded as bigraded_shear_module
from tensordev.core.bigraded.precompute import (
    BigradedPlanStore,
    colex_placements,
)
from tensordev.core.bigraded.shuffle import _dense_axis_permutation
from tensordev.core.shear._compiled_gamma import (
    ShearShuffleSupport,
    _ShearShuffleWorkspace,
    _binomial_table,
    _colex_output_pair_indices,
    _shear_shuffle_group_count,
    _shear_shuffle_term_count,
    compile_shear_shuffle_support,
    shear_shuffle_colex_output_pair_indices,
    shear_shuffle_dense_axis_permutations,
)
from tensordev.core.shear.bigraded import (
    BigradedShearShufflePlanStore,
    _expected_bigraded_gamma_memory_bytes_by_category,
    _gamma_rank_groups,
)
from tensordev.core.shear.symbolic import gamma_shuffle_pattern_support
from tensordev.core.utils.precompute import _unsigned_index_dtype


def _mask(positions) -> int:
    return sum(1 << position for position in positions)


def _symbolic_primitive_groups(left_grade, right_grade):
    n1, m1 = left_grade
    n2, m2 = right_grade
    grouped = {}
    for left_placement in colex_placements(n1 + m1, n1):
        for right_placement in colex_placements(n2 + m2, n2):
            for term in gamma_shuffle_pattern_support(
                n1 + m1,
                n2 + m2,
                left_placement,
                right_placement,
            ):
                permutation = _dense_axis_permutation(
                    left_grade,
                    right_grade,
                    term.prime_interleaving,
                    term.doubleprime_interleaving,
                )
                grouped.setdefault(permutation, []).append(
                    (
                        _mask(term.output_prime_positions),
                        _mask(left_placement),
                        _mask(right_placement),
                    )
                )
    return tuple(grouped.items())


def _compiled_primitive_groups(support):
    permutations = shear_shuffle_dense_axis_permutations(
        support, compiled=False
    )
    return tuple(
        (
            tuple(map(int, permutations[group])),
            list(
                zip(
                    map(int, support.output_masks[support.group_slice(group)]),
                    map(
                        int,
                        support.left_input_masks[
                            support.group_slice(group)
                        ],
                    ),
                    map(
                        int,
                        support.right_input_masks[
                            support.group_slice(group)
                        ],
                    ),
                )
            ),
        )
        for group in range(support.group_count)
    )


@pytest.mark.parametrize(
    ("left_grade", "right_grade"),
    (
        ((0, 0), (0, 0)),
        ((0, 3), (0, 2)),
        ((3, 0), (2, 0)),
        ((2, 1), (1, 2)),
        ((2, 2), (1, 1)),
    ),
)
def test_compiled_primitive_support_exactly_matches_symbolic_encounter_order(
    left_grade,
    right_grade,
):
    support = compile_shear_shuffle_support(
        *left_grade, *right_grade, compiled=False
    )

    assert _compiled_primitive_groups(support) == (
        _symbolic_primitive_groups(left_grade, right_grade)
    )


@pytest.mark.parametrize(
    ("left_grade", "right_grade"),
    tuple(
        ((n1, m1), (n2, total - n1 - m1 - n2))
        for total in range(6)
        for n1 in range(total + 1)
        for m1 in range(total - n1 + 1)
        for n2 in range(total - n1 - m1 + 1)
    ),
)
def test_compiled_bidegree_groups_exactly_match_symbolic_rank_groups(
    left_grade,
    right_grade,
):
    support = compile_shear_shuffle_support(
        *left_grade, *right_grade, compiled=False
    )
    encoded = shear_shuffle_colex_output_pair_indices(
        support, compiled=False
    )
    permutations = shear_shuffle_dense_axis_permutations(
        support, compiled=False
    )
    actual = {
        tuple(map(int, permutations[group])): encoded[
            support.group_slice(group)
        ].tolist()
        for group in range(support.group_count)
    }
    output_grade, output_count, pair_count, expected = _gamma_rank_groups(
        left_grade, right_grade
    )

    assert output_grade == support.output_grade
    assert output_count == support.output_placement_count
    assert pair_count == support.pair_count
    assert actual == expected
    assert support.term_count == _shear_shuffle_term_count(
        left_grade, right_grade
    )
    assert support.group_count == _shear_shuffle_group_count(
        left_grade, right_grade
    )


def test_compiled_and_python_emitters_produce_identical_minimal_tables():
    python = compile_shear_shuffle_support(2, 2, 2, 1, compiled=False)
    compiled = compile_shear_shuffle_support(2, 2, 2, 1, compiled=True)

    for name in (
        "output_masks",
        "left_input_masks",
        "right_input_masks",
        "prime_group_keys",
        "doubleprime_group_keys",
        "group_offsets",
    ):
        left = getattr(python, name)
        right = getattr(compiled, name)
        np.testing.assert_array_equal(left, right)
        assert left.dtype == right.dtype


def test_store_local_workspace_reuses_primitive_tables_without_changing_support():
    workspace = _ShearShuffleWorkspace()
    first = compile_shear_shuffle_support(
        2,
        2,
        1,
        1,
        compiled=False,
        _workspace=workspace,
    )
    second = compile_shear_shuffle_support(
        1,
        1,
        2,
        2,
        compiled=False,
        _workspace=workspace,
    )
    expected = compile_shear_shuffle_support(1, 1, 2, 2, compiled=False)

    assert len(workspace._masks) == 1
    assert len(workspace._binomials) == 1
    for name in (
        "output_masks",
        "left_input_masks",
        "right_input_masks",
        "prime_group_keys",
        "doubleprime_group_keys",
        "group_offsets",
    ):
        np.testing.assert_array_equal(getattr(second, name), getattr(expected, name))
    assert first.output_grade == second.output_grade


def test_large_compiled_support_contract_and_exact_counts():
    support = compile_shear_shuffle_support(5, 2, 5, 2)

    assert isinstance(support, ShearShuffleSupport)
    assert support.term_count == 123_732
    assert support.group_count == 1_512
    assert support.group_offsets[0] == 0
    assert support.group_offsets[-1] == support.term_count
    assert np.all(np.diff(support.group_offsets.astype(np.int64)) > 0)
    for array in (
        support.output_masks,
        support.left_input_masks,
        support.right_input_masks,
        support.prime_group_keys,
        support.doubleprime_group_keys,
        support.group_offsets,
    ):
        assert not array.flags.writeable
    for array in (
        support.output_masks,
        support.left_input_masks,
        support.right_input_masks,
    ):
        assert array.dtype == np.dtype(
            _unsigned_index_dtype(int(np.max(array, initial=0)))
        )
    assert support.group_offsets.dtype == np.dtype(
        _unsigned_index_dtype(support.term_count)
    )
    assert support.memory_bytes() == sum(
        array.nbytes
        for array in (
            support.output_masks,
            support.left_input_masks,
            support.right_input_masks,
            support.prime_group_keys,
            support.doubleprime_group_keys,
            support.group_offsets,
        )
    )


def test_rank_encoding_retains_values_above_signed_int64():
    output_mask = ((1 << 64) - 1) ^ ((1 << 32) - 1)
    arguments = (
        np.asarray((output_mask,), dtype=np.uint64),
        np.zeros(1, dtype=np.uint64),
        np.zeros(1, dtype=np.uint64),
        _binomial_table(64),
        32,
        32,
        0,
        0,
        1,
        10,
    )
    expected = (comb(64, 32) - 1) * 10
    assert expected > np.iinfo(np.int64).max
    assert expected <= np.iinfo(np.uint64).max
    for emitter in (
        _colex_output_pair_indices.py_func,
        _colex_output_pair_indices,
    ):
        assert int(emitter(*arguments)[0]) == expected


def test_bigraded_shuffle_construction_never_expands_symbolic_gamma(
    monkeypatch,
):
    def forbidden_symbolic_fallback(*args, **kwargs):
        raise AssertionError("symbolic shear-shuffle support was expanded")

    monkeypatch.setattr(
        bigraded_shear_module,
        "_gamma_rank_groups",
        forbidden_symbolic_fallback,
    )
    store = BigradedShearShufflePlanStore(
        BigradedPlanStore((1, 2), (4, 3))
    )

    assert store.block_plans


@pytest.mark.parametrize("scope", ("generator", "full"))
def test_gamma_memory_estimator_is_allocation_free_and_exact(
    monkeypatch,
    scope,
):
    def forbidden_symbolic_fallback(*args, **kwargs):
        raise AssertionError("memory estimation expanded symbolic support")

    monkeypatch.setattr(
        bigraded_shear_module,
        "_gamma_rank_groups",
        forbidden_symbolic_fallback,
    )
    plan_store = BigradedPlanStore((1, 2), (3, 2))
    actual = BigradedShearShufflePlanStore(plan_store, scope=scope)
    expected = _expected_bigraded_gamma_memory_bytes_by_category(
        (1, 2), (3, 2), scope=scope
    )

    assert dict(expected) == actual.memory_bytes_by_category()


@pytest.mark.parametrize(
    ("args", "error", "match"),
    (
        ((True, 0, 0, 0), TypeError, "n1"),
        ((-1, 0, 0, 0), ValueError, "n1"),
        ((65, 0, 0, 0), ValueError, "degree <= 64"),
        ((16, 16, 16, 16), OverflowError, "plan limit"),
    ),
)
def test_compiled_support_rejects_invalid_or_unrepresentable_grades(
    args,
    error,
    match,
):
    with pytest.raises(error, match=match):
        compile_shear_shuffle_support(*args)


def test_disabled_numba_jit_small_store_is_warning_free():
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["NUMBA_DISABLE_JIT"] = "1"
    environment["PYTHONPATH"] = str(repository / "src")
    code = """
from tensordev.core.bigraded.precompute import BigradedPlanStore
from tensordev.core.shear.bigraded import BigradedShearShufflePlanStore
store = BigradedShearShufflePlanStore(BigradedPlanStore((1, 2), (2, 2)))
assert store.block_plans
"""
    completed = subprocess.run(
        [sys.executable, "-W", "error", "-c", code],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_small_store_does_not_cold_compile_numba_emitters():
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    code = """
from tensordev.core.bigraded.precompute import BigradedPlanStore
from tensordev.core.shear._compiled_gamma import (
    _colex_output_pair_indices,
    _dense_axis_permutations,
    _emit_shear_shuffle_support,
)
from tensordev.core.shear.bigraded import BigradedShearShufflePlanStore
dispatchers = (
    _emit_shear_shuffle_support,
    _dense_axis_permutations,
    _colex_output_pair_indices,
)
assert all(not dispatcher.signatures for dispatcher in dispatchers)
store = BigradedShearShufflePlanStore(BigradedPlanStore((1, 2), (2, 2)))
assert not store._use_compiled_plan_builder
assert all(not dispatcher.signatures for dispatcher in dispatchers)
"""
    completed = subprocess.run(
        [sys.executable, "-W", "error", "-c", code],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
