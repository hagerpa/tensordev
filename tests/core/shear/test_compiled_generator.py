"""Exactness contracts for ordered-bidegree shear generator plans."""

from __future__ import annotations

import os
from math import comb
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import tensordev.core.shear.bigraded as generator_module
import tensordev.core.shear.symbolic as symbolic_module
from tensordev.core.bigraded.precompute import (
    BigradedPlanStore,
    colex_placements,
    colex_rank,
)
from tensordev.core.shear._compiled_generator import (
    BigradedShearGeneratorSupport,
    compile_bigraded_shear_generator_support,
)
from tensordev.core.shear.bigraded import (
    BigradedShearPlanStore,
    _expected_bigraded_shear_plan_memory_bytes_by_category,
)
from tensordev.core.shear.symbolic import (
    complement_positions,
    right_generator_pattern_support,
)
from tensordev.core.utils.precompute import _unsigned_index_dtype


def _packed_generator_permutation(term, grade):
    n, m = grade
    degree = n + m
    source_degree = degree - 1
    source_prime_positions = term.input_prime_positions
    source_doubleprime_positions = complement_positions(
        source_degree,
        source_prime_positions,
    )
    ordinary_to_packed = {
        position: axis
        for axis, position in enumerate(source_prime_positions)
    }
    ordinary_to_packed.update(
        {
            position: len(source_prime_positions) + axis
            for axis, position in enumerate(source_doubleprime_positions)
        }
    )
    ordinary_to_packed[source_degree] = source_degree
    output_doubleprime_positions = complement_positions(
        degree,
        term.output_prime_positions,
    )
    output_word_axes = (
        term.output_prime_positions + output_doubleprime_positions
    )
    return tuple(
        ordinary_to_packed[term.dense_permutation[position]]
        for position in output_word_axes
    )


def _symbolic_support(grade):
    n, m = grade
    degree = n + m
    output_placements = colex_placements(degree, n)
    output_count = len(output_placements)
    prime_source_count = comb(degree - 1, n - 1) if n else 1
    doubleprime_source_ranks = []
    doubleprime_output_ranks = []
    prime_groups = {}
    for output_rank, placement in enumerate(output_placements):
        for term in right_generator_pattern_support(placement, degree):
            if term.input_prime_positions == placement:
                doubleprime_source_ranks.append(
                    colex_rank(term.input_prime_positions)
                )
                doubleprime_output_ranks.append(output_rank)
                continue
            source_rank = colex_rank(term.input_prime_positions)
            permutation = _packed_generator_permutation(term, grade)
            prime_groups.setdefault(permutation, []).append(
                output_rank * prime_source_count + source_rank
            )
    rank_dtype = _unsigned_index_dtype(max(output_count - 1, 0))
    pair_dtype = _unsigned_index_dtype(
        max(output_count * prime_source_count - 1, 0)
    )
    permutation_dtype = _unsigned_index_dtype(max(degree - 1, 0))
    return (
        np.asarray(doubleprime_source_ranks, dtype=rank_dtype),
        np.asarray(doubleprime_output_ranks, dtype=rank_dtype),
        tuple(
            (
                permutation,
                np.asarray(encoded, dtype=pair_dtype),
            )
            for permutation, encoded in prime_groups.items()
        ),
        permutation_dtype,
    )


def _compiled_groups(support: BigradedShearGeneratorSupport):
    return tuple(
        (
            tuple(map(int, permutation)),
            support.prime_encoded_rank_pairs[
                support.prime_group_slice(group)
            ],
        )
        for group, permutation in enumerate(
            support.prime_dense_permutations
        )
    )


@pytest.mark.parametrize(
    "grade",
    [
        (n, m)
        for n in range(5)
        for m in range(5)
        if 0 < n + m <= 7
    ]
    + [(5, 3), (5, 4), (10, 4)],
)
def test_compiler_is_byte_exact_against_symbolic_oracle(grade):
    expected_double_source, expected_double_output, expected_groups, dtype = (
        _symbolic_support(grade)
    )
    placements = np.asarray(colex_placements(sum(grade), grade[0]))

    for compiled in (False, True):
        actual = compile_bigraded_shear_generator_support(
            placements,
            grade,
            compiled=compiled,
        )
        for array, expected in (
            (actual.doubleprime_source_ranks, expected_double_source),
            (actual.doubleprime_output_ranks, expected_double_output),
        ):
            assert array.dtype == expected.dtype
            assert array.shape == expected.shape
            assert array.tobytes() == expected.tobytes()
            assert not array.flags.writeable
        actual_groups = _compiled_groups(actual)
        assert len(actual_groups) == len(expected_groups)
        assert actual.prime_dense_permutations.dtype == dtype
        for (permutation, pairs), (expected_permutation, expected_pairs) in zip(
            actual_groups,
            expected_groups,
        ):
            assert permutation == expected_permutation
            assert pairs.dtype == expected_pairs.dtype
            assert pairs.tobytes() == expected_pairs.tobytes()
        assert not actual.prime_encoded_rank_pairs.flags.writeable
        assert not actual.prime_dense_permutations.flags.writeable
        assert not actual.prime_group_offsets.flags.writeable


def test_compiled_and_factorized_python_stores_are_byte_exact(monkeypatch):
    base = BigradedPlanStore((1, 2), (4, 3))
    monkeypatch.setattr(
        generator_module,
        "_COMPILED_SHEAR_GENERATOR_ENTRY_THRESHOLD",
        0,
    )
    compiled = BigradedShearPlanStore(base)
    monkeypatch.setattr(
        generator_module,
        "_COMPILED_SHEAR_GENERATOR_ENTRY_THRESHOLD",
        sys.maxsize,
    )
    python = BigradedShearPlanStore(base)

    assert compiled.generator_plans.keys() == python.generator_plans.keys()
    assert compiled.memory_bytes_by_category() == python.memory_bytes_by_category()
    for grade, compiled_plan in compiled.generator_plans.items():
        python_plan = python.generator_plans[grade]
        assert compiled_plan.output_grade == python_plan.output_grade
        assert compiled_plan.source_prime_grade == python_plan.source_prime_grade
        for left, right in (
            (
                compiled_plan.doubleprime_source_ranks,
                python_plan.doubleprime_source_ranks,
            ),
            (
                compiled_plan.doubleprime_output_ranks,
                python_plan.doubleprime_output_ranks,
            ),
        ):
            assert left.dtype == right.dtype
            assert left.tobytes() == right.tobytes()
        assert len(compiled_plan.prime_key_plans) == len(
            python_plan.prime_key_plans
        )
        for left, right in zip(
            compiled_plan.prime_key_plans,
            python_plan.prime_key_plans,
        ):
            assert left.dense_axis_permutation == right.dense_axis_permutation
            assert left.encoded_rank_pairs.dtype == right.encoded_rank_pairs.dtype
            assert (
                left.encoded_rank_pairs.tobytes()
                == right.encoded_rank_pairs.tobytes()
            )


def test_target_grade_has_exact_compact_payload_and_avoids_cold_jit():
    grade = (10, 4)
    placements = np.asarray(colex_placements(sum(grade), grade[0]))
    support = compile_bigraded_shear_generator_support(
        placements,
        grade,
        compiled=False,
    )

    assert support.output_count == 1_001
    assert support.prime_source_count == 715
    assert support.doubleprime_source_ranks.shape == (286,)
    assert support.doubleprime_output_ranks.shape == (286,)
    assert support.prime_encoded_rank_pairs.shape == (1_093,)
    assert support.prime_dense_permutations.shape == (12, 14)
    assert support.prime_group_offsets.shape == (13,)
    assert support.doubleprime_source_ranks.dtype == np.uint16
    assert support.doubleprime_output_ranks.dtype == np.uint16
    assert support.prime_encoded_rank_pairs.dtype == np.uint32
    assert support.prime_dense_permutations.dtype == np.uint8
    assert support.prime_group_offsets.dtype == np.int64
    assert support.prime_group_offsets[0] == 0
    assert support.prime_group_offsets[-1] == 1_093

    store = BigradedShearPlanStore(
        BigradedPlanStore((1, 2), grade)
    )
    assert not store._use_compiled_generator_builder
    plan = store.generator_plan(grade)
    assert len(plan.prime_key_plans) == 12
    assert sum(
        key.encoded_rank_pairs.size for key in plan.prime_key_plans
    ) == 1_093


def test_store_memory_matches_formula_estimator():
    dims = (1, 2)
    truncation = (4, 3)
    store = BigradedShearPlanStore(BigradedPlanStore(dims, truncation))
    expected = _expected_bigraded_shear_plan_memory_bytes_by_category(
        dims,
        truncation,
    )

    assert dict(store.memory_bytes_by_category()) == dict(expected)
    assert store.memory_bytes() == sum(expected.values())


def test_store_does_not_call_symbolic_generator_builder(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("symbolic generator construction was called")

    monkeypatch.setattr(
        symbolic_module,
        "right_generator_pattern_support",
        forbidden,
    )
    store = BigradedShearPlanStore(
        BigradedPlanStore((1, 2), (3, 2))
    )
    assert store.generator_plans


def test_small_store_does_not_cold_compile_generator_emitter():
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    code = """
from tensordev.core.bigraded.precompute import BigradedPlanStore
from tensordev.core.shear._compiled_generator import (
    _emit_prime_generator_stream,
)
from tensordev.core.shear.bigraded import BigradedShearPlanStore
assert not _emit_prime_generator_stream.signatures
store = BigradedShearPlanStore(BigradedPlanStore((1, 2), (2, 2)))
assert not store._use_compiled_generator_builder
assert not _emit_prime_generator_stream.signatures
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


def test_disabled_numba_jit_compiled_store_is_warning_free():
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["NUMBA_DISABLE_JIT"] = "1"
    environment["PYTHONPATH"] = str(repository / "src")
    code = """
import tensordev.core.shear.bigraded as module
from tensordev.core.bigraded.precompute import BigradedPlanStore
module._COMPILED_SHEAR_GENERATOR_ENTRY_THRESHOLD = 0
store = module.BigradedShearPlanStore(
    BigradedPlanStore((1, 2), (2, 2))
)
assert store._use_compiled_generator_builder
assert store.generator_plans
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
