"""Exactness tests for compiled partially symmetrized base-plan maps."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import tensordev.core.bigraded.symmetrized.plans as plans_module
from tensordev.core.bigraded.symmetrized._compiled import _binomial_table
from tensordev.core.bigraded.symmetrized._compiled_plans import (
    compile_concatenation_targets,
    compile_doubleprime_generator_targets,
    compile_placement_array,
)
from tensordev.core.bigraded.symmetrized.combinatorics import (
    _rank_normalized_blocks,
    multiset_placements,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
)
from tensordev.core.utils.precompute import _unsigned_index_dtype


def _placements(d_doubleprime, grade):
    n, m = grade
    return np.asarray(
        multiset_placements(d_doubleprime, grade),
        dtype=_unsigned_index_dtype(m),
    ).reshape(-1, n + 1, d_doubleprime)


def _rank_table(d_doubleprime, output_grade):
    n, m = output_grade
    parts = (n + 1) * d_doubleprime
    return _binomial_table(
        max(parts - 2 + m, 0),
        max_column=max(parts - 1, 0),
        max_complement=m,
    )


def _normal_form(row):
    return tuple(tuple(map(int, block)) for block in row)


@pytest.mark.parametrize(
    ("d_doubleprime", "grade"),
    ((1, (0, 0)), (1, (4, 3)), (2, (3, 4)), (3, (2, 3))),
)
def test_compiled_placements_match_canonical_tuple_enumeration(
    d_doubleprime,
    grade,
):
    actual = compile_placement_array(d_doubleprime, grade)
    expected = _placements(d_doubleprime, grade)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert not actual.flags.writeable


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.parametrize(
    ("d_doubleprime", "left_grade", "right_grade"),
    (
        (1, (0, 0), (0, 0)),
        (2, (2, 2), (1, 1)),
        (3, (1, 2), (2, 1)),
        (2, (0, 3), (3, 0)),
    ),
)
def test_concatenation_targets_match_normal_form_oracle(
    compiled,
    d_doubleprime,
    left_grade,
    right_grade,
):
    left = _placements(d_doubleprime, left_grade)
    right = _placements(d_doubleprime, right_grade)
    output_grade = (
        left_grade[0] + right_grade[0],
        left_grade[1] + right_grade[1],
    )
    output = _placements(d_doubleprime, output_grade)
    actual = compile_concatenation_targets(
        left,
        right,
        _rank_table(d_doubleprime, output_grade),
        output_rank_count=output.shape[0],
        compiled=compiled,
    )
    expected = []
    for left_row in left:
        alpha = _normal_form(left_row)
        for right_row in right:
            beta = _normal_form(right_row)
            merged = (
                alpha[:-1]
                + (tuple(a + b for a, b in zip(alpha[-1], beta[0])),)
                + beta[1:]
            )
            expected.append(_rank_normalized_blocks(merged))

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.dtype(
        _unsigned_index_dtype(output.shape[0] - 1)
    )
    assert not actual.flags.writeable


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.parametrize(
    ("d_doubleprime", "output_grade"),
    ((1, (0, 1)), (2, (3, 3)), (3, (2, 2))),
)
def test_doubleprime_generator_targets_match_normal_form_oracle(
    compiled,
    d_doubleprime,
    output_grade,
):
    n, m = output_grade
    source_grade = n, m - 1
    source = _placements(d_doubleprime, source_grade)
    output = _placements(d_doubleprime, output_grade)
    actual = compile_doubleprime_generator_targets(
        source,
        _rank_table(d_doubleprime, output_grade),
        output_rank_count=output.shape[0],
        compiled=compiled,
    )
    expected = []
    for letter in range(d_doubleprime):
        row = []
        for source_row in source:
            blocks = [list(block) for block in _normal_form(source_row)]
            blocks[-1][letter] += 1
            row.append(
                _rank_normalized_blocks(
                    tuple(tuple(block) for block in blocks)
                )
            )
        expected.append(row)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.dtype(
        _unsigned_index_dtype(output.shape[0] - 1)
    )
    assert not actual.flags.writeable


@pytest.mark.parametrize("builder", ("concatenation", "generator"))
def test_compiled_rank_maps_reject_negative_placements(builder):
    placements = np.asarray([[[0, -1]]], dtype=np.int64)
    table = _binomial_table(1)

    with pytest.raises(ValueError, match="non-negative"):
        if builder == "concatenation":
            compile_concatenation_targets(
                placements,
                placements,
                table,
                output_rank_count=1,
                compiled=True,
            )
        else:
            compile_doubleprime_generator_targets(
                placements,
                table,
                output_rank_count=1,
                compiled=True,
            )


def test_compiled_and_python_stores_have_identical_payloads(monkeypatch):
    monkeypatch.setattr(
        plans_module,
        "_COMPILED_PLAN_ENTRY_THRESHOLD",
        10**100,
    )
    python = PartiallySymmetrizedPlanStore((2, 3), (3, 3))
    assert not python._use_compiled_plan_builder

    monkeypatch.setattr(
        plans_module,
        "_COMPILED_PLAN_ENTRY_THRESHOLD",
        0,
    )
    compiled = PartiallySymmetrizedPlanStore((2, 3), (3, 3))
    assert compiled._use_compiled_plan_builder

    assert tuple(compiled.grade_plans) == tuple(python.grade_plans)
    assert tuple(compiled.concat_plans) == tuple(python.concat_plans)
    assert (
        tuple(compiled.doubleprime_generator_plans)
        == tuple(python.doubleprime_generator_plans)
    )
    assert (
        compiled.memory_bytes_by_category()
        == python.memory_bytes_by_category()
    )
    for grade, expected in python.grade_plans.items():
        actual = compiled.grade_plan(grade)
        np.testing.assert_array_equal(actual.placements, expected.placements)
        assert actual.placements.dtype == expected.placements.dtype
        assert not actual.placements.flags.writeable
    for grades, expected in python.concat_plans.items():
        actual = compiled.concat_plan(*grades)
        np.testing.assert_array_equal(
            actual.rank_plan.target_ranks,
            expected.rank_plan.target_ranks,
        )
        assert (
            actual.rank_plan.target_ranks.dtype
            == expected.rank_plan.target_ranks.dtype
        )
        assert not actual.rank_plan.target_ranks.flags.writeable
    for grade, expected in python.doubleprime_generator_plans.items():
        actual = compiled.doubleprime_generator_plan(grade)
        np.testing.assert_array_equal(actual.target_ranks, expected.target_ranks)
        assert actual.target_ranks.dtype == expected.target_ranks.dtype
        assert not actual.target_ranks.flags.writeable


@pytest.mark.parametrize("compiled", (False, True))
def test_zero_doubleprime_capacity_builds_all_base_plans(
    monkeypatch,
    compiled,
):
    monkeypatch.setattr(
        plans_module,
        "_COMPILED_PLAN_ENTRY_THRESHOLD",
        0 if compiled else 10**100,
    )
    store = PartiallySymmetrizedPlanStore((2, 2), (4, 0))

    assert store._use_compiled_plan_builder is compiled
    assert not store.doubleprime_generator_plans
    for plan in store.grade_plans.values():
        assert plan.rank_count == 1
        assert not plan.placements.flags.writeable
    for plan in store.concat_plans.values():
        np.testing.assert_array_equal(plan.rank_plan.target_ranks, [0])
        assert not plan.rank_plan.target_ranks.flags.writeable


def test_small_store_does_not_cold_compile_emitters():
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    code = """
from tensordev.core.bigraded.symmetrized._compiled import _rank_blocks
from tensordev.core.bigraded.symmetrized._compiled_plans import (
    _build_placement_array,
    _emit_concatenation_targets,
    _emit_doubleprime_generator_targets,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
)
dispatchers = (
    _rank_blocks,
    _build_placement_array,
    _emit_concatenation_targets,
    _emit_doubleprime_generator_targets,
)
assert all(not dispatcher.signatures for dispatcher in dispatchers)
store = PartiallySymmetrizedPlanStore((1, 2), (1, 1))
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


def test_disabled_numba_jit_compiled_store_is_warning_free():
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["NUMBA_DISABLE_JIT"] = "1"
    environment["PYTHONPATH"] = str(repository / "src")
    code = """
import tensordev.core.bigraded.symmetrized.plans as plans
plans._COMPILED_PLAN_ENTRY_THRESHOLD = 0
store = plans.PartiallySymmetrizedPlanStore((1, 2), (2, 2))
assert store._use_compiled_plan_builder
assert store.concat_plans
assert store.doubleprime_generator_plans
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
