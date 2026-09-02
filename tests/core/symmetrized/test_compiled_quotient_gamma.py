from __future__ import annotations

import os
from itertools import combinations
from math import comb
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import tensordev.core.bigraded.symmetrized.gamma as gamma_module
from tensordev.core.bigraded.symmetrized._compiled_gamma import (
    PartiallySymmetrizedGammaWorkspace,
    compile_partially_symmetrized_shear_shuffle_support,
)
from tensordev.core.bigraded.symmetrized._precompute import (
    _coefficient_dtype,
    _placement_tuple,
)
from tensordev.core.bigraded.symmetrized.combinatorics import (
    _rank_normalized_blocks,
)
from tensordev.core.bigraded.symmetrized.gamma import (
    PartiallySymmetrizedShearShufflePlanStore,
    _expected_gamma_memory_bytes_by_category,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
)
from tensordev.core.utils.precompute import _unsigned_index_dtype


def _interleave(left, right, left_positions):
    positions = frozenset(left_positions)
    left_values = iter(left)
    right_values = iter(right)
    return tuple(
        next(left_values) if position in positions else next(right_values)
        for position in range(len(left) + len(right))
    )


def _symbolic_support(base, left_grade, right_grade):
    left = tuple(
        _placement_tuple(row)
        for row in base.grade_plan(left_grade).placements
    )
    right = tuple(
        _placement_tuple(row)
        for row in base.grade_plan(right_grade).placements
    )
    output_grade = (
        left_grade[0] + right_grade[0],
        left_grade[1] + right_grade[1],
    )
    positions = tuple(
        combinations(range(output_grade[0]), left_grade[0])
    )
    targets = []
    coefficients = []
    for alpha in left:
        for beta in right:
            coefficient = 1
            for left_value, right_value in zip(alpha[-1], beta[-1]):
                coefficient *= comb(
                    left_value + right_value, left_value
                )
            coefficients.append(coefficient)
    for key in positions:
        key_targets = []
        for alpha in left:
            for beta in right:
                terminal = tuple(
                    left_value + right_value
                    for left_value, right_value in zip(alpha[-1], beta[-1])
                )
                placement = _interleave(alpha[:-1], beta[:-1], key) + (
                    terminal,
                )
                key_targets.append(_rank_normalized_blocks(placement))
        targets.append(key_targets)
    output_rank_count = base.grade_plan(output_grade).rank_count
    return (
        np.asarray(positions, dtype=np.int64).reshape(
            len(positions), left_grade[0]
        ),
        np.asarray(
            targets,
            dtype=_unsigned_index_dtype(output_rank_count - 1),
        ),
        np.asarray(
            coefficients,
            dtype=_coefficient_dtype(
                comb(output_grade[1], left_grade[1])
            ),
        ),
    )


@pytest.mark.parametrize("q", (1, 2, 3))
@pytest.mark.parametrize(
    ("left_grade", "right_grade"),
    (
        ((0, 0), (0, 0)),
        ((0, 2), (1, 1)),
        ((2, 0), (1, 1)),
        ((1, 2), (2, 1)),
        ((2, 2), (1, 0)),
    ),
)
def test_compiled_support_is_byte_exact_against_symbolic_oracle(
    q,
    left_grade,
    right_grade,
):
    capacity = (
        left_grade[0] + right_grade[0],
        left_grade[1] + right_grade[1],
    )
    base = PartiallySymmetrizedPlanStore((1, q), capacity)
    expected = _symbolic_support(base, left_grade, right_grade)

    for compiled in (False, True):
        actual = compile_partially_symmetrized_shear_shuffle_support(
            base.grade_plan(left_grade).placements,
            base.grade_plan(right_grade).placements,
            left_grade,
            right_grade,
            compiled=compiled,
        )
        for array, oracle in zip(
            (
                actual.left_prime_positions,
                actual.target_ranks,
                actual.coefficients,
            ),
            expected,
        ):
            assert array.dtype == oracle.dtype
            assert array.shape == oracle.shape
            assert array.tobytes() == oracle.tobytes()
            assert not array.flags.writeable


def test_python_and_compiled_plan_adapters_are_byte_exact(monkeypatch):
    base = PartiallySymmetrizedPlanStore((1, 2), (3, 2))
    monkeypatch.setattr(
        gamma_module, "_COMPILED_GAMMA_ENTRY_THRESHOLD", 0
    )
    compiled = PartiallySymmetrizedShearShufflePlanStore(base, "full")
    monkeypatch.setattr(
        gamma_module, "_COMPILED_GAMMA_ENTRY_THRESHOLD", sys.maxsize
    )
    python = PartiallySymmetrizedShearShufflePlanStore(base, "full")

    assert compiled.block_plans.keys() == python.block_plans.keys()
    for pair, compiled_plan in compiled.block_plans.items():
        python_plan = python.block_plans[pair]
        assert compiled_plan.coefficients.dtype == python_plan.coefficients.dtype
        assert (
            compiled_plan.coefficients.tobytes()
            == python_plan.coefficients.tobytes()
        )
        assert len(compiled_plan.key_plans) == len(python_plan.key_plans)
        for compiled_key, python_key in zip(
            compiled_plan.key_plans, python_plan.key_plans
        ):
            assert (
                compiled_key.dense_axis_permutation
                == python_key.dense_axis_permutation
            )
            left = compiled_key.rank_plan.target_ranks
            right = python_key.rank_plan.target_ranks
            assert left.dtype == right.dtype
            assert left.tobytes() == right.tobytes()


def test_store_local_workspace_reuses_primitive_tables():
    base = PartiallySymmetrizedPlanStore((1, 2), (3, 3))
    workspace = PartiallySymmetrizedGammaWorkspace()
    supports = []
    for left_grade, right_grade in (((2, 1), (1, 2)), ((2, 2), (1, 1))):
        supports.append(
            compile_partially_symmetrized_shear_shuffle_support(
                base.grade_plan(left_grade).placements,
                base.grade_plan(right_grade).placements,
                left_grade,
                right_grade,
                compiled=False,
                _workspace=workspace,
            )
        )

    assert len(workspace._prime_tables) == 1
    assert len(workspace._rank_tables) == 1
    assert len(workspace._coefficient_tables) == 2
    assert supports[0].output_grade == supports[1].output_grade


def test_former_worst_block_has_exact_shape_and_compact_storage():
    base = PartiallySymmetrizedPlanStore((1, 2), (10, 4))
    left_grade = right_grade = (5, 2)
    support = compile_partially_symmetrized_shear_shuffle_support(
        base.grade_plan(left_grade).placements,
        base.grade_plan(right_grade).placements,
        left_grade,
        right_grade,
    )

    assert support.target_ranks.shape == (252, 6_084)
    assert support.target_ranks.size == 1_533_168
    assert support.target_ranks.dtype == np.uint16
    assert support.coefficients.shape == (6_084,)
    assert support.coefficients.dtype == np.uint8


@pytest.mark.parametrize("scope", ("generator", "full"))
def test_memory_estimator_matches_compiled_store(scope):
    base = PartiallySymmetrizedPlanStore((1, 2), (3, 2))
    store = PartiallySymmetrizedShearShufflePlanStore(base, scope)

    assert dict(
        _expected_gamma_memory_bytes_by_category(
            (1, 2), (3, 2), scope=scope
        )
    ) == dict(store.memory_bytes_by_category())


def test_small_store_does_not_cold_compile_numba_emitter():
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    code = """
from tensordev.core.bigraded.symmetrized._compiled import _rank_blocks
from tensordev.core.bigraded.symmetrized._compiled_gamma import _emit_target_maps
from tensordev.core.bigraded.symmetrized.gamma import (
    PartiallySymmetrizedShearShufflePlanStore,
)
from tensordev.core.bigraded.symmetrized.plans import PartiallySymmetrizedPlanStore
assert not _emit_target_maps.signatures
assert not _rank_blocks.signatures
store = PartiallySymmetrizedShearShufflePlanStore(
    PartiallySymmetrizedPlanStore((1, 2), (2, 2)), "full"
)
assert store.block_plans
assert not store._use_compiled_plan_builder
assert not _emit_target_maps.signatures
assert not _rank_blocks.signatures
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


def test_disabled_numba_jit_store_is_warning_free():
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["NUMBA_DISABLE_JIT"] = "1"
    environment["PYTHONPATH"] = str(repository / "src")
    code = """
from tensordev.core.bigraded.symmetrized.gamma import (
    PartiallySymmetrizedShearShufflePlanStore,
)
from tensordev.core.bigraded.symmetrized.plans import PartiallySymmetrizedPlanStore
store = PartiallySymmetrizedShearShufflePlanStore(
    PartiallySymmetrizedPlanStore((1, 2), (2, 2)), "full"
)
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
