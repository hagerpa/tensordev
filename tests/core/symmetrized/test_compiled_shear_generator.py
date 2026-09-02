from __future__ import annotations

import os
from math import comb
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import tensordev.core.shear.symmetrized as generator_module
from tensordev.core.bigraded.symmetrized._compiled_generator import (
    compile_partially_symmetrized_prime_generator_support,
)
from tensordev.core.bigraded.symmetrized._precompute import (
    _coefficient_dtype,
    _placement_tuple,
)
from tensordev.core.bigraded.symmetrized.combinatorics import (
    _rank_normalized_blocks,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
)
from tensordev.core.shear.symmetrized import (
    PartiallySymmetrizedShearGeneratorPlanStore,
    _expected_partially_symmetrized_shear_generator_memory_bytes_by_category,
)
from tensordev.core.utils.precompute import _unsigned_index_dtype


def _symbolic_support(base, output_grade):
    n, m = output_grade
    source_rank_count = base.grade_plan((n - 1, m)).rank_count
    source_ranks = []
    coefficients = []
    for row in base.grade_plan(output_grade).placements:
        placement = _placement_tuple(row)
        merged = tuple(
            left + right
            for left, right in zip(placement[-2], placement[-1])
        )
        source_ranks.append(
            _rank_normalized_blocks(placement[:-2] + (merged,))
        )
        coefficient = 1
        for left, right in zip(placement[-2], placement[-1]):
            coefficient *= comb(left + right, left)
        coefficients.append(coefficient)
    return (
        np.asarray(
            source_ranks,
            dtype=_unsigned_index_dtype(source_rank_count - 1),
        ),
        np.asarray(
            coefficients,
            dtype=_coefficient_dtype(comb(m, m // 2)),
        ),
    )


@pytest.mark.parametrize("q", (1, 2, 3))
@pytest.mark.parametrize(
    "output_grade",
    ((1, 0), (1, 3), (2, 2), (4, 3)),
)
def test_compiler_is_byte_exact_against_symbolic_oracle(q, output_grade):
    base = PartiallySymmetrizedPlanStore((1, q), output_grade)
    expected_ranks, expected_coefficients = _symbolic_support(
        base, output_grade
    )

    for compiled in (False, True):
        actual = compile_partially_symmetrized_prime_generator_support(
            base.grade_plan(output_grade).placements,
            output_grade,
            compiled=compiled,
        )
        for array, expected in (
            (actual.source_ranks, expected_ranks),
            (actual.coefficients, expected_coefficients),
        ):
            assert array.dtype == expected.dtype
            assert array.shape == expected.shape
            assert array.tobytes() == expected.tobytes()
            assert not array.flags.writeable


def test_compiled_and_python_stores_are_byte_exact(monkeypatch):
    base = PartiallySymmetrizedPlanStore((1, 3), (4, 3))
    monkeypatch.setattr(
        generator_module,
        "_COMPILED_SHEAR_GENERATOR_ENTRY_THRESHOLD",
        0,
    )
    compiled = PartiallySymmetrizedShearGeneratorPlanStore(base)
    monkeypatch.setattr(
        generator_module,
        "_COMPILED_SHEAR_GENERATOR_ENTRY_THRESHOLD",
        sys.maxsize,
    )
    python = PartiallySymmetrizedShearGeneratorPlanStore(base)

    assert compiled.generator_plans.keys() == python.generator_plans.keys()
    assert compiled.memory_bytes_by_category() == python.memory_bytes_by_category()
    for grade, compiled_plan in compiled.generator_plans.items():
        python_plan = python.generator_plans[grade]
        for left, right in (
            (compiled_plan.source_ranks, python_plan.source_ranks),
            (compiled_plan.coefficients, python_plan.coefficients),
        ):
            assert left.dtype == right.dtype
            assert left.tobytes() == right.tobytes()


def test_target_capacity_uses_compiled_builder_and_exact_payload():
    base = PartiallySymmetrizedPlanStore((1, 2), (10, 4))
    store = PartiallySymmetrizedShearGeneratorPlanStore(base)
    plan = store.generator_plan((10, 4))

    assert store._use_compiled_plan_builder
    assert plan.source_ranks.shape == (12_650,)
    assert plan.source_ranks.dtype == np.uint16
    assert plan.coefficients.shape == (12_650,)
    assert plan.coefficients.dtype == np.uint8
    assert dict(store.memory_bytes_by_category()) == dict(
        _expected_partially_symmetrized_shear_generator_memory_bytes_by_category(
            (1, 2), (10, 4)
        )
    )


def test_small_store_does_not_cold_compile_emitters():
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    code = """
from tensordev.core.bigraded.symmetrized._compiled import _rank_blocks
from tensordev.core.bigraded.symmetrized._compiled_generator import (
    _emit_prime_generator_support,
)
from tensordev.core.bigraded.symmetrized._compiled_terminal import (
    _merge_terminal_coefficient,
)
from tensordev.core.bigraded.symmetrized.plans import PartiallySymmetrizedPlanStore
from tensordev.core.shear.symmetrized import (
    PartiallySymmetrizedShearGeneratorPlanStore,
)
dispatchers = (
    _rank_blocks,
    _emit_prime_generator_support,
    _merge_terminal_coefficient,
)
assert all(not dispatcher.signatures for dispatcher in dispatchers)
store = PartiallySymmetrizedShearGeneratorPlanStore(
    PartiallySymmetrizedPlanStore((1, 2), (2, 2))
)
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
import tensordev.core.shear.symmetrized as module
from tensordev.core.bigraded.symmetrized.plans import PartiallySymmetrizedPlanStore
module._COMPILED_SHEAR_GENERATOR_ENTRY_THRESHOLD = 0
store = module.PartiallySymmetrizedShearGeneratorPlanStore(
    PartiallySymmetrizedPlanStore((1, 2), (2, 2))
)
assert store._use_compiled_plan_builder
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
