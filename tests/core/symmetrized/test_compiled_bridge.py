from __future__ import annotations

import os
from itertools import product
from math import comb
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tensordev.core.bigraded.precompute import colex_placements
from tensordev.core.bigraded.symmetrized._compiled_bridge import (
    compile_symmetrization_bridge_targets,
)
from tensordev.core.bigraded.symmetrized.bridge import (
    SymmetrizationBridgePlanStore,
)
from tensordev.core.utils.precompute import _unsigned_index_dtype


def _oracle_targets(d_doubleprime, grade):
    n, m = grade
    targets = []
    for placement in colex_placements(n + m, n):
        prime_positions = frozenset(placement)
        for word in product(range(d_doubleprime), repeat=m):
            blocks = [[0] * d_doubleprime for _ in range(n + 1)]
            block = 0
            doubleprime = 0
            for position in range(n + m):
                if position in prime_positions:
                    block += 1
                else:
                    blocks[block][word[doubleprime]] += 1
                    doubleprime += 1
            flattened = tuple(value for values in blocks for value in values)
            prefix = 0
            rank = 0
            for separator, value in enumerate(flattened[:-1], start=1):
                prefix += value
                rank += comb(separator - 1 + prefix, separator)
            targets.append(rank)
    rank_count = comb(m + (n + 1) * d_doubleprime - 1, m)
    return np.asarray(
        targets,
        dtype=_unsigned_index_dtype(rank_count - 1),
    )


@pytest.mark.parametrize("d_doubleprime", (1, 2, 3))
@pytest.mark.parametrize(
    "grade",
    ((0, 0), (0, 3), (3, 0), (1, 3), (3, 2)),
)
def test_vectorized_bridge_targets_are_byte_exact(
    d_doubleprime,
    grade,
):
    actual = compile_symmetrization_bridge_targets(
        colex_placements(sum(grade), grade[0]),
        d_doubleprime=d_doubleprime,
        grade=grade,
    )
    expected = _oracle_targets(d_doubleprime, grade)

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    assert not actual.flags.writeable


def test_large_capacity_has_exact_compact_bridge_payload():
    store = SymmetrizationBridgePlanStore((1, 2), (10, 4))
    plan = store.grade_plan((10, 4))

    assert plan.target_ranks.shape == (16_016,)
    assert plan.target_ranks.dtype == np.uint16
    assert np.unique(plan.target_ranks).size == 12_650
    assert store.memory_bytes() == 112_503
    assert not plan.target_ranks.flags.writeable


def test_bridge_compiler_rejects_malformed_inputs_before_indexing():
    canonical = colex_placements(2, 1)
    with pytest.raises(ValueError, match="positive"):
        compile_symmetrization_bridge_targets(
            canonical,
            d_doubleprime=0,
            grade=(1, 1),
        )
    with pytest.raises(TypeError, match="non-negative integer"):
        compile_symmetrization_bridge_targets(
            canonical,
            d_doubleprime=True,
            grade=(1, 1),
        )
    with pytest.raises(TypeError, match=r"grade\[0\]"):
        compile_symmetrization_bridge_targets(
            canonical,
            d_doubleprime=2,
            grade=(True, 1),
        )
    with pytest.raises(ValueError, match="shape"):
        compile_symmetrization_bridge_targets(
            np.zeros((2, 2), dtype=np.int64),
            d_doubleprime=2,
            grade=(1, 1),
        )
    with pytest.raises(TypeError, match="integer positions"):
        compile_symmetrization_bridge_targets(
            np.asarray(canonical, dtype=float),
            d_doubleprime=2,
            grade=(1, 1),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        compile_symmetrization_bridge_targets(
            np.asarray([[-1], [1]]),
            d_doubleprime=2,
            grade=(1, 1),
        )
    with pytest.raises(ValueError, match="colexicographic"):
        compile_symmetrization_bridge_targets(
            np.asarray([[1], [0]]),
            d_doubleprime=2,
            grade=(1, 1),
        )


def test_bridge_builder_does_not_compile_numba_dispatchers():
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    code = """
from tensordev.core.bigraded.symmetrized._compiled import _rank_blocks
from tensordev.core.bigraded.symmetrized.bridge import SymmetrizationBridgePlanStore
assert not _rank_blocks.signatures
store = SymmetrizationBridgePlanStore((1, 2), (10, 4))
assert store.grade_plans
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


def test_bridge_builder_is_warning_free_with_numba_disabled():
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["NUMBA_DISABLE_JIT"] = "1"
    environment["PYTHONPATH"] = str(repository / "src")
    code = """
from tensordev.core.bigraded.symmetrized.bridge import SymmetrizationBridgePlanStore
store = SymmetrizationBridgePlanStore((1, 2), (3, 3))
assert store.grade_plans
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
