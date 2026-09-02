from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from quotient_word_oracle import hat_psi_map
from tensordev.core.bigraded.symmetrized import _compiled
from tensordev.core.bigraded.symmetrized._compiled import (
    PartiallySymmetrizedTransformWorkspace,
    _binomial_table,
    _rank_blocks,
    _transform_maximum_coefficient,
    _transform_term_count,
    compile_partially_symmetrized_transform_pair,
)
from tensordev.core.bigraded.symmetrized._precompute import (
    _coefficient_dtype,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
)
from tensordev.core.bigraded.symmetrized.transforms import (
    PartiallySymmetrizedShearPlanStore,
)
from tensordev.core.utils.precompute import _unsigned_index_dtype


def _parities(placements: np.ndarray) -> np.ndarray:
    values = []
    for placement in placements:
        exponent = 0
        multiplicity_prefix = 0
        for prime_index, block in enumerate(placement[:-1], start=1):
            multiplicity_prefix += sum(map(int, block))
            exponent += prime_index + multiplicity_prefix
        values.append(-1 if exponent % 2 else 1)
    return np.asarray(values, dtype=np.int8)


@pytest.mark.parametrize("alphabet_size", (1, 2, 3))
@pytest.mark.parametrize("grade", ((0, 0), (1, 2), (2, 2)))
def test_compiled_and_python_support_match_independent_word_oracle(
    alphabet_size,
    grade,
):
    base = PartiallySymmetrizedPlanStore((1, alphabet_size), grade)
    placements = base.grade_plan(grade).placements
    python = compile_partially_symmetrized_transform_pair(
        placements,
        grade,
        compiled=False,
    )
    compiled = compile_partially_symmetrized_transform_pair(
        placements,
        grade,
        compiled=True,
    )

    np.testing.assert_array_equal(python[2], compiled[2])
    np.testing.assert_array_equal(compiled[2], _parities(placements))
    for inverse, python_support, compiled_support in zip(
        (False, True),
        python[:2],
        compiled[:2],
    ):
        np.testing.assert_array_equal(
            python_support.encoded_rank_pairs,
            compiled_support.encoded_rank_pairs,
        )
        np.testing.assert_array_equal(
            python_support.coefficients,
            compiled_support.coefficients,
        )

        signed = np.asarray(
            hat_psi_map(
                ("p",),
                tuple(f"x{index}" for index in range(alphabet_size)),
                grade,
                inverse=inverse,
            ).rows,
            dtype=object,
        )
        raw = signed
        if inverse:
            parity = np.asarray(compiled[2], dtype=object)
            raw = parity[:, None] * signed * parity[None, :]
        encoded = np.flatnonzero(raw.reshape(-1))
        coefficients = np.asarray(raw.reshape(-1)[encoded], dtype=object)
        assert np.all(coefficients > 0)
        expected_index_dtype = np.dtype(_unsigned_index_dtype(max(raw.size - 1, 0)))
        expected_coefficient_dtype = np.dtype(
            _coefficient_dtype(max(map(int, coefficients), default=0))
        )
        np.testing.assert_array_equal(
            compiled_support.encoded_rank_pairs,
            encoded.astype(expected_index_dtype),
        )
        np.testing.assert_array_equal(
            compiled_support.coefficients,
            coefficients.astype(expected_coefficient_dtype),
        )
        assert compiled_support.encoded_rank_pairs.dtype == expected_index_dtype
        assert compiled_support.coefficients.dtype == expected_coefficient_dtype
        assert not compiled_support.encoded_rank_pairs.flags.writeable
        assert not compiled_support.coefficients.flags.writeable
    assert not compiled[2].flags.writeable


def test_target_grade_counts_and_coefficient_bounds_are_exact():
    assert _transform_term_count(10, 4, 2, inverse=False) == 6_926_634
    assert _transform_term_count(10, 4, 2, inverse=True) == 148_995
    assert _transform_maximum_coefficient(10, 4, 2, inverse=False) == 24
    assert _transform_maximum_coefficient(10, 4, 2, inverse=True) == 6


def test_workspace_writes_directly_to_final_width_arrays(monkeypatch):
    base = PartiallySymmetrizedPlanStore((1, 2), (2, 2))
    original = getattr(
        _compiled._emit_factorized_transform,
        "py_func",
        _compiled._emit_factorized_transform,
    )
    observed = []

    def recording_emitter(*args):
        encoded, coefficients = args[-2:]
        observed.append((encoded.dtype, coefficients.dtype, encoded.size))
        return original(*args)

    monkeypatch.setattr(
        _compiled,
        "_emit_factorized_transform",
        recording_emitter,
    )
    workspace = PartiallySymmetrizedTransformWorkspace(
        2,
        2,
        2,
        compiled=False,
    )
    forward, inverse, _ = workspace.compile_pair(
        base.grade_plan((2, 2)).placements,
        (2, 2),
    )

    assert observed == [
        (forward.encoded_rank_pairs.dtype, forward.coefficients.dtype, 76),
        (inverse.encoded_rank_pairs.dtype, inverse.coefficients.dtype, 55),
    ]
    assert forward.encoded_rank_pairs.dtype == np.dtype(np.uint16)
    assert inverse.encoded_rank_pairs.dtype == np.dtype(np.uint16)
    assert forward.coefficients.dtype == np.dtype(np.uint8)
    assert inverse.coefficients.dtype == np.dtype(np.uint8)


def test_store_reuses_scalar_placements_once_per_prime_degree(monkeypatch):
    base = PartiallySymmetrizedPlanStore((1, 2), (3, 2))
    original = _compiled._scalar_placement_array
    calls = []

    def recording_builder(prime_count, multiplicity):
        calls.append((prime_count, multiplicity))
        return original(prime_count, multiplicity)

    monkeypatch.setattr(_compiled, "_scalar_placement_array", recording_builder)
    first = PartiallySymmetrizedShearPlanStore(base)
    assert calls == [(0, 2), (1, 2), (2, 2), (3, 2)]
    assert tuple(first.forward_plans) == tuple(base.grade_plans)
    assert tuple(first.inverse_plans) == tuple(base.grade_plans)

    PartiallySymmetrizedShearPlanStore(base)
    assert calls == 2 * [(0, 2), (1, 2), (2, 2), (3, 2)]


def test_target_workspace_scalar_tables_use_minimal_coefficients():
    workspace = PartiallySymmetrizedTransformWorkspace(
        10,
        4,
        2,
        compiled=True,
    )
    matrices = tuple(workspace.scalar_matrices.values())
    assert tuple(matrix.dtype for matrix in matrices) == (
        np.dtype(np.uint8),
        np.dtype(np.uint8),
    )
    assert sum(matrix.nbytes for matrix in matrices) == 10_020_010
    assert sum(matrix.nbytes for matrix in matrices) * 8 == 80_160_080


def test_target_store_uses_one_compiled_emitter_and_metadata_layout(tmp_path):
    environment = os.environ.copy()
    source = str(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
    environment["PYTHONPATH"] = source
    environment["NUMBA_CACHE_DIR"] = str(tmp_path / "numba-cache")
    script = """
from tensordev.core.bigraded.symmetrized._compiled import (
    _emit_factorized_transform,
    _species_metadata,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
)
from tensordev.core.bigraded.symmetrized.transforms import (
    PartiallySymmetrizedShearPlanStore,
)
base = PartiallySymmetrizedPlanStore((1, 2), (10, 4))
PartiallySymmetrizedShearPlanStore(base)
assert len(_emit_factorized_transform.signatures) == 1
emitter_signature = _emit_factorized_transform.signatures[0]
assert str(emitter_signature[-2].dtype) == "uint32"
assert str(emitter_signature[-1].dtype) == "uint8"
assert len(_species_metadata.signatures) == 1
metadata_input = _species_metadata.signatures[0][0]
assert metadata_input.layout == "C"
assert metadata_input.mutable
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_rectangular_binomial_table_handles_zero_multiplicity_rank():
    table = _binomial_table(
        8,
        max_column=9,
        max_complement=0,
    )
    placement = np.zeros((10, 1), dtype=np.uint8)
    ranker = getattr(_rank_blocks, "py_func", _rank_blocks)
    assert int(ranker(placement, table)) == 0
    assert table.shape == (9, 10)


def test_compiler_validates_placement_shape_and_totals():
    with pytest.raises(ValueError, match="shape"):
        compile_partially_symmetrized_transform_pair(
            np.zeros((1, 1, 1), dtype=np.uint8),
            (1, 0),
        )
    with pytest.raises(ValueError, match="total multiplicity"):
        compile_partially_symmetrized_transform_pair(
            np.zeros((2, 2, 1), dtype=np.uint8),
            (1, 1),
        )


def test_compiler_runs_with_numba_disabled_and_warnings_as_errors(tmp_path):
    environment = os.environ.copy()
    source = str(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
    environment["PYTHONPATH"] = source
    environment["NUMBA_DISABLE_JIT"] = "1"
    environment["PYTHONWARNINGS"] = "error"
    script = """
from tensordev.core.bigraded.symmetrized._compiled import (
    compile_partially_symmetrized_transform_pair,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
)
base = PartiallySymmetrizedPlanStore((1, 2), (2, 2))
forward, inverse, parity = compile_partially_symmetrized_transform_pair(
    base.grade_plan((2, 2)).placements,
    (2, 2),
    compiled=False,
)
assert forward.term_count == 76
assert inverse.term_count == 55
assert parity.size == 21
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
