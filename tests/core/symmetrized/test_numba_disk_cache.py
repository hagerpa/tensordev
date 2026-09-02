"""Cross-process cache contracts for quotient precomputation kernels."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


_SCRIPT = r"""
from tensordev.core.bigraded.symmetrized._compiled import _binomial_table
from tensordev.core.bigraded.symmetrized._compiled_generator import (
    compile_partially_symmetrized_prime_generator_support,
)
from tensordev.core.bigraded.symmetrized._compiled_plans import (
    compile_concatenation_targets,
    compile_doubleprime_generator_targets,
)
from tensordev.core.bigraded.symmetrized._compiled_gamma import (
    compile_partially_symmetrized_shear_shuffle_support,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
)

base = PartiallySymmetrizedPlanStore((1, 2), (2, 2))
left = base.grade_plan((1, 1)).placements
output = base.grade_plan((2, 2))
parts = 6
rank_table = _binomial_table(
    parts,
    max_column=parts - 1,
    max_complement=2,
)
compile_concatenation_targets(
    left,
    left,
    rank_table,
    output_rank_count=output.rank_count,
    compiled=True,
)
compile_doubleprime_generator_targets(
    base.grade_plan((2, 1)).placements,
    rank_table,
    output_rank_count=output.rank_count,
    compiled=True,
)
compile_partially_symmetrized_shear_shuffle_support(
    left,
    left,
    (1, 1),
    (1, 1),
    compiled=True,
)
compile_partially_symmetrized_prime_generator_support(
    output.placements,
    (2, 2),
    compiled=True,
)
"""


def _run(repository: Path, environment: dict[str, str]):
    return subprocess.run(
        [sys.executable, "-W", "error", "-c", _SCRIPT],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_quotient_emitters_load_their_numba_disk_cache(tmp_path):
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    environment["NUMBA_CACHE_DIR"] = str(tmp_path / "numba-cache")

    first = _run(repository, environment)
    assert first.returncode == 0, first.stderr

    environment["NUMBA_DEBUG_CACHE"] = "1"
    second = _run(repository, environment)
    assert second.returncode == 0, second.stderr
    log = second.stdout + second.stderr
    emitter_names = (
        "_emit_concatenation_targets",
        "_emit_doubleprime_generator_targets",
        "_emit_target_maps",
        "_emit_prime_generator_support",
    )
    for name in emitter_names:
        lines = [line for line in log.splitlines() if name in line]
        assert any("data loaded from" in line for line in lines)
        assert all("data saved to" not in line for line in lines)
