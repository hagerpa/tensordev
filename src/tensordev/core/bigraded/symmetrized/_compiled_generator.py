"""Compiled prime-generator maps for partially symmetrized shear cores."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from numba import njit

from tensordev.core.bigraded.symmetrized._compiled import (
    _MAX_INT64,
    _MAX_UINT64,
    _binomial_table,
    _rank_blocks_impl,
)
from tensordev.core.bigraded.symmetrized._compiled_terminal import (
    _merge_terminal_coefficient_impl,
)
from tensordev.core.bigraded.symmetrized._precompute import (
    _coefficient_dtype,
)
from tensordev.core.utils.precompute import _readonly, _unsigned_index_dtype


@njit(cache=True, nogil=True)
def _emit_prime_generator_support(
    output_placements: np.ndarray,
    rank_binomial: np.ndarray,
    coefficient_binomial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Emit source ranks and coefficients in output-placement rank order."""
    output_rank_count = output_placements.shape[0]
    source_block_count = output_placements.shape[1] - 1
    alphabet_size = output_placements.shape[2]
    source_ranks = np.empty(output_rank_count, dtype=np.uint64)
    coefficients = np.empty(output_rank_count, dtype=np.uint64)
    source = np.empty(
        (source_block_count, alphabet_size), dtype=np.uint64
    )

    for output_rank in range(output_rank_count):
        for block in range(source_block_count - 1):
            for letter in range(alphabet_size):
                source[block, letter] = output_placements[
                    output_rank, block, letter
                ]
        coefficients[output_rank] = _merge_terminal_coefficient_impl(
            output_placements[output_rank, source_block_count - 1],
            output_placements[output_rank, source_block_count],
            source[source_block_count - 1],
            coefficient_binomial,
        )
        source_ranks[output_rank] = _rank_blocks_impl(source, rank_binomial)
    return source_ranks, coefficients


@dataclass(frozen=True, slots=True, eq=False)
class PartiallySymmetrizedPrimeGeneratorSupport:
    """Primitive source maps for one quotient prime-generator block."""

    output_grade: tuple[int, int]
    source_ranks: np.ndarray
    coefficients: np.ndarray


def compile_partially_symmetrized_prime_generator_support(
    output_placements: np.ndarray,
    output_grade: tuple[int, int],
    *,
    compiled: bool = True,
) -> PartiallySymmetrizedPrimeGeneratorSupport:
    """Compile one prime-generator plan from output placement rows."""
    if not isinstance(compiled, (bool, np.bool_)):
        raise TypeError(f"compiled must be boolean, got {compiled!r}.")
    compiled = bool(compiled)
    output_placements = np.asarray(output_placements)
    if output_placements.ndim != 3:
        raise ValueError("output placements must have three axes")

    n, m = output_grade
    if n <= 0:
        raise ValueError("output grade must have positive prime degree")
    if output_placements.shape[1] != n + 1:
        raise ValueError(
            "output placement block count disagrees with its grade"
        )
    if output_placements.shape[0] == 0:
        raise ValueError("output placements must contain at least one row")
    q = int(output_placements.shape[2])
    if q <= 0:
        raise ValueError("placement alphabet width must be positive")

    output_rank_count = int(output_placements.shape[0])
    source_rank_count = comb(m + n * q - 1, m)
    maximum_coefficient = comb(m, m // 2)
    if output_rank_count > _MAX_INT64:
        raise OverflowError(
            f"output rank count {output_rank_count} exceeds the int64 plan limit"
        )
    if source_rank_count - 1 > _MAX_UINT64:
        raise OverflowError("source placement ranks exceed the uint64 plan limit")
    if maximum_coefficient > _MAX_UINT64:
        raise OverflowError("generator coefficients exceed the uint64 plan limit")

    source_part_count = n * q
    rank_binomial = _binomial_table(
        max(
            source_part_count - 2 + m,
            source_part_count - 1,
        ),
        max_column=source_part_count - 1,
        max_complement=m,
    )
    coefficient_binomial = _binomial_table(m)
    if compiled:
        source_ranks, coefficients = _emit_prime_generator_support(
            output_placements,
            rank_binomial,
            coefficient_binomial,
        )
    else:
        emitter = getattr(
            _emit_prime_generator_support,
            "py_func",
            _emit_prime_generator_support,
        )
        source_ranks, coefficients = emitter(
            output_placements,
            rank_binomial,
            coefficient_binomial,
        )

    return PartiallySymmetrizedPrimeGeneratorSupport(
        output_grade=output_grade,
        source_ranks=_readonly(
            np.asarray(
                source_ranks,
                dtype=_unsigned_index_dtype(source_rank_count - 1),
            )
        ),
        coefficients=_readonly(
            np.asarray(
                coefficients,
                dtype=_coefficient_dtype(maximum_coefficient),
            )
        ),
    )


__all__ = [
    "PartiallySymmetrizedPrimeGeneratorSupport",
    "compile_partially_symmetrized_prime_generator_support",
]
