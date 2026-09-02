"""Compiled target maps for partially symmetrized shear shuffles.

For a fixed pair of quotient bidegrees, the coefficient depends only on the
two input multiset placements.  Each prime-skeleton interleaving then maps
that same placement pair to one output multiset-placement rank.  This module
emits those coefficients and target maps directly from the compact placement
arrays, without constructing Python normal forms.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
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


def _prime_position_matrix(
    output_prime_count: int,
    left_prime_count: int,
) -> np.ndarray:
    """Return lexicographically ordered left-prime positions."""
    rows = tuple(combinations(range(output_prime_count), left_prime_count))
    return np.asarray(rows, dtype=np.int64).reshape(
        len(rows), left_prime_count
    )


def _side_matrix(positions: np.ndarray, width: int) -> np.ndarray:
    sides = np.zeros((positions.shape[0], width), dtype=np.bool_)
    if positions.shape[1]:
        sides[np.arange(positions.shape[0])[:, None], positions] = True
    return sides


@njit(cache=True, nogil=True)
def _emit_target_maps(
    left_placements: np.ndarray,
    right_placements: np.ndarray,
    prime_sides: np.ndarray,
    rank_binomial: np.ndarray,
    coefficient_binomial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Emit key-major target ranks and pair-major coefficients."""
    left_count = left_placements.shape[0]
    right_count = right_placements.shape[0]
    left_prime_count = left_placements.shape[1] - 1
    right_prime_count = right_placements.shape[1] - 1
    output_prime_count = left_prime_count + right_prime_count
    alphabet_size = left_placements.shape[2]
    pair_count = left_count * right_count

    targets = np.empty(
        (prime_sides.shape[0], pair_count), dtype=np.uint64
    )
    coefficients = np.empty(pair_count, dtype=np.uint64)
    output = np.empty(
        (output_prime_count + 1, alphabet_size), dtype=np.uint64
    )

    for left_rank in range(left_count):
        for right_rank in range(right_count):
            pair = left_rank * right_count + right_rank
            coefficients[pair] = _merge_terminal_coefficient_impl(
                left_placements[left_rank, left_prime_count],
                right_placements[right_rank, right_prime_count],
                output[output_prime_count],
                coefficient_binomial,
            )

            for key in range(prime_sides.shape[0]):
                left_block = 0
                right_block = 0
                for output_block in range(output_prime_count):
                    if prime_sides[key, output_block]:
                        for letter in range(alphabet_size):
                            output[output_block, letter] = left_placements[
                                left_rank, left_block, letter
                            ]
                        left_block += 1
                    else:
                        for letter in range(alphabet_size):
                            output[output_block, letter] = right_placements[
                                right_rank, right_block, letter
                            ]
                        right_block += 1
                targets[key, pair] = _rank_blocks_impl(
                    output,
                    rank_binomial,
                )
    return targets, coefficients


@dataclass(frozen=True, slots=True, eq=False)
class PartiallySymmetrizedShearShuffleSupport:
    """Primitive target maps for one quotient shear-shuffle block."""

    left_grade: tuple[int, int]
    right_grade: tuple[int, int]
    left_prime_positions: np.ndarray
    target_ranks: np.ndarray
    coefficients: np.ndarray

    @property
    def output_grade(self) -> tuple[int, int]:
        return (
            self.left_grade[0] + self.right_grade[0],
            self.left_grade[1] + self.right_grade[1],
        )

    @property
    def key_count(self) -> int:
        return int(self.target_ranks.shape[0])

    @property
    def pair_count(self) -> int:
        return int(self.target_ranks.shape[1])


class PartiallySymmetrizedGammaWorkspace:
    """Store-local primitive tables shared across quotient shuffle blocks."""

    __slots__ = ("_coefficient_tables", "_prime_tables", "_rank_tables")

    def __init__(self) -> None:
        self._coefficient_tables: dict[tuple[int, int], np.ndarray] = {}
        self._prime_tables: dict[
            tuple[int, int], tuple[np.ndarray, np.ndarray]
        ] = {}
        self._rank_tables: dict[tuple[int, int, int], np.ndarray] = {}

    def prime_tables(
        self,
        output_prime_count: int,
        left_prime_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = output_prime_count, left_prime_count
        tables = self._prime_tables.get(key)
        if tables is None:
            positions = _prime_position_matrix(*key)
            tables = positions, _side_matrix(positions, output_prime_count)
            self._prime_tables[key] = tables
        return tables

    def rank_table(
        self,
        output_prime_count: int,
        output_multiplicity: int,
        alphabet_size: int,
    ) -> np.ndarray:
        key = output_prime_count, output_multiplicity, alphabet_size
        table = self._rank_tables.get(key)
        if table is None:
            part_count = (output_prime_count + 1) * alphabet_size
            table = _binomial_table(
                max(part_count - 2 + output_multiplicity, part_count - 1),
                max_column=part_count - 1,
                max_complement=output_multiplicity,
            )
            self._rank_tables[key] = table
        return table

    def coefficient_table(
        self,
        output_multiplicity: int,
        left_multiplicity: int,
    ) -> np.ndarray:
        key = output_multiplicity, left_multiplicity
        table = self._coefficient_tables.get(key)
        if table is None:
            table = _binomial_table(
                output_multiplicity,
                max_column=left_multiplicity,
                max_complement=output_multiplicity - left_multiplicity,
            )
            self._coefficient_tables[key] = table
        return table


def compile_partially_symmetrized_shear_shuffle_support(
    left_placements: np.ndarray,
    right_placements: np.ndarray,
    left_grade: tuple[int, int],
    right_grade: tuple[int, int],
    *,
    compiled: bool = True,
    _workspace: PartiallySymmetrizedGammaWorkspace | None = None,
) -> PartiallySymmetrizedShearShuffleSupport:
    """Compile one quotient shear-shuffle block from placement arrays."""
    if not isinstance(compiled, (bool, np.bool_)):
        raise TypeError(f"compiled must be boolean, got {compiled!r}.")
    compiled = bool(compiled)
    if _workspace is None:
        _workspace = PartiallySymmetrizedGammaWorkspace()
    elif not isinstance(_workspace, PartiallySymmetrizedGammaWorkspace):
        raise TypeError(
            "_workspace must be a PartiallySymmetrizedGammaWorkspace."
        )
    left_placements = np.asarray(left_placements)
    right_placements = np.asarray(right_placements)
    if left_placements.ndim != 3 or right_placements.ndim != 3:
        raise ValueError("placement arrays must have three axes")

    n1, m1 = left_grade
    n2, m2 = right_grade
    if left_placements.shape[1] != n1 + 1:
        raise ValueError("left placement block count disagrees with its grade")
    if right_placements.shape[1] != n2 + 1:
        raise ValueError("right placement block count disagrees with its grade")
    if left_placements.shape[2] != right_placements.shape[2]:
        raise ValueError("placement arrays must have the same alphabet width")
    if left_placements.shape[0] == 0 or right_placements.shape[0] == 0:
        raise ValueError("placement arrays must contain at least one row")

    q = int(left_placements.shape[2])
    if q <= 0:
        raise ValueError("placement alphabet width must be positive")
    output_grade = n1 + n2, m1 + m2
    key_count = comb(output_grade[0], n1)
    pair_count = left_placements.shape[0] * right_placements.shape[0]
    output_rank_count = comb(
        output_grade[1] + (output_grade[0] + 1) * q - 1,
        output_grade[1],
    )
    maximum_coefficient = comb(output_grade[1], m1)
    for name, count in (
        ("prime-interleaving count", key_count),
        ("placement-pair count", pair_count),
        ("target-map entry count", key_count * pair_count),
    ):
        if count > _MAX_INT64:
            raise OverflowError(f"{name} {count} exceeds the int64 plan limit")
    if output_rank_count - 1 > _MAX_UINT64:
        raise OverflowError("output placement ranks exceed the uint64 plan limit")
    if maximum_coefficient > _MAX_UINT64:
        raise OverflowError("shuffle coefficients exceed the uint64 plan limit")

    left_positions, prime_sides = _workspace.prime_tables(
        output_grade[0],
        n1,
    )
    rank_binomial = _workspace.rank_table(
        output_grade[0],
        output_grade[1],
        q,
    )
    coefficient_binomial = _workspace.coefficient_table(
        output_grade[1],
        m1,
    )
    if compiled:
        targets, coefficients = _emit_target_maps(
            left_placements,
            right_placements,
            prime_sides,
            rank_binomial,
            coefficient_binomial,
        )
    else:
        emitter = getattr(_emit_target_maps, "py_func", _emit_target_maps)
        targets, coefficients = emitter(
            left_placements,
            right_placements,
            prime_sides,
            rank_binomial,
            coefficient_binomial,
        )

    target_dtype = _unsigned_index_dtype(max(output_rank_count - 1, 0))
    coefficient_dtype = _coefficient_dtype(maximum_coefficient)
    return PartiallySymmetrizedShearShuffleSupport(
        left_grade=left_grade,
        right_grade=right_grade,
        left_prime_positions=_readonly(left_positions),
        target_ranks=_readonly(np.asarray(targets, dtype=target_dtype)),
        coefficients=_readonly(
            np.asarray(coefficients, dtype=coefficient_dtype)
        ),
    )


__all__ = [
    "PartiallySymmetrizedGammaWorkspace",
    "PartiallySymmetrizedShearShuffleSupport",
    "compile_partially_symmetrized_shear_shuffle_support",
]
