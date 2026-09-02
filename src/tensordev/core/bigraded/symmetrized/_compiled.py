"""Compiled primitive tables for partially symmetrized bidegrees."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, factorial
from numbers import Integral

import numpy as np
from numba import njit
from numba.extending import register_jitable

from tensordev.core.bigraded.symmetrized._precompute import (
    _coefficient_dtype,
)
from tensordev.core.bigraded.symmetrized.combinatorics import (
    multiset_placement_count,
    multiset_placements,
)
from tensordev.core.utils.precompute import _readonly, _unsigned_index_dtype


_MAX_INT64 = np.iinfo(np.int64).max
_MAX_UINT64 = np.iinfo(np.uint64).max


def _binomial_table(
    maximum: int,
    *,
    max_column: int | None = None,
    max_complement: int | None = None,
) -> np.ndarray:
    """Return an exact uint64 band of Pascal's triangle.

    ``max_column`` and ``max_complement`` restrict stored entries to
    ``column <= max_column`` and ``row - column <= max_complement``.  The
    latter keeps separator/colex tables narrow in the meaningful direction:
    their column can be large while the multiplicity complement stays small.
    """
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        raise TypeError(f"maximum must be a non-negative integer, got {maximum!r}.")
    if maximum < 0:
        raise ValueError(f"maximum must be non-negative, got {maximum}.")
    if max_column is None:
        max_column = maximum
    if max_complement is None:
        max_complement = maximum
    for name, value in (
        ("max_column", max_column),
        ("max_complement", max_complement),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-negative integer, got {value!r}.")
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    table = np.zeros((maximum + 1, max_column + 1), dtype=np.uint64)
    for row in range(maximum + 1):
        first_column = max(0, row - max_complement)
        for column in range(first_column, min(row, max_column) + 1):
            value = comb(row, column)
            if value > _MAX_UINT64:
                raise OverflowError(
                    f"binomial coefficient C({row}, {column}) exceeds uint64."
                )
            table[row, column] = value
    return table


@register_jitable
def _rank_blocks_impl(
    placement: np.ndarray,
    binomial: np.ndarray,
) -> np.uint64:
    """Rank one normalized block-multiplicity row in separator/colex order."""
    block_count, alphabet_size = placement.shape
    prefix = 0
    rank = np.uint64(0)
    separator = 1
    final_entry = block_count * alphabet_size - 1
    flattened = 0
    for block in range(block_count):
        for letter in range(alphabet_size):
            if flattened != final_entry:
                prefix += int(placement[block, letter])
                rank += binomial[separator - 1 + prefix, separator]
                separator += 1
            flattened += 1
    return rank


_rank_blocks = njit(cache=True, nogil=True)(_rank_blocks_impl)


def _forward_scalar_term_count(prime_count: int, multiplicity: int) -> int:
    """Count nonzero one-letter forward transform entries exactly."""
    return (
        comb(prime_count + multiplicity, prime_count)
        * comb(prime_count + multiplicity + 1, prime_count)
        // (prime_count + 1)
    )


def _inverse_scalar_term_count(prime_count: int, multiplicity: int) -> int:
    """Count nonzero one-letter inverse transform entries exactly."""
    return comb(multiplicity + 2 * prime_count, 2 * prime_count)


def _scalar_maximum_coefficient(
    prime_count: int,
    multiplicity: int,
    *,
    inverse: bool,
) -> int:
    if multiplicity == 0 or prime_count == 0:
        return 1
    if inverse:
        return comb(multiplicity, multiplicity // 2)
    part_count = prime_count + 1
    quotient, remainder = divmod(multiplicity, part_count)
    denominator = 1
    for _ in range(part_count - remainder):
        denominator *= factorial(quotient)
    for _ in range(remainder):
        denominator *= factorial(quotient + 1)
    return factorial(multiplicity) // denominator


def _composition_aggregate(
    alphabet_size: int,
    multiplicity: int,
    scalar_values: tuple[int, ...],
    *,
    maximize: bool,
) -> int:
    values = [0] * (multiplicity + 1)
    values[0] = 1
    for _ in range(alphabet_size):
        updated = [0] * (multiplicity + 1)
        for total in range(multiplicity + 1):
            candidates = (
                values[total - assigned] * scalar_values[assigned]
                for assigned in range(total + 1)
            )
            updated[total] = max(candidates) if maximize else sum(candidates)
        values = updated
    return values[multiplicity]


def _transform_term_count(
    prime_count: int,
    multiplicity: int,
    alphabet_size: int,
    *,
    inverse: bool,
) -> int:
    counter = _inverse_scalar_term_count if inverse else _forward_scalar_term_count
    return _composition_aggregate(
        alphabet_size,
        multiplicity,
        tuple(counter(prime_count, value) for value in range(multiplicity + 1)),
        maximize=False,
    )


def _transform_maximum_coefficient(
    prime_count: int,
    multiplicity: int,
    alphabet_size: int,
    *,
    inverse: bool,
) -> int:
    return _composition_aggregate(
        alphabet_size,
        multiplicity,
        tuple(
            _scalar_maximum_coefficient(
                prime_count,
                value,
                inverse=inverse,
            )
            for value in range(multiplicity + 1)
        ),
        maximize=True,
    )


@njit(cache=True, nogil=True)
def _species_metadata(
    species_placements: np.ndarray,
    binomial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rank_count, alphabet_size, _, _ = species_placements.shape
    totals = np.empty((rank_count, alphabet_size), dtype=np.int64)
    scalar_ranks = np.empty((rank_count, alphabet_size), dtype=np.int64)
    for rank in range(rank_count):
        for letter in range(alphabet_size):
            placement = species_placements[rank, letter]
            total = 0
            for block in range(placement.shape[0]):
                total += int(placement[block, 0])
            totals[rank, letter] = total
            scalar_ranks[rank, letter] = int(_rank_blocks(placement, binomial))
    return totals, scalar_ranks


def _python_species_metadata(
    species_placements: np.ndarray,
    binomial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rank_count, alphabet_size, _, _ = species_placements.shape
    totals = np.empty((rank_count, alphabet_size), dtype=np.int64)
    scalar_ranks = np.empty((rank_count, alphabet_size), dtype=np.int64)
    ranker = getattr(_rank_blocks, "py_func", _rank_blocks)
    for rank in range(rank_count):
        for letter in range(alphabet_size):
            placement = species_placements[rank, letter]
            totals[rank, letter] = int(np.sum(placement, dtype=np.int64))
            scalar_ranks[rank, letter] = int(ranker(placement, binomial))
    return totals, scalar_ranks


@njit(cache=True, nogil=True)
def _placement_parities(placements: np.ndarray) -> np.ndarray:
    rank_count, block_count, alphabet_size = placements.shape
    parities = np.empty(rank_count, dtype=np.int8)
    for rank in range(rank_count):
        exponent = 0
        multiplicity_prefix = 0
        for block in range(block_count - 1):
            for letter in range(alphabet_size):
                multiplicity_prefix += int(placements[rank, block, letter])
            exponent += block + 1 + multiplicity_prefix
        parities[rank] = -1 if exponent % 2 else 1
    return parities


@njit(cache=True, nogil=True)
def _build_scalar_matrices(
    scalar_placements: np.ndarray,
    scalar_counts: np.ndarray,
    binomial: np.ndarray,
    inverse: bool,
    matrices: np.ndarray,
) -> None:
    multiplicity_count, _, block_count = scalar_placements.shape
    matrices[...] = 0
    prime_count = block_count - 1
    for multiplicity in range(multiplicity_count):
        rank_count = int(scalar_counts[multiplicity])
        for output_rank in range(rank_count):
            output = scalar_placements[multiplicity, output_rank]
            for input_rank in range(rank_count):
                input_placement = scalar_placements[multiplicity, input_rank]
                coefficient = np.uint64(1)
                valid = True
                if inverse:
                    moved_previous = 0
                    for block in range(1, block_count):
                        moved = int(input_placement[block - 1]) - (
                            int(output[block - 1]) - moved_previous
                        )
                        if moved < 0 or moved > int(output[block]):
                            valid = False
                            break
                        retained = int(output[block - 1]) - moved_previous
                        coefficient *= binomial[retained + moved, retained]
                        moved_previous = moved
                    if valid and int(input_placement[prime_count]) != (
                        int(output[prime_count]) - moved_previous
                    ):
                        valid = False
                else:
                    output_tail = int(output[prime_count])
                    input_higher = int(input_placement[prime_count])
                    if input_higher > output_tail:
                        valid = False
                    for block in range(prime_count - 1, -1, -1):
                        if not valid:
                            break
                        output_tail += int(output[block])
                        available = output_tail - input_higher
                        output_value = int(output[block])
                        if available < output_value:
                            valid = False
                            break
                        coefficient *= binomial[available, output_value]
                        input_higher += int(input_placement[block])
                    if input_higher != output_tail:
                        valid = False
                if valid:
                    matrices[multiplicity, output_rank, input_rank] = coefficient


@njit(cache=True, nogil=True)
def _emit_factorized_transform(
    totals: np.ndarray,
    scalar_ranks: np.ndarray,
    group_ids: np.ndarray,
    group_offsets: np.ndarray,
    group_members: np.ndarray,
    scalar_matrices: np.ndarray,
    encoded: np.ndarray,
    coefficients: np.ndarray,
) -> int:
    rank_count, alphabet_size = totals.shape
    term_count = encoded.size
    emitted = 0
    for output_rank in range(rank_count):
        group = group_ids[output_rank]
        for member_index in range(group_offsets[group], group_offsets[group + 1]):
            input_rank = group_members[member_index]
            coefficient = np.uint64(1)
            for letter in range(alphabet_size):
                coefficient *= scalar_matrices[
                    totals[output_rank, letter],
                    scalar_ranks[output_rank, letter],
                    scalar_ranks[input_rank, letter],
                ]
                if coefficient == 0:
                    break
            if coefficient:
                if emitted >= term_count:
                    return emitted + 1
                encoded[emitted] = np.uint64(output_rank) * np.uint64(
                    rank_count
                ) + np.uint64(input_rank)
                coefficients[emitted] = coefficient
                emitted += 1
    return emitted


@dataclass(frozen=True, slots=True, eq=False)
class PartiallySymmetrizedTransformSupport:
    """Minimal fixed-width support for one quotient transform direction."""

    grade: tuple[int, int]
    inverse: bool
    rank_count: int
    encoded_rank_pairs: np.ndarray
    coefficients: np.ndarray

    @property
    def term_count(self) -> int:
        return int(self.encoded_rank_pairs.size)


def _grade(value: object) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(
            "grade must be a pair of non-negative integers, got " f"{value!r}."
        )
    result = []
    for index, entry in enumerate(value):
        if isinstance(entry, bool) or not isinstance(entry, Integral):
            raise TypeError(
                f"grade[{index}] must be a non-negative integer, got {entry!r}."
            )
        entry = int(entry)
        if entry < 0:
            raise ValueError(f"grade[{index}] must be non-negative, got {entry}.")
        result.append(entry)
    return result[0], result[1]


def _scalar_placement_array(
    prime_count: int,
    multiplicity: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(
        [
            multiset_placement_count(1, (prime_count, value))
            for value in range(multiplicity + 1)
        ],
        dtype=np.int64,
    )
    maximum_count = int(np.max(counts, initial=1))
    placements = np.zeros(
        (multiplicity + 1, maximum_count, prime_count + 1),
        dtype=np.int64,
    )
    for value, rank_count in enumerate(counts):
        placements[value, :rank_count] = np.asarray(
            multiset_placements(1, (prime_count, value)),
            dtype=np.int64,
        ).reshape(int(rank_count), prime_count + 1)
    return placements, counts


def _rank_groups(totals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_lookup: dict[tuple[int, ...], int] = {}
    members: list[list[int]] = []
    group_ids = np.empty(totals.shape[0], dtype=np.int64)
    for rank, row in enumerate(totals):
        key = tuple(map(int, row))
        group = group_lookup.get(key)
        if group is None:
            group = len(members)
            group_lookup[key] = group
            members.append([])
        group_ids[rank] = group
        members[group].append(rank)
    offsets = np.zeros(len(members) + 1, dtype=np.int64)
    for group, values in enumerate(members):
        offsets[group + 1] = offsets[group] + len(values)
    flattened = np.asarray(
        [rank for values in members for rank in values],
        dtype=np.int64,
    )
    return group_ids, offsets, flattened


def _validated_placements(
    placements: np.ndarray,
    prime_count: int,
    multiplicity: int,
    alphabet_size: int,
) -> np.ndarray:
    placements = np.asarray(placements)
    expected_shape = (
        multiset_placement_count(
            alphabet_size,
            (prime_count, multiplicity),
        ),
        prime_count + 1,
        alphabet_size,
    )
    if placements.shape != expected_shape:
        raise ValueError(
            f"placements must have shape {expected_shape}, got " f"{placements.shape}."
        )
    if placements.dtype.kind not in "iu":
        raise TypeError("placements must have an integer dtype.")
    if placements.dtype.kind == "i" and np.any(placements < 0):
        raise ValueError("placements must be non-negative.")
    row_totals = np.sum(placements, axis=(1, 2), dtype=np.uint64)
    if np.any(row_totals != multiplicity):
        raise ValueError(
            f"every placement must have total multiplicity {multiplicity}."
        )
    return placements


class PartiallySymmetrizedTransformWorkspace:
    """Construction-local scalar data shared across one prime degree."""

    def __init__(
        self,
        prime_count: int,
        max_multiplicity: int,
        alphabet_size: int,
        *,
        compiled: bool = True,
        working_index_dtype=None,
        working_coefficient_dtype=None,
    ) -> None:
        self.prime_count, self.max_multiplicity = _grade(
            (prime_count, max_multiplicity)
        )
        if isinstance(alphabet_size, bool) or not isinstance(alphabet_size, Integral):
            raise TypeError(
                "alphabet_size must be a positive integer, got " f"{alphabet_size!r}."
            )
        self.alphabet_size = int(alphabet_size)
        if self.alphabet_size <= 0:
            raise ValueError(
                f"alphabet_size must be positive, got {self.alphabet_size}."
            )
        if not isinstance(compiled, (bool, np.bool_)):
            raise TypeError(f"compiled must be boolean, got {compiled!r}.")
        self.compiled = bool(compiled)
        maximum_rank_count = multiset_placement_count(
            self.alphabet_size,
            (self.prime_count, self.max_multiplicity),
        )
        required_index_dtype = np.dtype(
            _unsigned_index_dtype(max(maximum_rank_count**2 - 1, 0))
        )
        maximum_coefficient = max(
            _transform_maximum_coefficient(
                self.prime_count,
                multiplicity,
                self.alphabet_size,
                inverse=inverse,
            )
            for multiplicity in range(self.max_multiplicity + 1)
            for inverse in (False, True)
        )
        required_coefficient_dtype = np.dtype(_coefficient_dtype(maximum_coefficient))
        self.working_index_dtype = self._working_dtype(
            working_index_dtype,
            required_index_dtype,
            name="working_index_dtype",
        )
        self.working_coefficient_dtype = self._working_dtype(
            working_coefficient_dtype,
            required_coefficient_dtype,
            name="working_coefficient_dtype",
        )
        self.scalar_placements, self.scalar_counts = _scalar_placement_array(
            self.prime_count,
            self.max_multiplicity,
        )
        self.coefficient_table = _binomial_table(self.max_multiplicity)
        matrix_builder = (
            _build_scalar_matrices
            if self.compiled
            else getattr(
                _build_scalar_matrices,
                "py_func",
                _build_scalar_matrices,
            )
        )
        matrix_shape = (
            self.max_multiplicity + 1,
            self.scalar_placements.shape[1],
            self.scalar_placements.shape[1],
        )
        scalar_matrices = {}
        for inverse in (False, True):
            matrices = np.empty(
                matrix_shape,
                dtype=self.working_coefficient_dtype,
            )
            matrix_builder(
                self.scalar_placements,
                self.scalar_counts,
                self.coefficient_table,
                inverse,
                matrices,
            )
            scalar_matrices[inverse] = matrices
        self.scalar_matrices = scalar_matrices

    @staticmethod
    def _working_dtype(value, required: np.dtype, *, name: str) -> np.dtype:
        if value is None:
            return required
        dtype = np.dtype(value)
        if dtype.kind != "u":
            raise TypeError(f"{name} must be an unsigned integer dtype.")
        if np.iinfo(dtype).max < np.iinfo(required).max:
            raise ValueError(
                f"{name}={dtype} is narrower than required dtype {required}."
            )
        return dtype

    def compile_pair(
        self,
        placements: np.ndarray,
        grade: object,
    ) -> tuple[
        PartiallySymmetrizedTransformSupport,
        PartiallySymmetrizedTransformSupport,
        np.ndarray,
    ]:
        prime_count, multiplicity = _grade(grade)
        if prime_count != self.prime_count:
            raise ValueError(
                f"grade prime count {prime_count} does not match workspace "
                f"prime count {self.prime_count}."
            )
        if multiplicity > self.max_multiplicity:
            raise ValueError(
                f"grade multiplicity {multiplicity} exceeds workspace "
                f"capacity {self.max_multiplicity}."
            )
        placements = _validated_placements(
            placements,
            prime_count,
            multiplicity,
            self.alphabet_size,
        )
        rank_count = int(placements.shape[0])
        maximum_encoded = rank_count**2 - 1
        if rank_count > _MAX_INT64 or maximum_encoded > _MAX_UINT64:
            raise OverflowError("quotient transform ranks exceed fixed-width limits.")

        part_count = (prime_count + 1) * self.alphabet_size
        rank_table = _binomial_table(
            max(0, part_count - 2 + multiplicity),
            max_column=max(0, part_count - 1),
            max_complement=multiplicity,
        )
        species_placements = np.array(
            np.transpose(placements, (0, 2, 1))[:, :, :, None],
            copy=True,
            order="C",
        )
        if self.compiled:
            totals, scalar_ranks = _species_metadata(
                species_placements,
                rank_table,
            )
            parities = _placement_parities(placements)
        else:
            totals, scalar_ranks = _python_species_metadata(
                species_placements,
                rank_table,
            )
            parity_builder = getattr(
                _placement_parities,
                "py_func",
                _placement_parities,
            )
            parities = parity_builder(placements)
        group_ids, group_offsets, group_members = _rank_groups(totals)
        emitter = (
            _emit_factorized_transform
            if self.compiled
            else getattr(
                _emit_factorized_transform,
                "py_func",
                _emit_factorized_transform,
            )
        )
        supports = []
        for inverse in (False, True):
            term_count = _transform_term_count(
                prime_count,
                multiplicity,
                self.alphabet_size,
                inverse=inverse,
            )
            maximum_coefficient = _transform_maximum_coefficient(
                prime_count,
                multiplicity,
                self.alphabet_size,
                inverse=inverse,
            )
            if term_count > _MAX_INT64 or maximum_coefficient > _MAX_UINT64:
                raise OverflowError(
                    "quotient transform support exceeds fixed-width limits."
                )
            final_index_dtype = np.dtype(_unsigned_index_dtype(max(maximum_encoded, 0)))
            final_coefficient_dtype = np.dtype(_coefficient_dtype(maximum_coefficient))
            encoded = np.empty(term_count, dtype=self.working_index_dtype)
            coefficients = np.empty(
                term_count,
                dtype=self.working_coefficient_dtype,
            )
            emitted = emitter(
                totals,
                scalar_ranks,
                group_ids,
                group_offsets,
                group_members,
                self.scalar_matrices[inverse],
                encoded,
                coefficients,
            )
            if emitted != term_count:
                raise AssertionError(
                    "factorized quotient transform count disagrees with its "
                    f"exact preallocation: emitted {emitted}, expected "
                    f"{term_count}."
                )
            supports.append(
                PartiallySymmetrizedTransformSupport(
                    grade=(prime_count, multiplicity),
                    inverse=inverse,
                    rank_count=rank_count,
                    encoded_rank_pairs=_readonly(
                        encoded.astype(final_index_dtype, copy=False)
                    ),
                    coefficients=_readonly(
                        coefficients.astype(
                            final_coefficient_dtype,
                            copy=False,
                        )
                    ),
                )
            )
            del encoded, coefficients
        return supports[0], supports[1], _readonly(parities)


def compile_partially_symmetrized_transform_pair(
    placements: np.ndarray,
    grade: object,
    *,
    compiled: bool = True,
) -> tuple[
    PartiallySymmetrizedTransformSupport,
    PartiallySymmetrizedTransformSupport,
    np.ndarray,
]:
    """Compile forward/inverse quotient supports and their shared parity."""
    prime_count, multiplicity = _grade(grade)
    placements = np.asarray(placements)
    if placements.ndim != 3 or placements.shape[2] == 0:
        raise ValueError("placements must be a nonempty three-dimensional array.")
    workspace = PartiallySymmetrizedTransformWorkspace(
        prime_count,
        multiplicity,
        int(placements.shape[2]),
        compiled=compiled,
    )
    return workspace.compile_pair(placements, (prime_count, multiplicity))


__all__ = [
    "PartiallySymmetrizedTransformSupport",
    "PartiallySymmetrizedTransformWorkspace",
    "_binomial_table",
    "_forward_scalar_term_count",
    "_inverse_scalar_term_count",
    "_rank_blocks",
    "_transform_maximum_coefficient",
    "_transform_term_count",
    "compile_partially_symmetrized_transform_pair",
]
