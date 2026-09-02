"""Primitive right-generator plans for ordered bidegree shear coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from numbers import Integral

import numpy as np
from numba import njit

from tensordev.core.bigraded.symmetrized._compiled import (
    _MAX_INT64,
    _MAX_UINT64,
    _binomial_table,
)
from tensordev.core.utils.precompute import _readonly, _unsigned_index_dtype


def _generator_prime_term_count(grade: tuple[int, int]) -> int:
    """Return the exact prime-generator term count for one output grade."""
    n, m = grade
    if n == 0:
        return 0
    counts = [1 << doubleprime_count for doubleprime_count in range(m + 1)]
    for _ in range(2, n + 1):
        for doubleprime_count in range(1, m + 1):
            counts[doubleprime_count] += counts[doubleprime_count - 1]
    return counts[m]


@njit(cache=True, nogil=True)
def _emit_prime_generator_stream(
    output_placements: np.ndarray,
    source_rank_binomial: np.ndarray,
    degree: int,
    prime_term_count: int,
    prime_source_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Emit encounter-order encoded pairs and packed permutations."""
    output_count, prime_count = output_placements.shape

    encoded_pairs = np.empty(prime_term_count, dtype=np.uint64)
    permutations = np.empty((prime_term_count, degree), dtype=np.int64)
    raw_labels = np.empty(degree, dtype=np.int64)
    raw_index_by_label = np.empty(degree, dtype=np.int64)
    source_packed_axis = np.empty(degree, dtype=np.int64)
    output_packed_labels = np.empty(degree, dtype=np.int64)
    combination = np.empty(degree, dtype=np.int64)

    emitted = 0
    for output_rank in range(output_count):
        final_prime = int(output_placements[output_rank, prime_count - 1])
        previous_prime = -1
        if prime_count >= 2:
            previous_prime = int(
                output_placements[output_rank, prime_count - 2]
            )
        prefix_size = previous_prime + 1
        left_size = final_prime - prefix_size
        right_size = degree - final_prime - 1
        shuffled_size = left_size + right_size

        source_rank = np.uint64(0)
        for selected in range(prime_count - 1):
            source_rank += source_rank_binomial[
                int(output_placements[output_rank, selected]),
                selected + 1,
            ]
        encoded = (
            np.uint64(output_rank) * np.uint64(prime_source_count)
            + source_rank
        )

        source_prime = 0
        source_doubleprime = 0
        for raw_axis in range(degree - 1):
            is_prime = (
                source_prime < prime_count - 1
                and int(
                    output_placements[output_rank, source_prime]
                )
                == raw_axis
            )
            if is_prime:
                source_packed_axis[raw_axis] = source_prime
                source_prime += 1
            else:
                source_packed_axis[raw_axis] = (
                    prime_count - 1 + source_doubleprime
                )
                source_doubleprime += 1
        source_packed_axis[degree - 1] = degree - 1

        for prime in range(prime_count):
            output_packed_labels[prime] = output_placements[
                output_rank, prime
            ]
        prime = 0
        packed_axis = prime_count
        for output_axis in range(degree):
            if (
                prime < prime_count
                and int(output_placements[output_rank, prime])
                == output_axis
            ):
                prime += 1
            else:
                output_packed_labels[packed_axis] = output_axis
                packed_axis += 1

        for axis in range(prefix_size):
            raw_labels[axis] = axis
        for axis in range(left_size):
            combination[axis] = axis
        combinations_finished = False
        while not combinations_finished:
            left_axis = 0
            right_axis = 0
            for shuffled_axis in range(shuffled_size):
                takes_left = (
                    left_axis < left_size
                    and combination[left_axis] == shuffled_axis
                )
                if takes_left:
                    label = prefix_size + left_axis
                    left_axis += 1
                else:
                    label = final_prime + 1 + right_axis
                    right_axis += 1
                raw_labels[prefix_size + shuffled_axis] = label
            raw_labels[degree - 1] = final_prime

            for raw_axis in range(degree):
                raw_index_by_label[raw_labels[raw_axis]] = raw_axis
            for output_axis in range(degree):
                raw_axis = raw_index_by_label[
                    output_packed_labels[output_axis]
                ]
                permutations[emitted, output_axis] = source_packed_axis[
                    raw_axis
                ]
            encoded_pairs[emitted] = encoded
            emitted += 1

            if left_size == 0:
                combinations_finished = True
            else:
                combination_axis = left_size - 1
                while (
                    combination_axis >= 0
                    and combination[combination_axis]
                    == shuffled_size - left_size + combination_axis
                ):
                    combination_axis -= 1
                if combination_axis < 0:
                    combinations_finished = True
                else:
                    combination[combination_axis] += 1
                    for axis in range(combination_axis + 1, left_size):
                        combination[axis] = combination[axis - 1] + 1

    if emitted != prime_term_count:
        raise AssertionError("prime-generator term-count mismatch")
    return encoded_pairs, permutations


def _stable_group_stream(
    encoded_pairs: np.ndarray,
    permutations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group a term stream by first permutation occurrence, stably."""
    if encoded_pairs.size == 0:
        return (
            encoded_pairs,
            permutations[:0],
            np.zeros(1, dtype=np.int64),
        )
    _, first_indices, inverse = np.unique(
        permutations,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    first_occurrence_order = np.argsort(first_indices, kind="stable")
    group_count = int(first_indices.size)
    old_to_new = np.empty(group_count, dtype=np.int64)
    old_to_new[first_occurrence_order] = np.arange(
        group_count, dtype=np.int64
    )
    group_ids = old_to_new[inverse]
    counts = np.bincount(group_ids, minlength=group_count)
    offsets = np.empty(group_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    term_order = np.argsort(group_ids, kind="stable")
    return (
        encoded_pairs[term_order],
        permutations[first_indices[first_occurrence_order]],
        offsets,
    )


@dataclass(frozen=True, slots=True, eq=False)
class BigradedShearGeneratorSupport:
    """Primitive ordered-bidegree right-generator support."""

    grade: tuple[int, int]
    output_count: int
    prime_source_count: int
    doubleprime_source_ranks: np.ndarray
    doubleprime_output_ranks: np.ndarray
    prime_encoded_rank_pairs: np.ndarray
    prime_dense_permutations: np.ndarray
    prime_group_offsets: np.ndarray

    @property
    def prime_group_count(self) -> int:
        return int(self.prime_dense_permutations.shape[0])

    def prime_group_slice(self, group: int) -> slice:
        if isinstance(group, bool) or not isinstance(group, Integral):
            raise TypeError(f"group must be an integer, got {group!r}.")
        group = int(group)
        if group < 0 or group >= self.prime_group_count:
            raise IndexError(
                f"group {group} is outside [0, {self.prime_group_count})."
            )
        return slice(
            int(self.prime_group_offsets[group]),
            int(self.prime_group_offsets[group + 1]),
        )


def compile_bigraded_shear_generator_support(
    output_placements: np.ndarray,
    grade: tuple[int, int],
    *,
    compiled: bool = True,
) -> BigradedShearGeneratorSupport:
    """Compile one right-generator plan from colex placement rows."""
    if not isinstance(compiled, (bool, np.bool_)):
        raise TypeError(f"compiled must be boolean, got {compiled!r}.")
    compiled = bool(compiled)
    output_placements = np.asarray(output_placements)
    if output_placements.ndim != 2:
        raise ValueError("output placements must have two axes")
    n, m = grade
    degree = n + m
    if degree <= 0:
        raise ValueError("generator output degree must be positive")
    if output_placements.shape[1] != n:
        raise ValueError("output placement width disagrees with its grade")
    output_count = comb(degree, n)
    if output_placements.shape[0] != output_count:
        raise ValueError(
            f"output placements has {output_placements.shape[0]} rows, "
            f"expected {output_count}"
        )
    prime_source_count = comb(degree - 1, n - 1) if n else 1
    doubleprime_term_count = comb(degree - 1, n) if m else 0
    prime_term_count = _generator_prime_term_count(grade)
    prime_group_count = (1 << m) - m if n else 0
    maximum_encoded = output_count * prime_source_count - 1
    for name, count in (
        ("output placement count", output_count),
        ("prime-source placement count", prime_source_count),
        ("prime-generator term count", prime_term_count),
        ("prime-generator group count", prime_group_count),
    ):
        if count > _MAX_INT64:
            raise OverflowError(f"{name} {count} exceeds the int64 plan limit")
    if maximum_encoded > _MAX_UINT64:
        raise OverflowError(
            "encoded generator rank pairs exceed the uint64 plan limit"
        )

    rank_dtype = _unsigned_index_dtype(max(output_count - 1, 0))
    double_ranks = _readonly(
        np.arange(doubleprime_term_count, dtype=rank_dtype)
    )
    if n:
        source_rank_binomial = _binomial_table(
            max(degree - 2, 0),
            max_column=n - 1,
            max_complement=m,
        )
        emitter = (
            _emit_prime_generator_stream
            if compiled
            else getattr(
                _emit_prime_generator_stream,
                "py_func",
                _emit_prime_generator_stream,
            )
        )
        encoded, permutations = emitter(
            output_placements,
            source_rank_binomial,
            degree,
            prime_term_count,
            prime_source_count,
        )
        encoded, permutations, offsets = _stable_group_stream(
            encoded, permutations
        )
        if permutations.shape[0] != prime_group_count:
            raise AssertionError("prime-generator group-count mismatch")
    else:
        encoded = np.empty(0, dtype=np.uint64)
        permutations = np.empty((0, degree), dtype=np.int64)
        offsets = np.zeros(1, dtype=np.int64)

    pair_dtype = _unsigned_index_dtype(max(maximum_encoded, 0))
    permutation_dtype = _unsigned_index_dtype(max(degree - 1, 0))
    return BigradedShearGeneratorSupport(
        grade=grade,
        output_count=output_count,
        prime_source_count=prime_source_count,
        doubleprime_source_ranks=double_ranks,
        doubleprime_output_ranks=_readonly(double_ranks.copy()),
        prime_encoded_rank_pairs=_readonly(
            np.asarray(encoded, dtype=pair_dtype)
        ),
        prime_dense_permutations=_readonly(
            np.asarray(permutations, dtype=permutation_dtype)
        ),
        prime_group_offsets=_readonly(offsets),
    )


__all__ = [
    "BigradedShearGeneratorSupport",
    "_generator_prime_term_count",
    "compile_bigraded_shear_generator_support",
]
