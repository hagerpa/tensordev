"""Compiled primitive support tables for ordered shear transforms.

This module compiles shear-transform support without constructing symbolic
words or per-term Python objects.  The returned tables are grouped by the
double-prime row permutation.  Groups follow first-occurrence order in the
canonical symbolic support, and terms within a group retain their canonical
encounter order.

Placement masks use bit ``position`` for zero-based word position
``position``.  ``doubleprime_permutations[g]`` lists, in input-word order,
the output double-prime labels for group ``g``.  The terms of that group are
the half-open slice ``group_offsets[g]:group_offsets[g + 1]`` of both mask
arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from numbers import Integral

import numpy as np
from numba import njit

from tensordev.core.utils.precompute import _readonly, _unsigned_index_dtype


_MAX_MASK_DEGREE = np.iinfo(np.uint64).bits
_MAX_INT64 = np.iinfo(np.int64).max
_MAX_UINT64 = np.iinfo(np.uint64).max


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer, got {value!r}.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}.")
    return result


def _forward_term_count(prime_count: int, doubleprime_count: int) -> int:
    """Exact number of terms in the complete forward support."""
    counts = [1] * (doubleprime_count + 1)
    for current_prime_count in range(1, prime_count + 1):
        for current_doubleprime_count in range(1, doubleprime_count + 1):
            counts[current_doubleprime_count] += (current_prime_count + 1) * counts[
                current_doubleprime_count - 1
            ]
    return counts[doubleprime_count]


def _inverse_term_count(prime_count: int, doubleprime_count: int) -> int:
    """Exact number of terms in the complete inverse support."""
    counts = [1] * (doubleprime_count + 1)
    for _ in range(1, prime_count + 1):
        for current_doubleprime_count in range(1, doubleprime_count + 1):
            counts[current_doubleprime_count] += (
                2 * counts[current_doubleprime_count - 1]
            )
    return counts[doubleprime_count]


def _forward_group_count(prime_count: int, doubleprime_count: int) -> int:
    """Exact number of occurring double-prime permutations."""
    if doubleprime_count <= 1:
        return 1

    # Eulerian(m, d) counts permutations of m labels with d descents.  A
    # forward row permutation occurs exactly when it has at most n descents.
    eulerian = [1]
    for size in range(2, doubleprime_count + 1):
        next_row = [0] * size
        for descents in range(size):
            if descents < len(eulerian):
                next_row[descents] += (descents + 1) * eulerian[descents]
            if descents:
                next_row[descents] += (size - descents) * eulerian[descents - 1]
        eulerian = next_row
    return sum(eulerian[: min(prime_count, doubleprime_count - 1) + 1])


def _inverse_group_count(prime_count: int, doubleprime_count: int) -> int:
    """Exact number of occurring inverse double-prime permutations."""
    if doubleprime_count <= 1:
        return 1

    # The coefficient of x**r counts permutations with ``r`` non-singleton
    # decreasing runs.  Inserting the largest label gives
    # P_m(x) = 3 P_{m-1}(x) + (x - 2) P_{m-2}(x), with P_0 = P_1 = 1.
    two_back = (1,)
    one_back = (1,)
    for size in range(2, doubleprime_count + 1):
        coefficients = []
        for run_count in range(size // 2 + 1):
            value = 3 * one_back[run_count] if run_count < len(one_back) else 0
            if run_count < len(two_back):
                value -= 2 * two_back[run_count]
            if run_count and run_count - 1 < len(two_back):
                value += two_back[run_count - 1]
            coefficients.append(value)
        two_back, one_back = one_back, tuple(coefficients)
    return sum(one_back[: min(prime_count, len(one_back) - 1) + 1])


def _maximum_forward_row_count(
    prime_count: int,
    doubleprime_count: int,
) -> int:
    """Exact largest row support, computed by a small dynamic program."""
    if prime_count == 0:
        return 1

    # ``previous[s]`` is the largest partial row after assigning ``s``
    # double-prime letters through the current run.  Before the first prime,
    # any initial run has one realization.
    previous = [1] * (doubleprime_count + 1)
    for current_prime_count in range(1, prime_count + 1):
        current = [0] * (doubleprime_count + 1)
        for assigned in range(doubleprime_count + 1):
            best = 0
            for previously_assigned in range(assigned + 1):
                value = previous[previously_assigned] * comb(
                    current_prime_count + assigned,
                    assigned - previously_assigned,
                )
                if value > best:
                    best = value
            current[assigned] = best
        previous = current
    return previous[doubleprime_count]


@dataclass(frozen=True, slots=True, eq=False)
class ShearTransformSupport:
    """Group-contiguous primitive table for one shear-transform bidegree.

    Arrays are read-only.  The table is independent of alphabet dimensions
    and numerical backends, so total-degree and bidegree plan builders can
    adapt the same representation.
    """

    prime_count: int
    doubleprime_count: int
    output_masks: np.ndarray
    input_masks: np.ndarray
    doubleprime_permutations: np.ndarray
    group_offsets: np.ndarray

    @property
    def total_degree(self) -> int:
        return self.prime_count + self.doubleprime_count

    @property
    def term_count(self) -> int:
        return int(self.output_masks.size)

    @property
    def group_count(self) -> int:
        return int(self.doubleprime_permutations.shape[0])

    def group_slice(self, group: int) -> slice:
        """Return the term slice for one permutation group."""
        if isinstance(group, bool) or not isinstance(group, Integral):
            raise TypeError(f"group must be an integer, got {group!r}.")
        group = int(group)
        if group < 0 or group >= self.group_count:
            raise IndexError(f"group {group} is outside [0, {self.group_count}).")
        return slice(
            int(self.group_offsets[group]),
            int(self.group_offsets[group + 1]),
        )

    def memory_bytes(self) -> int:
        """Return the retained NumPy payload size in bytes."""
        return int(
            self.output_masks.nbytes
            + self.input_masks.nbytes
            + self.doubleprime_permutations.nbytes
            + self.group_offsets.nbytes
        )


@njit(cache=True, nogil=True)
def _same_permutation(
    permutations: np.ndarray,
    group: int,
    candidate: np.ndarray,
    width: int,
) -> bool:
    for axis in range(width):
        if permutations[group, axis] != candidate[axis]:
            return False
    return True


@njit(cache=True, nogil=True)
def _permutation_hash(candidate: np.ndarray, width: int) -> np.uint64:
    # Equality is always checked after hashing.  Use a bitwise xorshift mix so
    # the undecorated function has the same silent uint64 wraparound behavior
    # as compiled machine arithmetic when Numba is disabled.
    value = np.uint64(1469598103934665603)
    for axis in range(width):
        value ^= np.uint64(candidate[axis] + 1)
        value ^= value << np.uint64(13)
        value ^= value >> np.uint64(7)
        value ^= value << np.uint64(17)
    return value


@njit(cache=True, nogil=True)
def _lookup_or_insert_group(
    candidate: np.ndarray,
    width: int,
    permutations: np.ndarray,
    group_counts: np.ndarray,
    hash_groups: np.ndarray,
    group_count: int,
) -> tuple[int, int]:
    slot = np.int64(
        _permutation_hash(candidate, width) & np.uint64(hash_groups.size - 1)
    )
    while True:
        stored = hash_groups[slot]
        if stored == 0:
            group = group_count
            for axis in range(width):
                permutations[group, axis] = candidate[axis]
            hash_groups[slot] = group + 1
            return group, group_count + 1
        group = stored - 1
        if _same_permutation(permutations, group, candidate, width):
            return group, group_count
        slot = (slot + 1) & (hash_groups.size - 1)


@njit(cache=True, nogil=True)
def _gather_grouped_masks(
    stream_output_masks: np.ndarray,
    stream_input_masks: np.ndarray,
    stream_group_ids: np.ndarray,
    group_counts: np.ndarray,
    group_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stably gather an encounter-order stream into discovered groups."""
    term_count = stream_output_masks.size
    group_offsets = np.empty(group_count + 1, dtype=np.int64)
    group_offsets[0] = 0
    for group in range(group_count):
        group_offsets[group + 1] = group_offsets[group] + group_counts[group]

    output_masks = np.empty(term_count, dtype=np.uint64)
    input_masks = np.empty(term_count, dtype=np.uint64)
    cursors = group_offsets[:-1].copy()
    for term in range(term_count):
        group = stream_group_ids[term]
        target = cursors[group]
        output_masks[target] = stream_output_masks[term]
        input_masks[target] = stream_input_masks[term]
        cursors[group] += 1
    return output_masks, input_masks, group_offsets


@njit(cache=True, nogil=True)
def _emit_forward_support(
    prime_count: int,
    doubleprime_count: int,
    term_count: int,
    expected_group_count: int,
    maximum_row_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Emit encounter-order terms, then stably gather permutation groups."""
    degree = prime_count + doubleprime_count
    stream_output_masks = np.empty(term_count, dtype=np.uint64)
    stream_input_masks = np.empty(term_count, dtype=np.uint64)
    stream_group_ids = np.empty(term_count, dtype=np.int64)

    permutations = np.empty(
        (expected_group_count, doubleprime_count),
        dtype=np.uint8,
    )
    group_counts = np.zeros(expected_group_count, dtype=np.int64)
    hash_capacity = 2
    while hash_capacity < 2 * expected_group_count:
        hash_capacity *= 2
    hash_groups = np.zeros(hash_capacity, dtype=np.int64)

    # A row is built one prime at a time.  Each stage shuffles the transformed
    # prefix plus its new prime with the next consecutive double-prime run.
    # Two reusable buffers suffice for every canonical output placement.
    words_a = np.empty((maximum_row_count, degree), dtype=np.uint8)
    words_b = np.empty((maximum_row_count, degree), dtype=np.uint8)
    positions = np.empty(prime_count, dtype=np.int64)
    runs = np.empty(prime_count + 1, dtype=np.int64)
    combination = np.empty(degree, dtype=np.int64)
    candidate_permutation = np.empty(doubleprime_count, dtype=np.uint8)

    for axis in range(prime_count):
        positions[axis] = axis

    stream_index = 0
    group_count = 0
    placements_finished = False
    while not placements_finished:
        output_mask = np.uint64(0)
        if prime_count == 0:
            runs[0] = doubleprime_count
        else:
            for axis in range(prime_count):
                output_mask |= np.uint64(1) << np.uint64(positions[axis])
            runs[0] = positions[0]
            for axis in range(1, prime_count):
                runs[axis] = positions[axis] - positions[axis - 1] - 1
            runs[prime_count] = degree - positions[prime_count - 1] - 1

        current_length = runs[0]
        current_count = 1
        for axis in range(current_length):
            words_a[0, axis] = axis
        current_is_a = True
        next_doubleprime_label = current_length

        for prime_label in range(prime_count):
            left_length = current_length + 1
            terminal_length = runs[prime_label + 1]
            next_length = left_length + terminal_length
            next_count = 0

            for parent_index in range(current_count):
                for axis in range(left_length):
                    combination[axis] = axis
                combinations_finished = False
                while not combinations_finished:
                    left_axis = 0
                    terminal_axis = 0
                    for output_axis in range(next_length):
                        takes_left = (
                            left_axis < left_length
                            and combination[left_axis] == output_axis
                        )
                        if takes_left:
                            if left_axis == current_length:
                                value = doubleprime_count + prime_label
                            elif current_is_a:
                                value = words_a[parent_index, left_axis]
                            else:
                                value = words_b[parent_index, left_axis]
                            left_axis += 1
                        else:
                            value = next_doubleprime_label + terminal_axis
                            terminal_axis += 1
                        if current_is_a:
                            words_b[next_count, output_axis] = value
                        else:
                            words_a[next_count, output_axis] = value
                    next_count += 1

                    combination_axis = left_length - 1
                    while (
                        combination_axis >= 0
                        and combination[combination_axis]
                        == next_length - left_length + combination_axis
                    ):
                        combination_axis -= 1
                    if combination_axis < 0:
                        combinations_finished = True
                    else:
                        combination[combination_axis] += 1
                        for axis in range(combination_axis + 1, left_length):
                            combination[axis] = combination[axis - 1] + 1

            current_is_a = not current_is_a
            current_count = next_count
            current_length = next_length
            next_doubleprime_label += terminal_length

        for row_term in range(current_count):
            input_mask = np.uint64(0)
            doubleprime_axis = 0
            for axis in range(degree):
                value = (
                    words_a[row_term, axis] if current_is_a else words_b[row_term, axis]
                )
                if value >= doubleprime_count:
                    input_mask |= np.uint64(1) << np.uint64(axis)
                else:
                    candidate_permutation[doubleprime_axis] = value
                    doubleprime_axis += 1

            group, group_count = _lookup_or_insert_group(
                candidate_permutation,
                doubleprime_count,
                permutations,
                group_counts,
                hash_groups,
                group_count,
            )
            stream_output_masks[stream_index] = output_mask
            stream_input_masks[stream_index] = input_mask
            stream_group_ids[stream_index] = group
            group_counts[group] += 1
            stream_index += 1

        if prime_count == 0:
            placements_finished = True
        else:
            combination_axis = prime_count - 1
            while (
                combination_axis >= 0
                and positions[combination_axis]
                == degree - prime_count + combination_axis
            ):
                combination_axis -= 1
            if combination_axis < 0:
                placements_finished = True
            else:
                positions[combination_axis] += 1
                for axis in range(combination_axis + 1, prime_count):
                    positions[axis] = positions[axis - 1] + 1

    output_masks, input_masks, group_offsets = _gather_grouped_masks(
        stream_output_masks,
        stream_input_masks,
        stream_group_ids,
        group_counts,
        group_count,
    )

    return (
        output_masks,
        input_masks,
        permutations[:group_count],
        group_offsets,
        stream_index,
    )


@njit(cache=True, nogil=True)
def _emit_inverse_support(
    prime_count: int,
    doubleprime_count: int,
    term_count: int,
    expected_group_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Emit exact inverse terms without symbolic words or deduplication."""
    degree = prime_count + doubleprime_count
    stream_output_masks = np.empty(term_count, dtype=np.uint64)
    stream_input_masks = np.empty(term_count, dtype=np.uint64)
    stream_group_ids = np.empty(term_count, dtype=np.int64)

    permutations = np.empty(
        (expected_group_count, doubleprime_count),
        dtype=np.uint8,
    )
    group_counts = np.zeros(expected_group_count, dtype=np.int64)
    hash_capacity = 2
    while hash_capacity < 2 * expected_group_count:
        hash_capacity *= 2
    hash_groups = np.zeros(hash_capacity, dtype=np.int64)

    # Write each row by its double-prime run lengths.  At prime ``i``, peel
    # ``k`` letters from the next run, reverse them, and shuffle them with the
    # parent's trailing double-prime block.  The new prime and unpeeled suffix
    # are then appended.  The prime position determines ``k`` and the labels
    # distinguish the two shuffled blocks, so separate branches cannot
    # collide and no term-combining dictionary is required.
    words_a = np.empty((term_count, degree), dtype=np.uint8)
    words_b = np.empty((term_count, degree), dtype=np.uint8)
    trailing_a = np.empty(term_count, dtype=np.int64)
    trailing_b = np.empty(term_count, dtype=np.int64)
    positions = np.empty(prime_count, dtype=np.int64)
    runs = np.empty(prime_count + 1, dtype=np.int64)
    combination = np.empty(degree, dtype=np.int64)
    candidate_permutation = np.empty(doubleprime_count, dtype=np.uint8)

    for axis in range(prime_count):
        positions[axis] = axis

    stream_index = 0
    group_count = 0
    placements_finished = False
    while not placements_finished:
        output_mask = np.uint64(0)
        if prime_count == 0:
            runs[0] = doubleprime_count
        else:
            for axis in range(prime_count):
                output_mask |= np.uint64(1) << np.uint64(positions[axis])
            runs[0] = positions[0]
            for axis in range(1, prime_count):
                runs[axis] = positions[axis] - positions[axis - 1] - 1
            runs[prime_count] = degree - positions[prime_count - 1] - 1

        current_length = runs[0]
        current_count = 1
        for axis in range(current_length):
            words_a[0, axis] = axis
        trailing_a[0] = current_length
        current_is_a = True
        next_doubleprime_label = current_length

        for prime_label in range(prime_count):
            terminal_length = runs[prime_label + 1]
            next_length = current_length + 1 + terminal_length
            next_count = 0

            for parent_index in range(current_count):
                trailing_length = (
                    trailing_a[parent_index]
                    if current_is_a
                    else trailing_b[parent_index]
                )
                prefix_length = current_length - trailing_length
                for peeled in range(terminal_length + 1):
                    shuffled_length = trailing_length + peeled
                    for axis in range(trailing_length):
                        combination[axis] = axis
                    combinations_finished = False
                    while not combinations_finished:
                        for axis in range(prefix_length):
                            value = (
                                words_a[parent_index, axis]
                                if current_is_a
                                else words_b[parent_index, axis]
                            )
                            if current_is_a:
                                words_b[next_count, axis] = value
                            else:
                                words_a[next_count, axis] = value

                        parent_axis = 0
                        peeled_axis = 0
                        for output_axis in range(shuffled_length):
                            takes_parent = (
                                parent_axis < trailing_length
                                and combination[parent_axis] == output_axis
                            )
                            if takes_parent:
                                source_axis = prefix_length + parent_axis
                                value = (
                                    words_a[parent_index, source_axis]
                                    if current_is_a
                                    else words_b[parent_index, source_axis]
                                )
                                parent_axis += 1
                            else:
                                value = (
                                    next_doubleprime_label + peeled - 1 - peeled_axis
                                )
                                peeled_axis += 1
                            target_axis = prefix_length + output_axis
                            if current_is_a:
                                words_b[next_count, target_axis] = value
                            else:
                                words_a[next_count, target_axis] = value

                        target_axis = prefix_length + shuffled_length
                        if current_is_a:
                            words_b[next_count, target_axis] = (
                                doubleprime_count + prime_label
                            )
                        else:
                            words_a[next_count, target_axis] = (
                                doubleprime_count + prime_label
                            )
                        target_axis += 1
                        for terminal_axis in range(peeled, terminal_length):
                            value = next_doubleprime_label + terminal_axis
                            if current_is_a:
                                words_b[next_count, target_axis] = value
                            else:
                                words_a[next_count, target_axis] = value
                            target_axis += 1
                        if current_is_a:
                            trailing_b[next_count] = terminal_length - peeled
                        else:
                            trailing_a[next_count] = terminal_length - peeled
                        next_count += 1

                        combination_axis = trailing_length - 1
                        while (
                            combination_axis >= 0
                            and combination[combination_axis]
                            == shuffled_length - trailing_length + combination_axis
                        ):
                            combination_axis -= 1
                        if combination_axis < 0:
                            combinations_finished = True
                        else:
                            combination[combination_axis] += 1
                            for axis in range(
                                combination_axis + 1,
                                trailing_length,
                            ):
                                combination[axis] = combination[axis - 1] + 1

            current_is_a = not current_is_a
            current_count = next_count
            current_length = next_length
            next_doubleprime_label += terminal_length

        for row_term in range(current_count):
            input_mask = np.uint64(0)
            doubleprime_axis = 0
            for axis in range(degree):
                value = (
                    words_a[row_term, axis] if current_is_a else words_b[row_term, axis]
                )
                if value >= doubleprime_count:
                    input_mask |= np.uint64(1) << np.uint64(axis)
                else:
                    candidate_permutation[doubleprime_axis] = value
                    doubleprime_axis += 1

            group, group_count = _lookup_or_insert_group(
                candidate_permutation,
                doubleprime_count,
                permutations,
                group_counts,
                hash_groups,
                group_count,
            )
            stream_output_masks[stream_index] = output_mask
            stream_input_masks[stream_index] = input_mask
            stream_group_ids[stream_index] = group
            group_counts[group] += 1
            stream_index += 1

        if prime_count == 0:
            placements_finished = True
        else:
            combination_axis = prime_count - 1
            while (
                combination_axis >= 0
                and positions[combination_axis]
                == degree - prime_count + combination_axis
            ):
                combination_axis -= 1
            if combination_axis < 0:
                placements_finished = True
            else:
                positions[combination_axis] += 1
                for axis in range(combination_axis + 1, prime_count):
                    positions[axis] = positions[axis - 1] + 1

    output_masks, input_masks, group_offsets = _gather_grouped_masks(
        stream_output_masks,
        stream_input_masks,
        stream_group_ids,
        group_counts,
        group_count,
    )
    return (
        output_masks,
        input_masks,
        permutations[:group_count],
        group_offsets,
        stream_index,
    )


@njit(cache=True, nogil=True)
def _colex_rank_pairs_from_masks(
    output_masks: np.ndarray,
    input_masks: np.ndarray,
    prime_count: int,
    degree: int,
    binomial: np.ndarray,
    placement_count: np.uint64,
) -> tuple[np.ndarray, bool]:
    encoded = np.empty(output_masks.size, dtype=np.uint64)
    valid = True
    for term in range(output_masks.size):
        output_rank = np.uint64(0)
        input_rank = np.uint64(0)
        output_selected = 0
        input_selected = 0
        for position in range(degree):
            bit = np.uint64(1) << np.uint64(position)
            if output_masks[term] & bit:
                output_selected += 1
                output_rank += binomial[position, output_selected]
            if input_masks[term] & bit:
                input_selected += 1
                input_rank += binomial[position, input_selected]
        if output_selected != prime_count or input_selected != prime_count:
            valid = False
            encoded[term] = np.uint64(0)
        else:
            encoded[term] = output_rank * placement_count + input_rank
    return encoded, valid


def shear_colex_rank_pairs(
    support: ShearTransformSupport,
) -> np.ndarray:
    """Encode grouped output/input colex ranks for a bidegree adapter.

    The returned array is aligned with the support's group-contiguous mask
    arrays.  A pair is encoded as ``output_rank * placement_count +
    input_rank`` and uses the smallest practical unsigned NumPy dtype.
    Conversion of every mask is Numba-compiled; Python work is bounded by the
    total degree rather than by the number of support terms.
    """
    if not isinstance(support, ShearTransformSupport):
        raise TypeError(
            "support must be a ShearTransformSupport, got " f"{type(support).__name__}."
        )
    placement_count = comb(support.total_degree, support.prime_count)
    maximum = max(placement_count**2 - 1, 0)
    if maximum > _MAX_UINT64:
        raise OverflowError(
            "encoded colex rank pairs exceed the uint64 plan limit for "
            f"bidegree {(support.prime_count, support.doubleprime_count)}."
        )

    binomial = np.zeros(
        (support.total_degree + 1, support.prime_count + 1),
        dtype=np.uint64,
    )
    for position in range(support.total_degree + 1):
        for selected in range(min(position, support.prime_count) + 1):
            binomial[position, selected] = comb(position, selected)
    encoded, valid = _colex_rank_pairs_from_masks(
        support.output_masks,
        support.input_masks,
        support.prime_count,
        support.total_degree,
        binomial,
        np.uint64(placement_count),
    )
    if not valid:
        raise ValueError(
            "support placement masks do not match its declared prime count."
        )
    return _readonly(encoded.astype(_unsigned_index_dtype(maximum), copy=False))


def _compiled_grade(
    prime_count: object,
    doubleprime_count: object,
) -> tuple[int, int, int]:
    prime_count = _non_negative_int(prime_count, name="prime_count")
    doubleprime_count = _non_negative_int(
        doubleprime_count,
        name="doubleprime_count",
    )
    degree = prime_count + doubleprime_count
    if degree > _MAX_MASK_DEGREE:
        raise ValueError(
            "compiled shear support uses uint64 placement masks and "
            f"therefore requires total degree <= {_MAX_MASK_DEGREE}, got "
            f"{degree}."
        )
    return prime_count, doubleprime_count, degree


def _require_int64_counts(direction: str, **counts: int) -> None:
    for name, value in counts.items():
        if value > _MAX_INT64:
            raise OverflowError(
                f"{direction} shear {name.replace('_', ' ')} {value} "
                "exceeds the int64 plan limit."
            )


def compile_forward_shear_support(
    prime_count: int,
    doubleprime_count: int,
) -> ShearTransformSupport:
    """Compile the complete forward shear support at one bidegree.

    Compilation is host-side and alphabet-independent.  It uses exact support
    counts to allocate primitive fixed-width arrays once; the term expansion,
    stable group discovery, and grouping are all Numba-compiled.
    """
    prime_count, doubleprime_count, _ = _compiled_grade(
        prime_count,
        doubleprime_count,
    )

    term_count = _forward_term_count(prime_count, doubleprime_count)
    group_count = _forward_group_count(prime_count, doubleprime_count)
    maximum_row_count = _maximum_forward_row_count(
        prime_count,
        doubleprime_count,
    )
    _require_int64_counts(
        "forward",
        term_count=term_count,
        group_count=group_count,
        maximum_row_count=maximum_row_count,
    )

    (
        output_masks,
        input_masks,
        permutations,
        group_offsets,
        emitted_terms,
    ) = _emit_forward_support(
        prime_count,
        doubleprime_count,
        term_count,
        group_count,
        maximum_row_count,
    )
    if emitted_terms != term_count:
        raise AssertionError(
            "compiled forward support count disagrees with its exact "
            f"preallocation: emitted {emitted_terms}, expected {term_count}."
        )
    if permutations.shape[0] != group_count:
        raise AssertionError(
            "compiled forward permutation count disagrees with its exact "
            f"preallocation: emitted {permutations.shape[0]}, expected "
            f"{group_count}."
        )

    return ShearTransformSupport(
        prime_count=prime_count,
        doubleprime_count=doubleprime_count,
        output_masks=_readonly(output_masks),
        input_masks=_readonly(input_masks),
        doubleprime_permutations=_readonly(permutations),
        group_offsets=_readonly(group_offsets),
    )


def compile_inverse_shear_support(
    prime_count: int,
    doubleprime_count: int,
) -> ShearTransformSupport:
    """Compile the complete inverse shear support at one bidegree.

    The compiled recursion emits primitive fixed-width arrays directly.  Its
    coefficients are not stored because each one is exactly the product of
    the output- and input-placement parities.
    """
    prime_count, doubleprime_count, _ = _compiled_grade(
        prime_count,
        doubleprime_count,
    )
    term_count = _inverse_term_count(prime_count, doubleprime_count)
    group_count = _inverse_group_count(prime_count, doubleprime_count)
    _require_int64_counts(
        "inverse",
        term_count=term_count,
        group_count=group_count,
    )

    (
        output_masks,
        input_masks,
        permutations,
        group_offsets,
        emitted_terms,
    ) = _emit_inverse_support(
        prime_count,
        doubleprime_count,
        term_count,
        group_count,
    )
    if emitted_terms != term_count:
        raise AssertionError(
            "compiled inverse support count disagrees with its exact "
            f"preallocation: emitted {emitted_terms}, expected {term_count}."
        )
    if permutations.shape[0] != group_count:
        raise AssertionError(
            "compiled inverse permutation count disagrees with its exact "
            f"preallocation: emitted {permutations.shape[0]}, expected "
            f"{group_count}."
        )

    return ShearTransformSupport(
        prime_count=prime_count,
        doubleprime_count=doubleprime_count,
        output_masks=_readonly(output_masks),
        input_masks=_readonly(input_masks),
        doubleprime_permutations=_readonly(permutations),
        group_offsets=_readonly(group_offsets),
    )


__all__ = [
    "ShearTransformSupport",
    "compile_forward_shear_support",
    "compile_inverse_shear_support",
    "shear_colex_rank_pairs",
]
