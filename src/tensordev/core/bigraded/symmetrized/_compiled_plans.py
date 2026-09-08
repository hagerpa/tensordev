"""Compiled rank maps for partially symmetrized base plans."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numba import njit

from tensordev.core.bigraded.symmetrized._compiled import (
    _binomial_table,
    _rank_blocks_impl,
)
from tensordev.core.bigraded.symmetrized.combinatorics import (
    multiset_placement_count,
)
from tensordev.core.utils.precompute import _readonly, _unsigned_index_dtype
from tensordev.core.utils.segmented import DestinationRankPlan


def _positive_count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer, got {value!r}.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}.")
    return result


def _placement_array(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3:
        raise ValueError(
            f"{name} must have shape (rank, block, letter), got {array.shape}."
        )
    if array.shape[0] == 0 or array.shape[1] == 0 or array.shape[2] == 0:
        raise ValueError(f"{name} axes must all be nonempty, got {array.shape}.")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer multiplicities.")
    if np.issubdtype(array.dtype, np.signedinteger) and np.any(array < 0):
        raise ValueError(f"{name} must contain non-negative multiplicities.")
    return array


def _execution_function(compiled: bool, emitter):
    if not isinstance(compiled, (bool, np.bool_)):
        raise TypeError(f"compiled must be boolean, got {compiled!r}.")
    if bool(compiled):
        return emitter
    return getattr(emitter, "py_func", emitter)


@njit(cache=True, nogil=True)
def _build_placement_array(
    rank_count: int,
    block_count: int,
    alphabet_size: int,
    multiplicity: int,
    binomial: np.ndarray,
) -> np.ndarray:
    """Unrank every placement directly into its compact rectangular layout."""
    part_count = block_count * alphabet_size
    placements = np.empty(
        (rank_count, block_count, alphabet_size),
        dtype=np.uint64,
    )
    if part_count == 1:
        placements[0, 0, 0] = multiplicity
        return placements

    separator_count = part_count - 1
    length = multiplicity + separator_count
    separators = np.empty(separator_count, dtype=np.int64)
    for rank in range(rank_count):
        remaining = rank
        upper = length - 1
        for subset_index in range(separator_count, 0, -1):
            lower = subset_index - 1
            high = upper
            while lower < high:
                candidate = (lower + high + 1) // 2
                if int(binomial[candidate, subset_index]) <= remaining:
                    lower = candidate
                else:
                    high = candidate - 1
            separators[subset_index - 1] = lower
            remaining -= int(binomial[lower, subset_index])
            upper = lower - 1

        for flat_index in range(part_count):
            if flat_index == 0:
                value = separators[0]
            elif flat_index == part_count - 1:
                value = length - separators[-1] - 1
            else:
                value = separators[flat_index] - separators[flat_index - 1] - 1
            placements[
                rank,
                flat_index // alphabet_size,
                flat_index % alphabet_size,
            ] = value
    return placements


def compile_placement_array(
    d_doubleprime: int,
    grade: tuple[int, int],
) -> np.ndarray:
    """Return all canonical placements without constructing Python tuples."""
    prime_count, multiplicity = grade
    rank_count = multiset_placement_count(d_doubleprime, grade)
    part_count = (prime_count + 1) * d_doubleprime
    separator_count = part_count - 1
    binomial = np.zeros((1, 1), dtype=np.uint64)
    if separator_count:
        binomial = _binomial_table(
            multiplicity + separator_count,
            max_column=separator_count,
        )
    placements = _build_placement_array(
        rank_count,
        prime_count + 1,
        d_doubleprime,
        multiplicity,
        binomial,
    )
    return _readonly(
        placements.astype(_unsigned_index_dtype(multiplicity), copy=False)
    )


@njit(cache=True, nogil=True)
def _emit_concatenation_targets(
    left_placements: np.ndarray,
    right_placements: np.ndarray,
    binomial: np.ndarray,
    targets: np.ndarray,
    output_rank_count: int,
) -> bool:
    """Fill one ordered pair's concatenation targets in rank-pair order."""
    left_block_count = left_placements.shape[1]
    right_block_count = right_placements.shape[1]
    alphabet_size = left_placements.shape[2]
    output_block_count = left_block_count + right_block_count - 1
    scratch = np.empty(
        (output_block_count, alphabet_size),
        dtype=np.uint64,
    )
    target = 0
    valid = True
    for left_rank in range(left_placements.shape[0]):
        for right_rank in range(right_placements.shape[0]):
            for block in range(left_block_count - 1):
                for letter in range(alphabet_size):
                    scratch[block, letter] = left_placements[
                        left_rank, block, letter
                    ]
            merged_block = left_block_count - 1
            for letter in range(alphabet_size):
                scratch[merged_block, letter] = (
                    left_placements[left_rank, merged_block, letter]
                    + right_placements[right_rank, 0, letter]
                )
            for block in range(1, right_block_count):
                for letter in range(alphabet_size):
                    scratch[merged_block + block, letter] = right_placements[
                        right_rank, block, letter
                    ]
            output_rank = _rank_blocks_impl(scratch, binomial)
            if output_rank >= output_rank_count:
                valid = False
            targets[target] = output_rank
            target += 1
    return valid


@njit(cache=True, nogil=True)
def _emit_doubleprime_generator_targets(
    source_placements: np.ndarray,
    binomial: np.ndarray,
    targets: np.ndarray,
    output_rank_count: int,
) -> bool:
    """Fill fixed-letter terminal-append targets in letter-major order."""
    block_count = source_placements.shape[1]
    alphabet_size = source_placements.shape[2]
    scratch = np.empty((block_count, alphabet_size), dtype=np.uint64)
    valid = True
    for letter in range(alphabet_size):
        for source_rank in range(source_placements.shape[0]):
            for block in range(block_count):
                for source_letter in range(alphabet_size):
                    scratch[block, source_letter] = source_placements[
                        source_rank, block, source_letter
                    ]
            scratch[block_count - 1, letter] += 1
            output_rank = _rank_blocks_impl(scratch, binomial)
            if output_rank >= output_rank_count:
                valid = False
            targets[letter, source_rank] = output_rank
    return valid


@njit(cache=True, nogil=True)
def _emit_doubleprime_generator_destination(
    output_placements: np.ndarray,
    binomial: np.ndarray,
    selected_edge_ids: np.ndarray,
    collision_target_ranks: np.ndarray,
    source_rank_count: int,
    doubleprime_rank_count: int,
) -> bool:
    """Fill a compact inverse terminal-append map in destination order."""
    block_count = output_placements.shape[1]
    alphabet_size = output_placements.shape[2]
    edge_count = alphabet_size * source_rank_count
    tail_primary_count = doubleprime_rank_count - source_rank_count
    collision_cursor = tail_primary_count
    scratch = np.empty((block_count, alphabet_size), dtype=np.uint64)

    for target_rank in range(output_placements.shape[0]):
        has_terminal_letter = False
        for letter in range(alphabet_size - 1, -1, -1):
            if output_placements[target_rank, block_count - 1, letter] == 0:
                continue
            if target_rank >= doubleprime_rank_count:
                return False
            for block in range(block_count):
                for source_letter in range(alphabet_size):
                    scratch[block, source_letter] = output_placements[
                        target_rank, block, source_letter
                    ]
            scratch[block_count - 1, letter] -= 1
            source_rank = _rank_blocks_impl(scratch, binomial)
            if source_rank >= source_rank_count:
                return False
            edge_id = letter * source_rank_count + source_rank
            if edge_id >= edge_count:
                return False

            if not has_terminal_letter:
                if target_rank < source_rank_count:
                    expected = (
                        (alphabet_size - 1) * source_rank_count
                        + target_rank
                    )
                    if edge_id != expected:
                        return False
                else:
                    tail_index = target_rank - source_rank_count
                    if tail_index >= tail_primary_count:
                        return False
                    selected_edge_ids[tail_index] = edge_id
                has_terminal_letter = True
            else:
                if collision_cursor >= selected_edge_ids.size:
                    return False
                selected_edge_ids[collision_cursor] = edge_id
                collision_target_ranks[
                    collision_cursor - tail_primary_count
                ] = target_rank
                collision_cursor += 1

        if target_rank < doubleprime_rank_count and not has_terminal_letter:
            return False
        if target_rank >= doubleprime_rank_count and has_terminal_letter:
            return False

    return collision_cursor == selected_edge_ids.size


def compile_concatenation_targets(
    left_placements: object,
    right_placements: object,
    binomial: np.ndarray,
    *,
    output_rank_count: int,
    compiled: bool,
) -> np.ndarray:
    """Return the compact concatenation target map for one grade pair."""
    left = _placement_array(left_placements, name="left_placements")
    right = _placement_array(right_placements, name="right_placements")
    if left.shape[2] != right.shape[2]:
        raise ValueError(
            "left_placements and right_placements must use the same alphabet."
        )
    output_rank_count = _positive_count(
        output_rank_count,
        name="output_rank_count",
    )
    target_dtype = _unsigned_index_dtype(output_rank_count - 1)
    targets = np.empty(
        left.shape[0] * right.shape[0],
        dtype=np.uint64,
    )
    emitter = _execution_function(compiled, _emit_concatenation_targets)
    valid = emitter(
        left,
        right,
        np.asarray(binomial, dtype=np.uint64),
        targets,
        output_rank_count,
    )
    if not valid:
        raise AssertionError("concatenation emitter produced an invalid rank")
    return _readonly(targets.astype(target_dtype, copy=False))


def compile_doubleprime_generator_targets(
    source_placements: object,
    binomial: np.ndarray,
    *,
    output_rank_count: int,
    compiled: bool,
) -> np.ndarray:
    """Return terminal-append targets for every double-prime letter."""
    source = _placement_array(source_placements, name="source_placements")
    output_rank_count = _positive_count(
        output_rank_count,
        name="output_rank_count",
    )
    target_dtype = _unsigned_index_dtype(output_rank_count - 1)
    targets = np.empty(
        (source.shape[2], source.shape[0]),
        dtype=np.uint64,
    )
    emitter = _execution_function(
        compiled,
        _emit_doubleprime_generator_targets,
    )
    valid = emitter(
        source,
        np.asarray(binomial, dtype=np.uint64),
        targets,
        output_rank_count,
    )
    if not valid:
        raise AssertionError("generator emitter produced an invalid rank")
    return _readonly(targets.astype(target_dtype, copy=False))


def compile_doubleprime_generator_destination(
    output_placements: object,
    binomial: np.ndarray,
    *,
    source_rank_count: int,
    doubleprime_rank_count: int,
    compiled: bool,
) -> DestinationRankPlan:
    """Return a compact destination-ordered terminal-append plan."""
    output = _placement_array(output_placements, name="output_placements")
    source_rank_count = _positive_count(
        source_rank_count,
        name="source_rank_count",
    )
    doubleprime_rank_count = _positive_count(
        doubleprime_rank_count,
        name="doubleprime_rank_count",
    )
    output_rank_count = output.shape[0]
    if doubleprime_rank_count > output_rank_count:
        raise ValueError(
            "doubleprime_rank_count cannot exceed the output rank count."
        )

    alphabet_size = output.shape[2]
    edge_count = alphabet_size * source_rank_count
    tail_primary_count = doubleprime_rank_count - source_rank_count
    collision_count = edge_count - doubleprime_rank_count
    if tail_primary_count < 0 or collision_count < 0:
        raise ValueError("invalid destination-plan rank counts")
    selected_edge_ids = np.empty(
        edge_count - source_rank_count,
        dtype=np.uint64,
    )
    collision_target_ranks = np.empty(
        collision_count,
        dtype=np.uint64,
    )
    emitter = _execution_function(
        compiled,
        _emit_doubleprime_generator_destination,
    )
    valid = emitter(
        output,
        np.asarray(binomial, dtype=np.uint64),
        selected_edge_ids,
        collision_target_ranks,
        source_rank_count,
        doubleprime_rank_count,
    )
    if not valid:
        raise AssertionError("destination generator emitter produced invalid ranks")
    return DestinationRankPlan.from_edges(
        selected_edge_ids,
        collision_target_ranks,
        edge_count=edge_count,
        output_rank_count=doubleprime_rank_count,
        primary_head_start=(alphabet_size - 1) * source_rank_count,
        primary_head_count=source_rank_count,
        tail_primary_count=tail_primary_count,
    )


__all__ = []
