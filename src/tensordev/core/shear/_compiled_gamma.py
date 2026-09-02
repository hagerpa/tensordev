"""Compiled primitive support for ordered shear shuffles.

For fixed input bidegrees, a shear-shuffle term is an ordinary shuffle in
which every double-prime block immediately preceding a prime letter stays
attached to that prime.  Only the two terminal double-prime blocks are
shuffled letter by letter.  This module emits that support directly as
fixed-width arrays; it does not construct symbolic words or per-term Python
objects.

Groups are indexed first by the lexicographic prime interleaving and then by
the lexicographic double-prime interleaving.  Terms within a group follow
increasing colexicographic input-placement-pair rank.  These are exactly the
first-occurrence group order and within-group encounter order of the symbolic
definition used by the ordered bidegree core.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from numbers import Integral

import numpy as np
from numba import njit

from tensordev.core.shear._compiled_support import (
    _MAX_INT64,
    _MAX_MASK_DEGREE,
    _MAX_UINT64,
    _non_negative_int,
)
from tensordev.core.utils.precompute import _readonly, _unsigned_index_dtype


def _terminal_tail_multiplicities(
    prime_count: int,
    doubleprime_count: int,
) -> tuple[tuple[int, int], ...]:
    """Return ``(terminal length, placement count)`` for one bidegree."""
    if prime_count == 0:
        return ((doubleprime_count, 1),)
    return tuple(
        (
            terminal_length,
            comb(
                prime_count + doubleprime_count - terminal_length - 1,
                prime_count - 1,
            ),
        )
        for terminal_length in range(doubleprime_count + 1)
    )


def _shear_shuffle_term_count(
    left_grade: tuple[int, int],
    right_grade: tuple[int, int],
) -> int:
    """Return the exact number of terms for one shear-shuffle block."""
    n1, m1 = left_grade
    n2, m2 = right_grade
    return comb(n1 + n2, n1) * sum(
        left_count
        * right_count
        * comb(left_tail + right_tail, left_tail)
        for left_tail, left_count in _terminal_tail_multiplicities(n1, m1)
        for right_tail, right_count in _terminal_tail_multiplicities(n2, m2)
    )


def _shear_shuffle_group_count(
    left_grade: tuple[int, int],
    right_grade: tuple[int, int],
) -> int:
    """Return the exact dense-permutation count for one shuffle block."""
    n1, m1 = left_grade
    n2, m2 = right_grade
    return comb(n1 + n2, n1) * comb(m1 + m2, m1)


def _combination_matrix(length: int, subset_size: int) -> np.ndarray:
    """Return lexicographically ordered combinations as an int64 matrix."""
    count = comb(length, subset_size)
    return np.asarray(
        tuple(combinations(range(length), subset_size)),
        dtype=np.int64,
    ).reshape(count, subset_size)


def _colex_masks(length: int, subset_size: int) -> np.ndarray:
    """Return uint64 placement masks at consecutive colex ranks."""
    masks = np.empty(comb(length, subset_size), dtype=np.uint64)
    for placement in combinations(range(length), subset_size):
        rank = sum(
            comb(position, selected + 1)
            for selected, position in enumerate(placement)
        )
        mask = 0
        for position in placement:
            mask |= 1 << position
        masks[rank] = mask
    return masks


def _side_matrix(keys: np.ndarray, length: int) -> np.ndarray:
    """Expand left-position keys into a small boolean side table."""
    sides = np.zeros((keys.shape[0], length), dtype=np.bool_)
    if keys.shape[1]:
        sides[np.arange(keys.shape[0])[:, None], keys] = True
    return sides


@njit(cache=True, nogil=True)
def _emit_shear_shuffle_support(
    output_masks: np.ndarray,
    prime_sides: np.ndarray,
    doubleprime_sides: np.ndarray,
    binomial: np.ndarray,
    n1: int,
    m1: int,
    n2: int,
    m2: int,
    term_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Emit masks group-contiguously in the symbolic encounter order."""
    prime_count = n1 + n2
    doubleprime_count = m1 + m2
    degree = prime_count + doubleprime_count
    left_degree = n1 + m1
    right_degree = n2 + m2
    doubleprime_key_count = doubleprime_sides.shape[0]
    group_count = prime_sides.shape[0] * doubleprime_key_count
    left_placement_count = binomial[left_degree, n1]
    right_placement_count = binomial[right_degree, n2]
    pair_count = left_placement_count * right_placement_count

    emitted_output_masks = np.empty(term_count, dtype=np.uint64)
    emitted_left_input_masks = np.empty(term_count, dtype=np.uint64)
    emitted_right_input_masks = np.empty(term_count, dtype=np.uint64)
    group_offsets = np.empty(group_count + 1, dtype=np.int64)
    group_offsets[0] = 0

    # For a fixed interleaving key, a placement pair contributes at most one
    # output word.  Pair-rank indexing preserves canonical encounter order
    # without sorting.
    pair_present = np.zeros(pair_count, dtype=np.bool_)
    pair_output_masks = np.empty(pair_count, dtype=np.uint64)
    pair_left_masks = np.empty(pair_count, dtype=np.uint64)
    pair_right_masks = np.empty(pair_count, dtype=np.uint64)

    emitted = 0
    for group in range(group_count):
        for pair in range(pair_count):
            pair_present[pair] = False

        prime_key = group // doubleprime_key_count
        doubleprime_key = group % doubleprime_key_count
        for output_index in range(output_masks.size):
            output_mask = output_masks[output_index]
            left_input_mask = np.uint64(0)
            right_input_mask = np.uint64(0)
            left_position = 0
            right_position = 0
            prime_index = 0
            doubleprime_index = 0
            valid = True

            for output_position in range(degree):
                is_prime = bool(
                    output_mask
                    & (np.uint64(1) << np.uint64(output_position))
                )
                if is_prime:
                    comes_from_left = prime_sides[prime_key, prime_index]
                    if comes_from_left:
                        left_input_mask |= (
                            np.uint64(1) << np.uint64(left_position)
                        )
                        left_position += 1
                    else:
                        right_input_mask |= (
                            np.uint64(1) << np.uint64(right_position)
                        )
                        right_position += 1
                    prime_index += 1
                else:
                    comes_from_left = doubleprime_sides[
                        doubleprime_key, doubleprime_index
                    ]
                    # ``prime_index`` is the next prime in the output word.
                    # A nonterminal D letter belongs to that prime's unit.
                    if (
                        prime_index < prime_count
                        and comes_from_left
                        != prime_sides[prime_key, prime_index]
                    ):
                        valid = False
                        break
                    if comes_from_left:
                        left_position += 1
                    else:
                        right_position += 1
                    doubleprime_index += 1

            if not valid:
                continue
            if left_position != left_degree or right_position != right_degree:
                raise AssertionError(
                    "shear-shuffle source-word width mismatch"
                )

            left_rank = 0
            selected = 0
            for position in range(left_degree):
                if left_input_mask & (
                    np.uint64(1) << np.uint64(position)
                ):
                    selected += 1
                    left_rank += binomial[position, selected]

            right_rank = 0
            selected = 0
            for position in range(right_degree):
                if right_input_mask & (
                    np.uint64(1) << np.uint64(position)
                ):
                    selected += 1
                    right_rank += binomial[position, selected]

            pair = left_rank * right_placement_count + right_rank
            if pair_present[pair]:
                raise AssertionError(
                    "a shear-shuffle key maps one input pair more than once"
                )
            pair_present[pair] = True
            pair_output_masks[pair] = output_mask
            pair_left_masks[pair] = left_input_mask
            pair_right_masks[pair] = right_input_mask

        for pair in range(pair_count):
            if pair_present[pair]:
                if emitted >= term_count:
                    raise AssertionError(
                        "shear-shuffle emission exceeded its preallocation"
                    )
                emitted_output_masks[emitted] = pair_output_masks[pair]
                emitted_left_input_masks[emitted] = pair_left_masks[pair]
                emitted_right_input_masks[emitted] = pair_right_masks[pair]
                emitted += 1
        group_offsets[group + 1] = emitted

    return (
        emitted_output_masks,
        emitted_left_input_masks,
        emitted_right_input_masks,
        group_offsets,
        emitted,
    )


@njit(cache=True, nogil=True)
def _dense_axis_permutations(
    prime_group_keys: np.ndarray,
    doubleprime_group_keys: np.ndarray,
    n1: int,
    m1: int,
    n2: int,
    m2: int,
) -> np.ndarray:
    """Convert lexicographic group keys to packed dense-axis rows."""
    prime_count = n1 + n2
    doubleprime_count = m1 + m2
    degree = prime_count + doubleprime_count
    left_degree = n1 + m1
    permutations = np.empty(
        (prime_group_keys.shape[0], degree), dtype=np.uint8
    )

    for group in range(prime_group_keys.shape[0]):
        left_prime = 0
        right_prime = 0
        selected = 0
        for output_prime in range(prime_count):
            comes_from_left = (
                selected < n1
                and prime_group_keys[group, selected] == output_prime
            )
            if comes_from_left:
                permutations[group, output_prime] = left_prime
                left_prime += 1
                selected += 1
            else:
                permutations[group, output_prime] = left_degree + right_prime
                right_prime += 1

        left_doubleprime = 0
        right_doubleprime = 0
        selected = 0
        for output_doubleprime in range(doubleprime_count):
            comes_from_left = (
                selected < m1
                and doubleprime_group_keys[group, selected]
                == output_doubleprime
            )
            output_axis = prime_count + output_doubleprime
            if comes_from_left:
                permutations[group, output_axis] = n1 + left_doubleprime
                left_doubleprime += 1
                selected += 1
            else:
                permutations[group, output_axis] = (
                    left_degree + n2 + right_doubleprime
                )
                right_doubleprime += 1
    return permutations


@njit(cache=True, nogil=True)
def _colex_output_pair_indices(
    output_masks: np.ndarray,
    left_input_masks: np.ndarray,
    right_input_masks: np.ndarray,
    binomial: np.ndarray,
    n1: int,
    m1: int,
    n2: int,
    m2: int,
    right_placement_count: int,
    pair_count: int,
) -> np.ndarray:
    """Encode output/pair ranks from primitive placement masks."""
    degree = n1 + m1 + n2 + m2
    left_degree = n1 + m1
    right_degree = n2 + m2
    encoded = np.empty(output_masks.size, dtype=np.uint64)
    for term in range(output_masks.size):
        output_rank = 0
        selected = 0
        for position in range(degree):
            if output_masks[term] & (
                np.uint64(1) << np.uint64(position)
            ):
                selected += 1
                output_rank += binomial[position, selected]

        left_rank = 0
        selected = 0
        for position in range(left_degree):
            if left_input_masks[term] & (
                np.uint64(1) << np.uint64(position)
            ):
                selected += 1
                left_rank += binomial[position, selected]

        right_rank = 0
        selected = 0
        for position in range(right_degree):
            if right_input_masks[term] & (
                np.uint64(1) << np.uint64(position)
            ):
                selected += 1
                right_rank += binomial[position, selected]

        pair = left_rank * right_placement_count + right_rank
        encoded[term] = (
            np.uint64(output_rank) * np.uint64(pair_count)
            + np.uint64(pair)
        )
    return encoded


@dataclass(frozen=True, slots=True, eq=False)
class ShearShuffleSupport:
    """Group-contiguous primitive support for one bidegree shuffle block."""

    left_grade: tuple[int, int]
    right_grade: tuple[int, int]
    output_masks: np.ndarray
    left_input_masks: np.ndarray
    right_input_masks: np.ndarray
    prime_group_keys: np.ndarray
    doubleprime_group_keys: np.ndarray
    group_offsets: np.ndarray

    @property
    def output_grade(self) -> tuple[int, int]:
        return (
            self.left_grade[0] + self.right_grade[0],
            self.left_grade[1] + self.right_grade[1],
        )

    @property
    def degree(self) -> int:
        return sum(self.output_grade)

    @property
    def term_count(self) -> int:
        return int(self.output_masks.size)

    @property
    def group_count(self) -> int:
        return int(self.prime_group_keys.shape[0])

    @property
    def output_placement_count(self) -> int:
        return comb(self.degree, self.output_grade[0])

    @property
    def left_placement_count(self) -> int:
        return comb(sum(self.left_grade), self.left_grade[0])

    @property
    def right_placement_count(self) -> int:
        return comb(sum(self.right_grade), self.right_grade[0])

    @property
    def pair_count(self) -> int:
        return self.left_placement_count * self.right_placement_count

    def group_slice(self, group: int) -> slice:
        if isinstance(group, bool) or not isinstance(group, Integral):
            raise TypeError(f"group must be an integer, got {group!r}.")
        group = int(group)
        if group < 0 or group >= self.group_count:
            raise IndexError(
                f"group {group} is outside [0, {self.group_count})."
            )
        return slice(
            int(self.group_offsets[group]),
            int(self.group_offsets[group + 1]),
        )

    def memory_bytes(self) -> int:
        return int(
            self.output_masks.nbytes
            + self.left_input_masks.nbytes
            + self.right_input_masks.nbytes
            + self.prime_group_keys.nbytes
            + self.doubleprime_group_keys.nbytes
            + self.group_offsets.nbytes
        )


def _compiled_grades(
    n1: object,
    m1: object,
    n2: object,
    m2: object,
) -> tuple[int, int, int, int, int]:
    n1 = _non_negative_int(n1, name="n1")
    m1 = _non_negative_int(m1, name="m1")
    n2 = _non_negative_int(n2, name="n2")
    m2 = _non_negative_int(m2, name="m2")
    degree = n1 + m1 + n2 + m2
    if degree > _MAX_MASK_DEGREE:
        raise ValueError(
            "compiled shear-shuffle support uses uint64 placement masks and "
            f"therefore requires output degree <= {_MAX_MASK_DEGREE}, got "
            f"{degree}."
        )
    return n1, m1, n2, m2, degree


def _binomial_table(degree: int) -> np.ndarray:
    table = np.zeros((degree + 1, degree + 1), dtype=np.int64)
    for row in range(degree + 1):
        for column in range(row + 1):
            table[row, column] = comb(row, column)
    return table


class _ShearShuffleWorkspace:
    """Store-local primitive tables shared across shuffle blocks."""

    __slots__ = ("_binomials", "_combinations", "_masks", "_sides")

    def __init__(self) -> None:
        self._binomials: dict[int, np.ndarray] = {}
        self._combinations: dict[tuple[int, int], np.ndarray] = {}
        self._masks: dict[tuple[int, int], np.ndarray] = {}
        self._sides: dict[tuple[int, int], np.ndarray] = {}

    def binomial(self, degree: int) -> np.ndarray:
        table = self._binomials.get(degree)
        if table is None:
            table = _binomial_table(degree)
            self._binomials[degree] = table
        return table

    def combinations(self, length: int, subset_size: int) -> np.ndarray:
        key = length, subset_size
        matrix = self._combinations.get(key)
        if matrix is None:
            matrix = _combination_matrix(length, subset_size)
            self._combinations[key] = matrix
        return matrix

    def masks(self, length: int, subset_size: int) -> np.ndarray:
        key = length, subset_size
        masks = self._masks.get(key)
        if masks is None:
            masks = _colex_masks(length, subset_size)
            self._masks[key] = masks
        return masks

    def sides(self, length: int, subset_size: int) -> np.ndarray:
        key = length, subset_size
        sides = self._sides.get(key)
        if sides is None:
            sides = _side_matrix(
                self.combinations(length, subset_size),
                length,
            )
            self._sides[key] = sides
        return sides


def compile_shear_shuffle_support(
    n1: int,
    m1: int,
    n2: int,
    m2: int,
    *,
    compiled: bool = True,
    _workspace: _ShearShuffleWorkspace | None = None,
) -> ShearShuffleSupport:
    """Compile one fixed-bidegree shear-shuffle support table."""
    n1, m1, n2, m2, degree = _compiled_grades(n1, m1, n2, m2)
    if not isinstance(compiled, (bool, np.bool_)):
        raise TypeError(f"compiled must be boolean, got {compiled!r}.")
    compiled = bool(compiled)
    if _workspace is None:
        _workspace = _ShearShuffleWorkspace()
    elif not isinstance(_workspace, _ShearShuffleWorkspace):
        raise TypeError("_workspace must be a _ShearShuffleWorkspace.")

    left_grade = n1, m1
    right_grade = n2, m2
    prime_count = n1 + n2
    doubleprime_count = m1 + m2
    output_count = comb(degree, prime_count)
    left_count = comb(n1 + m1, n1)
    right_count = comb(n2 + m2, n2)
    pair_count = left_count * right_count
    group_count = _shear_shuffle_group_count(left_grade, right_grade)
    term_count = _shear_shuffle_term_count(left_grade, right_grade)
    for name, count in (
        ("output placement count", output_count),
        ("input placement-pair count", pair_count),
        ("group count", group_count),
        ("term count", term_count),
    ):
        if count > _MAX_INT64:
            raise OverflowError(
                f"shear-shuffle {name} {count} exceeds the int64 plan limit."
            )
    maximum_encoded = output_count * pair_count - 1
    if maximum_encoded > _MAX_UINT64:
        raise OverflowError(
            "encoded shear-shuffle output/pair ranks exceed the uint64 plan "
            f"limit for grades {left_grade} and {right_grade}."
        )

    prime_keys = _workspace.combinations(prime_count, n1)
    doubleprime_keys = _workspace.combinations(doubleprime_count, m1)
    prime_sides = _workspace.sides(prime_count, n1)
    doubleprime_sides = _workspace.sides(doubleprime_count, m1)
    emitter = (
        _emit_shear_shuffle_support
        if compiled
        else getattr(
            _emit_shear_shuffle_support,
            "py_func",
            _emit_shear_shuffle_support,
        )
    )
    (
        output_masks,
        left_input_masks,
        right_input_masks,
        group_offsets,
        emitted,
    ) = emitter(
        _workspace.masks(degree, prime_count),
        prime_sides,
        doubleprime_sides,
        _workspace.binomial(degree),
        n1,
        m1,
        n2,
        m2,
        term_count,
    )
    if emitted != term_count:
        raise AssertionError(
            "compiled shear-shuffle support count disagrees with its exact "
            f"preallocation: emitted {emitted}, expected {term_count}."
        )
    if int(group_offsets[-1]) != term_count:
        raise AssertionError("shear-shuffle group offsets do not cover all terms.")

    key_dtype = _unsigned_index_dtype(max(degree - 1, 0))
    prime_group_keys = np.repeat(
        prime_keys, doubleprime_keys.shape[0], axis=0
    ).astype(key_dtype, copy=False)
    doubleprime_group_keys = np.tile(
        doubleprime_keys, (prime_keys.shape[0], 1)
    ).astype(key_dtype, copy=False)
    if prime_group_keys.shape[0] != group_count:
        raise AssertionError("shear-shuffle group-key count mismatch.")

    return ShearShuffleSupport(
        left_grade=left_grade,
        right_grade=right_grade,
        output_masks=_readonly(
            output_masks.astype(
                _unsigned_index_dtype(int(np.max(output_masks, initial=0))),
                copy=False,
            )
        ),
        left_input_masks=_readonly(
            left_input_masks.astype(
                _unsigned_index_dtype(
                    int(np.max(left_input_masks, initial=0))
                ),
                copy=False,
            )
        ),
        right_input_masks=_readonly(
            right_input_masks.astype(
                _unsigned_index_dtype(
                    int(np.max(right_input_masks, initial=0))
                ),
                copy=False,
            )
        ),
        prime_group_keys=_readonly(prime_group_keys),
        doubleprime_group_keys=_readonly(doubleprime_group_keys),
        group_offsets=_readonly(
            group_offsets.astype(
                _unsigned_index_dtype(term_count), copy=False
            )
        ),
    )


def shear_shuffle_dense_axis_permutations(
    support: ShearShuffleSupport,
    *,
    compiled: bool = True,
) -> np.ndarray:
    """Return packed dense-axis rows in the support's group order."""
    if not isinstance(support, ShearShuffleSupport):
        raise TypeError(
            "support must be a ShearShuffleSupport, got "
            f"{type(support).__name__}."
        )
    if not isinstance(compiled, (bool, np.bool_)):
        raise TypeError(f"compiled must be boolean, got {compiled!r}.")
    n1, m1 = support.left_grade
    n2, m2 = support.right_grade
    emitter = (
        _dense_axis_permutations
        if compiled
        else getattr(
            _dense_axis_permutations,
            "py_func",
            _dense_axis_permutations,
        )
    )
    permutations = emitter(
        support.prime_group_keys,
        support.doubleprime_group_keys,
        n1,
        m1,
        n2,
        m2,
    )
    return _readonly(
        permutations.astype(
            _unsigned_index_dtype(max(support.degree - 1, 0)), copy=False
        )
    )


def shear_shuffle_colex_output_pair_indices(
    support: ShearShuffleSupport,
    *,
    compiled: bool = True,
    _workspace: _ShearShuffleWorkspace | None = None,
) -> np.ndarray:
    """Encode output and input-pair colex ranks for a bidegree adapter."""
    if not isinstance(support, ShearShuffleSupport):
        raise TypeError(
            "support must be a ShearShuffleSupport, got "
            f"{type(support).__name__}."
        )
    if not isinstance(compiled, (bool, np.bool_)):
        raise TypeError(f"compiled must be boolean, got {compiled!r}.")
    if _workspace is None:
        binomial = _binomial_table(support.degree)
    elif isinstance(_workspace, _ShearShuffleWorkspace):
        binomial = _workspace.binomial(support.degree)
    else:
        raise TypeError("_workspace must be a _ShearShuffleWorkspace.")
    maximum = support.output_placement_count * support.pair_count - 1
    if maximum > _MAX_UINT64:
        raise OverflowError(
            "encoded shear-shuffle output/pair ranks exceed the uint64 plan "
            f"limit for grades {support.left_grade} and {support.right_grade}."
        )
    n1, m1 = support.left_grade
    n2, m2 = support.right_grade
    emitter = (
        _colex_output_pair_indices
        if compiled
        else getattr(
            _colex_output_pair_indices,
            "py_func",
            _colex_output_pair_indices,
        )
    )
    encoded = emitter(
        np.asarray(support.output_masks, dtype=np.uint64),
        np.asarray(support.left_input_masks, dtype=np.uint64),
        np.asarray(support.right_input_masks, dtype=np.uint64),
        binomial,
        n1,
        m1,
        n2,
        m2,
        support.right_placement_count,
        support.pair_count,
    )
    return _readonly(
        encoded.astype(_unsigned_index_dtype(maximum), copy=False)
    )


__all__ = [
    "ShearShuffleSupport",
    "compile_shear_shuffle_support",
    "shear_shuffle_colex_output_pair_indices",
    "shear_shuffle_dense_axis_permutations",
]
