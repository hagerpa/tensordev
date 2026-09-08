"""Vectorized target maps for ordered-to-quotient bridge plans."""

from __future__ import annotations

from math import comb
from numbers import Integral

import numpy as np

from tensordev.core.bigraded.symmetrized._compiled import (
    _MAX_INT64,
    _MAX_UINT64,
    _binomial_table,
)
from tensordev.core.utils.precompute import _readonly, _unsigned_index_dtype


_VECTOR_WORKSPACE_ENTRIES = 2_000_000


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"{name} must be a non-negative integer, got {value!r}."
        )
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}.")
    return result


def _grade(value: object) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(
            "grade must be a pair of non-negative integers, got "
            f"{value!r}."
        )
    return (
        _non_negative_int(value[0], name="grade[0]"),
        _non_negative_int(value[1], name="grade[1]"),
    )


def _prime_placement_matrix(
    values: object,
    *,
    prime_count: int,
    doubleprime_count: int,
) -> np.ndarray:
    placements = np.asarray(values)
    expected_count = comb(prime_count + doubleprime_count, prime_count)
    expected_shape = expected_count, prime_count
    if placements.ndim != 2 or placements.shape != expected_shape:
        raise ValueError(
            f"prime_placements must have shape {expected_shape}, got "
            f"{placements.shape}."
        )
    if prime_count and not np.issubdtype(placements.dtype, np.integer):
        raise TypeError("prime_placements must contain integer positions.")

    degree = prime_count + doubleprime_count
    for placement_rank, placement in enumerate(placements):
        previous = -1
        colex_rank = 0
        for selected, raw_position in enumerate(placement, start=1):
            position = int(raw_position)
            if position <= previous or position >= degree:
                raise ValueError(
                    "prime_placements rows must be strictly increasing with "
                    f"positions in [0, {degree})."
                )
            colex_rank += comb(position, selected)
            previous = position
        if colex_rank != placement_rank:
            raise ValueError(
                "prime_placements must be in consecutive colexicographic "
                "rank order."
            )
    # Normalize compact caller dtypes once so all vectorized index arithmetic
    # has one signed, fixed-width encoding.
    return np.array(placements, dtype=np.int64, order="C", copy=True)


def _doubleprime_block_matrix(
    prime_placements: np.ndarray,
    *,
    doubleprime_count: int,
) -> np.ndarray:
    """Return the normal-form block of every ordered double-prime slot."""
    placement_count, prime_count = prime_placements.shape
    degree = prime_count + doubleprime_count
    positions = np.arange(degree, dtype=np.int64)
    is_prime = np.any(
        positions[None, :, None] == prime_placements[:, None, :],
        axis=2,
    )
    blocks_at_positions = np.cumsum(is_prime, axis=1, dtype=np.int64)
    return np.ascontiguousarray(
        blocks_at_positions[~is_prime].reshape(
            placement_count,
            doubleprime_count,
        )
    )


def compile_symmetrization_bridge_targets(
    prime_placements: object,
    *,
    d_doubleprime: object,
    grade: object,
) -> np.ndarray:
    """Return one exact compact bridge target map."""
    prime_count, doubleprime_count = _grade(grade)
    alphabet_size = _non_negative_int(
        d_doubleprime,
        name="d_doubleprime",
    )
    if alphabet_size == 0:
        raise ValueError("d_doubleprime must be positive, got 0.")
    placements = _prime_placement_matrix(
        prime_placements,
        prime_count=prime_count,
        doubleprime_count=doubleprime_count,
    )
    placement_count = int(placements.shape[0])
    word_count = alphabet_size**doubleprime_count
    source_count = placement_count * word_count
    if source_count > _MAX_INT64:
        raise OverflowError(
            f"bridge source count {source_count} exceeds the int64 plan limit."
        )
    part_count = (prime_count + 1) * alphabet_size
    output_rank_count = comb(
        doubleprime_count + part_count - 1,
        doubleprime_count,
    )
    if output_rank_count - 1 > _MAX_UINT64:
        raise OverflowError("bridge target ranks exceed the uint64 plan limit.")
    target_dtype = _unsigned_index_dtype(output_rank_count - 1)

    if doubleprime_count == 0:
        return _readonly(np.zeros(source_count, dtype=target_dtype))

    blocks = _doubleprime_block_matrix(
        placements,
        doubleprime_count=doubleprime_count,
    )
    binomial = _binomial_table(
        max(part_count - 2 + doubleprime_count, 0),
        max_column=max(part_count - 1, 0),
        max_complement=doubleprime_count,
    )
    word_powers = np.asarray(
        [
            alphabet_size**power
            for power in range(doubleprime_count - 1, -1, -1)
        ],
        dtype=np.int64,
    )
    separators = np.arange(1, part_count, dtype=np.int64)
    targets = np.empty(source_count, dtype=target_dtype)
    row_workspace = part_count + 2 * doubleprime_count
    batch_size = max(1, _VECTOR_WORKSPACE_ENTRIES // row_workspace)
    count_dtype = _unsigned_index_dtype(doubleprime_count)

    for start in range(0, source_count, batch_size):
        stop = min(start + batch_size, source_count)
        source_ranks = np.arange(start, stop, dtype=np.int64)
        placement_ranks = source_ranks // word_count
        word_ranks = source_ranks % word_count
        letters = (
            word_ranks[:, None] // word_powers[None, :]
        ) % alphabet_size
        bins = blocks[placement_ranks] * alphabet_size + letters
        multiplicities = np.zeros(
            (stop - start, part_count),
            dtype=count_dtype,
        )
        rows = np.arange(stop - start, dtype=np.int64)[:, None]
        np.add.at(multiplicities, (rows, bins), 1)
        prefixes = np.cumsum(
            multiplicities[:, :-1],
            axis=1,
            dtype=np.int64,
        )
        ranks = np.sum(
            binomial[separators - 1 + prefixes, separators],
            axis=1,
            dtype=np.uint64,
        )
        if np.any(ranks >= output_rank_count):
            raise AssertionError(
                "bridge builder produced an invalid quotient rank."
            )
        targets[start:stop] = ranks

    return _readonly(targets)


__all__ = []
