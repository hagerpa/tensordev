"""Host-side multiset-placement combinatorics.

The routines in this module implement the separator/colexicographic layout
used for partially symmetrized bidegree blocks.  They deliberately operate on
Python integers and immutable tuples only: device arrays belong in the plan
layer built on top of these helpers.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb
from numbers import Integral
from typing import Iterable, Sequence, Tuple


Bidegree = Tuple[int, int]
BlockMultiplicities = Tuple[Tuple[int, ...], ...]


__all__ = [
    "multiset_placement_count",
    "multiset_placement_rank",
    "multiset_placement_unrank",
    "multiset_placements",
]


def _integer(value: object, *, name: str, positive: bool = False) -> int:
    qualifier = "positive" if positive else "non-negative"
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a {qualifier} integer, got {value!r}.")
    result = int(value)
    lower = 1 if positive else 0
    if result < lower:
        raise ValueError(f"{name} must be {qualifier}, got {result}.")
    return result


def _bidegree(value: object, *, name: str = "grade") -> Bidegree:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(
            f"{name} must be a pair of non-negative integers, got {value!r}."
        )
    return (
        _integer(value[0], name=f"{name}[0]"),
        _integer(value[1], name=f"{name}[1]"),
    )


def _tuple_from_iterable(value: object, *, name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an iterable, got {value!r}.")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable, got {value!r}.") from exc


def _block_multiplicities(
    value: Iterable[Sequence[int]],
) -> BlockMultiplicities:
    raw_blocks = _tuple_from_iterable(value, name="block_multiplicities")
    if not raw_blocks:
        raise ValueError("block_multiplicities must contain at least one block.")

    normalized = []
    width = None
    for block_index, raw_block in enumerate(raw_blocks):
        block = _tuple_from_iterable(
            raw_block,
            name=f"block_multiplicities[{block_index}]",
        )
        if width is None:
            width = len(block)
            if width == 0:
                raise ValueError(
                    "block_multiplicities blocks must have positive width."
                )
        elif len(block) != width:
            raise ValueError(
                "block_multiplicities must be rectangular; "
                f"block 0 has width {width}, while block {block_index} has "
                f"width {len(block)}."
            )
        normalized.append(
            tuple(
                _integer(
                    entry,
                    name=f"block_multiplicities[{block_index}][{entry_index}]",
                )
                for entry_index, entry in enumerate(block)
            )
        )
    return tuple(normalized)


@lru_cache(maxsize=None)
def _count(d_doubleprime: int, n: int, m: int) -> int:
    parts = (n + 1) * d_doubleprime
    return comb(m + parts - 1, m)


def multiset_placement_count(d_doubleprime: int, grade: Bidegree) -> int:
    """Return ``|MPl(n, m)|`` for ``grade=(n, m)``.

    ``d_doubleprime`` is the number of commuting letters.  The result is a
    Python integer, including when the count exceeds fixed-width ranges.
    """

    normalized_dimension = _integer(
        d_doubleprime,
        name="d_doubleprime",
        positive=True,
    )
    n, m = _bidegree(grade)
    return _count(normalized_dimension, n, m)


def multiset_placement_rank(
    block_multiplicities: Iterable[Sequence[int]],
) -> int:
    """Rank a multiset placement by the separator/colex formula.

    The outer sequence lists the ``n+1`` blocks and each inner sequence is a
    multiplicity vector over the ordered double-prime alphabet.  Consequently
    the bidegree and alphabet dimension are determined by the input itself.
    """

    blocks = _block_multiplicities(block_multiplicities)
    return _rank_normalized_blocks(blocks)


def _rank_normalized_blocks(blocks: BlockMultiplicities) -> int:
    """Rank an already validated normal form without repeating validation."""

    flattened = tuple(entry for block in blocks for entry in block)

    prefix_sum = 0
    rank = 0
    for separator_index, entry in enumerate(flattened[:-1], start=1):
        prefix_sum += entry
        rank += comb(separator_index - 1 + prefix_sum, separator_index)
    return rank


def _colex_unrank(rank: int, subset_size: int, length: int) -> Tuple[int, ...]:
    """Unrank a fixed-size subset of ``range(length)`` in colex order."""

    if subset_size == 0:
        return ()

    remaining = rank
    positions = [0] * subset_size
    upper = length - 1
    for subset_index in range(subset_size, 0, -1):
        lower = subset_index - 1
        high = upper
        while lower < high:
            candidate = (lower + high + 1) // 2
            if comb(candidate, subset_index) <= remaining:
                lower = candidate
            else:
                high = candidate - 1
        positions[subset_index - 1] = lower
        remaining -= comb(lower, subset_index)
        upper = lower - 1
    return tuple(positions)


def _unrank(
    rank: int,
    *,
    d_doubleprime: int,
    n: int,
    m: int,
) -> BlockMultiplicities:
    parts = (n + 1) * d_doubleprime
    if parts == 1:
        flattened = (m,)
    else:
        subset_size = parts - 1
        length = m + subset_size
        separators = _colex_unrank(rank, subset_size, length)

        entries = [0] * parts
        entries[0] = separators[0]
        for index in range(1, parts - 1):
            entries[index] = separators[index] - separators[index - 1] - 1
        entries[-1] = length - separators[-1] - 1
        flattened = tuple(entries)

    return tuple(
        tuple(flattened[start : start + d_doubleprime])
        for start in range(0, parts, d_doubleprime)
    )


def multiset_placement_unrank(
    rank: int,
    *,
    d_doubleprime: int,
    grade: Bidegree,
) -> BlockMultiplicities:
    """Return the unique multiset placement having ``rank`` at ``grade``."""

    normalized_rank = _integer(rank, name="rank")
    normalized_dimension = _integer(
        d_doubleprime,
        name="d_doubleprime",
        positive=True,
    )
    n, m = _bidegree(grade)
    count = _count(normalized_dimension, n, m)
    if normalized_rank >= count:
        raise ValueError(
            f"rank={normalized_rank} is outside [0, {count}) for "
            f"d_doubleprime={normalized_dimension} and grade={(n, m)}."
        )
    return _unrank(
        normalized_rank,
        d_doubleprime=normalized_dimension,
        n=n,
        m=m,
    )


def _placements(
    d_doubleprime: int,
    n: int,
    m: int,
) -> Tuple[BlockMultiplicities, ...]:
    return tuple(
        _unrank(rank, d_doubleprime=d_doubleprime, n=n, m=m)
        for rank in range(_count(d_doubleprime, n, m))
    )


def multiset_placements(
    d_doubleprime: int,
    grade: Bidegree,
) -> Tuple[BlockMultiplicities, ...]:
    """Enumerate all multiset placements in consecutive colex rank order."""

    normalized_dimension = _integer(
        d_doubleprime,
        name="d_doubleprime",
        positive=True,
    )
    n, m = _bidegree(grade)
    return _placements(normalized_dimension, n, m)
