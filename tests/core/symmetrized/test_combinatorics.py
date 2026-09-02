from itertools import product
from math import comb

import pytest

from tensordev.core.bigraded.symmetrized.combinatorics import (
    multiset_placement_count,
    multiset_placement_rank,
    multiset_placement_unrank,
    multiset_placements,
)


def _weak_compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for suffix in _weak_compositions(total - first, parts - 1):
            yield (first,) + suffix


def _note_rank(flattened):
    prefix_sum = 0
    rank = 0
    for separator_index, entry in enumerate(flattened[:-1], start=1):
        prefix_sum += entry
        rank += comb(separator_index - 1 + prefix_sum, separator_index)
    return rank


def _as_blocks(flattened, width):
    return tuple(
        tuple(flattened[start : start + width])
        for start in range(0, len(flattened), width)
    )


@pytest.mark.parametrize(
    ("d_doubleprime", "n", "m"),
    tuple(product(range(1, 4), range(4), range(5))),
)
def test_multiset_placement_combinatorics_exhaustively(d_doubleprime, n, m):
    parts = (n + 1) * d_doubleprime
    expected_count = comb(m + parts - 1, m)
    expected_flattened = tuple(
        sorted(_weak_compositions(m, parts), key=_note_rank)
    )
    expected = tuple(
        _as_blocks(flattened, d_doubleprime)
        for flattened in expected_flattened
    )

    placements = multiset_placements(d_doubleprime, (n, m))

    assert multiset_placement_count(d_doubleprime, (n, m)) == expected_count
    assert len(placements) == expected_count
    assert placements == expected
    assert len(set(placements)) == expected_count
    assert tuple(map(multiset_placement_rank, placements)) == tuple(
        range(expected_count)
    )
    assert tuple(
        multiset_placement_unrank(
            rank,
            d_doubleprime=d_doubleprime,
            grade=(n, m),
        )
        for rank in range(expected_count)
    ) == placements


def test_three_part_colex_order_is_not_lexicographic_order():
    assert multiset_placements(3, (0, 2)) == (
        ((0, 0, 2),),
        ((0, 1, 1),),
        ((1, 0, 1),),
        ((0, 2, 0),),
        ((1, 1, 0),),
        ((2, 0, 0),),
    )


def test_k_one_boundary_uses_the_unique_normal_form():
    for m in range(8):
        placement = ((m,),)
        assert multiset_placement_count(1, (0, m)) == 1
        assert multiset_placements(1, (0, m)) == (placement,)
        assert multiset_placement_rank(placement) == 0
        assert multiset_placement_unrank(
            0,
            d_doubleprime=1,
            grade=(0, m),
        ) == placement


def test_rank_and_count_use_python_arbitrary_precision_integers():
    d_doubleprime = 5
    grade = (9, 50)
    count = multiset_placement_count(d_doubleprime, grade)
    placement = multiset_placement_unrank(
        count - 1,
        d_doubleprime=d_doubleprime,
        grade=grade,
    )

    assert isinstance(count, int)
    assert count > 2**63
    assert multiset_placement_rank(placement) == count - 1


@pytest.mark.parametrize("value", (True, 1.5, "2", None))
def test_dimension_rejects_non_integer_values(value):
    with pytest.raises(TypeError, match="d_doubleprime"):
        multiset_placement_count(value, (1, 1))
    with pytest.raises(TypeError, match="d_doubleprime"):
        multiset_placements(value, (1, 1))
    with pytest.raises(TypeError, match="d_doubleprime"):
        multiset_placement_unrank(
            0,
            d_doubleprime=value,
            grade=(1, 1),
        )


@pytest.mark.parametrize("value", (0, -1))
def test_dimension_must_be_positive(value):
    with pytest.raises(ValueError, match="positive"):
        multiset_placement_count(value, (1, 1))


@pytest.mark.parametrize(
    "grade",
    (
        1,
        (1,),
        (1, 2, 3),
        (True, 1),
        (1, False),
        (1.0, 1),
        (1, "1"),
    ),
)
def test_grade_validation(grade):
    with pytest.raises(TypeError, match="grade"):
        multiset_placement_count(2, grade)


@pytest.mark.parametrize("grade", ((-1, 0), (0, -1)))
def test_grade_entries_must_be_non_negative(grade):
    with pytest.raises(ValueError, match="non-negative"):
        multiset_placement_count(2, grade)


@pytest.mark.parametrize(
    ("blocks", "error", "message"),
    (
        (1, TypeError, "iterable"),
        ("bad", TypeError, "iterable"),
        ((), ValueError, "at least one"),
        (((),), ValueError, "positive width"),
        (((0, 1), (2,)), ValueError, "rectangular"),
        (((0, -1),), ValueError, "non-negative"),
        (((0, True),), TypeError, "integer"),
        (((0, 1.5),), TypeError, "integer"),
    ),
)
def test_rank_validates_the_complete_normal_form(blocks, error, message):
    with pytest.raises(error, match=message):
        multiset_placement_rank(blocks)


@pytest.mark.parametrize("rank", (True, 1.5, "0", None))
def test_unrank_rejects_non_integer_rank(rank):
    with pytest.raises(TypeError, match="rank"):
        multiset_placement_unrank(
            rank,
            d_doubleprime=2,
            grade=(1, 1),
        )


def test_unrank_rejects_rank_outside_the_grade():
    count = multiset_placement_count(2, (1, 2))
    with pytest.raises(ValueError, match="non-negative"):
        multiset_placement_unrank(
            -1,
            d_doubleprime=2,
            grade=(1, 2),
        )
    with pytest.raises(ValueError, match="outside"):
        multiset_placement_unrank(
            count,
            d_doubleprime=2,
            grade=(1, 2),
        )
