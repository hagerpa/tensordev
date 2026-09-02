"""Exactness tests for compiled primitive shear support tables."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

import tensordev.core.shear.bigraded as bigraded_shear_module
from tensordev.core.bigraded.precompute import BigradedPlanStore, colex_rank
from tensordev.core.shear._compiled_support import (
    ShearTransformSupport,
    compile_forward_shear_support,
    compile_inverse_shear_support,
    shear_colex_rank_pairs,
)
from tensordev.core.shear.bigraded import (
    BigradedShearPlanStore,
    _transform_rank_groups,
)
from tensordev.core.shear.symbolic import (
    clear_symbolic_caches,
    psi_inverse_support,
    psi_support,
    transform_placement_sign,
)


def _placement_mask(positions) -> int:
    mask = 0
    for position in positions:
        mask |= 1 << position
    return mask


def _mask_positions(mask: int, degree: int) -> tuple[int, ...]:
    return tuple(position for position in range(degree) if mask & (1 << position))


def _symbolic_groups(
    prime_count: int,
    doubleprime_count: int,
    *,
    inverse: bool,
):
    grouped = {}
    support = psi_inverse_support if inverse else psi_support
    for term in support(prime_count, doubleprime_count):
        grouped.setdefault(term.doubleprime_permutation, []).append(
            (
                _placement_mask(term.output_prime_positions),
                _placement_mask(term.input_prime_positions),
            )
        )
    return tuple(grouped.items())


def _compiled_groups(table: ShearTransformSupport):
    groups = []
    for group, permutation in enumerate(table.doubleprime_permutations):
        term_slice = table.group_slice(group)
        groups.append(
            (
                tuple(map(int, permutation)),
                list(
                    zip(
                        map(int, table.output_masks[term_slice]),
                        map(int, table.input_masks[term_slice]),
                    )
                ),
            )
        )
    return tuple(groups)


def _inverse_symbolic_coefficients(
    prime_count: int,
    doubleprime_count: int,
):
    grouped = {}
    for term in psi_inverse_support(prime_count, doubleprime_count):
        grouped.setdefault(term.doubleprime_permutation, []).append(term.coefficient)
    return tuple(grouped.values())


@pytest.mark.parametrize(
    ("prime_count", "doubleprime_count"),
    [(n, m) for n in range(5) for m in range(5) if n + m <= 7] + [(5, 3), (5, 4)],
)
def test_compiled_forward_support_exactly_matches_symbolic_support(
    prime_count,
    doubleprime_count,
):
    table = compile_forward_shear_support(prime_count, doubleprime_count)

    assert _compiled_groups(table) == _symbolic_groups(
        prime_count,
        doubleprime_count,
        inverse=False,
    )
    clear_symbolic_caches()


@pytest.mark.parametrize(
    ("prime_count", "doubleprime_count"),
    [(n, m) for n in range(6) for m in range(6) if n + m <= 8] + [(5, 4)],
)
def test_compiled_inverse_support_exactly_matches_symbolic_support(
    prime_count,
    doubleprime_count,
):
    table = compile_inverse_shear_support(
        prime_count,
        doubleprime_count,
    )

    assert _compiled_groups(table) == _symbolic_groups(
        prime_count,
        doubleprime_count,
        inverse=True,
    )
    derived_coefficients = []
    for group in range(table.group_count):
        term_slice = table.group_slice(group)
        derived_coefficients.append(
            [
                transform_placement_sign(
                    _mask_positions(int(output_mask), table.total_degree),
                    _mask_positions(int(input_mask), table.total_degree),
                )
                for output_mask, input_mask in zip(
                    table.output_masks[term_slice],
                    table.input_masks[term_slice],
                )
            ]
        )
    assert tuple(derived_coefficients) == _inverse_symbolic_coefficients(
        prime_count,
        doubleprime_count,
    )
    clear_symbolic_caches()


@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize(
    ("prime_count", "doubleprime_count"),
    [(0, 0), (0, 4), (1, 3), (3, 2), (5, 4)],
)
def test_compiled_colex_pairs_match_bigraded_transform_groups(
    prime_count,
    doubleprime_count,
    inverse,
):
    compiler = (
        compile_inverse_shear_support if inverse else compile_forward_shear_support
    )
    table = compiler(prime_count, doubleprime_count)
    encoded = shear_colex_rank_pairs(table)
    placement_count, expected = _transform_rank_groups(
        (prime_count, doubleprime_count),
        inverse=inverse,
    )

    assert placement_count == comb(
        prime_count + doubleprime_count,
        prime_count,
    )
    assert len(expected) == table.group_count
    for group, (expected_permutation, expected_pairs) in enumerate(expected.items()):
        row_permutation = tuple(map(int, table.doubleprime_permutations[group]))
        inverse = tuple(
            row_permutation.index(axis) for axis in range(doubleprime_count)
        )
        assert expected_permutation == (
            tuple(range(prime_count)) + tuple(prime_count + axis for axis in inverse)
        )
        assert encoded[table.group_slice(group)].tolist() == expected_pairs
    clear_symbolic_caches()


def test_compiled_colex_pairs_are_directly_recoverable_from_masks():
    table = compile_forward_shear_support(4, 3)
    encoded = shear_colex_rank_pairs(table)
    placement_count = comb(table.total_degree, table.prime_count)

    expected = []
    for output_mask, input_mask in zip(
        table.output_masks,
        table.input_masks,
    ):
        output_rank = colex_rank(_mask_positions(int(output_mask), table.total_degree))
        input_rank = colex_rank(_mask_positions(int(input_mask), table.total_degree))
        expected.append(output_rank * placement_count + input_rank)
    assert encoded.tolist() == expected


def test_compiled_support_table_contract_and_large_exact_counts():
    forward = compile_forward_shear_support(10, 4)
    inverse = compile_inverse_shear_support(10, 4)

    assert isinstance(forward, ShearTransformSupport)
    assert forward.total_degree == inverse.total_degree == 14
    assert forward.term_count == 1_479_478
    assert inverse.term_count == 13_441
    assert forward.group_count == 24
    assert inverse.group_count == 13
    for table in (forward, inverse):
        assert table.output_masks.dtype == np.dtype(np.uint64)
        assert table.input_masks.dtype == np.dtype(np.uint64)
        assert table.doubleprime_permutations.dtype == np.dtype(np.uint8)
        assert table.group_offsets.dtype == np.dtype(np.int64)
        assert table.group_offsets[0] == 0
        assert table.group_offsets[-1] == table.term_count
        assert np.all(np.diff(table.group_offsets) > 0)
        assert table.memory_bytes() == sum(
            array.nbytes
            for array in (
                table.output_masks,
                table.input_masks,
                table.doubleprime_permutations,
                table.group_offsets,
            )
        )
        for array in (
            table.output_masks,
            table.input_masks,
            table.doubleprime_permutations,
            table.group_offsets,
        ):
            assert not array.flags.writeable


def test_bigraded_transform_construction_does_not_expand_symbolic_support(
    monkeypatch,
):
    def forbidden_symbolic_fallback(*args, **kwargs):
        raise AssertionError("symbolic transform support was expanded")

    monkeypatch.setattr(
        bigraded_shear_module,
        "_transform_rank_groups",
        forbidden_symbolic_fallback,
    )
    plans = BigradedShearPlanStore(
        BigradedPlanStore((1, 1), (4, 3))
    )

    assert plans.forward_plans[(4, 3)].key_plans
    assert plans.inverse_plans[(4, 3)].key_plans


@pytest.mark.parametrize(
    ("args", "error", "match"),
    [
        ((True, 1), TypeError, "prime_count"),
        ((1.0, 1), TypeError, "prime_count"),
        ((-1, 1), ValueError, "prime_count"),
        ((1, True), TypeError, "doubleprime_count"),
        ((1, -1), ValueError, "doubleprime_count"),
        ((65, 0), ValueError, "total degree <= 64"),
    ],
)
@pytest.mark.parametrize(
    "compiler",
    (compile_forward_shear_support, compile_inverse_shear_support),
)
def test_compiled_support_rejects_invalid_grades(
    compiler,
    args,
    error,
    match,
):
    with pytest.raises(error, match=match):
        compiler(*args)


def test_compiled_support_group_slice_validation():
    table = compile_forward_shear_support(1, 2)

    with pytest.raises(TypeError, match="group must be an integer"):
        table.group_slice(True)
    with pytest.raises(IndexError, match="outside"):
        table.group_slice(table.group_count)


def test_compiled_colex_pairs_reject_wrong_table_type():
    with pytest.raises(TypeError, match="ShearTransformSupport"):
        shear_colex_rank_pairs(object())
