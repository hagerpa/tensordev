"""Allocation-free exactness tests for ordered-bidegree shear estimates."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

import tensordev as td
import tensordev.core.shear.bigraded as bigraded_module
from tensordev.core.bigraded.precompute import BigradedPlanStore
from tensordev.core.shear._compiled_support import (
    compile_forward_shear_support,
    compile_inverse_shear_support,
)
from tensordev.core.shear.bigraded import (
    BigradedShearPlanStore,
    _expected_bigraded_shear_plan_memory_bytes_by_category,
    _transform_rank_matrix_group_count,
)
from tensordev.core.utils.precompute import _unsigned_index_dtype


def _actual_rank_matrix_groups(grade, *, inverse):
    n, m = grade
    placement_count = comb(n + m, n)
    itemsize = np.dtype(_unsigned_index_dtype(max(placement_count**2 - 1, 0))).itemsize
    compiler = (
        compile_inverse_shear_support if inverse else compile_forward_shear_support
    )
    support = compiler(n, m)
    matrix_bytes = placement_count**2 * np.dtype(np.int8).itemsize
    return tuple(
        tuple(map(int, support.doubleprime_permutations[group]))
        for group in range(support.group_count)
        if matrix_bytes
        <= 2
        * int(support.group_offsets[group + 1] - support.group_offsets[group])
        * itemsize
    )


@pytest.mark.parametrize("inverse", (False, True))
def test_rank_matrix_count_formula_covers_every_eligible_grade(inverse):
    for n in range(10):
        for m in range(10):
            placement_count = comb(n + m, n)
            if placement_count > 64:
                continue
            itemsize = np.dtype(
                _unsigned_index_dtype(max(placement_count**2 - 1, 0))
            ).itemsize

            expected_count = _transform_rank_matrix_group_count(
                (n, m),
                inverse=inverse,
                placement_count=placement_count,
                transform_itemsize=itemsize,
            )
            actual_groups = _actual_rank_matrix_groups((n, m), inverse=inverse)

            assert expected_count == len(actual_groups)
            if n > 1 and m > 1:
                expected_groups = (
                    (tuple(range(m)),) if expected_count and not inverse else ()
                )
                assert actual_groups == expected_groups


@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("grade", [(1, 15), (1, 16), (15, 1), (16, 1)])
def test_rank_matrix_count_formula_covers_index_width_boundary(grade, inverse):
    placement_count = comb(sum(grade), grade[0])
    itemsize = np.dtype(_unsigned_index_dtype(max(placement_count**2 - 1, 0))).itemsize

    assert _transform_rank_matrix_group_count(
        grade,
        inverse=inverse,
        placement_count=placement_count,
        transform_itemsize=itemsize,
    ) == len(_actual_rank_matrix_groups(grade, inverse=inverse))


@pytest.mark.parametrize(
    ("dims", "truncation"),
    [
        ((1, 1), (2, 2)),
        ((1, 2), (4, 3)),
        ((2, 1), (6, 3)),
        ((1, 2), (10, 4)),
    ],
)
def test_estimator_categories_are_byte_exact_against_store(dims, truncation):
    store = BigradedShearPlanStore(BigradedPlanStore(dims, truncation))

    assert dict(
        _expected_bigraded_shear_plan_memory_bytes_by_category(
            dims,
            truncation,
        )
    ) == dict(store.memory_bytes_by_category())


def test_target_public_estimator_is_byte_exact_against_full_core():
    kwargs = {
        "dims": (1, 2),
        "max_trunc": (10, 4),
        "precompute_shuffle": True,
        "representation": "ordered",
        "coordinates": "shear",
    }
    previous_pair = td.get_default_core_pair()
    try:
        core = td.set_default_core(**kwargs)
        breakdown = td.core_expected_memory(
            **kwargs,
            unit="bytes",
            breakdown=True,
        )
    finally:
        td.set_default_core(*previous_pair)

    assert breakdown == {
        **{
            name: float(value)
            for name, value in core.memory_bytes_by_category().items()
        },
        "total": float(core.memory_bytes()),
    }
    assert breakdown["total"] == 21_226_197.0


def test_target_estimator_does_not_expand_or_allocate_support(monkeypatch):
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("memory estimation expanded a support array")

    for name in (
        "_transform_rank_groups",
        "psi_support",
        "psi_inverse_support",
        "compile_forward_shear_support",
        "compile_inverse_shear_support",
    ):
        monkeypatch.setattr(bigraded_module, name, forbidden)
    for name in ("array", "asarray", "empty", "ones", "zeros"):
        monkeypatch.setattr(bigraded_module.np, name, forbidden)

    shear_categories = dict(
        _expected_bigraded_shear_plan_memory_bytes_by_category(
            (1, 2),
            (10, 4),
        )
    )
    assert shear_categories == {
        "transform_rank_pairs": 11_803_189,
        "transform_dense_permutations": 39_480,
        "transform_parities": 4_367,
        "transform_rank_matrices": 14_400,
        "generator_rank_pairs": 19_727,
        "generator_dense_permutations": 14_680,
    }
    assert (
        td.core_expected_memory(
            dims=(1, 2),
            max_trunc=(10, 4),
            precompute_shuffle=True,
            representation="ordered",
            coordinates="shear",
            unit="bytes",
        )
        == 21_226_197.0
    )
