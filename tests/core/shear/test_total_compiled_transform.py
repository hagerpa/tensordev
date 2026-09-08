"""Regression contracts for compiled dense-total shear transforms."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from tensordev.core.shear import total as total_module
from tensordev.core.shear.total import (
    TotalShearPlanBuilder,
    apply_total_masked_permutation_plan,
)


def _assert_exact_structure(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
        return
    if is_dataclass(expected) and not isinstance(expected, type):
        assert type(actual) is type(expected)
        for field in fields(expected):
            _assert_exact_structure(
                getattr(actual, field.name),
                getattr(expected, field.name),
            )
        return
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_exact_structure(actual_item, expected_item)
        return
    assert actual == expected


@pytest.mark.parametrize("degree", range(8))
@pytest.mark.parametrize("inverse", (False, True))
def test_compiled_total_transform_exactly_matches_symbolic_plan(
    degree,
    inverse,
):
    builder = TotalShearPlanBuilder((2, 1))
    compiled = builder.transform(degree, inverse=inverse)
    symbolic = builder.build(
        degree=degree,
        records=builder._transform_records(degree, inverse=inverse),
        source_domains=("full",) * degree,
        coefficient_mode="mask_parity" if inverse else "positive",
        with_transpose=True,
    )

    _assert_exact_structure(compiled, symbolic)
    assert compiled.memory_bytes_by_category() == (
        builder._expected_transform_memory(degree, inverse=inverse)
    )


def test_total_transform_and_memory_estimator_do_not_expand_symbolic_support(
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("compiled total transform expanded symbolic support")

    monkeypatch.setattr(total_module, "psi_support", forbidden)
    monkeypatch.setattr(total_module, "psi_inverse_support", forbidden)
    builder = TotalShearPlanBuilder((1, 2))

    for inverse in (False, True):
        plan = builder.transform(7, inverse=inverse)
        expected = builder._expected_transform_memory(7, inverse=inverse)
        assert plan.memory_bytes_by_category() == expected


def test_compiled_total_transform_retains_independent_read_only_arrays():
    builder = TotalShearPlanBuilder((1, 1))
    plan = builder.transform(9, inverse=False)
    arrays = (
        plan.permutation_table,
        plan.inverse_permutation_table,
        plan.term_output_masks,
        plan.term_input_masks,
        plan.term_permutation_ids,
    )

    assert all(not array.flags.writeable for array in arrays)
    assert len({id(array) for array in arrays}) == len(arrays)
    for array in arrays:
        owner = array
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        assert owner.nbytes == array.nbytes


def test_empty_record_plan_retains_the_empty_orientation_contract():
    builder = TotalShearPlanBuilder((1, 1))
    plan = builder.build(
        degree=1,
        records=(),
        source_domains=("full",),
        coefficient_mode="positive",
        with_transpose=True,
    )

    assert plan.term_count == 0
    assert plan.permutation_table.shape == (0, 1)
    for orientation in (plan.forward, plan.transpose):
        assert orientation.groups == ()
        assert orientation.group_masks.shape == (0,)
        np.testing.assert_array_equal(orientation.group_offsets, np.asarray([0]))
        assert not orientation.group_masks.flags.writeable
        assert not orientation.group_offsets.flags.writeable
    assert plan.forward.term_order is None
    assert plan.transpose.term_order.shape == (0,)
    assert not plan.transpose.term_order.flags.writeable

    source = np.asarray([2.0, -3.0])
    np.testing.assert_array_equal(
        apply_total_masked_permutation_plan(np, source, plan),
        np.zeros_like(source),
    )
    np.testing.assert_array_equal(
        apply_total_masked_permutation_plan(np, source, plan, transpose=True),
        np.zeros_like(source),
    )
    assert plan.memory_bytes_by_category() == (
        builder._expected_memory_bytes_by_category(
            degree=1,
            records=(),
            source_domains=("full",),
            with_transpose=True,
        )
    )


def test_total_transform_rejects_unrepresentable_mask_degree_before_expansion(
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("degree validation expanded symbolic support")

    monkeypatch.setattr(total_module, "psi_support", forbidden)
    monkeypatch.setattr(total_module, "psi_inverse_support", forbidden)
    builder = TotalShearPlanBuilder((1, 1))

    for method in (builder.transform, builder._expected_transform_memory):
        with pytest.raises(ValueError, match=r"degree must be <= 64, got 65"):
            method(65, inverse=False)


@pytest.mark.parametrize("dims", ((1, 1), (1, 2)))
@pytest.mark.parametrize("inverse", (False, True))
def test_vectorized_transform_memory_is_exact_at_degree_nine(dims, inverse):
    builder = TotalShearPlanBuilder(dims)
    plan = builder.transform(9, inverse=inverse)

    assert plan.memory_bytes_by_category() == (
        builder._expected_transform_memory(9, inverse=inverse)
    )
