from dataclasses import replace

import pytest

from tensordev.core.bigraded import BigradedPlanStore, BigradedSpec


def test_active_layout_contains_only_active_rectangle_and_shares_plans():
    store = BigradedPlanStore((2, 3), (3, 4))
    full = store.resolve()
    active = store.resolve((1, 2))

    assert active.truncation == (1, 2)
    assert all(n <= 1 and m <= 2 for n, m in active.grades)
    assert len(active.grade_plans) == (1 + 1) * (2 + 1)
    assert active.grade_plan((1, 2)) is full.grade_plan((1, 2))
    assert not active.contains((2, 0))


def test_resolve_is_cached_by_truncation_and_scalar_policy():
    store = BigradedPlanStore((1, 1), (2, 2))
    a = store.resolve((1, 2))
    b = store.resolve((1, 2))
    positive = store.resolve((1, 2), include_scalar=False)
    assert a is b
    assert positive is not a
    assert positive.with_scalar(True) is a
    assert (0, 0) not in positive.grades


def test_layout_exposes_symmetrization_neutral_rank_count():
    ordered = BigradedPlanStore((2, 3), (1, 2)).resolve()
    quotient = replace(
        ordered,
        spec=BigradedSpec(
            2,
            3,
            ordered.truncation,
            coordinates=ordered.coordinates,
            include_scalar=ordered.include_scalar,
            partially_symmetrized=True,
        ),
    )

    assert ordered.partially_symmetrized is False
    assert ordered.rank_count((1, 2)) == ordered.placement_count((1, 2)) == 3
    assert quotient.partially_symmetrized is True
    assert quotient.rank_count((1, 2)) == 21
    with pytest.raises(ValueError, match="only defined for ordered layouts.*rank_count"):
        quotient.placement_count((1, 2))


def test_resolve_rejects_capacity_overflow():
    store = BigradedPlanStore((1, 1), (2, 3))
    with pytest.raises(ValueError, match="exceeds plan-store capacity"):
        store.resolve((3, 1))
    with pytest.raises(ValueError, match="exceeds plan-store capacity"):
        store.resolve((1, 4))


def test_product_splits_follow_canonical_left_grade_order():
    layout = BigradedPlanStore((1, 1), (2, 2)).resolve()
    assert layout.product_splits((1, 1)) == (
        ((0, 0), (1, 1)),
        ((1, 0), (0, 1)),
        ((0, 1), (1, 0)),
        ((1, 1), (0, 0)),
    )


def test_positive_layout_product_splits_remain_complete_for_input_filtering():
    layout = BigradedPlanStore((1, 1), (2, 2)).resolve(
        (2, 2), include_scalar=False
    )
    assert layout.product_splits((1, 0)) == (
        ((0, 0), (1, 0)),
        ((1, 0), (0, 0)),
    )
    assert layout.product_splits((1, 1)) == (
        ((0, 0), (1, 1)),
        ((1, 0), (0, 1)),
        ((0, 1), (1, 0)),
        ((1, 1), (0, 0)),
    )


def test_active_concat_plan_requires_active_inputs_and_output():
    layout = BigradedPlanStore((1, 1), (2, 2)).resolve((1, 1))
    assert layout.concat_plan((1, 0), (0, 1)).output_grade == (1, 1)
    with pytest.raises(KeyError, match="left bidegree"):
        layout.concat_plan((2, 0), (0, 0))
    with pytest.raises(KeyError, match="outside active"):
        layout.concat_plan((1, 0), (1, 0))
