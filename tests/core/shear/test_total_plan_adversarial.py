"""Adversarial contracts for dense-total shear plan compilation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from tensordev.core.shear.jax import JaxShearTotal
from tensordev.core.shear.symbolic import (
    GammaShuffleSupportTerm,
    GeneratorSupportTerm,
    TransformSupportTerm,
)
from tensordev.core.shear.total import (
    TotalMaskedPermutationPlan,
    TotalShearPlanBuilder,
    TotalShearPlanStore,
    _expected_total_shear_memory_bytes_by_category,
    apply_total_masked_permutation_plan,
)


_FORCED_STRATEGIES = (
    ("static", 10**9, 10**9, "transpose_sum"),
    ("flat", 0, 10**9, "flat_gather"),
    ("coefficient", 0, 0, "coefficient_gather"),
)


def _force_strategy(monkeypatch, static_threshold, flat_max_bytes):
    monkeypatch.setattr(
        TotalShearPlanBuilder,
        "STATIC_TERM_THRESHOLD",
        static_threshold,
    )
    monkeypatch.setattr(
        TotalShearPlanBuilder,
        "FLAT_GATHER_MAX_BYTES",
        flat_max_bytes,
    )
    monkeypatch.setattr(TotalShearPlanBuilder, "COEFFICIENT_CHUNK_SIZE", 3)


def _representative_plans(builder):
    return {
        "F": builder.transform(4, inverse=False),
        "G": builder.transform(4, inverse=True),
        "generator": builder.generator(4),
        "shuffle": builder.shuffle(2, 2),
    }


def _representative_memory(builder):
    return {
        "F": builder._expected_transform_memory(4, inverse=False),
        "G": builder._expected_transform_memory(4, inverse=True),
        "generator": builder._expected_generator_memory(4),
        "shuffle": builder._expected_shuffle_memory(2, 2),
    }


def _integer_source(width):
    values = np.arange(width, dtype=np.int32)
    return np.stack((values, values[::-1] - 7))


@pytest.mark.parametrize(
    ("strategy_name", "static_threshold", "flat_max_bytes", "expected"),
    _FORCED_STRATEGIES,
)
def test_forced_total_strategies_are_equivalent_and_exactly_accounted(
    monkeypatch,
    strategy_name,
    static_threshold,
    flat_max_bytes,
    expected,
):
    del strategy_name
    _force_strategy(monkeypatch, 10**9, 10**9)
    reference = _representative_plans(TotalShearPlanBuilder((2, 1)))

    _force_strategy(monkeypatch, static_threshold, flat_max_bytes)
    builder = TotalShearPlanBuilder((2, 1))
    actual = _representative_plans(builder)
    expected_memory = _representative_memory(builder)

    sources = {
        "F": _integer_source(3**4),
        "G": _integer_source(3**4),
        "generator": _integer_source(3**3 * 2),
        "shuffle": _integer_source(3**4),
    }
    for family, plan in actual.items():
        nontrivial = [
            group for group in plan.forward.groups if group.term_count > 1
        ]
        assert nontrivial, f"{family} must exercise a non-direct strategy"
        assert {group.strategy for group in nontrivial} == {expected}
        assert plan.memory_bytes_by_category() == expected_memory[family]

        oracle = apply_total_masked_permutation_plan(
            np, sources[family], reference[family]
        )
        numpy_result = apply_total_masked_permutation_plan(
            np, sources[family], plan
        )
        jax_result = apply_total_masked_permutation_plan(
            jnp, jnp.asarray(sources[family]), plan
        )
        np.testing.assert_array_equal(numpy_result, oracle)
        np.testing.assert_array_equal(np.asarray(jax_result), oracle)

        if family in {"F", "G"}:
            transpose_source = _integer_source(3**4)
            transpose_oracle = apply_total_masked_permutation_plan(
                np,
                transpose_source,
                reference[family],
                transpose=True,
            )
            transpose_result = apply_total_masked_permutation_plan(
                jnp,
                jnp.asarray(transpose_source),
                plan,
                transpose=True,
            )
            np.testing.assert_array_equal(
                np.asarray(transpose_result), transpose_oracle
            )

    store = TotalShearPlanStore(
        (2, 1), 4, precompute_shuffle=True
    )
    estimated = _expected_total_shear_memory_bytes_by_category(
        (2, 1), 4, precompute_shuffle=True
    )
    assert dict(store.memory_bytes_by_category()) == dict(estimated)
    assert store.memory_bytes() == sum(estimated.values())


def _walk_deep(value, path="root"):
    """Walk slotted dataclasses, mappings, containers, and ordinary objects."""
    yield path, value
    if isinstance(value, np.ndarray):
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _walk_deep(
                getattr(value, field.name), f"{path}.{field.name}"
            )
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_deep(item, f"{path}[{key!r}]")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _walk_deep(item, f"{path}[{index}]")
    elif hasattr(value, "__dict__"):
        for name, item in vars(value).items():
            yield from _walk_deep(item, f"{path}.{name}")


def _assert_orientation_integrity(plan, orientation, *, transpose):
    count = plan.term_count
    order = (
        np.arange(count, dtype=np.int64)
        if orientation.term_order is None
        else np.asarray(orientation.term_order, dtype=np.int64)
    )
    np.testing.assert_array_equal(np.sort(order), np.arange(count))
    np.testing.assert_array_equal(
        orientation.group_offsets,
        np.asarray(
            [group.term_start for group in orientation.groups]
            + [count],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        orientation.group_masks,
        np.asarray([group.output_mask for group in orientation.groups]),
    )

    target_masks = (
        plan.term_input_masks if transpose else plan.term_output_masks
    )
    for group in orientation.groups:
        term_ids = order[group.term_start : group.term_stop]
        assert group.term_count == len(term_ids) > 0
        assert all(
            int(target_masks[term_id]) == group.output_mask
            for term_id in term_ids
        )
        expected_radices = tuple(
            plan.d_prime if group.output_mask & (1 << axis)
            else plan.d_doubleprime
            for axis in range(plan.degree)
        )
        assert group.output_radices == expected_radices

        if group.strategy in {"direct", "transpose_sum"}:
            assert group.flat_indices is None
            assert group.affine_offsets is None
            assert group.affine_coefficients is None
            assert group.validity_masks is None
            assert group.coefficient_signs.shape == (group.term_count,)
        elif group.strategy == "flat_gather":
            assert group.flat_indices.shape == (
                group.term_count,
                int(np.prod(group.output_radices)),
            )
            assert group.affine_offsets is None
            assert group.affine_coefficients is None
            assert group.validity_masks is None
            assert group.coefficient_signs.shape == (group.term_count,)
        else:
            assert group.flat_indices is None
            assert group.affine_offsets.shape[1:] == (group.chunk_size,)
            assert group.affine_coefficients.shape[1:] == (
                group.chunk_size,
                plan.degree,
            )
            assert group.validity_masks.shape == group.affine_offsets.shape
            assert group.coefficient_signs.shape == group.affine_offsets.shape
            assert int(group.validity_masks.sum()) == group.term_count


def _assert_plan_integrity(plan: TotalMaskedPermutationPlan):
    assert plan.permutation_table.shape == (
        plan.permutation_count,
        plan.degree,
    )
    assert plan.inverse_permutation_table.shape == plan.permutation_table.shape
    assert plan.term_output_masks.shape == (plan.term_count,)
    assert plan.term_input_masks.shape == (plan.term_count,)
    assert plan.term_permutation_ids.shape == (plan.term_count,)
    assert np.all(plan.term_permutation_ids < plan.permutation_count)
    for permutation, inverse in zip(
        plan.permutation_table, plan.inverse_permutation_table
    ):
        np.testing.assert_array_equal(
            np.asarray(permutation)[np.asarray(inverse)],
            np.arange(plan.degree),
        )
    _assert_orientation_integrity(plan, plan.forward, transpose=False)
    if plan.transpose is not None:
        _assert_orientation_integrity(plan, plan.transpose, transpose=True)


def test_packed_total_plan_store_is_structurally_sound_and_deeply_owned(
    monkeypatch,
):
    _force_strategy(monkeypatch, 0, 0)
    store = TotalShearPlanStore((2, 1), 4, precompute_shuffle=True)
    plans = tuple(store.forward_plans.values()) + tuple(
        store.inverse_plans.values()
    ) + tuple(store.generator_plans.values()) + tuple(store.shuffle_plans.values())
    for plan in plans:
        _assert_plan_integrity(plan)

    walked = tuple(_walk_deep(store))
    forbidden_symbolic = (
        TransformSupportTerm,
        GeneratorSupportTerm,
        GammaShuffleSupportTerm,
    )
    assert not any(isinstance(value, forbidden_symbolic) for _, value in walked)
    assert not any(
        ".bigraded" in type(value).__module__.lower()
        or "bigraded" in type(value).__name__.lower()
        for _, value in walked
    )
    assert not any("rank" in path.lower() for path, _ in walked)

    arrays = [value for _, value in walked if isinstance(value, np.ndarray)]
    assert arrays
    assert len({id(array) for array in arrays}) == len(arrays)
    assert all(not array.flags.writeable for array in arrays)
    owners = []
    for array in arrays:
        owner = array
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        assert owner.nbytes == array.nbytes
        owners.append(owner)
    assert len({id(owner) for owner in owners}) == len(owners)
    assert sum(array.nbytes for array in arrays) == store.memory_bytes()


def test_total_construction_and_execution_never_touch_bidegree_plans_or_ranks(
    monkeypatch,
):
    from tensordev.core.bigraded import precompute as bigraded_precompute
    from tensordev.core.shear import bigraded as shear_bigraded

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense-total shear touched bidegree plan machinery")

    monkeypatch.setattr(
        bigraded_precompute.BigradedPlanStore, "__init__", forbidden
    )
    monkeypatch.setattr(bigraded_precompute, "colex_rank", forbidden)
    monkeypatch.setattr(bigraded_precompute, "colex_placements", forbidden)
    for name in (
        "colex_rank",
        "colex_placements",
        "_transform_rank_groups",
        "_gamma_rank_groups",
    ):
        monkeypatch.setattr(shear_bigraded, name, forbidden)
    monkeypatch.setattr(
        shear_bigraded.BigradedShearPlanStore, "__init__", forbidden
    )
    monkeypatch.setattr(
        shear_bigraded.BigradedShearShufflePlanStore, "__init__", forbidden
    )

    core = JaxShearTotal(
        dims=(1, 1), max_trunc=3, precompute_shuffle=True
    )
    standard = tuple(
        jnp.arange(2**degree, dtype=jnp.float32) / (degree + 1)
        for degree in range(4)
    )
    shear = core.tensor_from_standard_coordinates(standard, trunc=3)
    outputs = (
        core.tensor_to_standard_coordinates(shear, trunc=3),
        core.tensor_product(shear, shear, trunc=3),
        core.tensor_shuffle_product(shear, shear, trunc=3),
        core.tensor_fmexp(
            (jnp.ones((1,), dtype=jnp.float32),),
            (jnp.asarray([0.2, -0.1], dtype=jnp.float32),),
            trunc=3,
        ),
    )
    for output in outputs:
        for block in output:
            np.asarray(block).sum()
