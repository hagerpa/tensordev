"""Public construction contracts for partially symmetrized cores."""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
import tensordev.core as core_api
from tensordev.core.bigraded import (
    JaxBigraded,
    JaxPartiallySymmetrizedBigraded,
)
from tensordev.core.bigraded.symmetrized.gamma import (
    PartiallySymmetrizedShearShufflePlanStore,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
)
from tensordev.core.shear import (
    JaxPartiallySymmetrizedShearBigraded,
    JaxShearBigraded,
)


@pytest.fixture(autouse=True)
def _restore_default_core_pair():
    pair = td.get_default_core_pair()
    yield
    td.set_default_core(*pair)


def _shuffle_scope(core):
    store = core.shuffle_plan_store
    return "none" if store is None else store.scope


def _assert_tensors_close(actual, expected, *, atol=2e-6, rtol=2e-6):
    assert actual.spec == expected.spec
    assert actual.grades == expected.grades
    for grade in actual.grades:
        np.testing.assert_allclose(
            actual[grade],
            expected[grade],
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize(
    ("partially_symmetrized", "expected_type"),
    (
        (False, JaxBigraded),
        (True, JaxPartiallySymmetrizedBigraded),
    ),
)
def test_make_core_selects_partial_symmetrization(
    partially_symmetrized,
    expected_type,
):
    core = td.make_core(
        dims=(1, 1),
        max_trunc=(1, 1),
        default_trunc=(1, 0),
        partially_symmetrized=partially_symmetrized,
        precompute_shuffle="generator",
    )

    assert isinstance(core, expected_type)
    assert core.partially_symmetrized is partially_symmetrized
    assert core.coordinates == "standard"
    assert core.default_truncation == (1, 0)
    assert _shuffle_scope(core) == "generator"


@pytest.mark.parametrize("value", ("quotient", 1))
def test_make_core_rejects_nonboolean_partial_symmetrization(value):
    with pytest.raises(
        TypeError,
        match="partially_symmetrized must be a bool",
    ):
        td.make_core(
            dims=(1, 1),
            max_trunc=(1, 1),
            partially_symmetrized=value,
        )


@pytest.mark.parametrize(
    "core_type",
    (
        JaxPartiallySymmetrizedBigraded,
        JaxPartiallySymmetrizedShearBigraded,
    ),
)
def test_explicit_empty_shuffle_store_does_not_advertise_shuffle(core_type):
    plans = PartiallySymmetrizedPlanStore((1, 1), (1, 1))
    shuffle = PartiallySymmetrizedShearShufflePlanStore(plans, scope="none")
    core = core_type(plan_store=plans, shuffle_plan_store=shuffle)

    assert not hasattr(core, "gamma_plan_store")
    assert core.shuffle_plan_store is None
    assert "shuffle" not in core.capabilities
    assert core.plan_statistics()["shuffle_enabled"] is False
    assert core.plan_statistics()["shuffle_scope"] == "none"


@pytest.mark.parametrize(
    "core_type",
    (
        JaxShearBigraded,
        JaxPartiallySymmetrizedBigraded,
        JaxPartiallySymmetrizedShearBigraded,
    ),
)
def test_explicit_shuffle_store_keyword_has_one_public_name(core_type):
    parameters = inspect.signature(core_type).parameters

    assert "shuffle_plan_store" in parameters
    assert "gamma_plan_store" not in parameters


@pytest.mark.parametrize(
    ("coordinates", "expected_type"),
    (
        ("standard", JaxPartiallySymmetrizedBigraded),
        ("shear", JaxPartiallySymmetrizedShearBigraded),
    ),
)
def test_set_default_core_constructs_and_directly_rebinds_quotient_cores(
    coordinates,
    expected_type,
):
    previous_product = td.tensor_product
    core = td.set_default_core(
        dims=(1, 1),
        max_trunc=(1, 1),
        default_trunc=(1, 1),
        partially_symmetrized=True,
        coordinates=coordinates,
        precompute_shuffle="generator",
    )

    assert isinstance(core, expected_type)
    assert core is td.get_default_core()
    assert core.partially_symmetrized is True
    assert core.coordinates == coordinates
    assert _shuffle_scope(core) == "generator"
    assert previous_product is not td.tensor_product
    for name in (
        "tensor_product",
        "tensor_exponential",
        "tensor_shear_pairing",
        "tensor_partially_symmetrize",
        "tensor_to_ordered",
        "tensor_from_standard_coordinates",
        "tensor_to_standard_coordinates",
    ):
        assert getattr(td, name).__self__ is core
    assert hasattr(core.tensor_adjoint_product.__func__, "lower")
    assert td.tensor_shear_pairing.__self__ is core

    generator = jnp.asarray([0.2, -0.3], dtype=jnp.float32)
    result = td.tensor_exponential((generator,), trunc=(1, 1))
    assert result.spec.partially_symmetrized is True
    assert result.spec.coordinates == coordinates
    ordered = td.tensor_to_ordered(result)
    assert ordered.spec.partially_symmetrized is False
    assert ordered.spec.coordinates == coordinates


@pytest.mark.parametrize("coordinates", ("standard", "shear"))
def test_symmetrize_core_inherits_source_metadata_and_shuffle_scope(
    coordinates,
):
    ordered_standard = td.make_core(
        dims=(1, 1),
        max_trunc=(2, 1),
        default_trunc=(1, 1),
        precompute_shuffle="generator",
    )
    source = (
        ordered_standard
        if coordinates == "standard"
        else td.shear_core(ordered_standard)
    )

    core = td.symmetrize_core(source)

    expected_type = (
        JaxPartiallySymmetrizedBigraded
        if coordinates == "standard"
        else JaxPartiallySymmetrizedShearBigraded
    )
    assert isinstance(core, expected_type)
    assert core.dims == source.dims
    assert core.max_truncation == source.max_truncation
    assert core.default_truncation == source.default_truncation
    assert core.coordinates == source.coordinates
    assert core.partially_symmetrized is True
    assert _shuffle_scope(core) == "generator"


@pytest.mark.parametrize(
    ("override", "expected_scope"),
    ((False, "none"), ("generator", "generator"), (True, "full")),
)
def test_symmetrize_core_allows_an_explicit_shuffle_override(
    override,
    expected_scope,
):
    source = td.make_core(
        dims=(1, 1),
        max_trunc=(1, 1),
        precompute_shuffle="generator",
    )

    core = td.symmetrize_core(
        source,
        precompute_shuffle=override,
    )

    assert _shuffle_scope(core) == expected_scope


def test_symmetrize_core_rejects_incompatible_sources_and_bad_overrides():
    ordered = td.make_core(dims=(1, 1), max_trunc=(1, 1))
    quotient = td.symmetrize_core(ordered)
    quotient_shear = td.shear_core(quotient)

    for source in (quotient, quotient_shear):
        with pytest.raises(TypeError, match="already partially symmetrized"):
            td.symmetrize_core(source)

    total = td.make_core(dims=2, max_trunc=1)
    total_shear = td.shear_core(total, dims=(1, 1))
    for source in (total, total_shear):
        with pytest.raises(TypeError, match="bidegree source core"):
            td.symmetrize_core(source)

    with pytest.raises(TypeError, match="supports built-in"):
        td.symmetrize_core(object())
    with pytest.raises(TypeError, match="precompute_shuffle"):
        td.symmetrize_core(ordered, precompute_shuffle=1)


def test_shear_core_preserves_partial_symmetrization_and_reuses_stores():
    standard = td.make_core(
        dims=(1, 1),
        max_trunc=(2, 1),
        default_trunc=(1, 1),
        partially_symmetrized=True,
        precompute_shuffle="generator",
    )

    shear = td.shear_core(standard)

    assert isinstance(shear, JaxPartiallySymmetrizedShearBigraded)
    assert shear.partially_symmetrized is standard.partially_symmetrized
    assert shear.coordinates == "shear"
    assert shear.dims == standard.dims
    assert shear.max_truncation == standard.max_truncation
    assert shear.default_truncation == standard.default_truncation
    assert shear.plan_store is standard.plan_store
    assert shear.bridge_plan_store is standard.bridge_plan_store
    assert shear.shear_plan_store is standard.shear_plan_store
    assert shear.shuffle_plan_store is standard.shuffle_plan_store

    full = td.shear_core(standard, precompute_shuffle=True)
    assert full.plan_store is standard.plan_store
    assert full.bridge_plan_store is standard.bridge_plan_store
    assert full.shear_plan_store is standard.shear_plan_store
    assert full.shuffle_plan_store is not standard.shuffle_plan_store
    assert full.shuffle_plan_store.scope == "full"

    asserted = td.shear_core(
        standard,
        dims=standard.dims,
        max_trunc=standard.max_truncation,
    )
    assert asserted.plan_store is standard.plan_store
    with pytest.raises(ValueError, match="dims"):
        td.shear_core(standard, dims=(2, 1))
    with pytest.raises(ValueError, match="max_trunc"):
        td.shear_core(standard, max_trunc=(1, 1))
    with pytest.raises(TypeError, match="max_trunc"):
        td.shear_core(standard, max_trunc=2)


def test_standard_shear_and_symmetrization_construction_square_commutes():
    ordered_standard = td.make_core(
        dims=(1, 1),
        max_trunc=(2, 2),
        default_trunc=(2, 2),
    )
    quotient_standard = td.symmetrize_core(ordered_standard)
    quotient_then_shear = td.shear_core(quotient_standard)
    ordered_shear = td.shear_core(ordered_standard)
    shear_then_quotient = td.symmetrize_core(ordered_shear)

    generator = jnp.asarray([0.2, -0.3], dtype=jnp.float32)
    ordered_value = ordered_standard.tensor_exponential(
        (generator,), trunc=(2, 2)
    )

    quotient_standard_value = quotient_standard.tensor_partially_symmetrize(
        ordered_value
    )
    quotient_then_shear_value = (
        quotient_then_shear.tensor_from_standard_coordinates(
            quotient_standard_value
        )
    )
    ordered_shear_value = ordered_shear.tensor_from_standard_coordinates(
        ordered_value
    )
    shear_then_quotient_value = (
        shear_then_quotient.tensor_partially_symmetrize(
            ordered_shear_value
        )
    )

    _assert_tensors_close(
        quotient_then_shear_value,
        shear_then_quotient_value,
    )
    for core in (quotient_then_shear, shear_then_quotient):
        direct = core.tensor_exponential((generator,), trunc=(2, 2))
        _assert_tensors_close(direct, quotient_then_shear_value)


@pytest.mark.parametrize(
    "kwargs",
    (
        {
            "dims": 2,
            "max_trunc": 2,
            "coordinates": "standard",
        },
        {
            "dims": (1, 1),
            "max_trunc": 2,
            "coordinates": "shear",
        },
    ),
)
def test_total_partially_symmetrized_default_failure_is_atomic(kwargs):
    initial_pair = td.get_default_core_pair()

    with pytest.raises(ValueError, match="requires bidegree truncation"):
        td.set_default_core(
            **kwargs,
            partially_symmetrized=True,
        )

    assert td.get_default_core_pair() is initial_pair
    assert td.tensor_product.__self__ is initial_pair[0]
    assert td.tensor_partially_symmetrize.__self__ is initial_pair[0]


def test_partially_symmetrized_factories_classes_and_operations_are_exported():
    expected_root_exports = {
        "JaxPartiallySymmetrizedBigraded",
        "JaxPartiallySymmetrizedShearBigraded",
        "make_core",
        "shear_core",
        "symmetrize_core",
        "tensor_partially_symmetrize",
        "tensor_partially_symmetrize_homogeneous",
        "tensor_to_ordered",
    }
    assert expected_root_exports <= set(td.__all__)
    assert {
        "JaxPartiallySymmetrizedBigraded",
        "JaxPartiallySymmetrizedShearBigraded",
        "symmetrize_core",
    } <= set(core_api.__all__)
    assert (
        td.JaxPartiallySymmetrizedBigraded
        is core_api.JaxPartiallySymmetrizedBigraded
        is JaxPartiallySymmetrizedBigraded
    )
    assert (
        td.JaxPartiallySymmetrizedShearBigraded
        is core_api.JaxPartiallySymmetrizedShearBigraded
        is JaxPartiallySymmetrizedShearBigraded
    )
    assert td.make_core is core_api.make_core
    assert td.shear_core is core_api.shear_core
    assert td.symmetrize_core is core_api.symmetrize_core
    for removed in (
        "total_degree_core",
        "bigraded_core",
        "symmetrized_core",
    ):
        assert not hasattr(td, removed)
        assert not hasattr(core_api, removed)
