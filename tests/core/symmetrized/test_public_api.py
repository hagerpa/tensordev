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
    ("representation", "expected_type"),
    (
        ("ordered", JaxBigraded),
        ("partially_symmetrized", JaxPartiallySymmetrizedBigraded),
    ),
)
def test_bigraded_core_selects_the_public_representation(
    representation,
    expected_type,
):
    core = td.bigraded_core(
        dims=(1, 1),
        max_trunc=(1, 1),
        default_trunc=(1, 0),
        representation=representation,
        precompute_shuffle="generator",
    )

    assert isinstance(core, expected_type)
    assert core.representation == representation
    assert core.coordinates == "standard"
    assert core.default_truncation == (1, 0)
    assert _shuffle_scope(core) == "generator"


def test_bigraded_core_rejects_an_unknown_representation():
    with pytest.raises(ValueError, match="partially_symmetrized"):
        td.bigraded_core(
            dims=(1, 1),
            max_trunc=(1, 1),
            representation="quotient",
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
        representation="partially_symmetrized",
        coordinates=coordinates,
        precompute_shuffle="generator",
    )

    assert isinstance(core, expected_type)
    assert core is td.get_default_core()
    assert core.representation == "partially_symmetrized"
    assert core.coordinates == coordinates
    assert _shuffle_scope(core) == "generator"
    assert previous_product is not td.tensor_product
    for name in (
        "tensor_product",
        "tensor_exponential",
        "tensor_signature_inner_product",
        "tensor_partially_symmetrize",
        "tensor_to_ordered",
        "tensor_from_standard_coordinates",
        "tensor_to_standard_coordinates",
    ):
        assert getattr(td, name).__self__ is core
    assert hasattr(core.tensor_adjoint_product.__func__, "lower")
    assert (
        td.tensor_signature_inner_product.__func__
        is td.tensor_shear_inner_product.__func__
    )

    generator = jnp.asarray([0.2, -0.3], dtype=jnp.float32)
    result = td.tensor_exponential((generator,), trunc=(1, 1))
    assert result.spec.representation == "partially_symmetrized"
    assert result.spec.coordinates == coordinates
    ordered = td.tensor_to_ordered(result)
    assert ordered.spec.representation == "ordered"
    assert ordered.spec.coordinates == coordinates


@pytest.mark.parametrize("coordinates", ("standard", "shear"))
def test_symmetrized_core_inherits_source_metadata_and_shuffle_scope(
    coordinates,
):
    ordered_standard = td.bigraded_core(
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

    core = td.symmetrized_core(source)

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
    assert core.representation == "partially_symmetrized"
    assert _shuffle_scope(core) == "generator"


@pytest.mark.parametrize(
    ("override", "expected_scope"),
    ((False, "none"), ("generator", "generator"), (True, "full")),
)
def test_symmetrized_core_allows_an_explicit_shuffle_override(
    override,
    expected_scope,
):
    source = td.bigraded_core(
        dims=(1, 1),
        max_trunc=(1, 1),
        precompute_shuffle="generator",
    )

    core = td.symmetrized_core(
        source,
        precompute_shuffle=override,
    )

    assert _shuffle_scope(core) == expected_scope


def test_symmetrized_core_rejects_incompatible_sources_and_bad_overrides():
    ordered = td.bigraded_core(dims=(1, 1), max_trunc=(1, 1))
    quotient = td.symmetrized_core(ordered)
    quotient_shear = td.shear_core(quotient)

    for source in (quotient, quotient_shear):
        with pytest.raises(TypeError, match="already partially symmetrized"):
            td.symmetrized_core(source)

    total = td.total_degree_core(d=2, max_trunc=1)
    total_shear = td.shear_core(total, dims=(1, 1))
    for source in (total, total_shear):
        with pytest.raises(TypeError, match="bidegree source core"):
            td.symmetrized_core(source)

    with pytest.raises(TypeError, match="supports built-in"):
        td.symmetrized_core(object())
    with pytest.raises(TypeError, match="precompute_shuffle"):
        td.symmetrized_core(ordered, precompute_shuffle=1)


def test_shear_core_preserves_quotient_representation_and_reuses_stores():
    standard = td.bigraded_core(
        dims=(1, 1),
        max_trunc=(2, 1),
        default_trunc=(1, 1),
        representation="partially_symmetrized",
        precompute_shuffle="generator",
    )

    shear = td.shear_core(standard)

    assert isinstance(shear, JaxPartiallySymmetrizedShearBigraded)
    assert shear.representation == standard.representation
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
    ordered_standard = td.bigraded_core(
        dims=(1, 1),
        max_trunc=(2, 2),
        default_trunc=(2, 2),
    )
    quotient_standard = td.symmetrized_core(ordered_standard)
    quotient_then_shear = td.shear_core(quotient_standard)
    ordered_shear = td.shear_core(ordered_standard)
    shear_then_quotient = td.symmetrized_core(ordered_shear)

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
            representation="partially_symmetrized",
        )

    assert td.get_default_core_pair() is initial_pair
    assert td.tensor_product.__self__ is initial_pair[0]
    assert td.tensor_partially_symmetrize.__self__ is initial_pair[0]


def test_partially_symmetrized_factories_classes_and_operations_are_exported():
    expected_root_exports = {
        "JaxPartiallySymmetrizedBigraded",
        "JaxPartiallySymmetrizedShearBigraded",
        "bigraded_core",
        "shear_core",
        "symmetrized_core",
        "tensor_partially_symmetrize",
        "tensor_partially_symmetrize_homogeneous",
        "tensor_to_ordered",
    }
    assert expected_root_exports <= set(td.__all__)
    assert {
        "JaxPartiallySymmetrizedBigraded",
        "JaxPartiallySymmetrizedShearBigraded",
        "symmetrized_core",
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
    assert td.bigraded_core is core_api.bigraded_core
    assert td.shear_core is core_api.shear_core
    assert td.symmetrized_core is core_api.symmetrized_core
