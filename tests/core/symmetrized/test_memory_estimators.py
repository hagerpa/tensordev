from __future__ import annotations

import pytest

import tensordev as td
from tensordev.core.bigraded.symmetrized.gamma import (
    PartiallySymmetrizedShearShufflePlanStore,
    _expected_gamma_memory_bytes_by_category,
)
from tensordev.core.bigraded.symmetrized.jax import (
    JaxPartiallySymmetrizedBigraded,
)
from tensordev.core.bigraded.symmetrized.memory import (
    _expected_partially_symmetrized_memory_bytes_by_category,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
)
from tensordev.core.bigraded.symmetrized.transforms import (
    PartiallySymmetrizedShearPlanStore,
    _expected_transform_memory_bytes_by_category,
)
from tensordev.core.shear.symmetrized import (
    JaxPartiallySymmetrizedShearBigraded,
    PartiallySymmetrizedShearGeneratorPlanStore,
    _expected_partially_symmetrized_shear_generator_memory_bytes_by_category,
)


@pytest.mark.parametrize(
    ("dims", "capacity"),
    (((1, 1), (2, 2)), ((1, 2), (2, 2)), ((2, 2), (1, 2))),
)
def test_transform_estimator_matches_retained_payload(dims, capacity):
    plans = PartiallySymmetrizedPlanStore(dims, capacity)
    transforms = PartiallySymmetrizedShearPlanStore(plans)

    assert dict(
        _expected_transform_memory_bytes_by_category(dims, capacity)
    ) == dict(transforms.memory_bytes_by_category())


@pytest.mark.parametrize("scope", ("generator", "full"))
@pytest.mark.parametrize(
    ("dims", "capacity"),
    (((1, 1), (2, 2)), ((1, 2), (2, 2)), ((2, 2), (1, 2))),
)
def test_gamma_estimator_matches_retained_payload(dims, capacity, scope):
    plans = PartiallySymmetrizedPlanStore(dims, capacity)
    gamma = PartiallySymmetrizedShearShufflePlanStore(plans, scope=scope)

    assert dict(
        _expected_gamma_memory_bytes_by_category(
            dims,
            capacity,
            scope=scope,
        )
    ) == dict(gamma.memory_bytes_by_category())


@pytest.mark.parametrize(
    ("dims", "capacity"),
    (
        ((1, 1), (2, 2)),
        ((1, 2), (2, 2)),
        ((2, 2), (1, 2)),
        ((1, 1), (4, 11)),
    ),
)
def test_shear_generator_estimator_matches_retained_payload(dims, capacity):
    plans = PartiallySymmetrizedPlanStore(dims, capacity)
    generators = PartiallySymmetrizedShearGeneratorPlanStore(plans)

    assert dict(
        _expected_partially_symmetrized_shear_generator_memory_bytes_by_category(
            dims,
            capacity,
        )
    ) == dict(generators.memory_bytes_by_category())


@pytest.mark.parametrize("precompute_shuffle", (False, "generator", True))
@pytest.mark.parametrize("coordinates", ("standard", "shear"))
def test_full_quotient_estimator_matches_constructed_core(
    coordinates,
    precompute_shuffle,
):
    core_type = (
        JaxPartiallySymmetrizedBigraded
        if coordinates == "standard"
        else JaxPartiallySymmetrizedShearBigraded
    )
    core = core_type(
        dims=(1, 2),
        max_trunc=(2, 2),
        precompute_shuffle=precompute_shuffle,
    )

    estimated = _expected_partially_symmetrized_memory_bytes_by_category(
        (1, 2),
        (2, 2),
        coordinates=coordinates,
        precompute_shuffle=precompute_shuffle,
    )
    assert dict(estimated) == dict(core.memory_bytes_by_category())
    assert sum(estimated.values()) == core.memory_bytes()

    public_breakdown = td.core_expected_memory(
        dims=(1, 2),
        max_trunc=(2, 2),
        partially_symmetrized=True,
        coordinates=coordinates,
        precompute_shuffle=precompute_shuffle,
        unit="bytes",
        breakdown=True,
    )
    assert public_breakdown["total"] == core.memory_bytes()
    assert {
        name: value
        for name, value in public_breakdown.items()
        if name != "total"
    } == core.memory_bytes_by_category()


def test_full_quotient_estimator_does_not_construct_cores_or_stores(
    monkeypatch,
):
    from tensordev.core.bigraded.symmetrized import bridge, gamma, plans, transforms
    from tensordev.core.bigraded.symmetrized import jax as quotient_jax
    from tensordev.core.shear import symmetrized as shear_quotient

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("memory estimation must not construct plans")

    monkeypatch.setattr(
        plans.PartiallySymmetrizedPlanStore,
        "__init__",
        forbidden,
    )
    monkeypatch.setattr(
        bridge.SymmetrizationBridgePlanStore,
        "__init__",
        forbidden,
    )
    monkeypatch.setattr(
        transforms.PartiallySymmetrizedShearPlanStore,
        "__init__",
        forbidden,
    )
    monkeypatch.setattr(
        gamma.PartiallySymmetrizedShearShufflePlanStore,
        "__init__",
        forbidden,
    )
    monkeypatch.setattr(
        shear_quotient.PartiallySymmetrizedShearGeneratorPlanStore,
        "__init__",
        forbidden,
    )
    monkeypatch.setattr(
        quotient_jax.JaxPartiallySymmetrizedBigraded,
        "__init__",
        forbidden,
    )
    monkeypatch.setattr(
        shear_quotient.JaxPartiallySymmetrizedShearBigraded,
        "__init__",
        forbidden,
    )

    for coordinates in ("standard", "shear"):
        for precompute_shuffle in (False, "generator", True):
            memory = _expected_partially_symmetrized_memory_bytes_by_category(
                (1, 2),
                (2, 2),
                coordinates=coordinates,
                precompute_shuffle=precompute_shuffle,
            )
            assert sum(memory.values()) > 0
            assert td.core_expected_memory(
                dims=(1, 2),
                max_trunc=(2, 2),
                partially_symmetrized=True,
                coordinates=coordinates,
                precompute_shuffle=precompute_shuffle,
                unit="bytes",
            ) == sum(memory.values())
