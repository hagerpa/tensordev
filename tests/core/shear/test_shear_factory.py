"""Construction of compatible shear cores from standard JAX cores."""

from __future__ import annotations

import pytest

import tensordev as td
from tensordev.core.bigraded.jax import JaxBigraded
from tensordev.core.jax import Jax
from tensordev.core.shear.bigraded import JaxShearBigraded
from tensordev.core.shear.jax import JaxShearTotal


@pytest.mark.parametrize(
    ("precompute_shuffle", "expected_scope"),
    ((False, "none"), (True, "full")),
)
def test_total_factory_inherits_capacity_default_and_shuffle_scope(
    precompute_shuffle, expected_scope
):
    source = Jax(
        d=2,
        max_trunc=2,
        default_trunc=1,
        precompute_shuffle=precompute_shuffle,
    )

    core = td.shear_core(source, dims=(1, 1))

    assert isinstance(core, JaxShearTotal)
    assert core.dims == (1, 1)
    assert core.max_truncation == 2
    assert core.default_truncation == 1
    assert core.shear_plan_store.shuffle_scope == expected_scope


@pytest.mark.parametrize(
    ("precompute_shuffle", "expected_scope"),
    ((False, "none"), ("generator", "generator"), (True, "full")),
)
def test_bigraded_factory_shares_base_plans_and_inherits_scope(
    precompute_shuffle, expected_scope
):
    source = JaxBigraded(
        dims=(1, 1),
        max_trunc=(1, 1),
        default_trunc=(1, 0),
        precompute_shuffle=precompute_shuffle,
    )

    core = td.shear_core(source)

    assert isinstance(core, JaxShearBigraded)
    assert core.dims == source.dims
    assert core.max_truncation == source.max_truncation
    assert core.default_truncation == source.default_truncation
    assert core.plan_store is source.plan_store
    assert core.shear_plan_store.plan_store is source.plan_store
    if expected_scope == "none":
        assert core.shuffle_plan_store is None
    else:
        assert core.shuffle_plan_store is not source.shuffle_plan_store
        assert core.shuffle_plan_store.plan_store is source.plan_store
        assert core.shuffle_plan_store.scope == expected_scope


def test_factory_shuffle_argument_explicitly_overrides_source_scope():
    total_source = Jax(d=2, max_trunc=2, precompute_shuffle=True)
    total = td.shear_core(
        total_source,
        dims=(1, 1),
        precompute_shuffle="generator",
    )
    assert total.shear_plan_store.shuffle_scope == "generator"

    bigraded_source = JaxBigraded(
        dims=(1, 1), max_trunc=(1, 1), precompute_shuffle=True
    )
    bigraded = td.shear_core(
        bigraded_source, precompute_shuffle=False
    )
    assert bigraded.shuffle_plan_store is None

    source_without_shuffle = JaxBigraded(
        dims=(1, 1), max_trunc=(1, 1), precompute_shuffle=False
    )
    full = td.shear_core(
        source_without_shuffle, precompute_shuffle=True
    )
    assert full.shuffle_plan_store.scope == "full"


def test_unbounded_default_jax_requires_dims_and_capacity_then_works():
    source = Jax()

    with pytest.raises(TypeError, match="dims"):
        td.shear_core(source)
    with pytest.raises(TypeError, match="max_trunc"):
        td.shear_core(source, dims=(1, 1))

    core = td.shear_core(source, dims=(1, 1), max_trunc=2)
    assert isinstance(core, JaxShearTotal)
    assert core.dims == (1, 1)
    assert core.max_truncation == 2
    assert core.default_truncation == 2


def test_total_factory_rejects_split_and_capacity_disagreement():
    source = Jax(d=3, max_trunc=2)

    with pytest.raises(TypeError, match="dims"):
        td.shear_core(source)
    with pytest.raises(ValueError, match="dims|split|dimension"):
        td.shear_core(source, dims=(1, 1))
    with pytest.raises(ValueError, match="max_trunc|capacity"):
        td.shear_core(source, dims=(1, 2), max_trunc=1)


def test_bigraded_factory_treats_dims_and_capacity_as_assertions():
    source = JaxBigraded(dims=(1, 2), max_trunc=(2, 1))

    same = td.shear_core(
        source, dims=(1, 2), max_trunc=(2, 1)
    )
    assert same.plan_store is source.plan_store

    with pytest.raises(ValueError, match="dims"):
        td.shear_core(source, dims=(2, 1))
    with pytest.raises(ValueError, match="max_trunc|capacity"):
        td.shear_core(source, max_trunc=(1, 1))
    with pytest.raises(TypeError, match="max_trunc"):
        td.shear_core(source, max_trunc=2)


@pytest.mark.parametrize(
    "source",
    (
        object(),
        JaxShearTotal(dims=(1, 1), max_trunc=1),
        JaxShearBigraded(dims=(1, 1), max_trunc=(1, 1)),
    ),
)
def test_factory_rejects_nonstandard_or_nonjax_sources(source):
    with pytest.raises(TypeError, match=type(source).__name__):
        td.shear_core(source)


@pytest.mark.parametrize("invalid", (0, 1, "full"))
def test_factory_strictly_validates_explicit_shuffle_override(invalid):
    source = Jax(d=2, max_trunc=1)
    with pytest.raises(TypeError, match="precompute_shuffle"):
        td.shear_core(
            source,
            dims=(1, 1),
            precompute_shuffle=invalid,
        )


def test_shear_core_factory_is_exported_from_package_root():
    from tensordev.core.shear import shear_core

    assert td.shear_core is shear_core
