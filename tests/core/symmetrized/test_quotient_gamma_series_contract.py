"""Public full-series shuffle contracts for both quotient coordinates."""

from __future__ import annotations

import re

import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from tensordev.core.bigraded.jax import JaxBigraded
from tensordev.core.bigraded.symmetrized.jax import (
    JaxPartiallySymmetrizedBigraded,
)
from tensordev.core.bigraded.types import BigradedTensor
from tensordev.core.shear.symmetrized import (
    JaxPartiallySymmetrizedShearBigraded,
)
from tensordev.development import path_signature


config.update("jax_enable_x64", True)


ATOL = 3e-11
RTOL = 3e-11


def _random_tensor(core, key, *, trunc, scale=0.04):
    layout = core.resolve_layout(trunc, include_scalar=True)
    keys = jr.split(key, len(layout.grades))
    return BigradedTensor(
        tuple(
            scale
            * jr.normal(
                block_key,
                (layout.block_width(grade),),
                dtype=jnp.float64,
            )
            for grade, block_key in zip(layout.grades, keys)
        ),
        layout.spec,
    )


def _assert_series_close(actual, expected):
    assert actual.spec == expected.spec
    for grade in actual.grades:
        np.testing.assert_allclose(
            actual[grade],
            expected[grade],
            atol=ATOL,
            rtol=RTOL,
        )


def _standard_quotient_lifted_oracle(core, ordered, left, right, trunc):
    ordered_left = core._lift_partially_symmetrized(left)
    ordered_right = core._lift_partially_symmetrized(right)
    return ordered.tensor_shuffle_product(
        ordered_left,
        ordered_right,
        trunc=trunc,
    )


def _shear_quotient_oracle(core, standard, left, right, trunc):
    standard_left = core._coordinate_forward_transpose(
        left,
        trunc=trunc,
        first_on=False,
    )
    standard_right = core._coordinate_forward_transpose(
        right,
        trunc=trunc,
        first_on=False,
    )
    standard_result = standard.tensor_shuffle_product(
        standard_left,
        standard_right,
        trunc=trunc,
    )
    return core._coordinate_inverse_transpose(
        standard_result,
        trunc=trunc,
        first_on=False,
    )


@pytest.mark.parametrize("coordinates", ("standard", "shear"))
def test_public_full_series_quotient_shuffle_laws(coordinates):
    trunc = (2, 2)
    ordered = JaxBigraded(
        dims=(1, 2),
        max_trunc=trunc,
        precompute_shuffle=True,
    )
    standard = JaxPartiallySymmetrizedBigraded(
        dims=(1, 2),
        max_trunc=trunc,
        precompute_shuffle=True,
    )
    core = (
        standard
        if coordinates == "standard"
        else JaxPartiallySymmetrizedShearBigraded(
            dims=(1, 2),
            max_trunc=trunc,
            precompute_shuffle=True,
        )
    )
    left, right, third = (
        _random_tensor(core, key, trunc=trunc)
        for key in jr.split(jr.PRNGKey(2301), 3)
    )
    unit = core.tensor_densify(
        {(0, 0): jnp.ones((1,), dtype=jnp.float64)},
        trunc=trunc,
        include_scalar=True,
    )

    product = core.tensor_shuffle_product(left, right, trunc=trunc)
    if coordinates == "standard":
        oracle = _standard_quotient_lifted_oracle(
            core, ordered, left, right, trunc
        )
        lifted_product = core._lift_partially_symmetrized(product)
        for grade in lifted_product.grades:
            np.testing.assert_allclose(
                lifted_product[grade],
                oracle[grade],
                atol=ATOL,
                rtol=RTOL,
            )
    else:
        oracle = _shear_quotient_oracle(
            core, standard, left, right, trunc
        )
        _assert_series_close(product, oracle)
    _assert_series_close(
        core.tensor_shuffle_product(unit, left, trunc=trunc),
        left,
    )
    _assert_series_close(
        core.tensor_shuffle_product(left, unit, trunc=trunc),
        left,
    )
    _assert_series_close(
        product,
        core.tensor_shuffle_product(right, left, trunc=trunc),
    )
    _assert_series_close(
        core.tensor_shuffle_product(product, third, trunc=trunc),
        core.tensor_shuffle_product(
            left,
            core.tensor_shuffle_product(right, third, trunc=trunc),
            trunc=trunc,
        ),
    )

    character_left = core.tensor_densify(
        {(1, 0): jnp.asarray([0.4], dtype=jnp.float64)},
        trunc=trunc,
        include_scalar=True,
    )
    character_right = core.tensor_densify(
        {(0, 1): jnp.asarray([-0.3, 0.1], dtype=jnp.float64)},
        trunc=trunc,
        include_scalar=True,
    )
    path = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.2, -0.1, 0.05],
            [-0.1, 0.25, 0.15],
            [0.15, 0.3, -0.2],
        ],
        dtype=jnp.float64,
    )
    ordered_signature = path_signature(path, trunc=trunc, core=ordered)
    character_product = core.tensor_shuffle_product(
        character_left,
        character_right,
        trunc=trunc,
    )
    lhs = (
        core.tensor_shear_pairing(
            character_left, ordered_signature
        )
        * core.tensor_shear_pairing(
            character_right, ordered_signature
        )
    )
    rhs = core.tensor_shear_pairing(
        character_product,
        ordered_signature,
    )
    np.testing.assert_allclose(rhs, lhs, atol=ATOL, rtol=RTOL)


def test_compact_shear_pairing_obeys_notebook_square_identity():
    trunc = (2, 2)
    standard = JaxPartiallySymmetrizedBigraded(
        dims=(1, 2),
        max_trunc=trunc,
    )
    shear = JaxPartiallySymmetrizedShearBigraded(
        plan_store=standard.plan_store,
        bridge_plan_store=standard.bridge_plan_store,
        precompute_shuffle=True,
    )
    path = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.2, -0.1, 0.05],
            [-0.1, 0.25, 0.15],
            [0.15, 0.3, -0.2],
        ],
        dtype=jnp.float64,
    )
    signature = path_signature(path, trunc=trunc, core=standard)
    ell = shear.tensor_from_standard_coordinates(signature[:2, :2])
    square = shear.tensor_shuffle_product(ell, ell, trunc=trunc)

    factor = shear.tensor_shear_pairing(ell, signature)
    product = shear.tensor_shear_pairing(square, signature)

    np.testing.assert_allclose(product, factor**2, atol=ATOL, rtol=RTOL)


def _plan_snapshot(core):
    store = core.shuffle_plan_store
    return (
        store,
        None if store is None else tuple(store.block_plans),
        tuple(sorted(core.memory_bytes_by_category().items())),
    )


@pytest.mark.parametrize(
    "core_type",
    (
        JaxPartiallySymmetrizedBigraded,
        JaxPartiallySymmetrizedShearBigraded,
    ),
)
@pytest.mark.parametrize("scope", (False, "generator"))
def test_public_full_series_scope_failures_do_not_grow_plans(
    core_type,
    scope,
):
    trunc = (2, 2)
    core = core_type(
        dims=(1, 1),
        max_trunc=trunc,
        precompute_shuffle=scope,
    )
    value = _random_tensor(core, jr.PRNGKey(2302), trunc=(1, 1))
    before = _plan_snapshot(core)

    assert not core.supports("shuffle_product")
    with pytest.raises(RuntimeError, match="precompute_shuffle"):
        core.tensor_shuffle_product(value, value, trunc=trunc)

    assert _plan_snapshot(core) == before


def _active_input(core, trunc):
    layout = core.resolve_layout(trunc, include_scalar=True)
    return BigradedTensor(
        tuple(
            jnp.arange(layout.block_width(grade), dtype=jnp.float32) + 0.1
            for grade in layout.grades
        ),
        layout.spec,
    )


def _active_shuffle_stablehlo(core, value, trunc):
    def apply(left, right):
        return core.tensor_shuffle_product(left, right, trunc=trunc)

    stablehlo = str(
        jax.jit(apply)
        .lower(value, value)
        .compiler_ir(dialect="stablehlo")
    )
    return re.sub(r"module @[^ ]+", "module @MODULE", stablehlo, count=1)


@pytest.mark.parametrize(
    "core_type",
    (
        JaxPartiallySymmetrizedBigraded,
        JaxPartiallySymmetrizedShearBigraded,
    ),
)
def test_active_view_shuffle_stablehlo_is_capacity_isolated(core_type):
    active = (1, 1)
    exact = core_type(
        dims=(1, 1),
        max_trunc=active,
        precompute_shuffle=True,
    )
    capacity = core_type(
        dims=(1, 1),
        max_trunc=(3, 3),
        precompute_shuffle=True,
    )
    view = capacity.at_truncation(active)

    assert view.plan_store is capacity.plan_store
    assert view.shuffle_plan_store is capacity.shuffle_plan_store
    exact_value = _active_input(exact, active)
    view_value = _active_input(view, active)
    assert _active_shuffle_stablehlo(
        view, view_value, active
    ) == _active_shuffle_stablehlo(exact, exact_value, active)
