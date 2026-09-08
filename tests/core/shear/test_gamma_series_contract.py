"""Public full-series Gamma contracts for both shear gradings."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from tensordev.core.bigraded.jax import JaxBigraded
from tensordev.core.bigraded.types import BigradedTensor
from tensordev.core.jax import Jax
from tensordev.core.shear.bigraded import JaxShearBigraded
from tensordev.core.shear.jax import JaxShearTotal
from tensordev.development import path_signature


ATOL = 4e-4
RTOL = 4e-4


def _assert_series_close(left, right):
    if isinstance(left, BigradedTensor):
        assert isinstance(right, BigradedTensor)
        assert left.spec == right.spec
        pairs = zip(left.blocks, right.blocks)
    else:
        assert len(left) == len(right)
        pairs = zip(left, right)
    for left_block, right_block in pairs:
        np.testing.assert_allclose(
            np.asarray(left_block),
            np.asarray(right_block),
            atol=ATOL,
            rtol=RTOL,
        )


def _transpose_transport_oracle(core, standard_core, left, right, trunc):
    standard_left = core._coordinate_forward_transpose(
        left, trunc=trunc, first_on=False
    )
    standard_right = core._coordinate_forward_transpose(
        right, trunc=trunc, first_on=False
    )
    standard_result = standard_core.tensor_shuffle_product(
        standard_left, standard_right, trunc=trunc
    )
    return core._coordinate_inverse_transpose(
        standard_result, trunc=trunc, first_on=False
    )


def _exercise_full_gamma_contract(
    core,
    standard_core,
    left,
    right,
    third,
    unit,
    signature,
    character_left,
    character_right,
    trunc,
):
    product = core.tensor_shuffle_product(left, right, trunc=trunc)
    oracle = _transpose_transport_oracle(
        core, standard_core, left, right, trunc
    )
    _assert_series_close(product, oracle)

    _assert_series_close(
        core.tensor_shuffle_product(unit, left, trunc=trunc), left
    )
    _assert_series_close(
        core.tensor_shuffle_product(left, unit, trunc=trunc), left
    )
    _assert_series_close(
        product,
        core.tensor_shuffle_product(right, left, trunc=trunc),
    )

    left_associated = core.tensor_shuffle_product(
        product, third, trunc=trunc
    )
    right_associated = core.tensor_shuffle_product(
        left,
        core.tensor_shuffle_product(right, third, trunc=trunc),
        trunc=trunc,
    )
    _assert_series_close(left_associated, right_associated)

    character_product = core.tensor_shuffle_product(
        character_left, character_right, trunc=trunc
    )
    lhs = (
        core.tensor_inner_product(character_left, signature)
        * core.tensor_inner_product(character_right, signature)
    )
    rhs = core.tensor_inner_product(character_product, signature)
    np.testing.assert_allclose(
        np.asarray(lhs), np.asarray(rhs), atol=ATOL, rtol=RTOL
    )


def _random_total(key, *, trunc, scale=0.06):
    keys = jr.split(key, trunc + 1)
    return tuple(
        scale * jr.normal(block_key, (2**degree,), dtype=jnp.float32)
        for degree, block_key in enumerate(keys)
    )


def _total_supported(*, trunc, blocks):
    return tuple(
        jnp.asarray(
            blocks.get(degree, np.zeros(2**degree)), dtype=jnp.float32
        )
        for degree in range(trunc + 1)
    )


def test_total_public_gamma_full_series_contract():
    trunc = 4
    core = JaxShearTotal(
        dims=(1, 1), max_trunc=trunc, precompute_shuffle=True
    )
    standard_core = Jax(
        d=2, max_trunc=trunc, precompute_shuffle=True
    )
    left, right, third = (
        _random_total(key, trunc=trunc)
        for key in jr.split(jr.PRNGKey(2101), 3)
    )
    unit = _total_supported(
        trunc=trunc, blocks={0: jnp.ones((1,), dtype=jnp.float32)}
    )
    character_left = _total_supported(
        trunc=trunc,
        blocks={1: jnp.asarray([0.4, -0.2], dtype=jnp.float32)},
    )
    character_right = _total_supported(
        trunc=trunc,
        blocks={1: jnp.asarray([-0.3, 0.5], dtype=jnp.float32)},
    )
    path = jnp.asarray(
        [[0.0, 0.0], [0.2, -0.1], [-0.1, 0.25], [0.15, 0.3]],
        dtype=jnp.float32,
    )
    signature = path_signature(path, trunc=trunc, core=core)
    _exercise_full_gamma_contract(
        core,
        standard_core,
        left,
        right,
        third,
        unit,
        signature,
        character_left,
        character_right,
        trunc,
    )


def _random_bigraded(core, key, *, trunc, scale=0.06):
    layout = core.resolve_layout(
        trunc, include_scalar=True
    )
    keys = jr.split(key, len(layout.grades))
    return BigradedTensor(
        tuple(
            scale
            * jr.normal(
                block_key,
                (layout.block_width(grade),),
                dtype=jnp.float32,
            )
            for grade, block_key in zip(layout.grades, keys)
        ),
        layout.spec,
    )


def test_bidegree_public_gamma_full_series_contract():
    trunc = (2, 2)
    core = JaxShearBigraded(
        dims=(1, 1), max_trunc=trunc, precompute_shuffle=True
    )
    standard_core = JaxBigraded(
        dims=(1, 1), max_trunc=trunc, precompute_shuffle=True
    )
    left, right, third = (
        _random_bigraded(core, key, trunc=trunc)
        for key in jr.split(jr.PRNGKey(2102), 3)
    )
    unit = core.tensor_densify(
        {(0, 0): jnp.ones((1,), dtype=jnp.float32)},
        trunc=trunc,
        include_scalar=True,
    )
    character_left = core.tensor_densify(
        {(1, 0): jnp.asarray([0.4], dtype=jnp.float32)},
        trunc=trunc,
        include_scalar=True,
    )
    character_right = core.tensor_densify(
        {(0, 1): jnp.asarray([-0.3], dtype=jnp.float32)},
        trunc=trunc,
        include_scalar=True,
    )
    path = jnp.asarray(
        [[0.0, 0.0], [0.2, -0.1], [-0.1, 0.25], [0.15, 0.3]],
        dtype=jnp.float32,
    )
    signature = path_signature(path, trunc=trunc, core=core)
    _exercise_full_gamma_contract(
        core,
        standard_core,
        left,
        right,
        third,
        unit,
        signature,
        character_left,
        character_right,
        trunc,
    )


@pytest.mark.parametrize("scope", (False, "generator"))
def test_total_public_gamma_scope_failures_do_not_grow_plans(scope):
    core = JaxShearTotal(
        dims=(1, 1), max_trunc=3, precompute_shuffle=scope
    )
    value = _random_total(jr.PRNGKey(2201), trunc=2)
    before = (
        tuple(core.shear_plan_store.shuffle_plans),
        core.shear_plan_store.memory_bytes(),
    )
    with pytest.raises(RuntimeError, match="precompute_shuffle"):
        core.tensor_shuffle_product(value, value, trunc=3)
    after = (
        tuple(core.shear_plan_store.shuffle_plans),
        core.shear_plan_store.memory_bytes(),
    )
    assert after == before


@pytest.mark.parametrize("scope", (False, "generator"))
def test_bidegree_public_gamma_scope_failures_do_not_grow_plans(scope):
    trunc = (2, 2)
    core = JaxShearBigraded(
        dims=(1, 1), max_trunc=trunc, precompute_shuffle=scope
    )
    value = _random_bigraded(core, jr.PRNGKey(2202), trunc=(1, 1))
    store = core.shuffle_plan_store
    before = (
        None
        if store is None
        else (tuple(store.block_plans), store.memory_bytes())
    )
    with pytest.raises(RuntimeError, match="precompute_shuffle"):
        core.tensor_shuffle_product(value, value, trunc=trunc)
    store = core.shuffle_plan_store
    after = (
        None
        if store is None
        else (tuple(store.block_plans), store.memory_bytes())
    )
    assert after == before
