from __future__ import annotations

from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import tensordev as td
import tensordev._wordwise.pallas_fssk_quotient as pallas_fssk_quotient
from tensordev._wordwise.dispatch import _supported_cuda_device
from tensordev._wordwise.layout import build_layout_plan
from tensordev._wordwise.pallas_fssk_q1 import PallasFSSKPlanError
from tensordev._wordwise.pallas_fssk_quotient import (
    clear_pallas_fssk_quotient_cache,
    quotient_fssk_q1_pallas,
    quotient_fssk_q1_readout_pallas,
)
from tensordev._wordwise.pallas_ordinary import PallasOrdinaryResourceError
from tensordev.core.bigraded.types import BigradedTensor
from tensordev.sss import DenseLambda, FSSK
from tensordev.sss.state_update import fssk_state_from_coef


def _kernel(dtype, *, latent_dim=2, state_dim=2):
    return FSSK(
        Lambda=DenseLambda(
            jnp.diag(
                jnp.linspace(0.3, 0.75, state_dim, dtype=dtype)
            )
        ),
        A=jnp.arange(
            1,
            1 + latent_dim * 3,
            dtype=dtype,
        ).reshape(1, latent_dim, 3) / 8,
        b=jnp.linspace(0.65, -0.15, state_dim, dtype=dtype)[None],
    )


def _case(
    dtype,
    *,
    batch=2,
    steps=4,
    truncation=(2, 1),
    state_dim=2,
):
    core = td.make_core(
        dims=(1, 1),
        max_trunc=truncation,
        partially_symmetrized=True,
    )
    plan = build_layout_plan(core, truncation)
    kernel = _kernel(dtype, state_dim=state_dim)
    y = 0.05 * jr.normal(
        jr.PRNGKey(745_000 + steps),
        (batch, steps, 2),
        dtype=dtype,
    )
    dt = jnp.linspace(0.06, 0.12, steps, dtype=dtype)
    coef = kernel.coef(dt, trunc=sum(truncation), dtype=dtype)
    canonical = (
        coef.E[None],
        coef.psi[None],
        coef.phi[:, 0][None],
    )
    return core, plan, y, coef, canonical


def _random_initial_state(core, plan, *, batch, state_dim, dtype):
    positive = plan.blocks[1:]
    keys = jr.split(jr.PRNGKey(745_101), len(positive))
    return BigradedTensor(
        tuple(
            0.025
            * jr.normal(
                key,
                (batch, 1, 1, state_dim, block.width),
                dtype=dtype,
            )
            for key, block in zip(keys, positive)
        ),
        core.resolve_layout(plan.truncation, include_scalar=False).spec,
    )


def _assert_state_close(actual, expected, *, tolerance):
    assert actual.spec == expected.spec
    for actual_block, expected_block in zip(actual.blocks, expected.blocks):
        np.testing.assert_allclose(
            actual_block,
            expected_block,
            atol=tolerance,
            rtol=tolerance,
        )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_interpreter_matches_native_partial_state_over_multiple_steps(dtype):
    core, plan, y, coef, (E, psi, phi) = _case(dtype)

    got = quotient_fssk_q1_pallas(
        y,
        E,
        psi,
        phi,
        plan=plan,
        tile_prime_words=2,
        interpret=True,
    )
    expected = fssk_state_from_coef(
        jnp.moveaxis(y, 1, 0),
        coef=coef,
        trunc=plan.truncation,
        axis=0,
        core=core,
    )

    tolerance = 4e-6 if dtype == jnp.float32 else 4e-12
    _assert_state_close(got, expected, tolerance=tolerance)


@pytest.mark.parametrize("accumulate", [True, False])
def test_compact_nonzero_seed_and_emitted_blocks_match_native(accumulate):
    core, plan, y, coef, (E, psi, phi) = _case(jnp.float64)
    initial = _random_initial_state(
        core,
        plan,
        batch=y.shape[0],
        state_dim=coef.R,
        dtype=y.dtype,
    )

    got = quotient_fssk_q1_pallas(
        y,
        E,
        psi,
        phi,
        plan=plan,
        initial_state=initial,
        block_size=2,
        accumulate=accumulate,
        tile_prime_words=2,
        interpret=True,
    )
    expected = fssk_state_from_coef(
        jnp.moveaxis(y, 1, 0),
        coef=coef,
        trunc=plan.truncation,
        axis=0,
        initial_state=initial,
        block_size=2,
        accumulate=accumulate,
        core=core,
    )
    expected = BigradedTensor(
        tuple(jnp.moveaxis(block, 0, 1) for block in expected.blocks),
        expected.spec,
    )

    _assert_state_close(got, expected, tolerance=4e-12)


def test_uniform_coefficients_and_jit_nesting_match_native():
    core, plan, y, _coef, _canonical = _case(jnp.float64, batch=1)
    kernel = _kernel(jnp.float64)
    coef = kernel.coef(0.09, trunc=sum(plan.truncation), dtype=y.dtype)
    E = coef.E[None, None]
    psi = coef.psi[None, None]
    phi = coef.phi[0][None, None]
    compute = jax.jit(
        lambda value: quotient_fssk_q1_pallas(
            value,
            E,
            psi,
            phi,
            plan=plan,
            tile_prime_words=2,
            interpret=True,
        )
    )

    got = compute(y)
    expected = fssk_state_from_coef(
        jnp.moveaxis(y, 1, 0),
        coef=coef,
        trunc=plan.truncation,
        axis=0,
        core=core,
    )

    _assert_state_close(got, expected, tolerance=4e-12)


def test_terminal_accumulate_flag_reuses_kernel_factories():
    _core, plan, y, _coef, operands = _case(jnp.float32, batch=1)
    clear_pallas_fssk_quotient_cache()
    quotient_fssk_q1_pallas(
        y,
        *operands,
        plan=plan,
        accumulate=True,
        tile_prime_words=2,
        interpret=True,
    )
    cache_size = pallas_fssk_quotient._quotient_fssk_call.cache_info().currsize
    quotient_fssk_q1_pallas(
        y,
        *operands,
        plan=plan,
        accumulate=False,
        tile_prime_words=2,
        interpret=True,
    )
    assert (
        pallas_fssk_quotient._quotient_fssk_call.cache_info().currsize
        == cache_size
    )


def test_mixed_singleton_and_batched_coefficients_match_native():
    core, plan, y, coef, (E, psi, phi) = _case(jnp.float64, batch=2)
    E = jnp.broadcast_to(E, (2,) + E.shape[1:]).at[1].multiply(0.96)
    phi = jnp.broadcast_to(phi, (2,) + phi.shape[1:]).at[1].multiply(1.03)
    mixed_coef = replace(
        coef,
        E=jnp.moveaxis(E, 0, 1),
        psi=jnp.moveaxis(psi, 0, 1),
        phi=jnp.moveaxis(phi, 0, 1)[:, :, None],
    )

    got = quotient_fssk_q1_pallas(
        y,
        E,
        psi,
        phi,
        plan=plan,
        tile_prime_words=2,
        interpret=True,
    )
    expected = fssk_state_from_coef(
        jnp.moveaxis(y, 1, 0),
        coef=mixed_coef,
        trunc=plan.truncation,
        axis=0,
        core=core,
    )

    _assert_state_close(got, expected, tolerance=4e-12)


def test_non_power_of_two_state_dimension_is_padded_and_sliced():
    core, plan, y, coef, (E, psi, phi) = _case(
        jnp.float64,
        batch=1,
        state_dim=3,
    )

    got = quotient_fssk_q1_pallas(
        y,
        E,
        psi,
        phi,
        plan=plan,
        tile_prime_words=2,
        interpret=True,
    )
    expected = fssk_state_from_coef(
        jnp.moveaxis(y, 1, 0),
        coef=coef,
        trunc=plan.truncation,
        axis=0,
        core=core,
    )

    assert all(block.shape[-2] == 3 for block in got.blocks)
    _assert_state_close(got, expected, tolerance=4e-12)


@pytest.mark.parametrize("weight_batch", (1, 2))
def test_terminal_readout_emission_matches_hidden_state_contraction(
    weight_batch,
):
    core, plan, y, _coef, operands = _case(
        jnp.float64,
        batch=2,
        state_dim=3,
    )
    weights = jnp.linspace(
        -0.25,
        0.75,
        weight_batch * 3,
        dtype=y.dtype,
    ).reshape(weight_batch, 3)

    hidden = quotient_fssk_q1_pallas(
        y,
        *operands,
        plan=plan,
        tile_prime_words=2,
        interpret=True,
    )
    got = quotient_fssk_q1_readout_pallas(
        y,
        *operands,
        weights,
        plan=plan,
        tile_prime_words=2,
        interpret=True,
    )
    hidden_blocks = hidden.blocks
    broadcast_weights = jnp.broadcast_to(weights, (y.shape[0], 3))
    positive = tuple(
        jnp.sum(
            block[:, 0, 0] * broadcast_weights[:, :, None],
            axis=-2,
        )
        for block in hidden_blocks
    )
    expected = plan.assemble_signature(
        (jnp.ones((y.shape[0], 1), dtype=y.dtype), *positive)
    )

    _assert_state_close(got, expected, tolerance=4e-12)
    assert got.spec == core.resolve_layout(plan.truncation).spec


def test_terminal_readout_values_do_not_expand_the_kernel_cache_key():
    _core, plan, y, _coef, operands = _case(
        jnp.float32,
        batch=1,
        state_dim=3,
    )
    clear_pallas_fssk_quotient_cache()
    quotient_fssk_q1_readout_pallas(
        y,
        *operands,
        jnp.ones((1, 3), dtype=y.dtype),
        plan=plan,
        tile_prime_words=2,
        interpret=True,
    )
    cache_size = pallas_fssk_quotient._quotient_fssk_call.cache_info().currsize
    quotient_fssk_q1_readout_pallas(
        y,
        *operands,
        jnp.zeros((1, 3), dtype=y.dtype),
        plan=plan,
        tile_prime_words=2,
        interpret=True,
    )
    assert (
        pallas_fssk_quotient._quotient_fssk_call.cache_info().currsize
        == cache_size
    )


def _supported_gpu_devices():
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        return ()
    return tuple(device for device in devices if _supported_cuda_device(device))


GPU_DEVICES = _supported_gpu_devices()


@pytest.mark.skipif(
    not GPU_DEVICES,
    reason="requires a supported NVIDIA JAX device",
)
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
@pytest.mark.parametrize("state_dim", (2, 3))
def test_numerical_operands_stay_on_real_gpu(dtype, state_dim):
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        core, plan, y, coef, operands = _case(
            dtype,
            batch=1,
            state_dim=state_dim,
        )
        initial = _random_initial_state(
            core,
            plan,
            batch=1,
            state_dim=state_dim,
            dtype=dtype,
        )
        expected = fssk_state_from_coef(
            jnp.moveaxis(y, 1, 0),
            coef=coef,
            trunc=plan.truncation,
            axis=0,
            initial_state=initial,
            core=core,
        )
        y_gpu = jax.device_put(np.asarray(y), GPU_DEVICES[0])
        weights = jnp.linspace(-0.2, 0.6, state_dim, dtype=dtype)[None]
        got = quotient_fssk_q1_pallas(
            y_gpu,
            *operands,
            plan=plan,
            initial_state=initial,
            tile_prime_words=2,
            interpret=False,
        )
        got_readout = quotient_fssk_q1_readout_pallas(
            y_gpu,
            *operands,
            weights,
            plan=plan,
            initial_state=initial,
            tile_prime_words=2,
            interpret=False,
        )

    tolerance = 4e-6 if dtype == jnp.float32 else 4e-12
    _assert_state_close(got, expected, tolerance=tolerance)
    broadcast_weights = jnp.broadcast_to(weights, (y.shape[0], state_dim))
    positive = tuple(
        jnp.sum(
            block[:, 0, 0] * broadcast_weights[:, :, None],
            axis=-2,
        )
        for block in expected.blocks
    )
    expected_readout = plan.assemble_signature(
        (jnp.ones((y.shape[0], 1), dtype=dtype), *positive)
    )
    _assert_state_close(got_readout, expected_readout, tolerance=tolerance)
    assert all(
        leaf.device == GPU_DEVICES[0]
        for leaf in jax.tree_util.tree_leaves((got, got_readout))
    )


def test_executor_consumes_packed_graphs_without_rank_views(monkeypatch):
    from tensordev._wordwise import plans as plans_module

    _core, plan, y, _coef, (E, psi, phi) = _case(
        jnp.float32,
        batch=1,
        truncation=(1, 1),
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("the packed FSSK executor created a rank view")

    monkeypatch.setattr(plans_module.PrefixGraphPlan, "graph", unexpected)
    got = quotient_fssk_q1_pallas(
        y,
        E,
        psi,
        phi,
        plan=plan,
        tile_prime_words=2,
        interpret=True,
    )

    assert all(np.all(np.isfinite(block)) for block in got.blocks)


def test_host_execution_plan_cache_has_exact_accounting_and_eviction(monkeypatch):
    _core, plan, _y, _coef, _canonical = _case(jnp.float32, batch=1)
    _other_core, other, _y, _coef, _canonical = _case(
        jnp.float32,
        batch=1,
        truncation=(1, 1),
    )
    clear_pallas_fssk_quotient_cache()

    first = pallas_fssk_quotient._build_execution_plan(plan)
    assert pallas_fssk_quotient._build_execution_plan(plan) is first
    assert len(pallas_fssk_quotient._EXECUTION_PLAN_CACHE) == 1
    assert (
        pallas_fssk_quotient._EXECUTION_PLAN_CACHE_BYTES_USED
        == first.memory_bytes()
        == first.grade_offsets.nbytes
    )

    monkeypatch.setattr(
        pallas_fssk_quotient,
        "_EXECUTION_PLAN_CACHE_SIZE",
        1,
    )
    second = pallas_fssk_quotient._build_execution_plan(other)
    assert tuple(pallas_fssk_quotient._EXECUTION_PLAN_CACHE) == (
        other.fingerprint,
    )
    assert pallas_fssk_quotient._EXECUTION_PLAN_CACHE_BYTES_USED == (
        second.memory_bytes()
    )

    monkeypatch.setattr(
        pallas_fssk_quotient,
        "_EXECUTION_PLAN_CACHE_BYTES",
        1,
    )
    clear_pallas_fssk_quotient_cache()
    uncached_first = pallas_fssk_quotient._build_execution_plan(plan)
    uncached_second = pallas_fssk_quotient._build_execution_plan(plan)
    assert uncached_first is not uncached_second
    assert not pallas_fssk_quotient._EXECUTION_PLAN_CACHE
    assert pallas_fssk_quotient._EXECUTION_PLAN_CACHE_BYTES_USED == 0


def test_plan_and_resource_guards_run_before_lowering():
    ordered = td.make_core(dims=(1, 1), max_trunc=(1, 1))
    ordered_plan = build_layout_plan(ordered, (1, 1))
    y = jnp.ones((1, 2, 2), dtype=jnp.float32)
    E = jnp.ones((1, 1, 1, 1), dtype=y.dtype)
    psi = jnp.ones((1, 1, 2, 1), dtype=y.dtype)
    phi = jnp.ones((1, 1, 1, 1, 1), dtype=y.dtype)

    with pytest.raises(PallasFSSKPlanError, match="partially symmetrized"):
        quotient_fssk_q1_pallas(
            y,
            E,
            psi,
            phi,
            plan=ordered_plan,
            interpret=True,
        )

    partial = td.make_core(
        dims=(1, 1),
        max_trunc=(1, 0),
        partially_symmetrized=True,
    )
    partial_plan = build_layout_plan(partial, (1, 0))
    state_dim = 129
    with pytest.raises(PallasOrdinaryResourceError, match="state dimension"):
        quotient_fssk_q1_pallas(
            y,
            jnp.zeros((1, 1, state_dim, state_dim), dtype=y.dtype),
            jnp.zeros((1, 1, 1, state_dim), dtype=y.dtype),
            jnp.zeros((1, 1, 0, state_dim, state_dim), dtype=y.dtype),
            plan=partial_plan,
            interpret=True,
        )
    with pytest.raises(PallasOrdinaryResourceError, match="power of two"):
        quotient_fssk_q1_pallas(
            y,
            E,
            psi[:, :, :1],
            phi[:, :, :0],
            plan=partial_plan,
            tile_prime_words=3,
            interpret=True,
        )
