from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import tensordev as td
import tensordev._wordwise.pallas_fssk_q1 as pallas_fssk
from tensordev._wordwise.dispatch import _supported_cuda_device
from tensordev._wordwise.layout import build_layout_plan
from tensordev._wordwise.pallas_fssk_q1 import (
    FSSKPlanLimits,
    PallasFSSKPlanError,
    build_ordered_fssk_execution_plan,
    clear_pallas_fssk_q1_cache,
    ordered_fssk_q1_pallas,
    ordered_fssk_q1_readout_pallas,
)
from tensordev._wordwise.pallas_fssk_quotient import quotient_fssk_q1_pallas
from tensordev._wordwise.pallas_ordinary import (
    PallasOrdinaryError,
    PallasOrdinaryResourceError,
    PallasOrdinaryUnsupportedError,
)
from tensordev.sss import DenseLambda, FSSK
from tensordev.sss.state_update import fssk_state_from_coef


def _kernel(dtype, *, state_dim=2, latent_dim=2) -> FSSK:
    diagonal = jnp.linspace(0.2, 0.7, state_dim, dtype=dtype)
    return FSSK(
        Lambda=DenseLambda(jnp.diag(diagonal)),
        A=jnp.eye(latent_dim, dtype=dtype)[None],
        b=jnp.linspace(0.8, -0.1, state_dim, dtype=dtype)[None],
    )


def _case(dtype, grading, *, batch=2, steps=4, state_dim=2):
    if grading == "total_degree":
        latent_dim = 2
        core = td.make_core(dims=2, max_trunc=3)
        truncation = 3
    else:
        latent_dim = 3
        core = td.make_core(dims=(1, 2), max_trunc=(2, 1))
        truncation = (2, 1)
    kernel = _kernel(dtype, state_dim=state_dim, latent_dim=latent_dim)
    y = 0.08 * jr.normal(
        jr.PRNGKey(7200 + steps),
        (batch, steps, latent_dim),
        dtype=dtype,
    )
    plan = build_layout_plan(core, truncation)
    coef = kernel.coef(
        jnp.linspace(0.04, 0.12, steps, dtype=dtype),
        trunc=3,
        dtype=dtype,
    )
    operands = (coef.E[None], coef.psi[None], coef.phi[:, 0][None])
    return kernel, core, truncation, plan, y, coef, operands


def _assert_state_close(actual, expected, *, dtype) -> None:
    if hasattr(actual, "spec"):
        assert actual.spec == expected.spec
        actual_blocks = actual.blocks
        expected_blocks = expected.blocks
    else:
        actual_blocks = tuple(actual)
        expected_blocks = tuple(expected)
    assert len(actual_blocks) == len(expected_blocks)
    tolerance = 3e-5 if dtype == jnp.float32 else 3e-12
    for actual_block, expected_block in zip(actual_blocks, expected_blocks):
        assert actual_block.shape == expected_block.shape
        np.testing.assert_allclose(
            np.asarray(actual_block),
            np.asarray(expected_block),
            atol=tolerance,
            rtol=tolerance,
        )


def _readout_reference(state, weights, *, plan):
    blocks = state.blocks if hasattr(state, "blocks") else state
    batch = int(blocks[0].shape[0])
    weights = jnp.broadcast_to(weights, (batch, weights.shape[-1]))
    positive = tuple(
        jnp.sum(
            block[:, 0, 0] * weights[:, :, None],
            axis=-2,
        )
        for block in blocks
    )
    unit = jnp.ones((batch, 1), dtype=blocks[0].dtype)
    return plan.assemble_signature((unit, *positive))


def _jaxpr_equations(value):
    nested = getattr(value, "jaxpr", None)
    jaxpr = nested if nested is not None else value
    if hasattr(jaxpr, "eqns"):
        equations = list(jaxpr.eqns)
        for equation in jaxpr.eqns:
            for parameter in equation.params.values():
                equations.extend(_jaxpr_equations(parameter))
        return equations
    if isinstance(value, Mapping):
        equations = []
        for parameter in value.values():
            equations.extend(_jaxpr_equations(parameter))
        return equations
    if isinstance(value, (tuple, list)):
        equations = []
        for parameter in value:
            equations.extend(_jaxpr_equations(parameter))
        return equations
    return []


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
@pytest.mark.parametrize("grading", ("total_degree", "bidegree"))
@pytest.mark.parametrize(
    ("block_size", "accumulate"),
    ((None, True), (2, True), (2, False)),
)
def test_zero_seed_interpreter_matches_every_portable_state_coordinate(
    dtype,
    grading,
    block_size,
    accumulate,
):
    _, core, truncation, plan, y, coef, operands = _case(dtype, grading)

    actual = ordered_fssk_q1_pallas(
        y,
        *operands,
        plan=plan,
        block_size=block_size,
        accumulate=accumulate,
        tile_words=4,
        interpret=True,
    )
    expected = fssk_state_from_coef(
        y,
        coef=coef,
        trunc=truncation,
        axis=1,
        block_size=block_size,
        accumulate=accumulate,
        core=core,
    )

    _assert_state_close(actual, expected, dtype=dtype)


def _initial_state(plan, *, state_dim, dtype):
    blocks = []
    offset = 1
    for block in plan.blocks:
        if block.total_degree == 0:
            continue
        size = state_dim * block.width
        value = jnp.arange(offset, offset + size, dtype=dtype)
        value = (0.003 * value).reshape(1, 1, state_dim, block.width)
        blocks.append(value)
        offset += size
    return plan.assemble_first_on(blocks)


@pytest.mark.parametrize("grading", ("total_degree", "bidegree"))
@pytest.mark.parametrize("accumulate", (True, False))
def test_compact_arbitrary_seed_matches_portable_blocks(grading, accumulate):
    dtype = jnp.float64
    kernel, core, truncation, plan, y, coef, operands = _case(dtype, grading)
    initial = _initial_state(
        plan,
        state_dim=kernel.state_dim,
        dtype=dtype,
    )

    actual = ordered_fssk_q1_pallas(
        y,
        *operands,
        plan=plan,
        initial_state=initial,
        block_size=2,
        accumulate=accumulate,
        tile_words=4,
        interpret=True,
    )
    expected = fssk_state_from_coef(
        y,
        coef=coef,
        trunc=truncation,
        axis=1,
        block_size=2,
        accumulate=accumulate,
        initial_state=initial,
        core=core,
    )

    _assert_state_close(actual, expected, dtype=dtype)


def test_batch_specific_compact_seed_matches_portable_bidegree_state():
    dtype = jnp.float64
    kernel, core, truncation, plan, y, coef, operands = _case(
        dtype,
        "bidegree",
        batch=2,
    )
    unbatched = _initial_state(
        plan,
        state_dim=kernel.state_dim,
        dtype=dtype,
    )
    initial = plan.assemble_first_on(
        tuple(
            jnp.broadcast_to(block[None], (2,) + block.shape)
            + jnp.asarray([0.0, 0.02], dtype=dtype)[:, None, None, None, None]
            for block in unbatched.blocks
        )
    )

    actual = ordered_fssk_q1_pallas(
        y,
        *operands,
        plan=plan,
        initial_state=initial,
        block_size=2,
        tile_words=4,
        interpret=True,
    )
    expected = fssk_state_from_coef(
        y,
        coef=coef,
        trunc=truncation,
        axis=1,
        block_size=2,
        initial_state=initial,
        core=core,
    )

    _assert_state_close(actual, expected, dtype=dtype)


def test_mixed_singleton_and_batched_coefficients_match_portable_state():
    dtype = jnp.float64
    _, core, truncation, plan, y, coef, (E, psi, phi) = _case(
        dtype,
        "bidegree",
        batch=2,
    )
    E = jnp.broadcast_to(E, (2,) + E.shape[1:]).at[1].multiply(0.97)
    phi = jnp.broadcast_to(phi, (2,) + phi.shape[1:]).at[1].multiply(1.04)
    mixed_coef = replace(
        coef,
        E=jnp.moveaxis(E, 0, 1),
        psi=jnp.moveaxis(psi, 0, 1),
        phi=jnp.moveaxis(phi, 0, 1)[:, :, None],
    )

    actual = ordered_fssk_q1_pallas(
        y,
        E,
        psi,
        phi,
        plan=plan,
        tile_words=4,
        interpret=True,
    )
    expected = fssk_state_from_coef(
        y,
        coef=mixed_coef,
        trunc=truncation,
        axis=1,
        core=core,
    )

    _assert_state_close(actual, expected, dtype=dtype)


@pytest.mark.parametrize("grading", ("total_degree", "bidegree"))
@pytest.mark.parametrize("weight_batch", (1, 2))
def test_terminal_readout_emission_matches_hidden_state_contraction(
    grading,
    weight_batch,
):
    dtype = jnp.float64
    _, _, _, plan, y, _, operands = _case(
        dtype,
        grading,
        batch=2,
        state_dim=3,
    )
    weights = jnp.linspace(
        -0.3,
        0.8,
        weight_batch * 3,
        dtype=dtype,
    ).reshape(weight_batch, 3)

    hidden = ordered_fssk_q1_pallas(
        y,
        *operands,
        plan=plan,
        tile_words=4,
        interpret=True,
    )
    got = ordered_fssk_q1_readout_pallas(
        y,
        *operands,
        weights,
        plan=plan,
        tile_words=4,
        interpret=True,
    )
    expected = _readout_reference(hidden, weights, plan=plan)

    _assert_state_close(got, expected, dtype=dtype)


def test_terminal_readout_values_do_not_expand_the_kernel_cache_key():
    _, _, _, plan, y, _, operands = _case(
        jnp.float32,
        "bidegree",
        batch=1,
        state_dim=3,
    )
    clear_pallas_fssk_q1_cache()
    ordered_fssk_q1_readout_pallas(
        y,
        *operands,
        jnp.ones((1, 3), dtype=y.dtype),
        plan=plan,
        tile_words=4,
        interpret=True,
    )
    cache_size = pallas_fssk._ordered_fssk_call.cache_info().currsize
    ordered_fssk_q1_readout_pallas(
        y,
        *operands,
        jnp.zeros((1, 3), dtype=y.dtype),
        plan=plan,
        tile_words=4,
        interpret=True,
    )
    assert pallas_fssk._ordered_fssk_call.cache_info().currsize == cache_size


def test_uniform_coefficients_use_one_coefficient_step():
    dtype = jnp.float64
    kernel, core, truncation, plan, y, _, _ = _case(dtype, "total_degree")
    coef = kernel.coef(0.07, trunc=3, dtype=dtype)

    actual = ordered_fssk_q1_pallas(
        y,
        coef.E[None, None],
        coef.psi[None, None],
        coef.phi[0][None, None],
        plan=plan,
        tile_words=4,
        interpret=True,
    )
    expected = fssk_state_from_coef(
        y,
        coef=coef,
        trunc=truncation,
        axis=1,
        core=core,
    )

    _assert_state_close(actual, expected, dtype=dtype)


def test_terminal_accumulate_flag_reuses_kernel_factories():
    _, _, _, plan, y, _, operands = _case(
        jnp.float32,
        "bidegree",
        batch=1,
    )
    clear_pallas_fssk_q1_cache()
    ordered_fssk_q1_pallas(
        y,
        *operands,
        plan=plan,
        accumulate=True,
        tile_words=4,
        interpret=True,
    )
    cache_size = pallas_fssk._ordered_fssk_call.cache_info().currsize
    ordered_fssk_q1_pallas(
        y,
        *operands,
        plan=plan,
        accumulate=False,
        tile_words=4,
        interpret=True,
    )
    assert pallas_fssk._ordered_fssk_call.cache_info().currsize == cache_size


def test_interpreter_can_be_nested_under_jit_with_an_arbitrary_seed():
    dtype = jnp.float64
    kernel, core, truncation, plan, y, coef, operands = _case(
        dtype,
        "bidegree",
        batch=1,
    )
    initial = _initial_state(
        plan,
        state_dim=kernel.state_dim,
        dtype=dtype,
    )
    compute = jax.jit(
        lambda value, state: ordered_fssk_q1_pallas(
            value,
            *operands,
            plan=plan,
            initial_state=state,
            block_size=2,
            accumulate=True,
            tile_words=4,
            interpret=True,
        )
    )

    actual = compute(y, initial)
    expected = fssk_state_from_coef(
        y,
        coef=coef,
        trunc=truncation,
        axis=1,
        block_size=2,
        accumulate=True,
        initial_state=initial,
        core=core,
    )

    _assert_state_close(actual, expected, dtype=dtype)


@pytest.mark.parametrize("executor", ("ordered", "quotient"))
def test_time_loop_jaxpr_does_not_grow_with_the_step_count(executor):
    dtype = jnp.float32
    kernel = _kernel(dtype, state_dim=3, latent_dim=2)
    coef = kernel.coef(0.07, trunc=3, dtype=dtype)
    operands = (
        coef.E[None, None],
        coef.psi[None, None],
        coef.phi[0][None, None],
    )
    if executor == "ordered":
        core = td.make_core(dims=2, max_trunc=3)
        plan = build_layout_plan(core, 3)
        execute = lambda value: ordered_fssk_q1_pallas(
            value,
            *operands,
            plan=plan,
            tile_words=4,
            interpret=True,
        )
    else:
        core = td.make_core(
            dims=(1, 1),
            max_trunc=(2, 1),
            partially_symmetrized=True,
        )
        plan = build_layout_plan(core, (2, 1))
        execute = lambda value: quotient_fssk_q1_pallas(
            value,
            *operands,
            plan=plan,
            tile_prime_words=2,
            interpret=True,
        )

    summaries = []
    for steps in (2, 8):
        closed = jax.make_jaxpr(execute)(
            jnp.ones((1, steps, 2), dtype=dtype)
        )
        equations = _jaxpr_equations(closed)
        scans = [
            equation
            for equation in equations
            if equation.primitive.name == "scan"
        ]
        summaries.append(
            (
                len(equations),
                tuple(equation.params["length"] for equation in scans),
            )
        )

    assert summaries[0][0] == summaries[1][0]
    assert summaries[0][1] and set(summaries[0][1]) == {2}
    assert summaries[1][1] and set(summaries[1][1]) == {8}


@pytest.mark.parametrize("grading", ("total_degree", "bidegree"))
def test_non_power_of_two_state_extent_matches_portable(grading):
    dtype = jnp.float32
    _, core, truncation, plan, y, coef, operands = _case(
        dtype,
        grading,
        batch=1,
        state_dim=3,
    )

    actual = ordered_fssk_q1_pallas(
        y,
        *operands,
        plan=plan,
        tile_words=4,
        interpret=True,
    )
    expected = fssk_state_from_coef(
        y,
        coef=coef,
        trunc=truncation,
        axis=1,
        core=core,
    )

    _assert_state_close(actual, expected, dtype=dtype)


@pytest.mark.parametrize("state_dim", (2, 3))
def test_numerical_operands_use_shared_colocation_before_padding(
    monkeypatch,
    state_dim,
):
    _, _, _, plan, y, _, operands = _case(
        jnp.float32,
        "bidegree",
        batch=10_000,
        state_dim=state_dim,
    )
    observed = []
    colocate = pallas_fssk._colocate_array

    def record(value, reference):
        observed.append((value, reference))
        return colocate(value, reference)

    monkeypatch.setattr(pallas_fssk, "_colocate_array", record)
    canonical = pallas_fssk._canonicalize_inputs(y, *operands)

    assert canonical.coefficient_batches == (1, 1, 1)
    assert canonical.flat_batch == 10_000
    assert all(value.shape[0] == 1 for value in (
        canonical.E,
        canonical.psi,
        canonical.phi,
    ))
    assert [id(value) for value, _ in observed] == [
        id(value) for value in operands
    ]
    assert all(reference is canonical.y for _, reference in observed)

    observed.clear()
    initial = pallas_fssk._pack_initial_state(
        None,
        plan=plan,
        flat_batch=canonical.flat_batch,
        state_dim=canonical.state_dim,
        dtype=canonical.y.dtype,
        reference=canonical.y,
    )
    assert initial.shape[0] == 1
    extent = pallas_fssk._fssk_state_extent(canonical.state_dim)
    padded = pallas_fssk._pad_fssk_state_operands(
        canonical,
        initial,
        state_extent=extent,
    )
    assert not observed
    assert all(value.device == canonical.y.device for value in padded)

    raw_weights = np.ones((1, state_dim), dtype=np.float32)
    weights, weight_mode = pallas_fssk._canonicalize_readout_weights(
        raw_weights,
        canonical,
        state_extent=extent,
    )
    assert observed == [(raw_weights, canonical.y)]
    assert weight_mode == 1
    assert weights.shape == (1, extent)
    assert weights.device == canonical.y.device


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
@pytest.mark.parametrize("grading", ("total_degree", "bidegree"))
@pytest.mark.parametrize("state_dim", (2, 3))
def test_numerical_operands_stay_on_real_gpu(dtype, grading, state_dim):
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        kernel, core, truncation, plan, y, coef, operands = _case(
            dtype,
            grading,
            batch=1,
            state_dim=state_dim,
        )
        initial = _initial_state(
            plan,
            state_dim=kernel.state_dim,
            dtype=dtype,
        )
        expected = fssk_state_from_coef(
            y,
            coef=coef,
            trunc=truncation,
            axis=1,
            block_size=2,
            initial_state=initial,
            core=core,
        )
        expected_terminal = fssk_state_from_coef(
            y,
            coef=coef,
            trunc=truncation,
            axis=1,
            initial_state=initial,
            core=core,
        )
        y_gpu = jax.device_put(np.asarray(y), GPU_DEVICES[0])
        weights = jnp.linspace(-0.2, 0.6, state_dim, dtype=dtype)[None]
        actual = ordered_fssk_q1_pallas(
            y_gpu,
            *operands,
            plan=plan,
            initial_state=initial,
            block_size=2,
            tile_words=4,
            interpret=False,
        )
        actual_readout = ordered_fssk_q1_readout_pallas(
            y_gpu,
            *operands,
            weights,
            plan=plan,
            initial_state=initial,
            tile_words=4,
            interpret=False,
        )
        expected_readout = _readout_reference(
            expected_terminal,
            weights,
            plan=plan,
        )

    _assert_state_close(actual, expected, dtype=dtype)
    _assert_state_close(actual_readout, expected_readout, dtype=dtype)
    assert all(
        leaf.device == GPU_DEVICES[0]
        for leaf in jax.tree_util.tree_leaves((actual, actual_readout))
    )


def test_bidegree_prefix_plan_is_compact_bounded_and_cached(monkeypatch):
    _, _, _, plan, _, _, _ = _case(jnp.float32, "bidegree", batch=1)
    first = build_ordered_fssk_execution_plan(plan)
    second = build_ordered_fssk_execution_plan(plan)

    assert first is second
    assert first.positive_width == plan.output_size - 1
    assert first.memory_bytes() == sum(
        group.width * group.total_degree * np.dtype(np.int32).itemsize
        for group in first.groups
    )
    for group in first.groups:
        assert group.prefix_indices.shape == (group.width, group.total_degree)
        assert np.all(group.prefix_indices >= 0)
        assert np.all(group.prefix_indices < first.positive_width)

    monkeypatch.setattr(pallas_fssk, "_EXECUTION_PLAN_CACHE_BYTES", 1)
    clear_pallas_fssk_q1_cache()
    uncached_first = build_ordered_fssk_execution_plan(plan)
    uncached_second = build_ordered_fssk_execution_plan(plan)
    assert uncached_first is not uncached_second

    with pytest.raises(PallasOrdinaryResourceError, match="prefix metadata"):
        build_ordered_fssk_execution_plan(
            plan,
            limits=FSSKPlanLimits(
                max_prefix_index_elements=1,
                max_plan_bytes=1024,
            ),
        )


def test_seed_resource_guard_precedes_seed_allocation(monkeypatch):
    _, _, _, plan, y, _, _ = _case(
        jnp.float32,
        "bidegree",
        batch=1,
    )

    def reject_seed(**_):
        raise PallasOrdinaryResourceError("seed guard sentinel")

    def fail_allocation(*_, **__):
        raise AssertionError("seed allocation ran before its resource guard")

    monkeypatch.setattr(pallas_fssk, "_check_seed_resources", reject_seed)
    monkeypatch.setattr(pallas_fssk.jnp, "zeros", fail_allocation)
    with pytest.raises(PallasOrdinaryResourceError, match="seed guard sentinel"):
        pallas_fssk._pack_initial_state(
            None,
            plan=plan,
            flat_batch=1,
            state_dim=2,
            dtype=y.dtype,
            reference=y,
        )

    initial = _initial_state(plan, state_dim=2, dtype=y.dtype)

    def fail_colocation(*_):
        raise AssertionError("seed colocation ran before its resource guard")

    monkeypatch.setattr(pallas_fssk, "_colocate_array", fail_colocation)
    with pytest.raises(PallasOrdinaryResourceError, match="seed guard sentinel"):
        pallas_fssk._pack_initial_state(
            initial,
            plan=plan,
            flat_batch=1,
            state_dim=2,
            dtype=y.dtype,
            reference=y,
        )


def test_state_resource_guard_precedes_coefficient_colocation(monkeypatch):
    y = jnp.ones((1, 2, 2), dtype=jnp.float32)
    state_dim = 129

    def fail_colocation(*_):
        raise AssertionError("coefficient colocation ran before resource guard")

    monkeypatch.setattr(pallas_fssk, "_colocate_array", fail_colocation)
    with pytest.raises(PallasOrdinaryResourceError, match="state dimension"):
        pallas_fssk._canonicalize_inputs(
            y,
            np.zeros((1, 1, state_dim, state_dim), dtype=np.float32),
            np.zeros((1, 1, 1, state_dim), dtype=np.float32),
            np.zeros((1, 1, 0, state_dim, state_dim), dtype=np.float32),
        )


def test_partial_layout_and_invalid_inputs_are_rejected_before_lowering():
    assert issubclass(PallasFSSKPlanError, PallasOrdinaryError)
    quotient = td.make_core(
        dims=(1, 1),
        max_trunc=(1, 1),
        partially_symmetrized=True,
    )
    with pytest.raises(PallasFSSKPlanError, match="ordered layout"):
        build_ordered_fssk_execution_plan(
            build_layout_plan(quotient, (1, 1))
        )

    _, _, _, plan, y, _, operands = _case(jnp.float32, "total_degree")
    with pytest.raises(PallasOrdinaryUnsupportedError, match="float32 or float64"):
        ordered_fssk_q1_pallas(
            y.astype(jnp.int32),
            *operands,
            plan=plan,
            interpret=True,
        )
    with pytest.raises(ValueError, match="must divide"):
        ordered_fssk_q1_pallas(
            y,
            *operands,
            plan=plan,
            block_size=3,
            interpret=True,
        )
    with pytest.raises(PallasOrdinaryResourceError, match="power of two"):
        ordered_fssk_q1_pallas(
            y,
            *operands,
            plan=plan,
            tile_words=3,
            interpret=True,
        )
