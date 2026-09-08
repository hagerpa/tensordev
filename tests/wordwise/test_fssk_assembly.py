from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import tensordev as td
import tensordev._wordwise.fssk as fssk_assembly
from tensordev._wordwise.fssk import (
    PreparedFSSKQ1Call,
    run_fssk_q1_wordwise,
    run_fssk_q1_wordwise_readout,
    try_fssk_q1_wordwise,
    try_fssk_q1_wordwise_readout,
)
from tensordev._wordwise.layout import build_layout_plan
from tensordev._wordwise.pallas_ordinary import (
    PallasOrdinaryResourceError,
)
from tensordev.sss import DenseLambda, FSSK
from tensordev.sss.state_update import fssk_state_from_coef


def _core_cases():
    return (
        pytest.param(
            td.make_core(dims=2, max_trunc=2),
            2,
            id="total-standard",
        ),
        pytest.param(
            td.make_core(dims=(1, 1), max_trunc=2, coordinates="shear"),
            2,
            id="total-shear",
        ),
        pytest.param(
            td.make_core(dims=(1, 1), max_trunc=(1, 1)),
            (1, 1),
            id="bidegree-standard",
        ),
        pytest.param(
            td.make_core(
                dims=(1, 1),
                max_trunc=(1, 1),
                coordinates="shear",
            ),
            (1, 1),
            id="bidegree-shear",
        ),
        pytest.param(
            td.make_core(
                dims=(1, 1),
                max_trunc=(1, 1),
                partially_symmetrized=True,
            ),
            (1, 1),
            id="partial-standard",
        ),
        pytest.param(
            td.make_core(
                dims=(1, 1),
                max_trunc=(1, 1),
                partially_symmetrized=True,
                coordinates="shear",
            ),
            (1, 1),
            id="partial-shear",
        ),
    )


def _kernel(*, dtype, state_dim=3, alphabet_dim=2):
    return FSSK(
        Lambda=DenseLambda(
            jnp.diag(
                jnp.linspace(0.25, 0.7, state_dim, dtype=dtype)
            )
        ),
        A=jnp.eye(alphabet_dim, dtype=dtype)[None],
        b=jnp.linspace(0.8, -0.2, state_dim, dtype=dtype)[None],
    )


def _standard_seed(
    plan,
    *,
    batch_shape,
    state_dim,
    dtype,
):
    blocks = []
    for index, block in enumerate(plan.blocks[1:]):
        key = jr.PRNGKey(880_100 + index)
        blocks.append(
            0.02
            * jr.normal(
                key,
                batch_shape + (1, 1, state_dim, block.width),
                dtype=dtype,
            )
        )
    return plan.assemble_first_on(blocks)


def _flatten_first_on(tensor, *, plan, batch_shape):
    blocks = tensor.blocks if hasattr(tensor, "blocks") else tensor
    flat_batch = int(np.prod(batch_shape, dtype=np.int64))
    return plan.assemble_first_on(
        tuple(
            block.reshape((flat_batch,) + block.shape[len(batch_shape) :])
            for block in blocks
        )
    )


def _prepared_case(
    core,
    trunc,
    *,
    batch_shape=(2, 2),
    steps=4,
    axis=1,
    block_size=None,
    accumulate=True,
    output_starting_state=False,
    uniform_coefficients=False,
):
    dtype = jnp.float64
    state_dim = 3
    plan = build_layout_plan(core, trunc)
    kernel = _kernel(dtype=dtype, state_dim=state_dim)
    y_time = 0.04 * jr.normal(
        jr.PRNGKey(880_000 + steps + axis),
        (steps,) + batch_shape + (2,),
        dtype=dtype,
    )
    y = jnp.moveaxis(y_time, 0, axis)
    dt = (
        jnp.asarray(0.08, dtype=dtype)
        if uniform_coefficients
        else jnp.linspace(0.04, 0.11, steps, dtype=dtype)
    )
    coef = kernel.coef(dt, trunc=2, dtype=dtype).with_time_axis()
    coefficient_steps = int(coef.E.shape[0])
    flat_batch = int(np.prod(batch_shape, dtype=np.int64))

    E_time = jnp.broadcast_to(
        coef.E[:, None],
        (coefficient_steps, flat_batch) + coef.E.shape[-2:],
    )
    psi_time = jnp.broadcast_to(
        coef.psi[:, None],
        (coefficient_steps, flat_batch) + coef.psi.shape[-2:],
    )
    scalar_phi = coef.phi[:, 0]
    phi_time = jnp.broadcast_to(
        scalar_phi[:, None],
        (coefficient_steps, flat_batch) + scalar_phi.shape[-3:],
    )
    standard_seed = _standard_seed(
        plan,
        batch_shape=batch_shape,
        state_dim=state_dim,
        dtype=dtype,
    )
    native_seed = core.tensor_from_standard_coordinates(
        standard_seed,
        trunc=trunc,
        first_on=True,
    )
    prepared = PreparedFSSKQ1Call(
        y_time=y_time.reshape((steps, flat_batch, 2)),
        E_time=E_time,
        psi_time=psi_time,
        phi_time=phi_time,
        initial_standard=_flatten_first_on(
            standard_seed,
            plan=plan,
            batch_shape=batch_shape,
        ),
        core=core,
        trunc=trunc,
        batch_shape=batch_shape,
        axis=axis,
        block_size=block_size,
        accumulate=accumulate,
        output_starting_state=output_starting_state,
    )
    expected = fssk_state_from_coef(
        y,
        coef=coef if not uniform_coefficients else kernel.coef(dt, trunc=2),
        trunc=trunc,
        axis=axis,
        block_size=block_size,
        accumulate=accumulate,
        initial_state=native_seed,
        output_starting_state=False,
        core=core,
    )
    expected = core.tensor_to_standard_coordinates(
        expected,
        trunc=trunc,
        first_on=True,
    )
    blocks = expected.blocks if hasattr(expected, "blocks") else expected
    block_count = 1 if block_size is None else steps // block_size
    if block_count == 1:
        canonical_blocks = tuple(
            block.reshape((flat_batch,) + block.shape[len(batch_shape) :])[None]
            for block in blocks
        )
    else:
        canonical_blocks = tuple(
            jnp.moveaxis(block, axis, 0).reshape(
                (block_count, flat_batch) + block.shape[-4:]
            )
            for block in blocks
        )
    return prepared, plan, plan.assemble_first_on(canonical_blocks)


def _assert_state_close(actual, expected):
    if hasattr(actual, "spec"):
        assert actual.spec == expected.spec
        actual_blocks = actual.blocks
        expected_blocks = expected.blocks
    else:
        actual_blocks = tuple(actual)
        expected_blocks = tuple(expected)
    assert len(actual_blocks) == len(expected_blocks)
    for actual_block, expected_block in zip(actual_blocks, expected_blocks):
        assert actual_block.shape == expected_block.shape
        np.testing.assert_allclose(
            np.asarray(actual_block),
            np.asarray(expected_block),
            atol=5e-12,
            rtol=5e-12,
        )


@pytest.mark.parametrize("core,trunc", _core_cases())
def test_terminal_all_core_families_match_portable_with_r3_seed(core, trunc):
    prepared, plan, expected = _prepared_case(
        core,
        trunc,
        batch_shape=(2, 2),
        steps=2,
        axis=1,
    )

    actual = run_fssk_q1_wordwise(
        prepared,
        plan=plan,
        interpret=True,
        tile_words=4,
        tile_prime_words=2,
    )

    _assert_state_close(actual, expected)


@pytest.mark.parametrize("core,trunc", _core_cases())
@pytest.mark.parametrize("weight_batch", (1, 2))
def test_fused_terminal_readout_all_core_families(core, trunc, weight_batch):
    prepared, plan, _ = _prepared_case(
        core,
        trunc,
        batch_shape=(2,),
        steps=2,
        axis=0,
    )
    weights = jnp.linspace(
        -0.4,
        0.7,
        weight_batch * 3,
        dtype=prepared.y_time.dtype,
    ).reshape(weight_batch, 3)
    hidden = run_fssk_q1_wordwise(
        prepared,
        plan=plan,
        interpret=True,
        tile_words=4,
        tile_prime_words=2,
    )

    got = run_fssk_q1_wordwise_readout(
        prepared,
        weights,
        plan=plan,
        interpret=True,
        tile_words=4,
        tile_prime_words=2,
    )
    hidden_blocks = hidden.blocks if hasattr(hidden, "blocks") else hidden
    broadcast_weights = jnp.broadcast_to(weights, (2, 3))
    positive = tuple(
        jnp.sum(
            block[0, :, 0, 0] * broadcast_weights[:, :, None],
            axis=-2,
        )
        for block in hidden_blocks
    )
    expected = plan.assemble_signature(
        (jnp.ones((2, 1), dtype=prepared.y_time.dtype), *positive)
    )

    _assert_state_close(got, expected)
    if hasattr(got, "spec"):
        assert got.spec.coordinates == "standard"


@pytest.mark.parametrize("core,trunc", _core_cases())
@pytest.mark.parametrize("accumulate,axis", [(True, 1), (False, 0)])
def test_blocked_all_core_families_return_canonical_standard_states(
    core,
    trunc,
    accumulate,
    axis,
):
    prepared, plan, expected = _prepared_case(
        core,
        trunc,
        batch_shape=(2, 1),
        steps=4,
        axis=axis,
        block_size=2,
        accumulate=accumulate,
        output_starting_state=True,
    )

    actual = run_fssk_q1_wordwise(
        prepared,
        plan=plan,
        interpret=True,
        tile_words=4,
        tile_prime_words=2,
    )

    _assert_state_close(actual, expected)

    blocks = actual.blocks if hasattr(actual, "blocks") else actual
    assert all(block.shape[:2] == (2, 2) for block in blocks)
    if hasattr(actual, "spec"):
        assert actual.spec.coordinates == "standard"


def test_single_coefficient_time_step_is_preserved():
    core = td.make_core(dims=2, max_trunc=2)
    prepared, plan, expected = _prepared_case(
        core,
        2,
        batch_shape=(2,),
        steps=2,
        axis=0,
        uniform_coefficients=True,
    )

    actual = run_fssk_q1_wordwise(
        prepared,
        plan=plan,
        interpret=True,
        tile_words=4,
    )

    assert prepared.E_time.shape[:2] == (1, 2)
    _assert_state_close(actual, expected)


@pytest.mark.parametrize(
    "error",
    [
        PallasOrdinaryResourceError("bounded resource failure"),
    ],
)
def test_fallback_helper_returns_none_only_for_bounded_errors(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(fssk_assembly, "run_fssk_q1_wordwise", fail)
    assert try_fssk_q1_wordwise(object()) is None

    def invalid(*args, **kwargs):
        raise ValueError("invalid prepared call")

    monkeypatch.setattr(fssk_assembly, "run_fssk_q1_wordwise", invalid)
    with pytest.raises(ValueError, match="invalid prepared call"):
        try_fssk_q1_wordwise(object())

    def unexpected(*args, **kwargs):
        raise RuntimeError("backend execution failed")

    monkeypatch.setattr(fssk_assembly, "run_fssk_q1_wordwise", unexpected)
    with pytest.raises(RuntimeError, match="backend execution failed"):
        try_fssk_q1_wordwise(object())


def test_readout_helper_falls_back_for_nonterminal_output_requests():
    core = td.make_core(dims=2, max_trunc=2)
    blocked, _, _ = _prepared_case(
        core,
        2,
        batch_shape=(1,),
        steps=2,
        axis=0,
        block_size=1,
    )
    starting, _, _ = _prepared_case(
        core,
        2,
        batch_shape=(1,),
        steps=2,
        axis=0,
        output_starting_state=True,
    )
    weights = jnp.ones((1, 3), dtype=blocked.y_time.dtype)

    assert try_fssk_q1_wordwise_readout(blocked, weights) is None
    assert try_fssk_q1_wordwise_readout(starting, weights) is None
