from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import tensordev as td
import tensordev._wordwise.pallas_quotient as pallas_quotient
from tensordev._wordwise.layout import build_layout_plan
from tensordev._wordwise.pallas_ordinary import (
    PallasOrdinaryResourceError,
    PallasOrdinaryUnsupportedError,
)
from tensordev._wordwise.pallas_quotient import (
    PallasQuotientPlanError,
    quotient_ordinary_pallas,
)
from tensordev._wordwise.reference import quotient_word_reference
from tensordev.development import path_signature


def _increments(dtype, *, batch=2, steps=4):
    return 0.08 * jr.normal(
        jr.PRNGKey(720_000 + steps),
        (batch, steps, 4),
        dtype=dtype,
    )


def _cores(truncation=(2, 2)):
    ordered = td.make_core(dims=(2, 2), max_trunc=truncation)
    quotient = td.make_core(
        dims=(2, 2),
        max_trunc=truncation,
        partially_symmetrized=True,
    )
    return ordered, quotient


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("grade", [(2, 0), (0, 2), (1, 1), (2, 2)])
def test_interpreter_matches_native_and_ordered_reduction(dtype, grade):
    ordered, quotient = _cores()
    plan = build_layout_plan(quotient, (2, 2))
    increments = _increments(dtype)
    block = plan.block(grade)

    got = quotient_ordinary_pallas(
        increments,
        block_plan=block,
        d_prime=2,
        tile_prime_words=2,
        interpret=True,
    )
    direct = path_signature(
        increments,
        trunc=(2, 2),
        increment_input=True,
        accumulate=False,
        core=quotient,
    )[grade]
    ordered_signature = path_signature(
        increments,
        trunc=(2, 2),
        increment_input=True,
        accumulate=False,
        core=ordered,
    )
    reduced = quotient.tensor_partially_symmetrize(ordered_signature)[grade]

    tolerance = 3e-6 if dtype == jnp.float32 else 3e-12
    np.testing.assert_allclose(got, direct, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(got, reduced, atol=tolerance, rtol=tolerance)


def test_packed_kernel_coordinate_matches_scalar_graph_reference():
    _ordered, quotient = _cores()
    plan = build_layout_plan(quotient, (2, 2))
    block = plan.block((2, 2))
    increments = _increments(jnp.float64, batch=1)
    rank = block.prefix_plan.graph_count - 1
    prime_code = block.dense_prime_width - 1
    prime_word = jnp.asarray([1, 1], dtype=jnp.int32)

    got = quotient_ordinary_pallas(
        increments,
        block_plan=block,
        d_prime=2,
        tile_prime_words=2,
        interpret=True,
    )
    expected = quotient_word_reference(
        increments,
        block.prefix_plan.graph(rank),
        prime_word,
    )
    coordinate = rank * block.dense_prime_width + prime_code

    np.testing.assert_allclose(
        got[..., coordinate], expected, atol=2e-12, rtol=2e-12
    )


@pytest.mark.parametrize("accumulate", [True, False])
def test_emitted_blocks_match_native_partial_signature(accumulate):
    _ordered, quotient = _cores((2, 1))
    plan = build_layout_plan(quotient, (2, 1))
    block = plan.block((2, 1))
    increments = _increments(jnp.float64, steps=6)

    got = quotient_ordinary_pallas(
        increments,
        block_plan=block,
        d_prime=2,
        block_size=2,
        accumulate=accumulate,
        tile_prime_words=2,
        interpret=True,
    )
    expected = path_signature(
        increments,
        trunc=(2, 1),
        increment_input=True,
        block_size=2,
        accumulate=accumulate,
        core=quotient,
    )[2, 1]

    assert got.shape == expected.shape == (2, 3, block.width)
    np.testing.assert_allclose(got, expected, atol=2e-12, rtol=2e-12)


def test_executor_never_constructs_per_rank_graph_views(monkeypatch):
    from tensordev._wordwise import plans as plans_module

    _ordered, quotient = _cores((1, 1))
    plan = build_layout_plan(quotient, (1, 1))
    block = plan.block((1, 1))
    increments = _increments(jnp.float32, batch=1)

    def unexpected(*args, **kwargs):
        raise AssertionError("the packed executor must not construct rank views")

    monkeypatch.setattr(plans_module.PrefixGraphPlan, "graph", unexpected)
    got = quotient_ordinary_pallas(
        increments,
        block_plan=block,
        d_prime=2,
        tile_prime_words=2,
        interpret=True,
    )

    assert got.shape == (1, block.width)
    assert np.all(np.isfinite(got))


def test_interpreter_can_be_nested_under_jit():
    _ordered, quotient = _cores((1, 1))
    plan = build_layout_plan(quotient, (1, 1))
    block = plan.block((1, 1))
    increments = _increments(jnp.float64, batch=1)
    compute = jax.jit(
        lambda value: quotient_ordinary_pallas(
            value,
            block_plan=block,
            d_prime=2,
            tile_prime_words=2,
            interpret=True,
        )
    )

    got = compute(increments)
    expected = path_signature(
        increments,
        trunc=(1, 1),
        increment_input=True,
        accumulate=False,
        core=quotient,
    )[1, 1]
    np.testing.assert_allclose(got, expected, atol=2e-12, rtol=2e-12)


def test_scalar_block_is_created_without_lowering():
    _ordered, quotient = _cores((1, 1))
    block = build_layout_plan(quotient, (1, 1)).block((0, 0))
    increments = _increments(jnp.float32, batch=1)

    got = quotient_ordinary_pallas(
        increments,
        block_plan=block,
        d_prime=2,
        block_size=2,
        accumulate=False,
        tile_prime_words=2,
        interpret=True,
    )

    np.testing.assert_array_equal(got, np.ones((1, 2, 1), dtype=np.float32))


def test_scalar_block_output_is_created_on_the_increment_device(monkeypatch):
    _ordered, quotient = _cores((1, 1))
    block = build_layout_plan(quotient, (1, 1)).block((0, 0))
    increments = _increments(jnp.float32, batch=1)
    expected_device = increments.device
    observed = []
    original_ones = pallas_quotient.jnp.ones

    def record_ones(*args, **kwargs):
        observed.append(kwargs.get("device"))
        return original_ones(*args, **kwargs)

    monkeypatch.setattr(pallas_quotient.jnp, "ones", record_ones)
    got = quotient_ordinary_pallas(
        increments,
        block_plan=block,
        d_prime=2,
        interpret=True,
    )

    assert observed == [expected_device]
    assert got.device == expected_device


def test_validation_and_resource_errors_precede_lowering():
    ordered, quotient = _cores((1, 1))
    quotient_plan = build_layout_plan(quotient, (1, 1))
    ordered_block = build_layout_plan(ordered, (1, 1)).block((1, 1))
    increments = _increments(jnp.float32, batch=1)

    with pytest.raises(PallasQuotientPlanError, match="packed prefix graph"):
        quotient_ordinary_pallas(
            increments,
            block_plan=ordered_block,
            d_prime=2,
            interpret=True,
        )
    with pytest.raises(PallasOrdinaryUnsupportedError, match="float32 or float64"):
        quotient_ordinary_pallas(
            increments.astype(jnp.int32),
            block_plan=quotient_plan.block((1, 1)),
            d_prime=2,
            interpret=True,
        )
    with pytest.raises(PallasOrdinaryResourceError, match="power of two"):
        quotient_ordinary_pallas(
            increments,
            block_plan=quotient_plan.block((1, 1)),
            d_prime=2,
            tile_prime_words=3,
            interpret=True,
        )
    with pytest.raises(ValueError, match="must divide"):
        quotient_ordinary_pallas(
            increments,
            block_plan=quotient_plan.block((1, 1)),
            d_prime=2,
            block_size=3,
            interpret=True,
        )
