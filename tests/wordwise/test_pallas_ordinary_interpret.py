from __future__ import annotations

import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from tensordev import Jax, make_core
import tensordev._wordwise.pallas_ordinary as pallas_ordinary
from tensordev._wordwise.layout import build_layout_plan
from tensordev._wordwise.pallas_ordinary import (
    PallasOrdinaryResourceError,
    PallasOrdinaryUnsupportedError,
    _effective_tile_words,
    ordered_ordinary_pallas,
)
from tensordev.core.jax import JaxSequentialCore
from tensordev.development import path_signature

CORE = Jax()
SEQ = JaxSequentialCore()


def _increments(dtype, *, batch=2, steps=6, alphabet=2):
    return 0.12 * jr.normal(
        jr.PRNGKey(310_000 + steps + alphabet),
        (batch, steps, alphabet),
        dtype=dtype,
    )


def _portable_block(
        increments,
        *,
        degree,
        block_size=None,
        accumulate=True,
):
    return path_signature(
        increments,
        trunc=degree,
        increment_input=True,
        axis=-2,
        block_size=block_size,
        accumulate=accumulate,
        core=CORE,
        seq_core=SEQ,
    )[degree]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize(
    ("block_size", "accumulate"),
    [(None, True), (2, True), (2, False)],
)
def test_dense_interpreter_matches_portable_signature(
    dtype, degree, block_size, accumulate
):
    increments = _increments(dtype)
    got = ordered_ordinary_pallas(
        increments,
        degree=degree,
        block_size=block_size,
        accumulate=accumulate,
        tile_words=4,
        interpret=True,
    )
    expected = _portable_block(
        increments,
        degree=degree,
        block_size=block_size,
        accumulate=accumulate,
    )

    tolerance = 2e-6 if dtype == jnp.float32 else 2e-12
    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize(
    ("block_size", "accumulate"),
    [(None, True), (2, True), (2, False)],
)
def test_explicit_codes_preserve_order_and_padding(block_size, accumulate):
    increments = _increments(jnp.float64)
    codes = np.asarray([7, 0, 3, 5, 1], dtype=np.int32)
    got = ordered_ordinary_pallas(
        increments,
        degree=3,
        word_codes=codes,
        block_size=block_size,
        accumulate=accumulate,
        tile_words=4,
        interpret=True,
    )
    dense = _portable_block(
        increments,
        degree=3,
        block_size=block_size,
        accumulate=accumulate,
    )
    expected = dense[..., codes]

    assert got.shape == expected.shape
    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        atol=2e-12,
        rtol=2e-12,
    )


def test_bidegree_decoder_matches_native_ordered_block():
    core = make_core(
        dims=(1, 2),
        max_trunc=(2, 1),
        precompute_shuffle=False,
    )
    plan = build_layout_plan(core, (2, 1))
    block = plan.block((2, 1))
    increments = _increments(jnp.float64, steps=4, alphabet=3)

    got = ordered_ordinary_pallas(
        increments,
        degree=block.total_degree,
        word_codes=block.word_codes,
        tile_words=4,
        interpret=True,
    )
    expected = path_signature(
        increments,
        trunc=(2, 1),
        increment_input=True,
        axis=-2,
        accumulate=False,
        core=core,
    )[2, 1]

    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        atol=2e-12,
        rtol=2e-12,
    )


@pytest.mark.parametrize("accumulate", [True, False])
def test_scalar_degree_is_assembled_without_a_pallas_launch(accumulate):
    increments = _increments(jnp.float32, steps=6)
    got = ordered_ordinary_pallas(
        increments,
        degree=0,
        block_size=2,
        accumulate=accumulate,
        tile_words=4,
        interpret=True,
    )

    assert got.shape == (2, 3, 1)
    np.testing.assert_array_equal(np.asarray(got), np.ones((2, 3, 1)))


def test_scalar_degree_output_is_created_on_the_increment_device(monkeypatch):
    increments = _increments(jnp.float32, batch=1, steps=2)
    expected_device = increments.device
    observed = []
    original_ones = pallas_ordinary.jnp.ones

    def record_ones(*args, **kwargs):
        observed.append(kwargs.get("device"))
        return original_ones(*args, **kwargs)

    monkeypatch.setattr(pallas_ordinary.jnp, "ones", record_ones)
    got = ordered_ordinary_pallas(
        increments,
        degree=0,
        interpret=True,
    )

    assert observed == [expected_device]
    assert got.device == expected_device


def test_interpreter_can_be_nested_under_jit():
    increments = _increments(jnp.float64, steps=4)
    compute = jax.jit(
        lambda x: ordered_ordinary_pallas(
            x,
            degree=3,
            block_size=2,
            accumulate=True,
            tile_words=4,
            interpret=True,
        )
    )

    got = compute(increments)
    expected = _portable_block(
        increments,
        degree=3,
        block_size=2,
        accumulate=True,
    )
    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        atol=2e-12,
        rtol=2e-12,
    )


def test_explicit_decoder_remains_an_operand_under_jit():
    increments = _increments(jnp.float32, steps=4)
    codes = jnp.asarray([3, 0, 2], dtype=jnp.int32)
    compute = jax.jit(
        lambda x, decoder: ordered_ordinary_pallas(
            x,
            degree=2,
            word_codes=decoder,
            tile_words=4,
            interpret=True,
        )
    )

    got = compute(increments, codes)
    expected = _portable_block(increments, degree=2)[..., codes]
    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize(
    ("word_count", "requested", "effective"),
    [(1, 128, 1), (2, 128, 2), (3, 128, 4), (5, 128, 8), (129, 128, 128)],
)
def test_requested_word_tile_is_an_upper_bound(
    word_count, requested, effective
):
    assert _effective_tile_words(word_count, requested) == effective


@pytest.mark.parametrize(
    ("increments", "match"),
    [
        (jnp.zeros((4, 2), dtype=jnp.float32), "canonical shape"),
        (jnp.zeros((1, 0, 2), dtype=jnp.float32), "at least one increment"),
        (jnp.zeros((0, 4, 2), dtype=jnp.float32), "flat batch"),
        (jnp.zeros((1, 4, 2), dtype=jnp.int32), "float32 or float64"),
    ],
)
def test_unsupported_inputs_fail_before_pallas_lowering(increments, match):
    with pytest.raises((ValueError, PallasOrdinaryUnsupportedError), match=match):
        ordered_ordinary_pallas(
            increments,
            degree=2,
            tile_words=4,
            interpret=True,
        )


def test_decoder_and_block_validation_errors_are_explicit():
    increments = _increments(jnp.float32, steps=6)
    with pytest.raises(PallasOrdinaryUnsupportedError, match="dtype int32"):
        ordered_ordinary_pallas(
            increments,
            degree=2,
            word_codes=np.asarray([0, 1], dtype=np.int64),
            tile_words=4,
            interpret=True,
        )
    with pytest.raises(ValueError, match="must lie in"):
        ordered_ordinary_pallas(
            increments,
            degree=2,
            word_codes=np.asarray([0, 4], dtype=np.int32),
            tile_words=4,
            interpret=True,
        )
    with pytest.raises(ValueError, match="must divide"):
        ordered_ordinary_pallas(
            increments,
            degree=2,
            block_size=4,
            tile_words=4,
            interpret=True,
        )


def test_static_resource_guards_do_not_attempt_large_allocations():
    increments = _increments(jnp.float64, alphabet=1)
    with pytest.raises(PallasOrdinaryResourceError, match="local bytes"):
        ordered_ordinary_pallas(
            increments,
            degree=32,
            word_codes=np.zeros((128,), dtype=np.int32),
            tile_words=128,
            interpret=True,
        )
    with pytest.raises(PallasOrdinaryResourceError, match="power of two"):
        ordered_ordinary_pallas(
            increments,
            degree=2,
            tile_words=3,
            interpret=True,
        )


def test_dense_word_space_is_checked_before_allocation():
    increments = _increments(jnp.float32, alphabet=2)
    with pytest.raises(PallasOrdinaryResourceError, match="int32 indexing"):
        ordered_ordinary_pallas(
            increments,
            degree=31,
            tile_words=4,
            interpret=True,
        )
