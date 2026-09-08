from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
from tensordev._wordwise.ordinary import run_ordinary_wordwise
from tensordev.development.free import (
    _execute_portable_free_development_call,
    _finalize_free_development_call,
    _prepare_free_development_call,
)


def _assert_tensors_close(actual, expected):
    actual_blocks = actual.blocks if hasattr(actual, "blocks") else tuple(actual)
    expected_blocks = (
        expected.blocks if hasattr(expected, "blocks") else tuple(expected)
    )
    assert len(actual_blocks) == len(expected_blocks)
    for actual_block, expected_block in zip(actual_blocks, expected_blocks):
        np.testing.assert_allclose(
            np.asarray(actual_block),
            np.asarray(expected_block),
            atol=2e-12,
            rtol=2e-12,
        )


@pytest.mark.parametrize(
    "core",
    [
        td.make_core(dims=2, max_trunc=3),
        td.make_core(
            dims=(1, 1),
            max_trunc=3,
            coordinates="shear",
        ),
        td.make_core(dims=(1, 1), max_trunc=(2, 1)),
        td.make_core(
            dims=(1, 1),
            max_trunc=(2, 1),
            coordinates="shear",
        ),
        td.make_core(
            dims=(1, 1),
            max_trunc=(2, 1),
            partially_symmetrized=True,
        ),
        td.make_core(
            dims=(1, 1),
            max_trunc=(2, 1),
            partially_symmetrized=True,
            coordinates="shear",
        ),
    ],
)
@pytest.mark.parametrize("accumulate", [True, False])
def test_wordwise_assembly_matches_portable_blocks_and_coordinates(
    core, accumulate
):
    increments = jnp.asarray(
        [
            [[0.10, -0.20], [0.05, 0.04]],
            [[0.02, 0.07], [-0.03, 0.09]],
            [[-0.04, 0.01], [0.08, -0.02]],
            [[0.03, -0.06], [0.01, 0.05]],
        ],
        dtype=jnp.float64,
    )
    call = _prepare_free_development_call(
        (increments,),
        trunc=core.default_truncation,
        increment_input=True,
        axis=0,
        block_size=2,
        accumulate=accumulate,
        core=core,
    )

    raw = run_ordinary_wordwise(
        call,
        interpret=True,
        tile_words=4,
    )
    actual = _finalize_free_development_call(
        call,
        raw,
        runner_applied_seed=False,
        runner_emitted_starting_point=False,
    )
    expected = _execute_portable_free_development_call(call)

    _assert_tensors_close(actual, expected)


@pytest.mark.parametrize(
    ("coordinates", "axis", "shape", "accumulate"),
    [
        ("standard", 0, (6, 2, 2), True),
        ("standard", -3, (2, 6, 3, 2), False),
        ("shear", 0, (6, 2, 2), False),
        ("shear", -3, (2, 6, 3, 2), True),
    ],
)
def test_partial_assembly_preserves_seed_axis_and_block_policy(
    coordinates, axis, shape, accumulate
):
    active = (2, 1)
    core = td.make_core(
        dims=(1, 1),
        max_trunc=active,
        partially_symmetrized=True,
        coordinates=coordinates,
    )
    increments = 0.06 * jax.random.normal(
        jax.random.PRNGKey(332_000 + int(accumulate) + len(shape)),
        shape,
        dtype=jnp.float64,
    )
    moved = jnp.moveaxis(increments, axis, -2)
    seed_increment = 0.04 * jax.random.normal(
        jax.random.PRNGKey(333_000 + int(accumulate) + len(shape)),
        moved.shape[:-2] + (sum(core.dims),),
        dtype=jnp.float64,
    )
    starting_point = core.tensor_exponential(
        (seed_increment,), trunc=active, output_zero_level=True
    )
    call = _prepare_free_development_call(
        (increments,),
        trunc=active,
        increment_input=True,
        axis=axis,
        block_size=2,
        accumulate=accumulate,
        starting_point=starting_point,
        output_starting_point=True,
        core=core,
    )

    raw = run_ordinary_wordwise(
        call,
        interpret=True,
        tile_prime_words=2,
    )
    actual = _finalize_free_development_call(
        call,
        raw,
        runner_applied_seed=False,
        runner_emitted_starting_point=False,
    )
    expected = _execute_portable_free_development_call(call)

    _assert_tensors_close(actual, expected)


def test_dimension_free_total_assembly_infers_alphabet_width():
    core = td.Jax()
    increments = jnp.asarray(
        [[[0.2, -0.1, 0.3], [0.1, 0.05, -0.02]]],
        dtype=jnp.float64,
    )
    call = _prepare_free_development_call(
        (increments,),
        trunc=2,
        increment_input=True,
        core=core,
    )

    actual = run_ordinary_wordwise(
        call,
        interpret=True,
        tile_words=4,
    )
    expected = _execute_portable_free_development_call(call)

    _assert_tensors_close(actual, expected)


def _arbitrary_dense_seed(key, *, batch_shape, dimension, truncation):
    keys = jax.random.split(key, truncation + 1)
    return tuple(
        0.1 * jax.random.normal(
            level_key,
            batch_shape + (dimension**degree,),
            dtype=jnp.float64,
        )
        for degree, level_key in enumerate(keys)
    )


@pytest.mark.parametrize(
    ("axis", "shape"),
    [
        (0, (6, 2, 2)),
        (-3, (2, 6, 3, 2)),
    ],
)
@pytest.mark.parametrize(
    ("block_size", "accumulate"),
    [(None, True), (None, False), (2, True), (2, False)],
)
def test_forced_interpreter_preserves_seed_axis_and_block_policy(
    axis, shape, block_size, accumulate
):
    core = td.make_core(dims=2, max_trunc=3)
    increments = 0.08 * jax.random.normal(
        jax.random.PRNGKey(330_000 + len(shape) + int(accumulate)),
        shape,
        dtype=jnp.float64,
    )
    moved = jnp.moveaxis(increments, axis, -2)
    starting_point = _arbitrary_dense_seed(
        jax.random.PRNGKey(331_000 + len(shape)),
        batch_shape=moved.shape[:-2],
        dimension=moved.shape[-1],
        truncation=3,
    )
    call = _prepare_free_development_call(
        (increments,),
        trunc=3,
        increment_input=True,
        axis=axis,
        block_size=block_size,
        accumulate=accumulate,
        starting_point=starting_point,
        output_starting_point=True,
        core=core,
    )

    raw = run_ordinary_wordwise(
        call,
        interpret=True,
        tile_words=4,
    )
    actual = _finalize_free_development_call(
        call,
        raw,
        runner_applied_seed=False,
        runner_emitted_starting_point=False,
    )
    expected = _execute_portable_free_development_call(call)

    _assert_tensors_close(actual, expected)
