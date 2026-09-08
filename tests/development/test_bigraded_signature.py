from __future__ import annotations

import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import tensordev as td
from tensordev import Jax, JaxSequentialCore, make_core
from tensordev.development import free_development, path_signature


config.update("jax_enable_x64", True)

TOTAL_CORE = Jax()
TOTAL_SEQ = JaxSequentialCore()


def _random_path(key, *, batch_shape=(), steps=8, dimension=3):
    increment_key, start_key = jr.split(key)
    increments = 0.12 * jr.normal(
        increment_key,
        batch_shape + (steps, dimension),
        dtype=jnp.float64,
    )
    start = 0.25 * jr.normal(
        start_key,
        batch_shape + (dimension,),
        dtype=jnp.float64,
    )
    return jnp.concatenate(
        (
            start[..., None, :],
            start[..., None, :] + jnp.cumsum(increments, axis=-2),
        ),
        axis=-2,
    )


def _assert_tensor_allclose(actual, expected, *, atol=1e-10, rtol=1e-10):
    assert actual.spec == expected.spec
    for grade in actual.grades:
        np.testing.assert_allclose(
            np.asarray(actual[grade]),
            np.asarray(expected[grade]),
            atol=atol,
            rtol=rtol,
            err_msg=f"bidegree {grade}",
        )


def _slice_block_axis(tensor, index):
    return tensor.with_blocks(tuple(block[..., index, :] for block in tensor.blocks))


@pytest.fixture(scope="module")
def core():
    return make_core(
        dims=(1, 2),
        max_trunc=(2, 2),
        default_trunc=(2, 1),
        precompute_shuffle=False,
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_signature_is_the_exact_rectangular_projection(core, parallel):
    active = (2, 1)
    path = _random_path(
        jr.PRNGKey(201 + parallel),
        batch_shape=(2,),
        steps=6,
        dimension=sum(core.dims),
    )

    got = path_signature(
        path,
        trunc=active,
        axis=-2,
        accumulate=False,
        parallel=parallel,
        core=core,
    )
    total = path_signature(
        path,
        trunc=sum(active),
        axis=-2,
        accumulate=False,
        parallel=parallel,
        core=TOTAL_CORE,
        seq_core=TOTAL_SEQ,
    )
    expected = core.tensor_from_total(total, trunc=active)

    _assert_tensor_allclose(got, expected)


def test_free_development_converts_to_projected_total_tensor(core):
    active = (2, 1)
    path = _random_path(
        jr.PRNGKey(207),
        batch_shape=(2,),
        steps=6,
        dimension=sum(core.dims),
    )

    bigraded = free_development(
        (path,),
        trunc=active,
        axis=-2,
        accumulate=False,
        core=core,
    )
    total = free_development(
        (path,),
        trunc=sum(active),
        axis=-2,
        accumulate=False,
        core=TOTAL_CORE,
        seq_core=TOTAL_SEQ,
    )

    converted = core.tensor_to_total(bigraded)
    projected = core.tensor_to_total(core.tensor_from_total(total, trunc=active))
    for degree, (actual, expected) in enumerate(zip(converted, projected)):
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"total degree {degree}",
        )


def test_streaming_parallel_and_increment_inputs_agree(core):
    active = (2, 1)
    path = _random_path(
        jr.PRNGKey(202),
        batch_shape=(2,),
        steps=8,
        dimension=sum(core.dims),
    )
    increments = jnp.diff(path, axis=-2)

    streaming = path_signature(
        path,
        trunc=active,
        axis=-2,
        accumulate=False,
        parallel=False,
        core=core,
    )
    parallel = path_signature(
        path,
        trunc=active,
        axis=-2,
        accumulate=False,
        parallel=True,
        core=core,
    )
    from_increments = path_signature(
        increments,
        trunc=active,
        axis=-2,
        increment_input=True,
        accumulate=False,
        core=core,
    )

    _assert_tensor_allclose(parallel, streaming)
    _assert_tensor_allclose(from_increments, streaming)


@pytest.mark.parametrize("accumulate_in_tree", [False, True])
def test_blocked_accumulation_matches_total_prefix_projection(core, accumulate_in_tree):
    active = (2, 1)
    path = _random_path(
        jr.PRNGKey(203 + accumulate_in_tree),
        batch_shape=(2,),
        steps=8,
        dimension=sum(core.dims),
    )
    kwargs = dict(
        axis=-2,
        block_size=2,
        accumulate=True,
        accumulate_in_tree=accumulate_in_tree,
        output_starting_point=True,
    )

    got = path_signature(path, trunc=active, core=core, **kwargs)
    total = path_signature(
        path,
        trunc=sum(active),
        core=TOTAL_CORE,
        seq_core=TOTAL_SEQ,
        **kwargs,
    )
    expected = core.tensor_from_total(total, trunc=active)

    _assert_tensor_allclose(got, expected)

    unit = core.tensor_exponential(
        (jnp.zeros(path.shape[:-2] + (sum(core.dims),), dtype=path.dtype),),
        trunc=active,
        output_zero_level=True,
    )
    _assert_tensor_allclose(_slice_block_axis(got, 0), unit)

    terminal = path_signature(
        path,
        trunc=active,
        axis=-2,
        accumulate=False,
        core=core,
    )
    _assert_tensor_allclose(_slice_block_axis(got, -1), terminal)


def test_starting_point_is_left_multiplied_and_can_be_emitted(core):
    active = (2, 1)
    path = _random_path(
        jr.PRNGKey(205),
        batch_shape=(2,),
        steps=6,
        dimension=sum(core.dims),
    )
    z = 0.1 * jr.normal(
        jr.PRNGKey(206),
        (2, sum(core.dims)),
        dtype=jnp.float64,
    )
    starting_point = core.tensor_exponential(
        (z,), trunc=active, output_zero_level=True
    )

    plain = path_signature(
        path,
        trunc=active,
        axis=-2,
        accumulate=False,
        core=core,
    )
    seeded = path_signature(
        path,
        trunc=active,
        axis=-2,
        accumulate=False,
        starting_point=starting_point,
        core=core,
    )
    expected = core.tensor_product(starting_point, plain, trunc=active)
    _assert_tensor_allclose(seeded, expected)

    prefixes = path_signature(
        path,
        trunc=active,
        axis=-2,
        block_size=2,
        accumulate=True,
        starting_point=starting_point,
        output_starting_point=True,
        core=core,
    )
    _assert_tensor_allclose(_slice_block_axis(prefixes, 0), starting_point)
    _assert_tensor_allclose(_slice_block_axis(prefixes, -1), expected)


def test_nonaccumulating_bigraded_blocks_seed_each_block_exactly_once(core):
    active = (2, 1)
    steps, block_size = 8, 2
    path = _random_path(
        jr.PRNGKey(208),
        batch_shape=(2,),
        steps=steps,
        dimension=sum(core.dims),
    )
    seed_increment = 0.1 * jr.normal(
        jr.PRNGKey(209),
        (2, sum(core.dims)),
        dtype=jnp.float64,
    )
    starting_point = core.tensor_exponential(
        (seed_increment,), trunc=active, output_zero_level=True
    )

    plain_blocks = path_signature(
        path,
        trunc=active,
        axis=-2,
        block_size=block_size,
        accumulate=False,
        core=core,
    )
    seeded_blocks = path_signature(
        path,
        trunc=active,
        axis=-2,
        block_size=block_size,
        accumulate=False,
        starting_point=starting_point,
        core=core,
    )
    emitted = path_signature(
        path,
        trunc=active,
        axis=-2,
        block_size=block_size,
        accumulate=False,
        starting_point=starting_point,
        output_starting_point=True,
        core=core,
    )

    for block_index in range(steps // block_size):
        expected = core.tensor_product(
            starting_point,
            _slice_block_axis(plain_blocks, block_index),
            trunc=active,
        )
        _assert_tensor_allclose(
            _slice_block_axis(seeded_blocks, block_index), expected
        )
        _assert_tensor_allclose(
            _slice_block_axis(emitted, block_index + 1), expected
        )

    _assert_tensor_allclose(_slice_block_axis(emitted, 0), starting_point)


def test_signature_class_binds_a_bidegree_core(core):
    path = _random_path(
        jr.PRNGKey(207),
        steps=6,
        dimension=sum(core.dims),
    )
    signature = td.Signature(trunc=(1, 2), core=core)

    got = signature(path, axis=-2, accumulate=False)
    expected = path_signature(
        path,
        trunc=(1, 2),
        axis=-2,
        accumulate=False,
        core=core,
    )

    assert signature.trunc == (1, 2)
    _assert_tensor_allclose(got, expected)


def test_package_default_drives_operations_and_signatures_then_resets():
    initial_core, initial_seq = td.get_default_core_pair()
    core = make_core(
        dims=(1, 1),
        max_trunc=(2, 2),
        default_trunc=(1, 1),
        precompute_shuffle=False,
    )
    path = _random_path(
        jr.PRNGKey(208),
        steps=4,
        dimension=sum(core.dims),
    )

    try:
        td.set_default_core(core)

        implicit = td.path_signature(path, axis=-2, accumulate=False)
        explicit = td.path_signature(
            path,
            trunc=(1, 1),
            axis=-2,
            accumulate=False,
            core=core,
        )
        bound = td.Signature()(path, axis=-2, accumulate=False)
        module_product = td.tensor_product(implicit, implicit)
        explicit_product = core.tensor_product(
            explicit, explicit, trunc=core.default_truncation
        )

        assert td.get_default_core() is core
        assert td.tensor_product.__self__ is core
        _assert_tensor_allclose(implicit, explicit)
        _assert_tensor_allclose(bound, explicit)
        _assert_tensor_allclose(module_product, explicit_product)
    finally:
        td.reset_default_core()

    assert td.get_default_core() is initial_core
    assert td.get_default_seq_core() is initial_seq
    assert td.tensor_product.__self__ is initial_core
