from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from tensordev import Jax


CORE = Jax()


def _list_element():
    return [
        jnp.arange(2, dtype=jnp.float64).reshape(2, 1),
        jnp.arange(4, dtype=jnp.float64).reshape(2, 2),
    ]


def test_list_component_operations_return_tuple_output():
    element = _list_element()

    scaled = CORE.tensor_scalar_multiply(element, 2.0)
    moved = CORE.tensor_moveaxis(element, source=0, destination=0)
    sliced = CORE.tensor_slice(element)[:1]
    stacked = CORE.tensor_stack([element, element], axis=0)

    for result in (scaled, moved, sliced, stacked):
        assert isinstance(result, tuple)

    for actual, expected in zip(scaled, element):
        np.testing.assert_array_equal(np.asarray(actual), 2.0 * np.asarray(expected))
    for actual, expected in zip(moved, element):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    for actual, expected in zip(sliced, element):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected[:1]))
    for actual, expected in zip(stacked, element):
        np.testing.assert_array_equal(
            np.asarray(actual),
            np.stack([np.asarray(expected), np.asarray(expected)], axis=0),
        )


def test_empty_list_scalar_multiply_returns_canonical_empty_tuple():
    assert CORE.tensor_scalar_multiply([], 2.0) == tuple()
