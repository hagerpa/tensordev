"""Whole-development JAX and background-core contracts for shear cores."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
from tensordev.core.bigraded.jax import JaxBigraded
from tensordev.core.jax import Jax
from tensordev.core.shear.bigraded import JaxShearBigraded
from tensordev.core.shear.jax import JaxShearTotal


def _assert_tree_allclose(actual, expected, *, atol=6e-5, rtol=6e-5):
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            atol=atol,
            rtol=rtol,
        )


def _sum_squares(element):
    return sum(jnp.sum(block**2) for block in jax.tree_util.tree_leaves(element))


@pytest.mark.parametrize("grading", ("total", "bidegree"))
def test_background_shear_signature_supports_jit_vmap_and_grad_on_axis_zero(
    grading,
):
    if grading == "total":
        trunc = 3
        standard_core = Jax(d=2, max_trunc=trunc, default_trunc=trunc)
        shear_core = JaxShearTotal(
            dims=(1, 1),
            max_trunc=trunc,
            default_trunc=trunc,
        )
    else:
        trunc = (2, 1)
        standard_core = JaxBigraded(
            dims=(1, 1),
            max_trunc=trunc,
            default_trunc=trunc,
        )
        shear_core = JaxShearBigraded(
            dims=(1, 1),
            max_trunc=trunc,
            default_trunc=trunc,
        )

    path = jnp.asarray(
        [
            [[0.0, 0.0], [0.1, -0.1], [-0.2, 0.2]],
            [[0.2, -0.1], [0.3, 0.0], [0.0, 0.4]],
            [[0.4, 0.2], [0.1, 0.2], [0.3, 0.1]],
            [[0.1, 0.3], [-0.2, 0.4], [0.5, -0.1]],
            [[0.3, 0.1], [0.0, 0.5], [0.2, 0.3]],
        ],
        dtype=jnp.float32,
    )
    previous_core, previous_seq_core = td.get_default_core_pair()
    try:
        td.set_default_core(shear_core)

        shear_signature = jax.jit(
            lambda value: td.path_signature(value, axis=0)
        )(path)
        standard_signature = jax.jit(
            lambda value: td.path_signature(
                value,
                axis=0,
                trunc=trunc,
                core=standard_core,
            )
        )(path)
        restored = shear_core.tensor_to_standard_coordinates(
            shear_signature,
            trunc=trunc,
        )
        _assert_tree_allclose(restored, standard_signature)

        vmapped = jax.vmap(
            lambda value: td.path_signature(value, axis=0),
            in_axes=1,
        )(path)
        _assert_tree_allclose(vmapped, shear_signature)

        def shear_loss(value):
            signature = td.path_signature(value, axis=0)
            standard = shear_core.tensor_to_standard_coordinates(
                signature,
                trunc=trunc,
            )
            return _sum_squares(standard)

        def standard_loss(value):
            return _sum_squares(
                td.path_signature(
                    value,
                    axis=0,
                    trunc=trunc,
                    core=standard_core,
                )
            )

        shear_gradient = jax.jit(jax.grad(shear_loss))(path[:, 0])
        standard_gradient = jax.jit(jax.grad(standard_loss))(path[:, 0])
        np.testing.assert_allclose(
            np.asarray(shear_gradient),
            np.asarray(standard_gradient),
            atol=8e-5,
            rtol=8e-5,
        )
    finally:
        td.set_default_core(previous_core, previous_seq_core)
