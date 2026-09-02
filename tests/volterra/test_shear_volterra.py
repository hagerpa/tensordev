"""Volterra integration tests for ordered shear tensor cores."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
from tensordev.core.bigraded.jax import JaxBigraded
from tensordev.core.jax import Jax
from tensordev.core.shear.bigraded import JaxShearBigraded
from tensordev.core.shear.jax import JaxShearTotal
from tensordev.volterra import FractionalKernel, vsig


_DX = jnp.array(
    [[0.20, -0.10], [-0.05, 0.25]],
    dtype=jnp.float64,
)
_PATH = jnp.concatenate(
    (jnp.zeros((1, 2), dtype=_DX.dtype), jnp.cumsum(_DX, axis=0)),
    axis=0,
)


def _q1_kernel():
    return FractionalKernel(
        beta=jnp.array([0.7], dtype=jnp.float64),
        A=jnp.eye(2, dtype=jnp.float64)[None, :, :],
    )


def _q2_kernel():
    return FractionalKernel(
        beta=jnp.array([0.65, 1.2], dtype=jnp.float64),
        A=jnp.array(
            [
                [[1.0, 0.0], [0.2, 0.7]],
                [[-0.3, 0.4], [0.8, 0.1]],
            ],
            dtype=jnp.float64,
        ),
    )


def _assert_dense_allclose(got, expected, *, atol=2e-10, rtol=2e-10):
    assert len(got) == len(expected)
    for grade, (actual, reference) in enumerate(zip(got, expected)):
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(reference),
            atol=atol,
            rtol=rtol,
            err_msg=f"total degree {grade}",
        )


def _assert_bigraded_allclose(got, expected, *, atol=2e-10, rtol=2e-10):
    assert got.spec == expected.spec
    for grade in got.grades:
        np.testing.assert_allclose(
            np.asarray(got[grade]),
            np.asarray(expected[grade]),
            atol=atol,
            rtol=rtol,
            err_msg=f"bidegree {grade}",
        )


def _assert_tree_allclose(got, expected, *, atol=6e-5, rtol=6e-5):
    got_leaves = jax.tree_util.tree_leaves(got)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(got_leaves) == len(expected_leaves)
    for actual, reference in zip(got_leaves, expected_leaves):
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(reference),
            atol=atol,
            rtol=rtol,
        )


def _tree_sum_squares(element):
    return sum(
        jnp.sum(block**2) for block in jax.tree_util.tree_leaves(element)
    )


@pytest.mark.parametrize("scheme", ("quadratic", "fft", "adams"))
def test_total_shear_q1_volterra_needs_no_shuffle_precomputation(scheme):
    trunc = 3
    standard_core = Jax(d=2, max_trunc=trunc)
    shear_core = JaxShearTotal(
        dims=(1, 1),
        max_trunc=trunc,
        precompute_shuffle=False,
    )

    got = vsig(
        _PATH,
        kernel=_q1_kernel(),
        trunc=trunc,
        dt=0.4,
        scheme=scheme,
        core=shear_core,
    )
    standard = vsig(
        _PATH,
        kernel=_q1_kernel(),
        trunc=trunc,
        dt=0.4,
        scheme=scheme,
        core=standard_core,
    )
    expected = shear_core.tensor_from_standard_coordinates(
        standard,
        trunc=trunc,
    )

    assert not shear_core.supports("shuffle")
    _assert_dense_allclose(got, expected)


@pytest.mark.parametrize("scheme", ("quadratic", "fft"))
def test_total_shear_multicomponent_volterra_uses_generator_scope(scheme):
    trunc = 2
    standard_core = Jax(d=2, max_trunc=trunc, precompute_shuffle=True)
    shear_core = JaxShearTotal(
        dims=(1, 1),
        max_trunc=trunc,
        precompute_shuffle="generator",
    )

    got = vsig(
        _PATH,
        kernel=_q2_kernel(),
        trunc=trunc,
        dt=0.4,
        scheme=scheme,
        core=shear_core,
    )
    standard = vsig(
        _PATH,
        kernel=_q2_kernel(),
        trunc=trunc,
        dt=0.4,
        scheme=scheme,
        core=standard_core,
    )
    expected = shear_core.tensor_from_standard_coordinates(
        standard,
        trunc=trunc,
    )

    assert shear_core.supports("shuffle")
    assert not shear_core.supports("shuffle_product")
    _assert_dense_allclose(got, expected)


@pytest.mark.parametrize(
    ("kernel_factory", "shuffle_scope"),
    ((_q1_kernel, False), (_q2_kernel, "generator")),
)
def test_bidegree_shear_volterra_matches_forward_rectangular_projection(
    kernel_factory,
    shuffle_scope,
):
    active = (1, 1)
    total_core = Jax(
        d=2,
        max_trunc=sum(active),
        precompute_shuffle=(shuffle_scope == "generator"),
    )
    standard_core = JaxBigraded(
        dims=(1, 1),
        max_trunc=active,
        precompute_shuffle=shuffle_scope,
    )
    shear_core = JaxShearBigraded(
        dims=(1, 1),
        max_trunc=active,
        precompute_shuffle=shuffle_scope,
    )
    kernel = kernel_factory()

    got = vsig(
        _PATH,
        kernel=kernel,
        trunc=active,
        dt=0.4,
        scheme="quadratic",
        core=shear_core,
    )
    standard = vsig(
        _PATH,
        kernel=kernel,
        trunc=active,
        dt=0.4,
        scheme="quadratic",
        core=standard_core,
    )
    total = vsig(
        _PATH,
        kernel=kernel,
        trunc=sum(active),
        dt=0.4,
        scheme="quadratic",
        core=total_core,
    )
    projected = standard_core.tensor_from_total(total, trunc=active)
    expected = shear_core.tensor_from_standard_coordinates(
        standard,
        trunc=active,
    )

    _assert_bigraded_allclose(standard, projected)
    _assert_bigraded_allclose(got, expected)


@pytest.mark.parametrize("grading", ("total", "bidegree"))
def test_background_shear_volterra_protocol_is_jittable_vmappable_and_differentiable(
    grading,
):
    if grading == "total":
        trunc = 2
        standard_core = Jax(
            d=2,
            max_trunc=trunc,
            default_trunc=trunc,
            precompute_shuffle=True,
        )
        shear_core = JaxShearTotal(
            dims=(1, 1),
            max_trunc=trunc,
            default_trunc=trunc,
            precompute_shuffle=True,
        )
    else:
        trunc = (1, 1)
        standard_core = JaxBigraded(
            dims=(1, 1),
            max_trunc=trunc,
            default_trunc=trunc,
            precompute_shuffle=True,
        )
        shear_core = JaxShearBigraded(
            dims=(1, 1),
            max_trunc=trunc,
            default_trunc=trunc,
            precompute_shuffle=True,
        )

    path = jnp.asarray(
        [
            [[0.0, 0.0], [0.1, -0.1]],
            [[0.2, -0.1], [0.3, 0.0]],
            [[0.4, 0.2], [0.1, 0.2]],
            [[0.1, 0.3], [-0.2, 0.4]],
            [[0.3, 0.1], [0.0, 0.5]],
        ],
        dtype=jnp.float64,
    )
    kernel = _q2_kernel()
    standard_start = standard_core.tensor_exponential(
        (jnp.zeros((2, 2), dtype=path.dtype),),
        trunc=trunc,
    )
    shear_start = shear_core.tensor_from_standard_coordinates(
        standard_start,
        trunc=trunc,
    )

    previous_core, previous_seq_core = td.get_default_core_pair()
    try:
        td.set_default_core(shear_core)
        options = dict(
            kernel=kernel,
            dt=0.25,
            axis=0,
            scheme="fft",
        )
        got_blocks = vsig(
            path,
            block_size=2,
            accumulate=True,
            starting_point=shear_start,
            output_starting_point=True,
            **options,
        )
        standard_blocks = vsig(
            path,
            trunc=trunc,
            block_size=2,
            accumulate=True,
            starting_point=standard_start,
            output_starting_point=True,
            core=standard_core,
            **options,
        )
        restored_blocks = shear_core.tensor_to_standard_coordinates(
            got_blocks,
            trunc=trunc,
        )
        _assert_tree_allclose(restored_blocks, standard_blocks)

        shear_solve = lambda value: vsig(value, **options)
        compiled = jax.jit(shear_solve)(path)
        vmapped = jax.vmap(shear_solve, in_axes=1)(path)
        _assert_tree_allclose(compiled, vmapped)

        def shear_loss(value):
            result = shear_core.tensor_to_standard_coordinates(
                shear_solve(value),
                trunc=trunc,
            )
            return _tree_sum_squares(result)

        def standard_loss(value):
            return _tree_sum_squares(
                vsig(
                    value,
                    kernel=kernel,
                    trunc=trunc,
                    dt=0.25,
                    axis=0,
                    scheme="fft",
                    core=standard_core,
                )
            )

        shear_gradient = jax.jit(jax.grad(shear_loss))(path[:, 0])
        standard_gradient = jax.jit(jax.grad(standard_loss))(path[:, 0])
        np.testing.assert_allclose(
            np.asarray(shear_gradient),
            np.asarray(standard_gradient),
            atol=2e-9,
            rtol=2e-9,
        )
    finally:
        td.set_default_core(previous_core, previous_seq_core)
