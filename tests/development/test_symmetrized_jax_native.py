"""End-to-end JAX tests for native partially symmetrized developments."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
from tensordev.core.bigraded.standard import StandardBigradedCore
from tensordev.core.bigraded.symmetrized import algebra as quotient_algebra
from tensordev.core.bigraded.types import BigradedTensor
from tensordev.development import free_development, path_signature


TRUNCATION = (1, 2)


def _assert_tree_allclose(actual, expected, *, atol=4e-6, rtol=4e-6):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(
        expected
    )
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
    ):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            atol=atol,
            rtol=rtol,
        )


def _higher_grade_seed(core):
    layout = core.resolve_layout(TRUNCATION, include_scalar=True)
    blocks = []
    for index, grade in enumerate(layout.grades):
        width = layout.block_width(grade)
        if grade == (0, 0):
            block = jnp.ones((width,), dtype=jnp.float32)
        else:
            block = jnp.linspace(
                -0.025 * (index + 1),
                0.035 * (index + 1),
                width,
                dtype=jnp.float32,
            )
        blocks.append(block)
    seed = BigradedTensor(tuple(blocks), layout.spec)
    assert float(jnp.linalg.norm(seed[1, 2])) > 0.0
    return seed


def _native_workflows(core, starting_point, path):
    signature = path_signature(
        path,
        trunc=TRUNCATION,
        axis=-2,
        accumulate=False,
        parallel=False,
        core=core,
    )
    developed = free_development(
        (path,),
        trunc=TRUNCATION,
        axis=-2,
        accumulate=False,
        parallel=True,
        starting_point=starting_point,
        core=core,
    )
    return signature, developed


def _sum_squares(element):
    return sum(jnp.sum(block**2) for block in element.blocks)


def _install_ordered_route_tripwires(monkeypatch, core):
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "a native quotient development entered an ordered/total route"
        )

    for name in (
        "_lift_partially_symmetrized",
        "_lift_partially_symmetrized_block",
        "tensor_partially_symmetrize",
        "tensor_partially_symmetrize_homogeneous",
        "tensor_to_total",
        "tensor_from_total",
    ):
        monkeypatch.setattr(core, name, forbidden)

    # Module helpers and ordered base-class conversions bypass instance-level
    # guards and must remain unused.
    monkeypatch.setattr(
        quotient_algebra,
        "_lift_partially_symmetrized_block",
        forbidden,
    )
    monkeypatch.setattr(
        quotient_algebra,
        "partially_symmetrize_block",
        forbidden,
    )
    monkeypatch.setattr(StandardBigradedCore, "tensor_to_total", forbidden)
    monkeypatch.setattr(StandardBigradedCore, "tensor_from_total", forbidden)


@pytest.mark.parametrize("coordinates", ("standard", "shear"))
def test_native_quotient_signature_and_free_development_jax_transforms(
    monkeypatch,
    coordinates,
):
    standard = td.bigraded_core(
        dims=(1, 2),
        max_trunc=TRUNCATION,
        default_trunc=TRUNCATION,
        representation="partially_symmetrized",
    )
    core = standard if coordinates == "standard" else td.shear_core(standard)
    starting_point = _higher_grade_seed(core)
    _install_ordered_route_tripwires(monkeypatch, core)

    paths = jnp.asarray(
        [
            [
                [0.00, 0.00, 0.00],
                [0.08, -0.04, 0.03],
                [0.02, 0.05, -0.06],
                [0.11, 0.01, 0.02],
                [0.04, -0.03, 0.09],
            ],
            [
                [0.03, -0.02, 0.01],
                [-0.05, 0.04, 0.08],
                [0.07, 0.02, -0.01],
                [0.01, -0.06, 0.05],
                [0.09, 0.03, 0.00],
            ],
        ],
        dtype=jnp.float32,
    )

    workflow = lambda path: _native_workflows(core, starting_point, path)
    direct = workflow(paths)
    _assert_tree_allclose(jax.jit(workflow)(paths), direct)

    vmapped = jax.vmap(workflow)(paths)
    _assert_tree_allclose(vmapped, direct)

    def scan_workflows(values):
        _, outputs = jax.lax.scan(
            lambda carry, path: (carry, workflow(path)),
            (),
            values,
        )
        return outputs

    scanned = jax.jit(scan_workflows)(paths)
    _assert_tree_allclose(scanned, vmapped)

    for output_index in (0, 1):
        def loss(path):
            return _sum_squares(workflow(path)[output_index])

        eager_gradient = jax.grad(loss)(paths[0])
        compiled_gradient = jax.jit(jax.grad(loss))(paths[0])
        np.testing.assert_allclose(
            np.asarray(compiled_gradient),
            np.asarray(eager_gradient),
            atol=8e-6,
            rtol=8e-6,
        )
        assert bool(jnp.all(jnp.isfinite(compiled_gradient)))
        assert float(jnp.linalg.norm(compiled_gradient)) > 0.0
