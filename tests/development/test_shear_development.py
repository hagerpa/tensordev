"""Development-level covariance tests for ordered shear coordinates."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tensordev.core.bigraded.jax import JaxBigraded
from tensordev.core.jax import Jax, JaxSequentialCore
from tensordev.core.shear.algebra import ShearCoordinateCore
from tensordev.core.shear.bigraded import JaxShearBigraded
from tensordev.core.shear.jax import JaxShearTotal
from tensordev.development import free_development, path_signature


_SEQ = JaxSequentialCore()


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


def test_total_shear_path_signature_is_forward_standard_signature():
    standard_core = Jax(d=3, max_trunc=3)
    shear_core = JaxShearTotal(dims=(1, 2), max_trunc=3)
    path = jnp.array(
        [
            [[0.0, 0.1, -0.2], [0.2, 0.0, -0.1], [0.1, 0.3, 0.2]],
            [[-0.1, 0.2, 0.0], [0.0, -0.2, 0.3], [0.4, 0.1, 0.1]],
        ],
        dtype=jnp.float64,
    )

    got = path_signature(
        path,
        trunc=3,
        axis=-2,
        accumulate=False,
        core=shear_core,
        seq_core=_SEQ,
    )
    standard = path_signature(
        path,
        trunc=3,
        axis=-2,
        accumulate=False,
        core=standard_core,
        seq_core=_SEQ,
    )
    expected = shear_core.tensor_from_standard_coordinates(standard, trunc=3)

    _assert_dense_allclose(got, expected)


def test_total_shear_higher_level_free_development_is_covariant():
    standard_core = Jax(d=2, max_trunc=3)
    shear_core = JaxShearTotal(dims=(1, 1), max_trunc=3)
    standard_path = (
        jnp.array(
            [[0.0, 0.0], [0.2, -0.1], [0.1, 0.3], [-0.2, 0.4]],
            dtype=jnp.float64,
        ),
        jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.1, -0.04, 0.02, 0.06],
                [0.12, -0.01, 0.04, 0.1],
                [0.05, 0.02, 0.09, 0.03],
            ],
            dtype=jnp.float64,
        ),
    )
    shear_path = shear_core.tensor_from_standard_coordinates(
        standard_path,
        trunc=2,
        first_on=True,
    )

    got = free_development(
        shear_path,
        trunc=3,
        axis=-2,
        accumulate=False,
        core=shear_core,
        seq_core=_SEQ,
    )
    standard = free_development(
        standard_path,
        trunc=3,
        axis=-2,
        accumulate=False,
        core=standard_core,
        seq_core=_SEQ,
    )
    expected = shear_core.tensor_from_standard_coordinates(standard, trunc=3)

    _assert_dense_allclose(got, expected)


def test_bidegree_shear_path_signature_is_forward_standard_signature():
    active = (2, 1)
    standard_core = JaxBigraded(dims=(1, 2), max_trunc=active)
    shear_core = JaxShearBigraded(dims=(1, 2), max_trunc=active)
    path = jnp.array(
        [
            [0.0, 0.1, -0.2],
            [0.2, 0.0, -0.1],
            [0.1, 0.3, 0.2],
            [0.4, -0.1, 0.3],
        ],
        dtype=jnp.float64,
    )

    got = path_signature(
        path,
        trunc=active,
        axis=-2,
        accumulate=False,
        core=shear_core,
        seq_core=_SEQ,
    )
    standard = path_signature(
        path,
        trunc=active,
        axis=-2,
        accumulate=False,
        core=standard_core,
        seq_core=_SEQ,
    )
    expected = shear_core.tensor_from_standard_coordinates(
        standard,
        trunc=active,
    )

    _assert_bigraded_allclose(got, expected)


@pytest.fixture(scope="module", params=("total", "bidegree"))
def shear_signature_core_pair(request):
    if request.param == "total":
        return (
            Jax(d=2, max_trunc=3),
            JaxShearTotal(dims=(1, 1), max_trunc=3),
            3,
        )
    return (
        JaxBigraded(dims=(1, 1), max_trunc=(2, 1)),
        JaxShearBigraded(dims=(1, 1), max_trunc=(2, 1)),
        (2, 1),
    )


@pytest.mark.parametrize(
    ("parallel", "accumulate", "block_size", "accumulate_in_tree"),
    (
        (False, False, None, False),
        (True, True, None, False),
        (False, False, 2, False),
        (True, True, 2, True),
    ),
)
def test_shear_signature_sequence_modes_are_covariant(
    shear_signature_core_pair,
    parallel,
    accumulate,
    block_size,
    accumulate_in_tree,
):
    standard_core, shear_core, trunc = shear_signature_core_pair
    path = jnp.array(
        [
            [0.0, 0.1],
            [0.2, 0.0],
            [0.1, 0.3],
            [0.4, -0.1],
            [0.25, 0.2],
        ],
        dtype=jnp.float64,
    )
    kwargs = dict(
        trunc=trunc,
        axis=-2,
        block_size=block_size,
        accumulate=accumulate,
        accumulate_in_tree=accumulate_in_tree,
        parallel=parallel,
        seq_core=_SEQ,
    )

    got = path_signature(path, core=shear_core, **kwargs)
    standard = path_signature(path, core=standard_core, **kwargs)
    expected = shear_core.tensor_from_standard_coordinates(
        standard,
        trunc=trunc,
    )

    if shear_core.grading == "total_degree":
        _assert_dense_allclose(got, expected)
    else:
        _assert_bigraded_allclose(got, expected)


@pytest.mark.parametrize("grading", ("total", "bidegree"))
def test_first_level_horner_uses_native_generator_plans_without_transport(
    monkeypatch,
    grading,
):
    if grading == "total":
        core = JaxShearTotal(dims=(1, 1), max_trunc=3)
        trunc = 3
        left = (
            jnp.ones((1,), dtype=jnp.float64),
            jnp.arange(2, dtype=jnp.float64),
            jnp.arange(4, dtype=jnp.float64),
            jnp.arange(8, dtype=jnp.float64),
        )
    else:
        core = JaxShearBigraded(dims=(1, 1), max_trunc=(2, 1))
        trunc = (2, 1)
        layout = core.resolve_layout(trunc, include_scalar=True)
        left = core._constant_element_for_layout(
            layout,
            batch_shape=(),
            dtype=jnp.float64,
            alphabet_dim=2,
            scalar=1.0,
        )

    native_action = core._right_multiply_generator_output_block
    calls = 0

    def counted_native_action(*args, **kwargs):
        nonlocal calls
        calls += 1
        return native_action(*args, **kwargs)

    def forbidden_transport(*_args, **_kwargs):
        raise AssertionError("first-level Horner used coordinate transport")

    monkeypatch.setattr(
        core,
        "_right_multiply_generator_output_block",
        counted_native_action,
    )
    for name in (
        "_coordinate_forward",
        "_coordinate_inverse",
        "_coordinate_forward_transpose",
        "_coordinate_inverse_transpose",
        "_coordinate_forward_block",
        "_coordinate_inverse_block",
        "_coordinate_forward_transpose_block",
        "_coordinate_inverse_transpose_block",
    ):
        monkeypatch.setattr(core, name, forbidden_transport)

    result = ShearCoordinateCore.tensor_fmexp(
        core,
        left,
        (jnp.array([0.2, -0.1], dtype=jnp.float64),),
        trunc=trunc,
    )

    assert calls > 0
    assert len(result) > 1
