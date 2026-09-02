from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
from tensordev import Jax, bigraded_core
from tensordev.volterra import (
    FractionalKernel,
    VolterraSignature,
    pc_iteration,
    quadratic_iteration,
    vsig,
)


_TOTAL_CORE = Jax()
_ACTIVE = (2, 1)
_DX = jnp.array(
    [
        [0.20, -0.10],
        [-0.05, 0.25],
    ],
    dtype=jnp.float64,
)
_X = jnp.concatenate(
    (jnp.zeros((1, 2), dtype=_DX.dtype), jnp.cumsum(_DX, axis=0)),
    axis=0,
)


@pytest.fixture(scope="module")
def q1_core():
    return bigraded_core(
        dims=(1, 1),
        max_trunc=_ACTIVE,
        default_trunc=(1, 1),
        precompute_shuffle=False,
    )


@pytest.fixture(scope="module")
def q1_kernel():
    return FractionalKernel(
        beta=jnp.array([0.7], dtype=jnp.float64),
        A=jnp.eye(2, dtype=jnp.float64)[None, :, :],
    )


def _assert_bigraded_projection(
        got,
        total,
        core,
        trunc,
        *,
        atol=2e-10,
        rtol=2e-10,
):
    expected = core.tensor_from_total(total, trunc=trunc)
    assert got.spec == expected.spec
    for grade in got.grades:
        np.testing.assert_allclose(
            np.asarray(got[grade]),
            np.asarray(expected[grade]),
            atol=atol,
            rtol=rtol,
            err_msg=f"bidegree {grade}",
        )


@pytest.mark.parametrize(
    ("order", "return_trajectory"),
    ((0, False), (1, True), (2, False)),
)
def test_quadratic_q1_is_exact_rectangular_projection(
        q1_core,
        q1_kernel,
        order,
        return_trajectory,
):
    total = quadratic_iteration(
        _DX,
        kernel=q1_kernel,
        trunc=sum(_ACTIVE),
        dt=jnp.array([0.3, 0.5], dtype=jnp.float64),
        order=order,
        return_trajectory=return_trajectory,
        core=_TOTAL_CORE,
    )
    got = quadratic_iteration(
        _DX,
        kernel=q1_kernel,
        trunc=_ACTIVE,
        dt=jnp.array([0.3, 0.5], dtype=jnp.float64),
        order=order,
        return_trajectory=return_trajectory,
        core=q1_core,
    )

    _assert_bigraded_projection(got, total, q1_core, _ACTIVE)


@pytest.mark.parametrize(
    ("order", "return_trajectory"),
    ((0, False), (1, True)),
)
def test_adams_q1_is_exact_rectangular_projection(
        q1_core,
        q1_kernel,
        order,
        return_trajectory,
):
    total = pc_iteration(
        _DX,
        kernel=q1_kernel,
        trunc=sum(_ACTIVE),
        dt=0.25,
        order=order,
        return_trajectory=return_trajectory,
        core=_TOTAL_CORE,
    )
    got = pc_iteration(
        _DX,
        kernel=q1_kernel,
        trunc=_ACTIVE,
        dt=0.25,
        order=order,
        return_trajectory=return_trajectory,
        core=q1_core,
    )

    _assert_bigraded_projection(
        got,
        total,
        q1_core,
        _ACTIVE,
        atol=5e-10,
        rtol=5e-10,
    )


def test_adams_rejects_nonuniform_grid(q1_kernel):
    with pytest.raises(ValueError, match="requires a scalar dt"):
        pc_iteration(
            _DX,
            kernel=q1_kernel,
            trunc=2,
            dt=jnp.array([0.2, 0.9], dtype=jnp.float64),
            core=_TOTAL_CORE,
        )
    with pytest.raises(ValueError, match="requires a scalar dt"):
        vsig(
            _X,
            kernel=q1_kernel,
            trunc=2,
            dt=jnp.array([0.2, 0.9], dtype=jnp.float64),
            scheme="adams",
            core=_TOTAL_CORE,
        )


def test_adams_validates_step_axis_and_path_width(q1_kernel):
    with pytest.raises(ValueError, match="not the trailing path dimension"):
        pc_iteration(
            _DX,
            kernel=q1_kernel,
            trunc=2,
            axis=-1,
            core=_TOTAL_CORE,
        )
    with pytest.raises(ValueError, match="trailing dimension must be 2"):
        pc_iteration(
            jnp.zeros((2, 3), dtype=jnp.float64),
            kernel=q1_kernel,
            trunc=2,
            core=_TOTAL_CORE,
        )


def test_high_level_adams_routes_the_native_core(q1_kernel):
    active = (1, 1)
    core = bigraded_core(dims=(1, 1), max_trunc=active)
    total = vsig(
        _X,
        kernel=q1_kernel,
        trunc=sum(active),
        dt=0.25,
        scheme="adams",
        core=_TOTAL_CORE,
    )
    got = vsig(
        _X,
        kernel=q1_kernel,
        trunc=active,
        dt=0.25,
        scheme="adams",
        core=core,
    )
    _assert_bigraded_projection(got, total, core, active)


@pytest.mark.parametrize("precompute_shuffle", [True, "generator"])
def test_quadratic_q2_projection_uses_native_shuffle_plans(
    precompute_shuffle,
):
    active = (1, 1)
    core = bigraded_core(
        dims=(1, 1),
        max_trunc=active,
        precompute_shuffle=precompute_shuffle,
    )
    kernel = FractionalKernel(
        beta=jnp.array([0.65, 1.2], dtype=jnp.float64),
        A=jnp.array(
            [
                [[1.0, 0.0], [0.2, 0.7]],
                [[-0.3, 0.4], [0.8, 0.1]],
            ],
            dtype=jnp.float64,
        ),
    )

    total = quadratic_iteration(
        _DX,
        kernel=kernel,
        trunc=sum(active),
        dt=0.4,
        order=0,
        core=_TOTAL_CORE,
    )
    got = quadratic_iteration(
        _DX,
        kernel=kernel,
        trunc=active,
        dt=0.4,
        order=0,
        core=core,
    )

    _assert_bigraded_projection(got, total, core, active)


def test_quadratic_q2_reports_missing_native_shuffle_plans():
    active = (1, 1)
    core = bigraded_core(
        dims=(1, 1),
        max_trunc=active,
        precompute_shuffle=False,
    )
    kernel = FractionalKernel(
        beta=jnp.array([0.65, 1.2], dtype=jnp.float64),
        A=jnp.ones((2, 2, 2), dtype=jnp.float64),
    )

    with pytest.raises(RuntimeError, match="precompute_shuffle='generator'"):
        quadratic_iteration(
            _DX[:1],
            kernel=kernel,
            trunc=active,
            dt=0.4,
            order=0,
            core=core,
        )


def test_vsig_uses_explicit_and_background_bigraded_core(q1_core, q1_kernel):
    active = q1_core.default_truncation
    total = vsig(
        _X,
        kernel=q1_kernel,
        trunc=sum(active),
        dt=0.4,
        scheme="quadratic",
        core=_TOTAL_CORE,
    )
    explicit = vsig(
        _X,
        kernel=q1_kernel,
        trunc=active,
        dt=0.4,
        scheme="quadratic",
        core=q1_core,
    )

    previous_core, previous_seq_core = td.get_default_core_pair()
    try:
        td.set_default_core(q1_core)
        implicit = vsig(
            _X,
            kernel=q1_kernel,
            dt=0.4,
            scheme="quadratic",
        )
    finally:
        td.set_default_core(previous_core, previous_seq_core)

    _assert_bigraded_projection(explicit, total, q1_core, active)
    _assert_bigraded_projection(implicit, total, q1_core, active)


def test_multicomponent_vsig_uses_generator_only_shuffle_plans():
    active = (1, 1)
    core = bigraded_core(
        dims=(1, 1),
        max_trunc=active,
        precompute_shuffle="generator",
    )
    kernel = FractionalKernel(
        beta=jnp.array([0.65, 1.2], dtype=jnp.float64),
        A=jnp.array(
            [
                [[1.0, 0.0], [0.2, 0.7]],
                [[-0.3, 0.4], [0.8, 0.1]],
            ],
            dtype=jnp.float64,
        ),
    )

    total = vsig(
        _X,
        kernel=kernel,
        trunc=sum(active),
        dt=0.4,
        scheme="quadratic",
        core=_TOTAL_CORE,
    )
    got = vsig(
        _X,
        kernel=kernel,
        trunc=active,
        dt=0.4,
        scheme="quadratic",
        core=core,
    )

    _assert_bigraded_projection(got, total, core, active)


def test_vsig_independent_blocking_and_starting_point_are_native_pytrees(
        q1_core,
        q1_kernel,
):
    active = (1, 1)
    dX = jnp.array(
        [
            [0.10, -0.20],
            [0.30, 0.05],
            [-0.15, 0.10],
            [0.05, 0.25],
        ],
        dtype=jnp.float64,
    )
    X = jnp.concatenate(
        (jnp.zeros((1, 2), dtype=dX.dtype), jnp.cumsum(dX, axis=0)),
        axis=0,
    )
    total_start = (
        jnp.array([2.0], dtype=jnp.float64),
        jnp.array([0.1, -0.2], dtype=jnp.float64),
        jnp.array([0.3, -0.1, 0.2, 0.4], dtype=jnp.float64),
    )
    native_start = q1_core.tensor_from_total(total_start, trunc=active)

    total = vsig(
        X,
        kernel=q1_kernel,
        trunc=sum(active),
        dt=0.5,
        block_size=2,
        accumulate=False,
        starting_point=total_start,
        output_starting_point=True,
        scheme="quadratic",
        core=_TOTAL_CORE,
    )
    got = vsig(
        X,
        kernel=q1_kernel,
        trunc=active,
        dt=0.5,
        block_size=2,
        accumulate=False,
        starting_point=native_start,
        output_starting_point=True,
        scheme="quadratic",
        core=q1_core,
    )

    _assert_bigraded_projection(got, total, q1_core, active)
    for grade in got.grades:
        np.testing.assert_allclose(got[grade][0], native_start[grade])


def test_volterra_signature_binds_core_and_uses_its_default_truncation(
        q1_core,
        q1_kernel,
):
    signature = VolterraSignature(kernel=q1_kernel, core=q1_core)

    got = signature.vsig(_X, dt=0.4, scheme="quadratic")
    expected = vsig(
        _X,
        kernel=q1_kernel,
        trunc=q1_core.default_truncation,
        dt=0.4,
        scheme="quadratic",
        core=q1_core,
    )

    assert signature.core is q1_core
    assert signature.trunc == q1_core.default_truncation
    assert signature.seq_core is q1_core.make_sequential_core()
    assert got.spec == expected.spec
    for grade in got.grades:
        np.testing.assert_allclose(got[grade], expected[grade])


def test_vsig_bigraded_core_is_jittable(q1_core, q1_kernel):
    active = (1, 1)

    compiled = jax.jit(
        lambda path: vsig(
            path,
            kernel=q1_kernel,
            trunc=active,
            dt=0.4,
            scheme="quadratic",
            core=q1_core,
        )
    )
    got = compiled(_X[:2])
    expected = vsig(
        _X[:2],
        kernel=q1_kernel,
        trunc=active,
        dt=0.4,
        scheme="quadratic",
        core=q1_core,
    )

    assert got.spec == expected.spec
    for grade in got.grades:
        np.testing.assert_allclose(got[grade], expected[grade])


def test_vsig_bigraded_core_supports_vmap_and_grad(q1_core, q1_kernel):
    active = (1, 1)

    def native(path):
        return vsig(
            path,
            kernel=q1_kernel,
            trunc=active,
            dt=0.4,
            scheme="quadratic",
            core=q1_core,
        )

    def projected_total(path):
        total = vsig(
            path,
            kernel=q1_kernel,
            trunc=sum(active),
            dt=0.4,
            scheme="quadratic",
            core=_TOTAL_CORE,
        )
        return q1_core.tensor_from_total(total, trunc=active)

    paths = jnp.stack((_X[:2], -0.5 * _X[:2]))
    got = jax.vmap(native)(paths)
    expected = jax.vmap(projected_total)(paths)

    assert got.spec == expected.spec
    for grade in got.grades:
        np.testing.assert_allclose(got[grade], expected[grade])

    def sum_blocks(element):
        return sum(jnp.sum(element[grade]) for grade in element.grades)

    got_grad = jax.grad(lambda path: sum_blocks(native(path)))(_X[:2])
    expected_grad = jax.grad(
        lambda path: sum_blocks(projected_total(path))
    )(_X[:2])
    np.testing.assert_allclose(got_grad, expected_grad)
