from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
from tensordev import Jax, bigraded_core
from tensordev.volterra import FractionalKernel
from tensordev.volterra.algebra import resolve_volterra_algebra
from tensordev.volterra.iteration_fft import (
    _sum_product_splits,
    fft_iteration,
    precompute_lag_tables,
)


_TOTAL_CORE = Jax()
_DX = jnp.array(
    [[0.20, -0.10], [-0.05, 0.25], [0.15, 0.05]],
    dtype=jnp.float64,
)


def _kernel(q: int) -> FractionalKernel:
    beta = jnp.array([0.7] if q == 1 else [0.7, 1.1], dtype=jnp.float64)
    matrix = jnp.eye(2, dtype=jnp.float64)
    A = matrix[None, ...] if q == 1 else jnp.stack((matrix, matrix))
    return FractionalKernel(beta=beta, A=A)


def _assert_projection(got, total, core, trunc, *, atol=2e-10):
    expected = core.tensor_from_total(total, trunc=trunc)
    assert got.spec == expected.spec
    for grade in got.grades:
        np.testing.assert_allclose(
            np.asarray(got[grade]),
            np.asarray(expected[grade]),
            atol=atol,
            rtol=atol,
            err_msg=f"bidegree {grade}",
        )


@pytest.mark.parametrize(
    ("order", "return_trajectory"),
    ((0, False), (1, True), (2, False)),
)
def test_scalar_fft_is_native_rectangular_projection(order, return_trajectory):
    active = (2, 1)
    kernel = _kernel(1)
    core = bigraded_core(
        dims=(1, 1),
        max_trunc=active,
        precompute_shuffle=False,
    )
    total = fft_iteration(
        _DX,
        kernel=kernel,
        trunc=sum(active),
        dt=0.25,
        order=order,
        return_trajectory=return_trajectory,
        core=_TOTAL_CORE,
    )
    got = fft_iteration(
        _DX,
        kernel=kernel,
        trunc=active,
        dt=0.25,
        order=order,
        return_trajectory=return_trajectory,
        core=core,
    )
    _assert_projection(got, total, core, active)


@pytest.mark.parametrize("precompute_shuffle", [True, "generator"])
def test_multicomponent_fft_is_native_rectangular_projection(
    precompute_shuffle,
):
    active = (2, 1)
    kernel = _kernel(2)
    core = bigraded_core(
        dims=(1, 1),
        max_trunc=active,
        precompute_shuffle=precompute_shuffle,
    )
    total = fft_iteration(
        _DX,
        kernel=kernel,
        trunc=sum(active),
        dt=0.25,
        order=1,
        core=_TOTAL_CORE,
    )
    got = fft_iteration(
        _DX,
        kernel=kernel,
        trunc=active,
        dt=0.25,
        order=1,
        core=core,
    )
    _assert_projection(got, total, core, active)


def test_fft_bidegree_source_group_fuses_multiple_grade_splits():
    active = (2, 1)
    core = bigraded_core(dims=(1, 1), max_trunc=active)
    algebra = resolve_volterra_algebra(core, active, 2)
    history_workset = algebra.diagonal(1)
    local_workset = algebra.diagonal(2)
    output_grade = (2, 1)
    history_values = tuple(
        jnp.arange(3 * width, dtype=jnp.float64).reshape(3, width) + index
        for index, width in enumerate(history_workset.widths)
    )
    local_values = tuple(
        jnp.arange(2 * 3 * width, dtype=jnp.float64).reshape(2, 3, width)
        + index
        for index, width in enumerate(local_workset.widths)
    )

    def grouped(*values):
        history = values[:history_workset.size]
        local = values[history_workset.size:]
        return _sum_product_splits(
            algebra=algebra,
            output_grade=output_grade,
            history_workset=history_workset,
            history_values=history,
            local_workset=local_workset,
            local_values=local,
        )

    contributions = tuple(
        (
            history_workset.block(history_values, history_grade)[None, ...],
            local_workset.block(local_values, local_grade),
            history_grade,
            local_grade,
        )
        for history_grade, local_grade in algebra.layout.product_splits(output_grade)
        if history_workset.contains(history_grade)
        and local_workset.contains(local_grade)
    )
    terms = tuple(
        core._product_block(left, right, left_grade, right_grade, output_grade)
        for left, right, left_grade, right_grade in contributions
    )
    expected = terms[0]
    for term in terms[1:]:
        expected = expected + term

    inputs = history_values + local_values
    got = grouped(*inputs)
    np.testing.assert_allclose(got, expected, atol=1e-13, rtol=1e-13)
    stablehlo = str(
        jax.jit(grouped).lower(*inputs).compiler_ir(dialect="stablehlo")
    )
    assert len(contributions) > 1
    assert stablehlo.count('"stablehlo.scatter"') == 1


def test_multicomponent_fft_projects_to_one_sided_rectangle():
    active = (0, 2)
    kernel = _kernel(2)
    core = bigraded_core(
        dims=(1, 1),
        max_trunc=active,
        precompute_shuffle=True,
    )
    total = fft_iteration(
        _DX,
        kernel=kernel,
        trunc=sum(active),
        dt=0.25,
        order=1,
        core=_TOTAL_CORE,
    )
    got = fft_iteration(
        _DX,
        kernel=kernel,
        trunc=active,
        dt=0.25,
        order=1,
        core=core,
    )
    _assert_projection(got, total, core, active)


def test_multicomponent_depth_one_does_not_require_shuffle_plans():
    active = (1, 0)
    kernel = _kernel(2)
    core = bigraded_core(
        dims=(1, 1),
        max_trunc=active,
        precompute_shuffle=False,
    )
    total = fft_iteration(
        _DX,
        kernel=kernel,
        trunc=1,
        dt=0.25,
        core=_TOTAL_CORE,
    )
    got = fft_iteration(
        _DX,
        kernel=kernel,
        trunc=active,
        dt=0.25,
        core=core,
    )
    _assert_projection(got, total, core, active)


def test_multicomponent_positive_depth_requires_shuffle_capability():
    core = bigraded_core(
        dims=(1, 1),
        max_trunc=(1, 1),
        precompute_shuffle=False,
    )
    with pytest.raises(RuntimeError, match="shuffle"):
        fft_iteration(
            _DX,
            kernel=_kernel(2),
            trunc=(1, 1),
            dt=0.25,
            core=core,
        )


def test_high_level_fft_routes_the_native_core():
    active = (1, 1)
    kernel = _kernel(1)
    core = bigraded_core(dims=(1, 1), max_trunc=active)
    path = jnp.concatenate(
        (jnp.zeros((1, 2), dtype=_DX.dtype), jnp.cumsum(_DX, axis=0)),
        axis=0,
    )
    total = td.vsig(
        path,
        kernel=kernel,
        trunc=sum(active),
        dt=0.25,
        scheme="fft",
        core=_TOTAL_CORE,
    )
    got = td.vsig(
        path,
        kernel=kernel,
        trunc=active,
        dt=0.25,
        scheme="fft",
        core=core,
    )
    _assert_projection(got, total, core, active)


def test_pair_and_larger_lag_tables_are_reused_across_active_rectangles():
    kernel = _kernel(1)
    core = bigraded_core(
        dims=(1, 1),
        max_trunc=(2, 1),
        default_trunc=(1, 1),
        precompute_shuffle=False,
    )
    tables = precompute_lag_tables(
        kernel,
        S=_DX.shape[0],
        h=0.25,
        order=1,
        trunc=(2, 1),
        dtype=_DX.dtype,
        core=core,
    )
    assert tables.max_order == 3
    assert tables.trunc == 3

    expected = fft_iteration(
        _DX,
        kernel=kernel,
        trunc=None,
        dt=0.25,
        order=1,
        core=core,
    )
    got = fft_iteration(
        _DX,
        kernel=kernel,
        trunc=None,
        dt=0.25,
        order=1,
        lag_tables=tables,
        core=core,
    )
    assert got.spec == expected.spec
    for grade in got.grades:
        np.testing.assert_allclose(got[grade], expected[grade], atol=1e-12, rtol=1e-12)


def test_integer_lag_precomputation_ignores_bigraded_background_core():
    kernel = _kernel(1)
    core = bigraded_core(
        dims=(1, 1),
        max_trunc=(1, 1),
        default_trunc=(1, 1),
    )
    previous_core, previous_seq_core = td.get_default_core_pair()
    try:
        td.set_default_core(core)
        tables = precompute_lag_tables(
            kernel,
            S=_DX.shape[0],
            h=0.25,
            order=0,
            trunc=3,
            dtype=_DX.dtype,
        )
    finally:
        td.set_default_core(previous_core, previous_seq_core)

    assert tables.max_order == 3


def test_fft_rejects_nonuniform_grid():
    kernel = _kernel(1)
    dt = jnp.array([0.2, 0.5, 0.3], dtype=jnp.float64)
    with pytest.raises(ValueError, match="requires a scalar dt"):
        fft_iteration(
            _DX,
            kernel=kernel,
            trunc=2,
            dt=dt,
            core=_TOTAL_CORE,
        )

    path = jnp.concatenate(
        (jnp.zeros((1, 2), dtype=_DX.dtype), jnp.cumsum(_DX, axis=0)),
        axis=0,
    )
    with pytest.raises(ValueError, match="requires a scalar dt"):
        td.vsig(
            path,
            kernel=kernel,
            trunc=2,
            dt=dt,
            scheme="fft",
            core=_TOTAL_CORE,
        )
