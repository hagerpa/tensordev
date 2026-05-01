from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from tensordev.core.jax import Jax
from tensordev.development import path_signature
from tensordev.volterra import VolterraKernel, vsig
from tensordev.volterra.eval_general import eval_e as eval_e_general
from tensordev.volterra.eval_general import eval_vte as eval_vte_general
from tensordev.volterra.eval_scalar import eval_e as eval_e_scalar
from tensordev.volterra.eval_scalar import eval_vte as eval_vte_scalar


_CORE = Jax()


def _np(x):
    return np.asarray(jax.device_get(x))


def _assert_dense_allclose(got, expected, *, atol=1e-10, rtol=1e-10):
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        np.testing.assert_allclose(_np(g), _np(e), atol=atol, rtol=rtol)


def _eval_e(y, coeffs):
    if coeffs.q == 1:
        return eval_e_scalar(y, coeffs)
    return eval_e_general(y, coeffs)


def _eval_vte(v, y, coeffs):
    if coeffs.q == 1:
        return eval_vte_scalar(v, y, coeffs)
    return eval_vte_general(v, y, coeffs)


def _unit(batch_shape, m, trunc, dtype=jnp.float64):
    return (
        jnp.ones(batch_shape + (1,), dtype=dtype),
        *(jnp.zeros(batch_shape + (m**n,), dtype=dtype) for n in range(1, trunc + 1)),
    )


def _manual_volterra(y, kernel, *, times, trunc):
    y = jnp.asarray(y)
    S = y.shape[0]
    batch_shape = y.shape[1:-1] if kernel.q == 1 else y.shape[1:-2]
    unit = _unit(batch_shape, kernel.m, trunc, dtype=y.dtype)
    history = [unit]
    for j in range(S):
        acc = unit
        for i in range(j + 1):
            if kernel.kind == "piecewise_constant":
                coeffs = kernel.coef_from_indices(i, j, trunc=trunc, dtype=y.dtype)
            else:
                coeffs = kernel.coef(times[i], times[i + 1], times[j + 1], trunc=trunc, dtype=y.dtype)
            term = _eval_vte(history[i], y[i], coeffs)
            acc = _CORE.tensor_summation(acc, term, trunc=trunc)
        history.append(acc)
    return history[-1], tuple(jnp.stack([h[n] for h in history], axis=0) for n in range(trunc + 1))


def test_volterra_vsig_one_interval_matches_unit_plus_local_e():
    trunc = 4
    X = jnp.array([[0.0, 0.0], [0.2, -0.5]], dtype=jnp.float64)
    A = jnp.eye(2, dtype=jnp.float64)[None, :, :]
    kernel = VolterraKernel.fractional(beta=jnp.array([1.0]), A=A)

    got = vsig(X, kernel=kernel, dt=0.7, trunc=trunc)

    dX = jnp.diff(X, axis=0)
    coeffs = kernel.coef(0.0, 0.7, 0.7, trunc=trunc, dtype=X.dtype)
    E = _eval_e(dX[0], coeffs)
    expected = _CORE.tensor_summation(_unit((), 2, trunc), E, trunc=trunc)
    _assert_dense_allclose(got, expected)


def test_volterra_vsig_two_intervals_matches_explicit_formula():
    trunc = 3
    X = jnp.array([[0.0, 0.0], [0.2, -0.5], [0.3, -0.2]], dtype=jnp.float64)
    A = jnp.eye(2, dtype=jnp.float64)[None, :, :]
    kernel = VolterraKernel.fractional(beta=jnp.array([0.8]), A=A)
    times = jnp.array([0.0, 0.4, 1.0], dtype=jnp.float64)

    got = vsig(X, kernel=kernel, times=times, trunc=trunc)
    dX = jnp.diff(X, axis=0)
    expected, _ = _manual_volterra(dX, kernel, times=times, trunc=trunc)

    _assert_dense_allclose(got, expected, atol=2e-10, rtol=2e-10)


def test_volterra_vsig_q_gt_one_piecewise_matches_explicit_triangular_recursion():
    trunc = 3
    B = jnp.array(
        [
            [[1.0, 0.5, -0.2], [0.0, 0.7, 0.4], [0.0, 0.0, 1.2]],
            [[0.3, -0.1, 0.8], [0.0, 1.1, -0.6], [0.0, 0.0, 0.9]],
        ],
        dtype=jnp.float64,
    )
    # Block-diagonal A: component 0 reads path dims 0:2, component 1 reads dims 2:4.
    A = jnp.zeros((2, 2, 4), dtype=jnp.float64)
    A = A.at[0, :, :2].set(jnp.eye(2, dtype=jnp.float64))
    A = A.at[1, :, 2:].set(jnp.eye(2, dtype=jnp.float64))
    kernel = VolterraKernel.piecewise_constant(B=B, A=A)
    times = jnp.arange(4, dtype=jnp.float64)

    # Flat path increments embedding independent (q, m) values into (S, d=4).
    dX = jnp.array(
        [[0.2, -0.5, 0.1, 0.3], [-0.4, 0.7, 0.2, -0.1], [0.3, 0.2, -0.2, 0.5]],
        dtype=jnp.float64,
    )
    X = jnp.concatenate(
        [jnp.zeros((1, 4), dtype=jnp.float64), jnp.cumsum(dX, axis=0)], axis=0
    )

    got = vsig(X, kernel=kernel, times=times, trunc=trunc)
    proj_y = jnp.einsum("qmd,sd->sqm", A, dX)
    expected, _ = _manual_volterra(proj_y, kernel, times=times, trunc=trunc)

    _assert_dense_allclose(got, expected, atol=2e-10, rtol=2e-10)


def test_volterra_vsig_beta_one_q_one_recovers_classical_signature():
    X = jnp.array(
        [
            [0.0, 0.0],
            [0.2, -0.1],
            [0.4, 0.3],
            [0.1, 0.5],
        ],
        dtype=jnp.float64,
    )
    trunc = 4
    A = jnp.eye(2, dtype=jnp.float64)[None, :, :]
    kernel = VolterraKernel.fractional(beta=jnp.array([1.0]), A=A)

    got = vsig(X, kernel=kernel, dt=1.0, trunc=trunc)
    expected = path_signature(X, trunc=trunc, axis=-2, core=_CORE)

    _assert_dense_allclose(got, expected, atol=2e-10, rtol=2e-10)


def test_volterra_vsig_output_starting_point_returns_padded_history_trajectory():
    trunc = 2
    X = jnp.array([[0.0], [0.2], [0.1], [0.5]], dtype=jnp.float64)
    A = jnp.ones((1, 1, 1), dtype=jnp.float64)
    kernel = VolterraKernel.fractional(beta=jnp.array([1.0]), A=A)
    times = jnp.arange(4, dtype=jnp.float64)

    got = vsig(X, kernel=kernel, times=times, trunc=trunc, output_starting_point=True)
    dX = jnp.diff(X, axis=0)
    terminal, expected_path = _manual_volterra(dX, kernel, times=times, trunc=trunc)

    assert got[0].shape == (4, 1)
    for level, expected in zip(got, expected_path):
        np.testing.assert_allclose(_np(level), _np(expected), atol=2e-10, rtol=2e-10)
    _assert_dense_allclose(tuple(level[-1] for level in got), terminal)


def test_volterra_vsig_batched_path_shapes():
    X = jnp.array(
        [
            [[0.0, 0.0], [0.1, 0.2], [0.3, 0.1]],
            [[0.0, 0.0], [-0.2, 0.4], [0.1, 0.5]],
        ],
        dtype=jnp.float64,
    )
    trunc = 3
    A = jnp.eye(2, dtype=jnp.float64)[None, :, :]
    kernel = VolterraKernel.fractional(beta=jnp.array([1.0]), A=A)

    got = vsig(X, kernel=kernel, dt=1.0, trunc=trunc, axis=1)

    assert got[0].shape == (2, 1)
    assert got[1].shape == (2, 2)
    assert got[2].shape == (2, 4)
    assert got[3].shape == (2, 8)
