from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import pytest

from tensordev.core.jax import Jax
from tensordev.development import path_signature
from tensordev.volterra import ConvolutionKernel, FractionalKernel, GammaKernel, vsig, vsig_fft
from tensordev.volterra.eval_general import eval_e as eval_e_general
from tensordev.volterra.eval_general import eval_vte as eval_vte_general
from tensordev.volterra.eval_scalar import eval_vte as eval_vte_scalar


_CORE = Jax()


def _np(x):
    return np.asarray(jax.device_get(x))


def _assert_dense_allclose(got, expected, *, atol=1e-10, rtol=1e-10):
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        np.testing.assert_allclose(_np(g), _np(e), atol=atol, rtol=rtol)


def _eval_e(y, coeffs):
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


def _manual_volterra(y, kernel, *, dt=1.0, trunc):
    y = jnp.asarray(y)
    S = y.shape[0]
    batch_shape = y.shape[1:-1] if kernel.q == 1 else y.shape[1:-2]
    unit = _unit(batch_shape, kernel.m, trunc, dtype=y.dtype)
    dt_arr = jnp.asarray(dt, dtype=y.dtype)
    if dt_arr.ndim == 0:
        times = jnp.arange(S + 1, dtype=y.dtype) * dt_arr
    else:
        times = jnp.concatenate([jnp.zeros((1,), dtype=y.dtype), jnp.cumsum(dt_arr)])
    history = [unit]
    for j in range(S):
        acc = unit
        for i in range(j + 1):
            coeffs = kernel.coef(times[i], times[i + 1], times[j + 1], trunc=trunc, dtype=y.dtype)
            term = _eval_vte(history[i], y[i], coeffs)
            acc = _CORE.tensor_summation(acc, term, trunc=trunc)
        history.append(acc)
    return history[-1], tuple(jnp.stack([h[n] for h in history], axis=0) for n in range(trunc + 1))


def test_volterra_vsig_one_interval_matches_unit_plus_local_e():
    trunc = 4
    X = jnp.array([[0.0, 0.0], [0.2, -0.5]], dtype=jnp.float64)
    A = jnp.eye(2, dtype=jnp.float64)[None, :, :]
    kernel = FractionalKernel(beta=jnp.array([1.0]), A=A)

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
    kernel = FractionalKernel(beta=jnp.array([0.8]), A=A)
    dt = jnp.array([0.4, 0.6], dtype=jnp.float64)

    got = vsig(X, kernel=kernel, dt=dt, trunc=trunc)
    dX = jnp.diff(X, axis=0)
    expected, _ = _manual_volterra(dX, kernel, dt=dt, trunc=trunc)

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
    kernel = FractionalKernel(beta=jnp.array([1.0]), A=A)

    got = vsig(X, kernel=kernel, dt=1.0, trunc=trunc)
    expected = path_signature(X, trunc=trunc, axis=-2, core=_CORE)

    _assert_dense_allclose(got, expected, atol=2e-10, rtol=2e-10)


def test_volterra_vsig_output_starting_point_returns_padded_history_trajectory():
    trunc = 2
    X = jnp.array([[0.0], [0.2], [0.1], [0.5]], dtype=jnp.float64)
    A = jnp.ones((1, 1, 1), dtype=jnp.float64)
    kernel = FractionalKernel(beta=jnp.array([1.0]), A=A)

    # block_size=1 + output_starting_point=True → [unit, V_1, V_2, V_3]
    got = vsig(X, kernel=kernel, dt=1.0, trunc=trunc, block_size=1, output_starting_point=True)
    dX = jnp.diff(X, axis=0)
    terminal, expected_path = _manual_volterra(dX, kernel, trunc=trunc)

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
    kernel = FractionalKernel(beta=jnp.array([1.0]), A=A)

    got = vsig(X, kernel=kernel, dt=1.0, trunc=trunc, axis=1)

    assert got[0].shape == (2, 1)
    assert got[1].shape == (2, 2)
    assert got[2].shape == (2, 4)
    assert got[3].shape == (2, 8)


# ---------------------------------------------------------------------------
# vsig_fft vs vsig agreement
# ---------------------------------------------------------------------------

# FFT introduces ~1e-13 rounding on float64; the einsum path matches exactly.
_FFT_ATOL = 1e-10


def _assert_fft_allclose(got, ref, *, atol=_FFT_ATOL):
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        np.testing.assert_allclose(_np(g), _np(r), atol=atol, rtol=0.0,
                                   err_msg="vsig_fft and vsig disagree")


@pytest.mark.parametrize("trunc", [1, 2, 3, 4])
@pytest.mark.parametrize("beta", [0.6, 1.0, 1.5])
def test_vsig_fft_fractional_uniform_grid_matches_vsig(trunc, beta):
    """FFT path (fractional kernel, uniform grid) matches the scan-based vsig."""
    X = jnp.array([[0.0, 0.0], [0.2, -0.1], [0.4, 0.3], [0.1, 0.5], [-0.2, 0.2]],
                  dtype=jnp.float64)
    A = jnp.eye(2, dtype=jnp.float64)[None, :, :]
    kernel = FractionalKernel(beta=jnp.array([beta]), A=A)

    ref = vsig(X, kernel=kernel, dt=0.5, trunc=trunc)
    got = vsig_fft(X, kernel=kernel, dt=0.5, trunc=trunc)
    _assert_fft_allclose(got, ref)


@pytest.mark.parametrize("trunc", [1, 2, 3])
@pytest.mark.parametrize("beta,rate", [(0.8, 0.5), (1.0, 1.0), (1.2, 2.0)])
def test_vsig_fft_gamma_uniform_grid_matches_vsig(trunc, beta, rate):
    """FFT path (gamma kernel, uniform grid) matches the scan-based vsig."""
    X = jnp.array([[0.0], [0.3], [-0.1], [0.4], [0.2]], dtype=jnp.float64)
    A = jnp.ones((1, 1, 1), dtype=jnp.float64)
    kernel = GammaKernel(
        beta=jnp.array([beta]), A=A, scale=jnp.array([1.5]), rate=jnp.array([rate]),
        quad_order=32,
    )

    ref = vsig(X, kernel=kernel, dt=0.25, trunc=trunc)
    got = vsig_fft(X, kernel=kernel, dt=0.25, trunc=trunc)
    _assert_fft_allclose(got, ref, atol=1e-7)


def test_vsig_fft_output_starting_point_matches_vsig():
    """Full trajectory (block_size=1, output_starting_point=True) matches vsig level-by-level."""
    X = jnp.array([[0.0, 0.0], [0.2, -0.1], [0.4, 0.3], [0.1, 0.5]], dtype=jnp.float64)
    A = jnp.eye(2, dtype=jnp.float64)[None, :, :]
    kernel = FractionalKernel(beta=jnp.array([0.8]), A=A)
    trunc = 3

    ref = vsig(X, kernel=kernel, dt=1.0, trunc=trunc, block_size=1, output_starting_point=True)
    got = vsig_fft(X, kernel=kernel, dt=1.0, trunc=trunc, block_size=1, output_starting_point=True)
    _assert_fft_allclose(got, ref)


def test_vsig_fft_batched_path_matches_vsig():
    """Batch axis is handled correctly and results match vsig per-batch-element."""
    X = jnp.array(
        [
            [[0.0, 0.0], [0.2, -0.1], [0.4, 0.3], [0.1, 0.5]],
            [[0.0, 0.0], [-0.1, 0.2], [0.3, -0.3], [0.2, 0.1]],
        ],
        dtype=jnp.float64,
    )
    A = jnp.eye(2, dtype=jnp.float64)[None, :, :]
    kernel = FractionalKernel(beta=jnp.array([0.9]), A=A)
    trunc = 3

    ref = vsig(X, kernel=kernel, dt=1.0, trunc=trunc, axis=1)
    got = vsig_fft(X, kernel=kernel, dt=1.0, trunc=trunc, axis=1)
    _assert_fft_allclose(got, ref)


@pytest.mark.parametrize("dyadic_order", [1, 2])
def test_vsig_fft_dyadic_refinement_matches_vsig(dyadic_order):
    """Dyadic path refinement gives the same result as vsig."""
    X = jnp.array([[0.0, 0.0], [0.3, -0.2], [0.1, 0.4]], dtype=jnp.float64)
    A = jnp.eye(2, dtype=jnp.float64)[None, :, :]
    kernel = FractionalKernel(beta=jnp.array([0.7]), A=A)
    trunc = 3

    ref = vsig(X, kernel=kernel, dt=1.0, trunc=trunc, dyadic_order=dyadic_order)
    got = vsig_fft(X, kernel=kernel, dt=1.0, trunc=trunc, dyadic_order=dyadic_order)
    _assert_fft_allclose(got, ref)


def test_vsig_fft_q_gt_one_runs():
    """vsig_fft with q = 2 completes without shape errors."""
    # q=2 kernel: two 2×4 projection matrices on a 4-d path.
    A = jnp.zeros((2, 2, 4), dtype=jnp.float64)
    A = A.at[0, :, :2].set(jnp.eye(2))
    A = A.at[1, :, 2:].set(jnp.eye(2))
    beta = jnp.array([1.0, 1.0], dtype=jnp.float64)
    kernel = FractionalKernel(beta=beta, A=A)
    X = jnp.zeros((3, 4), dtype=jnp.float64)

    result = vsig_fft(X, kernel=kernel, dt=1.0, trunc=2)
    assert len(result) == 3           # levels 0, 1, 2
    assert result[0].shape == (1,)    # scalar unit
    assert result[1].shape == (2,)    # m = 2
    assert result[2].shape == (4,)    # m**2 = 4
