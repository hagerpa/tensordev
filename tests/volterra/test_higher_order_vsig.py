"""Tests comparing vsig (higher-order, non-FFT) against vsig_fft.

The FFT branch is used as the reference for order=1 and order=2 on uniform
grids.  A separate convergence test verifies that higher orders converge
to the order=0 result under dyadic refinement.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tensordev.core.jax import Jax
from tensordev.volterra import FractionalKernel, GammaKernel, vsig, vsig_fft
from tensordev.volterra.eval_general import eval_vte as eval_vte_general
from tensordev.volterra.eval_scalar import eval_vte as eval_vte_scalar

_CORE = Jax()


def _unit(batch_shape, m, trunc, dtype=jnp.float64):
    return (
        jnp.ones(batch_shape + (1,), dtype=dtype),
        *(jnp.zeros(batch_shape + (m**n,), dtype=dtype) for n in range(1, trunc + 1)),
    )


def _manual_volterra_q1(y, kernel, *, dt=1.0, trunc):
    """Reference Euler recursion for q=1 kernels."""
    y = jnp.asarray(y)
    S = y.shape[0]
    unit = _unit(y.shape[1:-1], kernel.m, trunc, dtype=y.dtype)
    times = jnp.arange(S + 1, dtype=y.dtype) * jnp.asarray(dt, dtype=y.dtype)
    history = [unit]
    for j in range(S):
        acc = unit
        for i in range(j + 1):
            c = kernel.coef(times[i], times[i + 1], times[j + 1], trunc=trunc, dtype=y.dtype)
            term = eval_vte_scalar(history[i], y[i], c)
            acc = _CORE.tensor_summation(acc, term, trunc=trunc)
        history.append(acc)
    return history[-1]


def _manual_volterra_q2(y, kernel, *, dt=1.0, trunc):
    """Reference Euler recursion for q>1 kernels."""
    y = jnp.asarray(y)
    S = y.shape[0]
    unit = _unit(y.shape[1:-2], kernel.m, trunc, dtype=y.dtype)
    times = jnp.arange(S + 1, dtype=y.dtype) * jnp.asarray(dt, dtype=y.dtype)
    history = [unit]
    for j in range(S):
        acc = unit
        for i in range(j + 1):
            c = kernel.coef(times[i], times[i + 1], times[j + 1], trunc=trunc, dtype=y.dtype)
            term = eval_vte_general(history[i], y[i], c)
            acc = _CORE.tensor_summation(acc, term, trunc=trunc)
        history.append(acc)
    return history[-1]


def _np(x):
    return np.asarray(jax.device_get(x))


def _assert_allclose(got, ref, *, atol, label=""):
    assert len(got) == len(ref)
    for n, (g, r) in enumerate(zip(got, ref)):
        np.testing.assert_allclose(
            _np(g), _np(r), atol=atol, rtol=0.0,
            err_msg=f"{label} level {n} mismatch",
        )


# ---------------------------------------------------------------------------
# vsig (order>=1) vs vsig_fft (order>=1) on uniform grids
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("trunc", [1, 2, 3])
@pytest.mark.parametrize("beta", [0.6, 0.9, 1.3])
def test_higher_order_vsig_matches_fft_fractional(order, trunc, beta):
    """vsig with order>=1 matches vsig_fft on a uniform fractional grid."""
    X = jnp.array(
        [[0.0, 0.0], [0.2, -0.1], [0.4, 0.3], [0.1, 0.5], [-0.2, 0.2]],
        dtype=jnp.float64,
    )
    A = jnp.eye(2, dtype=jnp.float64)[None, :, :]
    kernel = FractionalKernel(beta=jnp.array([beta]), A=A)

    ref = vsig_fft(X, kernel=kernel, dt=0.5, trunc=trunc, order=order)
    got = vsig(X, kernel=kernel, dt=0.5, trunc=trunc, order=order)
    _assert_allclose(got, ref, atol=1e-8, label=f"order={order} trunc={trunc} beta={beta}")


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("trunc", [1, 2, 3])
def test_higher_order_vsig_matches_fft_gamma(order, trunc):
    """vsig with order>=1 matches vsig_fft for gamma kernel."""
    X = jnp.array([[0.0], [0.3], [-0.1], [0.4], [0.2]], dtype=jnp.float64)
    A = jnp.ones((1, 1, 1), dtype=jnp.float64)
    kernel = GammaKernel(
        beta=jnp.array([0.8]), A=A, scale=jnp.array([1.5]), rate=jnp.array([0.5]),
        quad_order=32,
    )

    ref = vsig_fft(X, kernel=kernel, dt=0.25, trunc=trunc, order=order)
    got = vsig(X, kernel=kernel, dt=0.25, trunc=trunc, order=order)
    _assert_allclose(got, ref, atol=1e-6, label=f"order={order} trunc={trunc}")


# ---------------------------------------------------------------------------
# output_starting_point consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2])
def test_higher_order_vsig_trajectory_matches_fft(order):
    """output_starting_point=True trajectory matches vsig_fft level-by-level."""
    X = jnp.array(
        [[0.0, 0.0], [0.2, -0.1], [0.4, 0.3], [0.1, 0.5]],
        dtype=jnp.float64,
    )
    A = jnp.eye(2, dtype=jnp.float64)[None, :, :]
    kernel = FractionalKernel(beta=jnp.array([0.8]), A=A)
    trunc = 3

    ref = vsig_fft(X, kernel=kernel, dt=1.0, trunc=trunc, order=order,
                   output_starting_point=True)
    got = vsig(X, kernel=kernel, dt=1.0, trunc=trunc, order=order,
               output_starting_point=True)
    _assert_allclose(got, ref, atol=1e-8, label=f"trajectory order={order}")


# ---------------------------------------------------------------------------
# terminal value consistency: trajectory[-1] == terminal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2])
def test_higher_order_vsig_terminal_consistent_with_trajectory(order):
    """Terminal value from output_starting_point=True matches the direct call."""
    X = jnp.array(
        [[0.0, 0.0], [0.2, -0.1], [0.4, 0.3], [0.1, 0.5]],
        dtype=jnp.float64,
    )
    A = jnp.eye(2, dtype=jnp.float64)[None, :, :]
    kernel = FractionalKernel(beta=jnp.array([0.7]), A=A)
    trunc = 3

    terminal = vsig(X, kernel=kernel, dt=1.0, trunc=trunc, order=order)
    traj = vsig(X, kernel=kernel, dt=1.0, trunc=trunc, order=order,
                output_starting_point=True)

    _assert_allclose(terminal, tuple(level[-1] for level in traj),
                     atol=1e-12, label=f"terminal vs traj[-1] order={order}")


# ---------------------------------------------------------------------------
# Non-uniform grid: higher order converges to order=0 under dyadic refinement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2])
def test_higher_order_vsig_nonuniform_self_convergence(order):
    """Higher-order vsig on a non-uniform grid self-converges under dyadic refinement."""
    X = jnp.array(
        [[0.0, 0.0], [0.2, -0.1], [0.4, 0.3], [0.1, 0.5], [-0.2, 0.2]],
        dtype=jnp.float64,
    )
    A = jnp.eye(2, dtype=jnp.float64)[None, :, :]
    kernel = FractionalKernel(beta=jnp.array([0.7]), A=A)
    dt = jnp.array([0.3, 0.5, 0.2, 0.4], dtype=jnp.float64)
    trunc = 3

    coarse = vsig(X, kernel=kernel, dt=dt, trunc=trunc, order=order, dyadic_order=4)
    fine = vsig(X, kernel=kernel, dt=dt, trunc=trunc, order=order, dyadic_order=5)
    _assert_allclose(coarse, fine, atol=5e-3, label=f"nonuniform order={order} self-convergence")


# ---------------------------------------------------------------------------
# q > 1 tests
# ---------------------------------------------------------------------------

def _make_q2_kernel(beta1=0.8, beta2=1.2, d=4, m=2):
    """FractionalKernel with q=2, two different betas."""
    A = jnp.zeros((2, m, d), dtype=jnp.float64)
    A = A.at[0, :, :d // 2].set(jnp.eye(m, d // 2))
    A = A.at[1, :, d // 2:].set(jnp.eye(m, d // 2))
    return FractionalKernel(beta=jnp.array([beta1, beta2], dtype=jnp.float64), A=A)


def test_vsig_q2_order0_matches_manual_euler():
    """Unified scan for q=2, order=0 matches the manual Euler recursion."""
    X = jnp.array(
        [[0.0, 0.0, 0.0, 0.0], [0.2, -0.1, 0.1, 0.3],
         [0.4, 0.3, -0.2, 0.1], [0.1, 0.5, 0.0, -0.2]],
        dtype=jnp.float64,
    )
    kernel = _make_q2_kernel()
    trunc = 3

    got = vsig(X, kernel=kernel, dt=0.5, trunc=trunc, order=0)
    dX = jnp.diff(X, axis=0)
    projected = jnp.einsum("qmd,sd->sqm", kernel.A, dX)
    ref = _manual_volterra_q2(projected, kernel, dt=0.5, trunc=trunc)
    _assert_allclose(got, ref, atol=1e-9, label="q=2 order=0 vs manual")


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("trunc", [1, 2, 3])
def test_vsig_q2_higher_order_self_convergence(order, trunc):
    """q=2 higher-order vsig self-converges under dyadic refinement."""
    X = jnp.array(
        [[0.0, 0.0, 0.0, 0.0], [0.2, -0.1, 0.1, 0.3],
         [0.4, 0.3, -0.2, 0.1], [0.1, 0.5, 0.0, -0.2]],
        dtype=jnp.float64,
    )
    kernel = _make_q2_kernel()

    coarse = vsig(X, kernel=kernel, dt=0.5, trunc=trunc, order=order, dyadic_order=3)
    fine = vsig(X, kernel=kernel, dt=0.5, trunc=trunc, order=order, dyadic_order=4)
    _assert_allclose(coarse, fine, atol=0.05, label=f"q=2 order={order} trunc={trunc}")


@pytest.mark.parametrize("order", [1, 2])
def test_vsig_q2_higher_order_converges_to_order0(order):
    """q=2 higher-order vsig converges to order=0 with sufficient dyadic refinement."""
    X = jnp.array(
        [[0.0, 0.0, 0.0, 0.0], [0.2, -0.1, 0.1, 0.3],
         [0.4, 0.3, -0.2, 0.1], [0.1, 0.5, 0.0, -0.2]],
        dtype=jnp.float64,
    )
    kernel = _make_q2_kernel()
    trunc = 2

    ref = vsig(X, kernel=kernel, dt=0.5, trunc=trunc, order=0, dyadic_order=5)
    got = vsig(X, kernel=kernel, dt=0.5, trunc=trunc, order=order, dyadic_order=5)
    _assert_allclose(got, ref, atol=0.05, label=f"q=2 order={order} converges to order=0")