"""Correctness tests: vsig_fft (q > 1) agrees with vsig on uniform grids.

Each test compares the FFT-based implementation against the general non-FFT
Volterra iteration for a q = 2 fractional kernel.  The key quantity under
test is the channel ordering

    channel = p * M_{r-1} + ell_index

used in both the lag-weight tables and the shuffle-monomial source channels.
A mismatch there would show up as wrong numerical values even for simple paths.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tensordev.volterra import FractionalKernel, vsig, vsig_fft
from tensordev.volterra.iteration_fft import precompute_lag_tables


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _np(x):
    return np.asarray(jax.device_get(x))


def _assert_allclose(got, ref, *, atol, rtol=1e-7, label=""):
    assert len(got) == len(ref), f"{label}: level count mismatch {len(got)} vs {len(ref)}"
    for n, (g, r) in enumerate(zip(got, ref)):
        np.testing.assert_allclose(
            _np(g), _np(r), atol=atol, rtol=rtol,
            err_msg=f"{label} level {n}",
        )


def _q2_kernel_m2(beta0=0.8, beta1=1.0):
    """q=2 fractional kernel, m=2 tensor letter dimension, 2-d path."""
    beta = jnp.array([beta0, beta1], dtype=jnp.float64)
    # A shape (q=2, m=2, d=2): each component is the 2×2 identity.
    A = jnp.stack([jnp.eye(2), jnp.eye(2)], axis=0).astype(jnp.float64)
    return FractionalKernel(beta=beta, A=A)


def _q2_kernel_m1(beta0=0.8, beta1=1.0):
    """q=2 fractional kernel, m=1 tensor letter dimension, 1-d path."""
    beta = jnp.array([beta0, beta1], dtype=jnp.float64)
    # A shape (q=2, m=1, d=1)
    A = jnp.ones((2, 1, 1), dtype=jnp.float64)
    return FractionalKernel(beta=beta, A=A)


_X_2d = jnp.array(
    [[0.0, 0.0], [0.3, -0.1], [0.1, 0.4], [-0.2, 0.2]],
    dtype=jnp.float64,
)  # S+1=4 nodes, d=2, so S=3 increments

_X_1d = jnp.array(
    [[0.0], [0.5], [-0.2], [0.3], [0.1]],
    dtype=jnp.float64,
)  # S+1=5 nodes, d=1, so S=4 increments


# ---------------------------------------------------------------------------
# 1. q=1 regression: scalar FFT still agrees with vsig
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("trunc", [1, 2, 3])
def test_q1_fft_matches_vsig(order, trunc):
    """q=1 FFT path is unaffected by the q>1 wiring."""
    X = jnp.array(
        [[0.0, 0.0], [0.2, -0.1], [0.4, 0.3], [0.1, 0.5]],
        dtype=jnp.float64,
    )
    kernel = FractionalKernel(beta=jnp.array([0.75]), A=jnp.eye(2)[None])
    ref = vsig(X, kernel=kernel, dt=0.5, trunc=trunc, order=order)
    got = vsig_fft(X, kernel=kernel, dt=0.5, trunc=trunc, order=order)
    _assert_allclose(got, ref, atol=1e-8, label=f"q1 order={order} trunc={trunc}")


# ---------------------------------------------------------------------------
# 2. q=2, m=1 — simplest layout, catches channel-order bugs clearly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("trunc", [1, 2, 3])
def test_q2_m1_order_vs_vsig(order, trunc):
    """vsig_fft (q=2, m=1) agrees with vsig for all basis orders."""
    kernel = _q2_kernel_m1(beta0=0.7, beta1=0.9)
    ref = vsig(_X_1d, kernel=kernel, dt=0.25, trunc=trunc, order=order)
    got = vsig_fft(_X_1d, kernel=kernel, dt=0.25, trunc=trunc, order=order)
    _assert_allclose(
        got, ref, atol=1e-8,
        label=f"q2 m1 order={order} trunc={trunc}",
    )


@pytest.mark.parametrize("beta_pair", [(0.7, 0.9), (1.0, 1.0), (0.6, 1.2)])
def test_q2_m1_various_betas(beta_pair):
    """q=2, m=1: agreement at trunc=2 across several beta pairs."""
    kernel = _q2_kernel_m1(*beta_pair)
    ref = vsig(_X_1d, kernel=kernel, dt=0.25, trunc=2, order=1)
    got = vsig_fft(_X_1d, kernel=kernel, dt=0.25, trunc=2, order=1)
    _assert_allclose(
        got, ref, atol=1e-8,
        label=f"q2 m1 betas={beta_pair}",
    )


# ---------------------------------------------------------------------------
# 3. q=2, m=2 — layout-sensitive tensor-letter dimension
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("trunc", [2, 3])
def test_q2_m2_order_vs_vsig(order, trunc):
    """vsig_fft (q=2, m=2) agrees with vsig — catches letter-axis bugs."""
    kernel = _q2_kernel_m2(beta0=0.8, beta1=1.0)
    ref = vsig(_X_2d, kernel=kernel, dt=1.0, trunc=trunc, order=order)
    got = vsig_fft(_X_2d, kernel=kernel, dt=1.0, trunc=trunc, order=order)
    _assert_allclose(
        got, ref, atol=1e-8,
        label=f"q2 m2 order={order} trunc={trunc}",
    )


# ---------------------------------------------------------------------------
# 4. output_starting_point=True: full trajectory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [0, 1])
def test_q2_m1_trajectory(order):
    """Trajectory output (output_starting_point=True) agrees level-by-level."""
    kernel = _q2_kernel_m1(beta0=0.8, beta1=0.9)
    ref = vsig(_X_1d, kernel=kernel, dt=0.25, trunc=2, order=order,
               output_starting_point=True)
    got = vsig_fft(_X_1d, kernel=kernel, dt=0.25, trunc=2, order=order,
                   output_starting_point=True)
    _assert_allclose(got, ref, atol=1e-8, label=f"q2 m1 trajectory order={order}")


# ---------------------------------------------------------------------------
# 5. Precomputed lag tables: must agree with on-the-fly computation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [0, 1, 2])
def test_q2_m1_precomputed_tables(order):
    """Precomputed lag tables give identical results to on-the-fly tables."""
    kernel = _q2_kernel_m1(beta0=0.7, beta1=0.9)
    dt = 0.25
    trunc = 2

    # On-the-fly (reference for this sub-test)
    ref = vsig_fft(_X_1d, kernel=kernel, dt=dt, trunc=trunc, order=order)

    # Precomputed
    S_eff = _X_1d.shape[0] - 1  # number of increments
    lag_tables = precompute_lag_tables(
        kernel, S=S_eff, h=dt, order=order, trunc=trunc, dtype=jnp.float64,
    )
    got = vsig_fft(_X_1d, kernel=kernel, dt=dt, trunc=trunc, order=order,
                   lag_tables=lag_tables)
    _assert_allclose(got, ref, atol=1e-12, label=f"q2 precomputed order={order}")

    # Also compare against vsig
    ref_vsig = vsig(_X_1d, kernel=kernel, dt=dt, trunc=trunc, order=order)
    _assert_allclose(got, ref_vsig, atol=1e-8, label=f"q2 precomputed vs vsig order={order}")


# ---------------------------------------------------------------------------
# 6. q=2 with non-trivial batch dimension
# ---------------------------------------------------------------------------

def test_q2_m1_batch():
    """q=2 FFT agrees with vsig when X has a leading batch axis."""
    kernel = _q2_kernel_m1(beta0=0.9, beta1=1.1)
    # X: (batch=2, S+1=4, d=1)
    X_batch = jnp.stack([_X_1d[:4], -_X_1d[:4]], axis=0)

    ref = vsig(X_batch, kernel=kernel, dt=0.25, trunc=2, order=1, axis=-2)
    got = vsig_fft(X_batch, kernel=kernel, dt=0.25, trunc=2, order=1, axis=-2)
    _assert_allclose(got, ref, atol=1e-8, label="q2 batch")

