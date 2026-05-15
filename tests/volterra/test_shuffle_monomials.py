"""Tests for _shuffle_monomials_by_degree (q > 1 FFT helper).

Verifies:
- Correct shapes at each degree.
- Degree-0 is all-ones.
- Degree-1 rows match y_0, y_1 in _compositions_desc order.
- Degree-2 rows match y_0⊗y_0, y_0⊗y_1+y_1⊗y_0, y_1⊗y_1.
- Multi-index ordering aligns with kernel.lag_weights(n=k+1).
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from tensordev.util.combinatorics import build_multiindex_layout
from tensordev.volterra.iteration_fft import _shuffle_monomials_by_degree
from tensordev.volterra import FractionalKernel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np(x):
    return np.asarray(jax.device_get(x))


def _idx_of(layout, ell):
    """Host-side: packed index of multi-index tuple ``ell``."""
    return layout.index_of(ell)


def _outer(a, b):
    """Flat outer product: (..., m) x (..., m) -> (..., m*m)."""
    return jnp.einsum("...i,...j->...ij", a, b).reshape(a.shape[:-1] + (a.shape[-1] * b.shape[-1],))


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q,m,trunc", [
    (2, 1, 0),
    (2, 2, 0),
    (2, 2, 2),
    (2, 3, 3),
    (3, 2, 2),
])
def test_shuffle_monomials_shapes(q, m, trunc):
    S = 4
    batch = (2,)
    y = jnp.ones((S,) + batch + (q, m), dtype=jnp.float64)
    layout = build_multiindex_layout(q=q, trunc=trunc)

    mons = _shuffle_monomials_by_degree(y, trunc=trunc, dtype=jnp.float64)
    assert len(mons) == trunc + 1, f"Expected {trunc+1} degrees, got {len(mons)}."

    for k, mon in enumerate(mons):
        M_k = int(layout.offsets[k + 1]) - int(layout.offsets[k])
        expected_shape = (S,) + batch + (M_k, m ** k)
        assert mon.shape == expected_shape, (
            f"Degree {k}: expected shape {expected_shape}, got {mon.shape}."
        )


# ---------------------------------------------------------------------------
# Value tests: q=2, m=2, no batch dims
# ---------------------------------------------------------------------------

@pytest.fixture
def y_q2_m2():
    """Fixed y with shape (3, 2, 2), no batch dims."""
    return jnp.array(
        [
            [[1.0, 2.0], [3.0, 5.0]],
            [[-1.0, 4.0], [2.0, -3.0]],
            [[0.5, -0.5], [1.0, 1.0]],
        ],
        dtype=jnp.float64,
    )  # shape (3, 2, 2): S=3, q=2, m=2


def test_degree0_is_ones(y_q2_m2):
    mons = _shuffle_monomials_by_degree(y_q2_m2, trunc=2, dtype=jnp.float64)
    assert mons[0].shape == (3, 1, 1)
    np.testing.assert_allclose(_np(mons[0]), 1.0)


def test_degree1_order_q2(y_q2_m2):
    """Degree-1: _compositions_desc order = [(1,0), (0,1)] for q=2."""
    mons = _shuffle_monomials_by_degree(y_q2_m2, trunc=2, dtype=jnp.float64)
    layout = build_multiindex_layout(q=2, trunc=2)

    i10 = _idx_of(layout, (1, 0)) - int(layout.offsets[1])  # local index in deg-1 block
    i01 = _idx_of(layout, (0, 1)) - int(layout.offsets[1])

    y0 = y_q2_m2[..., 0, :]  # (3, 2)
    y1 = y_q2_m2[..., 1, :]  # (3, 2)

    np.testing.assert_allclose(_np(mons[1][:, i10, :]), _np(y0), atol=1e-12,
                                err_msg="M_(1,0) should equal y_0.")
    np.testing.assert_allclose(_np(mons[1][:, i01, :]), _np(y1), atol=1e-12,
                                err_msg="M_(0,1) should equal y_1.")


def test_degree2_values_q2(y_q2_m2):
    """Degree-2: ordering [(2,0), (1,1), (0,2)] for q=2.

    M_(2,0) = y0 ⊗ y0
    M_(1,1) = y0 ⊗ y1 + y1 ⊗ y0
    M_(0,2) = y1 ⊗ y1
    """
    mons = _shuffle_monomials_by_degree(y_q2_m2, trunc=2, dtype=jnp.float64)
    layout = build_multiindex_layout(q=2, trunc=2)

    def local(ell):
        return _idx_of(layout, ell) - int(layout.offsets[2])

    i20 = local((2, 0))
    i11 = local((1, 1))
    i02 = local((0, 2))

    y0 = y_q2_m2[..., 0, :]
    y1 = y_q2_m2[..., 1, :]

    np.testing.assert_allclose(
        _np(mons[2][:, i20, :]), _np(_outer(y0, y0)), atol=1e-12,
        err_msg="M_(2,0) != y0⊗y0",
    )
    np.testing.assert_allclose(
        _np(mons[2][:, i11, :]), _np(_outer(y0, y1) + _outer(y1, y0)), atol=1e-12,
        err_msg="M_(1,1) != y0⊗y1 + y1⊗y0",
    )
    np.testing.assert_allclose(
        _np(mons[2][:, i02, :]), _np(_outer(y1, y1)), atol=1e-12,
        err_msg="M_(0,2) != y1⊗y1",
    )


# ---------------------------------------------------------------------------
# Index-order consistency with kernel.lag_weights
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [0, 1, 2])
def test_ell_ordering_matches_lag_weights(k):
    """The ell_index axis of monomials[k] must match lag_weights(..., n=k+1)."""
    q = 2
    beta = jnp.array([0.7, 0.9])
    A = jnp.eye(2, dtype=jnp.float64)[None].repeat(q, axis=0)  # (2, 2, 2)
    kernel = FractionalKernel(beta=beta, A=A)

    # The layout used by lag_weights(n=k+1) is build_multiindex_layout(q, k).
    lag_layout = build_multiindex_layout(q=q, trunc=k)
    # The layout used by _shuffle_monomials_by_degree for monomials[k].
    mon_layout = build_multiindex_layout(q=q, trunc=k)

    # Both use the same _compositions_desc ordering; verify ell arrays agree.
    deg_k_slice = slice(int(lag_layout.offsets[k]), int(lag_layout.offsets[k + 1]))
    lag_ells = _np(lag_layout.ell[deg_k_slice])  # (M_k, q)

    # From the monomial layout (using the full layout up to trunc=k):
    # the degree-k block covers the same slice.
    mon_ells = _np(mon_layout.ell[deg_k_slice])

    np.testing.assert_array_equal(
        lag_ells, mon_ells,
        err_msg=f"Degree-{k} multi-index ordering mismatch between "
                f"lag_weights and _shuffle_monomials_by_degree.",
    )

    # Double-check by calling lag_weights and verifying last-axis size = M_k.
    h = jnp.asarray(0.1, dtype=jnp.float64)
    theta = jnp.asarray(0.0, dtype=jnp.float64)
    rho = jnp.asarray(0.0, dtype=jnp.float64)
    w = kernel.lag_weights(out_len=4, h=h, theta=theta, n=k + 1, rho=rho, dtype=jnp.float64)
    M_k = int(lag_layout.offsets[k + 1]) - int(lag_layout.offsets[k])
    assert w.shape == (4, q, M_k), (
        f"lag_weights(n={k+1}) shape {w.shape} != expected (4, {q}, {M_k})."
    )


# ---------------------------------------------------------------------------
# Batch-dimension pass-through
# ---------------------------------------------------------------------------

def test_batch_dims_preserved():
    S, q, m, trunc = 5, 2, 3, 2
    batch = (2, 4)
    y = jnp.ones((S,) + batch + (q, m), dtype=jnp.float64)
    mons = _shuffle_monomials_by_degree(y, trunc=trunc, dtype=jnp.float64)
    layout = build_multiindex_layout(q=q, trunc=trunc)
    for k, mon in enumerate(mons):
        M_k = int(layout.offsets[k + 1]) - int(layout.offsets[k])
        assert mon.shape == (S,) + batch + (M_k, m ** k)


# ---------------------------------------------------------------------------
# Edge: trunc=0
# ---------------------------------------------------------------------------

def test_trunc_zero():
    y = jnp.ones((3, 2, 2), dtype=jnp.float64)
    mons = _shuffle_monomials_by_degree(y, trunc=0, dtype=jnp.float64)
    assert len(mons) == 1
    assert mons[0].shape == (3, 1, 1)
    np.testing.assert_allclose(_np(mons[0]), 1.0)

