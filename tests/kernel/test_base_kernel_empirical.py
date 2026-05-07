"""
Tests for BaseKernel empirical helpers (compute_kernel, compute_Gram,
compute_mmd, compute_scoring_rule, compute_expected_scoring_rule) exercised on
every concrete kernel wrapper:

  * SigKernel
  * FreeKernel
  * HigherOrderKernel
  * FSSKSigKernel

Checks verified per class
--------------------------
1. compute_Gram(sym=True)  matches  compute_Gram(sym=False) — symmetry
2. max_batch chunking produces the same result as no chunking
3. compute_mmd >= 0 (positive semi-definiteness sanity)
4. compute_scoring_rule / compute_expected_scoring_rule smoke-test (no crash,
   finite output)
"""

from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax.numpy as jnp
import jax.random as jr

from tensordev.kernel.sig import SigKernel
from tensordev.kernel.free import FreeKernel
from tensordev.kernel.higher_order import HigherOrderKernel
from tensordev.kernel.fssk import FSSKSigKernel
from tensordev.sss.kernel import FSSK

from random_paths import random_trigonometric_polynomial_paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_fssk(dim, *, dtype=jnp.float64):
    Lambda = jnp.zeros((1, 1), dtype=dtype)
    A = jnp.eye(dim, dtype=dtype)[None, :, :]
    b = jnp.ones((1, 1), dtype=dtype)
    return FSSK.from_matrix(Lambda=Lambda, A=A, b=b)


def _paths(key, batch, steps, dim):
    return random_trigonometric_polynomial_paths(key, batch=batch, steps=steps, dim=dim)


def _paths_as_increments(key, batch, steps, dim):
    """Return paths as a 1-tuple of increment arrays for FreeKernel."""
    X = _paths(key, batch=batch, steps=steps, dim=dim)
    dX = jnp.diff(X, axis=-2)   # (batch, steps-1, dim)
    return (dX,)


# ---------------------------------------------------------------------------
# Fixtures: (kernel_instance, X_batch, Y_batch)
# ---------------------------------------------------------------------------

DIM = 2
STEPS = 10
BATCH_X = 4
BATCH_Y = 5


@pytest.fixture(scope="module")
def sig_setup():
    key = jr.PRNGKey(1001)
    X = _paths(key, batch=BATCH_X, steps=STEPS, dim=DIM)
    Y = _paths(jr.fold_in(key, 1), batch=BATCH_Y, steps=STEPS, dim=DIM)
    kernel = SigKernel(dyadic_order=2, backend="scan")
    return kernel, X, Y


@pytest.fixture(scope="module")
def free_setup():
    key = jr.PRNGKey(2001)
    X = _paths_as_increments(key, batch=BATCH_X, steps=STEPS, dim=DIM)
    Y = _paths_as_increments(jr.fold_in(key, 1), batch=BATCH_Y, steps=STEPS, dim=DIM)
    kernel = FreeKernel(dyadic_order=1, increment_input=True)
    return kernel, X, Y


@pytest.fixture(scope="module")
def hok_setup():
    key = jr.PRNGKey(3001)
    X = _paths(key, batch=BATCH_X, steps=STEPS, dim=DIM)
    Y = _paths(jr.fold_in(key, 1), batch=BATCH_Y, steps=STEPS, dim=DIM)
    dX = jnp.diff(X, axis=-2)   # (batch, steps-1, dim)
    dY = jnp.diff(Y, axis=-2)
    kernel = HigherOrderKernel(log_steps=(5, 5), log_degree=(2, 2), increment_input=True)
    return kernel, dX, dY


@pytest.fixture(scope="module")
def fssk_setup():
    key = jr.PRNGKey(4001)
    X = _paths(key, batch=BATCH_X, steps=STEPS, dim=DIM)
    Y = _paths(jr.fold_in(key, 1), batch=BATCH_Y, steps=STEPS, dim=DIM)
    dt = 1.0 / STEPS
    kernel = FSSKSigKernel(
        kernel=_identity_fssk(DIM),
        dt_x=dt, dt_y=dt,
        backend="scan", dyadic_order=2,
    )
    return kernel, X, Y


ALL_SETUPS = ["sig_setup", "free_setup", "hok_setup", "fssk_setup"]


# ---------------------------------------------------------------------------
# 1. compute_Gram: sym=True is symmetric and matches sym=False
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("setup_name", ALL_SETUPS)
def test_compute_gram_sym_matches_nonsym(setup_name, request):
    kernel, X, Y = request.getfixturevalue(setup_name)

    G_nonsym = kernel.compute_Gram(X, Y, sym=False, max_batch=50)
    G_sym_xx = kernel.compute_Gram(X, sym=True, max_batch=50)

    # sym=True with X,X should produce a symmetric matrix
    np.testing.assert_allclose(
        np.asarray(G_sym_xx),
        np.asarray(G_sym_xx).T,
        rtol=1e-10, atol=1e-10,
        err_msg=f"{setup_name}: compute_Gram(sym=True) not symmetric",
    )

    # sym=False with X,X should equal sym=True with X,X
    G_nonsym_xx = kernel.compute_Gram(X, X, sym=False, max_batch=50)
    np.testing.assert_allclose(
        np.asarray(G_nonsym_xx),
        np.asarray(G_sym_xx),
        rtol=1e-10, atol=1e-10,
        err_msg=f"{setup_name}: sym and non-sym paths disagree for X,X",
    )

    # Shape check for X,Y gram
    assert G_nonsym.shape == (BATCH_X, BATCH_Y), (
        f"{setup_name}: unexpected Gram shape {G_nonsym.shape}"
    )


# ---------------------------------------------------------------------------
# 2. max_batch chunking consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("setup_name", ALL_SETUPS)
def test_max_batch_chunking_consistency(setup_name, request):
    kernel, X, Y = request.getfixturevalue(setup_name)

    # Full batch (no chunking)
    G_full = kernel.compute_Gram(X, Y, sym=False, max_batch=None)

    # Chunked (max_batch=2 forces multiple chunks)
    G_chunked = kernel.compute_Gram(X, Y, sym=False, max_batch=2)

    np.testing.assert_allclose(
        np.asarray(G_chunked),
        np.asarray(G_full),
        rtol=1e-10, atol=1e-10,
        err_msg=f"{setup_name}: chunked Gram differs from full Gram",
    )


@pytest.mark.parametrize("setup_name", ALL_SETUPS)
def test_max_batch_sym_chunking_consistency(setup_name, request):
    kernel, X, _ = request.getfixturevalue(setup_name)

    G_full = kernel.compute_Gram(X, sym=True, max_batch=None)
    G_chunked = kernel.compute_Gram(X, sym=True, max_batch=2)

    np.testing.assert_allclose(
        np.asarray(G_chunked),
        np.asarray(G_full),
        rtol=1e-10, atol=1e-10,
        err_msg=f"{setup_name}: chunked sym Gram differs from full sym Gram",
    )


# ---------------------------------------------------------------------------
# 3. compute_mmd matches manual formula
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("setup_name", ALL_SETUPS)
def test_compute_mmd_matches_formula(setup_name, request):
    """compute_mmd must equal E_offdiag[Kxx] + E_offdiag[Kyy] - 2*mean(Kxy)."""
    kernel, X, Y = request.getfixturevalue(setup_name)

    Kxx = kernel.compute_Gram(X, sym=True, max_batch=50)
    Kyy = kernel.compute_Gram(Y, sym=True, max_batch=50)
    Kxy = kernel.compute_Gram(X, Y, sym=False, max_batch=50)

    n_x = Kxx.shape[0]
    n_y = Kyy.shape[0]
    offdiag_xx = (jnp.sum(Kxx) - jnp.sum(jnp.diag(Kxx))) / (n_x * (n_x - 1))
    offdiag_yy = (jnp.sum(Kyy) - jnp.sum(jnp.diag(Kyy))) / (n_y * (n_y - 1))
    expected = offdiag_xx + offdiag_yy - 2.0 * jnp.mean(Kxy)

    mmd = kernel.compute_mmd(X, Y, max_batch=50)

    np.testing.assert_allclose(
        float(mmd), float(expected),
        rtol=1e-10, atol=1e-10,
        err_msg=f"{setup_name}: compute_mmd differs from manual formula",
    )


@pytest.mark.parametrize("setup_name", ALL_SETUPS)
def test_compute_mmd_finite(setup_name, request):
    kernel, X, Y = request.getfixturevalue(setup_name)
    mmd = kernel.compute_mmd(X, Y, max_batch=50)
    assert jnp.isfinite(mmd), f"{setup_name}: MMD is not finite"


# ---------------------------------------------------------------------------
# 4. compute_scoring_rule and compute_expected_scoring_rule smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("setup_name", ALL_SETUPS)
def test_compute_scoring_rule_smoke(setup_name, request):
    kernel, X, Y = request.getfixturevalue(setup_name)

    # Extract a single sample y from Y (handles arrays and tuples)
    if isinstance(Y, tuple):
        y = tuple(level[:1] for level in Y)
    else:
        y = Y[:1]

    sr = kernel.compute_scoring_rule(X, y, max_batch=50)
    assert jnp.isfinite(sr), f"{setup_name}: scoring rule is not finite"


@pytest.mark.parametrize("setup_name", ALL_SETUPS)
def test_compute_expected_scoring_rule_smoke(setup_name, request):
    kernel, X, Y = request.getfixturevalue(setup_name)
    esr = kernel.compute_expected_scoring_rule(X, Y, max_batch=50)
    assert jnp.isfinite(esr), f"{setup_name}: expected scoring rule is not finite"


# ---------------------------------------------------------------------------
# 5. FSSKSigKernel-specific: compute_Gram sym=True matches sym=False
#    (specifically exercises the _dispatch path added by the refactor)
# ---------------------------------------------------------------------------

def test_fssk_sym_gram_matches_nonsym(fssk_setup):
    kernel, X, _ = fssk_setup

    G_sym = kernel.compute_Gram(X, sym=True, max_batch=50)
    G_nonsym = kernel.compute_Gram(X, X, sym=False, max_batch=50)

    np.testing.assert_allclose(
        np.asarray(G_sym),
        np.asarray(G_nonsym),
        rtol=1e-10, atol=1e-10,
        err_msg="FSSKSigKernel sym=True vs sym=False disagree",
    )


def test_fssk_compute_mmd_finite(fssk_setup):
    kernel, X, Y = fssk_setup
    mmd = kernel.compute_mmd(X, Y, max_batch=50)
    assert jnp.isfinite(mmd), "FSSKSigKernel MMD is not finite"


# ---------------------------------------------------------------------------
# 6. HigherOrderKernel: call-time increment_input override
# ---------------------------------------------------------------------------

def test_hok_call_time_increment_input_agrees_with_ctor(hok_setup):
    """
    Passing increment_input at call time should give the exact same result as
    setting it at construction time (the notebook pattern).
    """
    ctor_kernel, dX, dY = hok_setup   # ctor_kernel has increment_input=True

    # Use equal-sized sub-batches for batchwise (non-pairwise) evaluation
    dX_sub = dX[:BATCH_X]
    dY_sub = dY[:BATCH_X]

    # Build an equivalent kernel WITHOUT increment_input in ctor, pass it at call time
    call_kernel = HigherOrderKernel(
        log_steps=ctor_kernel.log_steps,
        log_degree=ctor_kernel.log_degree,
        backend=ctor_kernel.backend,
        dyadic_order=ctor_kernel.dyadic_order,
        increment_input=False,   # default — will be overridden at call time
    )

    out_ctor = ctor_kernel(dX_sub, dY_sub, evaluate="terminal", pairwise=False)
    out_call = call_kernel(dX_sub, dY_sub, evaluate="terminal", pairwise=False, increment_input=True)

    np.testing.assert_allclose(
        np.asarray(out_call),
        np.asarray(out_ctor),
        rtol=1e-12, atol=1e-12,
        err_msg="call-time increment_input=True disagrees with ctor increment_input=True",
    )






