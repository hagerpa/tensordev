"""
Tests for the FSSK signature kernel:

1. **Lambda = 0 vs standard sig kernel** — When Lambda is the zero matrix the
   FSSK kernel reduces to the classical linear signature kernel.  We check that
   ``fssk_sigkernel`` with Lambda=0 closely matches ``SigKernel`` (which
   dispatches to ``free_kernel``) at high dyadic order.

2. **Dense vs Jordan consistency** — ``DenseLambda`` and ``JordanLambda``
   realizations of the same kernel must agree to machine precision.
"""

from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax.numpy as jnp
import jax.random as jr

from tensordev.kernel.fssk import FSSKSigKernel, fssk_sigkernel
from tensordev.kernel.sig import SigKernel
from tensordev.kernel.static_kernels import RBFKernel
from tensordev.sss.kernel import FSSK

from random_paths import random_trigonometric_polynomial_paths


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _identity_fssk_kernel_dense(dim, *, dtype=jnp.float64):
    """FSSK kernel with Lambda=0, A=I_d, b=1  (DenseLambda)."""
    Lambda = jnp.zeros((1, 1), dtype=dtype)
    A = jnp.eye(dim, dtype=dtype)[None, :, :]  # (1, d, d)
    b = jnp.ones((1, 1), dtype=dtype)  # (1, 1)
    return FSSK.from_matrix(Lambda=Lambda, A=A, b=b)


def _identity_fssk_kernel_jordan(dim, *, dtype=jnp.float64):
    """FSSK kernel with Lambda=0, A=I_d, b=1  (JordanLambda)."""
    A = jnp.eye(dim, dtype=dtype)[None, :, :]
    b = jnp.ones((1, 1), dtype=dtype)
    return FSSK.from_jordan(
        A=A, b=b,
        real_rates=jnp.array([0.0], dtype=dtype),
        real_sizes=jnp.array([1]),
    )


def _nontrivial_fssk_kernel_dense(dim, *, R=2, dtype=jnp.float64):
    """A small non-trivial FSSK kernel with DenseLambda."""
    key = jr.PRNGKey(9999)
    k1, k2, k3 = jr.split(key, 3)
    # Make Lambda positive semi-definite
    L = jr.normal(k1, (R, R), dtype=dtype)
    Lambda = L @ L.T + 0.1 * jnp.eye(R, dtype=dtype)
    A = jr.normal(k2, (1, dim, dim), dtype=dtype) * 0.5
    b = jr.normal(k3, (1, R), dtype=dtype)
    return FSSK.from_matrix(Lambda=Lambda, A=A, b=b)


def _nontrivial_fssk_kernel_jordan(dim, *, dtype=jnp.float64):
    """Same kernel spec but via JordanLambda (single real block, R=2)."""
    key = jr.PRNGKey(9999)
    k1, k2, k3 = jr.split(key, 3)
    # Use two distinct real rates
    A = jr.normal(k2, (1, dim, dim), dtype=dtype) * 0.5
    b = jr.normal(k3, (1, 2), dtype=dtype)
    return FSSK.from_jordan(
        A=A, b=b,
        real_rates=jnp.array([0.5, 1.5], dtype=dtype),
        real_sizes=jnp.array([1, 1]),
    )


def _random_paths(key, batch, steps, dim):
    return random_trigonometric_polynomial_paths(
        key, batch=batch, steps=steps, dim=dim,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Lambda = 0 vs standard sig kernel
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dyadic_order", [3, 4])
def test_fssk_lambda_zero_terminal_matches_sigkernel(dim, dyadic_order):
    """Terminal FSSK with Lambda=0 should match the classical sig kernel."""
    key = jr.PRNGKey(1000 + dim)
    X = _random_paths(key, batch=3, steps=16, dim=dim)
    Y = _random_paths(jr.fold_in(key, 1), batch=4, steps=14, dim=dim)

    dt = 1.0 / X.shape[-2]

    kernel = _identity_fssk_kernel_dense(dim)
    fssk_out = fssk_sigkernel(
        X, Y, kernel=kernel, dt_x=dt, dt_y=dt,
        evaluate="terminal", pairwise=True,
        backend="scan", scheme="heun", dyadic_order=dyadic_order,
    )

    sig = SigKernel(backend="scan", dyadic_order=dyadic_order)
    sig_out = sig(X, Y, evaluate="terminal", pairwise=True)

    np.testing.assert_allclose(
        np.asarray(fssk_out), np.asarray(sig_out),
        rtol=5e-3, atol=5e-3,
        err_msg=f"Lambda=0 terminal dim={dim} dy={dyadic_order}",
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_fssk_lambda_zero_gram_matches_sigkernel(dim):
    """Gram matrix from FSSK Lambda=0 should match SigKernel Gram."""
    key = jr.PRNGKey(2000 + dim)
    X = _random_paths(key, batch=3, steps=12, dim=dim)
    Y = _random_paths(jr.fold_in(key, 1), batch=4, steps=10, dim=dim)
    dyadic_order = 4
    dt = 1.0 / X.shape[-2]

    fssk_ker = FSSKSigKernel(
        kernel=_identity_fssk_kernel_dense(dim),
        dt_x=dt, dt_y=dt,
        backend="scan", scheme="heun", dyadic_order=dyadic_order,
    )
    fssk_gram = fssk_ker.compute_Gram(X, Y, sym=False)

    sig_ker = SigKernel(backend="scan", dyadic_order=dyadic_order)
    sig_gram = sig_ker.compute_Gram(X, Y, sym=False)

    np.testing.assert_allclose(
        np.asarray(fssk_gram), np.asarray(sig_gram),
        rtol=5e-3, atol=5e-3,
        err_msg=f"Lambda=0 Gram dim={dim}",
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_fssk_lambda_zero_grid_terminal_matches_sigkernel(dim):
    """Terminal corner of FSSK Lambda=0 grid should match SigKernel terminal."""
    key = jr.PRNGKey(3000 + dim)
    X = _random_paths(key, batch=1, steps=12, dim=dim)[0]  # single path
    Y = _random_paths(jr.fold_in(key, 1), batch=1, steps=10, dim=dim)[0]
    dyadic_order = 4
    dt = 1.0 / X.shape[-2]

    kernel = _identity_fssk_kernel_dense(dim)
    # Use terminal evaluation for a clean comparison
    fssk_term = fssk_sigkernel(
        X, Y, kernel=kernel, dt_x=dt, dt_y=dt,
        evaluate="terminal", pairwise=False,
        backend="scan", scheme="heun", dyadic_order=dyadic_order,
    )

    sig = SigKernel(backend="scan", dyadic_order=dyadic_order)
    sig_term = sig(X, Y, evaluate="terminal")

    np.testing.assert_allclose(
        np.asarray(fssk_term), np.asarray(sig_term),
        rtol=5e-3, atol=5e-3,
        err_msg=f"Lambda=0 grid terminal dim={dim}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Dense vs Jordan consistency
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dyadic_order", [0, 1, 2])
def test_dense_vs_jordan_lambda_zero_terminal(dim, dyadic_order):
    """DenseLambda(0) and JordanLambda(rate=0) must give identical terminals."""
    key = jr.PRNGKey(4000 + dim + dyadic_order)
    X = _random_paths(key, batch=3, steps=16, dim=dim)
    Y = _random_paths(jr.fold_in(key, 1), batch=4, steps=14, dim=dim)
    dt = 1.0 / X.shape[-2]

    dense_k = _identity_fssk_kernel_dense(dim)
    jordan_k = _identity_fssk_kernel_jordan(dim)

    out_dense = fssk_sigkernel(
        X, Y, kernel=dense_k, dt_x=dt, dt_y=dt,
        evaluate="terminal", pairwise=True,
        backend="scan", scheme="heun", dyadic_order=dyadic_order,
    )
    out_jordan = fssk_sigkernel(
        X, Y, kernel=jordan_k, dt_x=dt, dt_y=dt,
        evaluate="terminal", pairwise=True,
        backend="scan", scheme="heun", dyadic_order=dyadic_order,
    )

    np.testing.assert_allclose(
        np.asarray(out_dense), np.asarray(out_jordan),
        rtol=1e-10, atol=1e-10,
        err_msg=f"dense vs jordan Lambda=0 dim={dim} dy={dyadic_order}",
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dyadic_order", [0, 1])
def test_dense_vs_jordan_nontrivial_terminal(dim, dyadic_order):
    """DenseLambda and JordanLambda with the same diagonal rates must agree."""
    key = jr.PRNGKey(5000 + dim + dyadic_order)
    X = _random_paths(key, batch=3, steps=16, dim=dim)
    Y = _random_paths(jr.fold_in(key, 1), batch=4, steps=14, dim=dim)
    dt = 1.0 / X.shape[-2]

    # Build equivalent Dense and Jordan kernels with diagonal Lambda
    rates = jnp.array([0.5, 1.5], dtype=jnp.float64)
    Lambda_mat = jnp.diag(rates)

    rng = jr.PRNGKey(7777)
    k1, k2 = jr.split(rng)
    A = jr.normal(k1, (1, dim, dim), dtype=jnp.float64) * 0.3
    b = jr.normal(k2, (1, 2), dtype=jnp.float64)

    dense_k = FSSK.from_matrix(Lambda=Lambda_mat, A=A, b=b)
    jordan_k = FSSK.from_jordan(
        A=A, b=b,
        real_rates=rates,
        real_sizes=jnp.array([1, 1]),
    )

    out_dense = fssk_sigkernel(
        X, Y, kernel=dense_k, dt_x=dt, dt_y=dt,
        evaluate="terminal", pairwise=True,
        backend="scan", scheme="heun", dyadic_order=dyadic_order,
    )
    out_jordan = fssk_sigkernel(
        X, Y, kernel=jordan_k, dt_x=dt, dt_y=dt,
        evaluate="terminal", pairwise=True,
        backend="scan", scheme="heun", dyadic_order=dyadic_order,
    )

    np.testing.assert_allclose(
        np.asarray(out_dense), np.asarray(out_jordan),
        rtol=1e-8, atol=1e-8,
        err_msg=f"dense vs jordan nontrivial dim={dim} dy={dyadic_order}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Backend consistency (scan vs wavefront) for FSSK
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dyadic_order", [0, 1])
@pytest.mark.parametrize("scheme", ["heun", "etd1"])
def test_fssk_scan_vs_wavefront_terminal(dim, dyadic_order, scheme):
    """Scan and wavefront backends must agree for FSSK terminal values."""
    key = jr.PRNGKey(6000 + dim + dyadic_order)
    X = _random_paths(key, batch=3, steps=12, dim=dim)
    Y = _random_paths(jr.fold_in(key, 1), batch=4, steps=10, dim=dim)
    dt = 1.0 / X.shape[-2]

    kernel = _identity_fssk_kernel_dense(dim)

    results = {}
    for backend in ("scan", "wavefront"):
        results[backend] = fssk_sigkernel(
            X, Y, kernel=kernel, dt_x=dt, dt_y=dt,
            evaluate="terminal", pairwise=True,
            backend=backend, scheme=scheme, dyadic_order=dyadic_order,
        )

    np.testing.assert_allclose(
        np.asarray(results["scan"]), np.asarray(results["wavefront"]),
        rtol=1e-12, atol=1e-12,
        err_msg=f"FSSK scan vs wavefront dim={dim} dy={dyadic_order} {scheme}",
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dyadic_order", [0, 1])
def test_fssk_scan_vs_wavefront_grid(dim, dyadic_order):
    """Scan and wavefront backends must agree for FSSK grid output."""
    key = jr.PRNGKey(7000 + dim + dyadic_order)
    X = _random_paths(key, batch=1, steps=10, dim=dim)[0]
    Y = _random_paths(jr.fold_in(key, 1), batch=1, steps=8, dim=dim)[0]
    dt = 1.0 / X.shape[-2]

    kernel = _identity_fssk_kernel_dense(dim)

    results = {}
    for backend in ("scan", "wavefront"):
        results[backend] = fssk_sigkernel(
            X, Y, kernel=kernel, dt_x=dt, dt_y=dt,
            evaluate="grid", pairwise=False,
            backend=backend, scheme="heun", dyadic_order=dyadic_order,
        )

    np.testing.assert_allclose(
        np.asarray(results["scan"]), np.asarray(results["wavefront"]),
        rtol=1e-12, atol=1e-12,
        err_msg=f"FSSK grid scan vs wavefront dim={dim} dy={dyadic_order}",
    )


@pytest.mark.parametrize("dyadic_order", [0, 1])
def test_fssk_scan_vs_wavefront_nontrivial_kernel(dyadic_order):
    """Scan vs wavefront with a non-trivial Lambda."""
    dim = 2
    key = jr.PRNGKey(8000 + dyadic_order)
    X = _random_paths(key, batch=3, steps=12, dim=dim)
    Y = _random_paths(jr.fold_in(key, 1), batch=4, steps=10, dim=dim)
    dt = 1.0 / X.shape[-2]

    rates = jnp.array([0.5, 1.5], dtype=jnp.float64)
    rng = jr.PRNGKey(7777)
    k1, k2 = jr.split(rng)
    A = jr.normal(k1, (1, dim, dim), dtype=jnp.float64) * 0.3
    b = jr.normal(k2, (1, 2), dtype=jnp.float64)
    kernel = FSSK.from_jordan(
        A=A, b=b,
        real_rates=rates,
        real_sizes=jnp.array([1, 1]),
    )

    results = {}
    for backend in ("scan", "wavefront"):
        results[backend] = fssk_sigkernel(
            X, Y, kernel=kernel, dt_x=dt, dt_y=dt,
            evaluate="terminal", pairwise=True,
            backend=backend, scheme="heun", dyadic_order=dyadic_order,
        )

    np.testing.assert_allclose(
        np.asarray(results["scan"]), np.asarray(results["wavefront"]),
        rtol=1e-12, atol=1e-12,
        err_msg=f"FSSK nontrivial scan vs wavefront dy={dyadic_order}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Nonlinear static kernel tests
# ═══════════════════════════════════════════════════════════════════════════

_NONLINEAR_STATIC_KERNELS = [
    pytest.param(RBFKernel(sigma=1.0), id="RBF-sigma1"),
    pytest.param(RBFKernel(sigma=0.5), id="RBF-sigma0.5"),
]


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dyadic_order", [2, 3])
@pytest.mark.parametrize("static_kernel", _NONLINEAR_STATIC_KERNELS)
def test_fssk_lambda_zero_static_kernel_matches_sigkernel(dim, dyadic_order, static_kernel):
    """Lambda=0 FSSK with a nonlinear static kernel must match SigKernel using
    the same static kernel."""
    key = jr.PRNGKey(10000 + dim + dyadic_order)
    X = _random_paths(key, batch=3, steps=16, dim=dim)
    Y = _random_paths(jr.fold_in(key, 1), batch=4, steps=14, dim=dim)
    dt = 1.0 / X.shape[-2]

    kernel = _identity_fssk_kernel_dense(dim)
    fssk_out = fssk_sigkernel(
        X, Y, kernel=kernel, dt_x=dt, dt_y=dt,
        evaluate="terminal", pairwise=True,
        backend="scan", scheme="heun", dyadic_order=dyadic_order,
        static_kernel=static_kernel,
    )

    sig_out = SigKernel(
        backend="scan", dyadic_order=dyadic_order, static_kernel=static_kernel,
    )(X, Y, evaluate="terminal", pairwise=True)

    np.testing.assert_allclose(
        np.asarray(fssk_out), np.asarray(sig_out),
        rtol=5e-3, atol=5e-3,
        err_msg=f"Lambda=0 static_kernel terminal dim={dim} dy={dyadic_order}",
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dyadic_order", [0, 1])
@pytest.mark.parametrize("static_kernel", _NONLINEAR_STATIC_KERNELS)
def test_fssk_scan_vs_wavefront_nonlinear_static_kernel(dim, dyadic_order, static_kernel):
    """Scan and wavefront backends must agree when using a nonlinear static kernel."""
    key = jr.PRNGKey(11000 + dim + dyadic_order)
    X = _random_paths(key, batch=3, steps=12, dim=dim)
    Y = _random_paths(jr.fold_in(key, 1), batch=4, steps=10, dim=dim)
    dt = 1.0 / X.shape[-2]

    kernel = _identity_fssk_kernel_dense(dim)

    results = {}
    for backend in ("scan", "wavefront"):
        results[backend] = fssk_sigkernel(
            X, Y, kernel=kernel, dt_x=dt, dt_y=dt,
            evaluate="terminal", pairwise=True,
            backend=backend, scheme="heun", dyadic_order=dyadic_order,
            static_kernel=static_kernel,
        )

    np.testing.assert_allclose(
        np.asarray(results["scan"]), np.asarray(results["wavefront"]),
        rtol=1e-12, atol=1e-12,
        err_msg=f"FSSK static_kernel scan vs wavefront dim={dim} dy={dyadic_order}",
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dyadic_order", [0, 1])
@pytest.mark.parametrize("static_kernel", _NONLINEAR_STATIC_KERNELS)
def test_fssk_dense_vs_jordan_nonlinear_static_kernel(dim, dyadic_order, static_kernel):
    """Dense and Jordan Lambda representations must agree with a nonlinear static kernel.

    The static kernel enters only through G_ij and is independent of the
    Lambda/FSSK ODE structure — Dense and Jordan must therefore produce
    bitwise-identical results regardless of static_kernel choice.
    """
    key = jr.PRNGKey(12000 + dim + dyadic_order)
    X = _random_paths(key, batch=3, steps=16, dim=dim)
    Y = _random_paths(jr.fold_in(key, 1), batch=4, steps=14, dim=dim)
    dt = 1.0 / X.shape[-2]

    dense_k = _identity_fssk_kernel_dense(dim)
    jordan_k = _identity_fssk_kernel_jordan(dim)

    out_dense = fssk_sigkernel(
        X, Y, kernel=dense_k, dt_x=dt, dt_y=dt,
        evaluate="terminal", pairwise=True,
        backend="scan", scheme="heun", dyadic_order=dyadic_order,
        static_kernel=static_kernel,
    )
    out_jordan = fssk_sigkernel(
        X, Y, kernel=jordan_k, dt_x=dt, dt_y=dt,
        evaluate="terminal", pairwise=True,
        backend="scan", scheme="heun", dyadic_order=dyadic_order,
        static_kernel=static_kernel,
    )

    np.testing.assert_allclose(
        np.asarray(out_dense), np.asarray(out_jordan),
        rtol=1e-10, atol=1e-10,
        err_msg=f"FSSK dense vs jordan static_kernel dim={dim} dy={dyadic_order}",
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("static_kernel", _NONLINEAR_STATIC_KERNELS)
def test_fssk_static_kernel_gram_consistency(dim, static_kernel):
    """FSSKSigKernel.compute_Gram with a nonlinear static kernel must agree with
    the pairwise loop over fssk_sigkernel."""
    key = jr.PRNGKey(13000 + dim)
    X = _random_paths(key, batch=3, steps=12, dim=dim)
    Y = _random_paths(jr.fold_in(key, 1), batch=4, steps=10, dim=dim)
    dyadic_order = 2
    dt = 1.0 / X.shape[-2]

    fssk_ker = FSSKSigKernel(
        kernel=_identity_fssk_kernel_dense(dim),
        dt_x=dt, dt_y=dt,
        backend="scan", scheme="heun", dyadic_order=dyadic_order,
        static_kernel=static_kernel,
    )
    gram = fssk_ker.compute_Gram(X, Y, sym=False)

    # Reference: pairwise fssk_sigkernel call
    ref = fssk_sigkernel(
        X, Y,
        kernel=_identity_fssk_kernel_dense(dim),
        dt_x=dt, dt_y=dt,
        evaluate="terminal", pairwise=True,
        backend="scan", scheme="heun", dyadic_order=dyadic_order,
        static_kernel=static_kernel,
    )

    np.testing.assert_allclose(
        np.asarray(gram), np.asarray(ref),
        rtol=1e-10, atol=1e-10,
        err_msg=f"FSSK compute_Gram vs pairwise dim={dim}",
    )
