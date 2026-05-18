from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import pytest

from tensordev.sss.lambdas import DenseLambda, JordanLambda


def _assert_allclose(x, y, *, atol=1e-10, rtol=1e-10):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    assert x.shape == y.shape, f"shape mismatch: {x.shape} vs {y.shape}"
    assert jnp.allclose(x, y, atol=atol, rtol=rtol), (x, y)


def _sample_jordan_lambda() -> JordanLambda:
    return JordanLambda(
        real_rates=jnp.asarray([0.7]),
        real_sizes=jnp.asarray([2]),
        osc_decays=jnp.asarray([0.3]),
        osc_freqs=jnp.asarray([1.1]),
        osc_sizes=jnp.asarray([1]),
    )


def _phi1_dense(matrix, dt):
    """Compute phi_1(-dt * M) via the augmented exponential."""
    matrix = jnp.asarray(matrix)
    dt = jnp.asarray(dt, dtype=matrix.dtype)
    r = matrix.shape[0]
    eye = jnp.eye(r, dtype=matrix.dtype)
    zero = jnp.zeros_like(matrix)

    def one(h):
        aug = jnp.block([[-h * matrix, eye], [zero, zero]])
        return jsp_linalg.expm(aug)[:r, r:]

    return one(dt) if dt.ndim == 0 else jax.vmap(one)(dt)


# ---------------------------------------------------------------------------
# DenseLambda tests
# ---------------------------------------------------------------------------


def test_dense_lambda_rejects_non_square_matrix():
    with pytest.raises(ValueError, match=r"shape \(R, R\)"):
        DenseLambda(jnp.ones((2, 3)))


@pytest.mark.parametrize("dt", [0.2, jnp.asarray([0.1, 0.3, 0.7])])
def test_dense_lambda_expm_matches_direct_expm(dt):
    matrix = jnp.asarray([[1.0, -0.2], [0.0, 0.5]])
    lam = DenseLambda(matrix)

    got = lam.expm(dt)
    dt_arr = jnp.asarray(dt)
    if dt_arr.ndim == 0:
        expected = jsp_linalg.expm(-dt_arr * matrix)
    else:
        expected = jax.vmap(lambda h: jsp_linalg.expm(-h * matrix))(dt_arr)
    _assert_allclose(got, expected)


@pytest.mark.parametrize("dt", [0.2, jnp.asarray([0.1, 0.3, 0.7])])
def test_dense_lambda_phi1_matches_dense_block_formula(dt):
    matrix = jnp.asarray([[1.0, -0.2], [0.0, 0.5]])
    lam = DenseLambda(matrix)

    expected = _phi1_dense(matrix, dt)
    _assert_allclose(lam.phi1(dt), expected)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dt", [0.2, jnp.asarray([0.1, 0.3, 0.7])])
def test_dense_lambda_shifted_solves_match_direct_linear_solve(dt, transpose):
    matrix = jnp.asarray([[1.0, -0.2], [0.0, 0.5]])
    zeta = 0.8 + 0.6j
    lam = DenseLambda(matrix)
    dt_arr = jnp.asarray(dt)
    batched = dt_arr.ndim == 1

    # rhs shape: (R, k) for scalar dt, (m, R, k) for batched dt
    rhs_base = jnp.asarray([[1.0 + 0.2j, -0.5j], [0.3 - 0.1j, 2.0 + 0.4j]])
    if batched:
        rhs = jnp.stack([rhs_base, 1.5 * rhs_base, -0.25 * rhs_base], axis=0)
    else:
        rhs = rhs_base

    solve = lam.solve_shifted_transpose if transpose else lam.solve_shifted
    got = solve(zeta, dt, rhs)

    mat = matrix.astype(rhs.dtype)
    if transpose:
        mat = mat.T

    def expected_one(h, r):
        system = zeta * jnp.eye(matrix.shape[0], dtype=rhs.dtype) + h * mat
        return jnp.linalg.solve(system, r)

    if batched:
        expected = jax.vmap(expected_one)(dt_arr, rhs)
    else:
        expected = expected_one(dt_arr, rhs)
    _assert_allclose(got, expected)


@pytest.mark.parametrize("batched_dt", [False, True])
def test_dense_lambda_operator_actions_match_dense_materialization(batched_dt):
    matrix = jnp.asarray([[1.0, -0.2], [0.0, 0.5]])
    lam = DenseLambda(matrix)

    rhs = jnp.asarray([[1.0, -0.5, 0.2], [0.3, 2.0, -0.7]])
    lhs = jnp.asarray([[1.0, -0.2], [0.3, 0.5], [-0.4, 1.2]])
    dt = jnp.asarray([0.1, 0.3, 0.7]) if batched_dt else 0.2

    if batched_dt:
        rhs = jnp.stack([rhs, 1.5 * rhs, -0.25 * rhs], axis=0)
        lhs = jnp.stack([lhs, -0.5 * lhs, 2.0 * lhs], axis=0)

        exp_left_expected = jax.vmap(lambda h, x: jsp_linalg.expm(-h * matrix) @ x)(dt, rhs)
        exp_right_expected = jax.vmap(lambda h, x: x @ jsp_linalg.expm(-h * matrix).T)(dt, lhs)
        phi_left_expected = jax.vmap(lambda h, x: _phi1_dense(matrix, h) @ x)(dt, rhs)
        phi_right_expected = jax.vmap(lambda h, x: x @ _phi1_dense(matrix, h).T)(dt, lhs)
    else:
        exp_left_expected = jsp_linalg.expm(-dt * matrix) @ rhs
        exp_right_expected = lhs @ jsp_linalg.expm(-dt * matrix).T
        phi_left_expected = _phi1_dense(matrix, dt) @ rhs
        phi_right_expected = lhs @ _phi1_dense(matrix, dt).T

    _assert_allclose(lam.lambda_multiply_left(rhs), matrix @ rhs)
    _assert_allclose(lam.lambda_multiply_right(lhs), lhs @ matrix.T)
    _assert_allclose(lam.expm_multiply_left(dt, rhs), exp_left_expected)
    _assert_allclose(lam.expm_multiply_right(dt, lhs), exp_right_expected)
    _assert_allclose(lam.phi1_multiply_left(dt, rhs), phi_left_expected)
    _assert_allclose(lam.phi1_multiply_right(dt, lhs), phi_right_expected)


# ---------------------------------------------------------------------------
# JordanLambda tests
# ---------------------------------------------------------------------------


def test_jordan_lambda_state_dim_and_matrix_shape():
    lam = _sample_jordan_lambda()
    assert lam.state_dim == 4
    assert lam.matrix().shape == (4, 4)


@pytest.mark.parametrize("dt", [0.2, jnp.asarray([0.1, 0.3, 0.7])])
def test_jordan_lambda_expm_matches_dense_materialization(dt):
    lam = _sample_jordan_lambda()
    dense = DenseLambda(lam.matrix())
    _assert_allclose(lam.expm(dt), dense.expm(dt))


@pytest.mark.parametrize("dt", [0.2, jnp.asarray([0.1, 0.3, 0.7])])
def test_jordan_lambda_phi1_matches_dense_materialization(dt):
    lam = _sample_jordan_lambda()
    expected = _phi1_dense(lam.matrix(), dt)
    _assert_allclose(lam.phi1(dt), expected)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dt", [0.2, jnp.asarray([0.1, 0.3, 0.7])])
def test_jordan_lambda_shifted_solves_match_dense_materialization(dt, transpose):
    lam = _sample_jordan_lambda()
    dense = DenseLambda(lam.matrix())
    dt_arr = jnp.asarray(dt)
    batched = dt_arr.ndim == 1

    # rhs shape: (R, k) — R=state_dim=4 on second-to-last axis
    rhs_base = jnp.asarray([[1.0 + 0.2j, -0.5j],
                             [0.3 - 0.1j, 2.0 + 0.4j],
                             [0.7 - 0.1j, 1.1 + 0.3j],
                             [-0.8 + 0.5j, 0.1 - 0.9j]])
    if batched:
        rhs = jnp.stack([rhs_base, 1.5 * rhs_base, -0.25 * rhs_base], axis=0)
    else:
        rhs = rhs_base

    zeta = 0.8 + 0.6j
    solve_struct = lam.solve_shifted_transpose if transpose else lam.solve_shifted
    solve_dense = dense.solve_shifted_transpose if transpose else dense.solve_shifted
    _assert_allclose(solve_struct(zeta, dt, rhs), solve_dense(zeta, dt, rhs))


@pytest.mark.parametrize("batched_dt", [False, True])
def test_jordan_lambda_operator_actions_match_dense_materialization(batched_dt):
    lam = _sample_jordan_lambda()
    dense_matrix = lam.matrix()
    r = lam.state_dim

    rhs = jnp.reshape(jnp.arange(1, 1 + r * 3, dtype=jnp.float64), (r, 3)) / 7.0
    lhs = jnp.reshape(jnp.arange(1, 1 + 2 * r, dtype=jnp.float64), (2, r)) / 5.0
    dt = jnp.asarray([0.1, 0.3, 0.7]) if batched_dt else 0.2

    if batched_dt:
        rhs = jnp.stack([rhs, -0.5 * rhs, 1.25 * rhs], axis=0)
        lhs = jnp.stack([lhs, 2.0 * lhs, -1.5 * lhs], axis=0)

    _assert_allclose(
        lam.lambda_multiply_left(rhs),
        dense_matrix @ rhs if not batched_dt else jax.vmap(lambda x: dense_matrix @ x)(rhs),
    )
    _assert_allclose(
        lam.lambda_multiply_right(lhs),
        lhs @ dense_matrix.T if not batched_dt else jax.vmap(lambda x: x @ dense_matrix.T)(lhs),
    )
    _assert_allclose(
        lam.expm_multiply_left(dt, rhs),
        jax.vmap(lambda h, x: jsp_linalg.expm(-h * dense_matrix) @ x)(dt, rhs)
        if batched_dt else jsp_linalg.expm(-dt * dense_matrix) @ rhs,
    )
    _assert_allclose(
        lam.expm_multiply_right(dt, lhs),
        jax.vmap(lambda h, x: x @ jsp_linalg.expm(-h * dense_matrix).T)(dt, lhs)
        if batched_dt else lhs @ jsp_linalg.expm(-dt * dense_matrix).T,
    )
    _assert_allclose(
        lam.phi1_multiply_left(dt, rhs),
        jax.vmap(lambda h, x: _phi1_dense(dense_matrix, h) @ x)(dt, rhs)
        if batched_dt else _phi1_dense(dense_matrix, dt) @ rhs,
    )
    _assert_allclose(
        lam.phi1_multiply_right(dt, lhs),
        jax.vmap(lambda h, x: x @ _phi1_dense(dense_matrix, h).T)(dt, lhs)
        if batched_dt else lhs @ _phi1_dense(dense_matrix, dt).T,
    )


def test_jordan_lambda_b_from_prony_builds_expected_shape():
    lam = JordanLambda(
        real_rates=jnp.asarray([0.5, 1.2]),
        real_sizes=jnp.asarray([2, 1]),
        osc_decays=jnp.asarray([0.3]),
        osc_freqs=jnp.asarray([2.0]),
        osc_sizes=jnp.asarray([2]),
    )
    alpha = jnp.asarray([[1.0, 0.4, -0.2], [0.5, -0.1, 0.3]])
    beta = jnp.asarray([[0.6, -0.7], [0.2, 0.1]])
    delta = jnp.asarray([[0.4, 0.2], [-0.5, 0.8]])
    b = lam.b_from_prony(alpha=alpha, beta=beta, delta=delta)
    assert b.shape == (2, lam.state_dim)


@pytest.mark.parametrize(
    "kwargs,pattern",
    [
        ({"real_rates": jnp.asarray([0.1]), "real_sizes": jnp.asarray([1, 2])}, "same length"),
        ({"osc_decays": jnp.asarray([0.1]), "osc_freqs": jnp.asarray([1.0]), "osc_sizes": jnp.asarray([0])}, "strictly positive"),
        ({"osc_decays": jnp.asarray([0.1]), "osc_freqs": jnp.asarray([0.0]), "osc_sizes": jnp.asarray([1])}, "strictly positive"),
    ],
)
def test_jordan_lambda_validates_block_parameters(kwargs, pattern):
    with pytest.raises(ValueError, match=pattern):
        JordanLambda(**kwargs)


# ---------------------------------------------------------------------------
# phi2 dense helper
# ---------------------------------------------------------------------------


def _phi2_dense(matrix, dt):
    """Compute phi_2(-dt * M) via the augmented exponential."""
    matrix = jnp.asarray(matrix)
    dt = jnp.asarray(dt, dtype=matrix.dtype)
    r = matrix.shape[0]
    eye = jnp.eye(r, dtype=matrix.dtype)
    zero = jnp.zeros_like(matrix)

    def one(h):
        aug = jnp.block([[-h * matrix, eye, zero], [zero, zero, eye], [zero, zero, zero]])
        return jsp_linalg.expm(aug)[:r, 2 * r:]

    return one(dt) if dt.ndim == 0 else jax.vmap(one)(dt)


# ---------------------------------------------------------------------------
# Parametrized phi1 / phi2 correctness tests for JordanLambda (real blocks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.1, 0.5, 1.5])
@pytest.mark.parametrize("size", [1, 2, 3, 4])
@pytest.mark.parametrize("dt_val", [0.01, 0.04, 0.1, 0.3, 1.0, 2.0])
def test_jordan_lambda_phi1_real_pade_vs_dense(dt_val, size, rate):
    """phi1 for real Jordan blocks matches DenseLambda for all |dt*rate| including small."""
    lam = JordanLambda(
        real_rates=jnp.asarray([rate]),
        real_sizes=jnp.asarray([size]),
    )
    dense = DenseLambda(lam.matrix())
    dt = jnp.asarray([dt_val])
    got = lam.phi1(dt)
    expected = _phi1_dense(lam.matrix(), dt)
    _assert_allclose(got, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("rate", [0.1, 0.5, 1.5])
@pytest.mark.parametrize("size", [1, 2, 3, 4])
@pytest.mark.parametrize("dt_val", [0.01, 0.04, 0.1, 0.3, 1.0, 2.0])
def test_jordan_lambda_phi2_real_pade_vs_dense(dt_val, size, rate):
    """phi2 for real Jordan blocks matches DenseLambda for all |dt*rate| including small."""
    lam = JordanLambda(
        real_rates=jnp.asarray([rate]),
        real_sizes=jnp.asarray([size]),
    )
    dense = DenseLambda(lam.matrix())
    dt = jnp.asarray([dt_val])
    got = lam.phi2(dt)
    expected = _phi2_dense(lam.matrix(), dt)
    _assert_allclose(got, expected, atol=1e-8, rtol=1e-8)


# ---------------------------------------------------------------------------
# Small |dt*rate| regime — the exact bug domain (previously untested)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.1, 0.5])
@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("dt_val", [0.001, 0.01, 0.04])
def test_jordan_lambda_phi1_small_dt_rate_vs_dense(dt_val, size, rate):
    """phi1 in small |dt*rate| regime (was buggy) now matches DenseLambda."""
    lam = JordanLambda(
        real_rates=jnp.asarray([rate]),
        real_sizes=jnp.asarray([size]),
    )
    dt = jnp.asarray([dt_val])
    got = lam.phi1(dt)
    expected = _phi1_dense(lam.matrix(), dt)
    _assert_allclose(got, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("rate", [0.1, 0.5])
@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("dt_val", [0.001, 0.01, 0.04])
def test_jordan_lambda_phi2_small_dt_rate_vs_dense(dt_val, size, rate):
    """phi2 in small |dt*rate| regime matches DenseLambda."""
    lam = JordanLambda(
        real_rates=jnp.asarray([rate]),
        real_sizes=jnp.asarray([size]),
    )
    dt = jnp.asarray([dt_val])
    got = lam.phi2(dt)
    expected = _phi2_dense(lam.matrix(), dt)
    _assert_allclose(got, expected, atol=1e-8, rtol=1e-8)


# ---------------------------------------------------------------------------
# Large Jordan block sizes (3, 4, 5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [3, 4, 5])
@pytest.mark.parametrize("dt_val", [0.1, 1.0])
def test_jordan_lambda_phi1_large_block_sizes(dt_val, size):
    """phi1 is correct for large Jordan block sizes."""
    lam = JordanLambda(
        real_rates=jnp.asarray([0.5]),
        real_sizes=jnp.asarray([size]),
    )
    dt = jnp.asarray([dt_val])
    got = lam.phi1(dt)
    expected = _phi1_dense(lam.matrix(), dt)
    _assert_allclose(got, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("size", [3, 4, 5])
@pytest.mark.parametrize("dt_val", [0.1, 1.0])
def test_jordan_lambda_phi2_large_block_sizes(dt_val, size):
    """phi2 is correct for large Jordan block sizes."""
    lam = JordanLambda(
        real_rates=jnp.asarray([0.5]),
        real_sizes=jnp.asarray([size]),
    )
    dt = jnp.asarray([dt_val])
    got = lam.phi2(dt)
    expected = _phi2_dense(lam.matrix(), dt)
    _assert_allclose(got, expected, atol=1e-8, rtol=1e-8)


# ---------------------------------------------------------------------------
# Oscillatory phi1 / phi2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("osc_size", [1, 2, 3])
@pytest.mark.parametrize("dt_val", [0.01, 0.1, 1.0])
def test_jordan_lambda_phi1_oscillatory_vs_dense(dt_val, osc_size):
    """phi1 for oscillatory blocks matches DenseLambda."""
    lam = JordanLambda(
        osc_decays=jnp.asarray([0.3]),
        osc_freqs=jnp.asarray([1.1]),
        osc_sizes=jnp.asarray([osc_size]),
    )
    dt = jnp.asarray([dt_val])
    got = lam.phi1(dt)
    expected = _phi1_dense(lam.matrix(), dt)
    _assert_allclose(got, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("osc_size", [1, 2, 3])
@pytest.mark.parametrize("dt_val", [0.01, 0.1, 1.0])
def test_jordan_lambda_phi2_oscillatory_vs_dense(dt_val, osc_size):
    """phi2 for oscillatory blocks matches DenseLambda."""
    lam = JordanLambda(
        osc_decays=jnp.asarray([0.3]),
        osc_freqs=jnp.asarray([1.1]),
        osc_sizes=jnp.asarray([osc_size]),
    )
    dt = jnp.asarray([dt_val])
    got = lam.phi2(dt)
    expected = _phi2_dense(lam.matrix(), dt)
    _assert_allclose(got, expected, atol=1e-8, rtol=1e-8)


# ---------------------------------------------------------------------------
# phi2 was never tested for the mixed real+osc sample — add it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", [0.2, jnp.asarray([0.1, 0.3, 0.7])])
def test_jordan_lambda_phi2_matches_dense_materialization(dt):
    """phi2 for the standard mixed JordanLambda matches DenseLambda."""
    lam = _sample_jordan_lambda()
    got = lam.phi2(dt)
    expected = _phi2_dense(lam.matrix(), dt)
    _assert_allclose(got, expected, atol=1e-8, rtol=1e-8)
