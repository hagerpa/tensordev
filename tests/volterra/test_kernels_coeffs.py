from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tensordev.volterra import VolterraCoefficients, VolterraKernel
from tensordev.volterra.coeffs import validate_volterra_coefficients


def _np(x):
    return np.asarray(jax.device_get(x))


def _assert_allclose(got, expected, *, atol=1e-10, rtol=1e-10):
    got = _np(got)
    expected = np.asarray(expected, dtype=got.dtype)
    assert got.shape == expected.shape, f"shape mismatch: {got.shape} vs {expected.shape}"
    np.testing.assert_allclose(got, expected, atol=atol, rtol=rtol)


def _factorial_degree_plus_one(degree) -> np.ndarray:
    degree = _np(degree).astype(np.int64)
    return np.asarray([math.factorial(int(n) + 1) for n in degree], dtype=np.float64)


def test_fractional_q1_beta_one_tau_equals_t_gives_factorial_coefficients():
    A = jnp.ones((1, 2, 3), dtype=jnp.float64)
    kernel = VolterraKernel.fractional(beta=jnp.array([1.0]), A=A)

    coeffs = kernel.coef(s=0.25, t=0.75, tau=0.75, trunc=5)

    assert isinstance(coeffs, VolterraCoefficients)
    assert coeffs.q == 1
    assert coeffs.m == 2
    assert coeffs.layout.trunc == 4
    assert coeffs.alpha.shape == (1, coeffs.layout.size)
    assert coeffs.valid.shape == ()
    assert bool(coeffs.valid)
    validate_volterra_coefficients(coeffs)

    expected = 1.0 / _factorial_degree_plus_one(coeffs.layout.degree)
    _assert_allclose(coeffs.alpha[0], expected)


def test_fractional_coef_grid_shapes_and_invalid_future_mask():
    A = jnp.arange(12, dtype=jnp.float64).reshape(2, 2, 3) + 1.0
    beta = jnp.array([0.7, 1.25], dtype=jnp.float64)
    kernel = VolterraKernel.fractional(beta=beta, A=A)
    times = jnp.array([0.0, 0.2, 0.5, 1.0], dtype=jnp.float64)

    coeffs = kernel.coef_grid(times, trunc=4)

    assert coeffs.leading_shape == (3, 3)
    assert coeffs.alpha.shape == (3, 3, 2, coeffs.layout.size)
    assert coeffs.valid.shape == (3, 3)
    validate_volterra_coefficients(coeffs)

    expected_valid = np.triu(np.ones((3, 3), dtype=bool))
    np.testing.assert_array_equal(_np(coeffs.valid), expected_valid)

    invalid = np.tril(np.ones((3, 3), dtype=bool), k=-1)
    assert np.all(_np(coeffs.alpha)[invalid] == 0.0)
    assert np.all(_np(coeffs.alpha)[expected_valid] > 0.0)


def test_fractional_coef_broadcasts_input_triples():
    A = jnp.ones((2, 1, 1), dtype=jnp.float64)
    beta = jnp.array([0.5, 1.5], dtype=jnp.float64)
    kernel = VolterraKernel.fractional(beta=beta, A=A)

    s = jnp.array([0.0, 0.1], dtype=jnp.float64)[:, None]
    t = s + jnp.array([0.2, 0.4, 0.6], dtype=jnp.float64)[None, :]
    tau = jnp.array([0.8, 0.9, 1.0], dtype=jnp.float64)[None, :]
    coeffs = kernel.coef(s, t, tau, trunc=3)

    assert coeffs.leading_shape == (2, 3)
    assert coeffs.alpha.shape == (2, 3, 2, coeffs.layout.size)
    assert coeffs.valid.shape == (2, 3)
    validate_volterra_coefficients(coeffs)


def test_piecewise_constant_coefficients_match_closed_form():
    A = jnp.eye(2, dtype=jnp.float64).reshape(2, 1, 2)
    B = jnp.array(
        [
            [[2.0, 3.0], [5.0, 7.0]],
            [[11.0, 13.0], [17.0, 19.0]],
        ],
        dtype=jnp.float64,
    )
    kernel = VolterraKernel.piecewise_constant(B=B, A=A)

    coeffs = kernel.coef_from_indices(source=0, readout=1, trunc=4)

    validate_volterra_coefficients(coeffs)
    ell = _np(coeffs.layout.ell).astype(np.int64)
    degree = _np(coeffs.layout.degree).astype(np.int64)
    diag = np.asarray([2.0, 11.0])
    out = np.asarray([3.0, 13.0])
    prefix = np.prod(diag[None, :] ** ell, axis=1)
    inv_fact = np.asarray([1.0 / math.factorial(int(n) + 1) for n in degree])
    expected = out[:, None] * prefix[None, :] * inv_fact[None, :]

    _assert_allclose(coeffs.alpha, expected)
    assert bool(coeffs.valid)


def test_piecewise_constant_grid_marks_lower_triangle_invalid():
    A = jnp.ones((2, 1, 1), dtype=jnp.float64)
    B = jnp.ones((2, 3, 3), dtype=jnp.float64)
    kernel = VolterraKernel.piecewise_constant(B=B, A=A)
    times = jnp.array([0.0, 0.2, 0.5, 1.0], dtype=jnp.float64)

    coeffs = kernel.coef_grid(times, trunc=3)

    assert coeffs.leading_shape == (3, 3)
    expected_valid = np.triu(np.ones((3, 3), dtype=bool))
    np.testing.assert_array_equal(_np(coeffs.valid), expected_valid)
    assert np.all(_np(coeffs.alpha)[~expected_valid] == 0.0)

    degree = _np(coeffs.layout.degree).astype(np.int64)
    expected_by_degree = np.asarray([1.0 / math.factorial(int(n) + 1) for n in degree])
    _assert_allclose(coeffs.alpha[0, 0, 0], expected_by_degree)
    _assert_allclose(coeffs.alpha[0, 2, 1], expected_by_degree)


def test_gamma_beta_one_rate_zero_tau_equals_t_gives_factorial_coefficients():
    A = jnp.ones((1, 1, 2), dtype=jnp.float64)
    kernel = VolterraKernel.gamma(beta=1.0, scale=1.0, rate=0.0, A=A, quad_order=64)

    coeffs = kernel.coef(s=0.0, t=0.8, tau=0.8, trunc=5)

    assert coeffs.q == 1
    validate_volterra_coefficients(coeffs)
    expected = 1.0 / _factorial_degree_plus_one(coeffs.layout.degree)
    _assert_allclose(coeffs.alpha[0], expected, atol=2e-12, rtol=2e-12)


def test_invalid_time_triples_are_masked_to_zero():
    A = jnp.ones((1, 1, 1), dtype=jnp.float64)
    kernel = VolterraKernel.fractional(beta=jnp.array([1.0]), A=A)

    coeffs = kernel.coef(
        s=jnp.array([0.0, 0.5, 0.0]),
        t=jnp.array([0.5, 0.5, 0.8]),
        tau=jnp.array([0.5, 1.0, 0.7]),
        trunc=3,
    )

    np.testing.assert_array_equal(_np(coeffs.valid), np.asarray([True, False, False]))
    assert np.all(_np(coeffs.alpha)[1:] == 0.0)
    assert np.any(_np(coeffs.alpha)[0] != 0.0)


def test_coefficient_slicing_and_source_readout_broadcasting():
    A = jnp.ones((2, 1, 1), dtype=jnp.float64)
    beta = jnp.array([0.75, 1.25], dtype=jnp.float64)
    kernel = VolterraKernel.fractional(beta=beta, A=A)
    times = jnp.array([0.0, 0.3, 0.8], dtype=jnp.float64)
    coeffs = kernel.coef_grid(times, trunc=3)

    row = coeffs[0]
    assert row.leading_shape == (2,)
    assert row.alpha.shape == (2, 2, coeffs.layout.size)
    np.testing.assert_array_equal(_np(row.valid), _np(coeffs.valid[0]))

    single = kernel.coef(0.0, 0.25, 0.5, trunc=3)
    broadcast = single.broadcast_source_readout(steps=2, readouts=3)
    assert broadcast.leading_shape == (2, 3)
    _assert_allclose(broadcast.alpha[0, 0], single.alpha)
    _assert_allclose(broadcast.alpha[1, 2], single.alpha)
    np.testing.assert_array_equal(_np(broadcast.valid), np.ones((2, 3), dtype=bool))


def test_kernel_constructor_validation():
    with pytest.raises(ValueError, match="A must have shape"):
        VolterraKernel.fractional(beta=jnp.array([1.0]), A=jnp.ones((1, 2)))

    with pytest.raises(ValueError, match="fractional beta must have shape"):
        VolterraKernel.fractional(beta=jnp.array([1.0, 2.0]), A=jnp.ones((1, 1, 1)))

    with pytest.raises(ValueError, match="gamma kernels are scalar"):
        VolterraKernel.gamma(beta=1.0, A=jnp.ones((2, 1, 1)))

    with pytest.raises(ValueError, match="piecewise constant B"):
        VolterraKernel.piecewise_constant(B=jnp.ones((2, 3)), A=jnp.ones((2, 1, 1)))
