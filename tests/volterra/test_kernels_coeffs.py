from __future__ import annotations

import functools
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from tensordev.volterra import VolterraCoefficients, ConvolutionKernel, FractionalKernel, GammaKernel
from tensordev.volterra.coeffs import validate_volterra_coefficients
from tensordev.volterra.kernel import _gamma_series_x_max

from parameter_kernels import check_grads_each, parameter_kernel


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
    kernel = FractionalKernel(beta=jnp.array([1.0]), A=A)

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
    kernel = FractionalKernel(beta=beta, A=A)
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
    kernel = FractionalKernel(beta=beta, A=A)

    s = jnp.array([0.0, 0.1], dtype=jnp.float64)[:, None]
    t = s + jnp.array([0.2, 0.4, 0.6], dtype=jnp.float64)[None, :]
    tau = jnp.array([0.8, 0.9, 1.0], dtype=jnp.float64)[None, :]
    coeffs = kernel.coef(s, t, tau, trunc=3)

    assert coeffs.leading_shape == (2, 3)
    assert coeffs.alpha.shape == (2, 3, 2, coeffs.layout.size)
    assert coeffs.valid.shape == (2, 3)
    validate_volterra_coefficients(coeffs)


def test_gamma_beta_one_rate_zero_tau_equals_t_gives_factorial_coefficients():
    A = jnp.ones((1, 1, 2), dtype=jnp.float64)
    kernel = GammaKernel(beta=1.0, scale=1.0, rate=0.0, A=A, quad_order=64)

    coeffs = kernel.coef(s=0.0, t=0.8, tau=0.8, trunc=5)

    assert coeffs.q == 1
    validate_volterra_coefficients(coeffs)
    expected = 1.0 / _factorial_degree_plus_one(coeffs.layout.degree)
    _assert_allclose(coeffs.alpha[0], expected, atol=2e-12, rtol=2e-12)


def test_invalid_time_triples_are_masked_to_zero():
    A = jnp.ones((1, 1, 1), dtype=jnp.float64)
    kernel = FractionalKernel(beta=jnp.array([1.0]), A=A)

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
    kernel = FractionalKernel(beta=beta, A=A)
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
        FractionalKernel(beta=jnp.array([1.0]), A=jnp.ones((1, 2)))

    with pytest.raises(ValueError, match="fractional beta must have shape"):
        FractionalKernel(beta=jnp.array([1.0, 2.0]), A=jnp.ones((1, 1, 1)))

    with pytest.raises(ValueError, match="gamma kernels are scalar"):
        GammaKernel(beta=1.0, scale=1.0, rate=0.0, A=jnp.ones((2, 1, 1)))

    with pytest.raises(ValueError, match="quad_order must be positive"):
        GammaKernel(beta=1.0, scale=1.0, rate=0.0, A=jnp.ones((1, 1, 1)), quad_order=0)


# ---------------------------------------------------------------------------
# GammaKernel coefficients
# ---------------------------------------------------------------------------

def _gamma_kernel(beta, *, scale=1.0, rate=0.0, quad_order=64, dtype=jnp.float64):
    return GammaKernel(
        beta=jnp.asarray(beta, dtype=dtype),
        scale=jnp.asarray(scale, dtype=dtype),
        rate=jnp.asarray(rate, dtype=dtype),
        A=jnp.ones((1, 1, 1), dtype=dtype),
        quad_order=quad_order,
    )


def _fractional_kernel(beta):
    return FractionalKernel(beta=jnp.array([beta]), A=jnp.ones((1, 1, 1)))


def _weighted_coef_sum(kernel, s, t, tau, rho):
    """Mixed-sign weighted sum of all coefficients up to degree 4."""
    alpha = kernel.coef(s, t, tau, trunc=5, rho=rho).alpha
    weights = jnp.cos(jnp.arange(alpha.size, dtype=alpha.dtype)).reshape(alpha.shape)
    return jnp.sum(weights * alpha)


def _mpmath_alpha(s, t, tau, n, beta, scale, rate, rho):
    """alpha_n(s, t, tau; rho) of the gamma kernel at 40 digits.

    With h = t - s, d = tau - t and I the regularised incomplete beta function
    (I = 1 for n = 1),

        alpha_n = scale^n / (Gamma(n beta) h^n) int_0^h (h - v)^rho
                  e^(-rate (d + v)) (d + v)^(n beta - 1) I_{v/(d+v)}((n - 1) beta, beta) dv.

    The range is split geometrically towards v = 0, and on the first piece
    v = w^(1/p) removes the v^(p - 1) endpoint behaviour, with p = n beta if
    d = 0 and p = (n - 1) beta + 1 otherwise.
    """
    import mpmath as mp

    mp.mp.dps = 40
    s, t, tau, beta, scale, rate, rho = map(mp.mpf, (s, t, tau, beta, scale, rate, rho))
    h, d = t - s, tau - t

    def f(v):
        val = (h - v) ** rho * mp.exp(-rate * (d + v)) * (d + v) ** (n * beta - 1)
        if n > 1:
            val *= mp.betainc((n - 1) * beta, beta, 0, v / (d + v), regularized=True)
        return val

    pts = [h]
    while pts[-1] > min(r for r in (d, 1 / rate, h) if r > 0) / 8:
        pts.append(pts[-1] / 2)
    p = n * beta if d == 0 else (n - 1) * beta + 1
    head = mp.quad(lambda w: f(w ** (1 / p)) * w ** (1 / p - 1) / p, [0, pts[-1] ** p])
    return scale**n * (head + mp.quad(f, pts[::-1])) / (mp.gamma(n * beta) * h**n)


# alpha_n(s, t, tau; rho) for n = 1..5 from _mpmath_alpha, keyed by
# (s, t, tau, beta, scale, rate, rho).
_GAMMA_SERIES_REFERENCE = [
    ((0.1, 0.6, 0.6, 0.15, 1.3, 0.7, 0.0),
     (2.4066926558575722, 5.6691131458111761, 13.100213553555704,
      29.752883136747423, 66.519710518755617)),
    ((0.1, 0.6, 0.6, 0.3, 1.3, 5.0, 0.3),
     (1.237454495427373, 1.8635463537831203, 2.7699956735885033,
      4.0593527635119044, 5.8604737431244638)),
    ((0.0, 1.0, 1.0, 0.8, 0.9, 13.0, 2.0),
     (0.10238794253478435, 0.010409089131478886, 0.0010499452366101238,
      0.00010498993500365884, 1.0398124506665693e-5)),
    ((0.1, 0.6, 0.85, 0.3, 1.3, 5.0, 0.0),
     (0.090540938326657564, 0.13888460550554663, 0.20953902553253364,
      0.31065501367409051, 0.45237004303057693)),
    ((0.1, 0.6, 0.85, 1.7, 1.3, 2.0, 2.7),
     (0.013914131944185471, 0.00055616703867027124, 1.4908126172114212e-5,
      2.8647563448766887e-7, 4.1480109374801412e-9)),
    ((0.2, 0.3, 2.0, 0.6, 1.1, 3.0, 0.6),
     (0.0005066804986009882, 0.00079857356921377977, 0.0011134578521612747,
      0.0014010577328147334, 0.0016141859936498306)),
    ((0.0, 1.0, 1.001, 0.3, 1.0, 4.0, 0.3),
     (0.50182600147531291, 0.32039875504083937, 0.20349705142393981,
      0.12845810796437384, 0.080524139528642856)),
    ((0.0, 0.25, 0.25, 1.0, 1.0, 2.0, 1.0),
     (0.10653065971263342, 0.032653298563167118, 0.0077558852584514895,
      0.0014987900665763897, 0.00024373005385932697)),
]

# The same for rate * (t - s) above the float64 series threshold 13.85, where the
# coefficients come from tanh-sinh quadrature.
_GAMMA_QUAD_REFERENCE = [
    ((0.0, 0.5, 0.5, 0.3, 1.3, 60.0, 0.0),
     (0.76125116952361052, 0.57950334310105886, 0.44114759767850405,
      0.33582412466524622, 0.25564650765556368)),
    ((0.0, 0.5, 0.55, 0.8, 1.3, 400.0, 0.8),
     (1.1871720966310522e-11, 2.5494050434107767e-13, 5.4746748307698409e-15,
      1.175633898596249e-16, 2.5245275036817789e-18)),
    ((0.0, 1.0, 1.5, 0.6, 1.0, 20.0, 2.0),
     (1.7615492593050145e-6, 2.7434214255063819e-7, 4.2646138216084505e-8,
      6.6162244348637842e-9, 1.0243170982035009e-9)),
    ((0.0, 0.5, 0.5, 1.7, 1.0, 40.0, 1.7),
     (0.00100350277745681, 3.228196768504896e-6, 1.022107987458877e-8,
      3.1752036923316876e-11, 9.6400931107545515e-14)),
]


@pytest.mark.parametrize("beta", [0.15, 0.6, 1.0, 1.7])
def test_gamma_rate_zero_matches_fractional(beta):
    """``rate=0, scale=1`` is the fractional kernel, for every basis exponent."""
    s = jnp.array([0.1, 0.1, 0.0, 0.2, 0.0])
    t = jnp.array([0.6, 0.6, 0.25, 0.3, 1.0])
    tau = jnp.array([0.6, 0.85, 0.25, 2.0, 1.001])
    gamma = _gamma_kernel(beta)
    frac = _fractional_kernel(beta)
    for rho in (0.0, beta, 1.0, beta + 1.0, 2.0):
        got = gamma.coef(s, t, tau, trunc=5, rho=rho)
        ref = frac.coef(s, t, tau, trunc=5, rho=rho)
        _assert_allclose(got.alpha, ref.alpha, atol=0.0, rtol=1e-12)


@pytest.mark.parametrize("case, expected", _GAMMA_SERIES_REFERENCE)
def test_gamma_series_matches_high_precision_reference(case, expected):
    """The series matches the 40-digit references of ``_mpmath_alpha``.

    The cases cover ``tau == t`` with ``n * beta < 1``, a far lag,
    ``(tau - t) / (t - s) = 1e-3``, and ``rate * (t - s) = 13`` just below
    the series threshold.
    """
    s, t, tau, beta, scale, rate, rho = case
    assert rate * (t - s) <= _gamma_series_x_max(64, "float64")
    coeffs = _gamma_kernel(beta, scale=scale, rate=rate).coef(s, t, tau, trunc=5, rho=rho)
    _assert_allclose(coeffs.alpha[0], expected, atol=0.0, rtol=1e-12)


@pytest.mark.parametrize("case, expected", _GAMMA_QUAD_REFERENCE)
def test_gamma_quadrature_matches_high_precision_reference(case, expected):
    """Above the series threshold, 64 and 128 tanh-sinh nodes match to 1e-6 and 1e-10."""
    s, t, tau, beta, scale, rate, rho = case
    assert rate * (t - s) > _gamma_series_x_max(64, "float64")
    default = _gamma_kernel(beta, scale=scale, rate=rate).coef(s, t, tau, trunc=5, rho=rho)
    _assert_allclose(default.alpha[0], expected, atol=0.0, rtol=1e-6)
    fine = _gamma_kernel(beta, scale=scale, rate=rate, quad_order=128)
    _assert_allclose(fine.coef(s, t, tau, trunc=5, rho=rho).alpha[0], expected, atol=0.0, rtol=1e-10)


def test_gamma_series_and_quadrature_agree_at_threshold():
    """Coefficients at ``rate * (t - s) = x_max * (1 -+ 1e-12)`` agree."""
    x_max = _gamma_series_x_max(64, "float64")
    assert 13.8 < x_max < 13.9
    assert 18.9 < _gamma_series_x_max(64, "float32") < 19.0
    h = 0.5
    rates = jnp.array([x_max * (1.0 - 1e-12), x_max * (1.0 + 1e-12)]) / h
    for beta, tau in [(0.3, h), (1.7, h), (0.6, 1.5 * h)]:
        inside, outside = (
            _gamma_kernel(beta, rate=rate, quad_order=128).coef(0.0, h, tau, trunc=5, rho=beta).alpha
            for rate in rates
        )
        _assert_allclose(outside, inside, atol=0.0, rtol=1e-10)


def _gamma_coef_sum(scale, rate, s, t, tau, *, beta, dtype=jnp.float64):
    return _weighted_coef_sum(_gamma_kernel(beta, scale=scale, rate=rate, dtype=dtype), s, t, tau, rho=beta)


@pytest.mark.parametrize("beta", [0.3, 1.7])
@pytest.mark.parametrize(
    "s, t, tau, rate",
    [(0.1, 0.6, 0.85, 2.5), (0.0, 0.5, 0.7, 40.0), (0.1, 0.6, 0.85, 80.0)],
    ids=["series", "quad", "quad_c_is_1"],
)
def test_gamma_gradients_match_finite_differences(beta, s, t, tau, rate):
    """Gradients w.r.t. scale, rate, s, t and tau."""
    check_grads_each(
        functools.partial(_gamma_coef_sum, beta=beta), (1.3, rate, s, t, tau),
        eps=1e-6, rtol=1e-6, atol=1e-9,
    )


@pytest.mark.parametrize("beta", [0.3, 1.7])
@pytest.mark.parametrize("tau", [0.6, 0.85])
def test_gamma_rate_gradient_at_zero(beta, tau):
    """d alpha_n / d rate at rate 0 is scale^n (alpha^frac(rho+1) - (tau-s) alpha^frac(rho))."""
    s, t, scale = 0.1, 0.6, 1.3
    frac = _fractional_kernel(beta)
    n = jnp.arange(1.0, 6.0)
    for rho in (0.0, beta):
        got = jax.jacfwd(
            lambda rate: _gamma_kernel(beta, scale=scale, rate=rate)
            .coef(s, t, tau, trunc=5, rho=rho)
            .alpha[0]
        )(0.0)
        expected = scale**n * (
            frac.coef(s, t, tau, trunc=5, rho=rho + 1.0).alpha[0]
            - (tau - s) * frac.coef(s, t, tau, trunc=5, rho=rho).alpha[0]
        )
        _assert_allclose(got, expected, atol=0.0, rtol=1e-12)


@pytest.mark.parametrize("beta", [0.3, 1.0, 1.7])
@pytest.mark.parametrize("rate", [0.0, 2.5, 40.0])
def test_gamma_gradients_are_finite_on_diagonal_and_invalid_triples(beta, rate):
    """Gradients are finite at ``tau == t`` and zero for ``tau < t``.

    The derivative along ``t == tau`` matches central differences.
    """
    f = functools.partial(_gamma_coef_sum, beta=beta)
    grad_f = jax.grad(f, argnums=range(5))
    assert all(np.isfinite(float(g)) for g in grad_f(1.3, rate, 0.1, 0.6, 0.6))
    check_grads(
        lambda c: f(1.3, rate, 0.1, c, c), (0.6,),
        order=1, modes=("rev",), eps=1e-6, rtol=1e-6, atol=0.0,
    )
    assert all(float(g) == 0.0 for g in grad_f(1.3, rate, 0.1, 0.6, 0.5))


def test_gamma_float32_quadrature_gradients_match_float64():
    """Reverse-mode gradients at rate * (t - s) = 25, on the float32 quadrature path."""
    got, ref = (
        np.array(jax.grad(functools.partial(_gamma_coef_sum, beta=0.3, dtype=dtype), argnums=range(5))(
            *(jnp.asarray(v, dtype) for v in (1.3, 50.0, 0.1, 0.6, 0.85))
        ))
        for dtype in (jnp.float32, jnp.float64)
    )
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3 * np.max(np.abs(ref)))


def test_gamma_float32_matches_float64():
    """float32 coefficients are finite and close to float64, including far lags."""
    times = np.linspace(0.0, 1.0, 9)
    tau = np.array([0.125, 0.5, 1.0, 40.0])
    # With rate 400, rate * (t - s) >= 50 is above the float32 series threshold.
    for beta, rate in [(0.3, 0.0), (0.6, 1.5), (1.7, 3.0), (0.6, 400.0)]:
        got, ref = (
            _gamma_kernel(beta, scale=1.3, rate=rate, dtype=dtype)
            .coef_grid(jnp.asarray(times, dtype), trunc=5, rho=beta, tau=jnp.asarray(tau, dtype))
            .alpha
            for dtype in (jnp.float32, jnp.float64)
        )
        assert got.dtype == jnp.float32
        got, ref = _np(got).astype(np.float64), _np(ref)
        assert np.all(np.isfinite(got))
        representable = np.abs(ref) > 1e-30
        np.testing.assert_allclose(got[representable], ref[representable], rtol=5e-5)
        assert np.all(got[~representable] < 1e-30)

    # z = (t - s) / (tau - s) = 1 / 30000 and a = 2 n + 1 give z^a < 1e-38 for
    # n >= 4, below the float32 normal range.
    far32 = _gamma_kernel(2.0, dtype=jnp.float32).coef(0.0, 1e-3, 30.0, trunc=6, rho=2.0)
    far64 = _gamma_kernel(2.0).coef(0.0, 1e-3, 30.0, trunc=6, rho=2.0)
    assert np.all(_np(far32.alpha) > 0.0)
    _assert_allclose(_np(far32.alpha).astype(np.float64), far64.alpha, atol=0.0, rtol=1e-4)


# ---------------------------------------------------------------------------
# Derivatives with respect to the kernel parameters, including beta
# ---------------------------------------------------------------------------

# name -> parameters of parameter_kernel; rate * (t - s) is 1.25 for "gamma_series"
# and 20 for "gamma_quad".
_KERNEL_PARAMS = {
    "fractional": (0.3,),
    "fractional_q2": (0.45, 1.3),
    "gamma_series": (0.3, 1.3, 2.5),
    "gamma_quad": (0.3, 1.3, 40.0),
}


def _parameter_coef_sum(params, s, t, tau, *, kernel, rho):
    """``_weighted_coef_sum`` of ``parameter_kernel``; ``rho="beta"`` ties rho to beta."""
    return _weighted_coef_sum(
        parameter_kernel(kernel, params, jnp.ones((1, 1, 1))),
        s, t, tau, rho=params[0] if rho == "beta" else rho,
    )


@pytest.mark.parametrize("rho", [0.0, "beta", 1.0], ids=["rho0", "rho_beta", "rho1"])
@pytest.mark.parametrize("tau", [0.6, 0.85], ids=["tau_eq_t", "tau_gt_t"])
@pytest.mark.parametrize("kernel", list(_KERNEL_PARAMS))
def test_parameter_gradients_match_finite_differences(kernel, tau, rho):
    """Gradients w.r.t. beta (and scale, rate) of the coefficients."""
    check_grads_each(
        lambda p: _parameter_coef_sum(p, 0.1, 0.6, tau, kernel=kernel, rho=rho),
        (jnp.asarray(_KERNEL_PARAMS[kernel]),), eps=1e-6, rtol=1e-7, atol=1e-9,
    )


@pytest.mark.parametrize("kernel", list(_KERNEL_PARAMS))
def test_parameter_gradients_vanish_on_invalid_triples(kernel):
    """Triples with t == s, t < s or tau < t have gradient exactly zero, not NaN."""
    params = jnp.asarray(_KERNEL_PARAMS[kernel])
    s = jnp.array([0.5, 0.6, 0.3])
    t = jnp.array([0.5, 0.4, 0.7])
    tau = jnp.array([1.0, 0.9, 0.5])
    for rho in (0.0, "beta"):
        f = lambda p: _parameter_coef_sum(p, s, t, tau, kernel=kernel, rho=rho)
        np.testing.assert_array_equal(_np(jax.grad(f)(params)), 0.0)


if __name__ == "__main__":
    import mpmath as mp

    for case, _ in _GAMMA_SERIES_REFERENCE + _GAMMA_QUAD_REFERENCE:
        print(case, [mp.nstr(_mpmath_alpha(*case[:3], n, *case[3:]), 17) for n in range(1, 6)])
