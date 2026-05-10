from __future__ import annotations

import itertools
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tensordev.core.jax import Jax
from tensordev.volterra import VolterraKernel, FractionalKernel
from tensordev.volterra.eval_general import eval_e as eval_e_general
from tensordev.volterra.eval_general import eval_vte as eval_vte_general
from tensordev.volterra.eval_scalar import eval_vte as eval_vte_scalar


_CORE = Jax()


def eval_e(y, coeffs):
    return eval_e_general(y, coeffs)


def eval_vte(v, y, coeffs):
    if coeffs.q == 1:
        return eval_vte_scalar(v, y, coeffs)
    return eval_vte_general(v, y, coeffs)


def _np(x):
    return np.asarray(jax.device_get(x))


def _assert_level_allclose(got, expected, *, atol=1e-10, rtol=1e-10):
    got_np = _np(got)
    expected_np = np.asarray(expected, dtype=got_np.dtype)
    assert got_np.shape == expected_np.shape, f"shape mismatch: {got_np.shape} vs {expected_np.shape}"
    np.testing.assert_allclose(got_np, expected_np, atol=atol, rtol=rtol)


def _kron_word(y: np.ndarray, word: tuple[int, ...]) -> np.ndarray:
    out = np.asarray([1.0], dtype=y.dtype)
    for letter in word:
        out = np.kron(out, y[letter])
    return out


def _bruteforce_e_from_coeffs(y: jnp.ndarray, coeffs) -> tuple[np.ndarray, ...]:
    y_np = _np(y)
    alpha = _np(coeffs.alpha)
    q, m = coeffs.q, coeffs.m
    out = [np.zeros((m ** n,), dtype=alpha.dtype) for n in range(coeffs.trunc + 1)]
    for n in range(1, coeffs.trunc + 1):
        for word in itertools.product(range(q), repeat=n):
            prefix = word[:-1]
            last = word[-1]
            counts = [prefix.count(r) for r in range(q)]
            ell_idx = coeffs.layout.index_of(counts)
            out[n] += alpha[last, ell_idx] * _kron_word(y_np, word)
    return tuple(out)


def _make_dense_element(levels: list[np.ndarray | jnp.ndarray]) -> tuple[jnp.ndarray, ...]:
    return tuple(jnp.asarray(level, dtype=jnp.float64) for level in levels)


def test_eval_e_q1_beta_one_is_exponential_increment_minus_unit():
    m = 2
    trunc = 5
    A = jnp.ones((1, m, 3), dtype=jnp.float64)
    kernel = FractionalKernel(beta=jnp.array([1.0]), A=A)
    coeffs = kernel.coef(s=0.0, t=0.4, tau=0.4, trunc=trunc)
    y = jnp.array([0.3, -0.7], dtype=jnp.float64)

    E = eval_e(y, coeffs)

    _assert_level_allclose(E[0], np.zeros((1,)))
    power = np.asarray([1.0])
    y_np = _np(y)
    for n in range(1, trunc + 1):
        power = np.kron(power, y_np)
        expected = power / math.factorial(n)
        _assert_level_allclose(E[n], expected)


def test_eval_e_q1_accepts_explicit_component_axis():
    A = jnp.ones((1, 2, 1), dtype=jnp.float64)
    kernel = FractionalKernel(beta=jnp.array([1.0]), A=A)
    coeffs = kernel.coef(s=0.0, t=1.0, tau=1.0, trunc=3)
    y = jnp.array([[0.2, 0.5]], dtype=jnp.float64)

    E_component = eval_e(y, coeffs)
    E_flat = eval_e(y[0], coeffs)

    for got, expected in zip(E_component, E_flat):
        _assert_level_allclose(got, expected)


def test_eval_vte_matches_explicit_tensor_product_with_eval_e():
    m = 2
    trunc = 4
    A = jnp.ones((1, m, 1), dtype=jnp.float64)
    kernel = FractionalKernel(beta=jnp.array([1.0]), A=A)
    coeffs = kernel.coef(s=0.0, t=1.0, tau=1.0, trunc=trunc)
    y = jnp.array([0.4, -0.1], dtype=jnp.float64)
    v = _make_dense_element(
        [
            np.array([1.0]),
            np.array([0.2, -0.3]),
            np.arange(m**2, dtype=np.float64) / 10.0,
            np.arange(m**3, dtype=np.float64) / 20.0,
            np.arange(m**4, dtype=np.float64) / 30.0,
        ]
    )

    got = eval_vte(v, y, coeffs)
    E = eval_e(y, coeffs)
    expected_first = _CORE.tensor_product(v, E[1:], trunc=trunc, b_first_on=True)
    expected = (jnp.zeros_like(got[0]),) + tuple(expected_first)

    for g, e in zip(got, expected):
        _assert_level_allclose(g, e)


def test_eval_e_broadcasts_coefficients_and_increment_batches():
    A = jnp.ones((1, 2, 1), dtype=jnp.float64)
    kernel = FractionalKernel(beta=jnp.array([1.0]), A=A)
    coeffs = kernel.coef(
        s=jnp.array([0.0, 0.1]),
        t=jnp.array([0.5, 0.6]),
        tau=jnp.array([0.5, 0.6]),
        trunc=3,
    )
    y = jnp.array([0.2, -0.5], dtype=jnp.float64)

    E = eval_e(y, coeffs)

    assert E[0].shape == (2, 1)
    assert E[1].shape == (2, 2)
    assert E[2].shape == (2, 4)
    assert E[3].shape == (2, 8)


