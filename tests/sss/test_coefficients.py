from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

from tensordev.sss import FSSK


QUAD_ORDER = 32

EMPTY_PSI_ATOL = 1e-12
EMPTY_PSI_RTOL = 1e-12

NONEMPTY_PSI_ATOL = 1e-11
NONEMPTY_PSI_RTOL = 1e-11

PHI_ATOL = 1e-11
PHI_RTOL = 1e-11


def _np(x):
    return np.asarray(jax.device_get(x))


def _assert_allclose(got, expected, *, atol=1e-8, rtol=1e-8):
    got = _np(got)
    expected = np.asarray(expected, dtype=got.dtype)
    assert got.shape == expected.shape, f"shape mismatch: {got.shape} vs {expected.shape}"
    np.testing.assert_allclose(got, expected, atol=atol, rtol=rtol)


def _assert_psi_empty_allclose(got, expected, degree):
    got = _np(got)
    expected = np.asarray(expected, dtype=got.dtype)
    degree = np.asarray(degree, dtype=np.int64)

    assert got.shape == expected.shape, f"shape mismatch: {got.shape} vs {expected.shape}"

    empty = degree == 0
    assert np.count_nonzero(empty) == 1

    np.testing.assert_allclose(
        got[:, empty, :],
        expected[:, empty, :],
        atol=EMPTY_PSI_ATOL,
        rtol=EMPTY_PSI_RTOL,
    )


def _assert_psi_nonempty_allclose(got, expected, degree):
    got = _np(got)
    expected = np.asarray(expected, dtype=got.dtype)
    degree = np.asarray(degree, dtype=np.int64)

    assert got.shape == expected.shape, f"shape mismatch: {got.shape} vs {expected.shape}"

    nonempty = degree > 0
    assert np.any(nonempty)

    np.testing.assert_allclose(
        got[:, nonempty, :],
        expected[:, nonempty, :],
        atol=NONEMPTY_PSI_ATOL,
        rtol=NONEMPTY_PSI_RTOL,
    )


def _factorial_degree_plus_one(degree: np.ndarray) -> np.ndarray:
    return np.asarray([math.factorial(int(n) + 1) for n in degree], dtype=np.float64)


def _multiindex_products(layout, b_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ell = _np(layout.ell).astype(np.int64)
    degree = _np(layout.degree).astype(np.int64)
    prod = np.prod(b_vec[None, :] ** ell, axis=1)
    return ell, degree, prod


def _word_from_multiindex(multi: np.ndarray) -> list[int]:
    word: list[int] = []
    for p, count in enumerate(multi):
        word.extend([p] * int(count))
    return word


def _component_matrices(b: np.ndarray) -> list[np.ndarray]:
    # C_p = b_p 1^T.
    r = b.shape[1]
    one_row = np.ones((1, r), dtype=b.dtype)
    return [b[p, :, None] @ one_row for p in range(b.shape[0])]


def _block(i: int, j: int, r: int) -> tuple[slice, slice]:
    return slice(i * r, (i + 1) * r), slice(j * r, (j + 1) * r)


def _phi_word_block_expm(
    Lambda: np.ndarray,
    b: np.ndarray,
    word: list[int],
    delta: float,
) -> np.ndarray:
    """Independent block-exponential oracle for Phi^word(delta)."""
    r = Lambda.shape[0]
    n = len(word)
    C = _component_matrices(b)

    big = np.zeros(((n + 1) * r, (n + 1) * r), dtype=np.float64)

    for i in range(n + 1):
        rows, cols = _block(i, i, r)
        big[rows, cols] = -Lambda

    for i, p in enumerate(word):
        rows, cols = _block(i, i + 1, r)
        big[rows, cols] = C[p]

    exp_big = _np(jsp_linalg.expm(jnp.asarray(delta * big)))
    rows, cols = _block(0, n, r)
    return exp_big[rows, cols]


def _psi_word_block_expm(
    Lambda: np.ndarray,
    b: np.ndarray,
    word: list[int],
    delta: float,
) -> np.ndarray:
    """Independent block-exponential oracle for Psi^word(delta) = int_0^delta Phi^word(u) du."""
    r = Lambda.shape[0]
    n = len(word)
    C = _component_matrices(b)

    # Blocks 0..n have diagonal -Lambda; final block n+1 has diagonal 0.
    big = np.zeros(((n + 2) * r, (n + 2) * r), dtype=np.float64)

    for i in range(n + 1):
        rows, cols = _block(i, i, r)
        big[rows, cols] = -Lambda

    for i, p in enumerate(word):
        rows, cols = _block(i, i + 1, r)
        big[rows, cols] = C[p]

    rows, cols = _block(n, n + 1, r)
    big[rows, cols] = np.eye(r)

    exp_big = _np(jsp_linalg.expm(jnp.asarray(delta * big)))
    rows, cols = _block(0, n + 1, r)
    return exp_big[rows, cols]


def _scalar_positive_lambda_psi_hat(x: float, n: int) -> float:
    r"""Stable oracle for int_0^1 exp(-x s) s^n / n! ds."""
    nodes, weights = np.polynomial.legendre.leggauss(128)
    s = 0.5 * (nodes + 1.0)
    return float(
        0.5
        * np.sum(weights * np.exp(-x * s) * (s**n))
        / math.factorial(n)
    )


def _unique_multiplicities(values: list[float]) -> tuple[list[float], list[int]]:
    unique: list[float] = []
    multiplicities: list[int] = []

    for value in values:
        value = float(value)
        for i, old in enumerate(unique):
            if value == old:
                multiplicities[i] += 1
                break
        else:
            unique.append(value)
            multiplicities.append(1)

    return unique, multiplicities


def _exp_convolution_confluent(rates: list[float], delta: float) -> float:
    r"""
    Return

        int_{s_i >= 0, sum s_i = delta} prod_i exp(-rates[i] s_i) ds.

    This is the confluent partial-fraction expansion of

        prod_i (z + rates[i])^{-1}.

    Repeated rates are handled analytically.
    """
    unique, multiplicities = _unique_multiplicities(rates)
    out = 0.0

    for alpha, m_alpha in zip(unique, multiplicities):
        max_order = m_alpha - 1

        # Taylor coefficients of
        # prod_{beta != alpha} (x + beta - alpha)^(-m_beta)
        # around x = 0, up to order m_alpha - 1.
        coeff = np.zeros(max_order + 1, dtype=np.float64)
        coeff[0] = 1.0

        for beta, m_beta in zip(unique, multiplicities):
            if beta == alpha:
                continue

            diff = beta - alpha
            factor = np.asarray(
                [
                    ((-1.0) ** n)
                    * math.comb(m_beta + n - 1, n)
                    * diff ** (-m_beta - n)
                    for n in range(max_order + 1)
                ],
                dtype=np.float64,
            )
            coeff = np.convolve(coeff, factor)[: max_order + 1]

        exp_part = math.exp(-alpha * delta)

        # coeff[l] / (z + alpha)^{m_alpha-l}
        # inverts to coeff[l] * t^{m_alpha-l-1}/(m_alpha-l-1)! * exp(-alpha t).
        for k in range(m_alpha):
            l = m_alpha - 1 - k
            out += coeff[l] * (delta**k) / math.factorial(k) * exp_part

    return float(out)


def _diag_phi_word_confluent(
    lambdas: np.ndarray,
    b: np.ndarray,
    word: list[int],
    delta: float,
) -> np.ndarray:
    r"""Oracle for Phi^word(delta) when Lambda = diag(lambda_0, lambda_1)."""
    R = 2
    n = len(word)

    if n == 0:
        return np.diag(np.exp(-lambdas * delta))

    out = np.zeros((R, R), dtype=np.float64)

    for a0 in range(R):
        for an in range(R):
            total = 0.0

            # states = (a_0, ..., a_n), with a_0 fixed and a_n fixed.
            for middle in np.ndindex(*(R,) * max(n - 1, 0)):
                states = (a0,) + tuple(middle) + (an,)

                weight = 1.0
                for i, p in enumerate(word):
                    weight *= b[p, states[i]]

                rates = [float(lambdas[state]) for state in states]
                total += weight * _exp_convolution_confluent(rates, delta)

            out[a0, an] = total

    return out


def _diag_psi_row_word_confluent(
    lambdas: np.ndarray,
    b: np.ndarray,
    word: list[int],
    delta: float,
) -> np.ndarray:
    r"""Oracle for 1^T Psi^word(delta) when Lambda = diag(lambda_0, lambda_1)."""
    R = 2
    n = len(word)
    out = np.zeros((R,), dtype=np.float64)

    if n == 0:
        # Psi^empty(delta) = int_0^delta exp(-Lambda u) du.
        return np.asarray(
            [
                _exp_convolution_confluent([float(lambdas[a]), 0.0], delta)
                for a in range(R)
            ],
            dtype=np.float64,
        )

    for an in range(R):
        total_final = 0.0

        for a0 in range(R):
            for middle in np.ndindex(*(R,) * max(n - 1, 0)):
                states = (a0,) + tuple(middle) + (an,)

                weight = 1.0
                for i, p in enumerate(word):
                    weight *= b[p, states[i]]

                # The final 0-rate is the time integration in Psi = int Phi.
                rates = [float(lambdas[state]) for state in states] + [0.0]
                total_final += weight * _exp_convolution_confluent(rates, delta)

        out[an] = total_final

    return out


@pytest.mark.parametrize("n", [1, 3])
def test_multiindex_layout_starts_with_unique_empty_word(q):
    ker = FSSK.from_matrix(
        Lambda=jnp.zeros((1, 1), dtype=jnp.float64),
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=jnp.ones((q, 1), dtype=jnp.float64),
        quad_order=32,
    )

    coef = ker.coef(jnp.asarray([0.1], dtype=jnp.float64), trunc=5, dtype=jnp.float64)

    ell = _np(coef.layout.ell)
    degree = _np(coef.layout.degree)

    np.testing.assert_array_equal(ell[0], np.zeros(q, dtype=ell.dtype))
    assert degree[0] == 0
    assert np.count_nonzero(degree == 0) == 1


@pytest.mark.parametrize(
    "b",
    [
        jnp.asarray([[0.75]], dtype=jnp.float64),
        jnp.asarray([[0.80], [-0.35], [0.20]], dtype=jnp.float64),
    ],
)
def test_E_matches_closed_form_when_lambda_zero(b):
    q = int(b.shape[0])
    ker = FSSK.from_matrix(
        Lambda=jnp.zeros((1, 1), dtype=jnp.float64),
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.05, 0.3, 1.7], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=5, dtype=jnp.float64)

    expected_E = np.ones_like(_np(coef.E))
    _assert_allclose(coef.E, expected_E, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "b",
    [
        jnp.asarray([[0.75]], dtype=jnp.float64),
        jnp.asarray([[0.80], [-0.35], [0.20]], dtype=jnp.float64),
    ],
)
def test_psi_empty_matches_closed_form_when_lambda_zero(b):
    q = int(b.shape[0])
    ker = FSSK.from_matrix(
        Lambda=jnp.zeros((1, 1), dtype=jnp.float64),
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.05, 0.3, 1.7], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=5, dtype=jnp.float64)

    _, degree, _ = _multiindex_products(coef.layout, _np(b[:, 0]))
    expected_psi = np.zeros_like(_np(coef.psi))
    expected_psi[:, 0, 0] = 1.0

    _assert_psi_empty_allclose(coef.psi, expected_psi, degree)


@pytest.mark.parametrize(
    "b",
    [
        jnp.asarray([[0.75]], dtype=jnp.float64),
        jnp.asarray([[0.80], [-0.35], [0.20]], dtype=jnp.float64),
    ],
)
def test_psi_nonempty_matches_closed_form_when_lambda_zero(b):
    q = int(b.shape[0])
    trunc = 5

    ker = FSSK.from_matrix(
        Lambda=jnp.zeros((1, 1), dtype=jnp.float64),
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.05, 0.3, 1.7], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=trunc, dtype=jnp.float64)

    b_vec = _np(b[:, 0])
    _, degree, prod = _multiindex_products(coef.layout, b_vec)
    denom = _factorial_degree_plus_one(degree)

    expected_psi = prod / denom
    expected_psi = np.broadcast_to(expected_psi[None, :, None], coef.psi.shape)

    _assert_psi_nonempty_allclose(coef.psi, expected_psi, degree)


@pytest.mark.parametrize(
    "b",
    [
        jnp.asarray([[0.75]], dtype=jnp.float64),
        jnp.asarray([[0.80], [-0.35], [0.20]], dtype=jnp.float64),
    ],
)
def test_phi_matches_closed_form_when_lambda_zero(b):
    q = int(b.shape[0])
    trunc = 5

    ker = FSSK.from_matrix(
        Lambda=jnp.zeros((1, 1), dtype=jnp.float64),
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.05, 0.3, 1.7], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=trunc, dtype=jnp.float64)

    b_vec = _np(b[:, 0])
    _, degree, prod = _multiindex_products(coef.layout, b_vec)
    denom = _factorial_degree_plus_one(degree)

    mphi = coef.phi.shape[-3]
    expected_phi = b_vec[:, None] * (prod[:mphi] / denom[:mphi])[None, :]
    expected_phi = np.broadcast_to(expected_phi[None, :, :, None, None], coef.phi.shape)

    _assert_allclose(coef.phi, expected_phi, atol=PHI_ATOL, rtol=PHI_RTOL)


def test_E_matches_closed_form_for_positive_scalar_lambda():
    lam = 0.65
    b = jnp.asarray([[0.60], [-0.25]], dtype=jnp.float64)
    q = int(b.shape[0])

    ker = FSSK.from_matrix(
        Lambda=jnp.asarray([[lam]], dtype=jnp.float64),
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.1, 0.4, 1.2], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=5, dtype=jnp.float64)

    expected_E = np.exp(-lam * _np(dt))[:, None, None]
    _assert_allclose(coef.E, expected_E, atol=1e-12, rtol=1e-12)


def test_psi_empty_matches_closed_form_for_positive_scalar_lambda():
    lam = 0.65
    b = jnp.asarray([[0.60], [-0.25]], dtype=jnp.float64)
    q = int(b.shape[0])

    ker = FSSK.from_matrix(
        Lambda=jnp.asarray([[lam]], dtype=jnp.float64),
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.1, 0.4, 1.2], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=5, dtype=jnp.float64)

    _, degree, _ = _multiindex_products(coef.layout, _np(b[:, 0]))

    dt_np = _np(dt)
    expected_psi = np.zeros_like(_np(coef.psi))
    expected_psi[:, 0, 0] = (1.0 - np.exp(-lam * dt_np)) / (lam * dt_np)

    _assert_psi_empty_allclose(coef.psi, expected_psi, degree)


def test_psi_nonempty_matches_closed_form_for_positive_scalar_lambda():
    lam = 0.65
    b = jnp.asarray([[0.60], [-0.25]], dtype=jnp.float64)
    q = int(b.shape[0])
    trunc = 5

    ker = FSSK.from_matrix(
        Lambda=jnp.asarray([[lam]], dtype=jnp.float64),
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.1, 0.4, 1.2], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=trunc, dtype=jnp.float64)

    b_vec = _np(b[:, 0])
    _, degree, prod = _multiindex_products(coef.layout, b_vec)
    dt_np = _np(dt)

    expected_psi = np.empty(coef.psi.shape, dtype=np.float64)
    for ti, h in enumerate(dt_np):
        x = lam * float(h)
        for idx, n in enumerate(degree):
            expected_psi[ti, idx, 0] = prod[idx] * _scalar_positive_lambda_psi_hat(x, int(n))

    _assert_psi_nonempty_allclose(coef.psi, expected_psi, degree)


def test_phi_matches_closed_form_for_positive_scalar_lambda():
    lam = 0.65
    b = jnp.asarray([[0.60], [-0.25]], dtype=jnp.float64)
    q = int(b.shape[0])
    trunc = 5

    ker = FSSK.from_matrix(
        Lambda=jnp.asarray([[lam]], dtype=jnp.float64),
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.1, 0.4, 1.2], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=trunc, dtype=jnp.float64)

    b_vec = _np(b[:, 0])
    _, degree, prod = _multiindex_products(coef.layout, b_vec)
    dt_np = _np(dt)

    mphi = coef.phi.shape[-3]
    expected_phi = np.empty(coef.phi.shape, dtype=np.float64)

    for ti, h in enumerate(dt_np):
        x = lam * float(h)
        exp_factor = math.exp(-x)
        for p in range(q):
            for idx, n in enumerate(degree[:mphi]):
                expected_phi[ti, p, idx, 0, 0] = (
                    b_vec[p]
                    * prod[idx]
                    * exp_factor
                    / math.factorial(int(n) + 1)
                )

    _assert_allclose(coef.phi, expected_phi, atol=PHI_ATOL, rtol=PHI_RTOL)


def test_E_matches_block_exponential_reference_dense_matrix():
    Lambda = jnp.asarray(
        [
            [0.80, -0.15],
            [0.25, 0.50],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray(
        [
            [0.70, -0.20],
            [0.15, 0.45],
        ],
        dtype=jnp.float64,
    )
    q = int(b.shape[0])

    ker = FSSK.from_matrix(
        Lambda=Lambda,
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.2, 0.9], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=3, dtype=jnp.float64)

    expected_E = np.stack([_np(jsp_linalg.expm(-h * Lambda)) for h in _np(dt)], axis=0)
    _assert_allclose(coef.E, expected_E, atol=1e-12, rtol=1e-12)


def test_psi_empty_matches_block_exponential_reference_dense_matrix():
    Lambda = jnp.asarray(
        [
            [0.80, -0.15],
            [0.25, 0.50],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray(
        [
            [0.70, -0.20],
            [0.15, 0.45],
        ],
        dtype=jnp.float64,
    )
    q = int(b.shape[0])
    r = int(b.shape[1])

    ker = FSSK.from_matrix(
        Lambda=Lambda,
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.2, 0.9], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=3, dtype=jnp.float64)

    Lambda_np = _np(Lambda)
    ell = _np(coef.layout.ell).astype(np.int64)
    degree = _np(coef.layout.degree).astype(np.int64)
    one_row = np.ones((1, r), dtype=np.float64)

    expected_psi = np.zeros_like(_np(coef.psi))

    for ti, h in enumerate(_np(dt)):
        word = _word_from_multiindex(ell[0])
        psi_mat = _psi_word_block_expm(Lambda_np, _np(b), word, float(h))
        expected_psi[ti, 0] = (one_row @ psi_mat / float(h))[0]

    _assert_psi_empty_allclose(coef.psi, expected_psi, degree)


def test_psi_nonempty_matches_block_exponential_reference_dense_matrix():
    Lambda = jnp.asarray(
        [
            [0.80, -0.15],
            [0.25, 0.50],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray(
        [
            [0.70, -0.20],
            [0.15, 0.45],
        ],
        dtype=jnp.float64,
    )
    q = int(b.shape[0])
    r = int(b.shape[1])
    trunc = 3

    ker = FSSK.from_matrix(
        Lambda=Lambda,
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.2, 0.9], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=trunc, dtype=jnp.float64)

    Lambda_np = _np(Lambda)
    b_np = _np(b)
    ell = _np(coef.layout.ell).astype(np.int64)
    degree = _np(coef.layout.degree).astype(np.int64)
    one_row = np.ones((1, r), dtype=np.float64)

    expected_psi = np.empty(coef.psi.shape, dtype=np.float64)

    for ti, h in enumerate(_np(dt)):
        for idx, multi in enumerate(ell):
            word = _word_from_multiindex(multi)
            psi_mat = _psi_word_block_expm(Lambda_np, b_np, word, float(h))
            expected_psi[ti, idx] = (one_row @ psi_mat / (float(h) ** (int(degree[idx]) + 1)))[0]

    _assert_psi_nonempty_allclose(coef.psi, expected_psi, degree)


def test_phi_matches_block_exponential_reference_dense_matrix():
    Lambda = jnp.asarray(
        [
            [0.80, -0.15],
            [0.25, 0.50],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray(
        [
            [0.70, -0.20],
            [0.15, 0.45],
        ],
        dtype=jnp.float64,
    )
    q = int(b.shape[0])
    trunc = 3

    ker = FSSK.from_matrix(
        Lambda=Lambda,
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.2, 0.9], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=trunc, dtype=jnp.float64)

    Lambda_np = _np(Lambda)
    b_np = _np(b)
    ell = _np(coef.layout.ell).astype(np.int64)

    expected_phi = np.empty(coef.phi.shape, dtype=np.float64)
    mphi = coef.phi.shape[-3]

    for ti, h in enumerate(_np(dt)):
        for p in range(q):
            for idx, multi in enumerate(ell[:mphi]):
                word = [p] + _word_from_multiindex(multi)
                phi_mat = _phi_word_block_expm(Lambda_np, b_np, word, float(h))
                expected_phi[ti, p, idx] = phi_mat / (float(h) ** len(word))

    _assert_allclose(coef.phi, expected_phi, atol=PHI_ATOL, rtol=PHI_RTOL)


def test_E_matches_confluent_formula_for_R2_diagonal_lambda():
    lambdas = np.asarray([0.45, 1.30], dtype=np.float64)

    Lambda = jnp.diag(jnp.asarray(lambdas, dtype=jnp.float64))
    b = jnp.asarray(
        [
            [0.70, -0.20],
            [0.15, 0.45],
        ],
        dtype=jnp.float64,
    )
    q = int(b.shape[0])

    ker = FSSK.from_matrix(
        Lambda=Lambda,
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.17, 0.80], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=4, dtype=jnp.float64)

    expected_E = np.stack([np.diag(np.exp(-lambdas * float(h))) for h in _np(dt)], axis=0)
    _assert_allclose(coef.E, expected_E, atol=1e-12, rtol=1e-12)


def test_psi_empty_matches_confluent_formula_for_R2_diagonal_lambda():
    lambdas = np.asarray([0.45, 1.30], dtype=np.float64)

    Lambda = jnp.diag(jnp.asarray(lambdas, dtype=jnp.float64))
    b = jnp.asarray(
        [
            [0.70, -0.20],
            [0.15, 0.45],
        ],
        dtype=jnp.float64,
    )
    q = int(b.shape[0])

    ker = FSSK.from_matrix(
        Lambda=Lambda,
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.17, 0.80], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=4, dtype=jnp.float64)

    degree = _np(coef.layout.degree).astype(np.int64)

    expected_psi = np.zeros_like(_np(coef.psi))
    for ti, h in enumerate(_np(dt)):
        psi = _diag_psi_row_word_confluent(lambdas, _np(b), [], float(h))
        expected_psi[ti, 0] = psi / float(h)

    _assert_psi_empty_allclose(coef.psi, expected_psi, degree)


def test_psi_nonempty_matches_confluent_formula_for_R2_diagonal_lambda():
    lambdas = np.asarray([0.45, 1.30], dtype=np.float64)

    Lambda = jnp.diag(jnp.asarray(lambdas, dtype=jnp.float64))
    b = jnp.asarray(
        [
            [0.70, -0.20],
            [0.15, 0.45],
        ],
        dtype=jnp.float64,
    )
    q = int(b.shape[0])
    trunc = 4

    ker = FSSK.from_matrix(
        Lambda=Lambda,
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.17, 0.80], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=trunc, dtype=jnp.float64)

    ell = _np(coef.layout.ell).astype(np.int64)
    degree = _np(coef.layout.degree).astype(np.int64)
    b_np = _np(b)

    expected_psi = np.empty(coef.psi.shape, dtype=np.float64)

    for ti, h in enumerate(_np(dt)):
        for idx, multi in enumerate(ell):
            word = _word_from_multiindex(multi)
            psi = _diag_psi_row_word_confluent(lambdas, b_np, word, float(h))
            expected_psi[ti, idx] = psi / (float(h) ** (len(word) + 1))

    _assert_psi_nonempty_allclose(coef.psi, expected_psi, degree)


def test_phi_matches_confluent_formula_for_R2_diagonal_lambda():
    lambdas = np.asarray([0.45, 1.30], dtype=np.float64)

    Lambda = jnp.diag(jnp.asarray(lambdas, dtype=jnp.float64))
    b = jnp.asarray(
        [
            [0.70, -0.20],
            [0.15, 0.45],
        ],
        dtype=jnp.float64,
    )
    q = int(b.shape[0])
    trunc = 4

    ker = FSSK.from_matrix(
        Lambda=Lambda,
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=QUAD_ORDER,
    )

    dt = jnp.asarray([0.17, 0.80], dtype=jnp.float64)
    coef = ker.coef(dt, trunc=trunc, dtype=jnp.float64)

    ell = _np(coef.layout.ell).astype(np.int64)
    b_np = _np(b)

    expected_phi = np.zeros(coef.phi.shape, dtype=np.float64)
    mphi = coef.phi.shape[-3]

    for ti, h in enumerate(_np(dt)):
        for p in range(q):
            for idx, multi in enumerate(ell[:mphi]):
                word = [p] + _word_from_multiindex(multi)
                phi = _diag_phi_word_confluent(lambdas, b_np, word, float(h))
                expected_phi[ti, p, idx] = phi / (float(h) ** len(word))

    _assert_allclose(coef.phi, expected_phi, atol=PHI_ATOL, rtol=PHI_RTOL)


def test_coef_batched_dt_matches_loop_over_scalar_dt():
    Lambda = jnp.asarray(
        [
            [1.00, 0.20],
            [-0.10, 0.70],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray(
        [
            [0.70, -0.20],
            [0.10, 0.40],
        ],
        dtype=jnp.float64,
    )
    q = int(b.shape[0])
    trunc = 4

    ker = FSSK.from_matrix(
        Lambda=Lambda,
        A=jnp.ones((q, 1, 1), dtype=jnp.float64),
        b=b,
        quad_order=48,
    )

    dt = jnp.asarray([0.17, 0.43], dtype=jnp.float64)
    batched = ker.coef(dt, trunc=trunc, dtype=jnp.float64)
    scalar = [ker.coef(h, trunc=trunc, dtype=jnp.float64) for h in dt]

    expected_E = jnp.stack([c.E for c in scalar], axis=0)
    expected_psi = jnp.stack([c.psi for c in scalar], axis=0)
    expected_phi = jnp.stack([c.phi for c in scalar], axis=0)

    _assert_allclose(batched.E, expected_E, atol=1e-12, rtol=1e-12)
    _assert_allclose(batched.psi, expected_psi, atol=1e-12, rtol=1e-12)
    _assert_allclose(batched.phi, expected_phi, atol=1e-12, rtol=1e-12)