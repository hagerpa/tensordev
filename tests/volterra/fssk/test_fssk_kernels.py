from __future__ import annotations

from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tensordev.sss import DenseLambda, FSSKCoefficients, FSSK, JordanLambda


def _assert_allclose(x, y, *, atol=1e-9, rtol=1e-9):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    assert x.shape == y.shape
    assert jnp.allclose(x, y, atol=atol, rtol=rtol), (x, y)


def _sample_jordan() -> JordanLambda:
    return JordanLambda(
        real_rates=jnp.asarray([0.7]),
        real_sizes=jnp.asarray([2]),
        osc_decays=jnp.asarray([0.3]),
        osc_freqs=jnp.asarray([1.1]),
        osc_sizes=jnp.asarray([1]),
    )


def test_fssk_kernel_q1_uses_unified_layout_and_shapes():
    lam = DenseLambda(jnp.asarray([[1.0, 0.0], [0.0, 0.5]]))
    A = jnp.ones((1, 2, 3))
    b = jnp.asarray([[0.7, -0.2]])
    ker = FSSK(Lambda=lam, A=A, b=b)

    coef = ker.coef(jnp.asarray([0.1, 0.2, 0.3]), trunc=4)
    assert isinstance(coef, FSSKCoefficients)
    assert coef.layout.q == 1
    assert coef.layout.trunc == 3
    assert coef.E.shape == (3, 2, 2)
    assert coef.psi.shape == (3, 4, 2)
    assert coef.phi.shape == (3, 1, 3, 2, 2)


def test_fssk_kernel_multivariate_shapes_for_dt_array():
    lam = DenseLambda(jnp.asarray([[1.0, 0.0], [0.0, 0.5]]))
    A = jnp.ones((2, 2, 3))
    b = jnp.asarray([[0.7, -0.2], [0.1, 0.4]])
    ker = FSSK(Lambda=lam, A=A, b=b)

    coef = ker.coef(jnp.asarray([0.1, 0.2]), trunc=4)
    assert coef.layout.q == 2
    assert coef.layout.trunc == 3
    assert coef.E.shape == (2, 2, 2)
    assert coef.psi.shape == (2, 10, 2)
    assert coef.phi.shape == (2, 2, 6, 2, 2)


def test_shared_coef_algorithm_matches_dense_and_jordan_paths():
    jordan = _sample_jordan()
    dense = DenseLambda(jordan.matrix())

    A = jnp.ones((2, 1, 1))
    b = jnp.asarray([[0.8, 0.1, -0.3, 0.2], [0.2, -0.1, 0.4, 0.6]])

    ker_dense = FSSK(Lambda=dense, A=A, b=b)
    ker_jordan = FSSK(Lambda=jordan, A=A, b=b)

    dt = jnp.asarray([0.1, 0.3])
    coef_dense = ker_dense.coef(dt, trunc=3)
    coef_jordan = ker_jordan.coef(dt, trunc=3)

    _assert_allclose(coef_dense.E, coef_jordan.E)
    _assert_allclose(coef_dense.psi, coef_jordan.psi)
    _assert_allclose(coef_dense.phi, coef_jordan.phi)


def test_fssk_kernel_coef_is_jittable_when_trunc_is_static():
    lam = DenseLambda(jnp.asarray([[1.0, 0.0], [0.0, 0.5]]))
    A = jnp.ones((2, 2, 3))
    b = jnp.asarray([[0.7, -0.2], [0.1, 0.4]])
    ker = FSSK(Lambda=lam, A=A, b=b)

    @partial(jax.jit, static_argnames=("trunc",))
    def build(kernel, dt, *, trunc):
        return kernel.coef(dt, trunc=trunc)

    coef = build(ker, jnp.asarray([0.1, 0.2]), trunc=4)
    assert isinstance(coef, FSSKCoefficients)
    assert coef.psi.shape == (2, 10, 2)
    assert coef.phi.shape == (2, 2, 6, 2, 2)


def test_from_jordan_constructs_structured_kernel_with_explicit_b():
    A = jnp.ones((2, 1, 1))
    jordan = JordanLambda(real_rates=(0.7, 1.2), real_sizes=(1, 1))
    b = jnp.asarray([[1.0, -0.5], [0.25, 0.75]])

    ker = FSSK.from_jordan(
        A=A,
        b=b,
        real_rates=(0.7, 1.2),
        real_sizes=(1, 1),
    )

    assert isinstance(ker.Lambda, JordanLambda)
    coef = ker.coef(jnp.asarray([0.1, 0.2]), trunc=3)
    assert isinstance(coef, FSSKCoefficients)
    assert coef.E.shape == (2, 2, 2)
