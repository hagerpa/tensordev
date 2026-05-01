import numpy as np
import pytest

import jax
import jax.numpy as jnp

from tensordev.sss.rough_approx import fractional_fssk
from tensordev.sss.rough_approx import _bl2_quadrature_rule


jax.config.update("jax_enable_x64", True)


def test_fractional_fssk_returns_expected_fssk_shapes():
    dtype = jnp.float64
    beta = 0.75
    R = 3
    A = jnp.eye(2, dtype=dtype)[None, :, :]

    kernel = fractional_fssk(
        beta=beta,
        R=R,
        A=A,
        T=1.0,
        coef_quad_order=16,
    )

    assert kernel.q == 1
    assert kernel.m == 2
    assert kernel.path_dim == 2
    assert kernel.state_dim == R
    assert kernel.A.shape == (1, 2, 2)
    assert kernel.b.shape == (1, R)


def test_bl2_nodes_and_weights_are_finite_sorted_and_correct_size():
    beta = 0.75
    R = 4

    nodes, weights = _bl2_quadrature_rule(beta=beta, R=R, T=1.0)

    assert nodes.shape == (R,)
    assert weights.shape == (R,)
    assert np.all(np.isfinite(nodes))
    assert np.all(np.isfinite(weights))
    assert np.all(nodes >= 0.0)
    assert np.all(nodes[1:] >= nodes[:-1])


def test_bl2_exponential_sum_approximates_fractional_kernel_on_grid():
    beta = 0.75
    R = 4
    T = 1.0

    nodes, weights = _bl2_quadrature_rule(beta=beta, R=R, T=T)

    t = np.linspace(0.1, T, 8)
    target = t ** (beta - 1.0) / np.array(float(jax.scipy.special.gamma(beta)))
    approx = np.exp(-np.outer(t, nodes)) @ weights

    rel = np.max(np.abs(approx - target) / np.maximum(np.abs(target), 1e-12))
    assert rel < 0.2


def test_fractional_fssk_validates_beta_range():
    A = jnp.ones((1, 1, 1), dtype=jnp.float64)

    with pytest.raises(ValueError, match="beta"):
        fractional_fssk(beta=0.5, R=2, A=A)

    with pytest.raises(ValueError, match="beta"):
        fractional_fssk(beta=1.0, R=2, A=A)


def test_fractional_fssk_validates_R_and_T():
    A = jnp.ones((1, 1, 1), dtype=jnp.float64)

    with pytest.raises(ValueError, match="R"):
        fractional_fssk(beta=0.75, R=0, A=A)

    with pytest.raises(ValueError, match="T"):
        fractional_fssk(beta=0.75, R=2, T=0.0, A=A)


def test_fractional_fssk_validates_A_shape():
    with pytest.raises(ValueError, match="A must have shape"):
        fractional_fssk(
            beta=0.75,
            R=2,
            A=jnp.ones((2, 2), dtype=jnp.float64),
        )