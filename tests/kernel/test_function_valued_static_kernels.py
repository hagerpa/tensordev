"""RBF_SQR_Kernel and RBF_CEXP_Kernel act on node values, like every static kernel."""
from dataclasses import dataclass

import jax
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from tensordev import FSSKSigKernel
from tensordev.kernel import RBF_CEXP_Kernel, RBF_SQR_Kernel, SigKernel, StaticKernel
from tensordev.kernel.static_kernels import cos_exp_kernel
from tensordev.sss import StateSpaceSignature
from tensordev.util import random_trigonometric_polynomial_paths


def _sqr_reference(x, y, sigma1, sigma2):
    return np.exp(-np.sum((x - y) ** 2) / sigma1) * np.exp(-np.sum((x ** 2 - y ** 2) ** 2) / sigma2)


def _cexp_features(x, channels, n_freqs, sigma):
    f = x.reshape(-1, channels)                      # (length_x, channels)
    grid = np.linspace(0.0, 1.0, f.shape[0])
    diff = grid[:, None] - grid[None, :]
    T = np.asarray(cos_exp_kernel(jnp.asarray(diff), n_freqs=n_freqs, sigma=sigma))
    return (T.T @ f / f.shape[0]).ravel()            # (1/n) sum_i f(x_i) T(x_i, x_j)


def _cexp_reference(x, y, sigma1, sigma2, n_freqs, channels):
    fx = _cexp_features(x, channels, n_freqs, sigma1)
    fy = _cexp_features(y, channels, n_freqs, sigma1)
    return np.exp(-np.sum((fx - fy) ** 2) / sigma2)


@pytest.fixture
def nodes():
    rng = np.random.default_rng(0)
    return rng.normal(size=(3, 5, 8)), rng.normal(size=(3, 4, 8))


def test_rbf_sqr_batch_kernel_is_pointwise(nodes):
    X, Y = nodes
    k = RBF_SQR_Kernel(sigma1=2.0, sigma2=3.0)
    got = np.asarray(k.batch_kernel(jnp.asarray(X), jnp.asarray(Y)))
    want = np.array([[[_sqr_reference(X[b, s], Y[b, t], 2.0, 3.0) for t in range(4)] for s in range(5)]
                     for b in range(3)])
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_rbf_cexp_batch_kernel_is_pointwise(nodes):
    X, Y = nodes
    k = RBF_CEXP_Kernel(sigma1=0.5, sigma2=4.0, n_freqs=5, channels=2)
    got = np.asarray(k.batch_kernel(jnp.asarray(X), jnp.asarray(Y)))
    want = np.array([[[_cexp_reference(X[b, s], Y[b, t], 0.5, 4.0, 5, 2) for t in range(4)] for s in range(5)]
                     for b in range(3)])
    np.testing.assert_allclose(got, want, rtol=1e-12)


@pytest.mark.parametrize("kernel", [RBF_SQR_Kernel(sigma1=2.0, sigma2=3.0),
                                    RBF_CEXP_Kernel(sigma1=0.5, sigma2=4.0, n_freqs=5, channels=2)])
def test_gram_matrix_matches_batch_kernel(kernel, nodes):
    X, Y = (jnp.asarray(a) for a in nodes)
    gram = kernel.Gram_matrix(X, Y)                   # (3, 3, 5, 4)
    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(gram[i, j], kernel.batch_kernel(X[i], Y[j]), rtol=1e-12)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Pointwise(StaticKernel):
    """Reference static kernel built from a scalar function of two node values."""

    inner: StaticKernel

    def _one(self, x, y):
        return self.inner.batch_kernel(x[None, :], y[None, :])[0, 0]

    def batch_kernel(self, X, Y):
        pair = jax.vmap(jax.vmap(self._one, (None, 0)), (0, None))
        for _ in range(X.ndim - 2):
            pair = jax.vmap(pair)
        return pair(X, Y)

    def Gram_matrix(self, X, Y):
        f = self.batch_kernel
        for _ in range(Y.ndim - 2):
            f = jax.vmap(f, (None, 0))
        for _ in range(X.ndim - 2):
            f = jax.vmap(f, (0, None))
        return f(X, Y)


@pytest.mark.parametrize("kernel", [RBF_SQR_Kernel(sigma1=4.0, sigma2=8.0),
                                    RBF_CEXP_Kernel(sigma1=0.5, sigma2=4.0, n_freqs=5, channels=2)])
def test_sig_kernel_uses_node_values(kernel):
    # function-valued paths: 16 times, 3 grid points, 2 channels, flattened to 6 coordinates
    F = random_trigonometric_polynomial_paths(batch=6, steps=16, dim=6, key=0).reshape(2, 3, 17, 6)
    X, Y = F[0], F[1]
    got = SigKernel(static_kernel=kernel).compute_kernel(X, Y)
    want = SigKernel(static_kernel=_Pointwise(kernel)).compute_kernel(X, Y)
    np.testing.assert_allclose(got, want, rtol=1e-10)
    assert np.ptp(np.asarray(got)) > 1e-3             # different pairs give different values


@pytest.mark.parametrize("kernel", [RBF_SQR_Kernel(sigma1=4.0, sigma2=8.0),
                                    RBF_CEXP_Kernel(sigma1=0.5, sigma2=4.0, n_freqs=5)])
def test_fssk_sig_kernel_accepts_them(kernel):
    X = random_trigonometric_polynomial_paths(batch=3, steps=16, dim=2, key=1)
    sss = StateSpaceSignature.from_jordan(real_rates=jnp.array([1.0]), real_sizes=(1,),
                                          A=jnp.eye(2)[None], b=jnp.ones((1, 1)), trunc=3)
    G = FSSKSigKernel(kernel=sss.kernel, static_kernel=kernel, dt_x=1 / 16, dt_y=1 / 16).compute_Gram(X, X)
    assert np.all(np.isfinite(G))
    np.testing.assert_allclose(G, G.T, rtol=1e-10)


def test_rbf_cexp_rejects_incompatible_channels():
    with pytest.raises(ValueError, match="channels"):
        RBF_CEXP_Kernel(channels=3).batch_kernel(jnp.ones((2, 4)), jnp.ones((2, 4)))
