"""RBF_SQR_Kernel and RBF_CEXP_Kernel against their formulas on single node values."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from tensordev.kernel import RBF_CEXP_Kernel, RBF_SQR_Kernel
from tensordev.kernel.static_kernels import cexp

SQR = RBF_SQR_Kernel(sigma1=2.0, sigma2=3.0)
CEXP = RBF_CEXP_Kernel(sigma1=0.5, sigma2=4.0, n_freqs=5, channels=2)


def _sqr(x, y):
    return np.exp(-np.sum((x - y) ** 2) / SQR.sigma1 - np.sum((x ** 2 - y ** 2) ** 2) / SQR.sigma2)


def _transform(x):
    """:math:`Tx` of one node value, flattened grid-major."""
    return np.asarray(cexp(jnp.asarray(x).reshape(-1, CEXP.channels), CEXP.n_freqs, CEXP.sigma1)).ravel()


def _cexp(x, y):
    return np.exp(-np.sum((_transform(x) - _transform(y)) ** 2) / CEXP.sigma2)


@pytest.fixture
def nodes():
    rng = np.random.default_rng(0)
    return rng.normal(size=(3, 5, 8)), rng.normal(size=(3, 4, 8))


@pytest.mark.parametrize("kernel, formula", [(SQR, _sqr), (CEXP, _cexp)], ids=["sqr", "cexp"])
def test_batch_kernel_is_the_formula_on_node_pairs(kernel, formula, nodes):
    X, Y = nodes
    want = [[[formula(x, y) for y in Yb] for x in Xb] for Xb, Yb in zip(X, Y)]
    np.testing.assert_allclose(kernel.batch_kernel(jnp.asarray(X), jnp.asarray(Y)), want, rtol=1e-12)


@pytest.mark.parametrize("kernel", [SQR, CEXP], ids=["sqr", "cexp"])
def test_gram_matrix_matches_batch_kernel(kernel, nodes):
    X, Y = (jnp.asarray(a) for a in nodes)
    gram = kernel.Gram_matrix(X, Y)
    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(gram[i, j], kernel.batch_kernel(X[i], Y[j]), rtol=1e-12)


def test_rbf_cexp_rejects_incompatible_channels():
    with pytest.raises(ValueError, match="channels"):
        RBF_CEXP_Kernel(channels=3).batch_kernel(jnp.ones((2, 4)), jnp.ones((2, 4)))
