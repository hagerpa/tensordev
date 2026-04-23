"""
Tests for tensordev.kernel.static_kernels.

Coverage
--------
* Shape correctness for empty, 1-D, and multi-D batch prefixes
* Mathematical correctness against NumPy references
* Gram-matrix symmetry (K(X, X) == K(X, X).T element-wise)
* Positive semi-definiteness of RBF Gram matrices
* batch_kernel / Gram_matrix consistency  (batch diag == Gram diagonal)
* JAX pytree round-trip (register_dataclass)
* jax.jit compatibility
* cos_exp_kernel and cexp helper functions
"""

from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from tensordev.kernel.static_kernels import (
    LinearKernel,
    RBFKernel,
    RBF_CEXP_Kernel,
    RBF_SQR_Kernel,
    cos_exp_kernel,
    cexp,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KEY = jr.PRNGKey(0)


def _split(n=3):
    keys = jr.split(KEY, n)
    return keys


# ---------------------------------------------------------------------------
# Shape parametrisation
# ---------------------------------------------------------------------------

BATCH_SHAPES = [
    pytest.param((), id="batch=()"),
    pytest.param((4,), id="batch=(4,)"),
    pytest.param((2, 3), id="batch=(2,3)"),
]


def _rand(key, batch, length, dim):
    return jr.normal(key, batch + (length, dim), dtype=jnp.float64)


def _rand_fn(key, batch, length_t, length_x, dim):
    """Random function-space input: batch + (length_t, length_x, dim)."""
    return jr.normal(key, batch + (length_t, length_x, dim), dtype=jnp.float64)


# ===========================================================================
# LinearKernel
# ===========================================================================

class TestLinearKernel:

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_batch_kernel_shape(self, batch):
        k = LinearKernel(scale=1.0)
        keys = _split(2)
        X = _rand(keys[0], batch, 5, 3)
        Y = _rand(keys[1], batch, 7, 3)
        out = k.batch_kernel(X, Y)
        assert out.shape == batch + (5, 7)

    @pytest.mark.parametrize("batch_X", [(), (2,), (2, 3)])
    @pytest.mark.parametrize("batch_Y", [(), (4,), (3,)])
    def test_gram_matrix_shape(self, batch_X, batch_Y):
        k = LinearKernel(scale=1.0)
        keys = _split(2)
        X = _rand(keys[0], batch_X, 5, 3)
        Y = _rand(keys[1], batch_Y, 7, 3)
        out = k.Gram_matrix(X, Y)
        assert out.shape == batch_X + batch_Y + (5, 7)

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_correctness_vs_numpy(self, batch):
        """k(x, y) = scale² * (x · y) for each pair of points."""
        scale = 2.0
        k = LinearKernel(scale=scale)
        keys = _split(2)
        X = _rand(keys[0], batch, 4, 3)
        Y = _rand(keys[1], batch, 6, 3)
        got = np.asarray(k.batch_kernel(X, Y))
        ref = scale ** 2 * np.einsum("...pd,...qd->...pq", X, Y)
        np.testing.assert_allclose(got, ref, rtol=1e-12)

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_batch_kernel_symmetry(self, batch):
        """batch_kernel(X, X) must be symmetric in its last two axes."""
        k = LinearKernel(scale=1.5)
        X = _rand(_split(1)[0], batch, 5, 3)
        G = np.asarray(k.batch_kernel(X, X))
        np.testing.assert_allclose(G, G.swapaxes(-1, -2), atol=1e-14)

    def test_gram_symmetry_empty_batch(self):
        """Gram_matrix(X, X) is symmetric when batch is empty: G[p,q] == G[q,p]."""
        k = LinearKernel(scale=1.5)
        X = _rand(_split(1)[0], (), 5, 3)
        G = np.asarray(k.Gram_matrix(X, X))
        np.testing.assert_allclose(G, G.T, atol=1e-14)

    def test_gram_symmetry_single_batch(self):
        """Gram_matrix(X, X) satisfies G[i,j,p,q] == G[j,i,q,p]."""
        k = LinearKernel(scale=1.5)
        X = _rand(_split(1)[0], (4,), 5, 3)
        G = np.asarray(k.Gram_matrix(X, X))   # (4, 4, 5, 5)
        G_sym = G.transpose(1, 0, 3, 2)        # swap batch i↔j and point p↔q
        np.testing.assert_allclose(G, G_sym, atol=1e-14)

    def test_gram_diagonal_blocks_match_batch_kernel(self):
        """Gram_matrix(X,X)[i,i,:,:] == batch_kernel(X,X)[i,:,:] for each i."""
        k = LinearKernel(scale=1.0)
        X = _rand(_split(1)[0], (4,), 5, 3)
        batch_K = np.asarray(k.batch_kernel(X, X))   # (4, 5, 5)
        gram_K  = np.asarray(k.Gram_matrix(X, X))    # (4, 4, 5, 5)
        for i in range(4):
            np.testing.assert_allclose(batch_K[i], gram_K[i, i], rtol=1e-12)

    def test_pytree_roundtrip(self):
        k = LinearKernel(scale=3.0)
        leaves, treedef = jax.tree_util.tree_flatten(k)
        k2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert k2.scale == k.scale

    def test_jit(self):
        k = LinearKernel(scale=1.0)
        X = _rand(KEY, (3,), 4, 2)
        Y = _rand(jr.PRNGKey(1), (3,), 6, 2)
        f = jax.jit(k.batch_kernel)
        out = f(X, Y)
        assert out.shape == (3, 4, 6)


# ===========================================================================
# RBFKernel
# ===========================================================================

class TestRBFKernel:

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_batch_kernel_shape(self, batch):
        k = RBFKernel(sigma=1.0)
        keys = _split(2)
        X = _rand(keys[0], batch, 5, 3)
        Y = _rand(keys[1], batch, 7, 3)
        assert k.batch_kernel(X, Y).shape == batch + (5, 7)

    @pytest.mark.parametrize("batch_X", [(), (2,), (2, 3)])
    @pytest.mark.parametrize("batch_Y", [(), (4,), (3,)])
    def test_gram_matrix_shape(self, batch_X, batch_Y):
        k = RBFKernel(sigma=1.0)
        X = _rand(_split(1)[0], batch_X, 5, 3)
        Y = _rand(_split(2)[1], batch_Y, 7, 3)
        assert k.Gram_matrix(X, Y).shape == batch_X + batch_Y + (5, 7)

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_diagonal_is_one(self, batch):
        """k(x, x) = exp(0) = 1 for every point."""
        k = RBFKernel(sigma=2.0)
        X = _rand(_split(1)[0], batch, 5, 3)
        diag = np.diagonal(np.asarray(k.batch_kernel(X, X)), axis1=-2, axis2=-1)
        np.testing.assert_allclose(diag, 1.0, atol=1e-14)

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_correctness_vs_numpy(self, batch):
        sigma = 1.5
        k = RBFKernel(sigma=sigma)
        keys = _split(2)
        X = _rand(keys[0], batch, 4, 3)
        Y = _rand(keys[1], batch, 6, 3)
        got = np.asarray(k.batch_kernel(X, Y))
        # reference via broadcasting
        diff = X[..., :, None, :] - Y[..., None, :, :]  # batch + (4, 6, 3)
        ref = np.exp(-np.sum(diff ** 2, axis=-1) / sigma)
        np.testing.assert_allclose(got, ref, rtol=1e-12)

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_batch_kernel_symmetry(self, batch):
        """batch_kernel(X, X) must be symmetric in its last two axes."""
        k = RBFKernel(sigma=1.0)
        X = _rand(_split(1)[0], batch, 5, 3)
        G = np.asarray(k.batch_kernel(X, X))
        np.testing.assert_allclose(G, G.swapaxes(-1, -2), atol=1e-14)

    def test_gram_symmetry_empty_batch(self):
        k = RBFKernel(sigma=1.0)
        X = _rand(_split(1)[0], (), 5, 3)
        G = np.asarray(k.Gram_matrix(X, X))
        np.testing.assert_allclose(G, G.T, atol=1e-14)

    def test_gram_symmetry_single_batch(self):
        """Gram_matrix(X, X) satisfies G[i,j,p,q] == G[j,i,q,p]."""
        k = RBFKernel(sigma=1.0)
        X = _rand(_split(1)[0], (4,), 5, 3)
        G = np.asarray(k.Gram_matrix(X, X))   # (4, 4, 5, 5)
        G_sym = G.transpose(1, 0, 3, 2)
        np.testing.assert_allclose(G, G_sym, atol=1e-14)

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_positive_semi_definite(self, batch):
        """batch_kernel(X, X) must have non-negative eigenvalues for each batch."""
        k = RBFKernel(sigma=1.0)
        X = _rand(_split(1)[0], batch, 8, 3)
        G = np.asarray(k.batch_kernel(X, X))   # batch + (8, 8)
        G_flat = G.reshape((-1,) + G.shape[-2:]) if G.ndim > 2 else G[None]
        for mat in G_flat:
            eigvals = np.linalg.eigvalsh(mat)
            assert eigvals.min() >= -1e-10, f"PSD violated: min eigval = {eigvals.min()}"

    def test_gram_diagonal_blocks_match_batch_kernel(self):
        """Gram_matrix(X,X)[i,i,:,:] == batch_kernel(X,X)[i,:,:] for each i."""
        k = RBFKernel(sigma=1.0)
        X = _rand(_split(1)[0], (4,), 5, 3)
        batch_K = np.asarray(k.batch_kernel(X, X))   # (4, 5, 5)
        gram_K  = np.asarray(k.Gram_matrix(X, X))    # (4, 4, 5, 5)
        for i in range(4):
            np.testing.assert_allclose(batch_K[i], gram_K[i, i], rtol=1e-12)

    def test_pytree_roundtrip(self):
        k = RBFKernel(sigma=2.5)
        leaves, treedef = jax.tree_util.tree_flatten(k)
        k2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert k2.sigma == k.sigma

    def test_jit(self):
        k = RBFKernel(sigma=1.0)
        X = _rand(KEY, (3,), 4, 2)
        Y = _rand(jr.PRNGKey(1), (3,), 6, 2)
        out = jax.jit(k.batch_kernel)(X, Y)
        assert out.shape == (3, 4, 6)

    def test_sigma_scaling(self):
        """Larger sigma → kernel values closer to 1 everywhere."""
        keys = _split(2)
        X = _rand(keys[0], (5,), 4, 3)
        Y = _rand(keys[1], (5,), 4, 3)
        k_narrow = RBFKernel(sigma=0.01)
        k_wide = RBFKernel(sigma=1000.0)
        mean_narrow = float(jnp.mean(k_narrow.batch_kernel(X, Y)))
        mean_wide = float(jnp.mean(k_wide.batch_kernel(X, Y)))
        assert mean_wide > mean_narrow


# ===========================================================================
# RBF_CEXP_Kernel
# ===========================================================================

class TestRBF_CEXP_Kernel:

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_batch_kernel_shape(self, batch):
        k = RBF_CEXP_Kernel(sigma1=1.0, sigma2=1.0, n_freqs=5)
        keys = _split(2)
        X = _rand_fn(keys[0], batch, 6, 8, 3)
        Y = _rand_fn(keys[1], batch, 4, 8, 3)
        assert k.batch_kernel(X, Y).shape == batch + (6, 4)

    @pytest.mark.parametrize("batch_X", [(), (2,), (2, 3)])
    @pytest.mark.parametrize("batch_Y", [(), (4,)])
    def test_gram_matrix_shape(self, batch_X, batch_Y):
        k = RBF_CEXP_Kernel(sigma1=1.0, sigma2=1.0, n_freqs=5)
        X = _rand_fn(_split(1)[0], batch_X, 6, 8, 3)
        Y = _rand_fn(_split(2)[1], batch_Y, 4, 8, 3)
        assert k.Gram_matrix(X, Y).shape == batch_X + batch_Y + (6, 4)

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_batch_kernel_symmetry(self, batch):
        """batch_kernel(X, X) must be symmetric in its last two axes."""
        k = RBF_CEXP_Kernel(sigma1=1.0, sigma2=1.0, n_freqs=5)
        X = _rand_fn(_split(1)[0], batch, 5, 8, 3)
        G = np.asarray(k.batch_kernel(X, X))
        np.testing.assert_allclose(G, G.swapaxes(-1, -2), atol=1e-12)

    def test_gram_symmetry_single_batch(self):
        """Gram_matrix(X, X) satisfies G[i,j,p,q] == G[j,i,q,p]."""
        k = RBF_CEXP_Kernel(sigma1=1.0, sigma2=1.0, n_freqs=5)
        X = _rand_fn(_split(1)[0], (3,), 5, 8, 3)
        G = np.asarray(k.Gram_matrix(X, X))   # (3, 3, 5, 5)
        G_sym = G.transpose(1, 0, 3, 2)
        np.testing.assert_allclose(G, G_sym, atol=1e-12)

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_positive_semi_definite(self, batch):
        """batch_kernel(X, X) must have non-negative eigenvalues."""
        k = RBF_CEXP_Kernel(sigma1=1.0, sigma2=1.0, n_freqs=5)
        X = _rand_fn(_split(1)[0], batch, 6, 8, 3)
        G = np.asarray(k.batch_kernel(X, X))   # batch + (6, 6)
        G_flat = G.reshape((-1,) + G.shape[-2:]) if G.ndim > 2 else G[None]
        for mat in G_flat:
            eigvals = np.linalg.eigvalsh(mat)
            assert eigvals.min() >= -1e-9, f"PSD violated: min eigval = {eigvals.min()}"

    def test_pytree_roundtrip(self):
        k = RBF_CEXP_Kernel(sigma1=2.0, sigma2=0.5, n_freqs=10)
        leaves, treedef = jax.tree_util.tree_flatten(k)
        k2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert k2.sigma1 == k.sigma1
        assert k2.sigma2 == k.sigma2
        assert k2.n_freqs == k.n_freqs

    def test_jit(self):
        k = RBF_CEXP_Kernel(sigma1=1.0, sigma2=1.0, n_freqs=5)
        X = _rand_fn(KEY, (2,), 4, 6, 2)
        Y = _rand_fn(jr.PRNGKey(1), (2,), 5, 6, 2)
        out = jax.jit(k.batch_kernel)(X, Y)
        assert out.shape == (2, 4, 5)

    def test_matches_rbf_on_transformed_inputs(self):
        """Explicitly applying cexp then RBFKernel must match RBF_CEXP_Kernel."""
        sigma1, sigma2, n_freqs = 1.5, 0.8, 6
        k_cexp = RBF_CEXP_Kernel(sigma1=sigma1, sigma2=sigma2, n_freqs=n_freqs)
        k_rbf = RBFKernel(sigma=sigma2)
        keys = _split(2)
        X = _rand_fn(keys[0], (3,), 5, 8, 2)
        Y = _rand_fn(keys[1], (3,), 4, 8, 2)
        CX = cexp(X, n_freqs=n_freqs, sigma=sigma1).reshape(3, 5, -1)
        CY = cexp(Y, n_freqs=n_freqs, sigma=sigma1).reshape(3, 4, -1)
        ref = np.asarray(k_rbf.batch_kernel(CX, CY))
        got = np.asarray(k_cexp.batch_kernel(X, Y))
        np.testing.assert_allclose(got, ref, rtol=1e-12)


# ===========================================================================
# RBF_SQR_Kernel
# ===========================================================================

class TestRBF_SQR_Kernel:

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_batch_kernel_shape(self, batch):
        k = RBF_SQR_Kernel(sigma1=1.0, sigma2=1.0)
        keys = _split(2)
        X = _rand_fn(keys[0], batch, 6, 8, 3)
        Y = _rand_fn(keys[1], batch, 4, 8, 3)
        assert k.batch_kernel(X, Y).shape == batch + (6, 4)

    @pytest.mark.parametrize("batch_X", [(), (2,), (2, 3)])
    @pytest.mark.parametrize("batch_Y", [(), (4,)])
    def test_gram_matrix_shape(self, batch_X, batch_Y):
        k = RBF_SQR_Kernel(sigma1=1.0, sigma2=1.0)
        X = _rand_fn(_split(1)[0], batch_X, 6, 8, 3)
        Y = _rand_fn(_split(2)[1], batch_Y, 4, 8, 3)
        assert k.Gram_matrix(X, Y).shape == batch_X + batch_Y + (6, 4)

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_batch_kernel_symmetry(self, batch):
        """batch_kernel(X, X) must be symmetric in its last two axes."""
        k = RBF_SQR_Kernel(sigma1=1.0, sigma2=1.0)
        X = _rand_fn(_split(1)[0], batch, 5, 8, 3)
        G = np.asarray(k.batch_kernel(X, X))
        np.testing.assert_allclose(G, G.swapaxes(-1, -2), atol=1e-12)

    def test_gram_symmetry_single_batch(self):
        """Gram_matrix(X, X) satisfies G[i,j,p,q] == G[j,i,q,p]."""
        k = RBF_SQR_Kernel(sigma1=1.0, sigma2=1.0)
        X = _rand_fn(_split(1)[0], (3,), 5, 8, 3)
        G = np.asarray(k.Gram_matrix(X, X))   # (3, 3, 5, 5)
        G_sym = G.transpose(1, 0, 3, 2)
        np.testing.assert_allclose(G, G_sym, atol=1e-12)

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_positive_semi_definite(self, batch):
        """batch_kernel(X, X) must have non-negative eigenvalues."""
        k = RBF_SQR_Kernel(sigma1=1.0, sigma2=1.0)
        X = _rand_fn(_split(1)[0], batch, 6, 8, 3)
        G = np.asarray(k.batch_kernel(X, X))   # batch + (6, 6)
        G_flat = G.reshape((-1,) + G.shape[-2:]) if G.ndim > 2 else G[None]
        for mat in G_flat:
            eigvals = np.linalg.eigvalsh(mat)
            assert eigvals.min() >= -1e-9, f"PSD violated: min eigval = {eigvals.min()}"

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_correctness_is_product_of_rbfs(self, batch):
        """K_SQR(X, Y) == K_RBF1(X_flat, Y_flat) * K_RBF2(X_flat², Y_flat²)."""
        sigma1, sigma2 = 1.0, 2.0
        k = RBF_SQR_Kernel(sigma1=sigma1, sigma2=sigma2)
        rbf1 = RBFKernel(sigma=sigma1)
        rbf2 = RBFKernel(sigma=sigma2)
        keys = _split(2)
        X = _rand_fn(keys[0], batch, 4, 6, 3)
        Y = _rand_fn(keys[1], batch, 5, 6, 3)
        Xf = X.reshape(X.shape[:-2] + (-1,))
        Yf = Y.reshape(Y.shape[:-2] + (-1,))
        ref = np.asarray(rbf1.batch_kernel(Xf, Yf)) * np.asarray(rbf2.batch_kernel(Xf ** 2, Yf ** 2))
        got = np.asarray(k.batch_kernel(X, Y))
        np.testing.assert_allclose(got, ref, rtol=1e-12)

    def test_pytree_roundtrip(self):
        k = RBF_SQR_Kernel(sigma1=1.0, sigma2=3.0)
        leaves, treedef = jax.tree_util.tree_flatten(k)
        k2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert k2.sigma1 == k.sigma1
        assert k2.sigma2 == k.sigma2

    def test_jit(self):
        k = RBF_SQR_Kernel(sigma1=1.0, sigma2=1.0)
        X = _rand_fn(KEY, (2,), 4, 6, 2)
        Y = _rand_fn(jr.PRNGKey(1), (2,), 5, 6, 2)
        out = jax.jit(k.batch_kernel)(X, Y)
        assert out.shape == (2, 4, 5)


# ===========================================================================
# cos_exp_kernel and cexp helpers
# ===========================================================================

class TestCosExpHelpers:

    def test_cos_exp_kernel_shape(self):
        n = 10
        x_y = jnp.zeros((n, n))
        out = cos_exp_kernel(x_y, n_freqs=5, sigma=1.0)
        assert out.shape == (n, n)

    def test_cos_exp_kernel_at_zero(self):
        """At x = y (diff = 0): cos terms all = 1, exp(0) = 1 → value = n_freqs."""
        n_freqs = 7
        x_y = jnp.zeros((5, 5))
        out = np.asarray(cos_exp_kernel(x_y, n_freqs=n_freqs, sigma=1.0))
        np.testing.assert_allclose(out, n_freqs, rtol=1e-12)

    def test_cos_exp_kernel_symmetry(self):
        """cos_exp_kernel(x-y) is an even function: k(x-y) == k(y-x)."""
        obs = jnp.linspace(0, 1, 8)
        x_y = obs[:, None] - obs[None, :]
        K = np.asarray(cos_exp_kernel(x_y, n_freqs=5, sigma=1.0))
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    def test_cexp_shape(self, batch):
        length_t, length_x, dim = 6, 8, 3
        X = _rand_fn(_split(1)[0], batch, length_t, length_x, dim)
        out = cexp(X, n_freqs=5, sigma=1.0)
        assert out.shape == batch + (length_t, length_x, dim)

    def test_cexp_is_linear_in_X(self):
        """cexp is a linear operator in X (it is an integral operator)."""
        alpha = 3.7
        X = _rand_fn(KEY, (3,), 5, 8, 2)
        out_X = np.asarray(cexp(X, n_freqs=5, sigma=1.0))
        out_aX = np.asarray(cexp(alpha * X, n_freqs=5, sigma=1.0))
        np.testing.assert_allclose(out_aX, alpha * out_X, rtol=1e-12)





