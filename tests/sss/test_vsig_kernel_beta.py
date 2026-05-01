from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tensordev.kernel.fssk import FSSKSigKernel, fssk_sigkernel
from tensordev.sss import FSSK


def _make_scalar_identity_kernel(dim: int) -> FSSK:
    return FSSK.from_matrix(
        Lambda=jnp.asarray([[0.0]]),
        A=jnp.eye(dim)[None, :, :],
        b=jnp.asarray([[1.0]]),
    )


def test_fssk_kernel_returns_one_for_constant_paths():
    ker = _make_scalar_identity_kernel(dim=2)
    X = jnp.asarray([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    Y = jnp.asarray([[[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]]])

    out = fssk_sigkernel(X, Y, kernel=ker, dt_x=0.5, dt_y=jnp.asarray([0.25, 0.25]))
    assert out.shape == (1,)
    assert jnp.allclose(out, jnp.ones_like(out))



def test_fssk_kernel_pairwise_diagonal_matches_batchwise():
    ker = _make_scalar_identity_kernel(dim=2)
    X = jnp.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.5, 0.5]],
            [[0.0, 0.0], [0.0, 1.0], [0.5, 1.5]],
        ]
    )
    Y = jnp.asarray(
        [
            [[0.0, 0.0], [0.5, 0.5], [1.0, 0.5]],
            [[0.0, 0.0], [0.5, -0.5], [1.0, -1.0]],
        ]
    )
    dt = jnp.asarray([0.3, 0.7])

    batchwise = fssk_sigkernel(X, Y, kernel=ker, dt_x=dt, dt_y=dt, pairwise=False)
    pairwise = fssk_sigkernel(X, Y, kernel=ker, dt_x=dt, dt_y=dt, pairwise=True)

    assert batchwise.shape == (2,)
    assert pairwise.shape == (2, 2)
    assert jnp.allclose(batchwise, jnp.diag(pairwise), atol=1e-10, rtol=1e-10)



def test_fssk_sig_kernel_wrapper_compute_gram_is_symmetric_on_identical_inputs():
    ker = _make_scalar_identity_kernel(dim=2)
    X = jnp.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.5, 0.5]],
            [[0.0, 0.0], [0.0, 1.0], [0.5, 1.5]],
            [[0.0, 0.0], [1.0, 1.0], [1.5, 1.0]],
        ]
    )
    dt = jnp.asarray([0.4, 0.6])

    wrapper = FSSKSigKernel(kernel=ker, dt_x=dt, dt_y=dt)
    gram = wrapper.compute_Gram(X, sym=True, max_batch=2)

    assert gram.shape == (3, 3)
    assert jnp.allclose(gram, gram.T, atol=1e-10, rtol=1e-10)
    assert jnp.all(gram.diagonal() >= 1.0)



def test_fssk_kernel_grid_and_auxiliary_shapes():
    ker = _make_scalar_identity_kernel(dim=2)
    X = jnp.asarray([[[0.0, 0.0], [1.0, 0.0], [1.5, 0.5]]])
    Y = jnp.asarray([[[0.0, 0.0], [0.5, 0.5], [1.0, 0.5], [1.0, 1.0]]])

    eta, K, Psi, Phi = fssk_sigkernel(
        X,
        Y,
        kernel=ker,
        dt_x=jnp.asarray([0.2, 0.3]),
        dt_y=jnp.asarray([0.1, 0.2, 0.4]),
        evaluate="grid",
        return_fg=True,
    )

    assert eta.shape == (1, 3, 4)
    assert K.shape == (1, 3, 4, 1, 1)
    assert Psi.shape == (1, 3, 4, 1, 1)
    assert Phi.shape == (1, 3, 4, 1, 1)
    assert jnp.allclose(eta[:, 0, :], 1.0)
    assert jnp.allclose(eta[:, :, 0], 1.0)
