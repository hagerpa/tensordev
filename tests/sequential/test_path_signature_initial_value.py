from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax.numpy as jnp
import jax.random as jr

from tensordev import Jax

CORE = Jax()


def _assert_tuple_allclose(a, b, *, rtol=1e-12, atol=1e-12):
    assert len(a) == len(b)
    for xa, xb in zip(a, b):
        np.testing.assert_allclose(
            np.asarray(xa),
            np.asarray(xb),
            rtol=rtol,
            atol=atol,
        )


def _path_from_start_and_increments(start, increments):
    """
    start      : batch_shape + (dim,)
    increments : batch_shape + (steps, dim)

    returns    : batch_shape + (steps + 1, dim)
    """
    return jnp.concatenate(
        [
            start[..., None, :],
            start[..., None, :] + jnp.cumsum(increments, axis=-2),
        ],
        axis=-2,
    )


def _unit_dense(*, batch_shape=(), dim, trunc, dtype=jnp.float64):
    out = [jnp.ones(batch_shape + (1,), dtype=dtype)]
    for k in range(1, trunc + 1):
        out.append(jnp.zeros(batch_shape + (dim ** k,), dtype=dtype))
    return tuple(out)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("trunc", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_tensor_path_signature_is_translation_invariant(dim, trunc, batch_shape):
    """
    Ordinary path signatures depend only on increments, not on absolute position.
    """
    steps = 17
    key = jr.PRNGKey(1000 + 100 * dim + 10 * trunc + len(batch_shape))
    k_inc, k_start, k_shift = jr.split(key, 3)

    increments = 0.15 * jr.normal(
        k_inc,
        batch_shape + (steps, dim),
        dtype=jnp.float64,
    )
    start = 0.30 * jr.normal(
        k_start,
        batch_shape + (dim,),
        dtype=jnp.float64,
    )
    shift = 0.50 * jr.normal(
        k_shift,
        batch_shape + (dim,),
        dtype=jnp.float64,
    )

    X = _path_from_start_and_increments(start, increments)
    Y = X + shift[..., None, :]

    sig_X = CORE.tensor_path_signature(X, trunc=trunc, accumulate=False)
    sig_Y = CORE.tensor_path_signature(Y, trunc=trunc, accumulate=False)

    _assert_tuple_allclose(sig_X, sig_Y)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("trunc", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (4,)])
def test_tensor_path_signature_same_increments_different_start(dim, trunc, batch_shape):
    """
    Two paths built from the same increments but different initial values
    must have the same signature.
    """
    steps = 19
    key = jr.PRNGKey(2000 + 100 * dim + 10 * trunc + len(batch_shape))
    k_inc, k_start0, k_start1 = jr.split(key, 3)

    increments = 0.12 * jr.normal(
        k_inc,
        batch_shape + (steps, dim),
        dtype=jnp.float64,
    )
    start0 = 0.40 * jr.normal(
        k_start0,
        batch_shape + (dim,),
        dtype=jnp.float64,
    )
    start1 = 0.40 * jr.normal(
        k_start1,
        batch_shape + (dim,),
        dtype=jnp.float64,
    )

    X0 = _path_from_start_and_increments(start0, increments)
    X1 = _path_from_start_and_increments(start1, increments)

    sig0 = CORE.tensor_path_signature(X0, trunc=trunc, accumulate=False)
    sig1 = CORE.tensor_path_signature(X1, trunc=trunc, accumulate=False)

    _assert_tuple_allclose(sig0, sig1)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("trunc", [1, 3, 4])
@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_tensor_path_signature_constant_path_is_tensor_unit(dim, trunc, batch_shape):
    """
    A constant path should have trivial signature:
      level 0 = 1, higher levels = 0.
    """
    steps = 11
    key = jr.PRNGKey(3000 + 100 * dim + 10 * trunc + len(batch_shape))

    x0 = 0.70 * jr.normal(
        key,
        batch_shape + (dim,),
        dtype=jnp.float64,
    )
    X = jnp.repeat(x0[..., None, :], repeats=steps + 1, axis=-2)

    sig = CORE.tensor_path_signature(X, trunc=trunc, accumulate=False)
    ref = _unit_dense(
        batch_shape=batch_shape,
        dim=dim,
        trunc=trunc,
        dtype=jnp.float64,
    )

    _assert_tuple_allclose(sig, ref)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("trunc", [1, 3, 4])
@pytest.mark.parametrize("batch_shape", [(), (4,)])
def test_tensor_path_signature_single_increment_matches_tensor_exponential(dim, trunc, batch_shape):
    """
    For a 1-step path x0 -> x0 + v, the signature is exp(v), independent of x0.
    """
    key = jr.PRNGKey(4000 + 100 * dim + 10 * trunc + len(batch_shape))
    k_start, k_dx = jr.split(key, 2)

    x0 = 0.50 * jr.normal(
        k_start,
        batch_shape + (dim,),
        dtype=jnp.float64,
    )
    dx = 0.15 * jr.normal(
        k_dx,
        batch_shape + (dim,),
        dtype=jnp.float64,
    )

    X = jnp.stack([x0, x0 + dx], axis=-2)

    sig = CORE.tensor_path_signature(X, trunc=trunc, accumulate=False)
    ref = CORE.tensor_exponential(
        (dx,),
        trunc=trunc,
        output_zero_level=True,
    )

    _assert_tuple_allclose(sig, ref)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("trunc", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_tensor_path_signature_unchanged_by_prepended_flat_segment(dim, trunc, batch_shape):
    """
    Prepending a constant segment contributes zero increments and must not
    change the signature.
    """
    steps = 13
    key = jr.PRNGKey(5000 + 100 * dim + 10 * trunc + len(batch_shape))
    k_inc, k_start = jr.split(key, 2)

    increments = 0.10 * jr.normal(
        k_inc,
        batch_shape + (steps, dim),
        dtype=jnp.float64,
    )
    start = 0.25 * jr.normal(
        k_start,
        batch_shape + (dim,),
        dtype=jnp.float64,
    )

    X = _path_from_start_and_increments(start, increments)

    n_flat = 4
    flat_prefix = jnp.repeat(X[..., :1, :], repeats=n_flat, axis=-2)
    X_with_flat_prefix = jnp.concatenate([flat_prefix, X], axis=-2)

    sig_X = CORE.tensor_path_signature(X, trunc=trunc, accumulate=False)
    sig_Y = CORE.tensor_path_signature(X_with_flat_prefix, trunc=trunc, accumulate=False)

    _assert_tuple_allclose(sig_X, sig_Y)