from jax import config
config.update("jax_enable_x64", True)

from functools import partial

import numpy as np
import pytest
import jax.numpy as jnp
import jax.random as jr

from tensordev import Jax
from tensordev.core.jax import JaxSequentialCore

CORE = Jax()
SEQ = JaxSequentialCore()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(*, batch_shape=(), dim, trunc, dtype=jnp.float64):
    out = [jnp.ones(batch_shape + (1,), dtype=dtype)]
    for k in range(1, trunc + 1):
        out.append(jnp.zeros(batch_shape + (dim ** k,), dtype=dtype))
    return tuple(out)


def _random_dense(key, *, dim, trunc, batch_shape=(), scale=0.10):
    keys = jr.split(key, trunc + 1)
    return tuple(
        scale * jr.normal(keys[k], batch_shape + (dim ** k,), dtype=jnp.float64)
        for k in range(trunc + 1)
    )


def _random_increments(key, *, dim, trunc, steps, batch_shape=(), scale=0.08):
    """DenseElemFirstOn increments — suitable for tensor_fmexp-based ops."""
    keys = jr.split(key, trunc)
    return tuple(
        scale * jr.normal(keys[k - 1], batch_shape + (steps, dim ** k), dtype=jnp.float64)
        for k in range(1, trunc + 1)
    )


def _random_full_seq(key, *, dim, trunc, steps, batch_shape=(), scale=0.08):
    """Sequence of full DenseElems (level 0 to trunc) — suitable for tensor_product ops.

    Each element is a small tensor exponential so that the sequence is close to
    the unit and numerical precision stays high.
    """
    keys = jr.split(key, steps)
    seq = []
    for t in range(steps):
        inc = tuple(
            scale * jr.normal(keys[t], batch_shape + (dim ** k,), dtype=jnp.float64)
            for k in range(1, trunc + 1)
        )
        seq.append(CORE.tensor_exponential(inc, trunc=trunc, output_zero_level=True))
    return tuple(
        jnp.stack([seq[t][k] for t in range(steps)], axis=-2)
        for k in range(trunc + 1)
    )


def _assert_allclose(actual, expected, *, atol=1e-10, rtol=1e-10, msg=""):
    assert len(actual) == len(expected), f"{msg} level count mismatch"
    for n, (a, e) in enumerate(zip(actual, expected)):
        np.testing.assert_allclose(
            np.asarray(a, dtype=np.float64),
            np.asarray(e, dtype=np.float64),
            atol=atol, rtol=rtol,
            err_msg=f"{msg} level {n}",
        )


def _make_product_ops(trunc):
    op = partial(CORE.tensor_product, trunc=trunc)
    return op, op


def _make_fmexp_ops(trunc):
    reduce_op = partial(CORE.tensor_fmexp, trunc=trunc, output_zero_level=True)
    acc_op = partial(CORE.tensor_product, trunc=trunc)
    return reduce_op, acc_op


# ---------------------------------------------------------------------------
# tensor_reduce: single block equals full product
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("trunc", [2, 3])
@pytest.mark.parametrize("steps", [4, 8])
@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_tensor_reduce_matches_manual_product(dim, trunc, steps, batch_shape):
    """tensor_reduce with product op on full algebra elements equals iterative tensor_product."""
    key = jr.PRNGKey(1000 + 100 * dim + 10 * trunc + steps)
    X = _random_full_seq(key, dim=dim, trunc=trunc, steps=steps, batch_shape=batch_shape)
    neutral = _unit(batch_shape=batch_shape, dim=dim, trunc=trunc)

    reduce_op, _ = _make_product_ops(trunc)
    got = SEQ.tensor_reduce(X, reduce_op=reduce_op, neutral=neutral, axis=-2)

    # Reference: manual left-fold
    acc = neutral
    for t in range(steps):
        step = tuple(level[..., t, :] for level in X)
        acc = reduce_op(acc, step)

    _assert_allclose(got, acc, msg="tensor_reduce")


# ---------------------------------------------------------------------------
# tensor_reduce: seed override
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("trunc", [2, 3])
def test_tensor_reduce_with_seed(dim, trunc):
    """tensor_reduce with explicit seed equals seed ⊗ product."""
    steps = 6
    key = jr.PRNGKey(2000 + 100 * dim + 10 * trunc)
    k_path, k_seed = jr.split(key, 2)
    X = _random_full_seq(k_path, dim=dim, trunc=trunc, steps=steps)

    dx0 = 0.1 * jr.normal(k_seed, (dim,), dtype=jnp.float64)
    g = CORE.tensor_exponential((dx0,), trunc=trunc, output_zero_level=True)

    neutral = _unit(dim=dim, trunc=trunc)
    reduce_op, _ = _make_product_ops(trunc)

    got = SEQ.tensor_reduce(X, reduce_op=reduce_op, neutral=neutral, seed=g, axis=-2)
    plain = SEQ.tensor_reduce(X, reduce_op=reduce_op, neutral=neutral, axis=-2)
    expected = reduce_op(g, plain)

    _assert_allclose(got, expected, msg="tensor_reduce with seed")


# ---------------------------------------------------------------------------
# tensor_accumulate: prefix[t] matches reduce over first t+1 steps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("trunc", [2, 3])
@pytest.mark.parametrize("steps", [5, 8])
def test_tensor_accumulate_prefixes_match_reduce(dim, trunc, steps):
    """Each prefix from tensor_accumulate matches tensor_reduce on that prefix."""
    key = jr.PRNGKey(3000 + 100 * dim + 10 * trunc + steps)
    X = _random_full_seq(key, dim=dim, trunc=trunc, steps=steps)
    neutral = _unit(dim=dim, trunc=trunc)
    reduce_op, _ = _make_product_ops(trunc)

    prefixes = SEQ.tensor_accumulate(X, reduce_op=reduce_op, neutral=neutral, axis=-2)

    for t in range(steps):
        X_prefix = tuple(level[..., :t + 1, :] for level in X)
        ref = SEQ.tensor_reduce(X_prefix, reduce_op=reduce_op, neutral=neutral, axis=-2)
        got_t = tuple(a[..., t, :] for a in prefixes)
        _assert_allclose(got_t, ref, msg=f"prefix t={t}")


# ---------------------------------------------------------------------------
# tensor_accumulate: output_starting_point prepends neutral
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("trunc", [2, 3])
def test_tensor_accumulate_output_starting_point(dim, trunc):
    """With output_starting_point=True index 0 equals the seed (neutral)."""
    steps = 6
    key = jr.PRNGKey(4000 + 100 * dim + 10 * trunc)
    X = _random_full_seq(key, dim=dim, trunc=trunc, steps=steps)
    neutral = _unit(dim=dim, trunc=trunc)
    reduce_op, _ = _make_product_ops(trunc)

    got = SEQ.tensor_accumulate(
        X, reduce_op=reduce_op, neutral=neutral, axis=-2, output_starting_point=True
    )

    got_0 = tuple(a[..., 0, :] for a in got)
    _assert_allclose(got_0, neutral, atol=1e-12, rtol=1e-12, msg="index 0 == neutral")


# ---------------------------------------------------------------------------
# tensor_abra: single block, no accumulate equals tensor_reduce
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("trunc", [2, 3])
@pytest.mark.parametrize("steps", [6, 12])
@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_tensor_abra_single_block_equals_reduce(dim, trunc, steps, batch_shape):
    """tensor_abra with one block and accumulate=False equals tensor_reduce."""
    key = jr.PRNGKey(5000 + 100 * dim + 10 * trunc + steps)
    X = _random_full_seq(key, dim=dim, trunc=trunc, steps=steps, batch_shape=batch_shape)
    neutral = _unit(batch_shape=batch_shape, dim=dim, trunc=trunc)
    reduce_op, acc_op = _make_product_ops(trunc)

    abra = SEQ.tensor_abra(
        X, reduce_op=reduce_op, acc_op=acc_op, neutral=neutral, axis=-2, accumulate=False,
    )
    ref = SEQ.tensor_reduce(X, reduce_op=reduce_op, neutral=neutral, axis=-2)

    _assert_allclose(abra, ref, msg="abra single block == reduce")


# ---------------------------------------------------------------------------
# tensor_abra: blocking consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("trunc", [2, 3])
@pytest.mark.parametrize(("steps", "block_size"), [(8, 2), (12, 4), (12, 6)])
def test_tensor_abra_blocking_consistent_with_manual(dim, trunc, steps, block_size):
    """Each block result from tensor_abra(accumulate=False) matches a manual reduce on that block."""
    key = jr.PRNGKey(6000 + 100 * dim + 10 * trunc + steps + block_size)
    X = _random_full_seq(key, dim=dim, trunc=trunc, steps=steps)
    neutral = _unit(dim=dim, trunc=trunc)
    reduce_op, acc_op = _make_product_ops(trunc)

    blocks = SEQ.tensor_abra(
        X, reduce_op=reduce_op, acc_op=acc_op, neutral=neutral,
        axis=-2, block_size=block_size, accumulate=False,
    )

    n_blocks = steps // block_size
    for b in range(n_blocks):
        X_block = tuple(level[..., b * block_size:(b + 1) * block_size, :] for level in X)
        ref = SEQ.tensor_reduce(X_block, reduce_op=reduce_op, neutral=neutral, axis=-2)
        got_b = tuple(a[..., b, :] for a in blocks)
        _assert_allclose(got_b, ref, msg=f"block {b}")


# ---------------------------------------------------------------------------
# tensor_abra: accumulate=True prefix products
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("trunc", [2, 3])
@pytest.mark.parametrize(("steps", "block_size"), [(8, 2), (12, 3)])
def test_tensor_abra_accumulate_prefix_products(dim, trunc, steps, block_size):
    """tensor_abra(accumulate=True) result at index b equals product of blocks 0..b."""
    key = jr.PRNGKey(7000 + 100 * dim + 10 * trunc + steps + block_size)
    X = _random_full_seq(key, dim=dim, trunc=trunc, steps=steps)
    neutral = _unit(dim=dim, trunc=trunc)
    reduce_op, acc_op = _make_product_ops(trunc)

    # Per-block results
    block_results = SEQ.tensor_abra(
        X, reduce_op=reduce_op, acc_op=acc_op, neutral=neutral,
        axis=-2, block_size=block_size, accumulate=False,
    )
    # Prefix accumulated
    prefix_results = SEQ.tensor_abra(
        X, reduce_op=reduce_op, acc_op=acc_op, neutral=neutral,
        axis=-2, block_size=block_size, accumulate=True,
    )

    n_blocks = steps // block_size
    acc = neutral
    for b in range(n_blocks):
        blk_b = tuple(a[..., b, :] for a in block_results)
        acc = acc_op(acc, blk_b)
        got_b = tuple(a[..., b, :] for a in prefix_results)
        _assert_allclose(got_b, acc, msg=f"prefix product at block {b}")


# ---------------------------------------------------------------------------
# tensor_abra: first_apply_all consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("trunc", [2, 3])
@pytest.mark.parametrize("steps", [8, 16])
def test_tensor_abra_first_apply_all_consistent(dim, trunc, steps):
    """first_apply_all=True and False agree when using fmexp reduce with product acc."""
    key = jr.PRNGKey(8000 + 100 * dim + 10 * trunc + steps)
    X = _random_increments(key, dim=dim, trunc=trunc, steps=steps)
    neutral = _unit(dim=dim, trunc=trunc)

    reduce_op, acc_op = _make_fmexp_ops(trunc)

    got_stream = SEQ.tensor_abra(
        X, reduce_op=reduce_op, acc_op=acc_op, neutral=neutral,
        axis=-2, accumulate=False, first_apply_all=False,
    )
    got_parallel = SEQ.tensor_abra(
        X, reduce_op=reduce_op, acc_op=acc_op, neutral=neutral,
        axis=-2, accumulate=False, first_apply_all=True,
    )

    _assert_allclose(got_stream, got_parallel, msg="first_apply_all consistency")


# ---------------------------------------------------------------------------
# tensor_abra: output_starting_point prepends neutral/seed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("trunc", [2, 3])
def test_tensor_abra_output_starting_point(dim, trunc):
    """output_starting_point=True prepends the seed as index 0 on the block axis."""
    steps = 8
    block_size = 4
    key = jr.PRNGKey(9000 + 100 * dim + 10 * trunc)
    X = _random_full_seq(key, dim=dim, trunc=trunc, steps=steps)
    neutral = _unit(dim=dim, trunc=trunc)
    reduce_op, acc_op = _make_product_ops(trunc)

    got = SEQ.tensor_abra(
        X, reduce_op=reduce_op, acc_op=acc_op, neutral=neutral,
        axis=-2, block_size=block_size, accumulate=True, output_starting_point=True,
    )

    got_0 = tuple(a[..., 0, :] for a in got)
    _assert_allclose(got_0, neutral, atol=1e-12, rtol=1e-12, msg="index 0 == neutral")
