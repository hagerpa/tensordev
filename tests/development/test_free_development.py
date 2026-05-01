from jax import config
config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax.numpy as jnp
import jax.random as jr

from tensordev import Jax, FreeDevelopment
from tensordev.development import free_development
from tensordev.core.jax import JaxSequentialCore

from tensordev.util.random_paths import (
    integrated_ou_first_on_path,
    random_trigonometric_polynomial_paths_first_on,
)

CORE = Jax()
SEQ = JaxSequentialCore()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_level_one_path(path_kind: str, key, *, batch: int, steps: int, dim: int):
    if path_kind == "integrated_ou":
        X = integrated_ou_first_on_path(key, batch=batch, steps=steps, dim=dim, trunc=1)
        return np.asarray(X[0])
    if path_kind == "trig":
        X = random_trigonometric_polynomial_paths_first_on(key, batch=batch, steps=steps, dim=dim, trunc=1)
        return np.asarray(X[0])
    raise ValueError(f"Unknown path_kind={path_kind!r}.")


def _iisignature_levels(path: np.ndarray, trunc: int):
    iisignature = pytest.importorskip(
        "iisignature",
        reason="iisignature is required for direct path-signature reference tests.",
    )
    path = np.asarray(path, dtype=np.float64)
    dim = path.shape[-1]
    flat = np.asarray(iisignature.sig(path, trunc), dtype=np.float64).reshape(-1)

    levels = [np.ones((1,), dtype=np.float64)]
    offset = 0
    for k in range(1, trunc + 1):
        width = dim ** k
        levels.append(flat[offset:offset + width])
        offset += width

    if offset != flat.size:
        raise AssertionError(
            f"Unexpected iisignature output length {flat.size}; consumed {offset}."
        )
    return tuple(levels)


def _assert_dense_allclose(actual, expected, *, atol=1e-10, rtol=1e-10, msg=""):
    assert len(actual) == len(expected), (
        f"{msg} number of levels mismatch: got {len(actual)}, expected {len(expected)}"
    )
    for n, (a, e) in enumerate(zip(actual, expected)):
        a = np.asarray(a, dtype=np.float64)
        e = np.asarray(e, dtype=np.float64)
        assert a.shape == e.shape, (
            f"{msg} level {n} shape mismatch: got {a.shape}, expected {e.shape}"
        )
        np.testing.assert_allclose(a, e, atol=atol, rtol=rtol, err_msg=f"{msg} level {n} mismatch")


def _unit_dense(*, batch_shape=(), dim, trunc, dtype=jnp.float64):
    out = [jnp.ones(batch_shape + (1,), dtype=dtype)]
    for k in range(1, trunc + 1):
        out.append(jnp.zeros(batch_shape + (dim ** k,), dtype=dtype))
    return tuple(out)


# ---------------------------------------------------------------------------
# Level-1 sanity: endpoint increment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path_kind", ["integrated_ou", "trig"])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("steps", [8, 25])
@pytest.mark.parametrize("parallel", [False, True])
def test_free_development_level_1_matches_endpoint_increment(
    path_kind, dim, batch, steps, parallel
):
    """Level 1 of the free development equals the path increment X_T - X_0."""
    key = jr.PRNGKey(10_000 + 100 * dim + 10 * batch + steps)
    X = _random_level_one_path(path_kind, key, batch=batch, steps=steps, dim=dim)

    sig = free_development((jnp.asarray(X),), seq_core=SEQ, trunc=3, axis=-2, accumulate=False, parallel=parallel,
                           core=CORE)

    expected_level_1 = X[:, -1, :] - X[:, 0, :]
    np.testing.assert_allclose(np.asarray(sig[1]), expected_level_1, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Comparison with iisignature
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path_kind", ["integrated_ou", "trig"])
@pytest.mark.parametrize(
    ("dim", "trunc", "steps"),
    [(1, 4, 20), (2, 3, 20), (3, 2, 16)],
)
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("parallel", [False, True])
def test_free_development_matches_iisignature(
    path_kind, dim, trunc, steps, batch, parallel
):
    """free_development on a level-1 path matches iisignature."""
    key = jr.PRNGKey(20_000 + 1000 * dim + 100 * trunc + 10 * batch + steps)
    X = _random_level_one_path(path_kind, key, batch=batch, steps=steps, dim=dim)

    got = free_development((jnp.asarray(X),), seq_core=SEQ, trunc=trunc, axis=-2, accumulate=False, parallel=parallel,
                           core=CORE)
    got = tuple(np.asarray(level) for level in got)

    for b in range(batch):
        ref = _iisignature_levels(X[b], trunc=trunc)
        got_b = tuple(level[b] for level in got)
        _assert_dense_allclose(
            got_b,
            ref,
            atol=1e-10,
            rtol=1e-10,
            msg=f"path_kind={path_kind}, batch_index={b}, parallel={parallel}",
        )


# ---------------------------------------------------------------------------
# Prefix signatures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path_kind", ["integrated_ou", "trig"])
@pytest.mark.parametrize(("dim", "trunc", "steps"), [(1, 4, 12), (2, 3, 12)])
def test_free_development_prefixes_match_iisignature(path_kind, dim, trunc, steps):
    """Prefix accumulation matches iisignature on every prefix."""
    key = jr.PRNGKey(30_000 + 1000 * dim + 100 * trunc + steps)
    X = _random_level_one_path(path_kind, key, batch=1, steps=steps, dim=dim)

    got = free_development((jnp.asarray(X),), seq_core=SEQ, trunc=trunc, axis=-2, block_size=1, accumulate=True,
                           output_starting_point=True, core=CORE)
    got = tuple(np.asarray(level[0]) for level in got)

    for j in range(steps + 1):
        if j == 0:
            ref = tuple(
                np.zeros((dim ** k,), dtype=np.float64) if k > 0 else np.ones((1,), dtype=np.float64)
                for k in range(trunc + 1)
            )
        else:
            ref = _iisignature_levels(X[0, :j + 1, :], trunc=trunc)

        got_j = tuple(level[j] for level in got)
        _assert_dense_allclose(
            got_j,
            ref,
            atol=1e-10,
            rtol=1e-10,
            msg=f"path_kind={path_kind}, prefix_index={j}",
        )


# ---------------------------------------------------------------------------
# Translation invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("trunc", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_free_development_translation_invariant(dim, trunc, batch_shape):
    """Free development depends only on increments, not on absolute position."""
    steps = 17
    key = jr.PRNGKey(40_000 + 100 * dim + 10 * trunc + len(batch_shape))
    k_inc, k_start, k_shift = jr.split(key, 3)

    increments = 0.15 * jr.normal(k_inc, batch_shape + (steps, dim), dtype=jnp.float64)
    start = 0.30 * jr.normal(k_start, batch_shape + (dim,), dtype=jnp.float64)
    shift = 0.50 * jr.normal(k_shift, batch_shape + (dim,), dtype=jnp.float64)

    X = jnp.concatenate(
        [start[..., None, :], start[..., None, :] + jnp.cumsum(increments, axis=-2)],
        axis=-2,
    )
    Y = X + shift[..., None, :]

    sig_X = free_development((X,), seq_core=SEQ, trunc=trunc, axis=-2, accumulate=False, core=CORE)
    sig_Y = free_development((Y,), seq_core=SEQ, trunc=trunc, axis=-2, accumulate=False, core=CORE)

    _assert_dense_allclose(
        tuple(np.asarray(a) for a in sig_X),
        tuple(np.asarray(a) for a in sig_Y),
    )


# ---------------------------------------------------------------------------
# Constant path → tensor unit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("trunc", [1, 3, 4])
@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_free_development_constant_path_is_tensor_unit(dim, trunc, batch_shape):
    """A constant path has trivial signature: level 0 = 1, higher levels = 0."""
    steps = 11
    key = jr.PRNGKey(50_000 + 100 * dim + 10 * trunc + len(batch_shape))

    x0 = 0.70 * jr.normal(key, batch_shape + (dim,), dtype=jnp.float64)
    X = jnp.repeat(x0[..., None, :], repeats=steps + 1, axis=-2)

    sig = free_development((X,), seq_core=SEQ, trunc=trunc, axis=-2, accumulate=False, core=CORE)
    ref = _unit_dense(batch_shape=batch_shape, dim=dim, trunc=trunc)

    _assert_dense_allclose(
        tuple(np.asarray(a) for a in sig),
        tuple(np.asarray(a) for a in ref),
    )


# ---------------------------------------------------------------------------
# Single-step path matches tensor_exponential
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("trunc", [1, 3, 4])
@pytest.mark.parametrize("batch_shape", [(), (4,)])
def test_free_development_single_step_matches_exponential(dim, trunc, batch_shape):
    """For a 1-step path x0 -> x0 + v, the result is exp(v)."""
    key = jr.PRNGKey(60_000 + 100 * dim + 10 * trunc + len(batch_shape))
    k_start, k_dx = jr.split(key, 2)

    x0 = 0.50 * jr.normal(k_start, batch_shape + (dim,), dtype=jnp.float64)
    dx = 0.15 * jr.normal(k_dx, batch_shape + (dim,), dtype=jnp.float64)

    X = jnp.stack([x0, x0 + dx], axis=-2)

    sig = free_development((X,), seq_core=SEQ, trunc=trunc, axis=-2, accumulate=False, core=CORE)
    ref = CORE.tensor_exponential((dx,), trunc=trunc, output_zero_level=True)

    _assert_dense_allclose(
        tuple(np.asarray(a) for a in sig),
        tuple(np.asarray(a) for a in ref),
        atol=1e-12,
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# parallel consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path_kind", ["integrated_ou", "trig"])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("trunc", [2, 3])
@pytest.mark.parametrize("steps", [8, 16])
def test_free_development_parallel_consistent(path_kind, dim, trunc, steps):
    """parallel=True and False must agree on the final signature."""
    key = jr.PRNGKey(70_000 + 100 * dim + 10 * trunc + steps)
    X = _random_level_one_path(path_kind, key, batch=2, steps=steps, dim=dim)
    Xj = jnp.asarray(X)

    sig_stream = free_development((Xj,), seq_core=SEQ, trunc=trunc, axis=-2, accumulate=False, parallel=False,
                                  core=CORE)
    sig_parallel = free_development((Xj,), seq_core=SEQ, trunc=trunc, axis=-2, accumulate=False, parallel=True,
                                    core=CORE)

    _assert_dense_allclose(
        tuple(np.asarray(a) for a in sig_stream),
        tuple(np.asarray(a) for a in sig_parallel),
        atol=1e-10,
        rtol=1e-10,
        msg=f"parallel mismatch for path_kind={path_kind}",
    )


# ---------------------------------------------------------------------------
# Blocking consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path_kind", ["integrated_ou", "trig"])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("trunc", [2, 3])
@pytest.mark.parametrize(("steps", "block_size"), [(16, 4), (16, 8), (24, 6)])
def test_free_development_blocking_consistent(path_kind, dim, trunc, steps, block_size):
    """Different block sizes must yield the same final accumulated signature."""
    key = jr.PRNGKey(90_000 + 100 * dim + 10 * trunc + steps + block_size)
    X = _random_level_one_path(path_kind, key, batch=2, steps=steps, dim=dim)
    Xj = jnp.asarray(X)

    sig_no_block = free_development((Xj,), seq_core=SEQ, trunc=trunc, axis=-2, accumulate=False, core=CORE)
    sig_blocked = free_development((Xj,), seq_core=SEQ, trunc=trunc, axis=-2, block_size=block_size, accumulate=True,
                                   core=CORE)
    # Take the last block result
    sig_blocked_last = tuple(a[..., -1, :] for a in sig_blocked)

    _assert_dense_allclose(
        tuple(np.asarray(a) for a in sig_no_block),
        tuple(np.asarray(a) for a in sig_blocked_last),
        atol=1e-10,
        rtol=1e-10,
        msg=f"block_size={block_size}",
    )


# ---------------------------------------------------------------------------
# FreeDevelopment class
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path_kind", ["integrated_ou", "trig"])
@pytest.mark.parametrize(("dim", "trunc", "steps"), [(2, 3, 16), (3, 2, 12)])
@pytest.mark.parametrize("batch", [1, 2])
def test_free_development_class_matches_function(path_kind, dim, trunc, steps, batch):
    """FreeDevelopment(trunc=...) class produces the same result as free_development()."""
    key = jr.PRNGKey(100_000 + 1000 * dim + 100 * trunc + 10 * batch + steps)
    X = _random_level_one_path(path_kind, key, batch=batch, steps=steps, dim=dim)
    Xj = (jnp.asarray(X),)

    fn_result = free_development(Xj, seq_core=SEQ, trunc=trunc, axis=-2, accumulate=False, core=CORE)
    cls_result = FreeDevelopment(trunc=trunc, core=CORE, seq_core=SEQ)(Xj, axis=-2, accumulate=False)

    _assert_dense_allclose(
        tuple(np.asarray(a) for a in fn_result),
        tuple(np.asarray(a) for a in cls_result),
        atol=1e-12,
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# starting_point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("trunc", [2, 3])
def test_free_development_starting_point_equivalent_to_chen(dim, trunc):
    """
    free_development with starting_point=g equals
    tensor_product(g, free_development without starting_point).
    """
    steps = 12
    key = jr.PRNGKey(110_000 + 100 * dim + 10 * trunc)
    k_path, k_g = jr.split(key, 2)

    X = _random_level_one_path("trig", k_path, batch=2, steps=steps, dim=dim)
    Xj = (jnp.asarray(X),)

    dx0 = 0.15 * jr.normal(k_g, (2, dim), dtype=jnp.float64)
    g = CORE.tensor_exponential((dx0,), trunc=trunc, output_zero_level=True)

    sig_plain = free_development(Xj, seq_core=SEQ, trunc=trunc, axis=-2, accumulate=False, core=CORE)
    sig_seeded = free_development(Xj, seq_core=SEQ, trunc=trunc, axis=-2, accumulate=False, starting_point=g, core=CORE)

    expected = CORE.tensor_product(g, sig_plain, trunc=trunc)

    _assert_dense_allclose(
        tuple(np.asarray(a) for a in sig_seeded),
        tuple(np.asarray(a) for a in expected),
        atol=1e-10,
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# output_starting_point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("trunc", [2, 3])
def test_free_development_output_starting_point(dim, trunc):
    """With output_starting_point=True, index 0 equals the seed (unit)."""
    steps = 10
    key = jr.PRNGKey(120_000 + 100 * dim + 10 * trunc)
    X = _random_level_one_path("trig", key, batch=2, steps=steps, dim=dim)
    Xj = (jnp.asarray(X),)

    got = free_development(Xj, seq_core=SEQ, trunc=trunc, axis=-2, block_size=1, accumulate=True,
                           output_starting_point=True, core=CORE)
    # index 0 along the block axis (-2) should be the tensor unit
    got_0 = tuple(a[..., 0, :] for a in got)
    ref = _unit_dense(batch_shape=(2,), dim=dim, trunc=trunc)

    _assert_dense_allclose(
        tuple(np.asarray(a) for a in got_0),
        tuple(np.asarray(a) for a in ref),
        atol=1e-12,
        rtol=1e-12,
    )