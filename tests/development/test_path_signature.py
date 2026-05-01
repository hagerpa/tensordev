from jax import config
config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax.numpy as jnp
import jax.random as jr

from tensordev import Jax, Signature
from tensordev.development import path_signature
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

def _random_level_one_path(path_kind, key, *, batch, steps, dim):
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
def test_path_signature_level_1_matches_endpoint_increment(
    path_kind, dim, batch, steps, parallel
):
    """Level 1 of the signature equals X_T - X_0."""
    key = jr.PRNGKey(10_000 + 100 * dim + 10 * batch + steps)
    X = _random_level_one_path(path_kind, key, batch=batch, steps=steps, dim=dim)

    sig = path_signature(jnp.asarray(X), accumulate=False, trunc=3, axis=-2, parallel=parallel, core=CORE, seq_core=SEQ)

    np.testing.assert_allclose(
        np.asarray(sig[1]),
        X[:, -1, :] - X[:, 0, :],
        atol=1e-12,
        rtol=1e-12,
    )


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
def test_path_signature_matches_iisignature(
    path_kind, dim, trunc, steps, batch, parallel
):
    """path_signature matches iisignature on random paths."""
    key = jr.PRNGKey(20_000 + 1000 * dim + 100 * trunc + 10 * batch + steps)
    X = _random_level_one_path(path_kind, key, batch=batch, steps=steps, dim=dim)

    got = path_signature(jnp.asarray(X), accumulate=False, trunc=trunc, axis=-2, parallel=parallel, core=CORE,
                         seq_core=SEQ)
    got = tuple(np.asarray(level) for level in got)

    for b in range(batch):
        ref = _iisignature_levels(X[b], trunc=trunc)
        _assert_dense_allclose(
            tuple(level[b] for level in got),
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
def test_path_signature_prefixes_match_iisignature(path_kind, dim, trunc, steps):
    """Prefix accumulation matches iisignature at every timestep."""
    key = jr.PRNGKey(30_000 + 1000 * dim + 100 * trunc + steps)
    X = _random_level_one_path(path_kind, key, batch=1, steps=steps, dim=dim)

    got = path_signature(jnp.asarray(X), accumulate=True, trunc=trunc, axis=-2, block_size=1,
                         output_starting_point=True, core=CORE, seq_core=SEQ)
    got = tuple(np.asarray(level[0]) for level in got)

    for j in range(steps + 1):
        if j == 0:
            ref = tuple(
                np.zeros((dim ** k,), dtype=np.float64) if k > 0 else np.ones((1,), dtype=np.float64)
                for k in range(trunc + 1)
            )
        else:
            ref = _iisignature_levels(X[0, :j + 1, :], trunc=trunc)

        _assert_dense_allclose(
            tuple(level[j] for level in got),
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
def test_path_signature_translation_invariant(dim, trunc, batch_shape):
    """Signature depends only on increments, not on absolute position."""
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

    sig_X = path_signature(X, accumulate=False, trunc=trunc, axis=-2, core=CORE, seq_core=SEQ)
    sig_Y = path_signature(Y, accumulate=False, trunc=trunc, axis=-2, core=CORE, seq_core=SEQ)

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
def test_path_signature_constant_path_is_tensor_unit(dim, trunc, batch_shape):
    """A constant path has trivial signature: level 0 = 1, higher levels = 0."""
    steps = 11
    key = jr.PRNGKey(50_000 + 100 * dim + 10 * trunc + len(batch_shape))
    x0 = 0.70 * jr.normal(key, batch_shape + (dim,), dtype=jnp.float64)
    X = jnp.repeat(x0[..., None, :], repeats=steps + 1, axis=-2)

    sig = path_signature(X, accumulate=False, trunc=trunc, axis=-2, core=CORE, seq_core=SEQ)
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
def test_path_signature_single_step_matches_exponential(dim, trunc, batch_shape):
    """For a 1-step path x0 -> x0 + v, the signature equals exp(v)."""
    key = jr.PRNGKey(60_000 + 100 * dim + 10 * trunc + len(batch_shape))
    k_start, k_dx = jr.split(key, 2)

    x0 = 0.50 * jr.normal(k_start, batch_shape + (dim,), dtype=jnp.float64)
    dx = 0.15 * jr.normal(k_dx, batch_shape + (dim,), dtype=jnp.float64)
    X = jnp.stack([x0, x0 + dx], axis=-2)

    sig = path_signature(X, accumulate=False, trunc=trunc, axis=-2, core=CORE, seq_core=SEQ)
    ref = CORE.tensor_exponential((dx,), trunc=trunc, output_zero_level=True)

    _assert_dense_allclose(
        tuple(np.asarray(a) for a in sig),
        tuple(np.asarray(a) for a in ref),
        atol=1e-12,
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# Prepended flat segment leaves signature unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("trunc", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_path_signature_unchanged_by_flat_prefix(dim, trunc, batch_shape):
    """Prepending a constant segment (zero increments) must not change the signature."""
    steps = 13
    key = jr.PRNGKey(70_000 + 100 * dim + 10 * trunc + len(batch_shape))
    k_inc, k_start = jr.split(key, 2)

    increments = 0.10 * jr.normal(k_inc, batch_shape + (steps, dim), dtype=jnp.float64)
    start = 0.25 * jr.normal(k_start, batch_shape + (dim,), dtype=jnp.float64)
    X = jnp.concatenate(
        [start[..., None, :], start[..., None, :] + jnp.cumsum(increments, axis=-2)],
        axis=-2,
    )

    flat_prefix = jnp.repeat(X[..., :1, :], repeats=4, axis=-2)
    X_with_flat = jnp.concatenate([flat_prefix, X], axis=-2)

    sig_X = path_signature(X, accumulate=False, trunc=trunc, axis=-2, core=CORE, seq_core=SEQ)
    sig_Y = path_signature(X_with_flat, accumulate=False, trunc=trunc, axis=-2, core=CORE, seq_core=SEQ)

    _assert_dense_allclose(
        tuple(np.asarray(a) for a in sig_X),
        tuple(np.asarray(a) for a in sig_Y),
    )


# ---------------------------------------------------------------------------
# Signature class matches function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path_kind", ["integrated_ou", "trig"])
@pytest.mark.parametrize(("dim", "trunc", "steps"), [(2, 3, 16), (3, 2, 12)])
@pytest.mark.parametrize("batch", [1, 2])
def test_signature_class_matches_function(path_kind, dim, trunc, steps, batch):
    """Signature(trunc=...) class produces the same result as path_signature()."""
    key = jr.PRNGKey(80_000 + 1000 * dim + 100 * trunc + 10 * batch + steps)
    X = _random_level_one_path(path_kind, key, batch=batch, steps=steps, dim=dim)
    Xj = jnp.asarray(X)

    fn_result = path_signature(Xj, accumulate=False, trunc=trunc, axis=-2, core=CORE, seq_core=SEQ)
    cls_result = Signature(trunc=trunc, core=CORE, seq_core=SEQ)(Xj, axis=-2, accumulate=False)

    _assert_dense_allclose(
        tuple(np.asarray(a) for a in fn_result),
        tuple(np.asarray(a) for a in cls_result),
        atol=1e-12,
        rtol=1e-12,
    )
