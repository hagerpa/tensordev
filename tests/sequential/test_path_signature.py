from jax import config
config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax.random as jr
import jax.numpy as jnp

from tensordev import Jax

from random_paths import (
    integrated_ou_first_on_path,
    random_trigonometric_polynomial_paths_first_on,
)

CORE = Jax()


def _random_level_one_path(path_kind: str, key, *, batch: int, steps: int, dim: int):
    if path_kind == "integrated_ou":
        X = integrated_ou_first_on_path(
            key,
            batch=batch,
            steps=steps,
            dim=dim,
            trunc=1,
        )
        return np.asarray(X[0])

    if path_kind == "trig":
        X = random_trigonometric_polynomial_paths_first_on(
            key,
            batch=batch,
            steps=steps,
            dim=dim,
            trunc=1,
        )
        return np.asarray(X[0])

    raise ValueError(f"Unknown path_kind={path_kind!r}.")


def _iisignature_levels(path: np.ndarray, trunc: int):
    """
    Return dense tensor levels (1, S^1, ..., S^trunc) from iisignature.sig.
    """
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
        np.testing.assert_allclose(
            a, e,
            atol=atol,
            rtol=rtol,
            err_msg=f"{msg} level {n} mismatch",
        )


@pytest.mark.parametrize("path_kind", ["integrated_ou", "trig"])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("steps", [8, 25])
@pytest.mark.parametrize("memory_consumption", ["low", "high"])
def test_tensor_path_signature_level_1_matches_endpoint_increment(
    path_kind, dim, batch, steps, memory_consumption
):
    """
    First level of the signature must equal the path increment X_T - X_0.
    This is the most basic sanity check and should fail immediately if level 1
    is wrong.
    """
    key = jr.PRNGKey(10_000 + 100 * dim + 10 * batch + steps)
    X = _random_level_one_path(
        path_kind,
        key,
        batch=batch,
        steps=steps,
        dim=dim,
    )

    sig = CORE.tensor_path_signature(
        jnp.asarray(X),
        axis=-2,
        trunc=3,
        memory_consumption=memory_consumption,
    )

    expected_level_1 = X[:, -1, :] - X[:, 0, :]

    np.testing.assert_allclose(
        np.asarray(sig[1]),
        expected_level_1,
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("path_kind", ["integrated_ou", "trig"])
@pytest.mark.parametrize(
    ("dim", "trunc", "steps"),
    [
        (1, 4, 20),
        (2, 3, 20),
        (3, 2, 16),
    ],
)
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("memory_consumption", ["low", "high"])
def test_tensor_path_signature_matches_iisignature_on_random_paths(
    path_kind, dim, trunc, steps, batch, memory_consumption
):
    """
    Direct comparison of tensordev.tensor_path_signature against iisignature on
    the same random paths already used elsewhere in the test suite.
    """
    key = jr.PRNGKey(20_000 + 1000 * dim + 100 * trunc + 10 * batch + steps)
    X = _random_level_one_path(
        path_kind,
        key,
        batch=batch,
        steps=steps,
        dim=dim,
    )

    got = CORE.tensor_path_signature(
        jnp.asarray(X),
        axis=-2,
        trunc=trunc,
        memory_consumption=memory_consumption,
    )
    got = tuple(np.asarray(level) for level in got)

    for b in range(batch):
        ref = _iisignature_levels(X[b], trunc=trunc)
        got_b = tuple(level[b] for level in got)
        _assert_dense_allclose(
            got_b,
            ref,
            atol=1e-10,
            rtol=1e-10,
            msg=f"path_kind={path_kind}, batch_index={b}",
        )


@pytest.mark.parametrize("path_kind", ["integrated_ou", "trig"])
@pytest.mark.parametrize(("dim", "trunc", "steps"), [(1, 4, 12), (2, 3, 12)])
def test_tensor_path_signature_prefixes_match_iisignature(path_kind, dim, trunc, steps):
    """
    Prefix signatures are useful because they test the scan/product logic, not
    just the final whole-path result.
    """
    key = jr.PRNGKey(30_000 + 1000 * dim + 100 * trunc + steps)
    X = _random_level_one_path(
        path_kind,
        key,
        batch=1,
        steps=steps,
        dim=dim,
    )

    got = CORE.tensor_path_signature(
        jnp.asarray(X),
        axis=-2,
        trunc=trunc,
        block_size=1,
        accumulate=True,
        output_starting_point=True,
        memory_consumption="low",
    )
    got = tuple(np.asarray(level[0]) for level in got)

    # output_starting_point=True gives:
    # index 0   -> unit
    # index j>0 -> signature of prefix up to time-point j
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


