"""
Tests that the scan and wavefront backends produce identical results
for ``free_kernel`` across all evaluation modes and configurations.
"""

from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax.random as jr
from jax import numpy as jnp

from tensordev.kernel.free import free_kernel

from random_paths import (
    integrated_ou_first_on_path,
    random_trigonometric_polynomial_paths_first_on,
    path_to_increments,
)

BACKENDS = ("scan", "wavefront")

PATH_KINDS = ["integrated_ou", "trig"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_increments(path_kind, key, *, batch, steps, dim, trunc):
    """Build increment tuple from one of the random path families."""
    if path_kind == "integrated_ou":
        X = integrated_ou_first_on_path(
            key, batch=batch, steps=steps, dim=dim, trunc=trunc,
        )
    elif path_kind == "trig":
        X = random_trigonometric_polynomial_paths_first_on(
            key, batch=batch, steps=steps, dim=dim, trunc=trunc,
        )
    else:
        raise ValueError(path_kind)
    return path_to_increments(X)


def _assert_close(a, b, *, rtol=1e-12, atol=1e-12, label=""):
    np.testing.assert_allclose(
        np.asarray(a), np.asarray(b), rtol=rtol, atol=atol, err_msg=label
    )


# ── terminal evaluation ─────────────────────────────────────────────────────

@pytest.mark.parametrize("dyadic_order", [0, 1, 2])
@pytest.mark.parametrize("trunc", [1, 2])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("path_kind", PATH_KINDS)
def test_terminal_no_batch(path_kind, dim, trunc, dyadic_order):
    """Single-pair terminal: scan == wavefront."""
    key = jr.PRNGKey(100 + dim + trunc + dyadic_order)
    kx, ky = jr.split(key)
    dx = _make_increments(path_kind, kx, batch=1, steps=8, dim=dim, trunc=trunc)
    dy = _make_increments(path_kind, ky, batch=1, steps=6, dim=dim, trunc=trunc)
    # squeeze batch to get unbatched inputs
    dx = tuple(d[0] for d in dx)
    dy = tuple(d[0] for d in dy)

    results = {}
    for b in BACKENDS:
        results[b] = free_kernel(
            dx, dy, evaluate="terminal", backend=b,
            dyadic_order=dyadic_order, increment_in=True,
        )
    _assert_close(results["scan"], results["wavefront"],
                  label=f"terminal no_batch {path_kind} d={dim} P={trunc} dy={dyadic_order}")


@pytest.mark.parametrize("dyadic_order", [0, 1])
@pytest.mark.parametrize("trunc", [1, 2])
@pytest.mark.parametrize("path_kind", PATH_KINDS)
def test_terminal_pairwise(path_kind, trunc, dyadic_order):
    """Pairwise terminal: scan == wavefront."""
    key = jr.PRNGKey(200 + trunc + dyadic_order)
    kx, ky = jr.split(key)
    dx = _make_increments(path_kind, kx, batch=3, steps=8, dim=2, trunc=trunc)
    dy = _make_increments(path_kind, ky, batch=4, steps=6, dim=2, trunc=trunc)

    results = {}
    for b in BACKENDS:
        results[b] = free_kernel(
            dx, dy, evaluate="terminal", pairwise=True, backend=b,
            dyadic_order=dyadic_order, increment_in=True,
        )
    _assert_close(results["scan"], results["wavefront"],
                  label=f"terminal pairwise {path_kind} P={trunc} dy={dyadic_order}")


@pytest.mark.parametrize("dyadic_order", [0, 1])
@pytest.mark.parametrize("trunc", [1, 2])
@pytest.mark.parametrize("path_kind", PATH_KINDS)
def test_terminal_batch(path_kind, trunc, dyadic_order):
    """Batched terminal (non-pairwise): scan == wavefront."""
    key = jr.PRNGKey(300 + trunc + dyadic_order)
    kx, ky = jr.split(key)
    dx = _make_increments(path_kind, kx, batch=3, steps=8, dim=2, trunc=trunc)
    dy = _make_increments(path_kind, ky, batch=3, steps=6, dim=2, trunc=trunc)

    results = {}
    for b in BACKENDS:
        results[b] = free_kernel(
            dx, dy, evaluate="terminal", backend=b,
            dyadic_order=dyadic_order, increment_in=True,
        )
    _assert_close(results["scan"], results["wavefront"],
                  label=f"terminal batch {path_kind} P={trunc} dy={dyadic_order}")


# ── terminal with return_fg ──────────────────────────────────────────────────

@pytest.mark.parametrize("dyadic_order", [0, 1])
@pytest.mark.parametrize("trunc", [1, 2])
@pytest.mark.parametrize("path_kind", PATH_KINDS)
def test_terminal_return_fg(path_kind, trunc, dyadic_order):
    """Terminal with return_fg: scan == wavefront for w, f, g."""
    key = jr.PRNGKey(400 + trunc + dyadic_order)
    kx, ky = jr.split(key)
    dx = _make_increments(path_kind, kx, batch=2, steps=8, dim=2, trunc=trunc)
    dy = _make_increments(path_kind, ky, batch=3, steps=6, dim=2, trunc=trunc)

    results = {}
    for b in BACKENDS:
        results[b] = free_kernel(
            dx, dy, evaluate="terminal", return_fg=True, pairwise=True,
            backend=b, dyadic_order=dyadic_order, increment_in=True,
        )

    w_s, f_s, g_s = results["scan"]
    w_w, f_w, g_w = results["wavefront"]

    _assert_close(w_s, w_w, label=f"w {path_kind}")
    for k in range(len(f_s)):
        _assert_close(f_s[k], f_w[k], label=f"f[{k}] {path_kind}")
    for k in range(len(g_s)):
        _assert_close(g_s[k], g_w[k], label=f"g[{k}] {path_kind}")


# ── grid evaluation ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("dyadic_order", [0, 1])
@pytest.mark.parametrize("trunc", [1, 2])
@pytest.mark.parametrize("path_kind", PATH_KINDS)
def test_grid_no_batch(path_kind, trunc, dyadic_order):
    """Single-pair grid: scan == wavefront."""
    key = jr.PRNGKey(500 + trunc + dyadic_order)
    kx, ky = jr.split(key)
    dx = _make_increments(path_kind, kx, batch=1, steps=8, dim=2, trunc=trunc)
    dy = _make_increments(path_kind, ky, batch=1, steps=6, dim=2, trunc=trunc)
    dx = tuple(d[0] for d in dx)
    dy = tuple(d[0] for d in dy)

    results = {}
    for b in BACKENDS:
        results[b] = free_kernel(
            dx, dy, evaluate="grid", backend=b,
            dyadic_order=dyadic_order, increment_in=True,
        )
    _assert_close(results["scan"], results["wavefront"],
                  label=f"grid no_batch {path_kind} P={trunc} dy={dyadic_order}")


@pytest.mark.parametrize("dyadic_order", [0, 1])
@pytest.mark.parametrize("trunc", [1, 2])
@pytest.mark.parametrize("path_kind", PATH_KINDS)
def test_grid_pairwise(path_kind, trunc, dyadic_order):
    """Pairwise grid: scan == wavefront."""
    key = jr.PRNGKey(600 + trunc + dyadic_order)
    kx, ky = jr.split(key)
    dx = _make_increments(path_kind, kx, batch=2, steps=8, dim=2, trunc=trunc)
    dy = _make_increments(path_kind, ky, batch=3, steps=6, dim=2, trunc=trunc)

    results = {}
    for b in BACKENDS:
        results[b] = free_kernel(
            dx, dy, evaluate="grid", pairwise=True, backend=b,
            dyadic_order=dyadic_order, increment_in=True,
        )
    _assert_close(results["scan"], results["wavefront"],
                  label=f"grid pairwise {path_kind} P={trunc} dy={dyadic_order}")


@pytest.mark.parametrize("dyadic_order", [0, 1])
@pytest.mark.parametrize("trunc", [1, 2])
@pytest.mark.parametrize("path_kind", PATH_KINDS)
def test_grid_batch(path_kind, trunc, dyadic_order):
    """Batched grid (non-pairwise): scan == wavefront."""
    key = jr.PRNGKey(700 + trunc + dyadic_order)
    kx, ky = jr.split(key)
    dx = _make_increments(path_kind, kx, batch=3, steps=8, dim=2, trunc=trunc)
    dy = _make_increments(path_kind, ky, batch=3, steps=6, dim=2, trunc=trunc)

    results = {}
    for b in BACKENDS:
        results[b] = free_kernel(
            dx, dy, evaluate="grid", backend=b,
            dyadic_order=dyadic_order, increment_in=True,
        )
    _assert_close(results["scan"], results["wavefront"],
                  label=f"grid batch {path_kind} P={trunc} dy={dyadic_order}")


# ── grid with return_fg ──────────────────────────────────────────────────────

@pytest.mark.parametrize("dyadic_order", [0, 1])
@pytest.mark.parametrize("trunc", [1, 2])
@pytest.mark.parametrize("path_kind", PATH_KINDS)
def test_grid_return_fg(path_kind, trunc, dyadic_order):
    """Grid with return_fg: scan == wavefront for w, f, g."""
    key = jr.PRNGKey(800 + trunc + dyadic_order)
    kx, ky = jr.split(key)
    dx = _make_increments(path_kind, kx, batch=2, steps=7, dim=2, trunc=trunc)
    dy = _make_increments(path_kind, ky, batch=3, steps=5, dim=2, trunc=trunc)

    results = {}
    for b in BACKENDS:
        results[b] = free_kernel(
            dx, dy, evaluate="grid", return_fg=True, pairwise=True,
            backend=b, dyadic_order=dyadic_order, increment_in=True,
        )

    w_s, f_s, g_s = results["scan"]
    w_w, f_w, g_w = results["wavefront"]

    _assert_close(w_s, w_w, label=f"grid w {path_kind}")
    for k in range(len(f_s)):
        _assert_close(f_s[k], f_w[k], label=f"grid f[{k}] {path_kind}")
    for k in range(len(g_s)):
        _assert_close(g_s[k], g_w[k], label=f"grid g[{k}] {path_kind}")


# ── asymmetric dyadic order (tuple) ─────────────────────────────────────────

@pytest.mark.parametrize("dyadic_order", [(0, 1), (1, 0), (1, 2)])
@pytest.mark.parametrize("path_kind", PATH_KINDS)
def test_terminal_asymmetric_dyadic_order(path_kind, dyadic_order):
    """Asymmetric (tuple) dyadic order: scan == wavefront."""
    key = jr.PRNGKey(900 + dyadic_order[0] * 10 + dyadic_order[1])
    kx, ky = jr.split(key)
    dx = _make_increments(path_kind, kx, batch=2, steps=8, dim=2, trunc=2)
    dy = _make_increments(path_kind, ky, batch=3, steps=6, dim=2, trunc=2)

    results = {}
    for b in BACKENDS:
        results[b] = free_kernel(
            dx, dy, evaluate="terminal", pairwise=True, backend=b,
            dyadic_order=dyadic_order, increment_in=True,
        )
    _assert_close(results["scan"], results["wavefront"],
                  label=f"terminal asym dyadic={dyadic_order} {path_kind}")
