from jax import config

config.update("jax_enable_x64", True)

import os

import numpy as np
import pytest
import jax.random as jr
from jax import numpy as jnp

from tensordev import Jax
from tensordev.kernel.free import free_kernel

from random_paths import (
    integrated_ou_first_on_path,
    path_to_increments,
    random_trigonometric_polynomial_paths_first_on,
)

CORE = Jax()

STEP_SCALE = int(os.environ.get("TENSORDEV_TEST_STEP_SCALE", "1"))
MAX_REFERENCE_TRUNC = 16


def _scaled_steps(steps: int) -> int:
    return STEP_SCALE * steps


def _reference_trunc(trunc: int, extra: int) -> int:
    """
    Reference truncation used in the free-development comparison, capped to avoid
    overly expensive tests.
    """
    return min(MAX_REFERENCE_TRUNC, trunc + extra)


def _random_first_on_path(
        path_kind: str,
        key,
        *,
        batch,
        steps,
        dim,
        trunc,
        level_lambda: float = 1.0,
        factorial: bool = False,
):
    """
    Build a random DenseElemFirstOn path from one of the supported path families,
    with higher levels scaled according to the requested decay regime.
    """
    if path_kind == "integrated_ou":
        return integrated_ou_first_on_path(
            key,
            batch=batch,
            steps=steps,
            dim=dim,
            trunc=trunc,
            level_lambda=level_lambda,
            factorial=factorial,
        )

    if path_kind == "trig":
        return random_trigonometric_polynomial_paths_first_on(
            key,
            batch=batch,
            steps=steps,
            dim=dim,
            trunc=trunc,
            level_lambda=level_lambda,
            factorial=factorial,
        )

    raise ValueError(f"Unknown path_kind={path_kind!r}.")


def _broadcast_pairwise_levels(x, y):
    """
    Put batch axes of two graded elements into outer-product position, matching
    the broadcasting convention used in the PDE code.
    """
    batch_x = x[0].shape[:-1]
    batch_y = y[0].shape[:-1]

    nx = len(batch_x)
    ny = len(batch_y)

    x = tuple(level.reshape(batch_x + (1,) * ny + level.shape[-1:]) for level in x)
    y = tuple(level.reshape((1,) * nx + batch_y + level.shape[-1:]) for level in y)
    return x, y


def _pairwise_inner_products(dev_x, dev_y, core=CORE):
    """
    Pairwise inner products between two batches of developed tensors, formed by
    broadcasting rather than Python loops.
    """
    dev_x_bc, dev_y_bc = _broadcast_pairwise_levels(dev_x, dev_y)
    return np.asarray(core.tensor_inner_product(dev_x_bc, dev_y_bc))


DECAY_CASES = [
    pytest.param(1.0, False, id="no_decay"),
    pytest.param(0.5, False, id="geometric_decay"),
    pytest.param(1.0, True, id="factorial_decay"),
]


@pytest.mark.heavy
@pytest.mark.parametrize("level_lambda,factorial", DECAY_CASES)
@pytest.mark.parametrize(
    ("path_kind", "dim", "trunc"),
    [
        ("integrated_ou", 2, 2),
        ("integrated_ou", 3, 2),
        ("trig", 2, 3),
        ("trig", 2, 1),
        ("integrated_ou", 2, 4),
        ("integrated_ou", 3, 1),
    ],
)
def test_free_kernel_matches_inner_product_of_free_developments(
        path_kind,
        dim,
        trunc,
        level_lambda,
        factorial,
):
    """
    Compare the terminal PDE solver against the inner product of the corresponding
    free developments for different random path families and level-scaling regimes.

    Regimes:
    - no decay:         level k scaled like 1
    - geometric decay:  level k scaled like lambda**k
    - factorial decay:  level k scaled like lambda**k / k!
    """
    key = jr.PRNGKey(
        2026
        + 100 * (path_kind == "trig")
        + 10 * dim
        + trunc
        + 1000 * factorial
        + int(100 * level_lambda)
    )

    batch_x = 3
    batch_y = 4
    steps_x = _scaled_steps(40)
    steps_y = _scaled_steps(48)

    X = _random_first_on_path(
        path_kind,
        key,
        batch=batch_x,
        steps=steps_x,
        dim=dim,
        trunc=trunc,
        level_lambda=level_lambda,
        factorial=factorial,
    )
    Y = _random_first_on_path(
        path_kind,
        jr.fold_in(key, 1),
        batch=batch_y,
        steps=steps_y,
        dim=dim,
        trunc=trunc,
        level_lambda=level_lambda,
        factorial=factorial,
    )

    dx = path_to_increments(X)
    dy = path_to_increments(Y)

    ours = free_kernel(dx, dy, evaluate="terminal", return_fg=False, pairwise=True, backend="scan", dyadic_order=2,
                       increment_in=True)

    ref_trunc = _reference_trunc(trunc, extra=10)

    dev_x = CORE.tensor_development(
        dx,
        axis=-2,
        trunc=ref_trunc,
        accumulate=False,
        output_starting_point=False,
        increment_input=True,
    )
    dev_y = CORE.tensor_development(
        dy,
        axis=-2,
        trunc=ref_trunc,
        accumulate=False,
        output_starting_point=False,
        increment_input=True,
    )

    ref = _pairwise_inner_products(dev_x, dev_y, core=CORE)

    np.testing.assert_allclose(
        np.asarray(ours),
        ref,
        rtol=2e-3,
        atol=2e-3,
    )


@pytest.mark.parametrize("path_kind", ["integrated_ou", "trig"])
@pytest.mark.parametrize("dim", [2, 3])
def test_level_one_pairwise_matches_nested_single_pair_calls(path_kind, dim):
    """
    Level-1 sanity check: pairwise mode should agree with nested single-pair
    evaluations. This is the right P=1 regression test now that the scalar
    branch uses the quadratic sigkernel-style update.
    """
    key = jr.PRNGKey(5000 + 100 * (path_kind == "trig") + dim)

    batch_x = 2
    batch_y = 3
    steps_x = _scaled_steps(24)
    steps_y = _scaled_steps(28)

    X = _random_first_on_path(
        path_kind,
        key,
        batch=batch_x,
        steps=steps_x,
        dim=dim,
        trunc=1,
    )
    Y = _random_first_on_path(
        path_kind,
        jr.fold_in(key, 1),
        batch=batch_y,
        steps=steps_y,
        dim=dim,
        trunc=1,
    )

    dx = path_to_increments(X)
    dy = path_to_increments(Y)

    pairwise = free_kernel(dx, dy, evaluate="terminal", return_fg=False, pairwise=True, backend="scan", dyadic_order=1,
                           increment_in=True)

    ref = np.zeros((batch_x, batch_y), dtype=np.float64)
    for i in range(batch_x):
        dx_i = tuple(level[i] for level in dx)
        for j in range(batch_y):
            dy_j = tuple(level[j] for level in dy)
            ref[i, j] = float(
                free_kernel(dx_i, dy_j, evaluate="terminal", return_fg=False, pairwise=False, backend="scan",
                            dyadic_order=1, increment_in=True)
            )

    np.testing.assert_allclose(
        np.asarray(pairwise),
        ref,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("path_kind", "dim", "trunc"),
    [
        ("integrated_ou", 2, 2),
        ("integrated_ou", 3, 1),
        ("trig", 2, 1),
        ("trig", 3, 1),
    ],
)
def test_terminal_matches_last_grid_value(path_kind, dim, trunc):
    """
    Terminal output should coincide with the last value of the full grid output.
    This is checked for w as well as for the tensor-valued components f and g.
    """
    key = jr.PRNGKey(6000 + 100 * (path_kind == "trig") + 10 * dim + trunc)

    batch_x = 2
    batch_y = 3
    steps_x = _scaled_steps(16)
    steps_y = _scaled_steps(20)

    X = _random_first_on_path(
        path_kind,
        key,
        batch=batch_x,
        steps=steps_x,
        dim=dim,
        trunc=trunc,
    )
    Y = _random_first_on_path(
        path_kind,
        jr.fold_in(key, 1),
        batch=batch_y,
        steps=steps_y,
        dim=dim,
        trunc=trunc,
    )

    dx = path_to_increments(X)
    dy = path_to_increments(Y)

    w_term, f_term, g_term = free_kernel(dx, dy, evaluate="terminal", return_fg=True, pairwise=True, backend="scan",
                                         increment_in=True)
    w_grid, f_grid, g_grid = free_kernel(dx, dy, evaluate="grid", return_fg=True, pairwise=True, backend="scan",
                                         increment_in=True)

    np.testing.assert_allclose(
        np.asarray(w_term),
        np.asarray(w_grid[..., -1, -1]),
        rtol=1e-12,
        atol=1e-12,
    )

    for k in range(len(f_term)):
        np.testing.assert_allclose(
            np.asarray(f_term[k]),
            np.asarray(f_grid[k][..., -1, -1, :]),
            rtol=1e-12,
            atol=1e-12,
        )

    for k in range(len(g_term)):
        np.testing.assert_allclose(
            np.asarray(g_term[k]),
            np.asarray(g_grid[k][..., -1, -1, :]),
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    ("path_kind", "dim", "trunc"),
    [
        ("integrated_ou", 2, 2),
        ("integrated_ou", 3, 1),
        ("trig", 2, 1),
        ("trig", 3, 1),
    ],
)
def test_free_kernel_is_symmetric_under_swapping_inputs(path_kind, dim, trunc):
    """
    Swapping x and y should transpose the pairwise terminal kernel matrix, and
    should transpose the two grid axes in the single-pair grid output.
    """
    key = jr.PRNGKey(7000 + 100 * (path_kind == "trig") + 10 * dim + trunc)

    batch_x = 2
    batch_y = 3
    steps_x = _scaled_steps(18)
    steps_y = _scaled_steps(22)

    X = _random_first_on_path(
        path_kind,
        key,
        batch=batch_x,
        steps=steps_x,
        dim=dim,
        trunc=trunc,
    )
    Y = _random_first_on_path(
        path_kind,
        jr.fold_in(key, 1),
        batch=batch_y,
        steps=steps_y,
        dim=dim,
        trunc=trunc,
    )

    dx = path_to_increments(X)
    dy = path_to_increments(Y)

    k_xy = free_kernel(dx, dy, evaluate="terminal", return_fg=False, pairwise=True, backend="scan", increment_in=True)
    k_yx = free_kernel(dy, dx, evaluate="terminal", return_fg=False, pairwise=True, backend="scan", increment_in=True)

    np.testing.assert_allclose(
        np.asarray(k_xy),
        np.asarray(k_yx).T,
        rtol=1e-12,
        atol=1e-12,
    )

    dx0 = tuple(level[0] for level in dx)
    dy0 = tuple(level[0] for level in dy)

    w_xy = free_kernel(dx0, dy0, evaluate="grid", return_fg=False, pairwise=False, backend="scan", increment_in=True)
    w_yx = free_kernel(dy0, dx0, evaluate="grid", return_fg=False, pairwise=False, backend="scan", increment_in=True)

    np.testing.assert_allclose(
        np.asarray(w_xy),
        np.asarray(w_yx).T,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("path_kind", "dim", "trunc"),
    [
        ("integrated_ou", 2, 2),
        ("integrated_ou", 3, 1),
        ("trig", 2, 1),
        ("trig", 3, 1),
    ],
)
def test_pairwise_matches_nested_single_pair_calls(path_kind, dim, trunc):
    """
    Pairwise mode should agree with nested single-pair evaluations.
    """
    key = jr.PRNGKey(8000 + 100 * (path_kind == "trig") + 10 * dim + trunc)

    batch_x = 2
    batch_y = 3
    steps_x = _scaled_steps(14)
    steps_y = _scaled_steps(17)

    X = _random_first_on_path(
        path_kind,
        key,
        batch=batch_x,
        steps=steps_x,
        dim=dim,
        trunc=trunc,
    )
    Y = _random_first_on_path(
        path_kind,
        jr.fold_in(key, 1),
        batch=batch_y,
        steps=steps_y,
        dim=dim,
        trunc=trunc,
    )

    dx = path_to_increments(X)
    dy = path_to_increments(Y)

    pairwise = free_kernel(dx, dy, evaluate="terminal", return_fg=False, pairwise=True, backend="scan",
                           increment_in=True)

    ref = np.zeros((batch_x, batch_y), dtype=np.float64)
    for i in range(batch_x):
        dx_i = tuple(level[i] for level in dx)
        for j in range(batch_y):
            dy_j = tuple(level[j] for level in dy)
            ref[i, j] = float(
                free_kernel(dx_i, dy_j, evaluate="terminal", return_fg=False, pairwise=False, backend="scan",
                            increment_in=True)
            )

    np.testing.assert_allclose(
        np.asarray(pairwise),
        ref,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("quadrature", ["left", "midpoint", "trapezoid"])
def test_callable_input_matches_increment_input(quadrature):
    """
    For constant characteristic velocities, callable input should agree exactly
    with increment input for any supported quadrature rule.
    """
    grid_x = jnp.array([0.0, 0.1, 0.35, 0.9, 1.0], dtype=jnp.float64)
    grid_y = jnp.array([0.0, 0.2, 0.5, 0.7, 1.0], dtype=jnp.float64)

    x_consts = (
        jnp.array([0.2, -0.1], dtype=jnp.float64),
        jnp.array([0.05, -0.03, 0.04, 0.02], dtype=jnp.float64),
    )
    y_consts = (
        jnp.array([-0.15, 0.25], dtype=jnp.float64),
        jnp.array([0.01, 0.06, -0.02, 0.03], dtype=jnp.float64),
    )

    def x_velocity(t):
        t = jnp.asarray(t, dtype=jnp.float64)
        return tuple(jnp.broadcast_to(c, t.shape + c.shape) for c in x_consts)

    def y_velocity(t):
        t = jnp.asarray(t, dtype=jnp.float64)
        return tuple(jnp.broadcast_to(c, t.shape + c.shape) for c in y_consts)

    dt_x = jnp.diff(grid_x)
    dt_y = jnp.diff(grid_y)

    dx = tuple(dt_x[:, None] * c[None, :] for c in x_consts)
    dy = tuple(dt_y[:, None] * c[None, :] for c in y_consts)

    out_callable = free_kernel((x_velocity, grid_x), (y_velocity, grid_y), evaluate="terminal", return_fg=True,
                               pairwise=False, backend="scan", quadrature=quadrature, increment_in=True)
    out_increments = free_kernel(dx, dy, evaluate="terminal", return_fg=True, pairwise=False, backend="scan",
                                 increment_in=True)

    w_c, f_c, g_c = out_callable
    w_i, f_i, g_i = out_increments

    np.testing.assert_allclose(
        np.asarray(w_c),
        np.asarray(w_i),
        rtol=1e-12,
        atol=1e-12,
    )

    for k in range(len(f_c)):
        np.testing.assert_allclose(
            np.asarray(f_c[k]),
            np.asarray(f_i[k]),
            rtol=1e-12,
            atol=1e-12,
        )

    for k in range(len(g_c)):
        np.testing.assert_allclose(
            np.asarray(g_c[k]),
            np.asarray(g_i[k]),
            rtol=1e-12,
            atol=1e-12,
        )
