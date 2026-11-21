"""Steps beyond the reach of the coefficient contour are split exactly, or give NaN while traced."""
import jax
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
from tensordev.sss import FSSK
from tensordev.sss.kernel import contour_substeps
from tensordev.util import random_trigonometric_polynomial_paths

DT = 1.0 / 8


def _refine(X, k):
    """Insert k - 1 nodes on every segment; the piecewise-linear path is unchanged."""
    inc = jnp.diff(X, axis=-2)
    return jnp.concatenate([X[..., :1, :], X[..., :1, :] + jnp.cumsum(jnp.repeat(inc / k, k, axis=-2), axis=-2)], axis=-2)


def _oscillation(omega):
    return FSSK.from_jordan(
        A=jnp.eye(2)[None], b=jnp.ones((1, 3)), real_rates=jnp.array([1.0]), real_sizes=(1,),
        osc_decays=jnp.array([0.5]), osc_freqs=jnp.array([omega]), osc_sizes=(1,),
    )


def _growth(rate):
    return FSSK.from_jordan(A=jnp.eye(2)[None], b=jnp.ones((1, 1)), real_rates=jnp.array([rate]), real_sizes=(1,))


def _max_rel(a, b):
    return max(float(jnp.max(jnp.abs(x - y)) / jnp.max(jnp.abs(y))) for x, y in zip(a[1:], b[1:]))


@pytest.fixture
def X():
    return random_trigonometric_polynomial_paths(batch=2, steps=8, dim=2, key=3)


@pytest.mark.parametrize("kernel", [_oscillation(50 / DT), _oscillation(17 / DT), _growth(-7 / DT)],
                         ids=["omega_dt_50", "omega_dt_17", "rate_dt_-7"])
def test_long_steps_are_split_exactly(X, kernel):
    assert contour_substeps(kernel, DT) > 1
    reference = td.fssk_vsig(_refine(X, 128), kernel=kernel, dt=DT / 128, trunc=3)
    got = td.fssk_vsig(X, kernel=kernel, dt=DT, trunc=3)
    assert _max_rel(got, reference) < 1e-12


def test_short_steps_are_not_split():
    assert contour_substeps(_oscillation(2 / DT), DT) == 1
    assert contour_substeps(_growth(5.0), DT) == 1


def test_split_keeps_per_step_outputs_and_dt_arrays(X):
    kernel = _oscillation(30 / DT)
    dt_steps = jnp.full((8,), DT)
    got = td.fssk_vsig(X, kernel=kernel, dt=dt_steps, trunc=3, block_size=1, output_starting_state=True)
    k = 64
    reference = td.fssk_vsig(_refine(X, k), kernel=kernel, dt=DT / k, trunc=3, block_size=k,
                             output_starting_state=True)
    assert got[1].shape == reference[1].shape == (2, 9, 2)
    assert _max_rel(got, reference) < 1e-12


@pytest.mark.parametrize("kernel, finite", [(_oscillation(10 / DT), True), (_oscillation(14 / DT), False),
                                            (_growth(-5 / DT), True), (_growth(-6 / DT), False)])
def test_traced_kernel_gives_nan_beyond_the_contour(X, kernel, finite):
    out = jax.jit(lambda k, x: td.fssk_vsig(x, kernel=k, dt=DT, trunc=3))(kernel, X)
    assert all(bool(jnp.all(jnp.isfinite(level))) for level in out) == finite


def test_substeps_need_concrete_values():
    kernel = _oscillation(50 / DT)
    seen = []
    jax.jit(lambda k: seen.append(contour_substeps(k, DT)) or 0.0)(kernel)
    assert seen == [1]
    assert contour_substeps(kernel, DT) == 13
