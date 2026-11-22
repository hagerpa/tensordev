"""``fssk_vsig`` on steps beyond the bounds of :func:`contour_substeps`: exact sub-steps, or NaN when traced."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

import tensordev as td
from tensordev.sss import FSSK
from tensordev.sss.kernel import contour_substeps
from tensordev.util import random_trigonometric_polynomial_paths

DT = 1.0 / 8


def _substeps(X, k):
    """Increments of the same piecewise-linear path with every step split into ``k`` equal ones."""
    return jnp.repeat(jnp.diff(X, axis=-2) / k, k, axis=-2)


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
    reference = td.fssk_vsig(_substeps(X, 128), kernel=kernel, dt=DT / 128, trunc=3, increment_input=True)
    assert _max_rel(td.fssk_vsig(X, kernel=kernel, dt=DT, trunc=3), reference) < 1e-12


def test_short_steps_are_not_split():
    assert contour_substeps(_oscillation(2 / DT), DT) == 1
    assert contour_substeps(_growth(5.0), DT) == 1


def test_split_keeps_per_step_outputs_and_dt_arrays(X):
    kernel = _oscillation(30 / DT)
    got = td.fssk_vsig(X, kernel=kernel, dt=jnp.full((8,), DT), trunc=3, block_size=1, output_starting_state=True)
    reference = td.fssk_vsig(_substeps(X, 64), kernel=kernel, dt=DT / 64, trunc=3, block_size=64,
                             output_starting_state=True, increment_input=True)
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
