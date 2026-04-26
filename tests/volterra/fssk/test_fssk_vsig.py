"""Tests for tensordev.volterra.sss.vsig."""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from tensordev.kernel.fssk import fssk_sigkernel
from tensordev.sss import DenseLambda, FSSK, FSSK
from tensordev.sss import (
    fssk_readout,
    fssk_state,
    fssk_state_from_coef,
)
from tensordev.sss import (
    fssk_vsig,
    fssk_vsig_from_state,
)


def _allclose(a, b, *, atol=1e-10, rtol=1e-10):
    a, b = jnp.asarray(a), jnp.asarray(b)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    assert jnp.allclose(a, b, atol=atol, rtol=rtol), (
        f"max diff {jnp.max(jnp.abs(a - b))}"
    )


def _make_kernel(d: int = 3, m: int = 2, R: int = 2) -> FSSK:
    lam = DenseLambda(jnp.diag(jnp.asarray([1.0, 0.5][:R])))
    A = jnp.ones((1, m, d))
    b = jnp.asarray([[0.7, -0.2][:R]])
    return FSSK(Lambda=lam, A=A, b=b)


def _make_identity_kernel(dim: int = 2) -> FSSK:
    return FSSK.from_matrix(
        Lambda=jnp.asarray([[0.0]]),
        A=jnp.eye(dim)[None, :, :],
        b=jnp.asarray([[1.0]]),
    )


def _linear_path(S: int, d: int = 3) -> jax.Array:
    t = jnp.linspace(0.0, 1.0, S + 1)
    return jnp.stack([t * (i + 1) for i in range(d)], axis=-1)


def _batched_path(batch: int, S: int, d: int = 3) -> jax.Array:
    base = _linear_path(S, d)
    scale = jnp.linspace(0.5, 1.5, batch)[:, None, None]
    return base[None] * scale


def _project_increments(kernel: FSSK, X: jax.Array) -> jax.Array:
    dX = jnp.diff(X, axis=0)
    projected = jnp.einsum("qmd,...d->...qm", kernel.A.astype(X.dtype), dX)
    return projected[..., 0, :]


def test_fssk_vsig_matches_state_readout():
    ker = _make_kernel()
    X = _linear_path(S=6)
    trunc = 3
    lag = 0.25

    direct = fssk_vsig(X, kernel=ker, dt=0.1, trunc=trunc, tau_dt=lag)
    state = fssk_state(X, kernel=ker, dt=0.1, trunc=trunc)
    via_state = fssk_vsig_from_state(state, kernel=ker, tau_dt=lag)
    readout = fssk_readout(state, kernel=ker, tau_dt=lag)

    for r in range(trunc + 1):
        _allclose(direct[r], via_state[r])
        _allclose(direct[r], readout[r])


def test_fssk_vsig_shapes_unbatched_batched_and_blocked():
    ker = _make_kernel()
    trunc = 3

    unbatched = fssk_vsig(_linear_path(S=6), kernel=ker, dt=0.1, trunc=trunc)
    for r, z in enumerate(unbatched):
        assert z.shape == (ker.m ** r,)

    batched = fssk_vsig(_batched_path(batch=4, S=6), kernel=ker, dt=0.1, trunc=trunc)
    for r, z in enumerate(batched):
        assert z.shape == (4, ker.m ** r)

    blocked = fssk_vsig(
        _linear_path(S=6),
        kernel=ker,
        dt=0.1,
        trunc=trunc,
        block_size=2,
    )
    for r, z in enumerate(blocked):
        assert z.shape == (3, ker.m ** r)


def test_fssk_vsig_output_starting_state_prepends_unit_signature():
    ker = _make_kernel()
    trunc = 2

    sig = fssk_vsig(
        _linear_path(S=4),
        kernel=ker,
        dt=0.1,
        trunc=trunc,
        output_starting_state=True,
    )

    for r, z in enumerate(sig):
        assert z.shape == (2, ker.m ** r)
    _allclose(sig[0][0], jnp.ones((1,)))
    for r in range(1, trunc + 1):
        _allclose(sig[r][0], jnp.zeros((ker.m ** r,)))


def test_fssk_vsig_increment_input_matches_node_input():
    ker = _make_kernel()
    X = _linear_path(S=5)
    dX = jnp.diff(X, axis=0)
    trunc = 3

    from_nodes = fssk_vsig(X, kernel=ker, dt=0.1, trunc=trunc)
    from_increments = fssk_vsig(
        dX,
        kernel=ker,
        dt=0.1,
        trunc=trunc,
        axis=0,
        increment_input=True,
    )

    for r in range(trunc + 1):
        _allclose(from_nodes[r], from_increments[r])


def test_fssk_vsig_is_jittable():
    ker = _make_kernel()
    X = _linear_path(S=5)
    trunc = 2
    lag = 0.2

    out1 = fssk_vsig(X, kernel=ker, dt=0.1, trunc=trunc, tau_dt=lag)
    out2 = jax.jit(
        fssk_vsig,
        static_argnames=(
            "trunc",
            "axis",
            "block_size",
            "accumulate",
            "output_starting_state",
            "increment_input",
        ),
    )(X, kernel=ker, dt=0.1, trunc=trunc, tau_dt=lag)

    for r in range(trunc + 1):
        _allclose(out1[r], out2[r])


def test_fssk_class_vsig_matches_function_and_saves_state():
    ker = _make_kernel()
    X = _linear_path(S=6)
    trunc = 3
    model = FSSK(ker, dt=0.1, trunc=trunc, save_state=True)

    sig = model.vsig(X, tau_dt=0.25)
    expected_sig = fssk_vsig(X, kernel=ker, dt=0.1, trunc=trunc, tau_dt=0.25)
    expected_state = fssk_state(X, kernel=ker, dt=0.1, trunc=trunc)

    assert model.saved_state is not None
    for r in range(trunc + 1):
        _allclose(sig[r], expected_sig[r])
        _allclose(model.saved_state[r], expected_state[r])


def test_fssk_class_continue_vsig_matches_single_run():
    ker = _make_kernel()
    X = _linear_path(S=8)
    trunc = 3
    model = FSSK(ker, dt=0.1, trunc=trunc, save_state=True)

    model.vsig(X[:5])
    continued = model.continue_vsig(X[4:])
    full = fssk_vsig(X, kernel=ker, dt=0.1, trunc=trunc)

    for r in range(trunc + 1):
        _allclose(continued[r], full[r])


def test_fssk_class_continue_requires_saved_state():
    ker = _make_kernel()
    model = FSSK(ker, dt=0.1, trunc=2)

    with pytest.raises(ValueError, match="saved state"):
        model.continue_vsig(_linear_path(S=3))


def test_fssk_class_state_from_coef_then_readout():
    ker = _make_kernel()
    X = _linear_path(S=5)
    trunc = 3
    dt = jnp.full((5,), 0.1)
    coef = ker.coef(dt, trunc=trunc)
    y = _project_increments(ker, X)
    model = FSSK(ker, save_state=True)

    states = model.state_from_coef(y, coef=coef, save_state=True)
    expected_states = fssk_state_from_coef(y, coef=coef, axis=0)
    expected_sig = fssk_vsig(X, kernel=ker, dt=dt, trunc=trunc)
    sig = model.readout()

    for r in range(trunc + 1):
        _allclose(states[r], expected_states[r])
        _allclose(sig[r], expected_sig[r])


def test_fssk_class_vsig_kernel_matches_function():
    ker = _make_identity_kernel(dim=2)
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
    model = FSSK(ker, dt=dt, trunc=2)

    actual = model.vsig_kernel(X, Y, pairwise=True)
    expected = fssk_sigkernel(X, Y, kernel=ker, dt_x=dt, dt_y=dt, pairwise=True)

    _allclose(actual, expected)
