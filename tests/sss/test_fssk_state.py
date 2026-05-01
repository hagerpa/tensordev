from __future__ import annotations
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest
from tensordev.sss import DenseLambda, FSSK
from tensordev.sss.state_update import fssk_readout, fssk_state, fssk_state_from_coef


def _allclose(a, b, *, atol=1e-10, rtol=1e-10):
    a, b = jnp.asarray(a), jnp.asarray(b)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    assert jnp.allclose(a, b, atol=atol, rtol=rtol), f"max diff {jnp.max(jnp.abs(a - b))}"


def _make_kernel(d=3, m=2, R=2):
    lam = DenseLambda(jnp.diag(jnp.array([1.0, 0.5][:R])))
    return FSSK(Lambda=lam, A=jnp.ones((1, m, d)), b=jnp.array([[0.7, -0.2][:R]]))


def _linear_path(S, d=3):
    t = jnp.linspace(0.0, 1.0, S + 1)
    return jnp.stack([t * (i + 1) for i in range(d)], axis=-1)


def _batched_path(batch, S, d=3):
    return _linear_path(S, d)[None] * jnp.linspace(0.5, 1.5, batch)[:, None, None]


# -- shapes

def test_output_shape_unbatched_no_blocking():
    ker = _make_kernel()
    trunc = 3
    s = fssk_state(_linear_path(S=6), kernel=ker, dt=0.1, trunc=trunc)
    assert len(s) == trunc
    for r, z in enumerate(s):
        assert z.shape == (1, 1, ker.state_dim, ker.m ** (r + 1))


def test_output_shape_batched_no_blocking():
    ker = _make_kernel()
    trunc = 2
    s = fssk_state(_batched_path(4, S=6), kernel=ker, dt=0.1, trunc=trunc)
    assert len(s) == trunc
    for r, z in enumerate(s):
        assert z.shape == (4, 1, 1, ker.state_dim, ker.m ** (r + 1))


def test_output_shape_blocking_emits_block_axis():
    ker = _make_kernel()
    trunc = 2
    s = fssk_state(_linear_path(S=6), kernel=ker, dt=0.1, trunc=trunc, block_size=2)
    assert len(s) == trunc
    for r, z in enumerate(s):
        assert z.shape == (3, 1, 1, ker.state_dim, ker.m ** (r + 1))


def test_output_starting_state_prepends_seed():
    ker = _make_kernel()
    trunc = 2
    s = fssk_state(_linear_path(S=4), kernel=ker, dt=0.1, trunc=trunc, output_starting_state=True)
    assert len(s) == trunc
    for r, z in enumerate(s):
        assert z.shape[0] == 2
    for z in s:
        assert jnp.all(z[0] == 0.0)


def test_fssk_state_from_coef_shapes_match_fssk_state():
    ker = _make_kernel()
    S, trunc = 5, 3
    dt = jnp.full((S,), 0.1)
    X = _linear_path(S)
    y = jnp.einsum("qmd,...d->...qm", ker.A.astype(jnp.float64), jnp.diff(X, axis=0))[..., 0, :]
    sc = fssk_state_from_coef(y, coef=ker.coef(dt, trunc=trunc), axis=0)
    sf = fssk_state(X, kernel=ker, dt=dt, trunc=trunc)
    assert len(sc) == len(sf) == trunc
    for r in range(trunc):
        assert sc[r].shape == sf[r].shape


# -- correctness

def test_zero_increments_give_zero_state():
    ker = _make_kernel()
    s = fssk_state(jnp.zeros((9, 3)), kernel=ker, dt=0.1, trunc=3)
    for r, z in enumerate(s):
        assert jnp.allclose(z, 0.0), f"level {r} not zero"


@pytest.mark.parametrize("block_size", [1, 2, 3, 6])
def test_blocked_final_state_matches_full_run(block_size):
    ker = _make_kernel()
    X, trunc, dt = _linear_path(S=6), 3, 0.1
    n = 6 // block_size
    full = fssk_state(X, kernel=ker, dt=dt, trunc=trunc)
    bl = fssk_state(X, kernel=ker, dt=dt, trunc=trunc, block_size=block_size)
    for r in range(trunc):
        last = bl[r][n - 1] if n > 1 else bl[r]
        _allclose(full[r], last)


def test_continuation_matches_single_run():
    ker = _make_kernel()
    S1, S2, trunc, dt = 4, 4, 3, 0.1
    X = _linear_path(S=S1 + S2)
    full = fssk_state(X, kernel=ker, dt=dt, trunc=trunc)
    Z1 = fssk_state(X[:S1 + 1], kernel=ker, dt=dt, trunc=trunc)
    Z2 = fssk_state(X[S1:], kernel=ker, dt=dt, trunc=trunc, initial_state=Z1)
    for r in range(trunc):
        _allclose(full[r], Z2[r])


def test_continuation_with_blocking_matches_single_run():
    ker = _make_kernel()
    S1, S2, trunc, dt = 4, 4, 2, 0.1
    X = _linear_path(S=S1 + S2)
    full = fssk_state(X, kernel=ker, dt=dt, trunc=trunc)
    b1 = fssk_state(X[:S1 + 1], kernel=ker, dt=dt, trunc=trunc, block_size=2)
    Z1 = tuple(z[-1] for z in b1)
    Z2 = fssk_state(X[S1:], kernel=ker, dt=dt, trunc=trunc, initial_state=Z1)
    for r in range(trunc):
        _allclose(full[r], Z2[r])


def test_increment_input_matches_node_input():
    ker = _make_kernel()
    X, trunc = _linear_path(S=5), 3
    dt = jnp.full((5,), 0.1)
    a = fssk_state(X, kernel=ker, dt=dt, trunc=trunc)
    b = fssk_state(jnp.diff(X, axis=0), kernel=ker, dt=dt, trunc=trunc, increment_input=True)
    for r in range(trunc):
        _allclose(a[r], b[r])


def test_constant_and_perstep_dt_agree():
    ker = _make_kernel()
    S, trunc, v = 6, 2, 0.15
    X = _linear_path(S=S)
    a = fssk_state(X, kernel=ker, dt=v, trunc=trunc)
    b = fssk_state(X, kernel=ker, dt=jnp.array([v]), trunc=trunc)
    c = fssk_state(X, kernel=ker, dt=jnp.full((S,), v), trunc=trunc)
    for r in range(trunc):
        _allclose(a[r], b[r])
        _allclose(a[r], c[r])


def test_accumulate_false_each_block_starts_from_seed():
    ker = _make_kernel()
    trunc = 2
    na = fssk_state(_linear_path(S=6), kernel=ker, dt=0.1, trunc=trunc, block_size=3, accumulate=False)
    for r in range(trunc):
        _allclose(na[r][0], na[r][1])


def test_accumulate_false_first_block_matches_full_run_on_first_half():
    ker = _make_kernel()
    X, trunc, dt = _linear_path(S=6), 2, 0.1
    na = fssk_state(X, kernel=ker, dt=dt, trunc=trunc, block_size=3, accumulate=False)
    fh = fssk_state(X[:4], kernel=ker, dt=dt, trunc=trunc)
    for r in range(trunc):
        _allclose(na[r][0], fh[r])


# -- readout

def test_readout_returns_trunc_plus_one_levels():
    ker = _make_kernel()
    trunc = 3
    sig = fssk_readout(fssk_state(_linear_path(S=5), kernel=ker, dt=0.1, trunc=trunc), kernel=ker)
    assert len(sig) == trunc + 1
    for r, z in enumerate(sig):
        assert z.shape[-1] == ker.m ** r


def test_readout_zero_state_is_unit_signature():
    ker = _make_kernel()
    trunc = 3
    sig = fssk_readout(fssk_state(jnp.zeros((5, 3)), kernel=ker, dt=0.1, trunc=trunc), kernel=ker)
    _allclose(sig[0], jnp.ones((1,)))
    for r in range(1, trunc + 1):
        _allclose(sig[r], jnp.zeros((ker.m ** r,)))


def test_readout_is_jittable():
    ker = _make_kernel()
    s = fssk_state(_linear_path(S=4), kernel=ker, dt=0.1, trunc=2)
    out1 = fssk_readout(s, kernel=ker, tau_dt=0.2)
    out2 = jax.jit(fssk_readout)(s, kernel=ker, tau_dt=0.2)
    for r in range(3):
        _allclose(out1[r], out2[r])


# -- JIT

def test_fssk_state_is_jittable():
    ker = _make_kernel()
    X, trunc = _linear_path(S=4), 2
    kw = dict(kernel=ker, dt=0.1, trunc=trunc)
    out1 = fssk_state(X, **kw)
    out2 = jax.jit(
        fssk_state,
        static_argnames=("trunc", "axis", "block_size", "accumulate",
                         "output_starting_state", "increment_input"),
    )(X, **kw)
    for r in range(trunc):
        _allclose(out1[r], out2[r])


def test_fssk_state_from_coef_is_jittable():
    ker = _make_kernel()
    S, trunc = 4, 2
    dt = jnp.full((S,), 0.1)
    X = _linear_path(S)
    y = jnp.einsum("qmd,...d->...qm", ker.A.astype(jnp.float64), jnp.diff(X, axis=0))[..., 0, :]
    coef = ker.coef(dt, trunc=trunc)
    out1 = fssk_state_from_coef(y, coef=coef, axis=0)
    out2 = jax.jit(
        fssk_state_from_coef,
        static_argnames=("axis", "block_size", "accumulate", "output_starting_state"),
    )(y, coef=coef, axis=0)
    for r in range(trunc):
        _allclose(out1[r], out2[r])


# -- validation errors

def test_raises_for_non_positive_trunc():
    ker = _make_kernel()
    with pytest.raises(ValueError, match="trunc must be positive"):
        fssk_state(_linear_path(S=4), kernel=ker, dt=0.1, trunc=0)


def test_raises_for_1d_X():
    ker = _make_kernel()
    with pytest.raises(ValueError, match="at least a step axis"):
        fssk_state(jnp.ones((5,)), kernel=ker, dt=0.1, trunc=2)


def test_raises_for_axis_conflict_with_last_dim():
    ker = _make_kernel()
    with pytest.raises(ValueError, match="trailing path dimension"):
        fssk_state(_linear_path(S=4), kernel=ker, dt=0.1, trunc=2, axis=-1)


def test_raises_for_wrong_path_dim():
    ker = _make_kernel(d=3)
    with pytest.raises(ValueError, match="trailing dimension"):
        fssk_state(jnp.ones((6, 5)), kernel=ker, dt=0.1, trunc=2)


def test_raises_for_zero_increments():
    ker = _make_kernel()
    with pytest.raises(ValueError, match="at least one increment"):
        fssk_state(jnp.ones((1, 3)), kernel=ker, dt=0.1, trunc=2)


def test_raises_when_block_size_does_not_divide_S():
    ker = _make_kernel()
    with pytest.raises(ValueError, match="block_size must divide"):
        fssk_state(_linear_path(S=5), kernel=ker, dt=0.1, trunc=2, block_size=3)


def test_raises_for_initial_state_wrong_level_count():
    ker = _make_kernel()
    X, trunc = _linear_path(S=4), 2
    states = fssk_state(X, kernel=ker, dt=0.1, trunc=trunc)
    with pytest.raises(ValueError, match="levels"):
        fssk_state(X, kernel=ker, dt=0.1, trunc=trunc, initial_state=states[:-1])


def test_raises_for_invalid_1d_dt():
    ker = _make_kernel()
    with pytest.raises(ValueError, match="length 1 or S"):
        fssk_state(_linear_path(S=4), kernel=ker, dt=jnp.array([0.1, 0.2, 0.3]), trunc=2)

