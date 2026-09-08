from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

import tensordev as td
from tensordev.core.bigraded.types import BigradedTensor
from tensordev.sss import FSSK, StateSpaceSignature
from tensordev.sss import state_update as state_update_module
from tensordev.sss.recursion_scalar import eval_fg
from tensordev.sss.state_update import (
    fssk_readout,
    fssk_state,
    fssk_state_from_coef,
    fssk_vsig,
)


def _identity_kernel(dim: int = 2) -> FSSK:
    return FSSK.from_matrix(
        Lambda=jnp.zeros((1, 1)),
        A=jnp.eye(dim)[None, :, :],
        b=jnp.ones((1, 1)),
    )


def _nontrivial_kernel() -> FSSK:
    return FSSK.from_matrix(
        Lambda=jnp.asarray(
            [[0.7, 0.2], [-0.1, 0.4]], dtype=jnp.float64
        ),
        A=jnp.asarray(
            [[[1.0, 0.3], [-0.2, 0.8]]], dtype=jnp.float64
        ),
        b=jnp.asarray([[0.7, -0.2]], dtype=jnp.float64),
    )


def _path() -> jax.Array:
    return jnp.asarray(
        [[0.0, 0.0], [0.1, 0.2], [0.3, 0.1], [-0.1, 0.25]],
        dtype=jnp.float64,
    )


def _core_cases():
    return (
        (td.make_core(dims=2, max_trunc=3), 3),
        (
            td.make_core(
                dims=(1, 1),
                max_trunc=3,
                coordinates="shear",
            ),
            3,
        ),
        (td.make_core(dims=(1, 1), max_trunc=(2, 1)), (2, 1)),
        (
            td.make_core(
                dims=(1, 1),
                max_trunc=(2, 1),
                coordinates="shear",
            ),
            (2, 1),
        ),
        (
            td.make_core(
                dims=(1, 1),
                max_trunc=(2, 1),
                partially_symmetrized=True,
            ),
            (2, 1),
        ),
        (
            td.make_core(
                dims=(1, 1),
                max_trunc=(2, 1),
                partially_symmetrized=True,
                coordinates="shear",
            ),
            (2, 1),
        ),
    )


def _assert_tensor_close(left, right, *, atol: float = 1e-10) -> None:
    if isinstance(left, BigradedTensor):
        assert isinstance(right, BigradedTensor)
        assert left.spec == right.spec
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for a, b in zip(left_leaves, right_leaves):
        assert a.shape == b.shape
        assert jnp.allclose(a, b, atol=atol, rtol=atol)


@pytest.mark.parametrize("core,trunc", _core_cases())
def test_scalar_identity_fssk_recovers_ordinary_signature(core, trunc):
    X = _path()
    kernel = _identity_kernel()

    actual = fssk_vsig(X, kernel=kernel, dt=0.1, trunc=trunc, core=core)
    expected = td.path_signature(X, trunc=trunc, core=core)

    _assert_tensor_close(actual, expected)


@pytest.mark.parametrize("core,trunc", _core_cases())
def test_state_and_readout_use_the_selected_core(core, trunc):
    kernel = _identity_kernel()
    state = fssk_state(_path(), kernel=kernel, dt=0.1, trunc=trunc, core=core)

    if core.grading == "bidegree":
        assert isinstance(state, BigradedTensor)
        assert state.spec.truncation == trunc
        assert state.spec.include_scalar is False
        assert state.spec.coordinates == core.coordinates
        assert state.spec.partially_symmetrized == core.partially_symmetrized
    else:
        assert isinstance(state, tuple)
        assert len(state) == trunc

    direct = fssk_readout(state, kernel=kernel, core=core)
    fused = fssk_vsig(_path(), kernel=kernel, dt=0.1, trunc=trunc, core=core)
    _assert_tensor_close(direct, fused)


def test_partially_symmetrized_shear_state_is_boundary_transform():
    standard = td.make_core(
        dims=(1, 1),
        max_trunc=(2, 1),
        partially_symmetrized=True,
    )
    shear = td.make_core(
        dims=(1, 1),
        max_trunc=(2, 1),
        partially_symmetrized=True,
        coordinates="shear",
    )
    kernel = _identity_kernel()

    standard_state = fssk_state(
        _path(), kernel=kernel, dt=0.1, trunc=(2, 1), core=standard
    )
    shear_state = fssk_state(
        _path(), kernel=kernel, dt=0.1, trunc=(2, 1), core=shear
    )
    expected = shear.tensor_from_standard_coordinates(
        standard_state,
        trunc=(2, 1),
        first_on=True,
    )
    _assert_tensor_close(shear_state, expected)


def test_nontrivial_scalar_state_commutes_with_every_layout_and_coordinates():
    kernel = _nontrivial_kernel()
    X = _path()
    dt = jnp.asarray([0.1, 0.13, 0.08], dtype=X.dtype)
    total = td.make_core(dims=2, max_trunc=3)
    ordered = td.make_core(dims=(1, 1), max_trunc=(2, 1))
    partial = td.make_core(
        dims=(1, 1),
        max_trunc=(2, 1),
        partially_symmetrized=True,
    )

    total_state = fssk_state(
        X, kernel=kernel, dt=dt, trunc=3, core=total
    )
    total_shear = td.make_core(
        dims=(1, 1), max_trunc=3, coordinates="shear"
    )
    total_shear_expected = total_shear.tensor_from_standard_coordinates(
        total_state, trunc=3, first_on=True
    )
    total_shear_actual = fssk_state(
        X, kernel=kernel, dt=dt, trunc=3, core=total_shear
    )
    _assert_tensor_close(total_shear_actual, total_shear_expected)

    scalar_placeholder = jnp.zeros(
        total_state[0].shape[:-1] + (1,), dtype=total_state[0].dtype
    )
    ordered_expected = ordered.tensor_from_total(
        (scalar_placeholder,) + total_state,
        trunc=(2, 1),
        include_scalar=False,
    )
    partial_expected = partial.tensor_partially_symmetrize(
        ordered_expected,
        trunc=(2, 1),
        first_on=True,
    )

    cases = (
        (ordered, ordered_expected),
        (partial, partial_expected),
        (
            td.make_core(
                dims=(1, 1),
                max_trunc=(2, 1),
                coordinates="shear",
            ),
            ordered_expected,
        ),
        (
            td.make_core(
                dims=(1, 1),
                max_trunc=(2, 1),
                partially_symmetrized=True,
                coordinates="shear",
            ),
            partial_expected,
        ),
    )
    for core, standard_expected in cases:
        expected = (
            standard_expected
            if core.coordinates == "standard"
            else core.tensor_from_standard_coordinates(
                standard_expected,
                trunc=(2, 1),
                first_on=True,
            )
        )
        actual = fssk_state(
            X, kernel=kernel, dt=dt, trunc=(2, 1), core=core
        )
        _assert_tensor_close(actual, expected)


def test_bidegree_block_axis_and_starting_state_follow_path_axis():
    core = td.make_core(
        dims=(1, 1),
        max_trunc=(1, 2),
        partially_symmetrized=True,
        coordinates="shear",
    )
    kernel = _identity_kernel()
    X = jnp.stack((_path(), 0.5 * _path()), axis=0)

    state = fssk_state(
        X,
        kernel=kernel,
        dt=0.1,
        trunc=(1, 2),
        axis=1,
        block_size=1,
        output_starting_state=True,
        core=core,
    )
    assert state.batch_shape[:2] == (2, 4)

    actual = fssk_readout(state, kernel=kernel, core=core)
    expected = td.path_signature(
        X,
        trunc=(1, 2),
        axis=1,
        block_size=1,
        output_starting_point=True,
        core=core,
    )
    _assert_tensor_close(actual, expected)


@pytest.mark.parametrize("trunc", ((2, 0), (0, 2)))
def test_one_sided_bidegree_truncation(trunc):
    core = td.make_core(dims=(1, 1), max_trunc=(2, 2))
    kernel = _identity_kernel()
    actual = fssk_vsig(_path(), kernel=kernel, dt=0.1, trunc=trunc, core=core)
    expected = td.path_signature(_path(), trunc=trunc, core=core)
    _assert_tensor_close(actual, expected)


def test_coefficient_entry_point_uses_independent_bidegree_truncation():
    core = td.make_core(dims=(1, 1), max_trunc=(2, 2))
    kernel = _identity_kernel()
    X = _path()
    increments = jnp.diff(X, axis=0)
    y = jnp.einsum("qmd,...d->...qm", kernel.A, increments)[..., 0, :]
    coef = kernel.coef(0.1, trunc=4, dtype=X.dtype)

    from_coef = fssk_state_from_coef(
        y,
        coef=coef,
        trunc=(2, 1),
        core=core,
    )
    from_kernel = fssk_state(
        X,
        kernel=kernel,
        dt=0.1,
        trunc=(2, 1),
        core=core,
    )
    _assert_tensor_close(from_coef, from_kernel)


def test_state_space_signature_binds_core_and_continues_state():
    core = td.make_core(
        dims=(1, 1),
        max_trunc=(2, 1),
        partially_symmetrized=True,
        coordinates="shear",
    )
    kernel = _identity_kernel()
    X = _path()
    model = StateSpaceSignature(kernel, trunc=(2, 1), core=core)

    assert model.core is core
    assert isinstance(model.state, BigradedTensor)
    assert model.state.spec.include_scalar is False

    updated = model.update_with_path(X[:3], dt=0.1).update_with_path(
        X[2:], dt=0.1
    )
    expected = model.vsig(X, dt=0.1)
    _assert_tensor_close(updated.readout(), expected)

    jitted = jax.jit(lambda bound, path: bound.vsig(path, dt=0.1))(model, X)
    _assert_tensor_close(jitted, expected)


def test_partially_symmetrized_shear_path_gradient_matches_signature():
    core = td.make_core(
        dims=(1, 1),
        max_trunc=(1, 1),
        partially_symmetrized=True,
        coordinates="shear",
    )
    kernel = _identity_kernel()

    def fssk_objective(path):
        result = fssk_vsig(
            path, kernel=kernel, dt=0.1, trunc=(1, 1), core=core
        )
        return sum(jnp.sum(block**2) for block in result.blocks)

    def signature_objective(path):
        result = td.path_signature(path, trunc=(1, 1), core=core)
        return sum(jnp.sum(block**2) for block in result.blocks)

    actual = jax.grad(fssk_objective)(_path())
    expected = jax.grad(signature_objective)(_path())
    assert jnp.allclose(actual, expected, atol=1e-9, rtol=1e-9)


def test_direct_tuple_readout_keeps_legacy_total_degree_rule():
    kernel = _identity_kernel()
    X = _path()
    state = fssk_state(X, kernel=kernel, dt=0.1, trunc=2, core=td.Jax())
    old_core, old_seq_core = td.get_default_core_pair()
    try:
        td.set_default_core(
            td.make_core(dims=(1, 1), max_trunc=(1, 1)),
            old_seq_core,
        )
        actual = fssk_readout(state, kernel=kernel)
    finally:
        td.set_default_core(old_core, old_seq_core)

    expected = fssk_readout(state, kernel=kernel, core=td.Jax())
    _assert_tensor_close(actual, expected)


@pytest.mark.parametrize(
    "core,trunc",
    (
        (td.make_core(dims=2, max_trunc=2), 2),
        (
            td.make_core(
                dims=(1, 1),
                max_trunc=(1, 1),
                partially_symmetrized=True,
                coordinates="shear",
            ),
            (1, 1),
        ),
    ),
)
def test_readout_lag_broadcasts_over_state_batch(core, trunc):
    kernel = _identity_kernel()
    X = jnp.stack((_path(), 0.5 * _path()), axis=0)
    state = fssk_state(
        X,
        kernel=kernel,
        dt=0.1,
        trunc=trunc,
        axis=1,
        core=core,
    )
    lags = jnp.asarray([[0.0], [0.1], [0.2]], dtype=X.dtype)

    actual = fssk_readout(state, kernel=kernel, tau_dt=lags, core=core)
    expected = jax.tree.map(
        lambda *blocks: jnp.stack(blocks, axis=0),
        *(fssk_readout(state, kernel=kernel, tau_dt=lag, core=core)
          for lag in lags[:, 0]),
    )

    _assert_tensor_close(actual, expected)


def test_scalar_fssk_uses_the_background_core_and_default_truncation():
    core = td.make_core(
        dims=(1, 1),
        max_trunc=(2, 1),
        default_trunc=(1, 1),
        partially_symmetrized=True,
        coordinates="shear",
    )
    kernel = _identity_kernel()
    old_core, old_seq_core = td.get_default_core_pair()
    try:
        td.set_default_core(core, old_seq_core)
        actual = fssk_vsig(_path(), kernel=kernel, dt=0.1)
        bound = StateSpaceSignature(kernel, trunc=None)
    finally:
        td.set_default_core(old_core, old_seq_core)

    expected = fssk_vsig(
        _path(), kernel=kernel, dt=0.1, trunc=(1, 1), core=core
    )
    assert isinstance(actual, BigradedTensor)
    assert bound.core is core
    assert bound.trunc == (1, 1)
    _assert_tensor_close(actual, expected)


def test_q_greater_than_one_remains_standard_total_degree_only():
    kernel = FSSK.from_matrix(
        Lambda=jnp.zeros((1, 1)),
        A=jnp.ones((2, 2, 2)),
        b=jnp.ones((2, 1)),
    )
    X = _path()

    state = fssk_state(X, kernel=kernel, dt=0.1, trunc=2, core=td.Jax())
    assert isinstance(state, tuple)
    assert len(state) == 2

    core = td.make_core(dims=(1, 1), max_trunc=(1, 1))
    with pytest.raises(RuntimeError, match="only for scalar FSSK"):
        fssk_state(X, kernel=kernel, dt=0.1, trunc=(1, 1), core=core)


def test_q_greater_than_one_rejects_total_core_alphabet_mismatch():
    kernel = FSSK.from_matrix(
        Lambda=jnp.zeros((1, 1)),
        A=jnp.ones((2, 2, 2)),
        b=jnp.ones((2, 1)),
    )
    X = _path()
    wrong_core = td.make_core(dims=3, max_trunc=2)
    right_core = td.make_core(dims=2, max_trunc=2)
    coef = kernel.coef(0.1, trunc=2, dtype=X.dtype)
    y = jnp.einsum("qmd,...d->...qm", kernel.A, jnp.diff(X, axis=0))
    state = fssk_state(X, kernel=kernel, dt=0.1, trunc=2, core=right_core)

    calls = (
        lambda: fssk_state(
            X, kernel=kernel, dt=0.1, trunc=2, core=wrong_core
        ),
        lambda: fssk_state_from_coef(
            y, coef=coef, trunc=2, core=wrong_core
        ),
        lambda: fssk_readout(state, kernel=kernel, core=wrong_core),
        lambda: StateSpaceSignature(kernel, trunc=2, core=wrong_core),
    )
    for call in calls:
        with pytest.raises(ValueError, match="core alphabet dimension"):
            call()


def test_total_degree_g_horner_keeps_legacy_degree_bound():
    kernel = _nontrivial_kernel()
    coef = kernel.coef(0.1, trunc=4, dtype=jnp.float64)
    f, G = eval_fg(
        jnp.asarray([0.2, -0.1], dtype=jnp.float64),
        coef,
        core=td.Jax(),
    )

    assert len(f) == 4
    assert len(G) == 3
    assert tuple(block.shape[-1] for block in G) == (1, 2, 4)


def test_bidegree_g_horner_retains_the_active_rectangle():
    core = td.make_core(dims=(1, 1), max_trunc=(2, 1))
    kernel = _nontrivial_kernel()
    coef = kernel.coef(0.1, trunc=3, dtype=jnp.float64)
    _, G = eval_fg(
        jnp.asarray([0.2, -0.1], dtype=jnp.float64),
        coef,
        core=core,
        trunc=(2, 1),
    )

    assert isinstance(G, BigradedTensor)
    assert G.spec.truncation == (2, 1)
    assert G.grades == core.resolve_layout((2, 1), include_scalar=True).grades


def test_direct_fssk_vsig_uses_one_outer_compilation():
    state_update_module._fssk_vsig_impl.clear_cache()
    state_update_module._fssk_state_impl.clear_cache()
    state_update_module._fssk_readout_impl.clear_cache()

    result = fssk_vsig(
        _path(), kernel=_nontrivial_kernel(), dt=0.1, trunc=2
    )
    jax.block_until_ready(result)

    assert state_update_module._fssk_vsig_impl._cache_size() == 1
    assert state_update_module._fssk_state_impl._cache_size() == 0
    assert state_update_module._fssk_readout_impl._cache_size() == 0


def test_core_dimension_and_zero_bound_state_are_rejected():
    kernel = _identity_kernel()
    wrong_dimension = td.make_core(dims=(1, 2), max_trunc=(1, 1))
    with pytest.raises(ValueError, match="core alphabet dimension"):
        fssk_state(
            _path(),
            kernel=kernel,
            dt=0.1,
            trunc=(1, 1),
            core=wrong_dimension,
        )

    zero_core = td.make_core(dims=(1, 1), max_trunc=(0, 0))
    with pytest.raises(ValueError, match="trunc must be positive"):
        StateSpaceSignature(kernel, trunc=(0, 0), core=zero_core)
