from __future__ import annotations

import builtins
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
import tensordev._wordwise.dispatch as dispatch_module
import tensordev._wordwise.fssk as fssk_assembly
from tensordev._wordwise.dispatch import (
    fssk_q1_wordwise_candidate_eligible,
    fssk_q1_wordwise_device_eligible,
)
from tensordev.core.jax import JaxSequentialCore
from tensordev.sss import FSSK
from tensordev.sss import state_update as state_update_module


class _FakeDevice:
    platform = "gpu"
    device_kind = "NVIDIA test device"
    compute_capability = (8, 0)


class _FakeArray:
    dtype = np.dtype("float32")

    def devices(self):
        return {_FakeDevice()}


def _kernel() -> FSSK:
    return FSSK.from_matrix(
        Lambda=jnp.asarray([[0.6, 0.1], [-0.2, 0.4]]),
        A=jnp.asarray([[[1.0, 0.2], [-0.1, 0.8]]]),
        b=jnp.asarray([[0.7, -0.2]]),
    )


def _path() -> jax.Array:
    return jnp.asarray(
        [[0.0, 0.0], [0.1, 0.2], [0.3, 0.1], [-0.1, 0.25], [0.2, 0.2]],
        dtype=jnp.float64,
    )


def _assert_tree_close(actual, expected) -> None:
    if hasattr(actual, "spec") or hasattr(expected, "spec"):
        assert actual.spec == expected.spec
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert left.shape == right.shape
        np.testing.assert_allclose(
            np.asarray(left),
            np.asarray(right),
            atol=5e-11,
            rtol=5e-11,
        )


def test_cpu_fssk_fallback_never_imports_native_executor(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "tensordev._wordwise.fssk":
            raise AssertionError("CPU fallback imported the native FSSK executor")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = state_update_module.fssk_state(
        _path(), kernel=_kernel(), dt=0.1, trunc=2
    )
    assert len(result) == 2


def test_fssk_candidate_rejects_q2_and_every_traced_input():
    core = td.make_core(dims=2, max_trunc=2)
    seq_core = JaxSequentialCore()
    reference = _FakeArray()

    assert not fssk_q1_wordwise_candidate_eligible(
        core=core,
        seq_core=seq_core,
        q=2,
        reference=reference,
        differentiable_inputs=reference,
    )

    observed = []

    def trace(value):
        observed.append(
            fssk_q1_wordwise_candidate_eligible(
                core=core,
                seq_core=seq_core,
                q=1,
                reference=reference,
                differentiable_inputs=(reference, value),
            )
        )
        return value

    jax.make_jaxpr(trace)(jnp.ones((2,), dtype=jnp.float32))
    assert observed == [False]


def test_fssk_automatic_native_release_gate_defaults_closed():
    core = td.make_core(dims=2, max_trunc=2)
    seq_core = JaxSequentialCore()
    reference = _FakeArray()
    kwargs = dict(
        core=core,
        seq_core=seq_core,
        q=1,
        reference=reference,
        differentiable_inputs=reference,
    )

    assert fssk_q1_wordwise_candidate_eligible(**kwargs)
    assert not fssk_q1_wordwise_device_eligible(**kwargs)


def test_resource_fallback_reuses_the_single_preparation(monkeypatch):
    kernel = _kernel()
    path = _path()
    expected = state_update_module.fssk_state(
        path, kernel=kernel, dt=0.1, trunc=2
    )
    original_prepare = state_update_module._prepare_fssk_state_call
    preparations = []

    def observed_prepare(*args, **kwargs):
        preparations.append(None)
        return original_prepare(*args, **kwargs)

    state_update_module._prepare_fssk_state_for_wordwise.clear_cache()
    monkeypatch.setattr(
        state_update_module,
        "_fssk_q1_wordwise_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        state_update_module,
        "_prepare_fssk_state_call",
        observed_prepare,
    )
    monkeypatch.setattr(
        state_update_module,
        "_try_fssk_q1_wordwise",
        lambda call: None,
    )

    actual = state_update_module.fssk_state(
        path, kernel=kernel, dt=0.1, trunc=2
    )

    assert len(preparations) == 1
    _assert_tree_close(actual, expected)


def test_unexpected_native_errors_propagate(monkeypatch):
    monkeypatch.setattr(
        state_update_module,
        "_fssk_q1_wordwise_eligible",
        lambda **kwargs: True,
    )

    def fail(call):
        raise RuntimeError("native execution failed")

    monkeypatch.setattr(state_update_module, "_try_fssk_q1_wordwise", fail)
    with pytest.raises(RuntimeError, match="native execution failed"):
        state_update_module.fssk_state(
            _path(), kernel=_kernel(), dt=0.1, trunc=2
        )


def test_path_gradient_stays_on_the_portable_route(monkeypatch):
    observed_tracers = []

    def reject_transformed(**kwargs):
        leaves = jax.tree.leaves(kwargs["differentiable_inputs"])
        observed_tracers.append(
            any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)
        )
        return False

    monkeypatch.setattr(
        state_update_module,
        "_fssk_q1_wordwise_eligible",
        reject_transformed,
    )
    monkeypatch.setattr(
        state_update_module,
        "_try_fssk_q1_wordwise",
        lambda call: (_ for _ in ()).throw(
            AssertionError("gradient entered the native executor")
        ),
    )

    def objective(path):
        signature = state_update_module.fssk_vsig(
            path, kernel=_kernel(), dt=0.1, trunc=2
        )
        return sum(jnp.sum(block**2) for block in signature)

    gradient = jax.grad(objective)(_path())
    assert gradient.shape == _path().shape
    assert observed_tracers and all(observed_tracers)


def test_captured_concrete_inputs_stay_portable_inside_outer_trace(monkeypatch):
    path = _path()
    kernel = _kernel()
    monkeypatch.setattr(
        state_update_module,
        "_fssk_q1_wordwise_eligible",
        lambda **kwargs: True,
    )

    def unexpected_adapter(call):
        raise AssertionError("a traced prepared call built a native adapter")

    monkeypatch.setattr(
        state_update_module,
        "_fssk_q1_wordwise_adapter",
        unexpected_adapter,
    )

    def traced(dummy):
        state = state_update_module.fssk_state(
            path,
            kernel=kernel,
            dt=0.1,
            trunc=2,
        )
        signature = state_update_module.fssk_vsig(
            path,
            kernel=kernel,
            dt=0.1,
            trunc=2,
        )
        return (
            sum(jnp.sum(block) for block in state)
            + sum(jnp.sum(block) for block in signature)
            + dummy
        )

    jax.make_jaxpr(traced)(jnp.asarray(0.0))


def test_forced_interpreter_public_path_matches_portable(monkeypatch):
    core = td.make_core(
        dims=(1, 1),
        max_trunc=(1, 1),
        partially_symmetrized=True,
        coordinates="shear",
    )
    path = jnp.stack((_path(), 0.5 * _path()), axis=0)
    kwargs = dict(
        kernel=_kernel(),
        dt=0.1,
        trunc=(1, 1),
        axis=1,
        block_size=2,
        accumulate=False,
        output_starting_state=True,
        core=core,
    )
    expected = state_update_module.fssk_state(path, **kwargs)

    monkeypatch.setattr(
        state_update_module,
        "_fssk_q1_wordwise_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        fssk_assembly,
        "try_fssk_q1_wordwise",
        lambda call: fssk_assembly.run_fssk_q1_wordwise(
            call,
            interpret=True,
            tile_words=4,
            tile_prime_words=2,
        ),
    )
    actual = state_update_module.fssk_state(path, **kwargs)
    _assert_tree_close(actual, expected)


def test_uniform_coefficients_and_zero_seed_stay_batch_one_for_native():
    core = td.make_core(dims=2, max_trunc=2)
    path = jnp.broadcast_to(_path()[None], (17,) + _path().shape)
    call = state_update_module._prepare_fssk_state_call(
        path,
        kernel=_kernel(),
        dt=0.1,
        trunc=2,
        maximum_order=2,
        axis=1,
        block_size=None,
        accumulate=True,
        initial_state=None,
        output_starting_state=False,
        increment_input=False,
        core=core,
        seq_core=JaxSequentialCore(),
    )
    y_time, coef, seed = (
        state_update_module._canonicalize_fssk_q1_wordwise_inputs(call)
    )

    assert y_time.shape[:2] == (4, 17)
    assert coef.E.shape[:2] == (1, 1)
    assert coef.psi.shape[:2] == (1, 1)
    assert coef.phi.shape[:2] == (1, 1)
    assert all(block.shape[0] == 1 for block in seed)
    assert call.block_size is None


def test_colocation_device_puts_whole_pytrees_directly(monkeypatch):
    reference = object()
    device = object()
    values = (object(), object(), object())
    observed = []

    monkeypatch.setattr(
        dispatch_module,
        "_concrete_single_device",
        lambda value: device if value is reference else None,
    )

    def device_put(value, target):
        observed.append((value, target))
        return value

    monkeypatch.setattr(state_update_module.jax, "device_put", device_put)
    actual = state_update_module._colocate_fssk_inputs(reference, *values)

    assert actual == values
    assert observed == [(values, device)]


def test_portable_public_fssk_colocates_accelerator_inputs(monkeypatch):
    kernel = _kernel()
    path = _path()
    increments = jnp.diff(path, axis=0)
    projected = jnp.einsum("qmd,...d->...qm", kernel.A, increments)
    coefficients = kernel.coef(0.1, trunc=2)
    observed = []

    monkeypatch.setattr(
        state_update_module,
        "_fssk_q1_wordwise_eligible",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        state_update_module,
        "_fssk_accelerator_colocation_eligible",
        lambda **kwargs: True,
    )

    def colocate(reference, *values):
        observed.append((reference, len(values)))
        return values

    monkeypatch.setattr(state_update_module, "_colocate_fssk_inputs", colocate)

    state_update_module.fssk_state(path, kernel=kernel, dt=0.1, trunc=2)
    state_update_module.fssk_state_from_coef(
        projected,
        coef=coefficients,
        trunc=2,
    )
    state_update_module.fssk_vsig(path, kernel=kernel, dt=0.1, trunc=2)

    assert observed == [(path, 3), (projected, 2), (path, 4)]


def test_numpy_inputs_do_not_enter_accelerator_colocation(monkeypatch):
    kernel = _kernel()
    path = np.asarray(_path())
    increments = np.diff(path, axis=0)
    projected = np.einsum("qmd,...d->...qm", np.asarray(kernel.A), increments)
    coefficients = kernel.coef(0.1, trunc=2)

    def unexpected_colocation(*args, **kwargs):
        raise AssertionError("NumPy input entered accelerator colocation")

    monkeypatch.setattr(
        state_update_module,
        "_colocate_fssk_inputs",
        unexpected_colocation,
    )

    state = state_update_module.fssk_state(
        path, kernel=kernel, dt=0.1, trunc=2
    )
    state_update_module.fssk_state_from_coef(
        projected,
        coef=coefficients,
        trunc=2,
    )
    state_update_module.fssk_vsig(path, kernel=kernel, dt=0.1, trunc=2)
    state_update_module.fssk_readout(
        tuple(np.asarray(block) for block in state),
        kernel=kernel,
        tau_dt=np.asarray(0.2),
    )


def test_direct_readout_uses_the_gpu_colocation_seam(monkeypatch):
    kernel = _kernel()
    state = state_update_module.fssk_state(
        _path(), kernel=kernel, dt=0.1, trunc=2
    )
    observed = []
    monkeypatch.setattr(
        state_update_module,
        "_fssk_accelerator_colocation_eligible",
        lambda **kwargs: True,
    )

    def colocate(reference, *values):
        observed.append((reference, values))
        return values

    monkeypatch.setattr(state_update_module, "_colocate_fssk_inputs", colocate)
    result = state_update_module.fssk_readout(
        state,
        kernel=kernel,
        tau_dt=jnp.asarray(0.2),
    )

    assert len(result) == 3
    assert len(observed) == 1
    assert observed[0][1][0] is state


def test_terminal_readout_weights_keep_only_one_or_full_flat_batch():
    core = td.make_core(dims=2, max_trunc=2)
    path = jnp.broadcast_to(_path(), (2, 3) + _path().shape)
    call = state_update_module._prepare_fssk_state_call(
        path,
        kernel=_kernel(),
        dt=0.1,
        trunc=2,
        maximum_order=2,
        axis=2,
        block_size=None,
        accumulate=True,
        initial_state=None,
        output_starting_state=False,
        increment_input=False,
        core=core,
        seq_core=JaxSequentialCore(),
    )

    singleton = state_update_module._prepare_fssk_q1_wordwise_readout_weights(
        call,
        _kernel(),
        jnp.asarray(0.2),
    )
    batched = state_update_module._prepare_fssk_q1_wordwise_readout_weights(
        call,
        _kernel(),
        jnp.asarray([0.1, 0.2, 0.3]),
    )
    extra_output_axis = (
        state_update_module._prepare_fssk_q1_wordwise_readout_weights(
            call,
            _kernel(),
            jnp.ones((4, 1, 1)),
        )
    )

    assert singleton.shape == (1, 2)
    assert batched.shape == (6, 2)
    assert extra_output_axis is None


@pytest.mark.parametrize(
    "core,trunc,tau_dt",
    [
        pytest.param(td.Jax(), 2, 0.2, id="total-standard-singleton"),
        pytest.param(
            td.make_core(
                dims=(1, 1),
                max_trunc=(1, 1),
                partially_symmetrized=True,
                coordinates="shear",
            ),
            (1, 1),
            jnp.asarray([0.1, 0.2]),
            id="partial-shear-batched",
        ),
    ],
)
def test_forced_interpreter_fused_vsig_matches_portable(
    monkeypatch,
    core,
    trunc,
    tau_dt,
):
    path = jnp.stack((_path(), 0.5 * _path()), axis=0)
    kwargs = dict(
        kernel=_kernel(),
        dt=0.1,
        trunc=trunc,
        axis=1,
        tau_dt=tau_dt,
        core=core,
    )
    expected = state_update_module.fssk_vsig(path, **kwargs)

    monkeypatch.setattr(
        state_update_module,
        "_fssk_q1_wordwise_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        fssk_assembly,
        "try_fssk_q1_wordwise_readout",
        lambda call, weights: fssk_assembly.run_fssk_q1_wordwise_readout(
            call,
            weights,
            interpret=True,
            tile_words=4,
            tile_prime_words=2,
        ),
    )
    actual = state_update_module.fssk_vsig(path, **kwargs)

    _assert_tree_close(actual, expected)
