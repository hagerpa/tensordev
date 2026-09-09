from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
import tensordev._wordwise.dispatch as dispatch
import tensordev._wordwise.fssk as fssk_assembly
from tensordev._wordwise.pallas_ordinary import PallasOrdinaryResourceError
from tensordev.sss import FSSK, StateSpaceSignature
from tensordev.sss import state as state_module
from tensordev.sss import state_update


@pytest.fixture(autouse=True)
def clear_compilation_caches():
    yield
    jax.clear_caches()


def _kernel(q=1):
    return FSSK.from_matrix(
        Lambda=jnp.asarray([[0.4]]),
        A=jnp.broadcast_to(jnp.eye(2), (q, 2, 2)),
        b=jnp.ones((q, 1)),
    )


def _path():
    return jnp.asarray([[0.0, 0.0], [0.1, 0.2], [0.3, -0.1]])


def _call(api, *, execution, core=None, kernel=None, path=None, **kwargs):
    kernel = _kernel() if kernel is None else kernel
    path = _path() if path is None else path
    core = td.make_core(dims=2, max_trunc=2) if core is None else core
    if api == "from_coef":
        return state_update.fssk_state_from_coef(
            jnp.einsum("...d,pdm->...pm", jnp.diff(path, axis=0), kernel.A),
            coef=kernel.coef(0.1, trunc=2),
            core=core,
            execution=execution,
            **kwargs,
        )
    function = (
        state_update.fssk_state if api == "state" else state_update.fssk_vsig
    )
    return function(
        path, kernel=kernel, dt=0.1, core=core, execution=execution, **kwargs
    )


def _assert_tree_close(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(left, right, rtol=5e-11, atol=5e-11)


@pytest.mark.parametrize("api", ["state", "from_coef", "vsig"])
@pytest.mark.parametrize("execution", ["native", None, 1])
def test_invalid_execution(api, execution):
    with pytest.raises(ValueError, match="execution"):
        _call(api, execution=execution)


@pytest.mark.parametrize("api", ["state", "from_coef", "vsig"])
def test_explicit_wordwise_rejects_cpu(api):
    path = jax.device_put(_path(), jax.devices("cpu")[0])
    with pytest.raises(ValueError, match="NVIDIA|CUDA"):
        _call(api, execution="wordwise", path=path)


@pytest.mark.parametrize("api", ["state", "from_coef", "vsig"])
def test_explicit_wordwise_rejects_q_greater_than_one(api):
    with pytest.raises(ValueError, match="q.*1"):
        _call(api, execution="wordwise", kernel=_kernel(q=2))


@pytest.mark.parametrize("api", ["state", "from_coef", "vsig"])
def test_explicit_wordwise_rejects_unsupported_dtype(api):
    kernel = FSSK.from_matrix(
        Lambda=jnp.asarray([[0.4]], dtype=jnp.float32),
        A=jnp.eye(2, dtype=jnp.float16)[None],
        b=jnp.ones((1, 1), dtype=jnp.float32),
    )
    with pytest.raises(ValueError, match="float32.*float64"):
        _call(api, execution="wordwise", path=_path().astype(jnp.float16), kernel=kernel)


@pytest.mark.parametrize("api", ["state", "from_coef", "vsig"])
@pytest.mark.parametrize("transform", ["jit", "grad", "vmap"])
def test_explicit_wordwise_rejects_jax_transformations(api, transform):
    kernel = _kernel()
    core = td.make_core(dims=2, max_trunc=2)

    def evaluate(path):
        output = _call(
            api, execution="wordwise", path=path, kernel=kernel, core=core
        )
        return sum(jnp.sum(leaf) for leaf in jax.tree.leaves(output))

    path = _path()
    transformed = getattr(jax, transform)(evaluate)
    if transform == "vmap":
        path = path[None]
    with pytest.raises(ValueError, match="eager|trac|transform"):
        transformed(path)


@pytest.mark.parametrize("api", ["state", "from_coef", "vsig"])
def test_explicit_wordwise_rejects_captured_inputs_under_jit(api, monkeypatch):
    kernel = _kernel()
    path = _path()
    core = td.make_core(dims=2, max_trunc=2)
    monkeypatch.setattr(dispatch, "_supported_cuda_device", lambda device: True)

    def unexpected_adapter(*args, **kwargs):
        raise AssertionError("a traced call created a native adapter")

    monkeypatch.setattr(state_update, "_fssk_q1_wordwise_adapter", unexpected_adapter)

    def evaluate(dummy):
        output = _call(
            api, execution="wordwise", path=path, kernel=kernel, core=core
        )
        return dummy + sum(jnp.sum(leaf) for leaf in jax.tree.leaves(output))

    with pytest.raises(ValueError, match="eager|trac|transform"):
        jax.make_jaxpr(evaluate)(jnp.asarray(0.0))


@pytest.mark.parametrize("api", ["state", "from_coef", "vsig"])
def test_jax_execution_bypasses_wordwise_even_when_auto_enabled(api, monkeypatch):
    expected = _call(api, execution="auto")
    monkeypatch.setattr(dispatch, "_automatic_wordwise_release_eligible", lambda: True)
    monkeypatch.setattr(dispatch, "_supported_cuda_device", lambda device: True)

    def fail(*args, **kwargs):
        raise AssertionError("forced JAX execution entered the wordwise runner")

    monkeypatch.setattr(state_update, "_dispatch_fssk_q1_wordwise", fail)
    actual = _call(api, execution="jax")
    _assert_tree_close(actual, expected)


@pytest.mark.parametrize("api", ["state", "from_coef", "vsig"])
def test_explicit_wordwise_propagates_resource_failure(api, monkeypatch):
    monkeypatch.setattr(dispatch, "_supported_cuda_device", lambda device: True)

    def unavailable(*args, **kwargs):
        raise PallasOrdinaryResourceError("local state exceeds the configured limit")

    def portable(*args, **kwargs):
        raise AssertionError("explicit wordwise execution silently fell back")

    monkeypatch.setattr(fssk_assembly, "run_fssk_q1_wordwise", unavailable)
    monkeypatch.setattr(fssk_assembly, "run_fssk_q1_wordwise_readout", unavailable)
    monkeypatch.setattr(state_update, "_execute_portable_prepared_fssk_state", portable)
    monkeypatch.setattr(state_update, "_execute_portable_prepared_fssk_vsig", portable)
    with pytest.raises(PallasOrdinaryResourceError, match="local state"):
        _call(api, execution="wordwise")


@pytest.mark.parametrize("api", ["state", "from_coef", "vsig"])
def test_explicit_wordwise_cannot_silently_fall_back(api, monkeypatch):
    monkeypatch.setattr(dispatch, "_supported_cuda_device", lambda device: True)
    monkeypatch.setattr(fssk_assembly, "run_fssk_q1_wordwise", lambda *args: None)
    monkeypatch.setattr(
        fssk_assembly, "run_fssk_q1_wordwise_readout", lambda *args: None
    )
    with pytest.raises(RuntimeError, match="wordwise.*did not produce"):
        _call(api, execution="wordwise")


@pytest.mark.parametrize(
    "core_kwargs",
    [
        pytest.param(dict(dims=2, max_trunc=2), id="total-standard"),
        pytest.param(
            dict(dims=(1, 1), max_trunc=2, coordinates="shear"),
            id="total-shear",
        ),
        pytest.param(dict(dims=(1, 1), max_trunc=(1, 1)), id="bidegree-standard"),
        pytest.param(
            dict(dims=(1, 1), max_trunc=(1, 1), coordinates="shear"),
            id="bidegree-shear",
        ),
        pytest.param(
            dict(dims=(1, 1), max_trunc=(1, 1), partially_symmetrized=True),
            id="partial-standard",
        ),
        pytest.param(
            dict(
                dims=(1, 1),
                max_trunc=(1, 1),
                partially_symmetrized=True,
                coordinates="shear",
            ),
            id="partial-shear",
        ),
    ],
)
def test_explicit_wordwise_matches_jax_in_interpreter(core_kwargs, monkeypatch):
    core = td.make_core(**core_kwargs)
    monkeypatch.setattr(dispatch, "_supported_cuda_device", lambda device: True)
    original_state = fssk_assembly.run_fssk_q1_wordwise
    original_readout = fssk_assembly.run_fssk_q1_wordwise_readout
    routes = []

    def state(call):
        routes.append("state")
        return original_state(call, interpret=True, tile_words=4, tile_prime_words=4)

    def readout(call, weights):
        routes.append("readout")
        return original_readout(
            call, weights, interpret=True, tile_words=4, tile_prime_words=4
        )

    monkeypatch.setattr(fssk_assembly, "run_fssk_q1_wordwise", state)
    monkeypatch.setattr(fssk_assembly, "run_fssk_q1_wordwise_readout", readout)
    for api in ("state", "from_coef", "vsig"):
        expected = _call(api, execution="jax", core=core)
        actual = _call(api, execution="wordwise", core=core)
        _assert_tree_close(actual, expected)
    for options in (
        dict(block_size=1, output_starting_state=True),
        dict(tau_dt=jnp.asarray([0.0, 0.1])),
    ):
        expected = _call("vsig", execution="jax", core=core, **options)
        actual = _call("vsig", execution="wordwise", core=core, **options)
        _assert_tree_close(actual, expected)
    assert routes == ["state", "state", "readout", "state", "state"]


@pytest.mark.parametrize(
    "method", ["update_with_path", "update_with_increment", "states", "vsig"]
)
@pytest.mark.parametrize("execution", ["jax", "wordwise"])
def test_stateful_wrapper_forwards_execution(method, execution, monkeypatch):
    model = StateSpaceSignature(_kernel(), trunc=2)
    seen = []

    def observe(*args, **kwargs):
        seen.append(kwargs["execution"])
        return model.state

    monkeypatch.setattr(state_module, "fssk_state", observe)
    monkeypatch.setattr(state_module, "fssk_vsig", observe)
    path = _path()[1] if method == "update_with_increment" else _path()
    getattr(model, method)(path, dt=0.1, execution=execution)
    assert seen == [execution]
