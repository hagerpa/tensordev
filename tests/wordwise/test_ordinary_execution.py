from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
import tensordev._wordwise.dispatch as dispatch_module
from tensordev._wordwise import ordinary
from tensordev._wordwise.layout import PlanResourceError, estimate_layout_plan
from tensordev._wordwise.pallas_ordinary import PallasOrdinaryError
from tensordev.development import sig as signature_module


def _increments():
    return jnp.asarray(
        [
            [[0.1, -0.03, 0.02], [0.04, 0.08, -0.05],
             [-0.02, 0.06, 0.03], [0.07, -0.01, 0.04]],
            [[-0.02, 0.07, 0.01], [0.09, -0.02, 0.06],
             [0.03, 0.01, -0.04], [-0.01, 0.05, 0.02]],
        ],
        dtype=jnp.float64,
    )


def _core(family):
    if family == "total-standard":
        return td.make_core(dims=3, max_trunc=2)
    if family == "total-shear":
        return td.make_core(dims=(1, 2), max_trunc=2, coordinates="shear")
    return td.make_core(
        dims=(1, 2),
        max_trunc=(1, 2),
        partially_symmetrized="partial" in family,
        coordinates="shear" if family.endswith("shear") else "standard",
    )


def _assert_close(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_block, expected_block in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected)
    ):
        np.testing.assert_allclose(
            actual_block, expected_block, atol=2e-12, rtol=2e-12
        )


def _unexpected(*args, **kwargs):
    raise AssertionError("unexpected wordwise or fallback execution")


@pytest.fixture
def interpreted_wordwise(monkeypatch):
    monkeypatch.setattr(dispatch_module, "_supported_cuda_device", lambda _: True)
    monkeypatch.setattr(
        dispatch_module, "_automatic_wordwise_release_eligible", lambda: False
    )
    run = ordinary.run_ordinary_wordwise
    calls = []

    def interpreted(call, **kwargs):
        calls.append(call)
        return run(call, interpret=True, **kwargs)

    monkeypatch.setattr(ordinary, "run_ordinary_wordwise", interpreted)
    yield calls
    jax.clear_caches()


@pytest.mark.parametrize("execution", [None, "gpu", "WORDWISE", True, 1, [], {}])
def test_invalid_execution_is_rejected(execution):
    with pytest.raises(ValueError, match="execution"):
        td.path_signature(
            _increments(), trunc=2, increment_input=True, execution=execution
        )


def test_cpu_wordwise_is_rejected_before_planning(monkeypatch):
    increments = jax.device_put(_increments(), jax.devices("cpu")[0])
    monkeypatch.setattr(ordinary, "build_layout_plan", _unexpected)
    monkeypatch.setattr(
        signature_module, "_execute_portable_free_development_call", _unexpected
    )

    with pytest.raises(ValueError, match="wordwise"):
        td.path_signature(
            increments, trunc=2, increment_input=True, execution="wordwise"
        )


def test_jax_selection_bypasses_automatic_wordwise(monkeypatch):
    increments = _increments()
    expected = td.path_signature(
        increments, trunc=2, increment_input=True, execution="jax"
    )
    monkeypatch.setattr(dispatch_module, "_supported_cuda_device", lambda _: True)
    monkeypatch.setattr(
        dispatch_module, "_automatic_wordwise_release_eligible", lambda: True
    )
    monkeypatch.setattr(ordinary, "build_layout_plan", _unexpected)
    monkeypatch.setattr(ordinary, "run_ordinary_wordwise", _unexpected)

    actual = td.path_signature(
        increments, trunc=2, increment_input=True, execution="jax"
    )
    _assert_close(actual, expected)


@pytest.mark.parametrize("error_type", [PlanResourceError, PallasOrdinaryError])
def test_explicit_wordwise_propagates_executor_errors(monkeypatch, error_type):
    monkeypatch.setattr(dispatch_module, "_supported_cuda_device", lambda _: True)
    message = "executor resource or backend failure"
    error = (
        error_type(
            message,
            estimate=estimate_layout_plan(
                grading="total_degree", dims=3, truncation=2
            ),
        )
        if error_type is PlanResourceError
        else error_type(message)
    )

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(ordinary, "run_ordinary_wordwise", fail)
    monkeypatch.setattr(
        signature_module, "_execute_portable_free_development_call", _unexpected
    )

    with pytest.raises(error_type) as caught:
        td.path_signature(
            _increments(), trunc=2, increment_input=True, execution="wordwise"
        )
    assert caught.value is error


@pytest.mark.parametrize(
    "family",
    [
        "total-standard", "total-shear", "bidegree-standard", "bidegree-shear",
        "bidegree-partial-standard", "bidegree-partial-shear",
    ],
)
def test_explicit_wordwise_matches_jax_in_every_core_family(
    interpreted_wordwise, family
):
    core = _core(family)
    path = jnp.cumsum(_increments(), axis=-2)
    expected = td.path_signature(path, core=core, execution="jax")
    actual = td.path_signature(path, core=core, execution="wordwise")

    assert len(interpreted_wordwise) == 1
    _assert_close(actual, expected)


@pytest.mark.parametrize("accumulate", [False, True])
def test_blocked_wordwise_preserves_seed_axes_and_starting_output(
    interpreted_wordwise, accumulate
):
    core = _core("bidegree-partial-shear")
    increments = _increments()
    seed = td.path_signature(
        increments[:, :1], core=core, increment_input=True, execution="jax"
    )
    seed = core.tensor_scalar_multiply(seed, 0.7)
    options = dict(
        core=core,
        axis=0,
        increment_input=True,
        block_size=2,
        accumulate=accumulate,
        starting_point=seed,
        output_starting_point=True,
    )
    moved = jnp.moveaxis(increments, -2, 0)
    expected = td.path_signature(moved, execution="jax", **options)
    actual = td.path_signature(moved, execution="wordwise", **options)

    assert len(interpreted_wordwise) == 1
    _assert_close(actual, expected)


@pytest.mark.parametrize("tree_option", ["parallel", "accumulate_in_tree"])
def test_explicit_wordwise_rejects_tree_modes(monkeypatch, tree_option):
    monkeypatch.setattr(dispatch_module, "_supported_cuda_device", lambda _: True)
    monkeypatch.setattr(ordinary, "build_layout_plan", _unexpected)

    with pytest.raises(ValueError, match="wordwise"):
        td.path_signature(
            _increments(), trunc=2, increment_input=True,
            execution="wordwise", **{tree_option: True}
        )


@pytest.mark.parametrize("transform", ["jit", "grad", "vmap"])
def test_explicit_wordwise_rejects_transformed_calls(monkeypatch, transform):
    monkeypatch.setattr(dispatch_module, "_supported_cuda_device", lambda _: True)
    monkeypatch.setattr(ordinary, "build_layout_plan", _unexpected)

    def objective(increments):
        signature = td.path_signature(
            increments, trunc=2, increment_input=True, execution="wordwise"
        )
        return sum(jnp.sum(block) for block in jax.tree.leaves(signature))

    with pytest.raises(ValueError, match="wordwise"):
        getattr(jax, transform)(objective)(_increments())


def test_signature_wrapper_forwards_execution(monkeypatch):
    signature = td.Signature(trunc=2)
    calls = []
    sentinel = object()

    def record(path, **kwargs):
        calls.append(kwargs["execution"])
        return sentinel

    monkeypatch.setattr(signature_module, "path_signature", record)
    for options in ({}, {"execution": "jax"}, {"execution": "wordwise"}):
        assert signature(_increments(), **options) is sentinel
    assert calls == ["auto", "jax", "wordwise"]
