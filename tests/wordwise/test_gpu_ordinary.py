"""Real-GPU validation for ordinary wordwise execution."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
import tensordev._wordwise.dispatch as dispatch_module
from tensordev._wordwise import ordinary
from tensordev._wordwise.dispatch import _supported_cuda_device
from tensordev._wordwise.ordinary import run_ordinary_wordwise
from tensordev.development.free import (
    _execute_portable_free_development_call,
    _finalize_free_development_call,
    _prepare_free_development_call,
)


def _supported_devices():
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        return ()
    return tuple(device for device in devices if _supported_cuda_device(device))


GPU_DEVICES = _supported_devices()
pytestmark = pytest.mark.skipif(
    not GPU_DEVICES,
    reason="requires a supported NVIDIA JAX device",
)


def _case(name):
    if name == "total-standard":
        return td.make_core(dims=2, max_trunc=3), 3
    if name == "total-shear":
        return (
            td.make_core(
                dims=(1, 1),
                max_trunc=3,
                coordinates="shear",
            ),
            3,
        )
    if name == "bidegree-standard":
        return td.make_core(dims=(1, 1), max_trunc=(2, 1)), (2, 1)
    if name == "bidegree-shear":
        return (
            td.make_core(
                dims=(1, 1),
                max_trunc=(2, 1),
                coordinates="shear",
            ),
            (2, 1),
        )
    if name == "bidegree-partial-standard":
        return (
            td.make_core(
                dims=(1, 1),
                max_trunc=(2, 1),
                partially_symmetrized=True,
            ),
            (2, 1),
        )
    if name == "bidegree-partial-shear":
        return (
            td.make_core(
                dims=(1, 1),
                max_trunc=(2, 1),
                partially_symmetrized=True,
                coordinates="shear",
            ),
            (2, 1),
        )
    raise AssertionError(f"unknown test case {name!r}")


def _increments(dtype):
    host = np.asarray(
        [
            [
                [0.08, -0.02],
                [0.03, 0.05],
                [-0.04, 0.01],
                [0.02, -0.06],
            ],
            [
                [-0.01, 0.07],
                [0.06, -0.03],
                [0.02, 0.04],
                [-0.05, 0.02],
            ],
        ],
        dtype=dtype,
    )
    return jax.device_put(host, GPU_DEVICES[0])


def _assert_tensors_close(actual, expected, *, dtype):
    if hasattr(actual, "spec") or hasattr(expected, "spec"):
        assert actual.spec == expected.spec
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    tolerance = 3e-5 if dtype == np.float32 else 3e-11
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            atol=tolerance,
            rtol=tolerance,
        )


@pytest.mark.parametrize(
    "family",
    [
        "total-standard",
        "total-shear",
        "bidegree-standard",
        "bidegree-shear",
        "bidegree-partial-standard",
        "bidegree-partial-shear",
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("execution", ["auto", "wordwise"])
@pytest.mark.parametrize(
    ("block_size", "accumulate"),
    [(None, False), (2, True), (2, False)],
)
def test_real_gpu_candidate_public_forced_and_portable_agree(
    monkeypatch, family, dtype, execution, block_size, accumulate
):
    if execution == "auto":
        monkeypatch.setattr(
            dispatch_module,
            "_automatic_wordwise_release_eligible",
            lambda: True,
        )
    core, truncation = _case(family)
    increments = _increments(dtype)
    call = _prepare_free_development_call(
        (increments,),
        trunc=truncation,
        increment_input=True,
        axis=-2,
        block_size=block_size,
        accumulate=accumulate,
        core=core,
    )

    calls = []
    original = ordinary.run_ordinary_wordwise

    def observed_wordwise(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(ordinary, "run_ordinary_wordwise", observed_wordwise)
    portable = td.path_signature(
        increments,
        trunc=truncation,
        increment_input=True,
        axis=-2,
        block_size=block_size,
        accumulate=accumulate,
        core=core,
        execution="jax",
    )
    assert not calls, "execution='jax' attempted wordwise execution"
    raw_forced = run_ordinary_wordwise(call)
    forced = _finalize_free_development_call(
        call,
        raw_forced,
        runner_applied_seed=False,
        runner_emitted_starting_point=False,
    )

    public = td.path_signature(
        increments,
        trunc=truncation,
        increment_input=True,
        axis=-2,
        block_size=block_size,
        accumulate=accumulate,
        core=core,
        execution=execution,
    )

    assert calls, "the concrete supported GPU call did not enter wordwise execution"
    _assert_tensors_close(forced, portable, dtype=dtype)
    _assert_tensors_close(public, portable, dtype=dtype)


def _portable_signature(increments, *, core, truncation):
    call = _prepare_free_development_call(
        (increments,),
        trunc=truncation,
        increment_input=True,
        axis=-2,
        accumulate=False,
        core=core,
    )
    return _execute_portable_free_development_call(call)


def _squared_tree_norm(value):
    return sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(value))


def test_jit_and_grad_retain_the_portable_route(monkeypatch):
    core, truncation = _case("total-standard")
    increments = _increments(np.float32)

    def unexpected_plan(*args, **kwargs):
        del args, kwargs
        raise AssertionError("a traced call attempted wordwise planning")

    monkeypatch.setattr(ordinary, "build_layout_plan", unexpected_plan)
    public = lambda x: td.path_signature(
        x,
        trunc=truncation,
        increment_input=True,
        axis=-2,
        accumulate=False,
        core=core,
    )
    portable = lambda x: _portable_signature(
        x,
        core=core,
        truncation=truncation,
    )

    compiled = jax.jit(public)(increments)
    expected = portable(increments)
    _assert_tensors_close(compiled, expected, dtype=np.float32)

    public_gradient = jax.grad(lambda x: _squared_tree_norm(public(x)))(increments)
    portable_gradient = jax.grad(lambda x: _squared_tree_norm(portable(x)))(increments)
    np.testing.assert_allclose(
        np.asarray(public_gradient),
        np.asarray(portable_gradient),
        atol=3e-5,
        rtol=3e-5,
    )
