"""Real-GPU validation for public scalar-FSSK wordwise execution."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
import tensordev._wordwise.dispatch as dispatch_module
from tensordev._wordwise.dispatch import _supported_cuda_device
from tensordev._wordwise.layout import build_layout_plan
from tensordev.sss import FSSK
from tensordev.sss import state_update as state_update_module
from tensordev.sss.state_update import (
    fssk_readout,
    fssk_state,
    fssk_state_from_coef,
    fssk_vsig,
)


def _supported_devices():
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        return ()
    return tuple(device for device in devices if _supported_cuda_device(device))


GPU_DEVICES = _supported_devices()
CPU_DEVICE = jax.devices("cpu")[0]
pytestmark = pytest.mark.skipif(
    not GPU_DEVICES,
    reason="requires a supported NVIDIA JAX device",
)


_FAMILIES = (
    "total-standard",
    "total-shear",
    "bidegree-standard",
    "bidegree-shear",
    "bidegree-partial-standard",
    "bidegree-partial-shear",
)


def _case(name):
    if name == "total-standard":
        return td.Jax(), 2, None, False
    if name == "total-shear":
        return (
            td.make_core(dims=(1, 1), max_trunc=2, coordinates="shear"),
            2,
            2,
            True,
        )
    if name == "bidegree-standard":
        return td.make_core(dims=(1, 1), max_trunc=(1, 1)), (1, 1), 2, False
    if name == "bidegree-shear":
        return (
            td.make_core(
                dims=(1, 1),
                max_trunc=(1, 1),
                coordinates="shear",
            ),
            (1, 1),
            None,
            False,
        )
    if name == "bidegree-partial-standard":
        return (
            td.make_core(
                dims=(1, 1),
                max_trunc=(1, 1),
                partially_symmetrized=True,
            ),
            (1, 1),
            2,
            True,
        )
    if name == "bidegree-partial-shear":
        return (
            td.make_core(
                dims=(1, 1),
                max_trunc=(1, 1),
                partially_symmetrized=True,
                coordinates="shear",
            ),
            (1, 1),
            2,
            False,
        )
    raise AssertionError(f"unknown core family {name!r}")


def _device(value):
    device = getattr(value, "device", None)
    return device() if callable(device) else device


def _put(value, device):
    return jax.device_put(value, device)


def _q1_kernel(dtype, *, device):
    host = FSSK.from_matrix(
        Lambda=np.diag(np.asarray([0.25, 0.45, 0.7], dtype=dtype)),
        A=np.eye(2, dtype=dtype)[None],
        b=np.asarray([[0.8, 0.15, -0.2]], dtype=dtype),
        quad_order=16,
    )
    return _put(host, device)


def _q2_kernel(dtype, *, device):
    host = FSSK.from_matrix(
        Lambda=np.diag(np.asarray([0.2, 0.5, 0.75], dtype=dtype)),
        A=np.asarray(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.3, -0.2], [0.1, 0.6]],
            ],
            dtype=dtype,
        ),
        b=np.asarray(
            [[0.7, 0.2, -0.1], [0.1, -0.15, 0.5]],
            dtype=dtype,
        ),
        quad_order=16,
    )
    return _put(host, device)


def _increments(dtype, *, device):
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
    return _put(host, device)


def _initial_state(core, truncation, dtype, *, device):
    plan = build_layout_plan(core, truncation, alphabet_dim=2)
    blocks = []
    offset = 1
    for block in plan.blocks[1:]:
        size = 2 * 3 * block.width
        values = np.arange(offset, offset + size, dtype=dtype)
        blocks.append((0.002 * values).reshape(2, 1, 1, 3, block.width))
        offset += size
    standard = plan.assemble_first_on(tuple(jnp.asarray(block) for block in blocks))
    native = core.tensor_from_standard_coordinates(
        standard,
        trunc=truncation,
        first_on=True,
    )
    return _put(native, device)


def _assert_on_gpu(value):
    for leaf in jax.tree.leaves(value):
        assert _device(leaf) == GPU_DEVICES[0]


def _assert_tensors_close(actual, expected, *, dtype):
    if hasattr(actual, "spec") or hasattr(expected, "spec"):
        assert actual.spec == expected.spec
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    tolerance = 5e-5 if dtype == np.float32 else 5e-10
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        assert actual_leaf.shape == expected_leaf.shape
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            atol=tolerance,
            rtol=tolerance,
        )


def _tree_square_norm(value):
    return sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(value))


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_candidate_public_q1_apis_match_forced_portable(
    monkeypatch,
    family,
    dtype,
):
    monkeypatch.setattr(
        dispatch_module,
        "_automatic_wordwise_release_eligible",
        lambda: True,
    )
    gpu = GPU_DEVICES[0]
    core, truncation, block_size, accumulate = _case(family)
    varying = dtype == np.float64
    nonzero_seed = dtype == np.float64

    with jax.default_device(CPU_DEVICE):
        assert _device(jnp.zeros((), dtype=jnp.float32)) == CPU_DEVICE
        increments = _increments(dtype, device=gpu)
        kernel = _q1_kernel(dtype, device=CPU_DEVICE)
        dt = _put(
            np.linspace(0.04, 0.1, 4, dtype=dtype)
            if varying
            else np.asarray(0.07, dtype=dtype),
            CPU_DEVICE,
        )
        coef = kernel.coef(dt, trunc=2, dtype=jnp.dtype(dtype))
        tau_dt = _put(np.asarray(0.0, dtype=dtype), CPU_DEVICE)
        initial = (
            _initial_state(core, truncation, dtype, device=gpu)
            if nonzero_seed
            else None
        )
        output_starting_state = nonzero_seed

        assert _device(increments) == gpu
        assert all(_device(leaf) == CPU_DEVICE for leaf in jax.tree.leaves(kernel))
        assert _device(dt) == CPU_DEVICE
        assert all(
            _device(leaf) == CPU_DEVICE
            for leaf in jax.tree.leaves(coef)
            if isinstance(leaf, jax.Array)
        )
        assert _device(tau_dt) == CPU_DEVICE

        native_results = []
        original_try = state_update_module._try_fssk_q1_wordwise
        original_try_readout = (
            state_update_module._try_fssk_q1_wordwise_readout
        )

        def observed_native(*args, **kwargs):
            result = original_try(*args, **kwargs)
            native_results.append(result is not None)
            return result

        monkeypatch.setattr(
            state_update_module,
            "_try_fssk_q1_wordwise",
            observed_native,
        )

        def observed_native_readout(*args, **kwargs):
            result = original_try_readout(*args, **kwargs)
            native_results.append(result is not None)
            return result

        monkeypatch.setattr(
            state_update_module,
            "_try_fssk_q1_wordwise_readout",
            observed_native_readout,
        )
        state_kwargs = dict(
            trunc=truncation,
            axis=1,
            block_size=block_size,
            accumulate=accumulate,
            initial_state=initial,
            output_starting_state=output_starting_state,
            core=core,
        )
        automatic_state = fssk_state(
            increments,
            kernel=kernel,
            dt=dt,
            increment_input=True,
            **state_kwargs,
        )
        automatic_from_coef = fssk_state_from_coef(
            increments,
            coef=coef,
            **state_kwargs,
        )
        automatic_vsig = fssk_vsig(
            increments,
            kernel=kernel,
            dt=dt,
            increment_input=True,
            tau_dt=tau_dt,
            **state_kwargs,
        )

        assert len(native_results) >= 3
        assert all(native_results)

        monkeypatch.setattr(
            state_update_module,
            "_try_fssk_q1_wordwise",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            state_update_module,
            "_try_fssk_q1_wordwise_readout",
            lambda *args, **kwargs: None,
        )
        portable_state = fssk_state(
            increments,
            kernel=kernel,
            dt=dt,
            increment_input=True,
            **state_kwargs,
        )
        portable_from_coef = fssk_state_from_coef(
            increments,
            coef=coef,
            **state_kwargs,
        )
        portable_vsig = fssk_vsig(
            increments,
            kernel=kernel,
            dt=dt,
            increment_input=True,
            tau_dt=tau_dt,
            **state_kwargs,
        )

    _assert_tensors_close(automatic_state, portable_state, dtype=dtype)
    _assert_tensors_close(automatic_from_coef, portable_from_coef, dtype=dtype)
    _assert_tensors_close(automatic_vsig, portable_vsig, dtype=dtype)
    _assert_tensors_close(automatic_state, automatic_from_coef, dtype=dtype)
    _assert_on_gpu(automatic_state)
    _assert_on_gpu(automatic_from_coef)
    _assert_on_gpu(automatic_vsig)


def test_outer_jit_vmap_and_grad_remain_on_the_portable_route(monkeypatch):
    dtype = np.float32
    gpu = GPU_DEVICES[0]
    core = td.make_core(dims=2, max_trunc=2)

    with jax.default_device(CPU_DEVICE):
        y = _increments(dtype, device=gpu)
        kernel = _q1_kernel(dtype, device=gpu)
        dt = _put(np.asarray(0.07, dtype=dtype), gpu)
        coef = kernel.coef(dt, trunc=2, dtype=jnp.float32)

        monkeypatch.setattr(
            state_update_module,
            "_try_fssk_q1_wordwise",
            lambda *args, **kwargs: None,
        )

        def portable(value):
            return fssk_state_from_coef(
                value,
                coef=coef,
                trunc=2,
                axis=1,
                core=core,
            )

        expected = portable(y)

        def portable_objective(value):
            return _tree_square_norm(portable(value))

        expected_gradient = jax.grad(portable_objective)(y)

        def unexpected_native(*args, **kwargs):
            raise AssertionError("a transformed FSSK call attempted native execution")

        monkeypatch.setattr(
            state_update_module,
            "_try_fssk_q1_wordwise",
            unexpected_native,
        )

        def transformed(value):
            return fssk_state_from_coef(
                value,
                coef=coef,
                trunc=2,
                axis=1,
                core=core,
            )

        compiled = jax.jit(transformed)(y)
        vmapped = jax.vmap(
            lambda value: fssk_state_from_coef(
                value,
                coef=coef,
                trunc=2,
                axis=0,
                core=core,
            )
        )(y)

        def transformed_objective(value):
            return _tree_square_norm(transformed(value))

        gradient = jax.grad(transformed_objective)(y)

    _assert_tensors_close(compiled, expected, dtype=dtype)
    _assert_tensors_close(vmapped, expected, dtype=dtype)
    np.testing.assert_allclose(
        np.asarray(gradient),
        np.asarray(expected_gradient),
        atol=5e-5,
        rtol=5e-5,
    )


def test_public_readout_colocates_a_cpu_built_kernel_with_gpu_state():
    dtype = np.float32
    gpu = GPU_DEVICES[0]
    core = td.make_core(dims=2, max_trunc=2)

    with jax.default_device(CPU_DEVICE):
        kernel = _q1_kernel(dtype, device=CPU_DEVICE)
        state = _initial_state(core, 2, dtype, device=gpu)
        assert all(_device(leaf) == CPU_DEVICE for leaf in jax.tree.leaves(kernel))

        expected = fssk_readout(
            _put(state, CPU_DEVICE),
            kernel=kernel,
            core=core,
        )
        actual = fssk_readout(state, kernel=kernel, core=core)

    _assert_tensors_close(actual, expected, dtype=dtype)
    _assert_on_gpu(actual)


def test_q_greater_than_one_stays_portable_and_matches_public_forms(monkeypatch):
    dtype = np.float32
    gpu = GPU_DEVICES[0]
    core = td.make_core(dims=2, max_trunc=2)

    def unexpected_native(*args, **kwargs):
        raise AssertionError("q > 1 attempted scalar native execution")

    monkeypatch.setattr(
        state_update_module,
        "_try_fssk_q1_wordwise",
        unexpected_native,
    )
    with jax.default_device(CPU_DEVICE):
        increments = _increments(dtype, device=gpu)
        kernel = _q2_kernel(dtype, device=gpu)
        dt = _put(np.linspace(0.04, 0.1, 4, dtype=dtype), gpu)
        coef = kernel.coef(dt, trunc=2, dtype=jnp.float32)
        projected = jnp.einsum("qmd,...d->...qm", kernel.A, increments)

        state = fssk_state(
            increments,
            kernel=kernel,
            dt=dt,
            trunc=2,
            axis=1,
            increment_input=True,
            core=core,
        )
        from_coef = fssk_state_from_coef(
            projected,
            coef=coef,
            trunc=2,
            axis=1,
            core=core,
        )
        signature = fssk_vsig(
            increments,
            kernel=kernel,
            dt=dt,
            trunc=2,
            axis=1,
            increment_input=True,
            core=core,
        )
        expected_signature = fssk_readout(state, kernel=kernel, core=core)

    _assert_tensors_close(state, from_coef, dtype=dtype)
    _assert_tensors_close(signature, expected_signature, dtype=dtype)
    _assert_on_gpu(state)
    _assert_on_gpu(signature)
