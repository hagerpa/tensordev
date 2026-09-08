from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import tensordev as td
import tensordev._wordwise.dispatch as dispatch_module
from tensordev._wordwise import ordinary
from tensordev._wordwise.dispatch import (
    _colocate_array,
    ordinary_wordwise_candidate_eligible,
    ordinary_wordwise_device_eligible,
)
from tensordev._wordwise.pallas_quotient import PallasQuotientPlanError
from tensordev.core.capabilities import _WORDWISE_SIGNATURE_PROTOCOL
from tensordev.development.free import _prepare_free_development_call


class _FakeDevice:
    def __init__(
            self,
            *,
            platform="gpu",
            device_kind="NVIDIA test device",
            compute_capability=(8, 0),
    ):
        self.platform = platform
        self.device_kind = device_kind
        self.compute_capability = compute_capability


class _FakeArray:
    def __init__(self, *devices, dtype=np.float32):
        self.dtype = np.dtype(dtype)
        self._devices = devices

    def devices(self):
        return set(self._devices)


class _CompatibleCore:
    _wordwise_signature_protocol = _WORDWISE_SIGNATURE_PROTOCOL
    backend = "jax"
    grading = "total_degree"


class _CompatibleSequentialCore:
    _wordwise_signature_protocol = _WORDWISE_SIGNATURE_PROTOCOL
    backend = "jax"


def _call(increment, *, parallel=False, accumulate_in_tree=False):
    return SimpleNamespace(
        increments=(increment,),
        neutral=(),
        seed_policy=SimpleNamespace(canonical_start=()),
        parallel=parallel,
        accumulate_in_tree=accumulate_in_tree,
        core=_CompatibleCore(),
        seq_core=_CompatibleSequentialCore(),
    )


def _assert_dense_close(actual, expected):
    assert len(actual) == len(expected)
    for actual_level, expected_level in zip(actual, expected):
        np.testing.assert_allclose(
            np.asarray(actual_level),
            np.asarray(expected_level),
            atol=2e-12,
            rtol=2e-12,
        )


def test_cpu_automatic_fallback_does_not_build_a_layout_plan(monkeypatch):
    path = jnp.asarray(
        [[[0.0, 0.1], [0.2, -0.1], [0.3, 0.05]]],
        dtype=jnp.float64,
    )
    core = td.make_core(dims=2, max_trunc=3)
    expected = td.free_development(
        (path,),
        trunc=3,
        axis=-2,
        accumulate=False,
        core=core,
    )

    def unexpected_plan(*args, **kwargs):
        del args, kwargs
        raise AssertionError("CPU fallback constructed a wordwise layout plan")

    monkeypatch.setattr(ordinary, "build_layout_plan", unexpected_plan)
    actual = td.path_signature(
        path,
        trunc=3,
        axis=-2,
        accumulate=False,
        core=core,
    )

    _assert_dense_close(actual, expected)


def test_candidate_eligibility_rejects_traced_increments():
    observed = []

    def trace(increment):
        observed.append(ordinary_wordwise_candidate_eligible(_call(increment)))
        return increment

    jax.make_jaxpr(trace)(jnp.ones((1, 2, 2), dtype=jnp.float32))
    assert observed and not any(observed)


def test_candidate_eligibility_rejects_tree_modes():
    increment = _FakeArray(_FakeDevice())

    assert ordinary_wordwise_candidate_eligible(_call(increment))
    assert not ordinary_wordwise_candidate_eligible(
        _call(increment, parallel=True)
    )
    assert not ordinary_wordwise_candidate_eligible(
        _call(increment, accumulate_in_tree=True)
    )


def test_candidate_eligibility_rejects_unsupported_devices_and_dtypes():
    cpu = _FakeArray(_FakeDevice(platform="cpu", device_kind="CPU"))
    amd = _FakeArray(_FakeDevice(device_kind="AMD Radeon"))
    old_nvidia = _FakeArray(_FakeDevice(compute_capability=(7, 0)))
    turing_nvidia = _FakeArray(_FakeDevice(compute_capability=(7, 5)))
    sharded = _FakeArray(_FakeDevice(), _FakeDevice())
    low_precision = _FakeArray(_FakeDevice(), dtype=np.float16)

    for increment in (
        cpu,
        amd,
        old_nvidia,
        turing_nvidia,
        sharded,
        low_precision,
    ):
        assert not ordinary_wordwise_candidate_eligible(_call(increment))


def test_candidate_eligibility_rejects_cores_without_a_supported_plan():
    increment = _FakeArray(_FakeDevice())
    unsupported = _call(increment)
    unsupported.core = SimpleNamespace(backend="jax", grading="custom")
    unplanned_bidegree = _call(increment)

    class UnplannedBidegree:
        _wordwise_signature_protocol = _WORDWISE_SIGNATURE_PROTOCOL
        backend = "jax"
        grading = "bidegree"

    unplanned_bidegree.core = UnplannedBidegree()

    assert not ordinary_wordwise_candidate_eligible(unsupported)
    assert not ordinary_wordwise_candidate_eligible(unplanned_bidegree)


def test_candidate_eligibility_requires_an_explicit_core_contract():
    increment = _FakeArray(_FakeDevice())
    call = _call(increment)
    call.core = SimpleNamespace(backend="jax", grading="total_degree")
    assert not ordinary_wordwise_candidate_eligible(call)

    class InheritedOnly(_CompatibleCore):
        pass

    call = _call(increment)
    call.core = InheritedOnly()
    assert not ordinary_wordwise_candidate_eligible(call)


def test_automatic_native_release_gate_defaults_closed():
    increment = _FakeArray(_FakeDevice())

    assert ordinary_wordwise_candidate_eligible(_call(increment))
    assert not ordinary_wordwise_device_eligible(_call(increment))


def test_every_public_jax_core_family_declares_the_wordwise_contract():
    cores = (
        td.make_core(dims=2, max_trunc=2),
        td.make_core(
            dims=(1, 1), max_trunc=2, coordinates="shear"
        ),
        td.make_core(dims=(1, 1), max_trunc=(1, 1)),
        td.make_core(
            dims=(1, 1), max_trunc=(1, 1), coordinates="shear"
        ),
        td.make_core(
            dims=(1, 1),
            max_trunc=(1, 1),
            partially_symmetrized=True,
        ),
        td.make_core(
            dims=(1, 1),
            max_trunc=(1, 1),
            partially_symmetrized=True,
            coordinates="shear",
        ),
    )
    for core in cores:
        assert (
            type(core).__dict__.get("_wordwise_signature_protocol")
            is _WORDWISE_SIGNATURE_PROTOCOL
        )


def test_automatic_runner_admits_partial_plans(monkeypatch):
    core = td.make_core(
        dims=(1, 1),
        max_trunc=(1, 1),
        partially_symmetrized=True,
    )
    increments = jnp.ones((1, 2, 2), dtype=jnp.float32)
    call = _prepare_free_development_call(
        (increments,),
        trunc=(1, 1),
        increment_input=True,
        core=core,
    )
    sentinel = object()
    observed = []

    monkeypatch.setattr(
        ordinary, "ordinary_wordwise_device_eligible", lambda call: True
    )

    def execute(call, *, plan):
        observed.append(plan)
        return sentinel

    monkeypatch.setattr(ordinary, "run_ordinary_wordwise", execute)

    assert ordinary.try_ordinary_wordwise(call) is sentinel
    assert observed[0].partially_symmetrized


def test_automatic_runner_falls_back_on_quotient_plan_error(monkeypatch):
    core = td.make_core(
        dims=(1, 1),
        max_trunc=(1, 1),
        partially_symmetrized=True,
    )
    increments = jnp.ones((1, 2, 2), dtype=jnp.float32)
    call = _prepare_free_development_call(
        (increments,),
        trunc=(1, 1),
        increment_input=True,
        core=core,
    )

    monkeypatch.setattr(
        ordinary, "ordinary_wordwise_device_eligible", lambda call: True
    )

    def fail(*args, **kwargs):
        raise PallasQuotientPlanError("invalid packed plan")

    monkeypatch.setattr(ordinary, "run_ordinary_wordwise", fail)

    assert ordinary.try_ordinary_wordwise(call) is None


def test_metadata_colocation_does_not_materialize_on_default_device_first(
    monkeypatch,
):
    metadata = np.asarray([1, 2, 3], dtype=np.int32)
    device = object()
    reference = SimpleNamespace(device=device)
    sentinel = object()
    observed = []

    def device_put(value, target):
        observed.append((value, target))
        return sentinel

    def unexpected_asarray(value):
        raise AssertionError("metadata was materialized before device_put")

    monkeypatch.setattr(dispatch_module.jax, "device_put", device_put)
    monkeypatch.setattr(dispatch_module.jnp, "asarray", unexpected_asarray)

    assert _colocate_array(metadata, reference) is sentinel
    assert observed == [(metadata, device)]


def test_captured_concrete_path_stays_portable_under_jax_transforms(
    monkeypatch,
):
    core = td.Jax()
    captured = jnp.asarray(
        [[0.08, -0.02], [0.03, 0.05]],
        dtype=jnp.float64,
    )
    starting_point = (
        jnp.ones((1,), dtype=captured.dtype),
        jnp.asarray([0.02, -0.01], dtype=captured.dtype),
        jnp.zeros((4,), dtype=captured.dtype),
    )

    monkeypatch.setattr(dispatch_module, "_supported_cuda_device", lambda _: True)
    monkeypatch.setattr(
        dispatch_module,
        "_automatic_wordwise_release_eligible",
        lambda: True,
    )

    def unexpected_native(*args, **kwargs):
        raise AssertionError("a transformed call attempted wordwise execution")

    monkeypatch.setattr(ordinary, "try_ordinary_wordwise", unexpected_native)

    def transformed(scale):
        seed = tuple(level * scale for level in starting_point)
        return td.path_signature(
            captured,
            trunc=2,
            increment_input=True,
            starting_point=seed,
            core=core,
        )

    jax.make_jaxpr(transformed)(jnp.asarray(1.0, dtype=captured.dtype))
    compiled = jax.jit(transformed)(jnp.asarray(1.0, dtype=captured.dtype))
    mapped = jax.vmap(transformed)(jnp.asarray([0.8, 1.2], dtype=captured.dtype))
    gradient = jax.grad(
        lambda scale: sum(jnp.sum(level) for level in transformed(scale))
    )(jnp.asarray(1.0, dtype=captured.dtype))

    assert all(jnp.all(jnp.isfinite(level)) for level in compiled)
    assert all(jnp.all(jnp.isfinite(level)) for level in mapped)
    assert jnp.isfinite(gradient)


def test_path_signature_colocates_seed_and_portable_fallback():
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        import tensordev as td

        devices = jax.devices("cpu")
        assert len(devices) == 2
        core = td.Jax()
        increments_host = np.asarray(
            [[[0.08, -0.02], [0.03, 0.05]]], dtype=np.float32
        )
        increments = jax.device_put(increments_host, devices[1])
        seed_increment = jax.device_put(
            jnp.asarray([[0.02, -0.01]], dtype=jnp.float32), devices[0]
        )
        starting_point = core.tensor_exponential(
            (seed_increment,), trunc=2, output_zero_level=True
        )

        for accumulate in (True, False):
            actual = td.path_signature(
                increments,
                trunc=2,
                increment_input=True,
                starting_point=starting_point,
                output_starting_point=True,
                block_size=1,
                accumulate=accumulate,
                core=core,
            )
            expected = td.path_signature(
                jax.device_put(increments, devices[0]),
                trunc=2,
                increment_input=True,
                starting_point=starting_point,
                output_starting_point=True,
                block_size=1,
                accumulate=accumulate,
                core=core,
            )
            assert all(leaf.device == devices[1] for leaf in jax.tree.leaves(actual))
            for actual_leaf, expected_leaf in zip(actual, expected):
                np.testing.assert_allclose(
                    np.asarray(actual_leaf),
                    np.asarray(expected_leaf),
                    atol=2e-6,
                    rtol=2e-6,
                )
        """
    )
    env = os.environ.copy()
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env["JAX_PLATFORMS"] = "cpu"
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_sharded_path_does_not_treat_its_sharding_as_one_device():
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec
        import tensordev as td

        devices = np.asarray(jax.devices("cpu"))
        assert len(devices) == 2
        mesh = Mesh(devices, ("batch",))
        sharding = NamedSharding(mesh, PartitionSpec("batch"))
        increments = jax.device_put(
            jnp.asarray(
                [
                    [[0.08, -0.02], [0.03, 0.05]],
                    [[0.04, 0.01], [-0.02, 0.06]],
                ],
                dtype=jnp.float32,
            ),
            sharding,
        )
        core = td.Jax()
        starting_point = core.tensor_exponential(
            (jnp.asarray([0.02, -0.01], dtype=jnp.float32),),
            trunc=2,
            output_zero_level=True,
        )

        actual = td.path_signature(
            increments,
            trunc=2,
            increment_input=True,
            starting_point=starting_point,
            core=core,
        )
        expected = td.path_signature(
            jnp.asarray(increments),
            trunc=2,
            increment_input=True,
            starting_point=starting_point,
            core=core,
        )
        for actual_leaf, expected_leaf in zip(actual, expected):
            np.testing.assert_allclose(
                np.asarray(actual_leaf),
                np.asarray(expected_leaf),
                atol=2e-6,
                rtol=2e-6,
            )
        """
    )
    env = os.environ.copy()
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env["JAX_PLATFORMS"] = "cpu"
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
