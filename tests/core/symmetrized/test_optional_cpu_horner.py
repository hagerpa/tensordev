from __future__ import annotations

import importlib

import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from tensordev.core.bigraded.standard import StandardBigradedCore
from tensordev.core.bigraded.symmetrized import _cpu_horner
from tensordev.core.bigraded.symmetrized.jax import (
    JaxPartiallySymmetrizedBigraded,
)
from tensordev.core.bigraded.types import BigradedTensor


config.update("jax_enable_x64", True)


_TRUNCATION = (4, 4)
_NATIVE_BATCH = 9


def _core(*, dims=(1, 2), truncation=_TRUNCATION):
    return JaxPartiallySymmetrizedBigraded(
        dims=dims,
        max_trunc=truncation,
        default_trunc=truncation,
    )


def _random_tensor(core, key, *, batch, dtype, truncation=_TRUNCATION):
    layout = core.resolve_layout(truncation, include_scalar=True)
    keys = jr.split(key, len(layout.grades))
    return BigradedTensor(
        tuple(
            0.02
            * jr.normal(
                block_key,
                batch + (layout.block_width(grade),),
                dtype=dtype,
            )
            for block_key, grade in zip(keys, layout.grades)
        ),
        layout.spec,
    )


def _portable(core, left, generator, truncation=_TRUNCATION):
    return StandardBigradedCore._fmexp_first_level(
        core,
        left,
        generator,
        trunc=truncation,
    )


def _assert_tensor_close(actual, expected, *, atol, rtol):
    assert actual.spec == expected.spec
    for grade in actual.grades:
        np.testing.assert_allclose(
            actual[grade],
            expected[grade],
            atol=atol,
            rtol=rtol,
        )


def _native_extension_or_skip():
    try:
        extension = importlib.import_module("tensordev_native_cpu")
        registrations = extension.registrations()
    except (ImportError, OSError, AttributeError) as error:
        pytest.skip(f"optional TensorDev CPU extension is unavailable: {error}")
    if not frozenset(_cpu_horner._TARGETS.values()).issubset(registrations):
        pytest.skip("optional TensorDev CPU extension lacks Horner targets")
    return extension


def test_selector_rejects_ineligible_actions_before_native_registration(
    monkeypatch,
):
    def forbidden_registration():
        raise AssertionError("ineligible actions must not load the extension")

    monkeypatch.setattr(_cpu_horner, "_register_targets", forbidden_registration)

    small_core = _core(truncation=(3, 3))
    small_left = _random_tensor(
        small_core,
        jr.PRNGKey(810),
        batch=(),
        dtype=jnp.float32,
        truncation=(3, 3),
    )
    small_generator = jr.normal(jr.PRNGKey(811), (3,), dtype=jnp.float32)
    assert (
        _cpu_horner.try_fused_horner(
            small_core,
            small_left,
            small_generator,
            (3, 3),
        )
        is None
    )

    wrong_dims = _core(dims=(2, 2))
    wrong_left = _random_tensor(
        wrong_dims,
        jr.PRNGKey(812),
        batch=(_NATIVE_BATCH,),
        dtype=jnp.float32,
    )
    wrong_generator = jr.normal(
        jr.PRNGKey(813),
        (_NATIVE_BATCH, 4),
        dtype=jnp.float32,
    )
    assert (
        _cpu_horner.try_fused_horner(
            wrong_dims,
            wrong_left,
            wrong_generator,
            _TRUNCATION,
        )
        is None
    )


def test_missing_native_extension_falls_back_exactly(monkeypatch):
    core = _core()
    left = _random_tensor(
        core,
        jr.PRNGKey(820),
        batch=(_NATIVE_BATCH,),
        dtype=jnp.float64,
    )
    generator = jr.normal(
        jr.PRNGKey(821),
        (_NATIVE_BATCH, 3),
        dtype=jnp.float64,
    )
    expected = _portable(core, left, generator)
    real_import = _cpu_horner.importlib.import_module
    attempted = []

    def missing_extension(name, *args, **kwargs):
        if name == "tensordev_native_cpu":
            attempted.append(name)
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_cpu_horner, "_REGISTRATION_STATE", None)
    monkeypatch.setattr(
        _cpu_horner.importlib,
        "import_module",
        missing_extension,
    )
    actual = core._fmexp_first_level(left, generator, trunc=_TRUNCATION)

    assert attempted == ["tensordev_native_cpu"]
    assert _cpu_horner._REGISTRATION_STATE is False
    _assert_tensor_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    (
        (jnp.float32, 4e-6, 4e-6),
        (jnp.float64, 2e-12, 2e-12),
    ),
)
def test_native_horner_matches_portable_eager_and_jit(
    dtype,
    atol,
    rtol,
    monkeypatch,
):
    _native_extension_or_skip()
    monkeypatch.setattr(_cpu_horner, "_REGISTRATION_STATE", None)
    core = _core()
    left = _random_tensor(
        core,
        jr.PRNGKey(830),
        batch=(_NATIVE_BATCH,),
        dtype=dtype,
    )
    generator = jr.normal(
        jr.PRNGKey(831),
        (_NATIVE_BATCH, 3),
        dtype=dtype,
    )
    expected = _portable(core, left, generator)
    actual = _cpu_horner.try_fused_horner(
        core,
        left,
        generator,
        _TRUNCATION,
    )
    assert actual is not None
    _assert_tensor_close(actual, expected, atol=atol, rtol=rtol)

    function = jax.jit(
        lambda current, increment: core._fmexp_first_level(
            current,
            increment,
            trunc=_TRUNCATION,
        )
    )
    lowered = function.lower(left, generator)
    target = _cpu_horner._TARGETS[np.dtype(dtype)]
    assert target in lowered.compiler_ir(dialect="hlo").as_hlo_text()
    _assert_tensor_close(
        lowered.compile()(left, generator),
        expected,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    ("dims", "truncation"),
    (
        pytest.param((1, 3), (4, 4), id="q3"),
        pytest.param((1, 4), (4, 4), id="q4"),
        pytest.param((1, 2), (6, 3), id="q2_asymmetric"),
    ),
)
def test_native_horner_matches_portable_across_supported_shapes(
    dims,
    truncation,
    monkeypatch,
):
    _native_extension_or_skip()
    monkeypatch.setattr(_cpu_horner, "_REGISTRATION_STATE", None)
    core = _core(dims=dims, truncation=truncation)
    left = _random_tensor(
        core,
        jr.PRNGKey(835 + dims[1]),
        batch=(_NATIVE_BATCH,),
        dtype=jnp.float32,
        truncation=truncation,
    )
    generator = jr.normal(
        jr.PRNGKey(845 + dims[1]),
        (_NATIVE_BATCH, sum(dims)),
        dtype=jnp.float32,
    )

    expected = _portable(core, left, generator, truncation)
    actual = _cpu_horner.try_fused_horner(
        core,
        left,
        generator,
        truncation,
    )

    assert actual is not None
    _assert_tensor_close(actual, expected, atol=5e-6, rtol=5e-6)


def test_native_horner_custom_jvp_and_reverse_gradient(monkeypatch):
    _native_extension_or_skip()
    monkeypatch.setattr(_cpu_horner, "_REGISTRATION_STATE", None)
    core = _core()
    left = _random_tensor(
        core,
        jr.PRNGKey(840),
        batch=(_NATIVE_BATCH,),
        dtype=jnp.float64,
    )
    generator = jr.normal(
        jr.PRNGKey(841),
        (_NATIVE_BATCH, 3),
        dtype=jnp.float64,
    )
    left_dot = _random_tensor(
        core,
        jr.PRNGKey(842),
        batch=(_NATIVE_BATCH,),
        dtype=jnp.float64,
    )
    generator_dot = jr.normal(
        jr.PRNGKey(843),
        generator.shape,
        dtype=jnp.float64,
    )
    native = lambda current, increment: core._fmexp_first_level(
        current,
        increment,
        trunc=_TRUNCATION,
    )
    portable = lambda current, increment: _portable(
        core,
        current,
        increment,
    )

    expected, expected_dot = jax.jvp(
        portable,
        (left, generator),
        (left_dot, generator_dot),
    )
    actual, actual_dot = jax.jvp(
        native,
        (left, generator),
        (left_dot, generator_dot),
    )
    _assert_tensor_close(actual, expected, atol=2e-12, rtol=2e-12)
    _assert_tensor_close(actual_dot, expected_dot, atol=2e-11, rtol=2e-11)

    def loss(action, current, increment):
        output = action(current, increment)
        return sum(jnp.mean(block * block) for block in output.blocks)

    expected_grad = jax.jit(
        jax.grad(
            lambda current, increment: loss(portable, current, increment),
            argnums=(0, 1),
        )
    )(left, generator)
    actual_grad = jax.jit(
        jax.grad(
            lambda current, increment: loss(native, current, increment), argnums=(0, 1)
        )
    )(left, generator)
    _assert_tensor_close(
        actual_grad[0],
        expected_grad[0],
        atol=2e-10,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        actual_grad[1],
        expected_grad[1],
        atol=2e-10,
        rtol=2e-10,
    )


@pytest.mark.parametrize("outer_batch", ((2,), (2, 3)))
def test_native_horner_external_and_nested_vmap(outer_batch, monkeypatch):
    _native_extension_or_skip()
    monkeypatch.setattr(_cpu_horner, "_REGISTRATION_STATE", None)
    core = _core()
    batch = outer_batch + (_NATIVE_BATCH,)
    left = _random_tensor(
        core,
        jr.PRNGKey(850 + len(outer_batch)),
        batch=batch,
        dtype=jnp.float32,
    )
    generator = jr.normal(
        jr.PRNGKey(860 + len(outer_batch)),
        batch + (3,),
        dtype=jnp.float32,
    )
    native = lambda current, increment: core._fmexp_first_level(
        current,
        increment,
        trunc=_TRUNCATION,
    )
    portable = lambda current, increment: _portable(
        core,
        current,
        increment,
    )
    for _ in outer_batch:
        native = jax.vmap(native)
        portable = jax.vmap(portable)

    lowered = jax.jit(native).lower(left, generator)
    assert (
        _cpu_horner._TARGETS[np.dtype(jnp.float32)]
        in lowered.compiler_ir(dialect="hlo").as_hlo_text()
    )
    actual = lowered.compile()(left, generator)
    expected = jax.jit(portable)(left, generator)
    _assert_tensor_close(actual, expected, atol=5e-6, rtol=5e-6)


@pytest.mark.parametrize("mapped_argument", ("left", "generator"))
def test_native_horner_vmap_with_one_unmapped_argument(
    mapped_argument,
    monkeypatch,
):
    _native_extension_or_skip()
    monkeypatch.setattr(_cpu_horner, "_REGISTRATION_STATE", None)
    core = _core()
    outer_batch = 2
    left_batch = (
        (outer_batch, _NATIVE_BATCH) if mapped_argument == "left" else (_NATIVE_BATCH,)
    )
    generator_batch = (
        (_NATIVE_BATCH,) if mapped_argument == "left" else (outer_batch, _NATIVE_BATCH)
    )
    left = _random_tensor(
        core,
        jr.PRNGKey(870),
        batch=left_batch,
        dtype=jnp.float32,
    )
    generator = jr.normal(
        jr.PRNGKey(871),
        generator_batch + (3,),
        dtype=jnp.float32,
    )

    native_action = lambda current, increment: core._fmexp_first_level(
        current,
        increment,
        trunc=_TRUNCATION,
    )
    portable_action = lambda current, increment: _portable(
        core,
        current,
        increment,
    )
    in_axes = (0, None) if mapped_argument == "left" else (None, 0)
    native = jax.jit(jax.vmap(native_action, in_axes=in_axes))
    portable = jax.jit(jax.vmap(portable_action, in_axes=in_axes))
    lowered = native.lower(left, generator)
    target = _cpu_horner._TARGETS[np.dtype(jnp.float32)]
    assert target in lowered.compiler_ir(dialect="hlo").as_hlo_text()
    _assert_tensor_close(
        lowered.compile()(left, generator),
        portable(left, generator),
        atol=5e-6,
        rtol=5e-6,
    )
