from __future__ import annotations

from types import SimpleNamespace

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
from tensordev.development import path_signature


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
    if _cpu_horner._native_registrations() is None:
        pytest.skip(
            "TensorDev CPU extension with compatible Horner targets is unavailable"
        )


def _registration_module():
    registrations = {name: object() for name in _cpu_horner._TARGETS.values()}
    type_registrations = {_cpu_horner._STATE_TYPE_NAME: object()}
    return SimpleNamespace(
        registrations=lambda: registrations,
        type_registrations=lambda: type_registrations,
    )


def test_bundled_native_extension_takes_precedence(monkeypatch):
    extension = _registration_module()
    attempted = []

    def bundled_extension(name):
        attempted.append(name)
        assert name == "tensordev._native_cpu"
        return extension

    monkeypatch.setattr(_cpu_horner.importlib, "import_module", bundled_extension)
    registrations = _cpu_horner._native_registrations()

    assert attempted == ["tensordev._native_cpu"]
    assert registrations == (
        extension.registrations(),
        extension.type_registrations(),
    )


@pytest.mark.parametrize("unavailable", ("missing", "unloadable", "incompatible"))
def test_standalone_native_extension_fallback(unavailable, monkeypatch):
    extension = _registration_module()
    attempted = []

    def missing_library():
        raise OSError("shared library cannot be loaded")

    def standalone_extension(name):
        attempted.append(name)
        if name == "tensordev._native_cpu":
            if unavailable == "missing":
                raise ModuleNotFoundError(name)
            if unavailable == "unloadable":
                return SimpleNamespace(registrations=missing_library)
            return SimpleNamespace(registrations=dict, type_registrations=dict)
        assert name == "tensordev_native_cpu"
        return extension

    monkeypatch.setattr(_cpu_horner.importlib, "import_module", standalone_extension)
    registrations = _cpu_horner._native_registrations()

    assert attempted == ["tensordev._native_cpu", "tensordev_native_cpu"]
    assert registrations == (
        extension.registrations(),
        extension.type_registrations(),
    )


@pytest.mark.parametrize("attribute", ("registrations", "type_registrations"))
def test_invalid_native_registration_mapping_raises(attribute, monkeypatch):
    extension = _registration_module()
    setattr(extension, attribute, lambda: None)
    monkeypatch.setattr(_cpu_horner.importlib, "import_module", lambda _: extension)

    with pytest.raises(TypeError, match="must return a mapping"):
        _cpu_horner._native_registrations()


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
        if name in ("tensordev._native_cpu", "tensordev_native_cpu"):
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

    assert attempted == ["tensordev._native_cpu", "tensordev_native_cpu"]
    assert _cpu_horner._REGISTRATION_STATE is False
    _assert_tensor_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("platforms", ("gpu", "cuda", "rocm", "tpu", "cuda,rocm"))
@pytest.mark.parametrize("registration_state", (None, True))
def test_cpu_excluded_skips_registration_and_plan_preparation(
    platforms,
    registration_state,
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU-disabled dispatch must not prepare native execution")

    monkeypatch.setattr(
        _cpu_horner,
        "jax",
        SimpleNamespace(
            config=SimpleNamespace(jax_platforms=platforms),
            devices=forbidden,
        ),
    )
    monkeypatch.setattr(_cpu_horner, "_REGISTRATION_STATE", registration_state)
    monkeypatch.setattr(_cpu_horner.importlib, "import_module", forbidden)
    monkeypatch.setattr(_cpu_horner, "_compile_plan", forbidden)
    core = SimpleNamespace(
        coordinates="standard",
        dims=(1, 2),
        plan_store=SimpleNamespace(resolve=forbidden),
    )
    left = SimpleNamespace(spec=SimpleNamespace(include_scalar=True))

    assert _cpu_horner._register_targets() is False
    assert _cpu_horner._REGISTRATION_STATE is registration_state
    assert _cpu_horner.try_fused_horner(core, left, None, _TRUNCATION) is None


@pytest.mark.parametrize("platforms", (None, "", "cpu", "cuda,cpu", "cpu,tpu"))
def test_cpu_enabled_platform_configurations(platforms, monkeypatch):
    monkeypatch.setattr(
        _cpu_horner,
        "jax",
        SimpleNamespace(config=SimpleNamespace(jax_platforms=platforms)),
    )
    assert _cpu_horner._cpu_backend_enabled()


def test_cpu_excluded_falls_back_exactly_eager_and_jit(monkeypatch):
    core = _core()
    left = _random_tensor(
        core,
        jr.PRNGKey(822),
        batch=(_NATIVE_BATCH,),
        dtype=jnp.float64,
    )
    generator = jr.normal(
        jr.PRNGKey(823),
        (_NATIVE_BATCH, 3),
        dtype=jnp.float64,
    )

    def forbidden_registration():
        raise AssertionError("CPU-disabled dispatch must not register native targets")

    monkeypatch.setattr(_cpu_horner, "_cpu_backend_enabled", lambda: False)
    monkeypatch.setattr(_cpu_horner, "_register_targets", forbidden_registration)
    native = lambda current, increment: core._fmexp_first_level(
        current, increment, trunc=_TRUNCATION
    )
    portable = lambda current, increment: _portable(core, current, increment)

    _assert_tensor_close(
        native(left, generator), portable(left, generator), atol=0.0, rtol=0.0
    )
    lowered = jax.jit(native).lower(left, generator)
    hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    assert not any(target in hlo for target in _cpu_horner._TARGETS.values())
    _assert_tensor_close(
        lowered.compile()(left, generator),
        jax.jit(portable)(left, generator),
        atol=0.0,
        rtol=0.0,
    )


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


def test_native_path_signature_matches_portable_horner_scan(monkeypatch):
    _native_extension_or_skip()
    monkeypatch.setattr(_cpu_horner, "_REGISTRATION_STATE", None)
    core = _core()
    paths = 0.05 * jr.normal(
        jr.PRNGKey(834), (_NATIVE_BATCH, 5, 3), dtype=jnp.float64
    )

    def signature(values):
        return path_signature(
            values,
            trunc=_TRUNCATION,
            axis=-2,
            accumulate=False,
            parallel=False,
            core=core,
        )

    def portable_signature(values):
        increments = jnp.diff(values, axis=-2)
        neutral = core.development_neutral(
            (increments,), trunc=_TRUNCATION, axis=-2
        )
        terminal, _ = jax.lax.scan(
            lambda carry, increment: (_portable(core, carry, increment), None),
            neutral,
            jnp.moveaxis(increments, -2, 0),
        )
        return terminal

    lowered = jax.jit(signature).lower(paths)
    target = _cpu_horner._TARGETS[np.dtype(jnp.float64)]
    assert target in lowered.compiler_ir(dialect="hlo").as_hlo_text()
    _assert_tensor_close(
        lowered.compile()(paths),
        jax.jit(portable_signature)(paths),
        atol=2e-12,
        rtol=2e-12,
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
