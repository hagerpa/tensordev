from __future__ import annotations

import jax.numpy as jnp
import pytest

import tensordev as td
from tensordev import _backend
from tensordev.core import (
    Jax,
    JaxBigraded,
    JaxSequentialCore,
    JaxShearBigraded,
    JaxShearTotal,
)


class _DelegatingCore:
    """Small complete core stand-in with a selectable array namespace."""

    def __init__(self, delegate, xp):
        self._delegate = delegate
        self.xp = xp

    def __getattr__(self, name):
        return getattr(self._delegate, name)


class _FactoryCore(_DelegatingCore):
    def __init__(self, delegate, xp, seq_core):
        super().__init__(delegate, xp)
        self._seq_core = seq_core

    def make_sequential_core(self):
        return self._seq_core


class _InvalidFactoryCore(_DelegatingCore):
    def make_sequential_core(self):
        return None


class _MarkerCore(_DelegatingCore):
    def tensor_product(self, *args, **kwargs):
        return "marker product", args, kwargs

    def tensor_exponential(self, *args, **kwargs):
        return "marker exponential", args, kwargs

    def tensor_to_flat(self, *args, **kwargs):
        return "marker flat", args, kwargs


@pytest.fixture(autouse=True)
def _restore_default_core():
    _backend.reset_default_core()
    yield
    _backend.reset_default_core()


def test_environment_default_is_one_coherent_jax_pair():
    core, seq_core = _backend.get_default_core_pair()

    assert isinstance(core, Jax)
    assert isinstance(seq_core, JaxSequentialCore)
    assert _backend.get_default_core() is core
    assert _backend.get_default_seq_core() is seq_core


def test_set_default_core_sets_explicit_pair_atomically():
    initial_core = _backend.get_default_core()
    core = _DelegatingCore(initial_core, xp=object())
    seq_core = object()

    installed = _backend.set_default_core(core, seq_core)

    assert installed is core
    assert _backend.get_default_core_pair() == (core, seq_core)
    assert _backend.get_default_core() is core
    assert _backend.get_default_seq_core() is seq_core


def test_set_default_core_prefers_core_supplied_sequential_core():
    initial_core = _backend.get_default_core()
    seq_core = object()
    core = _FactoryCore(initial_core, xp=object(), seq_core=seq_core)

    _backend.set_default_core(core)

    assert _backend.get_default_core_pair() == (core, seq_core)


def test_set_default_core_infers_sequential_core_from_jax_namespace():
    initial_core, initial_seq_core = _backend.get_default_core_pair()
    composed_jax_core = _DelegatingCore(initial_core, xp=initial_core.xp)

    _backend.set_default_core(composed_jax_core)

    assert _backend.get_default_core() is composed_jax_core
    assert _backend.get_default_seq_core() is initial_seq_core


def test_unresolvable_sequential_core_is_rejected_without_changing_pair():
    initial_pair = _backend.get_default_core_pair()
    unsupported_core = _DelegatingCore(initial_pair[0], xp=object())

    with pytest.raises(TypeError, match="seq_core must be provided"):
        _backend.set_default_core(unsupported_core)

    assert _backend.get_default_core_pair() is initial_pair


def test_malformed_sequential_core_provider_is_rejected():
    initial_pair = _backend.get_default_core_pair()
    invalid_core = _InvalidFactoryCore(initial_pair[0], xp=initial_pair[0].xp)

    with pytest.raises(TypeError, match="make_sequential_core returned None"):
        _backend.set_default_core(invalid_core)

    assert _backend.get_default_core_pair() is initial_pair


def test_callbacks_observe_set_and_reset_and_can_be_unregistered():
    initial_pair = _backend.get_default_core_pair()
    events = []

    unregister = _backend.register_default_core_callback(
        lambda core, seq_core: events.append((core, seq_core)),
        notify=True,
    )
    assert events == [initial_pair]

    custom_core = _DelegatingCore(initial_pair[0], xp=object())
    custom_seq_core = object()
    _backend.set_default_core(custom_core, custom_seq_core)
    assert events[-1] == (custom_core, custom_seq_core)

    _backend.reset_default_core()
    assert events[-1] == initial_pair

    unregister()
    _backend.set_default_core(custom_core, custom_seq_core)
    assert events == [initial_pair, (custom_core, custom_seq_core), initial_pair]


def test_callback_failure_restores_previous_pair():
    initial_pair = _backend.get_default_core_pair()
    custom_core = _DelegatingCore(initial_pair[0], xp=object())
    custom_seq_core = object()
    events = []

    def fail_on_custom(core, seq_core):
        events.append((core, seq_core))
        if core is custom_core:
            raise RuntimeError("cannot rebind")

    unregister = _backend.register_default_core_callback(fail_on_custom)
    try:
        with pytest.raises(RuntimeError, match="cannot rebind"):
            _backend.set_default_core(custom_core, custom_seq_core)
    finally:
        unregister()

    assert _backend.get_default_core_pair() is initial_pair
    assert events == [(custom_core, custom_seq_core), initial_pair]


def test_set_default_core_rejects_none():
    initial_pair = _backend.get_default_core_pair()

    with pytest.raises(TypeError, match="core must not be None"):
        _backend.set_default_core(None)

    assert _backend.get_default_core_pair() is initial_pair


def test_set_default_core_constructs_total_degree_from_integer_dims():
    core = td.set_default_core(
        dims=2,
        max_trunc=4,
        default_trunc=2,
        precompute_shuffle=True,
    )

    assert isinstance(core, Jax)
    assert core is td.get_default_core()
    assert core.d == 2
    assert core.max_truncation == 4
    assert core.default_truncation == 2
    assert core.shuffle_plan_store is not None
    assert td.tensor_product.__self__ is core


def test_set_default_core_constructs_bidegree_from_tuple_dims():
    core = td.set_default_core(
        dims=(1, 2),
        max_trunc=(2, 1),
        default_trunc=(1, 1),
    )

    assert isinstance(core, JaxBigraded)
    assert core is td.get_default_core()
    assert core.grading == "bidegree"
    assert core.dims == (1, 2)
    assert core.max_truncation == (2, 1)
    assert core.default_truncation == (1, 1)
    assert td.tensor_product.__self__ is core


def test_set_default_core_constructs_generator_shuffle_bidegree():
    core = td.set_default_core(
        dims=(1, 2),
        max_trunc=(3, 2),
        precompute_shuffle="generator",
    )

    assert core is td.get_default_core()
    assert core.shuffle_plan_store.scope == "generator"
    assert core.supports("shuffle")
    assert not core.supports("shuffle_product")


def test_set_default_core_constructs_dense_total_shear_from_pair_and_integer():
    core = td.set_default_core(
        dims=(1, 2),
        max_trunc=3,
        default_trunc=2,
        coordinates="shear",
        precompute_shuffle="generator",
    )

    assert isinstance(core, JaxShearTotal)
    assert core is td.get_default_core()
    assert core.grading == "total_degree"
    assert core.coordinates == "shear"
    assert core.dims == (1, 2)
    assert core.max_truncation == 3
    assert core.default_truncation == 2
    assert core.supports("shuffle")
    assert not core.supports("shuffle_product")
    assert td.tensor_product.__self__ is core


def test_set_default_core_constructs_bidegree_shear_from_two_pairs():
    core = td.set_default_core(
        dims=(1, 2),
        max_trunc=(2, 1),
        default_trunc=(1, 1),
        coordinates="shear",
    )

    assert isinstance(core, JaxShearBigraded)
    assert core is td.get_default_core()
    assert core.grading == "bidegree"
    assert core.coordinates == "shear"
    assert core.dims == (1, 2)
    assert core.max_truncation == (2, 1)
    assert core.default_truncation == (1, 1)
    assert td.tensor_product.__self__ is core


@pytest.mark.parametrize(
    "kwargs",
    (
        {"dims": 2, "max_trunc": (1, 1)},
        {"dims": (1, 1), "max_trunc": 2},
        {"dims": 2, "max_trunc": 2, "coordinates": "shear"},
    ),
)
def test_set_default_core_rejects_unsupported_dispatch_combinations_atomically(
    kwargs,
):
    initial_pair = td.get_default_core_pair()

    with pytest.raises(ValueError, match="Accepted combinations"):
        td.set_default_core(**kwargs)

    assert td.get_default_core_pair() is initial_pair


def test_set_default_core_rejects_unknown_coordinates_atomically():
    initial_pair = td.get_default_core_pair()

    with pytest.raises(ValueError, match="either 'standard' or 'shear'"):
        td.set_default_core(
            dims=(1, 1),
            max_trunc=2,
            coordinates="other",
        )

    assert td.get_default_core_pair() is initial_pair


def test_failed_shear_core_construction_does_not_change_default_pair():
    initial_pair = td.get_default_core_pair()

    with pytest.raises(ValueError, match="exceeds core capacity"):
        td.set_default_core(
            dims=(1, 1),
            max_trunc=2,
            default_trunc=3,
            coordinates="shear",
        )

    assert td.get_default_core_pair() is initial_pair


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({}, "constructed core or dims"),
        ({"dims": 2}, "max_trunc is required"),
        ({"dims": True, "max_trunc": 2}, "positive integer or a bidegree tuple"),
        ({"dims": [1, 1], "max_trunc": (1, 1)}, "positive integer or a bidegree tuple"),
    ),
)
def test_set_default_core_construction_validation_is_atomic(kwargs, message):
    initial_pair = td.get_default_core_pair()

    with pytest.raises(TypeError, match=message):
        td.set_default_core(**kwargs)

    assert td.get_default_core_pair() is initial_pair


def test_set_default_core_rejects_mixing_core_and_factory_arguments():
    initial_pair = td.get_default_core_pair()

    with pytest.raises(TypeError, match="cannot be combined"):
        td.set_default_core(initial_pair[0], dims=2, max_trunc=3)

    assert td.get_default_core_pair() is initial_pair


def test_public_module_operations_rebind_directly_and_reset():
    initial_core, initial_seq_core = td.get_default_core_pair()
    captured_product = td.tensor_product
    custom_core = _MarkerCore(initial_core, xp=initial_core.xp)

    td.set_default_core(custom_core)

    assert td.get_default_core() is custom_core
    assert td.get_default_seq_core() is initial_seq_core
    assert td.tensor_product.__self__ is custom_core
    assert td.tensor_exponential.__self__ is custom_core
    assert td.tensor_flatten.__self__ is custom_core
    assert td.tensor_product(1, 2, trunc=3) == (
        "marker product",
        (1, 2),
        {"trunc": 3},
    )

    # A directly imported alias is an ordinary bound method.  Rebinding the
    # package attribute deliberately cannot mutate an already captured object.
    assert captured_product is not td.tensor_product
    assert captured_product.__self__ is initial_core

    td.reset_default_core()

    assert td.get_default_core() is initial_core
    assert td.get_default_seq_core() is initial_seq_core
    assert td.tensor_product.__self__ is initial_core
    assert captured_product.__self__ is initial_core


def test_public_default_api_and_forwarded_operations_are_exported():
    expected = {
        "get_default_core",
        "get_default_core_pair",
        "get_default_seq_core",
        "set_default_core",
        "reset_default_core",
        "tensor_product",
        "tensor_shuffle_product",
        "tensor_exponential",
        "tensor_shear_inner_product",
        "tensor_flatten",
        "shear_core",
        "total_degree_core",
        "core_expected_memory",
        "tensor_from_standard_coordinates",
        "tensor_to_standard_coordinates",
    }

    assert expected <= set(td.__all__)
    assert "JaxSequentialCoreFreeDevelopment" not in td.__all__


def test_shear_pairing_is_rebound_with_the_default_core():
    standard = td.total_degree_core(d=2, max_trunc=1)
    core = td.shear_core(standard, dims=(1, 1))
    words = (
        jnp.ones((1,), dtype=jnp.float32),
        jnp.asarray([0.25, -0.5], dtype=jnp.float32),
    )

    td.set_default_core(core)

    assert td.tensor_shear_inner_product.__self__ is core
    assert td.tensor_shear_inner_product(words, words) == (
        core.tensor_shear_inner_product(words, words)
    )


def test_bounded_total_core_drives_background_operations_and_signatures():
    core = td.set_default_core(
        dims=2,
        max_trunc=4,
        default_trunc=2,
        precompute_shuffle=True,
    )

    A = tuple(jnp.ones((2**degree,)) for degree in range(4))
    X = jnp.zeros((5, 2))

    assert td.get_default_core() is core
    assert td.tensor_product.__self__ is core
    assert td.tensor_shuffle_product.__self__ is core
    assert len(td.tensor_product(A, A)) == 3
    assert len(td.tensor_shuffle_product(A, A)) == 3
    assert len(td.path_signature(X)) == 3


def test_shear_coordinate_conversions_are_bound_through_root_api():
    core = td.set_default_core(
        dims=(1, 1),
        max_trunc=2,
        coordinates="shear",
    )
    standard = (
        jnp.asarray([1.0]),
        jnp.arange(2, dtype=jnp.float32),
        jnp.arange(4, dtype=jnp.float32),
    )

    shear = td.tensor_from_standard_coordinates(standard)
    restored = td.tensor_to_standard_coordinates(shear)

    assert td.tensor_from_standard_coordinates.__self__ is core
    assert td.tensor_to_standard_coordinates.__self__ is core
    assert all(
        jnp.allclose(actual, expected)
        for actual, expected in zip(restored, standard)
    )


def test_background_total_shear_signature_matches_standard_signature():
    X = jnp.asarray(
        [
            [0.0, 0.0],
            [0.2, -0.1],
            [0.5, 0.3],
            [0.1, 0.4],
        ],
        dtype=jnp.float32,
    )
    standard_core = td.total_degree_core(
        d=2,
        max_trunc=3,
        default_trunc=3,
    )
    expected = td.path_signature(X, core=standard_core)

    core = td.set_default_core(
        dims=(1, 1),
        max_trunc=3,
        default_trunc=3,
        coordinates="shear",
    )
    actual = td.path_signature(X)
    restored = td.tensor_to_standard_coordinates(actual)

    assert td.get_default_core() is core
    assert all(
        jnp.allclose(actual_level, expected_level, atol=1e-6, rtol=1e-6)
        for actual_level, expected_level in zip(restored, expected)
    )


def test_background_bidegree_shear_signature_matches_standard_signature():
    X = jnp.asarray(
        [
            [0.0, 0.0],
            [0.2, -0.1],
            [0.5, 0.3],
            [0.1, 0.4],
        ],
        dtype=jnp.float32,
    )
    standard_core = td.bigraded_core(
        dims=(1, 1),
        max_trunc=(2, 2),
        default_trunc=(2, 1),
    )
    expected = td.path_signature(X, core=standard_core)

    core = td.set_default_core(
        dims=(1, 1),
        max_trunc=(2, 2),
        default_trunc=(2, 1),
        coordinates="shear",
    )
    actual = td.path_signature(X)
    restored = td.tensor_to_standard_coordinates(actual)

    assert td.get_default_core() is core
    assert restored.grades == expected.grades
    assert all(
        jnp.allclose(restored[grade], expected[grade], atol=1e-6, rtol=1e-6)
        for grade in restored.grades
    )


@pytest.mark.parametrize(
    ("dimension", "truncation"),
    ((1, 8), (2, 8), (3, 8)),
)
def test_shuffle_memory_estimator_matches_allocated_plan_payload(
    dimension,
    truncation,
):
    core = td.total_degree_core(
        d=dimension,
        max_trunc=truncation,
        precompute_shuffle=True,
    )

    expected_bytes = core.memory_bytes()
    for unit, divisor in (
        ("bytes", 1),
        ("KiB", 1024),
        ("MiB", 1024**2),
        ("GiB", 1024**3),
    ):
        assert td.core_expected_memory(
            dims=dimension,
            max_trunc=truncation,
            unit=unit,
            precompute_shuffle=True,
        ) == expected_bytes / divisor
        breakdown = td.core_expected_memory(
            dims=dimension,
            max_trunc=truncation,
            unit=unit,
            precompute_shuffle=True,
            breakdown=True,
        )
        assert breakdown["total"] == expected_bytes / divisor
        assert {
            name: value / divisor
            for name, value in core.memory_bytes_by_category().items()
        } == {name: value for name, value in breakdown.items() if name != "total"}
    assert td.core_expected_memory(
        dims=dimension,
        max_trunc=truncation,
        unit="bytes",
    ) == 0.0
    assert td.core_expected_memory(
        dims=dimension,
        max_trunc=truncation,
        unit="bytes",
        breakdown=True,
    ) == {"total": 0.0}
    assert expected_bytes == core.memory_bytes()


def test_separate_shuffle_core_api_is_removed():
    assert "shuffle_core" not in td.__all__
    assert "shuffle_core_expected_memory" not in td.__all__
    assert not hasattr(td, "shuffle_core")


def test_individual_memory_estimator_apis_are_removed():
    removed = {
        "total_degree_core_expected_memory",
        "total_degree_core_expected_memory_breakdown",
        "bigraded_core_expected_memory",
        "bigraded_core_expected_memory_breakdown",
    }

    assert removed.isdisjoint(td.__all__)
    assert all(not hasattr(td, name) for name in removed)
    assert not hasattr(Jax, "expected_memory_bytes")
    assert not hasattr(Jax, "expected_memory_bytes_by_category")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {"dims": True, "max_trunc": 2, "unit": "MiB"},
            "positive integer or a bidegree tuple",
        ),
        (
            {"dims": [1, 1], "max_trunc": (1, 1), "unit": "MiB"},
            "positive integer or a bidegree tuple",
        ),
        ({"dims": (1,), "max_trunc": (1, 1), "unit": "MiB"}, "dims must be a pair"),
        (
            {"dims": (1, 1, 1), "max_trunc": (1, 1), "unit": "MiB"},
            "dims must be a pair",
        ),
        (
            {"dims": 2, "max_trunc": 2, "unit": "MiB", "precompute_shuffle": 1},
            "must be a boolean",
        ),
        (
            {
                "dims": 2,
                "max_trunc": 2,
                "unit": "MiB",
                "precompute_shuffle": "generator",
            },
            "must be a boolean",
        ),
        (
            {"dims": 2, "max_trunc": 2, "unit": "MiB", "breakdown": 1},
            "must be a boolean",
        ),
    ),
)
def test_unified_memory_estimator_validates_dispatch_arguments(kwargs, message):
    with pytest.raises(TypeError, match=message):
        td.core_expected_memory(**kwargs)


def test_unified_memory_estimator_defaults_to_mib_and_validates_unit():
    assert td.core_expected_memory(
        dims=(1, 1),
        max_trunc=(1, 1),
    ) == td.core_expected_memory(
        dims=(1, 1),
        max_trunc=(1, 1),
        unit="MiB",
    )
    with pytest.raises(TypeError, match="unit must be a string"):
        td.core_expected_memory(dims=2, max_trunc=2, unit=1)
    with pytest.raises(ValueError, match="exactly one of"):
        td.core_expected_memory(dims=2, max_trunc=2, unit="MB")


@pytest.mark.parametrize(
    ("feature", "call"),
    [
        ("free_kernel", lambda path: td.free_kernel(path, path)),
        (
            "higher_order_kernel",
            lambda path: td.higher_order_kernel(
                path,
                path,
                log_steps=(1, 1),
                log_degree=(1, 1),
            ),
        ),
        (
            "fssk_state",
            lambda path: td.fssk_state(path, kernel=None, dt=1.0, trunc=1),
        ),
        (
            "fssk_vsig",
            lambda path: td.fssk_vsig(path, kernel=None, dt=1.0, trunc=1),
        ),
    ],
)
def test_total_degree_specific_entry_points_reject_bidegree_default(feature, call):
    core = td.bigraded_core(
        dims=(1, 1),
        max_trunc=(1, 1),
    )
    td.set_default_core(core)

    path = jnp.zeros((3, 2))
    with pytest.raises(RuntimeError, match=rf"{feature} is total-degree-specific"):
        call(path)


@pytest.mark.parametrize("precompute_shuffle", [False, "generator", True])
def test_bigraded_memory_estimator_matches_allocated_plan_payload(
    precompute_shuffle,
):
    dims = (1, 2)
    capacity = (2, 1)
    core = td.bigraded_core(
        dims=dims,
        max_trunc=capacity,
        precompute_shuffle=precompute_shuffle,
    )

    expected = td.core_expected_memory(
        dims=dims,
        max_trunc=capacity,
        unit="MiB",
        precompute_shuffle=precompute_shuffle,
    )
    assert expected == core.memory_mb()

    breakdown = td.core_expected_memory(
        dims=dims,
        max_trunc=capacity,
        unit="MiB",
        precompute_shuffle=precompute_shuffle,
        breakdown=True,
    )
    actual_categories = {
        name: value / 1024**2
        for name, value in core.memory_bytes_by_category().items()
    }

    assert breakdown["total"] == expected
    assert {
        name: value
        for name, value in breakdown.items()
        if name != "total"
    } == actual_categories


@pytest.mark.parametrize("grading", ["total_degree", "bidegree"])
@pytest.mark.parametrize("precompute_shuffle", [False, "generator", True])
def test_shear_memory_estimator_matches_allocated_plan_payload_without_rebinding(
    grading,
    precompute_shuffle,
):
    kwargs = {
        "dims": (1, 1),
        "max_trunc": 2 if grading == "total_degree" else (1, 1),
        "coordinates": "shear",
        "precompute_shuffle": precompute_shuffle,
    }
    core = td.set_default_core(**kwargs)
    installed_pair = td.get_default_core_pair()

    expected = td.core_expected_memory(**kwargs, unit="bytes")
    breakdown = td.core_expected_memory(
        **kwargs,
        unit="bytes",
        breakdown=True,
    )

    assert expected == core.memory_bytes()
    assert breakdown["total"] == expected
    assert {
        name: value
        for name, value in breakdown.items()
        if name != "total"
    } == core.memory_bytes_by_category()
    assert td.get_default_core_pair() is installed_pair


@pytest.mark.parametrize("capacity", [(1, 8), (6, 3)])
def test_bigraded_shear_memory_estimator_matches_counted_payload(capacity):
    kwargs = {
        "dims": (1, 1),
        "max_trunc": capacity,
        "coordinates": "shear",
        "precompute_shuffle": False,
    }
    core = JaxShearBigraded(
        dims=kwargs["dims"],
        max_trunc=kwargs["max_trunc"],
        precompute_shuffle=False,
    )

    assert td.core_expected_memory(**kwargs, unit="bytes") == core.memory_bytes()
    assert td.core_expected_memory(
        **kwargs,
        unit="bytes",
        breakdown=True,
    ) == {
        **{
            name: float(value)
            for name, value in core.memory_bytes_by_category().items()
        },
        "total": float(core.memory_bytes()),
    }


def test_large_bigraded_shear_memory_estimate_avoids_support_expansion(
    monkeypatch,
):
    from math import comb

    from tensordev.core.shear import bigraded as shear_bigraded

    original = shear_bigraded._transform_rank_groups

    def reject_large_expansion(grade, *, inverse):
        if comb(sum(grade), grade[0]) > 64:
            raise AssertionError("large symbolic support was expanded")
        return original(grade, inverse=inverse)

    monkeypatch.setattr(
        shear_bigraded,
        "_transform_rank_groups",
        reject_large_expansion,
    )
    kwargs = {
        "dims": (1, 1),
        "max_trunc": (10, 4),
        "coordinates": "shear",
        "precompute_shuffle": False,
    }

    breakdown = td.core_expected_memory(
        **kwargs,
        unit="bytes",
        breakdown=True,
    )

    assert breakdown == {
        "placements": 142692.0,
        "conversion": 17468.0,
        "concatenation": 73964.0,
        "shear_transform_rank_pairs": 11803189.0,
        "shear_transform_dense_permutations": 39480.0,
        "shear_transform_parities": 4367.0,
        "shear_transform_rank_matrices": 14400.0,
        "shear_generator_rank_pairs": 19727.0,
        "shear_generator_dense_permutations": 14680.0,
        "total": 12129967.0,
    }
    assert not any(name.startswith("shuffle_") for name in breakdown)
    assert td.core_expected_memory(**kwargs, unit="GiB") == (
        12129967 / 1024**3
    )


def test_shear_memory_estimator_does_not_construct_cores_or_plan_stores(
    monkeypatch,
):
    from tensordev.core.shear import bigraded as shear_bigraded
    from tensordev.core.shear import total as shear_total

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("memory estimation must not construct plans")

    monkeypatch.setattr(_backend._CoreConfiguration, "construct", forbidden)
    monkeypatch.setattr(shear_total.TotalShearPlanStore, "__init__", forbidden)
    monkeypatch.setattr(
        shear_bigraded.BigradedPlanStore, "__init__", forbidden
    )
    monkeypatch.setattr(
        shear_bigraded.BigradedShearPlanStore, "__init__", forbidden
    )
    monkeypatch.setattr(
        shear_bigraded.BigradedShearShufflePlanStore,
        "__init__",
        forbidden,
    )

    assert td.core_expected_memory(
        dims=(1, 1),
        max_trunc=3,
        coordinates="shear",
        precompute_shuffle=True,
    ) > 0
    assert td.core_expected_memory(
        dims=(1, 1),
        max_trunc=(2, 2),
        coordinates="shear",
        precompute_shuffle=True,
    ) > 0


def test_shear_construction_and_estimation_release_symbolic_support_caches():
    from tensordev.core.shear.symbolic import (
        clear_symbolic_caches,
        psi_support,
        symbolic_cache_entry_count,
    )

    clear_symbolic_caches()
    psi_support(2, 2)
    assert symbolic_cache_entry_count() > 0
    clear_symbolic_caches()

    JaxShearTotal(
        dims=(1, 1),
        max_trunc=3,
        precompute_shuffle=True,
    )
    assert symbolic_cache_entry_count() == 0
    JaxShearBigraded(
        dims=(1, 1),
        max_trunc=(2, 2),
        precompute_shuffle=True,
    )
    assert symbolic_cache_entry_count() == 0

    td.core_expected_memory(
        dims=(1, 1),
        max_trunc=3,
        coordinates="shear",
        precompute_shuffle=True,
    )
    assert symbolic_cache_entry_count() == 0
    td.core_expected_memory(
        dims=(1, 1),
        max_trunc=(2, 2),
        coordinates="shear",
        precompute_shuffle=True,
    )
    assert symbolic_cache_entry_count() == 0


def test_shear_plan_statistics_report_compile_and_memory_structure():
    total = JaxShearTotal(
        dims=(2, 2),
        max_trunc=4,
        precompute_shuffle="generator",
    ).plan_statistics()

    assert set(total["plan_counts"]) == {
        "forward",
        "inverse",
        "generator",
        "shuffle",
    }
    assert set(total["term_counts"]) == set(total["plan_counts"])
    assert set(total["permutation_counts"]) == set(total["plan_counts"])
    assert total["shuffle_terms"] == total["term_counts"]["shuffle"]
    assert "gamma_terms" not in total
    assert sum(total["strategy_counts"].values()) == sum(
        total["execution_group_counts"].values()
    )
    assert (
        total["authoritative_memory_bytes"]
        + total["derived_execution_memory_bytes"]
        == total["memory_bytes"]
    )

    bigraded = JaxShearBigraded(
        dims=(1, 1),
        max_trunc=(2, 2),
        precompute_shuffle=True,
    ).plan_statistics()
    mandatory = bigraded["shear_plan_statistics"]
    shuffle = bigraded["shuffle_plan_statistics"]

    assert set(mandatory["plan_counts"]) == {
        "forward",
        "inverse",
        "generator",
    }
    assert set(mandatory["term_counts"]) == set(mandatory["plan_counts"])
    assert set(mandatory["permutation_group_counts"]) == set(
        mandatory["plan_counts"]
    )
    assert shuffle["block_plan_count"] > 0
    assert shuffle["term_count"] > 0
    assert shuffle["permutation_group_count"] > 0
    assert (
        bigraded["authoritative_memory_bytes"]
        + bigraded["derived_execution_memory_bytes"]
        == bigraded["memory_bytes"]
    )


def test_bigraded_factory_and_estimator_exclude_shuffle_by_default():
    dims = (1, 1)
    capacity = (2, 2)
    core = td.bigraded_core(dims=dims, max_trunc=capacity)

    assert core.shuffle_plan_store is None
    assert td.core_expected_memory(
        dims=dims,
        max_trunc=capacity,
        unit="MiB",
    ) == core.memory_mb()


def test_bigraded_factory_validates_capacity_and_supports_scalar_only_core():
    with pytest.raises(ValueError, match="strictly positive"):
        td.bigraded_core(dims=(0, 1), max_trunc=(1, 1))
    with pytest.raises(TypeError, match="dims must be a pair"):
        td.bigraded_core(dims=(1,), max_trunc=(1, 1))
    with pytest.raises(ValueError, match="non-negative"):
        td.bigraded_core(dims=(1, 1), max_trunc=(1, -1))
    with pytest.raises(ValueError, match="exceeds core capacity"):
        td.bigraded_core(
            dims=(1, 1),
            max_trunc=(1, 1),
            default_trunc=(2, 1),
        )

    scalar_core = td.bigraded_core(dims=(1, 1), max_trunc=(0, 0))
    identity = scalar_core.tensor_exponential(tuple(), trunc=(0, 0))
    assert identity.grades == ((0, 0),)
    assert jnp.array_equal(identity[0, 0], jnp.ones((1,), dtype=identity[0, 0].dtype))
