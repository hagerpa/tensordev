"""Focused contract tests for configured ordinary total-degree cores."""

from __future__ import annotations

import numpy as np
import pytest

from tensordev.core.jax import Jax
from tensordev.core.numba import Numba
from tensordev.development.sig import path_signature


CORE_TYPES = (Jax, Numba)


def _dense(core, *, dimension: int = 2, degree: int = 3):
    return tuple(
        core.xp.ones((dimension**grade,), dtype=np.float32) * (grade + 1)
        for grade in range(degree + 1)
    )


def _matrix_dense(core, *, dimension: int = 2, degree: int = 3):
    return tuple(
        core.xp.ones((1, 1, dimension**grade), dtype=np.float32)
        for grade in range(degree + 1)
    )


@pytest.mark.parametrize("core_type", CORE_TYPES)
def test_bounded_configuration_defaults_to_capacity(core_type):
    core = core_type(d=2, max_trunc=4)

    assert core.d == 2
    assert core.max_truncation == 4
    assert core.default_truncation == 4
    assert core.shuffle_plan_store is None


@pytest.mark.parametrize("core_type", CORE_TYPES)
@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    (
        ({"d": True}, TypeError, "d must be a positive integer"),
        ({"d": 0}, ValueError, "d must be positive"),
        ({"max_trunc": True}, TypeError, "max_trunc must be"),
        ({"max_trunc": -1}, ValueError, "max_trunc must be non-negative"),
        (
            {"default_trunc": 1},
            ValueError,
            "default_trunc requires a finite max_trunc",
        ),
        (
            {"max_trunc": 1, "default_trunc": 2},
            ValueError,
            "exceeds core capacity",
        ),
        (
            {"d": 2, "precompute_shuffle": True},
            ValueError,
            "requires both d and max_trunc",
        ),
        (
            {"max_trunc": 2, "precompute_shuffle": True},
            ValueError,
            "requires both d and max_trunc",
        ),
        (
            {"precompute_shuffle": 1},
            TypeError,
            "precompute_shuffle must be a boolean",
        ),
    ),
)
def test_invalid_total_degree_configuration(core_type, kwargs, error, message):
    with pytest.raises(error, match=message):
        core_type(**kwargs)


def test_unbounded_jax_infers_output_depths_from_inputs():
    core = Jax()
    A = _dense(core, degree=2)
    B = _dense(core, degree=1)
    vector = core.xp.ones((2,), dtype=np.float32)

    assert core.max_truncation is None
    assert core.default_truncation is None
    assert len(core.tensor_summation(A, B)) == 3
    assert len(core.tensor_product(A, B)) == 4
    assert len(core.tensor_shuffle_vector(A, vector)) == 3


@pytest.mark.parametrize("core_type", CORE_TYPES)
def test_empty_inner_product_returns_scalar_zero(core_type):
    core = core_type()
    batched = (core.xp.ones((3, 1), dtype=np.int8),)

    result = core.tensor_inner_product(tuple(), batched)

    assert np.shape(result) == ()
    assert np.asarray(result).dtype == np.asarray(core.xp.asarray(0.0)).dtype


@pytest.mark.parametrize("core_type", CORE_TYPES)
def test_bounded_default_caps_all_truncation_aware_operations(core_type):
    core = core_type(d=2, max_trunc=4, default_trunc=2)
    A = _dense(core)
    matrices = _matrix_dense(core)
    vector = core.xp.ones((2,), dtype=np.float32)

    assert len(core.tensor_summation(A, A)) == 3
    assert len(core.tensor_product(A, A)) == 3
    assert len(core.tensor_shuffle_vector(A, vector)) == 2
    assert len(core.tensor_matrix_product(matrices, matrices)) == 3
    assert len(core.tensor_adjoint_product(A, A)) == 3


@pytest.mark.parametrize("core_type", CORE_TYPES)
@pytest.mark.parametrize(
    "operation",
    ("summation", "product", "vector", "matrix", "adjoint"),
)
def test_explicit_truncation_above_capacity_is_rejected(core_type, operation):
    core = core_type(d=2, max_trunc=2)
    A = _dense(core)
    matrices = _matrix_dense(core)
    vector = core.xp.ones((2,), dtype=np.float32)

    calls = {
        "summation": lambda: core.tensor_summation(A, A, trunc=3),
        "product": lambda: core.tensor_product(A, A, trunc=3),
        "vector": lambda: core.tensor_shuffle_vector(A, vector, trunc=3),
        "matrix": lambda: core.tensor_matrix_product(
            matrices, matrices, trunc=3
        ),
        "adjoint": lambda: core.tensor_adjoint_product(A, A, trunc=3),
    }
    with pytest.raises(ValueError, match="exceeds core capacity"):
        calls[operation]()


@pytest.mark.parametrize("core_type", CORE_TYPES)
def test_truncation_views_are_cached_and_share_shuffle_plans(core_type):
    core = core_type(
        d=2,
        max_trunc=4,
        precompute_shuffle=True,
    )
    view = core.at_truncation(2)

    assert view is core.at_truncation(2)
    assert view.at_truncation(4) is core
    assert view.d == core.d
    assert view.max_truncation == core.max_truncation
    assert view.default_truncation == 2
    assert view.shuffle_plan_store is core.shuffle_plan_store
    assert view.memory_bytes() == core.memory_bytes()


def test_jax_truncation_views_share_compiled_wrapper_objects():
    core = Jax(d=2, max_trunc=4, precompute_shuffle=True)
    view = core.at_truncation(2)

    assert view.tensor_product.__func__ is core.tensor_product.__func__
    assert (
        view.tensor_shuffle_product.__func__
        is core.tensor_shuffle_product.__func__
    )


@pytest.mark.parametrize("core_type", CORE_TYPES)
def test_shared_store_infers_capacity_and_accepts_smaller_default(core_type):
    owner = core_type(d=2, max_trunc=4, precompute_shuffle=True)
    view = core_type(
        default_trunc=2,
        shuffle_plan_store=owner.shuffle_plan_store,
    )

    assert view.d == 2
    assert view.max_truncation == 4
    assert view.default_truncation == 2
    assert view.shuffle_plan_store is owner.shuffle_plan_store


@pytest.mark.parametrize("core_type", CORE_TYPES)
def test_full_shuffle_requires_attached_plans(core_type):
    core = core_type(d=2, max_trunc=2)
    A = _dense(core, degree=1)

    with pytest.raises(RuntimeError, match="precompute_shuffle=True"):
        core.tensor_shuffle_product(A, A)


@pytest.mark.parametrize("core_type", CORE_TYPES)
def test_shuffle_enabled_core_computes_full_product(core_type):
    core = core_type(
        d=2,
        max_trunc=2,
        precompute_shuffle=True,
    )
    A = (
        core.xp.asarray([1.0]),
        core.xp.asarray([1.0, 2.0]),
    )
    B = (
        core.xp.asarray([1.0]),
        core.xp.asarray([3.0, 4.0]),
    )

    result = core.tensor_shuffle_product(A, B)

    assert len(result) == 3
    np.testing.assert_allclose(np.asarray(result[0]), [1.0])
    np.testing.assert_allclose(np.asarray(result[1]), [4.0, 6.0])
    np.testing.assert_allclose(np.asarray(result[2]), [6.0, 10.0, 10.0, 16.0])


def test_fixed_dimension_is_validated_at_signature_boundary():
    core = Jax(d=2, max_trunc=2)
    wrong_dimension_path = core.xp.zeros((4, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="configured dimension 2"):
        path_signature(wrong_dimension_path, core=core)
