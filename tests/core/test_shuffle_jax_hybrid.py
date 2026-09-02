"""Focused coverage for the compile-aware JAX shuffle kernels."""

from collections import Counter
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
from tensordev.core.jax import Jax
from tensordev.core.shuffle import TotalDegreeShufflePlanStore


@pytest.fixture(scope="module")
def d2_core():
    return Jax(d=2, max_trunc=8, precompute_shuffle=True)


@pytest.fixture(scope="module")
def d3_core():
    return Jax(d=3, max_trunc=8, precompute_shuffle=True)


def _integer_inputs(dimension, left_degree, right_degree):
    left = (np.arange(2 * dimension**left_degree) % 7).reshape(
        2, 1, dimension**left_degree
    )
    right = (np.arange(3 * dimension**right_degree) % 5).reshape(
        1, 3, dimension**right_degree
    )
    return left.astype(np.int32), right.astype(np.int32)


def test_strategy_selection_covers_all_execution_paths(d2_core, d3_core):
    d2_store = d2_core.shuffle_plan_store
    d3_store = d3_core.shuffle_plan_store
    assert d2_store.STATIC_PERMUTATION_THRESHOLD == 24
    assert d2_store.FLAT_GATHER_CHUNK_SIZE == 24
    assert d3_store.COEFFICIENT_GATHER_CHUNK_SIZE == 8
    assert d2_core.plan_strategy(4, 0) == "direct"
    assert d2_core.plan_strategy(3, 3) == "transpose"
    assert d2_core.plan_strategy(4, 4) == "flat_gather"
    assert d3_core.plan_strategy(4, 4) == "coefficient_gather"

    assert (
        d2_store._runtime_plans[(4, 4)].flat_indices.shape[-2]
        == d2_store.FLAT_GATHER_CHUNK_SIZE
    )
    assert (
        d3_store._runtime_plans[(4, 4)].coefficients.shape[-2]
        == d3_store.COEFFICIENT_GATHER_CHUNK_SIZE
    )


@pytest.mark.parametrize(
    ("dimension", "left_degree", "right_degree"),
    ((1, 4, 4), (2, 3, 3), (2, 4, 4), (3, 4, 4)),
)
def test_each_strategy_matches_backend_neutral_permutations(
    dimension,
    left_degree,
    right_degree,
):
    jax_core = Jax(
        d=dimension,
        max_trunc=left_degree + right_degree,
        precompute_shuffle=True,
    )
    numpy_store = TotalDegreeShufflePlanStore(
        dimension,
        left_degree + right_degree,
    )
    left, right = _integer_inputs(dimension, left_degree, right_degree)

    actual = jax_core.permutation_einsum(
        jnp.asarray(left),
        jnp.asarray(right),
        left_degree,
        right_degree,
    )
    expected = numpy_store.apply(
        np,
        left,
        right,
        left_degree,
        right_degree,
    )

    assert isinstance(actual, jax.Array)
    assert actual.shape == (2, 3, dimension ** (left_degree + right_degree))
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_flat_maps_obey_cap_and_derived_memory_is_reported(d2_core):
    store = d2_core.shuffle_plan_store
    categories = d2_core.memory_bytes_by_category()
    flat_plans = tuple(
        plan
        for plan in store._runtime_plans.values()
        if plan.strategy == "flat_gather"
    )

    assert flat_plans
    assert all(
        plan.flat_indices.nbytes <= store.FLAT_GATHER_MAX_BYTES
        for plan in flat_plans
    )
    assert categories["derived_execution_maps"] == sum(
        plan.flat_indices.nbytes for plan in flat_plans
    )
    assert d2_core.memory_bytes() == sum(categories.values())
    assert d2_core.plan_statistics()["memory_bytes"] == d2_core.memory_bytes()


@pytest.mark.parametrize(("dimension", "truncation"), ((1, 8), (2, 8), (3, 8)))
def test_expected_memory_matches_allocated_buffers(dimension, truncation):
    core = Jax(
        d=dimension,
        max_trunc=truncation,
        precompute_shuffle=True,
    )

    expected = td.core_expected_memory(
        dims=dimension,
        max_trunc=truncation,
        unit="MiB",
        precompute_shuffle=True,
        breakdown=True,
    )
    assert {
        name: value / 1024**2
        for name, value in core.memory_bytes_by_category().items()
    } == {name: value for name, value in expected.items() if name != "total"}
    assert expected["total"] == core.memory_mb()


@pytest.mark.parametrize(
    ("dimension", "truncation", "error"),
    ((True, 1, TypeError), (0, 1, ValueError), (2, -1, ValueError)),
)
def test_expected_memory_validates_like_core(dimension, truncation, error):
    with pytest.raises(error):
        td.core_expected_memory(
            dims=dimension,
            max_trunc=truncation,
            unit="MiB",
            precompute_shuffle=True,
        )


def test_large_maps_stay_compact_as_coefficient_chunks(d3_core):
    store = d3_core.shuffle_plan_store
    runtime = store._runtime_plans[(4, 4)]
    plan = store.plans[(4, 4)]
    unmaterialized_map_bytes = (
        plan.permutation_count * plan.output_width * np.dtype(np.int32).itemsize
    )

    assert runtime.strategy == "coefficient_gather"
    assert runtime.flat_indices is None
    assert runtime.coefficients is not None
    assert runtime.memory_bytes() < unmaterialized_map_bytes // 100
    assert d3_core.plan_statistics()["coefficient_gather_plan_count"] > 0


def test_flat_map_uses_chunked_execution_for_large_batches(d2_core):
    left = (np.arange(8 * 16) % 7).reshape(8, 1, 16).astype(np.int32)
    right = (np.arange(8 * 16) % 5).reshape(1, 8, 16).astype(np.int32)
    numpy_store = TotalDegreeShufflePlanStore(2, 8)
    store = d2_core.shuffle_plan_store
    runtime = store._runtime_plans[(4, 4)]
    assert (
        64 * runtime.flat_indices.size * np.dtype(np.int32).itemsize
        > store.FULL_GATHER_MAX_TEMPORARY_BYTES
    )

    actual = d2_core.permutation_einsum(
        jnp.asarray(left),
        jnp.asarray(right),
        4,
        4,
    )
    expected = numpy_store.apply(np, left, right, 4, 4)

    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_scanned_gather_is_differentiable(d2_core):
    left = jnp.arange(16.0).reshape(1, 16)
    right = jnp.arange(16.0, 32.0).reshape(1, 16)
    permutation_count = (
        d2_core.shuffle_plan_store.plans[(4, 4)].permutation_count
    )

    def total(left_):
        return jnp.sum(d2_core.permutation_einsum(left_, right, 4, 4))

    gradient = jax.grad(total)(left)
    expected = jnp.full_like(
        left,
        permutation_count * jnp.sum(right),
    )
    np.testing.assert_allclose(np.asarray(gradient), np.asarray(expected))


@pytest.mark.parametrize(
    ("degree", "small_capacity", "large_capacity"),
    ((3, 6, 8), (4, 8, 10)),
)
def test_active_shuffle_program_does_not_capture_capacity_only_plans(
    degree,
    small_capacity,
    large_capacity,
):
    left = jnp.ones((2**degree,), dtype=jnp.float32)
    right = jnp.ones((2**degree,), dtype=jnp.float32)

    def inventory(capacity):
        core = Jax(d=2, max_trunc=capacity, precompute_shuffle=True)
        lowered = jax.jit(
            lambda a, b: core.permutation_einsum(a, b, degree, degree)
        ).lower(left, right)
        text = str(lowered.compiler_ir(dialect="stablehlo"))
        return Counter(re.findall(r"stablehlo\.([a-zA-Z0-9_]+)", text))

    assert inventory(small_capacity) == inventory(large_capacity)
