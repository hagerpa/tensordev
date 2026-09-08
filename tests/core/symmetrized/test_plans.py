from math import comb

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quotient_word_oracle import NormalForm, concatenate_forms, sym_rank
from tensordev.core.bigraded.symmetrized._precompute import (
    _coefficient_dtype,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
    _expected_plan_memory_bytes_by_category,
    apply_doubleprime_generator_block,
    apply_doubleprime_generator_prefix,
)
from tensordev.core.utils.precompute import _unsigned_index_dtype
from tensordev.core.utils.segmented import apply_segmented_rank_plan


def _scatter_add(output, targets, values):
    return output.at[..., targets, :].add(values)


def _form(row, prime_degree):
    return NormalForm(
        blocks=tuple(tuple(map(int, block)) for block in row),
        primes=("p",) * prime_degree,
    )


def test_grade_plans_match_quotient_widths_and_are_immutable():
    store = PartiallySymmetrizedPlanStore((3, 2), (2, 3))

    for (n, m), plan in store.grade_plans.items():
        parts = (n + 1) * 2
        rank_count = comb(m + parts - 1, m)
        assert plan.rank_count == rank_count
        assert plan.block_width == rank_count * 3**n
        assert plan.dense_shape == (rank_count, 3**n)
        assert plan.placements.shape == (rank_count, n + 1, 2)
        assert not plan.placements.flags.writeable
        np.testing.assert_array_equal(
            plan.placements.sum(axis=(1, 2)),
            np.full(rank_count, m),
        )

    with pytest.raises(TypeError):
        store.grade_plans[(0, 0)] = store.grade_plan((0, 0))


def test_active_layouts_share_capacity_plans_and_preserve_partial_symmetrization():
    store = PartiallySymmetrizedPlanStore((2, 2), (3, 2))
    layout = store.resolve(
        (2, 1),
        include_scalar=False,
        coordinates="shear",
    )

    assert layout is store.resolve(
        (2, 1),
        include_scalar=False,
        coordinates="shear",
    )
    assert layout.store is store
    assert layout.partially_symmetrized is True
    assert layout.coordinates == "shear"
    assert not layout.include_scalar
    assert all(
        layout.grade_plan(grade) is store.grade_plan(grade)
        for grade in layout.grades
    )

    scalar_layout = layout.with_scalar(True)
    assert scalar_layout.store is store
    assert scalar_layout.coordinates == "shear"
    assert scalar_layout.partially_symmetrized is True
    assert scalar_layout.include_scalar


@pytest.mark.parametrize(
    ("left_grade", "right_grade"),
    (
        ((0, 0), (1, 2)),
        ((0, 1), (0, 1)),
        ((1, 1), (1, 1)),
        ((2, 0), (0, 2)),
    ),
)
def test_concat_rank_maps_match_independent_normal_form_oracle(
    left_grade,
    right_grade,
):
    store = PartiallySymmetrizedPlanStore((2, 2), (2, 2))
    plan = store.concat_plan(left_grade, right_grade)
    left = store.grade_plan(left_grade)
    right = store.grade_plan(right_grade)

    expected = np.empty(
        (left.rank_count, right.rank_count),
        dtype=np.int64,
    )
    for left_rank, left_row in enumerate(left.placements):
        for right_rank, right_row in enumerate(right.placements):
            output = concatenate_forms(
                _form(left_row, left_grade[0]),
                _form(right_row, right_grade[0]),
            )
            expected[left_rank, right_rank] = sym_rank(output)

    np.testing.assert_array_equal(plan.target_ranks(), expected)
    assert not plan.rank_plan.target_ranks.flags.writeable
    assert plan.rank_plan.source_count == left.rank_count * right.rank_count


def test_concat_segmented_plan_accumulates_required_collisions_under_jit():
    store = PartiallySymmetrizedPlanStore((1, 2), (0, 2))
    plan = store.concat_plan((0, 1), (0, 1))
    targets = np.asarray(plan.rank_plan.target_ranks)
    assert len(np.unique(targets)) < targets.size

    values = jnp.arange(1, targets.size + 1, dtype=jnp.float32).reshape(
        targets.size,
        1,
    )

    def apply(current):
        return apply_segmented_rank_plan(
            jnp,
            current,
            plan.rank_plan,
            scatter_add=_scatter_add,
        )

    expected = np.zeros((plan.output_rank_count, 1), dtype=np.float32)
    for source, target in enumerate(targets):
        expected[target] += np.asarray(values[source])
    np.testing.assert_allclose(jax.jit(apply)(values), expected)


def test_concat_prime_axis_permutation_is_factored_and_static():
    plan = PartiallySymmetrizedPlanStore((3, 2), (3, 2)).concat_plan(
        (2, 1),
        (1, 1),
    )
    # raw expanded axes: rank1, p1, p1, rank2, p2
    assert plan.outer_axis_permutation == (0, 3, 1, 2, 4)
    assert plan.dense_axis_permutation == (0, 1, 2)
    assert plan.dense_input_shape == (3, 3, 3)
    assert plan.dense_output_shape == (3, 3, 3)


def _doubleprime_target_oracle(store, plan):
    source = store.grade_plan(plan.source_grade)
    expected = np.empty(
        (plan.d_doubleprime, plan.source_rank_count),
        dtype=np.int64,
    )
    for letter in range(plan.d_doubleprime):
        for source_rank, row in enumerate(source.placements):
            blocks = [list(map(int, block)) for block in row]
            blocks[-1][letter] += 1
            expected[letter, source_rank] = sym_rank(
                NormalForm(
                    blocks=tuple(tuple(block) for block in blocks),
                    primes=("p",) * plan.output_grade[0],
                )
            )
    return expected


def _targets_encoded_by_generator_plan(plan):
    if plan.target_ranks is not None:
        return np.asarray(plan.target_ranks, dtype=np.int64)

    destination = plan.destination_plan
    assert destination is not None
    source_count = plan.source_rank_count
    encoded = np.full(
        (plan.d_doubleprime, source_count),
        -1,
        dtype=np.int64,
    )

    def install(edge_id, target_rank):
        letter, source_rank = divmod(int(edge_id), source_count)
        assert encoded[letter, source_rank] == -1
        encoded[letter, source_rank] = int(target_rank)

    head = range(
        destination.primary_head_start,
        destination.primary_head_start + destination.primary_head_count,
    )
    for target_rank, edge_id in enumerate(head):
        install(edge_id, target_rank)
    selected = np.asarray(destination.selected_edge_ids)
    for offset, edge_id in enumerate(
        selected[: destination.tail_primary_count]
    ):
        install(edge_id, destination.primary_head_count + offset)
    for edge_id, target_rank in zip(
        selected[destination.tail_primary_count :],
        destination.collision_target_ranks,
    ):
        install(edge_id, target_rank)

    assert np.all(encoded >= 0)
    return encoded


@pytest.mark.parametrize("d_doubleprime", (1, 2, 3, 4))
def test_doubleprime_generator_plans_match_terminal_block_oracle(
    d_doubleprime,
):
    store = PartiallySymmetrizedPlanStore(
        (2, d_doubleprime),
        (2, 3),
    )
    for output_grade, plan in store.doubleprime_generator_plans.items():
        expected = _doubleprime_target_oracle(store, plan)
        actual = _targets_encoded_by_generator_plan(plan)
        np.testing.assert_array_equal(actual, expected)
        assert np.array_equal(
            np.unique(actual),
            np.arange(plan.doubleprime_rank_count),
        )
        for letter_targets in actual:
            assert np.unique(letter_targets).size == plan.source_rank_count

        if plan.target_ranks is not None:
            assert not plan.uses_destination_order
            assert not plan.target_ranks.flags.writeable
            continue

        assert plan.uses_destination_order
        destination = plan.destination_plan
        assert destination is not None
        edge_count = plan.d_doubleprime * plan.source_rank_count
        assert destination.edge_count == edge_count
        assert destination.output_rank_count == plan.doubleprime_rank_count
        assert destination.primary_head_start == (
            (plan.d_doubleprime - 1) * plan.source_rank_count
        )
        assert destination.primary_head_count == plan.source_rank_count
        assert destination.tail_primary_count == (
            plan.doubleprime_rank_count - plan.source_rank_count
        )
        assert destination.collision_count == (
            edge_count - plan.doubleprime_rank_count
        )
        head_edges = np.arange(
            destination.primary_head_start,
            destination.primary_head_start + destination.primary_head_count,
        )
        np.testing.assert_array_equal(
            np.sort(
                np.concatenate(
                    (head_edges, destination.selected_edge_ids)
                )
            ),
            np.arange(edge_count),
        )
        assert destination.selected_edge_ids.dtype == np.dtype(
            _unsigned_index_dtype(edge_count - 1)
        )
        assert destination.collision_target_ranks.dtype == np.dtype(
            _unsigned_index_dtype(plan.doubleprime_rank_count - 1)
        )
        assert not destination.selected_edge_ids.flags.writeable
        assert not destination.collision_target_ranks.flags.writeable
        assert plan.memory_bytes() == destination.memory_bytes()


@pytest.mark.parametrize("d_doubleprime", (1, 2, 3, 4))
def test_shared_doubleprime_generator_executor_is_jittable_and_exact(
    d_doubleprime,
):
    store = PartiallySymmetrizedPlanStore(
        (2, d_doubleprime),
        (2, 2),
    )
    plan = store.doubleprime_generator_plan((1, 2))
    source_width = plan.source_rank_count * plan.dense_prime_width
    block = jnp.arange(2 * source_width, dtype=jnp.float32).reshape(
        2,
        source_width,
    )
    generator = jnp.linspace(
        -1.0,
        2.0,
        2 * d_doubleprime,
        dtype=jnp.float32,
    ).reshape(2, d_doubleprime)

    def apply(source, increment):
        return apply_doubleprime_generator_block(
            jnp,
            source,
            increment,
            plan,
            scatter_add=_scatter_add,
        )

    expected = np.zeros(
        (2, plan.output_rank_count, plan.dense_prime_width),
        dtype=np.float32,
    )
    source = np.asarray(block).reshape(
        2,
        plan.source_rank_count,
        plan.dense_prime_width,
    )
    targets = _doubleprime_target_oracle(store, plan)
    for batch in range(2):
        for letter in range(plan.d_doubleprime):
            for source_rank, target_rank in enumerate(targets[letter]):
                expected[batch, target_rank] += (
                    source[batch, source_rank] * generator[batch, letter]
                )
    expected = expected.reshape(2, -1)

    np.testing.assert_allclose(jax.jit(apply)(block, generator), expected)
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda source: apply(source, generator).sum()))(block),
        jax.grad(lambda source: apply(source, generator).sum())(block),
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda increment: apply(block, increment).sum()))(
            generator
        ),
        jax.grad(lambda increment: apply(block, increment).sum())(generator),
    )
    np.testing.assert_allclose(
        jax.vmap(apply)(block, generator),
        jnp.stack(tuple(apply(row, inc) for row, inc in zip(block, generator))),
    )


def test_q1_destination_prefix_and_block_skip_scatter():
    store = PartiallySymmetrizedPlanStore((2, 1), (2, 3))
    plan = store.doubleprime_generator_plan((1, 3))
    assert plan.uses_destination_order
    destination = plan.destination_plan
    assert destination is not None
    assert destination.selected_edge_ids.size == 0
    assert destination.collision_count == 0
    assert plan.memory_bytes() == 0

    source_width = plan.source_rank_count * plan.dense_prime_width
    block = jnp.arange(2 * source_width, dtype=jnp.float32).reshape(
        2, source_width
    )
    generator = jnp.asarray(((2.0,), (-0.5,)), dtype=jnp.float32)

    def forbidden_scatter(*args, **kwargs):
        del args, kwargs
        raise AssertionError("q=1 destination execution must not scatter")

    prefix = jax.jit(
        lambda source, increment: apply_doubleprime_generator_prefix(
            jnp,
            source,
            increment,
            plan,
            scatter_add=forbidden_scatter,
        )
    )(block, generator)
    full = jax.jit(
        lambda source, increment: apply_doubleprime_generator_block(
            jnp,
            source,
            increment,
            plan,
            scatter_add=forbidden_scatter,
        )
    )(block, generator)

    expected_prefix = (block * generator).reshape(2, -1)
    np.testing.assert_allclose(prefix, expected_prefix)
    suffix_width = (
        plan.output_rank_count - plan.doubleprime_rank_count
    ) * plan.dense_prime_width
    np.testing.assert_allclose(full[..., : prefix.shape[-1]], prefix)
    np.testing.assert_array_equal(
        full[..., prefix.shape[-1] :],
        np.zeros((2, suffix_width), dtype=np.float32),
    )


def test_hybrid_generator_plan_selects_destination_only_when_favorable():
    hybrid = PartiallySymmetrizedPlanStore((1, 3), (2, 3))
    assert not hybrid.doubleprime_generator_plan((1, 1)).uses_destination_order
    assert hybrid.doubleprime_generator_plan((2, 3)).uses_destination_order

    dense = PartiallySymmetrizedPlanStore((2, 2), (3, 3))
    assert not dense.doubleprime_generator_plan((1, 1)).uses_destination_order
    assert dense.doubleprime_generator_plan((3, 3)).uses_destination_order

    memory_guard = PartiallySymmetrizedPlanStore((1, 8), (0, 3))
    assert not memory_guard.doubleprime_generator_plan(
        (0, 3)
    ).uses_destination_order


@pytest.mark.parametrize(
    ("dims", "capacity"),
    (
        ((2, 3), (2, 2)),
        ((1, 3), (2, 3)),
        ((1, 1), (2, 3)),
        ((1, 8), (0, 3)),
    ),
)
def test_memory_categories_are_exact_and_count_every_array_once(
    dims,
    capacity,
):
    store = PartiallySymmetrizedPlanStore(dims, capacity)
    expected = _expected_plan_memory_bytes_by_category(dims, capacity)

    assert store.memory_bytes_by_category() == expected
    assert set(expected) == {
        "multiset_placements",
        "symmetrized_grade_metadata",
        "symmetrized_concatenation_target_maps",
        "symmetrized_concatenation_segment_metadata",
        "shared_doubleprime_generator_maps",
    }
    assert store.memory_bytes() == sum(expected.values())
    assert store.plan_statistics()["memory_bytes"] == store.memory_bytes()


def test_store_validation_and_capacity_failures_are_eager():
    with pytest.raises(ValueError, match="strictly positive"):
        PartiallySymmetrizedPlanStore((0, 2), (1, 1))
    with pytest.raises(TypeError, match="dims"):
        PartiallySymmetrizedPlanStore((True, 2), (1, 1))

    store = PartiallySymmetrizedPlanStore((1, 2), (1, 1))
    with pytest.raises(ValueError, match="exceeds"):
        store.resolve((2, 1))
    with pytest.raises(KeyError, match="no double-prime predecessor"):
        store.doubleprime_generator_plan((1, 0))


def test_coefficient_storage_rejects_values_beyond_uint64():
    assert _coefficient_dtype(np.iinfo(np.uint64).max) is np.uint64
    with pytest.raises(OverflowError, match="exceeds uint64"):
        _coefficient_dtype(np.iinfo(np.uint64).max + 1)
