import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensordev.core.utils.segmented import (
    DestinationRankPlan,
    SegmentedRankPlan,
    apply_destination_rank_plan,
    apply_segmented_rank_plan,
)


def _scatter_add(output, targets, values):
    return output.at[..., targets, :].add(values)


def test_segmented_rank_plan_accumulates_collisions_and_batches():
    plan = SegmentedRankPlan.from_targets([2, 0, 2, 1], output_rank_count=3)
    values = jnp.arange(16.0).reshape(2, 4, 2)

    result = apply_segmented_rank_plan(
        jnp,
        values,
        plan,
        scatter_add=_scatter_add,
    )

    expected = np.stack(
        [
            np.stack((batch[1], batch[3], batch[0] + batch[2]))
            for batch in np.asarray(values)
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_segmented_rank_plan_is_jittable_and_differentiable():
    plan = SegmentedRankPlan.from_targets([0, 1, 0], output_rank_count=2)

    def objective(values):
        result = apply_segmented_rank_plan(
            jnp,
            values,
            plan,
            scatter_add=_scatter_add,
        )
        return jnp.square(result).sum()

    values = jnp.arange(6.0).reshape(3, 2)
    np.testing.assert_allclose(jax.jit(objective)(values), objective(values))
    np.testing.assert_allclose(
        jax.jit(jax.grad(objective))(values),
        jax.grad(objective)(values),
    )


def test_segmented_rank_plan_validates_targets_and_source_axis():
    with pytest.raises(ValueError, match="output_rank_count"):
        SegmentedRankPlan.from_targets([], output_rank_count=0)
    with pytest.raises(ValueError, match="target ranks"):
        SegmentedRankPlan.from_targets([2], output_rank_count=2)

    plan = SegmentedRankPlan.from_targets([0, 1], output_rank_count=2)
    with pytest.raises(ValueError, match="source axis"):
        apply_segmented_rank_plan(
            jnp,
            jnp.ones((3, 1)),
            plan,
            scatter_add=_scatter_add,
        )


def _destination_plan():
    # Head edges 4, 5 supply destinations 0, 1.  The first two selected
    # edges supply destinations 2, 3; the final two are corrections.
    return DestinationRankPlan.from_edges(
        [0, 3, 1, 2],
        [0, 3],
        edge_count=6,
        output_rank_count=4,
        primary_head_start=4,
        primary_head_count=2,
        tail_primary_count=2,
    )


def test_destination_rank_plan_assembles_primary_edges_and_collisions():
    plan = _destination_plan()
    values = jnp.arange(24.0).reshape(2, 6, 2)

    result = apply_destination_rank_plan(
        jnp,
        values,
        plan,
        scatter_add=_scatter_add,
    )
    expected = np.stack(
        [
            np.stack(
                (
                    batch[4] + batch[1],
                    batch[5],
                    batch[0],
                    batch[3] + batch[2],
                )
            )
            for batch in np.asarray(values)
        ]
    )

    np.testing.assert_allclose(result, expected)
    assert plan.selected_edge_ids.dtype == np.uint8
    assert plan.collision_target_ranks.dtype == np.uint8
    assert not plan.selected_edge_ids.flags.writeable
    assert not plan.collision_target_ranks.flags.writeable
    assert plan.memory_bytes() == (
        plan.selected_edge_ids.nbytes
        + plan.collision_target_ranks.nbytes
    )


def test_destination_rank_plan_is_jittable_vmappable_and_differentiable():
    plan = _destination_plan()

    def apply(values):
        return apply_destination_rank_plan(
            jnp,
            values,
            plan,
            scatter_add=_scatter_add,
        )

    values = jnp.arange(36.0, dtype=jnp.float32).reshape(3, 6, 2)
    np.testing.assert_allclose(jax.jit(apply)(values), apply(values))
    np.testing.assert_allclose(
        jax.vmap(apply)(values),
        jnp.stack(tuple(apply(row) for row in values)),
    )

    def objective(current):
        return jnp.square(apply(current)).sum()

    np.testing.assert_allclose(
        jax.jit(jax.grad(objective))(values),
        jax.grad(objective)(values),
    )


def test_destination_rank_plan_skips_selection_and_scatter_for_a_pure_head():
    plan = DestinationRankPlan.from_edges(
        [],
        [],
        edge_count=2,
        output_rank_count=2,
        primary_head_start=0,
        primary_head_count=2,
        tail_primary_count=0,
    )

    def forbidden_scatter(*args, **kwargs):
        del args, kwargs
        raise AssertionError("a pure destination head must not scatter")

    values = jnp.arange(12.0).reshape(3, 2, 2)
    result = jax.jit(
        lambda current: apply_destination_rank_plan(
            jnp,
            current,
            plan,
            scatter_add=forbidden_scatter,
        )
    )(values)
    np.testing.assert_array_equal(result, values)


def test_destination_rank_plan_validates_partition_and_edge_axis():
    with pytest.raises(ValueError, match="sorted"):
        DestinationRankPlan.from_edges(
            [0, 3, 1, 2],
            [3, 0],
            edge_count=6,
            output_rank_count=4,
            primary_head_start=4,
            primary_head_count=2,
            tail_primary_count=2,
        )

    plan = _destination_plan()
    with pytest.raises(ValueError, match="edge axis"):
        apply_destination_rank_plan(
            jnp,
            jnp.ones((5, 1)),
            plan,
            scatter_add=_scatter_add,
        )
