import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensordev.core.utils.segmented import (
    SegmentedRankPlan,
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
