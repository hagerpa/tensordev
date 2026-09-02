from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quotient_word_oracle import hat_psi_map
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
)
from tensordev.core.bigraded.symmetrized.transforms import (
    PartiallySymmetrizedShearPlanStore,
    apply_partially_symmetrized_transform,
    apply_partially_symmetrized_transform_plan,
)


def _numpy_scatter_add(output, targets, values):
    np.add.at(output, (..., np.asarray(targets), slice(None)), values)
    return output


def _jax_scatter_add(output, targets, values):
    return output.at[..., targets, :].add(values)


def _matrix(plan):
    matrix = np.zeros((plan.rank_count, plan.rank_count), dtype=object)
    output_ranks, input_ranks = plan.rank_pairs()
    for output_rank, input_rank, coefficient in zip(
        output_ranks,
        input_ranks,
        plan.coefficients,
    ):
        matrix[int(output_rank), int(input_rank)] += int(coefficient)
    if plan.inverse:
        parity = np.asarray(plan.parities, dtype=object)
        matrix = parity[:, None] * matrix * parity[None, :]
    return matrix


@pytest.mark.parametrize("d_doubleprime", (1, 2))
def test_transform_integer_matrices_match_the_independent_word_oracle(
    d_doubleprime,
):
    base = PartiallySymmetrizedPlanStore(
        (1, d_doubleprime),
        (2, 3),
    )
    store = PartiallySymmetrizedShearPlanStore(base)
    doubleprime = tuple(f"x{index}" for index in range(d_doubleprime))

    for grade in base.grade_plans:
        matrices = []
        for inverse in (False, True):
            plan = store.transform_plan(grade, inverse=inverse)
            actual = _matrix(plan)
            expected = np.asarray(
                hat_psi_map(
                    ("p",),
                    doubleprime,
                    grade,
                    inverse=inverse,
                ).rows,
                dtype=object,
            )
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(np.tril(actual, -1), 0)
            np.testing.assert_array_equal(np.diag(actual), 1)
            assert not plan.encoded_rank_pairs.flags.writeable
            assert not plan.coefficients.flags.writeable
            assert not plan.parities.flags.writeable
            matrices.append(actual)
        identity = np.eye(matrices[0].shape[0], dtype=object)
        np.testing.assert_array_equal(matrices[0] @ matrices[1], identity)
        np.testing.assert_array_equal(matrices[1] @ matrices[0], identity)


def test_sparse_dense_and_all_transpose_orientations_are_identical():
    base = PartiallySymmetrizedPlanStore((2, 2), (2, 2))
    sparse = PartiallySymmetrizedShearPlanStore(
        base,
        dense_rank_threshold=0,
    )
    dense = PartiallySymmetrizedShearPlanStore(
        base,
        dense_rank_threshold=10_000,
        dense_relative_bytes=10_000,
    )
    grade = (2, 2)
    width = base.grade_plan(grade).block_width
    values = np.arange(3 * width, dtype=float).reshape(3, width) / 11
    probe = np.arange(3 * width, dtype=float).reshape(3, width)[::-1] / 7

    for inverse in (False, True):
        sparse_plan = sparse.transform_plan(grade, inverse=inverse)
        dense_plan = dense.transform_plan(grade, inverse=inverse)
        assert sparse_plan.strategy == "sparse"
        assert dense_plan.strategy == "dense"
        assert sparse_plan.parities is sparse.forward_plans[grade].parities
        for transpose in (False, True):
            sparse_result = apply_partially_symmetrized_transform_plan(
                np,
                values,
                sparse_plan,
                transpose=transpose,
                scatter_add=_numpy_scatter_add,
            )
            dense_result = apply_partially_symmetrized_transform_plan(
                np,
                values,
                dense_plan,
                transpose=transpose,
                scatter_add=_numpy_scatter_add,
            )
            np.testing.assert_allclose(sparse_result, dense_result)

        forward = apply_partially_symmetrized_transform_plan(
            np,
            values,
            sparse_plan,
            scatter_add=_numpy_scatter_add,
        )
        transpose_probe = apply_partially_symmetrized_transform_plan(
            np,
            probe,
            sparse_plan,
            transpose=True,
            scatter_add=_numpy_scatter_add,
        )
        np.testing.assert_allclose(
            np.sum(forward * probe),
            np.sum(values * transpose_probe),
        )

    restored = apply_partially_symmetrized_transform(
        np,
        apply_partially_symmetrized_transform(
            np,
            values,
            sparse,
            grade,
            orientation="forward",
            scatter_add=_numpy_scatter_add,
        ),
        sparse,
        grade,
        orientation="inverse",
        scatter_add=_numpy_scatter_add,
    )
    np.testing.assert_allclose(restored, values, atol=1e-12)


def test_transform_executor_batches_jits_vmaps_and_differentiates():
    base = PartiallySymmetrizedPlanStore((2, 2), (2, 2))
    store = PartiallySymmetrizedShearPlanStore(
        base,
        dense_rank_threshold=0,
    )
    plan = store.transform_plan((2, 2), inverse=True)
    width = base.grade_plan((2, 2)).block_width

    def transform(values):
        return apply_partially_symmetrized_transform_plan(
            jnp,
            values,
            plan,
            transpose=True,
            scatter_add=_jax_scatter_add,
        )

    values = jnp.arange(3 * width, dtype=jnp.float32).reshape(3, width) / 19
    eager = transform(values)
    np.testing.assert_allclose(jax.jit(transform)(values), eager)
    np.testing.assert_allclose(jax.vmap(transform)(values), eager)

    def objective(value):
        return jnp.square(transform(value)).sum()

    gradient = jax.jit(jax.grad(objective))(values[0])
    assert gradient.shape == values[0].shape
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_transform_memory_categories_are_exact_and_shared_parity_is_once():
    base = PartiallySymmetrizedPlanStore((1, 2), (2, 2))
    store = PartiallySymmetrizedShearPlanStore(base)
    memory = store.memory_bytes_by_category()
    plans = tuple(store.forward_plans.values()) + tuple(
        store.inverse_plans.values()
    )

    assert memory["transform_rank_pairs"] == sum(
        plan.encoded_rank_pairs.nbytes for plan in plans
    )
    assert memory["transform_coefficients"] == sum(
        plan.coefficients.nbytes for plan in plans
    )
    assert memory["transform_parities"] == sum(
        plan.parities.nbytes for plan in store.forward_plans.values()
    )
    assert store.memory_bytes() == sum(memory.values())
    assert store.plan_statistics()["memory_bytes"] == store.memory_bytes()
    assert all(
        store.forward_plans[grade].parities
        is store.inverse_plans[grade].parities
        for grade in base.grade_plans
    )
