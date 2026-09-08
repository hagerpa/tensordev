from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensordev.core.bigraded import (
    BigradedSpec,
    BigradedTensor,
    JaxBigraded,
)
from tensordev.core.bigraded.symmetrized.bridge import (
    SymmetrizationBridgePlanStore,
    _lift_partially_symmetrized_block,
    partially_symmetrize_block,
)
from tensordev.core.bigraded.symmetrized.jax import (
    JaxPartiallySymmetrizedBigraded,
)
from tensordev.core.shear.bigraded import JaxShearBigraded
from tensordev.core.shear.symmetrized import (
    JaxPartiallySymmetrizedShearBigraded,
)


def _scatter_add(output, targets, values):
    return output.at[..., targets, :].add(values)


def _constant_tensor(core, value, *, trunc=(0, 0)):
    layout = core.resolve_layout(trunc)
    return BigradedTensor(
        tuple(
            jnp.full((layout.block_width(grade),), value)
            for grade in layout.grades
        ),
        layout.spec,
    )


@pytest.fixture(scope="module")
def cores():
    kwargs = {"dims": (1, 2), "max_trunc": (1, 2)}
    return {
        "ordered_standard": JaxBigraded(**kwargs),
        "quotient_standard": JaxPartiallySymmetrizedBigraded(**kwargs),
        "ordered_shear": JaxShearBigraded(**kwargs),
        "quotient_shear": JaxPartiallySymmetrizedShearBigraded(**kwargs),
    }


def test_quotient_specs_are_hashable_pytree_metadata_across_coordinates(cores):
    tensors = {
        name: _constant_tensor(core, index + 1)
        for index, (name, core) in enumerate(cores.items())
    }
    standard_spec = tensors["quotient_standard"].spec
    equal_standard_spec = BigradedSpec(
        *standard_spec.dims,
        standard_spec.truncation,
        coordinates="standard",
        include_scalar=True,
        partially_symmetrized=True,
    )

    assert standard_spec == equal_standard_spec
    assert hash(standard_spec) == hash(equal_standard_spec)
    assert len({tensor.spec for tensor in tensors.values()}) == 4

    leaves, tree = jax.tree_util.tree_flatten(tensors["quotient_standard"])
    assert len(leaves) == 1
    rebuilt = jax.tree_util.tree_unflatten(tree, leaves)
    assert rebuilt.spec == standard_spec
    assert jax.tree_util.tree_structure(rebuilt) == tree
    assert tree != jax.tree_util.tree_structure(tensors["quotient_shear"])
    assert tree != jax.tree_util.tree_structure(tensors["ordered_standard"])

    traces = []

    @jax.jit
    def static_metadata_code(tensor):
        spec = tensor.spec
        traces.append((spec.partially_symmetrized, spec.coordinates))
        code = int(spec.partially_symmetrized)
        code += 2 * int(spec.coordinates == "shear")
        return tensor[(0, 0)] + code

    expected_codes = {
        "ordered_standard": 0,
        "quotient_standard": 1,
        "ordered_shear": 2,
        "quotient_shear": 3,
    }
    for name, tensor in tensors.items():
        expected = np.asarray(tensor[(0, 0)]) + expected_codes[name]
        np.testing.assert_array_equal(static_metadata_code(tensor), expected)

    # An independently constructed tensor with equal metadata reuses the
    # quotient-standard trace instead of treating the spec as a dynamic leaf.
    np.testing.assert_array_equal(
        static_metadata_code(_constant_tensor(cores["quotient_standard"], 9)),
        np.asarray([10]),
    )
    assert traces == [
        (False, "standard"),
        (True, "standard"),
        (False, "shear"),
        (True, "shear"),
    ]


@pytest.mark.parametrize(
    ("ordered_name", "quotient_name"),
    (
        ("ordered_standard", "quotient_standard"),
        ("ordered_shear", "quotient_shear"),
    ),
)
def test_public_algebra_rejects_mismatched_symmetrization(
    cores,
    ordered_name,
    quotient_name,
):
    ordered_core = cores[ordered_name]
    quotient_core = cores[quotient_name]
    ordered = _constant_tensor(ordered_core, 1, trunc=(1, 1))
    quotient = _constant_tensor(quotient_core, 1, trunc=(1, 1))

    with pytest.raises(
        ValueError,
        match="partially_symmetrized.*expected False",
    ):
        ordered_core.tensor_summation(quotient, quotient)
    with pytest.raises(
        ValueError,
        match="partially_symmetrized.*expected True",
    ):
        quotient_core.tensor_summation(ordered, ordered)


def test_quotient_algebra_rejects_cross_coordinate_inputs(cores):
    standard = _constant_tensor(
        cores["quotient_standard"], 1, trunc=(1, 1)
    )
    shear = _constant_tensor(cores["quotient_shear"], 1, trunc=(1, 1))

    with pytest.raises(ValueError, match="expected 'standard'"):
        cores["quotient_standard"].tensor_summation(shear, shear)
    with pytest.raises(ValueError, match="expected 'shear'"):
        cores["quotient_shear"].tensor_summation(standard, standard)


@pytest.mark.parametrize(
    ("dims", "grade", "edge"),
    (
        ((2, 2), (0, 3), "zero_prime"),
        ((2, 2), (3, 0), "zero_doubleprime"),
        ((2, 1), (2, 3), "one_doubleprime_letter"),
    ),
)
def test_bridge_edge_grades_preserve_the_q_q_transpose_pairing(
    dims,
    grade,
    edge,
):
    plan = SymmetrizationBridgePlanStore(dims, grade).grade_plan(grade)
    ordered = (
        jnp.arange(plan.ordered_block_width, dtype=jnp.float32) + 0.5
    )
    quotient_words = (
        jnp.arange(plan.quotient_block_width, dtype=jnp.float32) - 0.25
    )
    quotient = partially_symmetrize_block(
        jnp,
        ordered,
        plan,
        scatter_add=_scatter_add,
    )
    lifted = _lift_partially_symmetrized_block(
        jnp,
        quotient_words,
        plan,
    )

    assert set(map(int, plan.target_ranks)) == set(
        range(plan.quotient_rank_count)
    )
    np.testing.assert_allclose(
        jnp.sum(quotient * quotient_words),
        jnp.sum(ordered * lifted),
    )

    if edge == "zero_prime":
        orbit_sizes = partially_symmetrize_block(
            jnp,
            jnp.ones((plan.ordered_block_width,), dtype=jnp.float32),
            plan,
            scatter_add=_scatter_add,
        )
        np.testing.assert_array_equal(jnp.sort(orbit_sizes), [1, 1, 3, 3])
    elif edge == "zero_doubleprime":
        np.testing.assert_array_equal(quotient, ordered)
        np.testing.assert_array_equal(lifted, quotient_words)
    else:
        np.testing.assert_array_equal(
            np.sort(plan.target_ranks),
            np.arange(plan.source_count),
        )
        np.testing.assert_array_equal(
            _lift_partially_symmetrized_block(jnp, quotient, plan),
            ordered,
        )


@pytest.mark.parametrize(
    "core_name",
    ("quotient_standard", "quotient_shear"),
)
def test_complex_shear_pairing_is_bilinear_q_q_transpose(cores, core_name):
    core = cores[core_name]
    grade = (1, 2)
    plan = core.bridge_plan_store.grade_plan(grade)
    word_index = jnp.arange(
        2 * plan.quotient_block_width, dtype=jnp.float32
    ).reshape(2, 1, plan.quotient_block_width)
    signature_index = jnp.arange(
        3 * plan.ordered_block_width, dtype=jnp.float32
    ).reshape(1, 3, plan.ordered_block_width)
    words = (
        0.1 + 0.03 * word_index + 1j * (0.2 - 0.02 * word_index)
    ).astype(jnp.complex64)
    signature = (
        0.2
        - 0.01 * signature_index
        + 1j * (0.05 + 0.025 * signature_index)
    ).astype(jnp.complex64)

    standard_words = core._coordinate_forward_transpose_block(words, grade)
    lifted_words = core._lift_partially_symmetrized_block(
        standard_words, grade
    )
    quotient_signature = partially_symmetrize_block(
        jnp,
        signature,
        plan,
        scatter_add=_scatter_add,
    )
    result = core.tensor_shear_pairing_homogeneous(
        words,
        signature,
        grade=grade,
    )

    np.testing.assert_allclose(
        result,
        jnp.sum(lifted_words * signature, axis=-1),
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        result,
        jnp.sum(standard_words * quotient_signature, axis=-1),
        atol=2e-6,
        rtol=2e-6,
    )
    assert not np.allclose(
        result,
        jnp.sum(jnp.conj(lifted_words) * signature, axis=-1),
    )
