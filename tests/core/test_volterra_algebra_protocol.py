from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tensordev as td
from tensordev import Jax, bigraded_core
from tensordev.core.bigraded import BigradedTensor
from tensordev.core.jax import JaxSequentialCore
from tensordev.core.sequential import SequentialCore
from tensordev.core.universal import Universal
from tensordev.volterra.algebra import (
    GradeWorkset,
    resolve_volterra_algebra,
    resolve_volterra_core_pair,
    require_volterra_shuffle,
)


def _sum_terms(terms):
    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return result


class _CoreProxy:
    def __init__(self, delegate):
        self.delegate = delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class _MissingProductOutputHook(_CoreProxy):
    def __getattr__(self, name):
        if name == "_product_output_block":
            raise AttributeError(name)
        return super().__getattr__(name)


class _NativeProductOutputHook(_CoreProxy):
    def __init__(self, delegate):
        super().__init__(delegate)
        self.calls = []

    def _product_output_block(self, contributions, output_grade):
        self.calls.append((tuple(contributions), output_grade))
        width = self._block_width_for_layout(
            self.resolve_layout(output_grade, include_scalar=True),
            output_grade,
            alphabet_dim=2,
        )
        return jnp.full((width,), 17.0)

    def _product_grade(self, *_args, **_kwargs):
        raise AssertionError("the opportunistic hook must not be used")


class _ShearMarkerCore(Jax):
    coordinates = "shear"


def test_total_layouts_cache_and_native_element_boundaries():
    core = Jax()
    algebra = resolve_volterra_algebra(core, 3, 2)

    assert resolve_volterra_algebra(core, 3, 2) is algebra
    assert algebra.backend == "jax"
    assert algebra.grades == (0, 1, 2, 3)
    assert algebra.positive_grades == (1, 2, 3)
    assert algebra.grades_by_total_order == ((0,), (1,), (2,), (3,))
    assert algebra.first_level_grades == (1,)
    assert algebra.diagonal_offsets_by_total_order == (
        (0, 1),
        (0, 2),
        (0, 4),
        (0, 8),
    )
    assert algebra.supports("concatenation")
    assert algebra.supports("generator_action")
    assert algebra.supports("shuffle")

    unit = algebra.unit(batch_shape=(2,), dtype=jnp.float32)
    assert isinstance(unit, tuple)
    assert tuple(block.shape for block in unit) == ((2, 1), (2, 2), (2, 4), (2, 8))
    np.testing.assert_array_equal(unit[0], jnp.ones((2, 1)))
    for block in unit[1:]:
        np.testing.assert_array_equal(block, jnp.zeros_like(block))

    positive = algebra.assemble(
        tuple(
            jnp.full((2, algebra.block_width(grade)), float(grade))
            for grade in algebra.positive_grades
        ),
        positive=True,
    )
    full = algebra.embed_positive(positive)
    np.testing.assert_array_equal(full[0], jnp.zeros((2, 1)))
    for grade in algebra.positive_grades:
        np.testing.assert_array_equal(
            algebra.block(full, grade), algebra.block(positive, grade, positive=True)
        )


def test_total_diagonal_pack_is_identity_and_actions_match_existing_kernels():
    core = Jax()
    algebra = resolve_volterra_algebra(core, 2, 3)
    source_workset = algebra.diagonal(1)
    target_workset = algebra.diagonal(2)
    source = (jnp.arange(6.0).reshape(2, 3),)
    z = jnp.array([[0.5, -1.0, 2.0]])

    packed = algebra.pack_diagonal(1, source)
    assert packed is source[0]
    assert algebra.split_diagonal(1, packed)[0] is packed

    right = algebra.right_generator_action(
        source_workset, source, z, target_workset
    )
    expected_right = core.tensor_product_homogeneous(source[0], z)
    np.testing.assert_allclose(right[0], expected_right)

    shuffle = algebra.shuffle_generator_action(
        source_workset, source, z, target_workset
    )
    expected_shuffle = core.tensor_shuffle_vector_homogeneous(source[0], z, 1)
    np.testing.assert_allclose(shuffle[0], expected_shuffle)

    with pytest.raises(ValueError, match="generator block 1 has width 2, expected 3"):
        algebra.generator_blocks(jnp.ones((2,)))


def test_total_product_output_block_matches_reference_lowering_and_order():
    core = Jax()
    algebra = resolve_volterra_algebra(core, 3, 2)
    output_grade = 3
    splits = algebra.layout.product_splits(output_grade)
    inputs = tuple(
        block
        for left_grade, right_grade in splits
        for block in (
            jnp.arange(2 * 2**left_grade, dtype=jnp.float32).reshape(
                2, 2**left_grade
            ),
            jnp.arange(2 * 2**right_grade, dtype=jnp.float32).reshape(
                2, 2**right_grade
            ),
        )
    )

    def contributions(values):
        return tuple(
            (
                values[2 * index],
                values[2 * index + 1],
                left_grade,
                right_grade,
            )
            for index, (left_grade, right_grade) in enumerate(splits)
        )

    def through_adapter(*values):
        return algebra.product_output_block(
            contributions(values), output_grade
        )

    def reference(*values):
        ordered = contributions(values)
        left, right, left_grade, right_grade = ordered[0]
        result = core._product_block(
            left, right, left_grade, right_grade, output_grade
        )
        for left, right, left_grade, right_grade in ordered[1:]:
            result = result + core._product_block(
                left, right, left_grade, right_grade, output_grade
            )
        return result

    assert callable(core._product_output_block)
    np.testing.assert_array_equal(through_adapter(*inputs), reference(*inputs))
    assert str(jax.make_jaxpr(through_adapter)(*inputs)) == str(
        jax.make_jaxpr(reference)(*inputs)
    )


def test_product_output_block_uses_only_required_coordinate_native_hook():
    core = _NativeProductOutputHook(Jax())
    algebra = resolve_volterra_algebra(core, 2, 2)
    contributions = (
        (jnp.array([2.0]), jnp.arange(4.0), 0, 2),
        (jnp.arange(2.0), jnp.arange(2.0), 1, 1),
        (jnp.arange(4.0), jnp.array([3.0]), 2, 0),
    )

    got = algebra.product_output_block(contributions, 2)

    np.testing.assert_array_equal(got, jnp.full((4,), 17.0))
    assert core.calls == [(contributions, 2)]


def test_resolution_requires_coordinate_native_product_output_hook():
    core = _MissingProductOutputHook(Jax())
    with pytest.raises(TypeError, match="coordinate-native _product_output_block"):
        resolve_volterra_algebra(core, 2, 2)


def test_total_shared_product_driver_matches_reference_lowering_and_order():
    core = Jax()
    trunc = 3
    left = tuple(
        jnp.arange(2 * 2**grade, dtype=jnp.float32).reshape(2, 2**grade)
        for grade in range(trunc + 1)
    )
    right = tuple(block + 1.0 for block in left)

    def through_shared_driver(A, B):
        return Universal.tensor_product(core, A, B, trunc=trunc)

    def reference(A, B):
        blocks = []
        for output_grade in range(trunc + 1):
            result = core._product_block(
                A[0], B[output_grade], 0, output_grade, output_grade
            )
            for left_grade in range(1, output_grade + 1):
                right_grade = output_grade - left_grade
                result = result + core._product_block(
                    A[left_grade],
                    B[right_grade],
                    left_grade,
                    right_grade,
                    output_grade,
                )
            blocks.append(result)
        return tuple(blocks)

    actual = through_shared_driver(left, right)
    expected = reference(left, right)
    for actual_block, expected_block in zip(actual, expected):
        np.testing.assert_array_equal(actual_block, expected_block)
    assert str(jax.make_jaxpr(through_shared_driver)(left, right)) == str(
        jax.make_jaxpr(reference)(left, right)
    )


def test_bidegree_metadata_is_active_layout_only_and_capacity_independent():
    small = bigraded_core(dims=(1, 2), max_trunc=(2, 1))
    large = bigraded_core(dims=(1, 2), max_trunc=(3, 3))
    algebra = resolve_volterra_algebra(small, (2, 1), 3)
    other = resolve_volterra_algebra(large, (2, 1), 3)

    assert algebra.grades == (
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (2, 1),
    )
    assert algebra.positive_grades == algebra.grades[1:]
    assert algebra.grades_by_total_order == (
        ((0, 0),),
        ((1, 0), (0, 1)),
        ((2, 0), (1, 1)),
        ((2, 1),),
    )
    assert algebra.first_level_grades == ((1, 0), (0, 1))
    assert algebra.diagonal_offsets_by_total_order == (
        (0, 1),
        (0, 1, 3),
        (0, 1, 5),
        (0, 6),
    )
    assert algebra.grades == other.grades
    assert tuple(
        algebra.diagonal(order).widths for order in range(algebra.max_order + 1)
    ) == tuple(
        other.diagonal(order).widths for order in range(other.max_order + 1)
    )
    assert algebra.diagonal_offsets_by_total_order == other.diagonal_offsets_by_total_order


def test_bidegree_diagonal_pack_split_and_positive_embedding():
    core = bigraded_core(dims=(1, 2), max_trunc=(2, 1))
    algebra = resolve_volterra_algebra(core, (2, 1), 3)
    workset = algebra.diagonal(2)
    values = (
        jnp.array([[1.0], [2.0]]),
        jnp.arange(4.0).reshape(1, 4),
    )

    packed = workset.pack(core.xp, values)
    expected = jnp.concatenate(
        (values[0], jnp.broadcast_to(values[1], (2, 4))), axis=-1
    )
    np.testing.assert_array_equal(packed, expected)
    split = workset.split(packed)
    np.testing.assert_array_equal(split[0], expected[..., :1])
    np.testing.assert_array_equal(split[1], expected[..., 1:])

    positive_blocks = tuple(
        jnp.full((2, algebra.block_width(grade)), float(sum(grade)))
        for grade in algebra.positive_grades
    )
    positive = algebra.assemble(positive_blocks, positive=True)
    assert isinstance(positive, BigradedTensor)
    assert not positive.spec.include_scalar
    full = algebra.embed_positive(positive)
    assert full.spec.include_scalar
    np.testing.assert_array_equal(full[(0, 0)], jnp.zeros((2, 1)))
    for grade in algebra.positive_grades:
        np.testing.assert_array_equal(full[grade], positive[grade])


@pytest.mark.parametrize("action", ["right", "shuffle"])
def test_bidegree_fused_generator_actions_match_native_block_kernels(action):
    core = bigraded_core(
        dims=(1, 2), max_trunc=(2, 1), precompute_shuffle=True
    )
    algebra = resolve_volterra_algebra(core, (2, 1), 3)
    source_workset = algebra.diagonal(1)
    target_workset = algebra.diagonal(2)
    source_values = (
        jnp.array([[1.0], [2.0]]),
        jnp.array([[3.0, 4.0], [5.0, 6.0]]),
    )
    z = jnp.array([[0.25, -0.5, 1.5], [2.0, 0.75, -1.0]])
    generator = dict(algebra.generator_blocks(z))

    if action == "right":
        got = algebra.right_generator_action(
            source_workset, source_values, z, target_workset
        )
        block_action = core._product_block
    else:
        got = algebra.shuffle_generator_action(
            source_workset, source_values, z, target_workset
        )
        block_action = core._shuffle_block

    expected = []
    for output_grade in target_workset.grades:
        terms = []
        for source_grade, generator_grade in algebra.generator_splits(output_grade):
            terms.append(
                block_action(
                    source_workset.block(source_values, source_grade),
                    generator[generator_grade],
                    source_grade,
                    generator_grade,
                    output_grade,
                )
            )
        expected.append(_sum_terms(terms))

    for actual, reference in zip(got, expected):
        np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-6)

    compiled = jax.jit(
        lambda first, second, generator_value: (
            algebra.right_generator_action(
                source_workset, (first, second), generator_value, target_workset
            )
            if action == "right"
            else algebra.shuffle_generator_action(
                source_workset, (first, second), generator_value, target_workset
            )
        )
    )(*source_values, z)
    for actual, reference in zip(compiled, expected):
        np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-6)


def test_one_sided_active_rectangle_filters_generator_schedule():
    core = bigraded_core(dims=(1, 2), max_trunc=(2, 2))
    algebra = resolve_volterra_algebra(core, (0, 2), 3)
    z = jnp.array([[10.0, 1.0, 2.0]])

    generator = algebra.generator_blocks(z)
    assert tuple(grade for grade, _ in generator) == ((0, 1),)
    np.testing.assert_array_equal(generator[0][1], z[..., 1:])

    source_workset = algebra.diagonal(0)
    target_workset = algebra.diagonal(1)
    result = algebra.right_generator_action(
        source_workset, (jnp.ones((1, 1)),), z, target_workset
    )
    np.testing.assert_array_equal(result[0], z[..., 1:])


def test_workset_pytree_map_add_and_validation():
    workset = GradeWorkset(("a", "b"), (2, 1))
    values = (jnp.array([[1.0, 2.0]]), jnp.array([[3.0]]))
    doubled = workset.map(lambda value: {"leaf": 2 * value}, values)
    assert tuple(item.keys() for item in doubled) == ({"leaf"}, {"leaf"})
    np.testing.assert_array_equal(doubled[0]["leaf"], 2 * values[0])
    added = workset.add(values, values)
    np.testing.assert_array_equal(added[1], 2 * values[1])

    with pytest.raises(ValueError, match="received 1 blocks"):
        workset.validate_values(values[:1])
    with pytest.raises(ValueError, match="expected 2"):
        workset.validate_values((jnp.ones((1, 3)), values[1]))
    with pytest.raises(KeyError, match="not in this workset"):
        workset.block(values, "missing")


def test_resolution_rejects_wrong_backend_dimension_and_empty_positive_layout():
    with pytest.raises(TypeError, match="requires a JAX tensor core"):
        resolve_volterra_algebra(Universal(np), 2, 2)

    core = bigraded_core(dims=(1, 2), max_trunc=(2, 1))
    with pytest.raises(ValueError, match="does not match bidegree dimensions"):
        resolve_volterra_algebra(core, (2, 1), 4)
    with pytest.raises(ValueError, match="at least one positive grade"):
        resolve_volterra_algebra(core, (0, 0), 3)


def test_shuffle_capability_is_explicit():
    core = bigraded_core(dims=(1, 1), max_trunc=(1, 1))
    algebra = resolve_volterra_algebra(core, (1, 1), 2)

    assert not algebra.supports("shuffle")
    with pytest.raises(RuntimeError, match="'shuffle' algebra capability"):
        algebra.shuffle_generator_action(
            algebra.diagonal(0),
            (jnp.ones((1,)),),
            jnp.ones((2,)),
            algebra.diagonal(1),
        )


def test_volterra_shuffle_configuration_error_is_coordinate_neutral():
    core = bigraded_core(dims=(1, 1), max_trunc=(2, 1))
    algebra = resolve_volterra_algebra(core, (2, 1), 2)

    with pytest.raises(RuntimeError, match="configure the selected core") as error:
        require_volterra_shuffle(algebra, feature="test evaluator")
    assert "bidegree" not in str(error.value)


def test_core_pair_resolution_preserves_explicit_pair_and_checks_backends():
    core = bigraded_core(dims=(1, 1), max_trunc=(1, 1))
    seq_core = JaxSequentialCore()

    assert resolve_volterra_core_pair(core, seq_core) == (core, seq_core)
    inferred_core, inferred_seq = resolve_volterra_core_pair(core)
    assert inferred_core is core
    assert inferred_seq.backend == "jax"

    with pytest.raises(TypeError, match="compatible JAX algebra"):
        resolve_volterra_core_pair(core, SequentialCore(np))


@pytest.mark.parametrize(
    ("feature", "call"),
    (
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
    ),
)
def test_standard_specific_consumers_reject_total_degree_shear_default(
        feature, call
):
    td.set_default_core(_ShearMarkerCore())
    try:
        with pytest.raises(
                RuntimeError,
                match=rf"{feature} requires standard coordinates",
        ):
            call(jnp.zeros((3, 2)))
    finally:
        td.reset_default_core()
