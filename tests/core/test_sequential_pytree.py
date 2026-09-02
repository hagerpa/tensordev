from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensordev.core.jax import JaxSequentialCore
from tensordev.core.sequential import SequentialCore
from tensordev.core.bigraded import BigradedSpec, BigradedTensor
from tensordev.core.utils.pytrees import tree_take


SEQ = JaxSequentialCore()
EAGER_SEQ = SequentialCore(np)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _TreeElement:
    """Small graded-element stand-in with static layout metadata."""

    scalar: Any
    vector: Any
    layout: str

    def tree_flatten(self):
        return (self.scalar, self.vector), self.layout

    @classmethod
    def tree_unflatten(cls, layout, children):
        scalar, vector = children
        return cls(scalar=scalar, vector=vector, layout=layout)


def _add(left: _TreeElement, right: _TreeElement) -> _TreeElement:
    assert left.layout == right.layout
    return _TreeElement(
        scalar=left.scalar + right.scalar,
        vector=left.vector + right.vector,
        layout=left.layout,
    )


def _assert_tree_allclose(actual: _TreeElement, expected: _TreeElement) -> None:
    assert isinstance(actual, _TreeElement)
    assert actual.layout == expected.layout
    np.testing.assert_allclose(np.asarray(actual.scalar), np.asarray(expected.scalar))
    np.testing.assert_allclose(np.asarray(actual.vector), np.asarray(expected.vector))


def _primitive_inventory(closed_jaxpr) -> Counter:
    return Counter(eqn.primitive.name for eqn in closed_jaxpr.jaxpr.eqns)


def test_jax_sequential_core_reports_execution_backend():
    assert SEQ.backend == "jax"
    assert SEQ.supports("map")
    assert SEQ.supports("scan")
    assert SEQ.supports("functional_indexed_update")
    assert EAGER_SEQ.backend == "numpy"
    assert EAGER_SEQ.supports("map")
    assert EAGER_SEQ.supports("scan")
    assert not EAGER_SEQ.supports("functional_indexed_update")


def test_tensor_map_normalizes_per_input_axes_and_nested_pytrees():
    left_array = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    right_array = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)

    def run(seq, array):
        left = {
            "value": array(left_array),
            "nested": (array(left_array + 1),),
        }
        right = array(right_array)

        def map_op(left_step, right_step):
            return {
                "sum": left_step["value"] + right_step,
                "nested": (left_step["nested"][0] - right_step,),
            }

        return seq.tensor_map(
            (left, right),
            map_op=map_op,
            in_axes=(-2, 0),
            out_axis=-2,
        )

    eager = run(EAGER_SEQ, np.asarray)
    jax_result = run(SEQ, jnp.asarray)
    expected_right = np.moveaxis(right_array, 0, 1)
    expected = {
        "sum": left_array + expected_right,
        "nested": (left_array + 1 - expected_right,),
    }
    for got in (eager, jax_result):
        np.testing.assert_allclose(np.asarray(got["sum"]), expected["sum"])
        np.testing.assert_allclose(
            np.asarray(got["nested"][0]),
            expected["nested"][0],
        )


def test_tensor_map_preserves_bigraded_tensor_metadata():
    steps = 4
    spec = BigradedSpec(1, 2, (1, 1))
    tensor = BigradedTensor(
        tuple(
            jnp.arange(steps * spec.block_width(grade), dtype=jnp.float32).reshape(
                steps,
                spec.block_width(grade),
            )
            for grade in spec.grades
        ),
        spec,
    )

    result = SEQ.tensor_map(
        (tensor,),
        map_op=lambda step: jax.tree.map(lambda block: 2 * block + 1, step),
        in_axes=0,
        out_axis=0,
    )

    assert isinstance(result, BigradedTensor)
    assert result.spec == spec
    for actual, source in zip(result.blocks, tensor.blocks):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(2 * source + 1))


def test_tree_take_is_leafwise_and_preserves_bigraded_tensor_metadata():
    spec = BigradedSpec(1, 1, (1, 1))
    tensor = BigradedTensor(
        tuple(
            jnp.arange(5 * spec.block_width(grade), dtype=jnp.float32).reshape(
                5,
                spec.block_width(grade),
            )
            for grade in spec.grades
        ),
        spec,
    )
    indices = jnp.asarray([0, 3, 4])

    selected = tree_take(jnp, tensor, indices, axis=0)

    assert isinstance(selected, BigradedTensor)
    assert selected.spec == spec
    for actual, source in zip(selected.blocks, tensor.blocks):
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(source)[np.asarray(indices)],
        )


def test_tensor_map_jaxpr_matches_direct_vmap_inventory():
    X = jnp.arange(15, dtype=jnp.float32).reshape(5, 3)

    def map_op(step):
        return {"linear": 3 * step + 1, "quadratic": (step**2,)}

    wrapped = lambda x: SEQ.tensor_map(
        (x,),
        map_op=map_op,
        in_axes=(0,),
        out_axis=0,
    )
    baseline = lambda x: jax.vmap(map_op)(x)

    wrapped_jaxpr = jax.make_jaxpr(wrapped)(X)
    baseline_jaxpr = jax.make_jaxpr(baseline)(X)
    assert _primitive_inventory(wrapped_jaxpr) == _primitive_inventory(baseline_jaxpr)
    wrapped_shapes = tuple(leaf.shape for leaf in jax.tree.leaves(wrapped(X)))
    baseline_shapes = tuple(leaf.shape for leaf in jax.tree.leaves(baseline(X)))
    assert wrapped_shapes == baseline_shapes


def test_tensor_scan_heterogeneous_carry_and_output_base_jax_agree():
    batch, steps, width = 2, 5, 3
    values = np.arange(batch * steps * width, dtype=np.float32).reshape(
        batch,
        steps,
        width,
    )

    def run(seq, array):
        X = {"value": array(values)}
        initial = _TreeElement(
            scalar=array(np.zeros((batch, 1), dtype=np.float32)),
            vector=array(np.zeros((batch, width), dtype=np.float32)),
            layout="heterogeneous-scan",
        )

        def scan_op(carry, step):
            updated = _TreeElement(
                scalar=carry.scalar + step["value"].sum(axis=-1, keepdims=True),
                vector=carry.vector + step["value"],
                layout=carry.layout,
            )
            emitted = {
                "sum": updated.scalar,
                "nested": (updated.vector,),
            }
            return updated, emitted

        return seq.tensor_scan(
            X,
            initial=initial,
            scan_op=scan_op,
            axis=-2,
            out_axis=-2,
        )

    expected_vector = np.cumsum(values, axis=1)
    expected_scalar = np.cumsum(values.sum(axis=-1, keepdims=True), axis=1)
    for final, outputs in (run(EAGER_SEQ, np.asarray), run(SEQ, jnp.asarray)):
        assert isinstance(final, _TreeElement)
        assert final.layout == "heterogeneous-scan"
        np.testing.assert_allclose(np.asarray(final.vector), expected_vector[:, -1])
        np.testing.assert_allclose(np.asarray(final.scalar), expected_scalar[:, -1])
        np.testing.assert_allclose(np.asarray(outputs["sum"]), expected_scalar)
        np.testing.assert_allclose(
            np.asarray(outputs["nested"][0]),
            expected_vector,
        )


def test_tensor_scan_is_one_direct_jax_scan_primitive():
    X = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    initial = {"sum": jnp.zeros((3,), dtype=X.dtype)}

    def scan_op(carry, step):
        updated = {"sum": carry["sum"] + step}
        return updated, (2 * updated["sum"],)

    wrapped = lambda x: SEQ.tensor_scan(
        x,
        initial=initial,
        scan_op=scan_op,
        axis=0,
        out_axis=0,
    )
    baseline = lambda x: jax.lax.scan(scan_op, initial, x)

    wrapped_jaxpr = jax.make_jaxpr(wrapped)(X)
    baseline_jaxpr = jax.make_jaxpr(baseline)(X)
    assert _primitive_inventory(wrapped_jaxpr)["scan"] == 1
    assert _primitive_inventory(wrapped_jaxpr) == _primitive_inventory(baseline_jaxpr)


def test_tensor_update_index_supports_traced_indices_inside_scan():
    steps = {
        "index": jnp.arange(3),
        "values": {
            "matrix": jnp.arange(6, dtype=jnp.float32).reshape(3, 2),
            "nested": (jnp.arange(3, dtype=jnp.float32)[:, None] + 10,),
        },
    }
    initial = {
        "matrix": jnp.zeros((3, 2), dtype=jnp.float32),
        "nested": (jnp.zeros((3, 1), dtype=jnp.float32),),
    }

    def run(X):
        def scan_op(carry, step):
            updated = SEQ.tensor_update_index(
                carry,
                step["index"],
                step["values"],
            )
            return updated, step["index"]

        return SEQ.tensor_scan(
            X,
            initial=initial,
            scan_op=scan_op,
            axis=0,
        )

    final, indices = jax.jit(run)(steps)
    np.testing.assert_allclose(
        np.asarray(final["matrix"]),
        np.asarray(steps["values"]["matrix"]),
    )
    np.testing.assert_allclose(
        np.asarray(final["nested"][0]),
        np.asarray(steps["values"]["nested"][0]),
    )
    np.testing.assert_array_equal(np.asarray(indices), np.arange(3))


def test_eager_tensor_update_index_reports_missing_backend_capability():
    with pytest.raises(NotImplementedError, match="functional indexed updates"):
        EAGER_SEQ.tensor_update_index(
            np.zeros((2, 2), dtype=np.float32),
            0,
            np.ones((2,), dtype=np.float32),
        )


def test_tensor_map_and_scan_validate_sequence_lengths():
    with pytest.raises(ValueError, match="same mapped length"):
        SEQ.tensor_map(
            (jnp.ones((2, 1)), jnp.ones((3, 1))),
            map_op=lambda left, right: left + right,
            in_axes=(0, 0),
        )

    with pytest.raises(ValueError, match="must be nonempty"):
        SEQ.tensor_scan(
            jnp.ones((0, 2)),
            initial=jnp.zeros((2,)),
            scan_op=lambda carry, step: (carry + step, carry + step),
            axis=0,
        )


@pytest.mark.parametrize("in_tree", [False, True])
def test_reduce_and_accumulate_registered_pytree(in_tree):
    batch, steps = 2, 5
    scalar = jnp.arange(batch * steps, dtype=jnp.float64).reshape(batch, steps, 1)
    vector = jnp.arange(batch * steps * 3, dtype=jnp.float64).reshape(batch, steps, 3)
    sequence = _TreeElement(scalar=scalar, vector=vector, layout="test-layout")
    neutral = _TreeElement(
        scalar=jnp.zeros((batch, 1), dtype=scalar.dtype),
        vector=jnp.zeros((batch, 3), dtype=vector.dtype),
        layout=sequence.layout,
    )

    reduced = SEQ.tensor_reduce(
        sequence,
        reduce_op=_add,
        neutral=neutral,
        axis=-2,
        accumulate_in_tree=in_tree,
    )
    expected_reduced = _TreeElement(
        scalar=jnp.sum(scalar, axis=-2),
        vector=jnp.sum(vector, axis=-2),
        layout=sequence.layout,
    )
    _assert_tree_allclose(reduced, expected_reduced)

    prefixes = SEQ.tensor_accumulate(
        sequence,
        reduce_op=_add,
        neutral=neutral,
        axis=-2,
        output_starting_point=True,
        accumulate_in_tree=in_tree,
    )
    expected_prefixes = _TreeElement(
        scalar=jnp.concatenate([neutral.scalar[:, None, :], jnp.cumsum(scalar, axis=-2)], axis=-2),
        vector=jnp.concatenate([neutral.vector[:, None, :], jnp.cumsum(vector, axis=-2)], axis=-2),
        layout=sequence.layout,
    )
    _assert_tree_allclose(prefixes, expected_prefixes)


@pytest.mark.parametrize("first_apply_all", [False, True])
def test_abra_different_step_and_carry_pytrees(first_apply_all):
    steps, block_size = 8, 2
    delta = jnp.arange(steps * 2, dtype=jnp.float64).reshape(steps, 2) / 10.0
    raw_steps = {"increment": delta}
    neutral = _TreeElement(
        scalar=jnp.zeros((1,), dtype=delta.dtype),
        vector=jnp.zeros((2,), dtype=delta.dtype),
        layout="test-layout",
    )

    def lift(step):
        increment = step["increment"]
        return _TreeElement(
            scalar=jnp.sum(increment, keepdims=True),
            vector=increment,
            layout=neutral.layout,
        )

    def reduce_op(carry, step):
        return _add(carry, lift(step))

    result = SEQ.tensor_abra(
        raw_steps,
        reduce_op=reduce_op,
        acc_op=_add,
        neutral=neutral,
        axis=-2,
        block_size=block_size,
        accumulate=True,
        output_starting_point=True,
        first_apply_all=first_apply_all,
        reduce_in_tree=first_apply_all,
        accumulate_in_tree=True,
    )

    block_ends = jnp.arange(block_size - 1, steps, block_size)
    vector_prefixes = jnp.concatenate(
        [neutral.vector[None, :], jnp.cumsum(delta, axis=0)[block_ends]],
        axis=0,
    )
    expected = _TreeElement(
        scalar=jnp.sum(vector_prefixes, axis=-1, keepdims=True),
        vector=vector_prefixes,
        layout=neutral.layout,
    )
    _assert_tree_allclose(result, expected)


@pytest.mark.parametrize("in_tree", [False, True])
def test_accumulation_applies_non_neutral_seed_exactly_once(in_tree):
    steps = 6
    sequence = _TreeElement(
        scalar=jnp.arange(steps, dtype=jnp.float64)[:, None],
        vector=jnp.arange(2 * steps, dtype=jnp.float64).reshape(steps, 2),
        layout="seed-layout",
    )
    neutral = _TreeElement(
        scalar=jnp.zeros((1,), dtype=jnp.float64),
        vector=jnp.zeros((2,), dtype=jnp.float64),
        layout=sequence.layout,
    )
    seed = _TreeElement(
        scalar=jnp.asarray([7.0]),
        vector=jnp.asarray([11.0, 13.0]),
        layout=sequence.layout,
    )

    prefixes = SEQ.tensor_accumulate(
        sequence,
        reduce_op=_add,
        neutral=neutral,
        seed=seed,
        axis=0,
        output_starting_point=True,
        accumulate_in_tree=in_tree,
    )
    expected_prefixes = _TreeElement(
        scalar=jnp.concatenate(
            [seed.scalar[None, :], seed.scalar + jnp.cumsum(sequence.scalar, axis=0)],
            axis=0,
        ),
        vector=jnp.concatenate(
            [seed.vector[None, :], seed.vector + jnp.cumsum(sequence.vector, axis=0)],
            axis=0,
        ),
        layout=sequence.layout,
    )
    _assert_tree_allclose(prefixes, expected_prefixes)

    block_prefixes = SEQ.tensor_abra(
        sequence,
        reduce_op=_add,
        acc_op=_add,
        neutral=neutral,
        seed=seed,
        axis=0,
        block_size=2,
        accumulate=True,
        output_starting_point=True,
        reduce_in_tree=in_tree,
        accumulate_in_tree=in_tree,
    )
    block_ends = jnp.asarray([1, 3, 5])
    expected_blocks = _TreeElement(
        scalar=jnp.concatenate(
            [
                seed.scalar[None, :],
                seed.scalar + jnp.cumsum(sequence.scalar, axis=0)[block_ends],
            ],
            axis=0,
        ),
        vector=jnp.concatenate(
            [
                seed.vector[None, :],
                seed.vector + jnp.cumsum(sequence.vector, axis=0)[block_ends],
            ],
            axis=0,
        ),
        layout=sequence.layout,
    )
    _assert_tree_allclose(block_prefixes, expected_blocks)


def test_nonaccumulating_abra_prepends_seed_on_requested_axis():
    batch, steps = 2, 6
    sequence = _TreeElement(
        scalar=jnp.arange(batch * steps, dtype=jnp.float64).reshape(
            batch, steps, 1
        ),
        vector=jnp.arange(batch * steps * 2, dtype=jnp.float64).reshape(
            batch, steps, 2
        ),
        layout="nonaccumulating-seed-layout",
    )
    neutral = _TreeElement(
        scalar=jnp.zeros((batch, 1), dtype=jnp.float64),
        vector=jnp.zeros((batch, 2), dtype=jnp.float64),
        layout=sequence.layout,
    )

    result = SEQ.tensor_abra(
        sequence,
        reduce_op=_add,
        acc_op=_add,
        neutral=neutral,
        axis=-2,
        block_size=2,
        accumulate=False,
        output_starting_point=True,
    )

    expected = _TreeElement(
        scalar=jnp.concatenate(
            [
                neutral.scalar[:, None, :],
                sequence.scalar.reshape(batch, 3, 2, 1).sum(axis=2),
            ],
            axis=-2,
        ),
        vector=jnp.concatenate(
            [
                neutral.vector[:, None, :],
                sequence.vector.reshape(batch, 3, 2, 2).sum(axis=2),
            ],
            axis=-2,
        ),
        layout=sequence.layout,
    )
    _assert_tree_allclose(result, expected)
