import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensordev.core.bigraded import BigradedSpec, BigradedTensor


def _tensor(spec, *, batch_shape=(2,), dtype=jnp.float32):
    return BigradedTensor(
        tuple(
            jnp.zeros(batch_shape + (spec.block_width(grade),), dtype=dtype)
            for grade in spec.grades
        ),
        spec,
    )


def test_spec_canonical_grade_order_and_indices():
    spec = BigradedSpec(2, 3, (2, 2))
    assert spec.grades == (
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
        (2, 1),
        (1, 2),
        (2, 2),
    )
    for index, grade in enumerate(spec.grades):
        assert spec.index(grade) == index
    assert spec.block_width((1, 1)) == 2 * 2 * 3
    assert spec.block_width((2, 1)) == 3 * 2**2 * 3


def test_spec_partial_symmetrization_defaults_without_changing_positional_construction():
    spec = BigradedSpec(2, 3, (2, 1), "shear", False)

    assert spec.coordinates == "shear"
    assert spec.include_scalar is False
    assert spec.partially_symmetrized is False
    assert spec == BigradedSpec(
        2,
        3,
        (2, 1),
        coordinates="shear",
        include_scalar=False,
        partially_symmetrized=False,
    )


def test_partially_symmetrized_rank_counts_and_block_widths():
    spec = BigradedSpec(
        2,
        3,
        (2, 2),
        partially_symmetrized=True,
    )

    assert spec.rank_count((0, 0)) == 1
    assert spec.rank_count((0, 2)) == 6
    assert spec.rank_count((1, 2)) == 21
    assert spec.block_width((0, 2)) == 6
    assert spec.block_width((1, 2)) == 42
    # ``placement_count`` counts ordinary-word placements.
    assert spec.placement_count((1, 2)) == 3


def test_with_scalar_preserves_coordinates_and_partial_symmetrization():
    spec = BigradedSpec(
        1,
        2,
        (1, 1),
        coordinates="shear",
        partially_symmetrized=True,
    )

    positive = spec.with_scalar(False)

    assert positive.coordinates == "shear"
    assert positive.partially_symmetrized is True
    assert positive.include_scalar is False


@pytest.mark.parametrize("value", ("other", 1))
def test_spec_rejects_nonboolean_partial_symmetrization(value):
    with pytest.raises(TypeError, match="partially_symmetrized must be a bool"):
        BigradedSpec(1, 1, (1, 1), partially_symmetrized=value)


def test_spec_without_scalar_is_structurally_distinct():
    with_scalar = BigradedSpec(1, 1, (1, 1), include_scalar=True)
    positive = with_scalar.with_scalar(False)
    assert positive.grades == ((1, 0), (0, 1), (1, 1))
    assert not positive.contains((0, 0))
    assert positive != with_scalar
    assert hash(positive) != hash(with_scalar)
    with pytest.raises(KeyError, match="not present"):
        positive.index((0, 0))


@pytest.mark.parametrize(
    "args, error",
    [
        ((0, 1, (1, 1)), ValueError),
        ((1, -1, (1, 1)), ValueError),
        ((True, 1, (1, 1)), TypeError),
        ((1, 1, (-1, 1)), ValueError),
        ((1, 1, (1,)), TypeError),
    ],
)
def test_spec_rejects_invalid_parameters(args, error):
    with pytest.raises(error):
        BigradedSpec(*args)


def test_spec_accepts_shear_but_rejects_unknown_coordinates_and_nonbool_scalar_policy():
    assert BigradedSpec(1, 1, (1, 1), coordinates="shear").coordinates == "shear"
    with pytest.raises(ValueError, match="standard.*shear"):
        BigradedSpec(1, 1, (1, 1), coordinates="unknown")
    with pytest.raises(TypeError, match="include_scalar"):
        BigradedSpec(1, 1, (1, 1), include_scalar=1)


def test_tensor_validates_and_supports_true_bidegree_access():
    spec = BigradedSpec(2, 1, (2, 1))
    tensor = BigradedTensor(
        tuple(
            jnp.full((3, spec.block_width(grade)), index, dtype=jnp.float32)
            for index, grade in enumerate(spec.grades)
        ),
        spec,
    )
    np.testing.assert_array_equal(tensor[1, 1], tensor.blocks[spec.index((1, 1))])
    np.testing.assert_array_equal(tensor.block(2, 1), tensor[2, 1])
    assert tensor.batch_shape == (3,)
    assert tensor.dtype == np.dtype("float32")
    assert tensor.truncation == (2, 1)
    with pytest.raises(KeyError):
        _ = tensor[0, 2]


def test_tensor_prefix_slice_uses_upper_exclusive_bidegree_stops():
    spec = BigradedSpec(1, 1, (5, 2))
    tensor = BigradedTensor(
        tuple(
            jnp.full((2, spec.block_width(grade)), index, dtype=jnp.float32)
            for index, grade in enumerate(spec.grades)
        ),
        spec,
    )

    cropped = tensor[:5, :2]

    assert cropped.truncation == (4, 1)
    assert cropped.grades == BigradedSpec(1, 1, (4, 1)).grades
    assert cropped.batch_shape == tensor.batch_shape
    for grade in cropped.grades:
        assert cropped[grade] is tensor[grade]


@pytest.mark.parametrize(
    ("coordinates", "partially_symmetrized", "include_scalar"),
    [
        ("standard", False, True),
        ("shear", False, False),
        ("standard", True, False),
        ("shear", True, True),
    ],
)
def test_tensor_prefix_slice_preserves_static_layout_metadata(
    coordinates,
    partially_symmetrized,
    include_scalar,
):
    spec = BigradedSpec(
        1,
        2,
        (3, 2),
        coordinates=coordinates,
        include_scalar=include_scalar,
        partially_symmetrized=partially_symmetrized,
    )
    tensor = _tensor(spec, batch_shape=(2, 3))

    cropped = tensor[:3, :2]

    assert cropped.spec.dims == spec.dims
    assert cropped.spec.truncation == (2, 1)
    assert cropped.spec.coordinates == coordinates
    assert cropped.spec.partially_symmetrized is partially_symmetrized
    assert cropped.spec.include_scalar is include_scalar
    assert cropped.batch_shape == (2, 3)


def test_tensor_full_or_oversized_prefix_slice_returns_same_object():
    tensor = _tensor(BigradedSpec(1, 1, (2, 3)))

    assert tensor[:, :] is tensor
    assert tensor[0::1, 0::1] is tensor
    assert tensor[:100, :100] is tensor


def test_tensor_prefix_slice_can_produce_empty_positive_layout():
    tensor = _tensor(
        BigradedSpec(1, 1, (2, 2), include_scalar=False),
        batch_shape=(3,),
    )

    cropped = tensor[:1, :1]

    assert cropped.truncation == (0, 0)
    assert cropped.spec.include_scalar is False
    assert cropped.grades == ()
    assert cropped.blocks == ()


def test_tensor_prefix_slice_is_valid_under_jit():
    tensor = _tensor(
        BigradedSpec(
            1,
            2,
            (2, 2),
            partially_symmetrized=True,
        )
    )

    cropped = jax.jit(lambda value: value[:2, :2])(tensor)

    assert isinstance(cropped, BigradedTensor)
    assert cropped.truncation == (1, 1)
    assert cropped.spec.partially_symmetrized is True


@pytest.mark.parametrize(
    ("key", "error", "match"),
    [
        (slice(None, 2), TypeError, "two prefix slices"),
        ((slice(None, 2), 1), TypeError, "two prefix slices"),
        ((slice(1, 2), slice(None, 2)), ValueError, "start at 0"),
        ((slice(0.0, 2), slice(None, 2)), TypeError, "start"),
        ((slice(None, 2, 2), slice(None, 2)), ValueError, "step must be 1"),
        ((slice(None, 2, 1.0), slice(None, 2)), TypeError, "step"),
        ((slice(None, 0), slice(None, 2)), ValueError, "stop must be positive"),
        ((slice(None, -1), slice(None, 2)), ValueError, "stop must be positive"),
        ((slice(None, True), slice(None, 2)), TypeError, "positive integer"),
        ((slice(None, 1.5), slice(None, 2)), TypeError, "positive integer"),
    ],
)
def test_tensor_prefix_slice_rejects_nonrectangular_or_invalid_keys(
    key,
    error,
    match,
):
    tensor = _tensor(BigradedSpec(1, 1, (2, 2)))

    with pytest.raises(error, match=match):
        _ = tensor[key]


def test_tensor_accepts_partially_symmetrized_spec_widths():
    spec = BigradedSpec(
        2,
        2,
        (1, 2),
        partially_symmetrized=True,
    )
    tensor = _tensor(spec)

    assert tensor.spec is spec
    assert tensor[1, 2].shape == (2, spec.rank_count((1, 2)) * 2)


def test_tensor_validation_rejects_malformed_blocks():
    spec = BigradedSpec(2, 1, (1, 1))
    valid = _tensor(spec)
    with pytest.raises(ValueError, match="expected 4 blocks"):
        BigradedTensor(valid.blocks[:-1], spec)

    wrong_width = list(valid.blocks)
    wrong_width[2] = jnp.zeros((2, wrong_width[2].shape[-1] + 1))
    with pytest.raises(ValueError, match="final width"):
        BigradedTensor(tuple(wrong_width), spec)

    wrong_batch = list(valid.blocks)
    wrong_batch[1] = jnp.zeros((3, wrong_batch[1].shape[-1]))
    with pytest.raises(ValueError, match="batch shape"):
        BigradedTensor(tuple(wrong_batch), spec)

    wrong_dtype = list(valid.blocks)
    wrong_dtype[1] = jnp.zeros(wrong_dtype[1].shape, dtype=jnp.int32)
    with pytest.raises(TypeError, match="same dtype"):
        BigradedTensor(tuple(wrong_dtype), spec)


def test_empty_positive_tensor_at_zero_truncation_is_valid():
    spec = BigradedSpec(2, 3, (0, 0), include_scalar=False)
    tensor = BigradedTensor((), spec)
    assert tensor.grades == ()
    assert tensor.batch_shape == ()
    assert tensor.dtype is None


def test_registered_pytree_preserves_static_spec_under_jit_vmap_and_grad():
    spec = BigradedSpec(2, 1, (1, 1))
    tensor = _tensor(spec, batch_shape=())
    leaves, treedef = jax.tree.flatten(tensor)
    assert len(leaves) == spec.size
    assert jax.tree.unflatten(treedef, leaves).spec is spec

    doubled = jax.jit(lambda x: jax.tree.map(lambda block: 2 * block, x))(tensor)
    assert isinstance(doubled, BigradedTensor)
    assert doubled.spec == spec

    batched = _tensor(spec, batch_shape=(4,))
    vmapped = jax.vmap(lambda x: jax.tree.map(lambda block: block + 1, x))(batched)
    assert vmapped.batch_shape == (4,)
    assert vmapped.spec == spec

    def loss(x):
        return sum(jnp.sum(block**2) for block in x.blocks)

    gradient = jax.grad(loss)(tensor)
    assert isinstance(gradient, BigradedTensor)
    assert gradient.spec == spec


def test_registered_pytree_is_valid_scan_carry():
    spec = BigradedSpec(1, 1, (1, 1))
    initial = _tensor(spec, batch_shape=())

    def step(carry, value):
        updated = jax.tree.map(lambda block: block + value, carry)
        return updated, updated

    final, outputs = jax.lax.scan(step, initial, jnp.arange(3, dtype=jnp.float32))
    assert isinstance(final, BigradedTensor)
    assert isinstance(outputs, BigradedTensor)
    assert outputs.batch_shape == (3,)
