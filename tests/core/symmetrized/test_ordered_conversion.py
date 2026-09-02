"""Conversion from partially symmetrized to ordered representation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import tensordev as td
from tensordev.core.bigraded import BigradedTensor


def _cores(coordinates):
    ordered_standard = td.bigraded_core(
        dims=(1, 2),
        max_trunc=(1, 2),
    )
    ordered = (
        ordered_standard
        if coordinates == "standard"
        else td.shear_core(ordered_standard)
    )
    return ordered, td.symmetrized_core(ordered)


def _random_tensor(core, key, *, trunc, batch, include_scalar=True):
    layout = core.resolve_layout(trunc, include_scalar=include_scalar)
    keys = jr.split(key, len(layout.grades))
    return BigradedTensor(
        tuple(
            jr.normal(
                block_key,
                batch + (layout.block_width(grade),),
                dtype=jnp.float32,
            )
            for block_key, grade in zip(keys, layout.grades)
        ),
        layout.spec,
    )


@pytest.mark.parametrize("coordinates", ("standard", "shear"))
def test_tensor_to_ordered_is_jittable_and_preserves_tensor_metadata(
    coordinates,
):
    _, core = _cores(coordinates)
    words = _random_tensor(
        core,
        jr.PRNGKey(700),
        trunc=(1, 2),
        batch=(2, 3),
        include_scalar=False,
    )

    ordered = jax.jit(
        lambda value: core.tensor_to_ordered(
            value,
            trunc=(1, 1),
            first_on=True,
        )
    )(words)

    assert ordered.spec.dims == words.spec.dims
    assert ordered.truncation == (1, 1)
    assert ordered.spec.include_scalar is False
    assert ordered.spec.coordinates == coordinates
    assert ordered.spec.representation == "ordered"
    assert ordered.batch_shape == words.batch_shape
    assert ordered.dtype == words.dtype
    for grade in ordered.grades:
        np.testing.assert_array_equal(
            ordered[grade],
            core._lift_partially_symmetrized_block(words[grade], grade),
        )


@pytest.mark.parametrize("coordinates", ("standard", "shear"))
def test_tensor_to_ordered_is_adjoint_to_partial_symmetrization(coordinates):
    ordered_core, core = _cores(coordinates)
    ordered = _random_tensor(
        ordered_core,
        jr.PRNGKey(701),
        trunc=(1, 2),
        batch=(1, 3),
    )
    words = _random_tensor(
        core,
        jr.PRNGKey(702),
        trunc=(1, 2),
        batch=(2, 1),
    )

    symmetrized = core.tensor_partially_symmetrize(ordered)
    lifted = core.tensor_to_ordered(words)
    left = sum(
        jnp.sum(words[grade] * symmetrized[grade], axis=-1)
        for grade in words.grades
    )
    right = sum(
        jnp.sum(lifted[grade] * ordered[grade], axis=-1)
        for grade in lifted.grades
    )

    np.testing.assert_allclose(left, right, atol=2e-6, rtol=2e-6)


def test_tensor_to_ordered_rejects_unsupported_cores_and_scalar_policy():
    ordered_core, core = _cores("standard")
    ordered = _random_tensor(
        ordered_core,
        jr.PRNGKey(703),
        trunc=(1, 1),
        batch=(),
    )
    words = core.tensor_partially_symmetrize(ordered)

    with pytest.raises(RuntimeError, match="partially symmetrized"):
        ordered_core.tensor_to_ordered(ordered)
    with pytest.raises(ValueError, match="first_on"):
        core.tensor_to_ordered(words, first_on=True)
