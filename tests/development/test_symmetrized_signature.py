from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
from jax import config
import numpy as np
import pytest

from tensordev import make_core, shear_core, symmetrize_core
from tensordev.core.bigraded import BigradedTensor
from tensordev.development import free_development, path_signature


config.update("jax_enable_x64", True)


def _assert_close(actual, expected, *, atol=2e-10, rtol=2e-10):
    assert actual.spec == expected.spec
    for grade in actual.grades:
        np.testing.assert_allclose(
            actual[grade], expected[grade], atol=atol, rtol=rtol
        )


@pytest.fixture(scope="module")
def cores():
    kwargs = dict(dims=(1, 2), max_trunc=(2, 2), default_trunc=(2, 1))
    return (
        make_core(**kwargs),
        make_core(**kwargs, partially_symmetrized=True),
    )


@pytest.mark.parametrize("parallel", (False, True))
def test_direct_quotient_signature_equals_q_of_ordered(cores, parallel):
    ordered, quotient = cores
    path = 0.1 * jr.normal(jr.PRNGKey(401 + parallel), (2, 7, 3))

    ordered_signature = path_signature(
        path,
        trunc=(2, 1),
        axis=-2,
        accumulate=False,
        parallel=parallel,
        core=ordered,
    )
    direct = path_signature(
        path,
        trunc=(2, 1),
        axis=-2,
        accumulate=False,
        parallel=parallel,
        core=quotient,
    )

    _assert_close(
        direct,
        quotient.tensor_partially_symmetrize(ordered_signature),
    )


def test_accumulated_quotient_signature_preserves_pytree_blocks(cores):
    ordered, quotient = cores
    path = 0.08 * jr.normal(jr.PRNGKey(410), (2, 9, 3))
    kwargs = dict(
        trunc=(2, 1),
        axis=-2,
        block_size=2,
        accumulate=True,
        accumulate_in_tree=True,
        output_starting_point=True,
    )

    direct = path_signature(path, core=quotient, **kwargs)
    ordered_signature = path_signature(path, core=ordered, **kwargs)

    _assert_close(
        direct,
        quotient.tensor_partially_symmetrize(ordered_signature),
    )


@pytest.mark.parametrize("parallel", (False, True))
def test_direct_quotient_shear_signature_obeys_both_commuting_routes(
    cores,
    parallel,
):
    ordered_standard, quotient_standard = cores
    ordered_shear = shear_core(ordered_standard)
    quotient_shear = symmetrize_core(ordered_shear)
    path = 0.09 * jr.normal(jr.PRNGKey(415 + parallel), (2, 7, 3))
    options = dict(
        trunc=(2, 1),
        axis=-2,
        accumulate=False,
        parallel=parallel,
    )

    direct = path_signature(path, core=quotient_shear, **options)
    ordered_shear_signature = path_signature(
        path,
        core=ordered_shear,
        **options,
    )
    quotient_standard_signature = path_signature(
        path,
        core=quotient_standard,
        **options,
    )

    _assert_close(
        direct,
        quotient_shear.tensor_partially_symmetrize(
            ordered_shear_signature
        ),
    )
    _assert_close(
        direct,
        quotient_shear.tensor_from_standard_coordinates(
            quotient_standard_signature
        ),
    )


def test_free_development_accepts_nontrivial_higher_grade_seed(cores):
    _, core = cores
    path = 0.07 * jr.normal(jr.PRNGKey(420), (2, 7, 3))
    layout = core.resolve_layout((2, 1), include_scalar=True)
    keys = jr.split(jr.PRNGKey(421), len(layout.grades))
    starting_point = BigradedTensor(
        tuple(
            jr.normal(
                key,
                (2, layout.block_width(grade)),
                dtype=jnp.float64,
            )
            for key, grade in zip(keys, layout.grades)
        ),
        layout.spec,
    )
    # A nonzero higher bidegree distinguishes this from unit and first-level
    # seeds.
    assert np.linalg.norm(np.asarray(starting_point[2, 1])) > 0

    plain = free_development(
        (path,),
        trunc=(2, 1),
        axis=-2,
        accumulate=False,
        core=core,
    )
    seeded = free_development(
        (path,),
        trunc=(2, 1),
        axis=-2,
        accumulate=False,
        starting_point=starting_point,
        core=core,
    )

    _assert_close(
        seeded,
        core.tensor_product(starting_point, plain, trunc=(2, 1)),
    )
