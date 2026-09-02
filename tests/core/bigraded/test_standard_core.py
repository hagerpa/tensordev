from __future__ import annotations

from collections import Counter
from functools import partial
import re

import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from tensordev import Jax, bigraded_core, path_signature, total_degree_core
from tensordev.core.bigraded import BigradedTensor
from tensordev.core.bigraded.standard import StandardBigradedCore
from tensordev.core.universal import Universal


config.update("jax_enable_x64", True)

TOTAL_CORE = Jax()


def _random_tensor(
    core,
    key,
    *,
    trunc,
    batch_shape=(),
    include_scalar=True,
    scale=0.15,
):
    layout = core.resolve_layout(trunc, include_scalar=include_scalar)
    keys = jr.split(key, len(layout.grades))
    blocks = tuple(
        scale
        * jr.normal(
            block_key,
            batch_shape + (layout.block_width(grade),),
            dtype=jnp.float64,
        )
        for block_key, grade in zip(keys, layout.grades)
    )
    return BigradedTensor(blocks, layout.spec)


def _assert_tensor_allclose(actual, expected, *, atol=1e-11, rtol=1e-11):
    assert actual.spec == expected.spec
    assert actual.grades == expected.grades
    for grade in actual.grades:
        np.testing.assert_allclose(
            np.asarray(actual[grade]),
            np.asarray(expected[grade]),
            atol=atol,
            rtol=rtol,
            err_msg=f"bidegree {grade}",
        )


def _project_total(core, levels, *, trunc, include_scalar=True):
    return core.tensor_from_total(
        levels,
        trunc=trunc,
        include_scalar=include_scalar,
    )


@pytest.mark.parametrize(
    "name",
    (
        "tensor_summation",
        "tensor_scalar_multiply",
        "tensor_dilation",
        "tensor_inner_product",
        "tensor_product",
        "tensor_adjoint_product",
        "tensor_fmexp",
        "tensor_exponential",
        "tensor_logarithm",
        "tensor_matrix_product_right",
        "tensor_matrix_product_left",
        "tensor_matrix_product",
    ),
)
def test_public_orchestration_is_inherited_from_universal(name):
    assert name not in StandardBigradedCore.__dict__
    assert getattr(StandardBigradedCore, name) is getattr(Universal, name)


@pytest.fixture(scope="module")
def core():
    return bigraded_core(
        dims=(1, 2),
        max_trunc=(2, 2),
        default_trunc=(2, 1),
    )


def test_product_is_exact_total_projection(core):
    active = (2, 1)
    ka, kb = jr.split(jr.PRNGKey(101))
    A = _random_tensor(core, ka, trunc=active, batch_shape=(2,))
    B = _random_tensor(core, kb, trunc=active, batch_shape=(2,))

    got = core.tensor_product(A, B, trunc=active)
    total = TOTAL_CORE.tensor_product(
        core.tensor_to_total(A),
        core.tensor_to_total(B),
        trunc=sum(active),
    )
    expected = _project_total(core, total, trunc=active)

    _assert_tensor_allclose(got, expected)


def test_fused_product_grade_matches_sum_of_homogeneous_blocks(core):
    output_grade = (2, 1)
    layout = core.resolve_layout(output_grade)
    ka, kb = jr.split(jr.PRNGKey(147))
    A = _random_tensor(core, ka, trunc=output_grade, batch_shape=(2,))
    B = _random_tensor(core, kb, trunc=output_grade, batch_shape=(2,))
    contributions = tuple(
        (A[left_grade], B[right_grade], left_grade, right_grade)
        for left_grade, right_grade in layout.product_splits(output_grade)
    )

    got = core._product_grade(contributions, output_grade)
    left, right, left_grade, right_grade = contributions[0]
    expected = core._product_block(
        left, right, left_grade, right_grade, output_grade
    )
    for left, right, left_grade, right_grade in contributions[1:]:
        expected = expected + core._product_block(
            left, right, left_grade, right_grade, output_grade
        )

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), atol=1e-13, rtol=1e-13
    )


def test_product_is_associative_and_has_a_unit(core):
    active = (2, 1)
    ka, kb, kc = jr.split(jr.PRNGKey(102), 3)
    A = _random_tensor(core, ka, trunc=active)
    B = _random_tensor(core, kb, trunc=active)
    C = _random_tensor(core, kc, trunc=active)
    unit = core.tensor_exponential((), trunc=active, output_zero_level=True)

    left = core.tensor_product(core.tensor_product(A, B, trunc=active), C, trunc=active)
    right = core.tensor_product(A, core.tensor_product(B, C, trunc=active), trunc=active)

    _assert_tensor_allclose(left, right)
    _assert_tensor_allclose(core.tensor_product(unit, A, trunc=active), A)
    _assert_tensor_allclose(core.tensor_product(A, unit, trunc=active), A)


@pytest.mark.parametrize("side", ["left", "right"])
def test_adjoint_product_satisfies_the_defining_pairing(core, side):
    active = (1, 1)
    kw, kx, ky = jr.split(jr.PRNGKey(103 + (side == "right")), 3)
    W = _random_tensor(core, kw, trunc=active)
    X = _random_tensor(core, kx, trunc=active)
    Y = _random_tensor(core, ky, trunc=active)

    adjoint = core.tensor_adjoint_product(W, Y, trunc=active, side=side)
    product = (
        core.tensor_product(W, X, trunc=active)
        if side == "left"
        else core.tensor_product(X, W, trunc=active)
    )

    lhs = core.tensor_inner_product(product, Y)
    rhs = core.tensor_inner_product(X, adjoint)
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("side", ["left", "right"])
def test_adjoint_product_broadcasts_multiplier_and_target_batches(side):
    active = (1, 1)
    core = bigraded_core(
        dims=(1, 2),
        max_trunc=active,
        precompute_shuffle=False,
    )
    kw, kx, ky = jr.split(jr.PRNGKey(140 + (side == "right")), 3)
    W = _random_tensor(core, kw, trunc=active, batch_shape=(3, 1))
    X = _random_tensor(core, kx, trunc=active, batch_shape=(3, 2))
    Y = _random_tensor(core, ky, trunc=active, batch_shape=(1, 2))

    adjoint = core.tensor_adjoint_product(W, Y, trunc=active, side=side)
    product = (
        core.tensor_product(W, X, trunc=active)
        if side == "left"
        else core.tensor_product(X, W, trunc=active)
    )

    assert adjoint.batch_shape == (3, 2)
    lhs = core.tensor_inner_product(product, Y)
    rhs = core.tensor_inner_product(X, adjoint)
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-11, rtol=1e-11)


def test_total_and_flat_conversions_round_trip(core):
    active = (2, 1)
    A = _random_tensor(
        core,
        jr.PRNGKey(104),
        trunc=active,
        batch_shape=(2,),
    )

    total = core.tensor_to_total(A)
    _assert_tensor_allclose(_project_total(core, total, trunc=active), A)

    flat = core.tensor_to_flat(A)
    from_flat = core.tensor_from_flat(flat, trunc=active, include_scalar=True)
    _assert_tensor_allclose(from_flat, A)
    np.testing.assert_array_equal(
        np.asarray(core.tensor_to_flat(A, start_at_level_one=True)),
        np.asarray(jnp.concatenate(A.blocks[1:], axis=-1)),
    )

    sparse = core.tensor_densify(
        {(0, 0): A[0, 0], (1, 0): A[1, 0]},
        trunc=active,
    )
    np.testing.assert_array_equal(np.asarray(sparse[0, 0]), np.asarray(A[0, 0]))
    np.testing.assert_array_equal(np.asarray(sparse[1, 0]), np.asarray(A[1, 0]))
    for grade in sparse.grades:
        if grade not in ((0, 0), (1, 0)):
            np.testing.assert_array_equal(
                np.asarray(sparse[grade]), np.zeros_like(np.asarray(sparse[grade]))
            )


def test_sum_scaling_and_bidegree_dilations_are_blockwise(core):
    active = (2, 1)
    ka, kb = jr.split(jr.PRNGKey(105))
    A = _random_tensor(core, ka, trunc=active, batch_shape=(2,))
    B = _random_tensor(core, kb, trunc=active, batch_shape=(2,))

    summed = core.tensor_summation(A, B, trunc=active)
    scaled = core.tensor_scalar_multiply(A, 1.75)
    dilated = core.tensor_dilation(A, -0.4)
    bidilated = core.tensor_bidilation(A, 0.6, -0.3)

    for n, m in A.grades:
        np.testing.assert_allclose(np.asarray(summed[n, m]), np.asarray(A[n, m] + B[n, m]))
        np.testing.assert_allclose(np.asarray(scaled[n, m]), np.asarray(1.75 * A[n, m]))
        np.testing.assert_allclose(
            np.asarray(dilated[n, m]),
            np.asarray((-0.4) ** (n + m) * A[n, m]),
        )
        np.testing.assert_allclose(
            np.asarray(bidilated[n, m]),
            np.asarray(0.6**n * (-0.3) ** m * A[n, m]),
        )


def test_first_level_exponential_and_fmexp_match_total_projection():
    active = (2, 2)
    core = bigraded_core(dims=(1, 1), max_trunc=active)
    z = 0.2 * jr.normal(jr.PRNGKey(106), (2, 2), dtype=jnp.float64)

    got = core.tensor_exponential((z,), trunc=active, output_zero_level=True)
    total = TOTAL_CORE.tensor_exponential(
        (z,), trunc=sum(active), output_zero_level=True
    )
    expected = _project_total(core, total, trunc=active)
    _assert_tensor_allclose(got, expected)

    g = _random_tensor(core, jr.PRNGKey(107), trunc=active, batch_shape=(2,))
    fmexp = core.tensor_fmexp(g, (z,), trunc=active, output_zero_level=True)
    product = core.tensor_product(g, got, trunc=active)
    _assert_tensor_allclose(fmexp, product)

    positive_g = core.tensor_densify(
        {grade: g[grade] for grade in g.grades if grade != (0, 0)},
        trunc=active,
        include_scalar=False,
    )
    with pytest.raises(ValueError, match="g must include the scalar block"):
        core.tensor_fmexp(positive_g, (z,), trunc=active)


def test_general_exponential_and_logarithm_are_inverse(core):
    active = (2, 1)
    X = _random_tensor(
        core,
        jr.PRNGKey(108),
        trunc=active,
        include_scalar=False,
        scale=0.05,
    )

    exponential = core.tensor_exponential(X, trunc=active, output_zero_level=True)
    total_X = core.tensor_to_total(X)
    total_exponential = TOTAL_CORE.tensor_exponential(
        total_X[1:], trunc=sum(active), output_zero_level=True
    )
    expected = _project_total(core, total_exponential, trunc=active)
    _assert_tensor_allclose(exponential, expected)

    positive_exponential = core.tensor_densify(
        {
            grade: exponential[grade]
            for grade in exponential.grades
            if grade != (0, 0)
        },
        trunc=active,
        include_scalar=False,
    )
    recovered = core.tensor_logarithm(
        positive_exponential,
        trunc=active,
        output_zero_level=False,
    )
    _assert_tensor_allclose(recovered, X, atol=2e-11, rtol=2e-11)


def test_standard_shuffle_matches_total_shuffle_projection():
    active = (2, 1)
    core = bigraded_core(
        dims=(1, 1), max_trunc=active, precompute_shuffle=True
    )
    ka, kb = jr.split(jr.PRNGKey(109))
    A = _random_tensor(core, ka, trunc=active)
    B = _random_tensor(core, kb, trunc=active)

    got = core.tensor_shuffle_product(A, B, trunc=active)
    total_shuffle = total_degree_core(
        d=2,
        max_trunc=sum(active),
        precompute_shuffle=True,
    )
    total = total_shuffle.tensor_shuffle_product(
        core.tensor_to_total(A),
        core.tensor_to_total(B),
        trunc=sum(active),
    )
    expected = _project_total(core, total, trunc=active)

    _assert_tensor_allclose(got, expected)
    _assert_tensor_allclose(
        got,
        core.tensor_shuffle_product(B, A, trunc=active),
    )


def test_full_shuffle_is_associative_and_supports_vmap_and_grad():
    active = (2, 1)
    core = bigraded_core(
        dims=(1, 1), max_trunc=active, precompute_shuffle=True
    )
    ka, kb, kc = jr.split(jr.PRNGKey(144), 3)
    A = _random_tensor(core, ka, trunc=active, include_scalar=False)
    B = _random_tensor(core, kb, trunc=active, include_scalar=False)
    C = _random_tensor(core, kc, trunc=active, include_scalar=False)

    left = core.tensor_shuffle_product(
        core.tensor_shuffle_product(A, B, trunc=active), C, trunc=active
    )
    right = core.tensor_shuffle_product(
        A, core.tensor_shuffle_product(B, C, trunc=active), trunc=active
    )
    _assert_tensor_allclose(left, right, atol=2e-11, rtol=2e-11)

    batched_A = BigradedTensor(
        tuple(jnp.stack((block, 0.7 * block, -0.3 * block)) for block in A.blocks),
        A.spec,
    )
    direct = core.tensor_shuffle_product(batched_A, B, trunc=active)
    vmapped = jax.vmap(
        lambda element: core.tensor_shuffle_product(element, B, trunc=active)
    )(batched_A)
    _assert_tensor_allclose(direct, vmapped, atol=2e-11, rtol=2e-11)

    def loss(element):
        shuffled = core.tensor_shuffle_product(element, B, trunc=active)
        return jnp.sum(core.tensor_to_flat(shuffled) ** 2)

    gradient = jax.grad(loss)(A)
    assert isinstance(gradient, BigradedTensor)
    assert all(bool(jnp.all(jnp.isfinite(block))) for block in gradient.blocks)


def test_signature_coordinates_satisfy_bigraded_shuffle_identity():
    active = (1, 1)
    core = bigraded_core(
        dims=(1, 1), max_trunc=active, precompute_shuffle=True
    )
    path = jnp.asarray(
        [[0.0, 0.0], [0.2, -0.1], [-0.05, 0.3], [0.15, 0.25]],
        dtype=jnp.float64,
    )
    signature = path_signature(path, trunc=active, core=core)
    left = core.tensor_densify(
        {(1, 0): jnp.asarray([0.7], dtype=jnp.float64)},
        trunc=active,
        include_scalar=False,
    )
    right = core.tensor_densify(
        {(0, 1): jnp.asarray([-0.4], dtype=jnp.float64)},
        trunc=active,
        include_scalar=False,
    )
    shuffled = core.tensor_shuffle_product(left, right, trunc=active)

    lhs = (
        core.tensor_inner_product(left, signature)
        * core.tensor_inner_product(right, signature)
    )
    rhs = core.tensor_inner_product(shuffled, signature)
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-12, rtol=1e-12)


def test_matrix_tensor_product_matches_total_projection():
    active = (1, 1)
    core = bigraded_core(dims=(1, 1), max_trunc=active)
    ka, kb = jr.split(jr.PRNGKey(110))
    A = _random_tensor(core, ka, trunc=active, batch_shape=(2, 3))
    B = _random_tensor(core, kb, trunc=active, batch_shape=(3, 2))

    got = core.tensor_matrix_product(A, B, trunc=active)
    total = TOTAL_CORE.tensor_matrix_product(
        core.tensor_to_total(A),
        core.tensor_to_total(B),
        trunc=sum(active),
    )
    expected = _project_total(core, total, trunc=active)

    _assert_tensor_allclose(got, expected)


def test_adjoint_and_matrix_products_create_no_hidden_host_plan_caches():
    active = (1, 1)
    core = bigraded_core(
        dims=(1, 2),
        max_trunc=(2, 2),
        default_trunc=active,
        precompute_shuffle=False,
    )
    kw, ky, ka, kb = jr.split(jr.PRNGKey(142), 4)
    W = _random_tensor(core, kw, trunc=active)
    Y = _random_tensor(core, ky, trunc=active)
    A = _random_tensor(core, ka, trunc=active, batch_shape=(2, 2))
    B = _random_tensor(core, kb, trunc=active, batch_shape=(2, 2))

    reported_before = core.memory_bytes()
    statistics_before = core.plan_statistics()
    core_keys_before = set(vars(core))
    core_mapping_sizes_before = {
        name: len(value)
        for name, value in vars(core).items()
        if isinstance(value, dict)
    }
    store_mapping_sizes_before = {
        name: len(value)
        for name, value in vars(core.plan_store).items()
        if isinstance(value, dict)
    }

    adjoint = core.tensor_adjoint_product(W, Y, trunc=active, side="left")
    matrix = core.tensor_matrix_product(A, B, trunc=active)
    jax.tree_util.tree_map(lambda block: block.block_until_ready(), (adjoint, matrix))

    assert not hasattr(core, "_coordinate_target_cache")
    assert core.memory_bytes() == reported_before
    assert core.plan_statistics() == statistics_before
    assert set(vars(core)) == core_keys_before
    assert {
        name: len(value)
        for name, value in vars(core).items()
        if isinstance(value, dict)
    } == core_mapping_sizes_before
    assert {
        name: len(value)
        for name, value in vars(core.plan_store).items()
        if isinstance(value, dict)
    } == store_mapping_sizes_before


def test_active_subtruncation_uses_the_same_capacity_without_padding():
    core = bigraded_core(
        dims=(1, 1),
        max_trunc=(2, 2),
        default_trunc=(2, 2),
        precompute_shuffle=False,
    )
    ka, kb = jr.split(jr.PRNGKey(111))
    A = _random_tensor(core, ka, trunc=(2, 2))
    B = _random_tensor(core, kb, trunc=(2, 2))

    direct = core.tensor_product(A, B, trunc=(1, 1))
    view = core.at_truncation((1, 1))
    through_view = view.tensor_product(A, B)

    assert view.plan_store is core.plan_store
    assert core.at_truncation((1, 1)) is view
    assert view.at_truncation((2, 2)) is core
    assert view.tensor_product.__func__ is core.tensor_product.__func__
    assert direct.truncation == (1, 1)
    assert direct.grades == ((0, 0), (1, 0), (0, 1), (1, 1))
    _assert_tensor_allclose(direct, through_view)

    with pytest.raises(ValueError, match="exceeds core capacity"):
        core.tensor_product(A, B, trunc=(3, 1))


def test_densify_rejects_grades_outside_the_requested_layout():
    core = bigraded_core(
        dims=(1, 1),
        max_trunc=(1, 1),
        precompute_shuffle=False,
    )
    full = _random_tensor(core, jr.PRNGKey(143), trunc=(1, 1))

    with pytest.raises(ValueError, match="outside the requested layout"):
        core.tensor_densify(full, trunc=(1, 1), include_scalar=False)
    with pytest.raises(ValueError, match="outside the requested layout"):
        core.tensor_densify(
            {(0, 0): full[0, 0], (1, 0): full[1, 0]},
            trunc=(1, 1),
            include_scalar=False,
        )
    with pytest.raises(ValueError, match="outside the requested layout"):
        core.tensor_densify(
            {(1, 1): full[1, 1]},
            trunc=(1, 0),
            include_scalar=False,
        )


def test_shuffle_vector_accepts_empty_positive_zero_truncation():
    core = bigraded_core(
        dims=(1, 1), max_trunc=(1, 1), precompute_shuffle=True
    )
    empty = BigradedTensor(
        tuple(),
        core.resolve_layout((0, 0), include_scalar=False).spec,
    )
    vector = jnp.asarray([[0.2, -0.3], [0.4, 0.1]], dtype=jnp.float32)

    result = core.tensor_shuffle_vector(empty, vector, trunc=(1, 1))

    assert result.batch_shape == (2,)
    assert result.dtype == np.dtype(np.float32)
    for block in result.blocks:
        np.testing.assert_array_equal(np.asarray(block), np.zeros_like(np.asarray(block)))


def test_disabled_shuffle_fails_without_growing_plan_memory():
    active = (1, 1)
    core = bigraded_core(dims=(1, 1), max_trunc=active)
    A = _random_tensor(core, jr.PRNGKey(145), trunc=active)
    memory_before = core.memory_bytes()

    with pytest.raises(RuntimeError, match="precompute_shuffle=False"):
        core.tensor_shuffle_product(A, A, trunc=active)

    assert core.shuffle_plan_store is None
    assert core.memory_bytes() == memory_before


def test_float32_product_and_horner_preserve_dtype_and_projection():
    active = (1, 1)
    core = bigraded_core(
        dims=(1, 1),
        max_trunc=active,
        precompute_shuffle=False,
    )
    z = jnp.asarray([[0.15, -0.2], [-0.05, 0.1]], dtype=jnp.float32)
    exponential = core.tensor_exponential((z,), trunc=active)
    product = core.tensor_product(exponential, exponential, trunc=active)
    total = TOTAL_CORE.tensor_product(
        core.tensor_to_total(exponential),
        core.tensor_to_total(exponential),
        trunc=sum(active),
    )
    expected = core.tensor_from_total(total, trunc=active)

    assert exponential.dtype == np.dtype(np.float32)
    assert product.dtype == np.dtype(np.float32)
    _assert_tensor_allclose(product, expected, atol=2e-6, rtol=2e-6)


def test_bigraded_pytree_composes_with_jit_vmap_and_grad():
    active = (1, 1)
    core = bigraded_core(
        dims=(1, 1),
        max_trunc=(2, 2),
        default_trunc=active,
        precompute_shuffle=False,
    )
    z = jnp.asarray([0.17, -0.11], dtype=jnp.float64)

    exponential = partial(
        core.tensor_exponential,
        trunc=active,
        output_zero_level=True,
    )
    eager = exponential((z,))
    compiled = jax.jit(lambda value: exponential((value,)))(z)
    _assert_tensor_allclose(compiled, eager)

    zs = jnp.stack((z, 1.3 * z, -0.7 * z))
    vmapped = jax.vmap(lambda value: exponential((value,)))(zs)
    batched = exponential((zs,))
    _assert_tensor_allclose(vmapped, batched)

    def loss(value):
        tensor = exponential((value,))
        return jnp.sum(core.tensor_to_flat(tensor) ** 2)

    gradient = jax.grad(loss)(z)
    direction = jnp.asarray([0.4, -0.25], dtype=jnp.float64)
    epsilon = 1e-5
    finite_difference = (
        loss(z + epsilon * direction) - loss(z - epsilon * direction)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        np.asarray(jnp.vdot(gradient, direction)),
        np.asarray(finite_difference),
        atol=2e-8,
        rtol=2e-8,
    )


def _stablehlo_inventory(lowered):
    text = str(lowered.compiler_ir(dialect="stablehlo"))
    return Counter(re.findall(r"stablehlo\.([a-zA-Z0-9_]+)", text))


def test_product_uses_one_placement_scatter_per_output_grade():
    active = (2, 2)
    core = bigraded_core(dims=(1, 1), max_trunc=active)
    ka, kb = jr.split(jr.PRNGKey(148))
    A = _random_tensor(core, ka, trunc=active)
    B = _random_tensor(core, kb, trunc=active)

    lowered = core.tensor_product.__func__.lower(core, A, B, trunc=active)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    scatter_count = stablehlo.count('"stablehlo.scatter"')
    split_count = sum(
        len(core.resolve_layout(active).product_splits(grade))
        for grade in core.resolve_layout(active).grades
    )

    assert scatter_count == len(core.resolve_layout(active).grades)
    assert scatter_count < split_count


def test_small_active_program_does_not_capture_larger_capacity_plans():
    active = (1, 1)
    small = bigraded_core(dims=(1, 1), max_trunc=active)
    large = bigraded_core(dims=(1, 1), max_trunc=(4, 4), default_trunc=active)
    ka, kb = jr.split(jr.PRNGKey(146))
    A_small = _random_tensor(small, ka, trunc=active)
    B_small = _random_tensor(small, kb, trunc=active)
    A_large = _random_tensor(large, ka, trunc=active)
    B_large = _random_tensor(large, kb, trunc=active)

    small_product = small.tensor_product.__func__.lower(
        small, A_small, B_small, trunc=active
    )
    large_product = large.tensor_product.__func__.lower(
        large, A_large, B_large, trunc=active
    )
    assert _stablehlo_inventory(small_product) == _stablehlo_inventory(
        large_product
    )

    z = jnp.asarray([0.1, -0.2], dtype=jnp.float64)
    small_horner = small.tensor_exponential.__func__.lower(
        small, (z,), trunc=active
    )
    large_horner = large.tensor_exponential.__func__.lower(
        large, (z,), trunc=active
    )
    assert _stablehlo_inventory(small_horner) == _stablehlo_inventory(
        large_horner
    )
