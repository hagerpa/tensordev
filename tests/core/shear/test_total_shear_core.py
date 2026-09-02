"""End-to-end contracts for the dense total-degree shear core.

The numerical references in this file build ordinary dense word-coordinate
matrices from the backend-independent symbolic support.  They intentionally do
not use the production total plan store or its masked-permutation executor.
"""

from __future__ import annotations

from functools import lru_cache
import inspect
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensordev.core.jax import Jax, JaxSequentialCore
from tensordev.core.shear.jax import JaxShearTotal
from tensordev.core.shear.total import (
    TotalShearPlanBuilder,
    TotalShearPlanStore,
    apply_total_masked_permutation_plan,
)
from tensordev.core.shear.symbolic import (
    interleave,
    ordinary_shuffle_placements,
    psi_inverse_total_support,
    psi_total_support,
)
from tensordev.development import path_signature


ATOL = 3e-5
RTOL = 3e-5


def _input_word(output_word, term, d_prime):
    prime_values = tuple(value for value in output_word if value < d_prime)
    doubleprime_values = tuple(
        value for value in output_word if value >= d_prime
    )
    input_doubleprime_values = tuple(
        doubleprime_values[label]
        for label in term.doubleprime_permutation
    )
    return interleave(
        prime_values,
        input_doubleprime_values,
        term.input_prime_positions,
    )


@lru_cache(maxsize=None)
def _coordinate_matrix(dims, degree, inverse=False):
    """Return the exact dense matrix for ``Psi`` or ``Psi^-1``."""
    d_prime, d_doubleprime = dims
    dimension = d_prime + d_doubleprime
    words = tuple(product(range(dimension), repeat=degree))
    word_indices = {word: index for index, word in enumerate(words)}
    support = (
        psi_inverse_total_support(degree)
        if inverse
        else psi_total_support(degree)
    )
    rows = {}
    for term in support:
        rows.setdefault(term.output_prime_positions, []).append(term)

    matrix = np.zeros((dimension**degree, dimension**degree), dtype=np.int64)
    for output_index, output_word in enumerate(words):
        output_pattern = tuple(
            position
            for position, value in enumerate(output_word)
            if value < d_prime
        )
        for term in rows[output_pattern]:
            input_word = _input_word(output_word, term, d_prime)
            matrix[output_index, word_indices[input_word]] += term.coefficient
    return matrix


def _apply_matrix(matrix, value):
    return np.einsum("oi,...i->...o", matrix, np.asarray(value))


def _coordinate_element(element, dims, *, inverse=False, transpose=False):
    result = []
    for degree, level in enumerate(element):
        matrix = _coordinate_matrix(dims, degree, inverse=inverse)
        if transpose:
            matrix = matrix.T
        result.append(_apply_matrix(matrix, level))
    return tuple(result)


def _outer_flat(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    batch_shape = np.broadcast_shapes(left.shape[:-1], right.shape[:-1])
    left = np.broadcast_to(left, batch_shape + (left.shape[-1],))
    right = np.broadcast_to(right, batch_shape + (right.shape[-1],))
    return (left[..., :, None] * right[..., None, :]).reshape(
        batch_shape + (left.shape[-1] * right.shape[-1],)
    )


def _ordinary_product(left, right, trunc):
    levels = []
    for degree in range(trunc + 1):
        terms = [
            _outer_flat(left[left_degree], right[degree - left_degree])
            for left_degree in range(degree + 1)
            if left_degree < len(left) and degree - left_degree < len(right)
        ]
        if not terms:
            break
        levels.append(sum(terms[1:], start=terms[0]))
    return tuple(levels)


def _ordinary_shuffle(left, right, dimension, left_degree, right_degree):
    """Dense transpose-sum oracle for one ordinary shuffle product."""
    raw = _outer_flat(left, right)
    batch_ndim = raw.ndim - 1
    output_degree = left_degree + right_degree
    raw = raw.reshape(raw.shape[:-1] + (dimension,) * output_degree)
    result = np.zeros_like(raw)
    left_axes = tuple(range(left_degree))
    right_axes = tuple(range(left_degree, output_degree))
    for left_positions in ordinary_shuffle_placements(
        left_degree, right_degree
    ):
        permutation = interleave(left_axes, right_axes, left_positions)
        axes = tuple(range(batch_ndim)) + tuple(
            batch_ndim + axis for axis in permutation
        )
        result += np.transpose(raw, axes)
    return result.reshape(result.shape[:batch_ndim] + (dimension**output_degree,))


def _gamma_dual_oracle(left, right, dims, left_degree, right_degree):
    dimension = sum(dims)
    forward_left = _apply_matrix(
        _coordinate_matrix(dims, left_degree).T, left
    )
    forward_right = _apply_matrix(
        _coordinate_matrix(dims, right_degree).T, right
    )
    shuffled = _ordinary_shuffle(
        forward_left,
        forward_right,
        dimension,
        left_degree,
        right_degree,
    )
    return _apply_matrix(
        _coordinate_matrix(
            dims, left_degree + right_degree, inverse=True
        ).T,
        shuffled,
    )


def _random_element(rng, dimension, trunc, *, batch_shape=(), scale=0.15):
    return tuple(
        jnp.asarray(
            scale * rng.normal(size=batch_shape + (dimension**degree,)),
            dtype=jnp.float32,
        )
        for degree in range(trunc + 1)
    )


def _assert_element_allclose(actual, expected, *, atol=ATOL, rtol=RTOL):
    assert len(actual) == len(expected)
    for actual_level, expected_level in zip(actual, expected):
        np.testing.assert_allclose(
            np.asarray(actual_level),
            np.asarray(expected_level),
            atol=atol,
            rtol=rtol,
        )


def _inner_product(left, right):
    return sum(jnp.sum(a * b) for a, b in zip(left, right))


@pytest.fixture(scope="module")
def shear_core():
    return JaxShearTotal(
        dims=(1, 2),
        max_trunc=3,
        default_trunc=3,
        precompute_shuffle=False,
    )


@pytest.fixture(scope="module")
def gamma_generator_core():
    return JaxShearTotal(
        dims=(1, 2),
        max_trunc=3,
        precompute_shuffle="generator",
    )


@pytest.fixture(scope="module")
def gamma_full_core():
    return JaxShearTotal(
        dims=(1, 2),
        max_trunc=3,
        precompute_shuffle=True,
    )


@pytest.mark.parametrize(
    ("precompute_shuffle", "expected"),
    ((False, "False"), ("generator", "'generator'"), (True, "True")),
)
def test_repr_uses_public_shuffle_constructor_values(
    precompute_shuffle, expected
):
    core = JaxShearTotal(
        dims=(1, 1),
        max_trunc=1,
        precompute_shuffle=precompute_shuffle,
    )

    assert f"precompute_shuffle={expected}" in repr(core)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"dims": (True, 1)},
        {"dims": (1.0, 1)},
        {"max_trunc": False},
        {"max_trunc": 1.0},
        {"precompute_shuffle": 0},
    ),
)
def test_supplied_store_constructor_normalizes_before_comparison(kwargs):
    store = TotalShearPlanStore((1, 1), 1)

    with pytest.raises(TypeError):
        JaxShearTotal(plan_store=store, **kwargs)


def test_scalar_only_series_and_coordinate_maps_remain_composable(shear_core):
    unit = shear_core.tensor_exponential((), trunc=0, output_zero_level=True)
    zero = shear_core.tensor_logarithm((), trunc=0, output_zero_level=True)
    densified_zero = shear_core.tensor_densify((None,))

    assert unit[0].shape == (1,)
    assert zero[0].shape == (1,)
    assert densified_zero[0].shape == (1,)
    _assert_element_allclose(
        shear_core.tensor_product(unit, unit, trunc=0),
        unit,
    )
    _assert_element_allclose(
        shear_core.tensor_product(zero, unit, trunc=0),
        zero,
    )
    _assert_element_allclose(
        shear_core.tensor_product(densified_zero, unit, trunc=0),
        densified_zero,
    )
    _assert_element_allclose(
        shear_core.tensor_to_standard_coordinates(densified_zero, trunc=0),
        densified_zero,
    )
    _assert_element_allclose(
        shear_core.tensor_from_standard_coordinates(unit, trunc=0),
        unit,
    )
    _assert_element_allclose(
        shear_core.tensor_to_standard_coordinates(unit, trunc=0),
        unit,
    )


@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_scalar_only_flatten_preserves_empty_positive_batch_and_dtype(
    shear_core,
    batch_shape,
):
    scalar = jnp.ones(batch_shape + (1,), dtype=jnp.float32)
    flat = shear_core.tensor_to_flat(
        (scalar,),
        start_at_level_one=True,
    )

    assert flat.shape == batch_shape + (0,)
    assert flat.dtype == scalar.dtype


def test_forward_inverse_and_dense_word_matrix_oracle(shear_core):
    dims = (1, 2)
    rng = np.random.default_rng(1201)
    standard = _random_element(rng, 3, 3, batch_shape=(2, 1))
    expected_forward = _coordinate_element(standard, dims)

    shear = shear_core.tensor_from_standard_coordinates(standard, trunc=3)
    _assert_element_allclose(shear, expected_forward)
    _assert_element_allclose(
        shear_core.tensor_to_standard_coordinates(shear, trunc=3),
        standard,
    )

    arbitrary_shear = _random_element(rng, 3, 3, batch_shape=(1, 2))
    standard_again = shear_core.tensor_to_standard_coordinates(
        arbitrary_shear, trunc=3
    )
    _assert_element_allclose(
        standard_again,
        _coordinate_element(arbitrary_shear, dims, inverse=True),
    )
    _assert_element_allclose(
        shear_core.tensor_from_standard_coordinates(standard_again, trunc=3),
        arbitrary_shear,
    )


def test_forward_and_inverse_transposes_satisfy_pairing(shear_core):
    rng = np.random.default_rng(1202)
    x = _random_element(rng, 3, 3)
    y = _random_element(rng, 3, 3)

    for transform, transpose, inverse in (
        (
            shear_core.tensor_from_standard_coordinates,
            shear_core._coordinate_forward_transpose,
            False,
        ),
        (
            shear_core.tensor_to_standard_coordinates,
            shear_core._coordinate_inverse_transpose,
            True,
        ),
    ):
        transformed = transform(x, trunc=3)
        transposed = transpose(y, trunc=3, first_on=False)
        expected = _coordinate_element(y, (1, 2), inverse=inverse, transpose=True)
        _assert_element_allclose(transposed, expected)
        np.testing.assert_allclose(
            np.asarray(_inner_product(transformed, y)),
            np.asarray(_inner_product(x, transposed)),
            atol=ATOL,
            rtol=RTOL,
        )


def test_transform_supports_batch_jit_vmap_and_grad(shear_core):
    dims = (1, 2)
    rng = np.random.default_rng(1203)
    batched = _random_element(rng, 3, 3, batch_shape=(2, 3))

    compiled = jax.jit(
        lambda value: shear_core.tensor_from_standard_coordinates(
            value, trunc=3
        )
    )(batched)
    expected = _coordinate_element(batched, dims)
    _assert_element_allclose(compiled, expected)

    vmapped = jax.vmap(
        lambda value: shear_core.tensor_from_standard_coordinates(
            value, trunc=3
        )
    )(tuple(level[:, 0] for level in batched))
    _assert_element_allclose(
        vmapped,
        tuple(level[:, 0] for level in expected),
    )

    point = tuple(level[0, 0] for level in batched)

    def loss(value):
        transformed = shear_core.tensor_from_standard_coordinates(
            value, trunc=3
        )
        return sum(jnp.sum(level**2) for level in transformed)

    gradient = jax.grad(loss)(point)
    expected_gradient = tuple(
        _apply_matrix(
            _coordinate_matrix(dims, degree).T,
            2.0
            * _apply_matrix(_coordinate_matrix(dims, degree), level),
        )
        for degree, level in enumerate(point)
    )
    _assert_element_allclose(gradient, expected_gradient)


def test_affine_fallback_uses_one_device_scan_per_output_group(monkeypatch):
    monkeypatch.setattr(TotalShearPlanBuilder, "STATIC_TERM_THRESHOLD", 0)
    monkeypatch.setattr(TotalShearPlanBuilder, "FLAT_GATHER_MAX_BYTES", 0)
    monkeypatch.setattr(TotalShearPlanBuilder, "COEFFICIENT_CHUNK_SIZE", 2)

    builder = TotalShearPlanBuilder((1, 1))
    plan = builder.transform(5, inverse=False)
    coefficient_groups = tuple(
        group
        for group in plan.forward.groups
        if group.strategy == "coefficient_gather"
    )
    assert coefficient_groups
    assert any(group.affine_offsets.shape[0] > 1 for group in coefficient_groups)

    raw = np.random.default_rng(1200).normal(size=(3, 2**5)).astype(np.float32)
    expected = apply_total_masked_permutation_plan(np, raw, plan)
    actual = apply_total_masked_permutation_plan(jnp, jnp.asarray(raw), plan)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)

    jaxpr = jax.make_jaxpr(
        lambda value: apply_total_masked_permutation_plan(jnp, value, plan)
    )(jnp.asarray(raw)).jaxpr
    scan_count = sum(equation.primitive.name == "scan" for equation in jaxpr.eqns)
    assert scan_count == len(coefficient_groups)


def test_native_product_covariance_associativity_and_unit(shear_core):
    dims = (1, 2)
    rng = np.random.default_rng(1301)
    left = _random_element(rng, 3, 3)
    right = _random_element(rng, 3, 2)
    third = _random_element(rng, 3, 2)

    got = shear_core.tensor_product(left, right, trunc=3)
    standard_left = _coordinate_element(left, dims, inverse=True)
    standard_right = _coordinate_element(right, dims, inverse=True)
    expected = _coordinate_element(
        _ordinary_product(standard_left, standard_right, trunc=3), dims
    )
    _assert_element_allclose(got, expected)

    left_associated = shear_core.tensor_product(
        shear_core.tensor_product(left, right, trunc=3), third, trunc=3
    )
    right_associated = shear_core.tensor_product(
        left, shear_core.tensor_product(right, third, trunc=3), trunc=3
    )
    _assert_element_allclose(left_associated, right_associated)

    unit = (jnp.ones((1,), dtype=jnp.float32),)
    _assert_element_allclose(
        shear_core.tensor_product(unit, left, trunc=3), left
    )
    _assert_element_allclose(
        shear_core.tensor_product(left, unit, trunc=3), left
    )


def test_explicit_grade_homogeneous_product_matches_transport(shear_core):
    dims = (1, 2)
    rng = np.random.default_rng(1302)
    left = jnp.asarray(rng.normal(size=(2, 3**2)), dtype=jnp.float32)
    right = jnp.asarray(rng.normal(size=(1, 3)), dtype=jnp.float32)

    got = shear_core.tensor_product_homogeneous(
        left,
        right,
        left_grade=2,
        right_grade=1,
    )
    standard_left = _apply_matrix(_coordinate_matrix(dims, 2, True), left)
    standard_right = _apply_matrix(_coordinate_matrix(dims, 1, True), right)
    expected = _apply_matrix(
        _coordinate_matrix(dims, 3),
        _outer_flat(standard_left, standard_right),
    )
    np.testing.assert_allclose(got, expected, atol=ATOL, rtol=RTOL)


def test_native_right_generator_action_matches_transport(shear_core):
    dims = (1, 2)
    rng = np.random.default_rng(1303)
    source = jnp.asarray(rng.normal(size=(2, 3**2)), dtype=jnp.float32)
    generator = jnp.asarray(rng.normal(size=(1, 3)), dtype=jnp.float32)

    got = shear_core._right_multiply_generator_output_block(
        (source,),
        (generator,),
        predecessor_grades=(2,),
        generator_grades=(1,),
        output_grade=3,
    )
    expected = _apply_matrix(
        _coordinate_matrix(dims, 3),
        _outer_flat(
            _apply_matrix(_coordinate_matrix(dims, 2, True), source),
            generator,
        ),
    )
    np.testing.assert_allclose(got, expected, atol=ATOL, rtol=RTOL)


def test_first_level_fmexp_and_path_signature_are_covariant(shear_core):
    dims = (1, 2)
    rng = np.random.default_rng(1401)
    standard_core = Jax(d=3, max_trunc=3)
    standard_left = _random_element(rng, 3, 3)
    shear_left = tuple(
        jnp.asarray(level)
        for level in _coordinate_element(standard_left, dims)
    )
    generator = jnp.asarray(rng.normal(size=(3,)), dtype=jnp.float32)

    got = shear_core.tensor_fmexp(
        shear_left, (generator,), trunc=3, output_zero_level=True
    )
    standard = standard_core.tensor_fmexp(
        standard_left, (generator,), trunc=3, output_zero_level=True
    )
    _assert_element_allclose(got, _coordinate_element(standard, dims))

    path = jnp.asarray(
        rng.normal(size=(2, 5, 3)).cumsum(axis=1), dtype=jnp.float32
    )
    sequential = JaxSequentialCore()
    shear_signature = path_signature(
        path,
        trunc=3,
        axis=-2,
        accumulate=False,
        parallel=False,
        core=shear_core,
        seq_core=sequential,
    )
    standard_signature = path_signature(
        path,
        trunc=3,
        axis=-2,
        accumulate=False,
        parallel=False,
        core=standard_core,
        seq_core=sequential,
    )
    _assert_element_allclose(
        shear_signature,
        _coordinate_element(standard_signature, dims),
        atol=8e-5,
        rtol=8e-5,
    )


def test_gamma_generator_scope_matches_full_scope_and_dual_identity(
    gamma_generator_core,
    gamma_full_core,
):
    dims = (1, 2)
    rng = np.random.default_rng(1501)
    left = jnp.asarray(rng.normal(size=(2, 3**2)), dtype=jnp.float32)
    generator = jnp.asarray(rng.normal(size=(1, 3)), dtype=jnp.float32)
    expected = _gamma_dual_oracle(left, generator, dims, 2, 1)

    generator_result = gamma_generator_core.tensor_shuffle_vector_homogeneous(
        left, generator, 2
    )
    full_result = gamma_full_core.tensor_shuffle_vector_homogeneous(
        left, generator, 2
    )
    np.testing.assert_allclose(
        generator_result, expected, atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(full_result, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(("left_degree", "right_degree"), ((1, 1), (1, 2)))
def test_full_gamma_shuffle_matches_transpose_transport_dual_identity(
    gamma_full_core,
    left_degree,
    right_degree,
):
    dims = (1, 2)
    rng = np.random.default_rng(1502 + 10 * left_degree + right_degree)
    left = jnp.asarray(
        rng.normal(size=(2, 3**left_degree)), dtype=jnp.float32
    )
    right = jnp.asarray(
        rng.normal(size=(1, 3**right_degree)), dtype=jnp.float32
    )

    got = gamma_full_core.tensor_shuffle_product_homogeneous(
        left, right, left_degree, right_degree
    )
    expected = _gamma_dual_oracle(
        left, right, dims, left_degree, right_degree
    )
    np.testing.assert_allclose(got, expected, atol=ATOL, rtol=RTOL)


def test_matrix_product_and_adjoint_vjp_respect_shear_coordinates(shear_core):
    dims = (1, 2)
    rng = np.random.default_rng(1601)
    standard_core = Jax(d=3, max_trunc=2)
    left_matrices = tuple(
        jnp.asarray(
            rng.normal(size=(2, 2, 3**degree)), dtype=jnp.float32
        )
        for degree in range(3)
    )
    right_matrices = tuple(
        jnp.asarray(
            rng.normal(size=(2, 2, 3**degree)), dtype=jnp.float32
        )
        for degree in range(3)
    )
    standard_left = _coordinate_element(left_matrices, dims, inverse=True)
    standard_right = _coordinate_element(right_matrices, dims, inverse=True)
    expected_standard = standard_core.tensor_matrix_product(
        standard_left, standard_right, trunc=2
    )
    got = shear_core.tensor_matrix_product(
        left_matrices, right_matrices, trunc=2
    )
    _assert_element_allclose(
        got, _coordinate_element(expected_standard, dims)
    )

    fixed = _random_element(rng, 3, 2)
    variable = _random_element(rng, 3, 2)
    target = _random_element(rng, 3, 2)
    for side in ("left", "right"):
        if side == "left":
            product_function = lambda value: shear_core.tensor_product(
                fixed, value, trunc=2
            )
        else:
            product_function = lambda value: shear_core.tensor_product(
                value, fixed, trunc=2
            )
        product_value, pullback = jax.vjp(product_function, variable)
        explicit = shear_core.tensor_adjoint_product(
            fixed, target, trunc=2, side=side
        )
        automatic = pullback(target)[0]
        _assert_element_allclose(explicit, automatic)
        np.testing.assert_allclose(
            np.asarray(_inner_product(product_value, target)),
            np.asarray(_inner_product(variable, explicit)),
            atol=ATOL,
            rtol=RTOL,
        )


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    (
        (
            "tensor_adjoint_left_homogeneous",
            {"multiplier_grade": True, "output_grade": 1},
        ),
        (
            "tensor_adjoint_right_homogeneous",
            {"multiplier_grade": 1.5, "output_grade": 1},
        ),
        (
            "tensor_matrix_product_homogeneous",
            {"left_grade": "1", "right_grade": 1},
        ),
    ),
)
def test_homogeneous_grade_keywords_reject_nonintegers(
    shear_core, method_name, kwargs
):
    if "adjoint" in method_name:
        arguments = (jnp.ones((3,)), jnp.ones((9,)))
    else:
        arguments = (
            jnp.ones((2, 2, 3)),
            jnp.ones((2, 2, 3)),
        )
    with pytest.raises(TypeError, match="non-negative integer"):
        getattr(shear_core, method_name)(*arguments, **kwargs)


@pytest.mark.parametrize("invalid_grade", (True, 1.5, "1"))
def test_gamma_homogeneous_grades_reject_nonintegers(
    gamma_full_core, invalid_grade
):
    vector = jnp.ones((3,))
    with pytest.raises(TypeError):
        gamma_full_core.tensor_shuffle_product_homogeneous(
            vector, vector, invalid_grade, 1
        )
    with pytest.raises(TypeError, match="non-negative integer"):
        gamma_full_core.tensor_shuffle_vector_homogeneous(
            vector, vector, invalid_grade
        )


def test_homogeneous_adjoint_and_matrix_accept_capacity_boundary(shear_core):
    multiplier = jnp.arange(3, dtype=jnp.float32)
    target = jnp.arange(3**3, dtype=jnp.float32)
    left = shear_core.tensor_adjoint_left_homogeneous(
        multiplier,
        target,
        multiplier_grade=1,
        output_grade=2,
    )
    right = shear_core.tensor_adjoint_right_homogeneous(
        multiplier,
        target,
        multiplier_grade=1,
        output_grade=2,
    )
    assert left.shape == right.shape == (3**2,)

    matrix = shear_core.tensor_matrix_product_homogeneous(
        jnp.ones((2, 2, 3)),
        jnp.ones((2, 2, 3**2)),
        left_grade=1,
        right_grade=2,
    )
    assert matrix.shape == (2, 2, 3**3)


def _walk_owned_objects(root):
    """Walk small Python plan objects without descending into array payloads."""
    stack = [root]
    seen = set()
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, (np.ndarray, jax.Array)) or callable(value):
            continue
        yield value
        if isinstance(value, dict):
            stack.extend(value.keys())
            stack.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            stack.extend(value)
        elif hasattr(value, "__dict__"):
            stack.extend(vars(value).values())


def test_total_core_owns_no_bidegree_plans_or_rank_metadata(shear_core):
    assert shear_core.grading == "total_degree"
    assert shear_core.coordinates == "shear"
    assert all(
        "bigraded" not in cls.__module__.lower()
        and "bigraded" not in cls.__name__.lower()
        for cls in type(shear_core).__mro__
    )

    shear_sources = "\n".join(
        inspect.getsource(cls)
        for cls in type(shear_core).__mro__
        if cls.__module__.startswith("tensordev.core.shear")
    ).lower()
    assert "colex_rank" not in shear_sources
    assert "bigradedplanstore" not in shear_sources

    plan_objects = []
    for value in _walk_owned_objects(shear_core):
        value_type = type(value)
        qualified_name = f"{value_type.__module__}.{value_type.__name__}".lower()
        assert ".bigraded" not in qualified_name
        if "plan" in qualified_name or "store" in qualified_name:
            plan_objects.append(value)
            if hasattr(value, "__dict__"):
                assert all("rank" not in name.lower() for name in vars(value))
    assert plan_objects, "the bounded shear core must own precomputed plans"
