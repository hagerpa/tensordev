"""Focused correctness tests for the compact Numba shuffle path."""

from itertools import combinations, product
from math import comb

import numpy as np

import tensordev.core.numba as numba_backend
from tensordev.core.numba import Numba, NumbaTotalDegreeShufflePlanStore
from tensordev.core.shuffle import TotalDegreeShufflePlanStore


def _numba_core(d: int, max_trunc: int) -> Numba:
    return Numba(
        d=d,
        max_trunc=max_trunc,
        precompute_shuffle=True,
    )


def _flat_index(word, dimension):
    index = 0
    for letter in word:
        index = index * dimension + letter
    return index


def _reference_shuffle(left, right, dimension, left_degree, right_degree):
    left = np.asarray(left)
    right = np.asarray(right)
    batch_shape = np.broadcast_shapes(left.shape[:-1], right.shape[:-1])
    left = np.broadcast_to(
        left, batch_shape + (dimension ** left_degree,)
    ).reshape(-1, dimension ** left_degree)
    right = np.broadcast_to(
        right, batch_shape + (dimension ** right_degree,)
    ).reshape(-1, dimension ** right_degree)

    output_degree = left_degree + right_degree
    output = np.zeros(
        (left.shape[0], dimension ** output_degree), dtype=np.float64
    )
    for output_index, word in enumerate(
        product(range(dimension), repeat=output_degree)
    ):
        for left_positions in combinations(range(output_degree), left_degree):
            left_positions = set(left_positions)
            left_word = tuple(
                word[position]
                for position in range(output_degree)
                if position in left_positions
            )
            right_word = tuple(
                word[position]
                for position in range(output_degree)
                if position not in left_positions
            )
            output[:, output_index] += (
                left[:, _flat_index(left_word, dimension)]
                * right[:, _flat_index(right_word, dimension)]
            )
    return output.reshape(batch_shape + (dimension ** output_degree,))


def test_numba_backend_does_not_expose_sparse_shuffle():
    assert not hasattr(Numba, "sparse_einsum")
    assert not hasattr(numba_backend, "NumbaShuffleCore")
    assert not hasattr(NumbaTotalDegreeShufflePlanStore, "sparse_einsum")


def test_numba_permutation_shuffle_degree_one_by_one():
    core = _numba_core(d=2, max_trunc=2)
    assert isinstance(
        core.shuffle_plan_store,
        NumbaTotalDegreeShufflePlanStore,
    )
    left = np.array([[2.0, 3.0]])
    right = np.array([[5.0, 7.0]])

    actual = core.tensor_shuffle_product_homogeneous(left, right, 1, 1)

    np.testing.assert_allclose(actual, [[20.0, 29.0, 29.0, 42.0]])


def test_numba_permutation_shuffle_matches_reference_with_broadcast_batch():
    rng = np.random.default_rng(20260828)
    core = _numba_core(d=2, max_trunc=5)
    left = rng.normal(size=(2, 1, 2 ** 3))
    right = rng.normal(size=(1, 3, 2 ** 2))

    actual = core.tensor_shuffle_product_homogeneous(left, right, 3, 2)
    expected = _reference_shuffle(left, right, 2, 3, 2)

    assert actual.shape == (2, 3, 2 ** 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_numba_large_output_small_batch_uses_numpy_fallback(monkeypatch):
    rng = np.random.default_rng(42)
    core = _numba_core(d=3, max_trunc=8)
    numpy_store = TotalDegreeShufflePlanStore(d=3, max_trunc=8)
    left = rng.normal(size=(2, 1, 3 ** 4))
    right = rng.normal(size=(1, 3, 3 ** 4))

    def unexpected_numba_kernel(*_args):
        raise AssertionError("small flattened batches should use NumPy")

    monkeypatch.setattr(
        numba_backend,
        "_permutation_shuffle_nb",
        unexpected_numba_kernel,
    )
    actual = core.permutation_einsum(left, right, 4, 4)
    expected = numpy_store.apply(np, left, right, 4, 4)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_numba_direct_plans_cover_one_dimension_and_scalar_factor():
    one_dimensional = _numba_core(d=1, max_trunc=5)
    left = np.array([[2.0], [-3.0]])
    right = np.array([[5.0], [7.0]])
    actual = one_dimensional.tensor_shuffle_product_homogeneous(
        left, right, 3, 2
    )
    np.testing.assert_allclose(actual, left * right * comb(5, 3))

    core = _numba_core(d=2, max_trunc=3)
    tensor = np.arange(16.0).reshape(2, 8)
    scalar = np.array([[2.0], [-0.5]])
    actual = core.tensor_shuffle_product_homogeneous(tensor, scalar, 3, 0)
    np.testing.assert_allclose(actual, tensor * scalar)
