"""Backend-neutral ordinary shuffle-plan and NumPy execution tests."""

from itertools import combinations

import numpy as np
import pytest

from tensordev.core.shuffle import (
    HomogeneousShufflePlan,
    TotalDegreeShufflePlanStore,
    ordinary_axis_permutations,
)
from tensordev.core.numba import Numba


def _make_core(N: int) -> Numba:
    return Numba(d=1, max_trunc=N, precompute_shuffle=True)


def _dense(scalars):
    """Wrap scalars as (batch=1, dim=1) arrays."""
    return tuple(np.array([[v]]) for v in scalars)


def _digits(index, dimension, degree):
    if degree == 0:
        return ()
    return np.unravel_index(index, (dimension,) * degree)


def _manual_shuffle_homogeneous(left, right, dimension, left_degree, right_degree):
    """Coordinate-loop reference independent of the permutation kernel."""
    batch_shape = np.broadcast_shapes(left.shape[:-1], right.shape[:-1])
    left = np.broadcast_to(left, batch_shape + (dimension**left_degree,))
    right = np.broadcast_to(right, batch_shape + (dimension**right_degree,))
    output_degree = left_degree + right_degree
    output = np.zeros(
        batch_shape + (dimension**output_degree,),
        dtype=np.result_type(left, right),
    )
    for batch_index in np.ndindex(batch_shape):
        for left_index in range(dimension**left_degree):
            left_word = _digits(left_index, dimension, left_degree)
            for right_index in range(dimension**right_degree):
                right_word = _digits(right_index, dimension, right_degree)
                value = (
                    left[batch_index + (left_index,)]
                    * right[batch_index + (right_index,)]
                )
                for left_positions in combinations(
                    range(output_degree), left_degree
                ):
                    positions = set(left_positions)
                    left_iter = iter(left_word)
                    right_iter = iter(right_word)
                    output_word = tuple(
                        next(left_iter) if position in positions else next(right_iter)
                        for position in range(output_degree)
                    )
                    output_index = np.ravel_multi_index(
                        output_word, (dimension,) * output_degree
                    )
                    output[batch_index + (output_index,)] += value
    return output


# ---------------------------------------------------------------------------
# Structure: hash, equality, repr
# ---------------------------------------------------------------------------

class TestShuffleCoreStructure:
    def test_repr(self):
        sc = _make_core(3)
        assert sc.d == 1
        assert sc.max_truncation == 3

    def test_hash_is_identity(self):
        sc = _make_core(2)
        assert hash(sc) == id(sc)

    def test_eq_is_identity(self):
        sc1 = _make_core(2)
        sc2 = _make_core(2)
        assert sc1 == sc1
        assert sc1 != sc2

    def test_hashable_in_dict(self):
        sc = _make_core(2)
        d = {sc: "value"}
        assert d[sc] == "value"

    def test_plans_populated(self):
        sc = _make_core(4)
        # All (i,j) with i>=j, i+j<=4 must be present
        for total in range(5):
            for i in range(total, -1, -1):
                j = total - i
                if j > i:
                    break
                assert (i, j) in sc.shuffle_plan_store.plans

    def test_plans_are_compact_and_sparse_api_is_absent(self):
        sc = Numba(d=2, max_trunc=4, precompute_shuffle=True)
        assert all(
            isinstance(plan, HomogeneousShufflePlan)
            for plan in sc.shuffle_plan_store.plans.values()
        )
        assert not hasattr(sc, "operators")
        assert not hasattr(sc, "sparse_einsum")

    def test_memory_reports_only_permutation_payloads(self):
        sc = Numba(d=2, max_trunc=8, precompute_shuffle=True)
        expected_bytes = sum(
            plan.axis_permutations.nbytes
            for plan in sc.shuffle_plan_store.plans.values()
        )
        assert sc.memory_bytes_by_category() == {
            "axis_permutations": expected_bytes
        }
        assert sc.memory_bytes() == expected_bytes
        assert sc.memory_mb() == sc.memory_bytes() / 1024**2

        plan = sc.shuffle_plan_store.plans[(4, 4)]
        assert plan.axis_permutations.shape == (70, 8)
        assert plan.memory_bytes() == 70 * 8 * np.dtype(np.int8).itemsize

        statistics = sc.plan_statistics()
        assert statistics["d"] == 2
        assert statistics["max_truncation"] == 8
        assert statistics["default_truncation"] == 8
        assert statistics["plan_count"] == len(sc.shuffle_plan_store.plans)
        assert statistics["permutation_count"] == sum(
            item.permutation_count
            for item in sc.shuffle_plan_store.plans.values()
        )
        assert statistics["direct_plan_count"] == 9
        assert statistics["memory_bytes"] == expected_bytes
        assert statistics["shuffle_enabled"]


class TestPermutationPlans:
    def test_axis_permutations_are_cached_read_only_tables(self):
        permutations = ordinary_axis_permutations(2, 1)
        np.testing.assert_array_equal(
            permutations,
            np.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [2, 0, 1],
                ]
            ),
        )
        assert not permutations.flags.writeable
        assert ordinary_axis_permutations(2, 1) is permutations

    def test_scalar_dimension_uses_direct_binomial_scaling(self):
        store = TotalDegreeShufflePlanStore(d=1, max_trunc=8)
        plan = store.plans[(4, 4)]
        assert plan.uses_direct_scaling
        assert plan.direct_multiplicity == 70
        assert plan.permutation_count == 70
        assert plan.axis_permutations.shape == (0, 8)
        assert plan.memory_bytes() == 0

    def test_degree_zero_uses_direct_multiplication(self):
        sc = Numba(d=3, max_trunc=3, precompute_shuffle=True)
        plan = sc.shuffle_plan_store.plans[(3, 0)]
        assert plan.uses_direct_scaling
        assert plan.direct_multiplicity == 1
        left = np.arange(27.0).reshape(1, 27)
        right = np.array([[2.0]])
        np.testing.assert_array_equal(
            sc.tensor_shuffle_product_homogeneous(left, right, 3, 0),
            2.0 * left,
        )


# ---------------------------------------------------------------------------
# tensor_shuffle_product_homogeneous
# ---------------------------------------------------------------------------

class TestShuffleHomogeneous:
    def setup_method(self):
        self.sc = _make_core(4)

    def test_degree_0_0(self):
        Ai = np.array([[2.0]])
        Bj = np.array([[3.0]])
        res = self.sc.tensor_shuffle_product_homogeneous(Ai, Bj, 0, 0)
        np.testing.assert_allclose(res, [[6.0]])

    def test_degree_1_0(self):
        Ai = np.array([[5.0]])
        Bj = np.array([[2.0]])
        res = self.sc.tensor_shuffle_product_homogeneous(Ai, Bj, 1, 0)
        np.testing.assert_allclose(res, [[10.0]])

    def test_degree_1_1(self):
        # C(2,1) = 2
        Ai = np.array([[3.0]])
        Bj = np.array([[4.0]])
        res = self.sc.tensor_shuffle_product_homogeneous(Ai, Bj, 1, 1)
        np.testing.assert_allclose(res, [[2.0 * 3.0 * 4.0]])

    def test_degree_2_1(self):
        # C(3,2) = 3
        Ai = np.array([[2.0]])
        Bj = np.array([[5.0]])
        res = self.sc.tensor_shuffle_product_homogeneous(Ai, Bj, 2, 1)
        np.testing.assert_allclose(res, [[3.0 * 2.0 * 5.0]])

    @pytest.mark.parametrize(
        ("dimension", "left_degree", "right_degree"),
        [(2, 1, 1), (2, 2, 1), (3, 2, 2)],
    )
    def test_non_scalar_matches_coordinate_reference(
        self,
        dimension,
        left_degree,
        right_degree,
    ):
        sc = Numba(
            d=dimension,
            max_trunc=left_degree + right_degree,
            precompute_shuffle=True,
        )
        rng = np.random.default_rng(
            100 * dimension + 10 * left_degree + right_degree
        )
        left = rng.normal(size=(2, 1, dimension**left_degree))
        right = rng.normal(size=(1, 3, dimension**right_degree))
        actual = sc.tensor_shuffle_product_homogeneous(
            left,
            right,
            left_degree,
            right_degree,
        )
        expected = _manual_shuffle_homogeneous(
            left,
            right,
            dimension,
            left_degree,
            right_degree,
        )
        assert actual.shape == (2, 3, dimension ** (left_degree + right_degree))
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_rejects_incorrect_level_width(self):
        sc = Numba(d=2, max_trunc=3, precompute_shuffle=True)
        with pytest.raises(ValueError, match="left degree-2 level has width 3"):
            sc.tensor_shuffle_product_homogeneous(
                np.zeros((1, 3)),
                np.zeros((1, 2)),
                2,
                1,
            )


# ---------------------------------------------------------------------------
# tensor_shuffle_product (full graded)
# ---------------------------------------------------------------------------

class TestShuffleProduct:
    def setup_method(self):
        self.sc = _make_core(6)

    def test_constant_times_constant(self):
        A = _dense([2.0])
        B = _dense([3.0])
        C = self.sc.tensor_shuffle_product(A, B)
        assert len(C) == 1
        np.testing.assert_allclose(C[0], [[6.0]])

    def test_degree1_times_degree1(self):
        # C_0 = a0*b0,  C_1 = a0*b1 + a1*b0,  C_2 = C(2,1)*a1*b1 = 2*a1*b1
        A = _dense([1.0, 2.0])
        B = _dense([1.0, 3.0])
        C = self.sc.tensor_shuffle_product(A, B)
        assert len(C) == 3
        np.testing.assert_allclose(C[0], [[1.0]])
        np.testing.assert_allclose(C[1], [[1.0 * 3.0 + 2.0 * 1.0]])  # 5
        np.testing.assert_allclose(C[2], [[2.0 * 2.0 * 3.0]])         # 12

    def test_trunc(self):
        A = _dense([1.0, 2.0])
        B = _dense([1.0, 3.0])
        C = self.sc.tensor_shuffle_product(A, B, trunc=1)
        assert len(C) == 2

    def test_trunc_exceeds_precomputed_raises(self):
        sc = _make_core(4)
        A = _dense([1.0, 2.0])
        B = _dense([1.0, 2.0])
        with pytest.raises(ValueError, match="core capacity"):
            sc.tensor_shuffle_product(A, B, trunc=5)  # explicit trunc > self.trunc=4

    def test_no_explicit_trunc_silently_caps(self):
        # NA+NB=6 > self.trunc=4: output should be capped at degree 4, no error
        sc = _make_core(4)
        A = _dense([1.0, 2.0, 0.5, 1.0])  # NA=3
        B = _dense([1.0, 2.0, 0.5, 1.0])  # NB=3
        C = sc.tensor_shuffle_product(A, B)
        assert len(C) == 5  # degrees 0..4

    def test_commutativity(self):
        A = _dense([1.0, 2.0, 0.5])
        B = _dense([3.0, 1.0, 4.0])
        C_AB = self.sc.tensor_shuffle_product(A, B)
        C_BA = self.sc.tensor_shuffle_product(B, A)
        for k in range(len(C_AB)):
            np.testing.assert_allclose(C_AB[k], C_BA[k], err_msg=f"degree {k}")

    def test_a_first_on(self):
        A = _dense([2.0])       # A_1 = 2
        B = _dense([3.0, 4.0])  # B_0 = 3, B_1 = 4
        C = self.sc.tensor_shuffle_product(A, B, a_first_on=True)
        # degree 1: C(1,1)*A1*B0 = 1*2*3 = 6
        # degree 2: C(2,1)*A1*B1 = 2*2*4 = 16
        assert len(C) == 2
        np.testing.assert_allclose(C[0], [[6.0]])
        np.testing.assert_allclose(C[1], [[16.0]])

    def test_first_on_out(self):
        A = _dense([1.0, 2.0])
        B = _dense([1.0, 3.0])
        C_full = self.sc.tensor_shuffle_product(A, B)
        C_drop0 = self.sc.tensor_shuffle_product(A, B, first_on_out=True)
        assert len(C_drop0) == len(C_full) - 1
        for k in range(len(C_drop0)):
            np.testing.assert_allclose(C_drop0[k], C_full[k + 1])

    def test_both_first_on_zero_padding_degree1(self):
        A = _dense([2.0])  # A_1
        B = _dense([3.0])  # B_1
        C = self.sc.tensor_shuffle_product(A, B, a_first_on=True, b_first_on=True)
        assert len(C) == 2
        np.testing.assert_allclose(C[0], [[0.0]])              # degree 1 = zero
        np.testing.assert_allclose(C[1], [[2.0 * 2.0 * 3.0]]) # C(2,1)*A1*B1

    def test_batch(self):
        batch = 5
        A = tuple(np.random.randn(batch, 1) for _ in range(3))
        B = tuple(np.random.randn(batch, 1) for _ in range(2))
        C = self.sc.tensor_shuffle_product(A, B)
        assert len(C) == 4
        for k, Ck in enumerate(C):
            assert Ck.shape == (batch, 1), f"degree {k}: wrong shape {Ck.shape}"
