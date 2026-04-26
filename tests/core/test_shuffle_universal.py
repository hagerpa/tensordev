"""
Tests for ShuffleCore (numpy backend).
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, "src")

from tensordev.core.shuffle import ShuffleCore


def _make_core(N: int) -> ShuffleCore:
    return ShuffleCore(np, d=1, trunc=N)


def _dense(scalars):
    """Wrap scalars as (batch=1, dim=1) arrays."""
    return tuple(np.array([[v]]) for v in scalars)


# ---------------------------------------------------------------------------
# Structure: hash, equality, repr
# ---------------------------------------------------------------------------

class TestShuffleCoreStructure:
    def test_repr(self):
        sc = _make_core(3)
        assert "d=1" in repr(sc)
        assert "trunc=3" in repr(sc)

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

    def test_operators_populated(self):
        sc = _make_core(4)
        # All (i,j) with i>=j, i+j<=4 must be present
        for total in range(5):
            for i in range(total, -1, -1):
                j = total - i
                if j > i:
                    break
                assert (i, j) in sc.operators, f"Missing operator ({i},{j})"


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
        with pytest.raises(ValueError, match="precomputed trunc"):
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