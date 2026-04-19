from functools import wraps

from .universal import *


class Einsum(Universal[Array]):
    @wraps(Universal.tensor_product_homogeneous)
    def tensor_product_homogeneous(self, Ai: Array, Bj: Array) -> Array:
        return self.xp.einsum(
            "...i,...j->...ij",
            Ai, Bj
        ).reshape(Ai.shape[:-1] + (Ai.shape[-1] * Bj.shape[-1],))

    @wraps(Universal.tensor_inner_product_homogeneous)
    def tensor_inner_product_homogeneous(self, Ak: Array, Bk: Array) -> Array:
        return self.xp.einsum("...i,...i->...", Ak, Bk)

    @wraps(Universal.tensor_adjoint_left_homogeneous)
    def tensor_adjoint_left_homogeneous(self, Ai: Array, Yni: Array) -> Array:
        y = Yni.reshape(
            Yni.shape[:-1] + (Ai.shape[-1], Yni.shape[-1] // Ai.shape[-1])
        )
        return self.xp.einsum("...i,...iN->...N", Ai, y)

    @wraps(Universal.tensor_adjoint_right_homogeneous)
    def tensor_adjoint_right_homogeneous(self, Bj: Array, Ynj: Array) -> Array:
        y = Ynj.reshape(
            Ynj.shape[:-1] + (Ynj.shape[-1] // Bj.shape[-1], Bj.shape[-1])
        )
        return self.xp.einsum("...Nj,...j->...N", y, Bj)

    @wraps(Universal.tensor_matrix_product_right_homogeneous)
    def tensor_matrix_product_right_homogeneous(
            self,
            A: Array,
            M: Array,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> Array:
        A, row_axis, col_axis = self._canonicalize_matrix_axes(A, row_axis, col_axis)
        out = self.xp.einsum("...nka,...kl->...nla", A, M)
        return self._restore_matrix_axes(out, row_axis, col_axis)

    @wraps(Universal.tensor_matrix_product_left_homogeneous)
    def tensor_matrix_product_left_homogeneous(
            self,
            M: Array,
            A: Array,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> Array:
        A, row_axis, col_axis = self._canonicalize_matrix_axes(A, row_axis, col_axis)
        out = self.xp.einsum("...nk,...kla->...nla", M, A)
        return self._restore_matrix_axes(out, row_axis, col_axis)

    @wraps(Universal.tensor_matrix_product_homogeneous)
    def tensor_matrix_product_homogeneous(
            self,
            A: Array,
            B: Array,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> Array:
        A, row_axis, col_axis = self._canonicalize_matrix_axes(A, row_axis, col_axis)
        B, _, _ = self._canonicalize_matrix_axes(B, row_axis, col_axis)

        out = self.xp.einsum("...nka,...klb->...nlab", A, B)
        out = out.reshape(out.shape[:-2] + (out.shape[-2] * out.shape[-1],))
        return self._restore_matrix_axes(out, row_axis, col_axis)