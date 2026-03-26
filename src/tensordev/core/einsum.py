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
