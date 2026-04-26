from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union, List, TypeVar, Generic

import array_api._2023_12 as array_types

from tensordev.core.utils.annotations import jit as dummy_jit

Array = TypeVar("Array", bound=array_types.Array)
DenseElem = Tuple[Array, ...]
DenseElemFirstOn = Tuple[Array, ...]


class ShuffleCore(Generic[Array]):
    """
    Self-contained shuffle product engine for a fixed base dimension ``d``
    and maximum output degree ``trunc``.

    All operators are precomputed at construction time — no lazy growth,
    no cache management.  Memory cost is paid once; subsequent calls are
    pure compute.

    Parameters
    ----------
    xp : array namespace
        The array backend (e.g. ``numpy``, ``jax.numpy``).
    d : int
        Base dimension of the tensor algebra.
    trunc : int
        Maximum **output** degree this instance can produce.
        ``tensor_shuffle_product`` may request any ``trunc' <= trunc``.

    JIT safety
    ----------
    Hashing and equality are identity-based so the instance can be used as a
    JAX ``static_argnums`` argument without triggering spurious retraces.
    """

    def __init__(self, xp: array_types.ArrayNamespace, d: int, trunc: int) -> None:
        self.xp = xp
        self.d = d
        self.trunc = trunc
        self.operators: dict = {}
        self._precompute()

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return f"{type(self).__name__}(d={self.d}, trunc={self.trunc})"

    def _precompute(self) -> None:
        from tensordev.core.utils.shuffle_precalculation import assemble_shuffle_algebra_homogeneous
        for t in range(self.trunc + 1):
            for i in range(t, -1, -1):
                j = t - i
                if j > i:
                    break
                meta, vals = assemble_shuffle_algebra_homogeneous(self.d, i, j)
                idx_sizes, rows, cols, data = vals
                idx_ = np.arange(len(idx_sizes))
                segment_ids = np.repeat(idx_, idx_sizes)
                self.operators[(i, j)] = meta, (segment_ids, rows, cols, data)

    @dummy_jit(static_argnums=(0, 3, 4), dynamic_batchtime=("Ai", "Bj"))
    def sparse_einsum(self, Ai, Bj, i: int, j: int):
        """
        Sparse bilinear product ``C = einsum('bi,bj,ijl->bl', Ai, Bj, Q)``
        using the precomputed operator for degrees ``(i, j)``.

        This is the numpy fallback.  Backend subclasses (JAX, Numba) override it.
        """
        _, Q = self.operators[(i, j)]
        segment_ids, rows, cols, data = Q
        res = self.xp.zeros((Ai.shape[0], Ai.shape[1] * Bj.shape[1]), dtype=Ai.dtype)
        for k in range(len(segment_ids)):
            res[:, segment_ids[k]] += Ai[:, rows[k]] * Bj[:, cols[k]] * data[k]
        return res

    @dummy_jit(static_argnums=(0, 3, 4), dynamic_batchtime=("Ai", "Bj"))
    def tensor_shuffle_product_homogeneous(self, Ai: Array, Bj: Array, i: int, j: int) -> Array:
        """
        Homogeneous shuffle product ``(A_i ⊔ B_j)_{i+j}``.

        Parameters
        ----------
        Ai : Array, shape ``batch + (d**i,)``
        Bj : Array, shape ``batch + (d**j,)``
        i, j : int
            Degrees; must satisfy ``i >= j`` (commutativity is resolved by the caller).
        """
        return self.sparse_einsum(Ai, Bj, i, j)

    @dummy_jit(
        static_argnums=(0,),
        static_argnames=("trunc", "a_first_on", "b_first_on", "first_on_out"),
        dynamic_batchtime=("A", "B"),
    )
    def tensor_shuffle_product(
            self,
            A: Union[DenseElem, DenseElemFirstOn],
            B: Union[DenseElem, DenseElemFirstOn],
            *,
            trunc: Optional[int] = None,
            a_first_on: bool = False,
            b_first_on: bool = False,
            first_on_out: bool = False,
    ) -> Union[DenseElem, DenseElemFirstOn]:
        """
        Graded shuffle (commutative) product ``C = A ⊔ B``.

        For each output degree ``n``,
            C_n = ∑_{i+j=n} (A_i ⊔_shuffle B_j).

        Parameters
        ----------
        A, B : tuple of Array
            Graded levels; degree ``k`` has last dimension ``d**k``.
        trunc : int, optional
            Cap on the output degree.  Defaults to ``self.trunc``.
            Must not exceed ``self.trunc``; operators beyond that were not precomputed.
        a_first_on, b_first_on : bool, default False
            Whether ``A`` / ``B`` starts at degree 1 instead of 0.
        first_on_out : bool, default False
            If ``True``, the returned tuple starts at degree 1.
        """
        A = tuple(A)
        B = tuple(B)

        if len(A) == 0 or len(B) == 0:
            return tuple()

        a0 = 1 if a_first_on else 0
        b0 = 1 if b_first_on else 0

        NA = len(A) + a0 - 1
        NB = len(B) + b0 - 1

        if trunc is not None and trunc > self.trunc:
            raise ValueError(
                f"Requested trunc={trunc} exceeds precomputed trunc={self.trunc}. "
                f"Construct a new ShuffleCore with a larger trunc."
            )
        effective_trunc = self.trunc if trunc is None else trunc
        N = min(NA + NB, effective_trunc)

        out_start = 1 if (first_on_out or a_first_on or b_first_on) else 0
        out: List[Array] = []

        if a_first_on and b_first_on and out_start == 1 and N >= 1:
            out.append(self.xp.zeros_like(A[0]))
            loop_start = 2
        else:
            loop_start = out_start

        for n in range(loop_start, N + 1):
            i_min = max(a0, n - NB)
            i_max = min(NA, n - b0)

            i = i_min
            j = n - i
            if i >= j:
                term = self.tensor_shuffle_product_homogeneous(A[i - a0], B[j - b0], i, j)
            else:
                term = self.tensor_shuffle_product_homogeneous(B[j - b0], A[i - a0], j, i)

            for i in range(i_min + 1, i_max + 1):
                j = n - i
                if i >= j:
                    new = self.tensor_shuffle_product_homogeneous(A[i - a0], B[j - b0], i, j)
                else:
                    new = self.tensor_shuffle_product_homogeneous(B[j - b0], A[i - a0], j, i)
                term = term + new

            out.append(term)

        return tuple(out)