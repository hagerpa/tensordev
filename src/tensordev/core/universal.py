from __future__ import annotations

import functools
import itertools
from typing import Optional, Sequence, Tuple, Union, Literal, List, TypeVar, Generic, Callable

import array_api._2023_12 as array_types

from .annotations import jit as dummy_jit

Array = TypeVar("Array", bound=array_types.Array)
Elem = Sequence[Optional[Array]]  # one tensor-algebra element (level-list)
DenseElem = Tuple[Array, ...]  # level k has last dim d**k stating at level k=0; no Nones; shared batch shape
DenseElemFirstOn = Tuple[Array, ...]  # level k has last dim d**k starting at level k=1 no Nones; shared batch shape,


class Universal(Generic[Array]):
    def __init__(self, xp: array_types.ArrayNamespace):
        self.xp = xp

    # ----------------------------------------------------------------------
    # Axes iteration and reduction utilites
    # ----------------------------------------------------------------------

    @dummy_jit(static_argnums=0, static_argnames=("axis",), dynamic_batchtime=("X",))
    def tensor_stack(self, X: List[DenseElem], *, axis: int) -> tuple:
        L = len(X[-1])
        ndim = X[-1][-1].ndim
        stack_axis = axis if axis >= 0 else (ndim + 1 + axis)
        return tuple(self.xp.stack([e[k] for e in X], axis=stack_axis) for k in range(L))

    @dummy_jit(static_argnums=0, static_argnames=("source", "destination"), dynamic_batchtime=("X",))
    def tensor_moveaxis(
            self,
            X: DenseElem,
            *,
            source: int,
            destination: int
    ) -> DenseElem:
        return tuple(self.xp.moveaxis(a, source, destination) for a in X)

    def _mapper(self, fun: Callable[[DenseElem], DenseElem]):
        unstack = lambda x: [tuple(x[k][i] for k in range(len(x))) for i in range(len(x[0]))]
        return lambda seq: self.tensor_stack(list(map(fun, unstack(seq))), axis=0)

    def _reducer(
            self,
            fun: Callable[[DenseElem, DenseElem], DenseElem],  # (acc, step) -> acc
            *,
            neutral: DenseElem,  # guaranteed; kept for API symmetry (unused here)
            seed: DenseElem,  # guaranteed
            associative: bool = False,  # accepted for future use; not used here
    ):
        """
        Maker: returns reduce_fn(X) -> single graded element.
        Sequential left-fold from `seed` over axis 0 using `fun`.
        """

        def reduce_fn(X):
            X = tuple(X)
            S = int(X[0].shape[0])
            step = lambda t: tuple(a[t] for a in X)
            steps = (step(t) for t in range(S))
            return functools.reduce(fun, steps, seed)

        return reduce_fn

    def _accumulator(
            self,
            fun: Callable[[DenseElem, DenseElem], DenseElem],
            # (carry, step) -> y  (we build the body so that carry' := y)
            *,
            neutral: DenseElem,  # guaranteed; kept for API symmetry (unused here)
            seed: DenseElem,  # guaranteed
            associative: bool = False,  # accepted for future use; not used here
    ):
        """
        Maker: returns scan_fn(X) -> (final, ys).

        Semantics:
          - We lift `fun` into a scan body by setting carry' := y.
          - Using itertools.accumulate over [seed, s0, s1, ...]:
              prefixes = [seed, fun(seed,s0), fun(fun(seed,s0),s1), ...]
            We drop the initial `seed` from the emitted sequence.
          - final = ys[-1], ys are the prefixes without the bare seed.
        """

        def scan_fn(X):
            X = tuple(X)
            S = int(X[0].shape[0])
            step = lambda t: tuple(a[t] for a in X)
            prefixes = itertools.accumulate(
                itertools.chain([seed], (step(t) for t in range(S))),
                fun
            )
            ys = list(itertools.islice(prefixes, 1, None))  # drop the bare seed
            return ys[-1], self.tensor_stack(ys, axis=0)

        return scan_fn

    # ----------------------------------------------------------------------
    # Single & Binary Operations
    # ----------------------------------------------------------------------

    @dummy_jit(static_argnums=0, static_argnames=("trunc",), dynamic_batchtime=("A", "B"))
    def tensor_summation(self, A: DenseElem, B: DenseElem, trunc: Optional[int] = None) -> DenseElem:
        """
        Level-wise sum ``C = A + B`` (with implicit zero-padding if degrees differ).

        Parameters
        ----------
        A, B : sequence of ndarray
            Inputs with shapes ``A_k.shape == B_k.shape`` whenever both exist.
        trunc : int, optional
            If given, keep only degrees ``0..trunc``.

        Returns
        -------
        tuple of ndarray
            ``C_k = A_k + B_k`` for all available ``k`` (zeros used if one side
            lacks degree ``k``).
        """
        A, B = tuple(A), tuple(B)
        NA, NB = len(A) - 1, len(B) - 1
        N = max(NA, NB)
        if trunc is not None:
            N = min(N, trunc)
        out: List[Array] = []
        for k in range(N + 1):
            if k <= NA and k <= NB:
                out.append(A[k] + B[k])
            elif k <= NA:
                out.append(A[k])
            else:
                out.append(B[k])
        return tuple(out)

    @dummy_jit(static_argnums=0, dynamic_batchtime=("Ai", "Bj"))
    def tensor_product_homogeneous(self, Ai: Array, Bj: Array) -> Array:
        """
        Homogeneous-degree tensor (Chen) product on the last axis.

        Given degree-i and degree-j levels with shapes
            Ai.shape == batch + (d**i,),   Bj.shape == batch + (d**j,),
        return their (flattened) tensor product
            (Ai ⊗ Bj).shape == batch + (d**(i+j),).

        This is just a Kronecker on the last axis, preserving the batch shape.
        """
        tmp = self.xp.expand_dims(Ai, -1) * self.xp.expand_dims(Bj, -2)
        return self.xp.reshape(tmp, tmp.shape[:-2] + (tmp.shape[-2] * tmp.shape[-1],))

    @dummy_jit(static_argnums=0, static_argnames=("trunc",), dynamic_batchtime=("A", "B"))
    def tensor_product(self, A: DenseElem, B: DenseElem, trunc: Optional[int] = None) -> DenseElem:
        """
        Graded (Cauchy-type) product ``C = A ⊗ B`` in the free tensor algebra.

        For each degree ``n``,
            C_n = ∑_{i=0..n} flatten( tensor_product_homogeneous(A_i, B_{n-i}) ).

        Parameters
        ----------
        A, B : sequence of ndarray
            Levels of the left/right factors. Each ``A_k`` / ``B_k`` has shape
            ``batch + (d**k,)``. Batch shapes and dtypes should be compatible.
        trunc : int, optional
            If given, truncate the result to degrees ``0..trunc``.

        Returns
        -------
        tuple of ndarray
            Levels ``C_0, ..., C_N`` with ``N = min(trunc, deg(A)+deg(B))`` if ``trunc``
            is provided, else ``deg(A)+deg(B)``.
        """
        A = tuple(A)
        B = tuple(B)
        NA, NB = len(A) - 1, len(B) - 1
        N = (NA + NB) if trunc is None else min(NA + NB, trunc)

        out: List[Array] = []
        for n in range(N + 1):
            i_min, i_max = max(0, n - NB), min(n, NA)
            term = self.tensor_product_homogeneous(A[i_min], B[n - i_min])
            for i in range(i_min + 1, i_max + 1):
                term = term + self.tensor_product_homogeneous(A[i], B[n - i])
            out.append(term)
        return tuple(out)

    @dummy_jit(static_argnums=0, dynamic_batchtime=("A",), full_dynamic=("alpha",))
    def tensor_scalar_multiply(self, A: DenseElem, alpha: Union[Array, float]) -> DenseElem:
        """
        Uniform scalar multiply: ``Y = a * X``.

        Parameters
        ----------
        A : sequence of ndarray
            Input levels.
        alpha : float or ndarray
            Either a scalar, or a batch-shaped array broadcastable to the
            leading batch shape. When array-valued, scaling is done elementwise
            per batch example (broadcast over the last axis).

        Returns
        -------
        tuple of ndarray
            Levels ``B_k = alpha * A_k``.
        """
        A = tuple(A)
        if not A:
            return A
        a0 = self.xp.asarray(alpha, dtype=A[0].dtype)
        return tuple(Ak * self.xp.expand_dims(a0, axis=-1) for Ak in A)

    @dummy_jit(static_argnums=0, dynamic_batchtime=("A",), full_dynamic=("c",))
    def tensor_dilation(self, A: DenseElem, c: Union[Array, float]) -> DenseElem:
        """Dense version of `tensor_dilation`. See `tensor_dilation` for details; here we assume dense.
        Scales level k by c**k; optional `trunc` limits levels."""
        A = tuple(A)
        if not A:
            return A
        c0 = self.xp.asarray(c, dtype=A[0].dtype)
        out: List[Array] = []
        for k, Xk in enumerate(A):
            scale_k = self.xp.expand_dims(c0 ** k, axis=-1)  # shape: batch + (1,)
            out.append(Xk * scale_k)
        return tuple(out)

    @dummy_jit(static_argnums=0, dynamic_batchtime=("Ak", "Bk"))
    def tensor_inner_product_homogeneous(self, Ak: Array, Bk: Array) -> Array:
        """
        Level-wise (homogeneous) Euclidean inner product over the last axis.

        Parameters
        ----------
        Ak, Bk : ndarray
            Same-degree levels with shape ``batch + (d**k,)``.

        Returns
        -------
        ndarray
            Batch-shaped array ``batch`` with ⟨Ak, Bk⟩ = sum over the last axis.
        """
        return (Ak * Bk).sum(-1)

    @dummy_jit(static_argnums=0, dynamic_batchtime=("A", "B"))
    def tensor_inner_product(self, A: DenseElem, B: DenseElem) -> Array:
        """
        Canonical Euclidean inner product ⟨A, B⟩ = ∑_k ⟨A_k, B_k⟩,
        summing level-wise dot products over the last axis.

        Parameters
        ----------
        A, B : DenseElem
            Dense graded elements (X₀, X₁, …) with matching batch shape.

        Returns
        -------
        Array
            Batch-shaped array with the inner product.
        """
        K = min(len(A), len(B))
        if K == 0:
            # Degenerate case: return scalar
            return self.xp.asarray(0.0)

        acc = self.tensor_inner_product_homogeneous(A[0], B[0])
        for k in range(1, K):
            acc = acc + self.tensor_inner_product_homogeneous(A[k], B[k])
        return acc

    @dummy_jit(static_argnums=0, dynamic_batchtime=("Ai", "Yni"))
    def tensor_adjoint_left_homogeneous(sefl, Ai: Array, Yni: Array) -> Array:
        """
        Homogeneous left-adjoint contraction for degree i.

        Given
            Ai.shape  == batch + (d**i,)
            Yni.shape == batch + (d**(n+i),)
        reshape Yni as (..., d**i, -1) and contract the first width with Ai:
            out = Ai^T • Yni_(..., i, N)  ->  shape batch + (N,)
        """
        y = Yni.reshape((*Yni.shape[:-1], Ai.shape[-1], -1))  # (..., i, N)
        return (Ai[..., :, None] * y).sum(axis=-2)  # contract over i → (..., N)

    @dummy_jit(static_argnums=0, dynamic_batchtime=("Bj", "Ynj"))
    def tensor_adjoint_right_homogeneous(self, Bj: Array, Ynj: Array) -> Array:
        """
        Homogeneous right-adjoint contraction for degree j.

        Given
            Bj.shape  == batch + (d**j,)
            Ynj.shape == batch + (d**(n+j),)
        reshape Ynj as (..., -1, d**j) and contract the last width with Bj:
            out = Ynj_(..., N, j) • Bj  ->  shape batch + (N,)
        """
        y = Ynj.reshape((*Ynj.shape[:-1], -1, Bj.shape[-1]))  # (..., N, j)
        return (y * Bj[..., None, :]).sum(axis=-1)  # contract over j → (..., N)

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "side"), dynamic_batchtime=("W", "Y"))
    def tensor_adjoint_product(
            self,
            W: DenseElem,  # multiplier levels (left: A, right: B)
            Y: DenseElem,  # target levels
            trunc: Optional[int],
            side: Literal["left", "right"],
    ) -> DenseElem:
        """
        Shared accumulation for left/right adjoints.
        For each n: sum_i/j contract(W_i/j, Y_{n+i/j}) using the side-specific homogeneous op.
        """
        W, Y = tuple(W), tuple(Y)
        NW, NY = len(W) - 1, len(Y) - 1
        N = NY if trunc is None else min(NY, trunc)
        contract = self.tensor_adjoint_left_homogeneous if side == "left" else self.tensor_adjoint_right_homogeneous

        out = []
        for n in range(N + 1):
            m = min(NW, NY - n)
            terms = [contract(W[i], Y[n + i]) for i in range(m + 1)]
            out.append(terms[0] if len(terms) == 1 else sum(terms[1:], terms[0]))
        return tuple(out)

    @dummy_jit(static_argnums=0, static_argnames=("trunc",), dynamic_batchtime=("A", "Y"))
    def tensor_adjoint_left(self, A: DenseElem, Y: DenseElem, trunc: Optional[int] = None) -> DenseElem:
        """
        Left-adjoint of multiplication by ``A``: find ``Z`` such that
        ``<Z, X> = <Y, A ⊗ X>`` for all ``X`` (canonical Euclidean inner product).

        In components, for each ``n``:
          ``Z_n = ∑_i  (A_i^T • reshape(Y_{n+i}, (..., A_i_width, -1)))``,
        where ``A_i_width = A_i.shape[-1]`` and ``•`` contracts that width.

        Parameters
        ----------
        A : DenseElem
            Left multiplier levels.
        Y : DenseElem
            Target element levels.
        trunc : int, optional
            If given, return degrees ``0..trunc`` only.

        Returns
        -------
        DenseElem
            Levels of the adjoint result ``Z``.
        """
        return self.tensor_adjoint_product(A, Y, trunc, side="left")

    @dummy_jit(static_argnums=0, static_argnames=("trunc",), dynamic_batchtime=("B", "Y"))
    def tensor_adjoint_right(self, B: DenseElem, Y: DenseElem, trunc: Optional[int] = None) -> DenseElem:
        """
        Right-adjoint of multiplication by ``B``: find ``Z`` such that
        ``<Z, X> = <Y, X ⊗ B>`` for all ``X``.

        In components, for each ``n``:
          ``Z_n = ∑_j  (reshape(Y_{n+j}, (..., -1, B_j_width)) • B_j)``,
        contracting over the right width axis.

        Parameters
        ----------
        B : DenseElem
            Right multiplier levels.
        Y : DenseElem
            Target element levels.
        trunc : int, optional
            If given, return degrees ``0..trunc`` only.

        Returns
        -------
        DenseElem
            Levels of the adjoint result ``Z``.
        """
        return self.tensor_adjoint_product(B, Y, trunc, side="right")

    # ----------------------------------------------------------------------
    # Polynomial / Series Operations
    # ----------------------------------------------------------------------

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "output_zero_level"), dynamic_batchtime=("g", "X"))
    def tensor_fmexp(
            self,
            g: DenseElem,  # graded element with explicit level-0
            X: DenseElemFirstOn,  # levels start at 1; treat X₀ ≡ 0
            *,
            trunc: int,
            output_zero_level: bool = True,
    ) -> Tuple[Array, ...]:
        """
        Efficient formal multiplicative exponential: compute Y = g ⊗ exp(X).

        Conventions
        -----------
        - `g` is a dense graded element (g₀, g₁, …).
        - `X` has levels starting at 1: (X₁, X₂, …); we treat X₀ ≡ 0.

        Algorithm
        ---------
        Recurrences (free tensor algebra):
            k E_k = Σ_{r=1..k} r Σ_{a+b=k-r} β(a,b) · E_a X_r E_b,
            k Y_k = k g_k + Σ_{r=1..k} r Σ_{a+b=k-r} β(a,b) · Y_a X_r E_b,
        where β(a,b) = a! b! / (a+b+1)! and X_r is the degree-r part of X.

        Parameters
        ----------
        g : DenseElem
            Left factor with explicit level-0.
        X : DenseElemFirstOn
            Exponent argument with levels starting at degree 1.
        trunc : int
            Maximum degree (inclusive) of the output.
        output_zero_level : bool, default True
            If False, drop degree-0 from the returned tuple.

        Returns
        -------
        Tuple[Array, ...]
            (Y₀, …, Y_trunc) for `output_zero_level=True`; otherwise (Y₁, …, Y_trunc).
        """
        if trunc < 0: raise ValueError("trunc>=0")
        if trunc == 0 or len(X) == 0:
            return g if output_zero_level else g[1:]

        # --- degree=1 fused path (Signatory-style fmexp):
        # For each k, compute  Y_k = Σ_{i=0..k} g_i ⊗ (X1^{⊗(k-i)} / (k-i)!)
        # Horner (nested form): Y_k = (((g0 ⊗ (z/k) + g1) ⊗ (z/(k-1)) + g2) ⊗ ... ⊗ (z/1) + gk),  with z := X[0].
        if len(X) == 1:
            z = X[0]
            Y = (g[0],)
            for k in range(1, trunc + 1):
                # Horner for the k-th level of g ⊗ exp(z)
                t = self.tensor_product_homogeneous(g[0], z) * (1.0 / float(k))

                # i runs 1..k-1:  t <- (t + g[i]) ⊗ (z/(k-i))
                for i in range(1, k):
                    gi = g[i] if i < len(g) else self.xp.zeros_like(t)
                    t = self.tensor_product_homogeneous(t + gi, z) * (1.0 / float(k - i))

                gk = g[k] if k < len(g) else self.xp.zeros_like(t)
                Y += (t + gk,)
            return Y if output_zero_level else Y[1:]
        # --- degree>1
        # E_k = (1/k) * sum_{r=1..min(k,R)} r * (X_r ⊗ E_{k-r})
        # Y_k = sum_{i=0..min(k,deg(g))} g_i ⊗ E_{k-i}
        E = (self.xp.ones_like(g[0]),)
        Y = (g[0],)
        for k in range(1, trunc + 1):
            invk = 1.0 / float(k)
            m = min(k, len(X))
            t = self.tensor_product_homogeneous(X[0], E[k - 1]) * invk
            for r in range(2, m + 1):
                t += self.tensor_product_homogeneous(X[r - 1], E[k - r]) * (r * invk)
            E += (t,)
            hi = min(k, len(g) - 1)
            yk = self.tensor_product_homogeneous(g[0], E[k])
            for i in range(1, hi + 1):
                yk += self.tensor_product_homogeneous(g[i], E[k - i])
            Y += (yk,)

        return Y if output_zero_level else Y[1:]

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "output_zero_level"), dynamic_batchtime=("X",))
    def tensor_exponential(
            self, X: DenseElemFirstOn, *, trunc: int, output_zero_level: bool = True
    ) -> Tuple[Array, ...]:
        """
        Algebra exponential for inputs starting at degree 1 (treat X₀ ≡ 0).

        Implemented via the efficient primitive:
            exp(X) = I ⊗ exp(X)  with  I = (1,)  (level-0 only)
        """
        if trunc < 0:
            raise ValueError("tensor_exponential: trunc must be >= 0.")

        if len(X) == 0:
            # exp(0) = 1
            return (self.xp.asarray(1.0),) if output_zero_level else tuple()

        I: DenseElem = (self.xp.ones_like(X[0][..., :1]),)  # level-0 only
        return self.tensor_fmexp(I, X, trunc=trunc, output_zero_level=output_zero_level)

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "output_zero_level"), dynamic_batchtime=("X",))
    def tensor_logarithm(self, X: DenseElemFirstOn, *, trunc: int, output_zero_level: bool = True) -> Tuple[Array, ...]:
        """
        Algebra logarithm with input levels starting at 1.

        Convention
        ----------
        Input X = (X₁, X₂, …) and we *treat* X₀ ≡ 1.
        Let H := (0, X₁, X₂, …) (zero scalar part). Then
            log(1 + H) = (0, ∑_{n≥1} (-1)^{n+1} H^{⊗ n} / n),
        truncated to degree `trunc`. No padding of higher degrees is performed; the
        convolution grows degrees naturally via `tensor_product(..., trunc=trunc)`.

        Parameters
        ----------
        X : DenseElemFirstOn
            Levels start at degree 1, last dim d**k, shared batch shape.
        trunc : int
            Maximum degree (inclusive) of the output.
        output_zero_level : bool, default True
            Whether to include degree-0 in the returned tuple (it's identically zero).

        Returns
        -------
        DenseElem (or degrees 1..trunc when output_zero_level=False)
        """
        if trunc < 0:
            raise ValueError("tensor_log: trunc must be >= 0.")
        if len(X) == 0:
            # log(1) = 0
            return (self.xp.asarray(0.0),) if output_zero_level else tuple()

        zero0 = self.xp.zeros_like(X[0][..., :1])
        H: DenseElem = (zero0,) + tuple(X)

        # Start with zeros (log(1)=0)
        res: DenseElem = (zero0,) + tuple(self.xp.zeros_like(L) for L in X)

        term: DenseElem = H
        for n in range(1, trunc + 1):
            coeff = (1.0 if (n % 2 == 1) else -1.0) / n
            res = self.tensor_summation(res, self.tensor_scalar_multiply(term, coeff), trunc=trunc)
            if n != trunc:
                term = self.tensor_product(term, H, trunc=trunc)

        return res if output_zero_level else res[1:]

    # ----------------------------------------------------------------------
    # Sequential Operations (Reductions & Developments)
    # ----------------------------------------------------------------------

    def _pre_processing_for_sequential(
            self,
            len_X: int,
            *,
            op: Literal["product", "sum", "fmexp"] = "product",
            trunc: Optional[int] = None,
            memory_consumption: Literal["high", "low"] = "low",
    ) -> Tuple[
        int,  # N
        Callable,  # reduce_op
        Callable,  # acc_op
        Callable[[DenseElem], DenseElem],  # neutral_gen(X) -> neutral
        Callable[[DenseElem, Optional[DenseElem]], DenseElem],  # seed_gen(neutral, starting_point) -> seed
        Callable[[DenseElem], DenseElem],
    ]:
        K = len_X - 1

        if op == "sum":
            N = min(K, trunc) if trunc is not None else K
            reduce_op = functools.partial(self.tensor_summation, trunc=N)
            acc_op = reduce_op

            def neutral_gen(X: DenseElem) -> DenseElem:
                step0 = tuple(a[0] for a in X)
                return self.tensor_logarithm(
                    (self.xp.zeros_like(step0[1]),), trunc=N, output_zero_level=True
                )

        elif op == "fmexp":
            if trunc is None:
                raise ValueError("`trunc` is required for op='fmexp'.")
            N = trunc
            if memory_consumption == "high":
                reduce_op = functools.partial(self.tensor_product, trunc=N)
            elif memory_consumption == "low":
                reduce_op = functools.partial(self.tensor_fmexp, trunc=N, output_zero_level=True)
            else:
                raise ValueError("`memory_consumption` must be 'high' or 'low'.")
            acc_op = functools.partial(self.tensor_product, trunc=N)

            def neutral_gen(X: DenseElem) -> DenseElem:
                step0 = tuple(a[0] for a in X)
                return self.tensor_exponential(
                    (self.xp.zeros_like(step0[0]),), trunc=N, output_zero_level=True
                )

        elif op == "product":
            N = K if trunc is None else trunc
            reduce_op = functools.partial(self.tensor_product, trunc=N)
            acc_op = reduce_op

            def neutral_gen(X: DenseElem) -> DenseElem:
                step0 = tuple(a[0] for a in X)
                return self.tensor_exponential(
                    (self.xp.zeros_like(step0[1]),), trunc=N, output_zero_level=True
                )

        else:
            raise ValueError("`op` must be one of {'sum','product','fmexp'}.")

        def seed_gen(neutral: DenseElem, starting_point: Optional[DenseElem] = None) -> DenseElem:
            return acc_op(starting_point, neutral) if starting_point is not None else neutral

        def truncator(X: DenseElem) -> DenseElem:
            return X[: min(N + 1, len(X) + (op == "fmexp"))]

        return N, reduce_op, acc_op, neutral_gen, seed_gen, truncator

    @dummy_jit(static_argnums=0, static_argnames=("op", "axis", "trunc", "memory_consumption"),
               dynamic_batchtime=("X",))
    def tensor_reduce(
            self,
            X: DenseElem | DenseElemFirstOn,
            *,
            op: Literal["product", "sum", "fmexp"] = "product",
            axis: int = -2,
            trunc: Optional[int] = None,
            memory_consumption: Literal["high", "low"] = "low",
    ):
        """
        Reduce a **packed graded element** `X` along `axis`.

        Modes
        -----
        - ``"sum"``     : degree-wise addition (cap degrees via `_resolve_truncation`).
        - ``"product"`` : Chen/tensor product (cap as above).
        - ``"fmexp"``   : per-step *formal multiplicative exponential* (requires `trunc`):
            • ``memory_consumption="low"``  → fused streaming via ``tensor_fmexp(carry, step, trunc)``;
            • ``memory_consumption="high"`` → pre-map per-step exponentials
              ``step ↦ tensor_exponential(step, trunc)`` and then plain product.
          In all cases we start from the degree-0 **neutral** (zeros for sum, ones for product/fmexp).

        Returns
        -------
        DenseElem or []  (empty if there are no steps).
        """
        N, reduce_op, _, neutral_gen, seed_gen, truncator = self._pre_processing_for_sequential(len(X), op=op,
                                                                                                trunc=trunc,
                                                                                                memory_consumption=memory_consumption)
        X = self.tensor_moveaxis(truncator(X), soure=axis, destination=0)
        neutral = neutral_gen(X)
        seed = seed_gen(neutral, None)

        associative = not (op == "fmexp" and memory_consumption == "low")
        if op == "fmexp" and memory_consumption == "high":
            X = self.tensor_exponential(X, trunc=N)
        return self._reducer(reduce_op, neutral=neutral, seed=seed, associative=associative)(X)

    @dummy_jit(static_argnums=0, static_argnames=("op", "axis", "trunc", "memory_consumption", "starting_point",
                                                  "output_starting_point"), dynamic_batchtime=("X",))
    def tensor_accumulate(
            self,
            X,
            *,
            op: Literal["product", "sum", "fmexp"] = "product",
            axis: int = -2,
            trunc: Optional[int] = None,
            memory_consumption: Literal["high", "low"] = "low",
            starting_point: Optional[tuple] = None,
            output_starting_point: bool = False,
    ):
        """
        Inclusive prefixes along `axis`, with optional `starting_point` and
        `output_starting_point`. Always returns a **packed** graded element stacked
        on `axis` (no Python list return).

        - sum/product: associative prefix scan (optionally seeded).
        - fmexp/low:   fused streaming via `tensor_fmexp`.
        - fmexp/high:  vectorized `tensor_exponential` then associative product scan.
        """
        N, reduce_op, _, neutral_gen, seed_gen, truncator = self._pre_processing_for_sequential(len(X), op=op,
                                                                                                trunc=trunc,
                                                                                                memory_consumption=memory_consumption)
        X = self.tensor_moveaxis(truncator(X), source=axis, destination=0)
        neutral = neutral_gen(X)
        seed = seed_gen(neutral, starting_point)

        associative = not (op == "fmexp" and memory_consumption == "low")
        if op == "fmexp" and memory_consumption == "high":
            X = self.tensor_exponential(X, trunc=N)
        accumulator = self._accumulator(reduce_op, neutral=neutral, seed=seed, associative=associative)
        stacked = tuple(self.xp.stack([seed[k], *(X[k])], axis=0) for k in range(N + 1))
        _, zs = accumulator(stacked)
        if not output_starting_point:
            zs = tuple(a[1:] for a in zs)
        return self.tensor_moveaxis(zs, source=0, destination=axis)

    @dummy_jit(static_argnums=0, static_argnames=("op", "axis", "trunc", "block_size", "accumulate", "starting_point",
                                                  "output_starting_point", "memory_consumption"),
               dynamic_batchtime=("X",))
    def tensor_abra(
            self,
            X: DenseElem | DenseElemFirstOn,
            *,
            op: Literal["product", "sum", "fmexp"] = "product",
            axis: int = -2,
            trunc: Optional[int] = None,
            block_size: Optional[int] = None,
            accumulate: bool = False,
            starting_point: Optional[DenseElem] = None,
            output_starting_point: bool = False,
            memory_consumption: Literal["high", "low"] = "low",
    ) -> DenseElem:
        """
        Aplly, Block, Reduce and Accumulate (ABRA) a **packed graded element**
        along a steps axis and return an **already-packed** graded element.

        Ops
        ---
        - ``"product"`` : left-fold Chen/tensor product over steps.
        - ``"sum"``     : levelwise sum over steps.
        - ``"fmexp"``   : per-step algebra exponential, then product.

        Input (dense, flat levels)
        --------------------------
        ``X = (X₀, X₁, …, X_K)`` (or X = (X₁, …, X_K) in case of ``"fmexp"``) with flat last axis (width ``d**k``).
        The **steps axis** is at ``axis`` (default ``-2``):
          • for ``k ≥ 1``:  ``X_k.shape == batch + (S, d**k)``
          • for ``k = 0``:  either ``batch + (S, 1)`` or ``batch + (1,)``

        Truncation
        ----------
        - ``op="sum"``     → degree cap ``N = K`` (or ``min(K, trunc)`` if ``trunc`` given).
        - ``op="product"`` → degree cap ``N = trunc`` if provided, else ``N = K``.
        - ``op="fmexp"``   → **requires** ``trunc`` and uses ``N = trunc``.

        What gets computed
        ------------------
        Given the per-step graded elements (slicing along ``axis``),
        - ``"product"``:  ``acc ← acc ⊗ step``
        - ``"sum"``:      ``acc ← acc + step``  (levelwise)
        - ``"fmexp"``:    ``acc ← acc ⊗ exp(step)`` (algebra exponential per step)

        Blocking
        --------
        If ``block_size`` is ``None``/``-1``, the whole steps axis is a single block.
        Otherwise, it must divide ``S``; blocks are contiguous slices of length
        ``block_size``.

        Accumulation (prefixes) & starting point
        ----------------------------------------
        - If ``accumulate=False``:
            * ``output_starting_point=False`` → emit per-block reductions: ``[B₀, B₁, …]``.
            * ``output_starting_point=True``  → emit the **head** then the blocks:
              ``[g, B₀, B₁, …]`` if ``starting_point=g`` is given,
              else ``[I, B₀, B₁, …]`` (neutral; degree-0 identity for product/fmexp, zero for sum).
        - If ``accumulate=True`` (streamed prefixes):
            * First block starts from the **seed** (``g`` if given, otherwise the neutral).
            * Each subsequent block starts from the **previous block’s outcome**.
            * ``output_starting_point=False`` → ``[g⊗B₀, g⊗B₀⊗B₁, …]`` if ``g`` given,
              else ``[B₀, B₀⊗B₁, …]``.
            * ``output_starting_point=True``  → prepend the seed:
              ``[g, g⊗B₀, g⊗B₀⊗B₁, …]`` (or ``[I, B₀, B₀⊗B₁, …]`` if no ``g``).

        Memory consumption (relevant for "fmexp" only)
        ----------------------------------------------
        - "high": Within each block, compute per-step exponentials G_t = exp(ΔX_t) in batch (via vmap)
            and then reduce them with an associative prefix/product. This maximizes parallelism and is
            typically fastest, but materializes the entire block of G_t (higher peak memory).
        - "low": Stream within each block using fmexp inside a scan, so G_t is never materialized
            (lowest peak memory). This is strictly sequential within a block and can be slower.

        Output packing
        --------------
        - If **exactly one** item is emitted, return that single graded element
          (no block axis).
        - Otherwise, each degree is stacked with a **block axis at `axis`**. The number
          of returned degrees equals the maximum number present among the outputs
          (we don’t fabricate higher empty degrees).

        Parameters
        ----------
        X : DenseElem | DenseElemFirstOn
            Packed input graded element with a steps axis at ``axis``.
        op : {"product","sum","fmexp"}, default: "product"
            Reduction mode (see above).
        axis : int, default: -2
            Steps axis (and the position where the block axis will be inserted).
        trunc : int, optional
            Degree cap (inclusive). Required for ``"fmexp"``.
        block_size : int or None, optional
            ``None/-1`` → one block. Otherwise must divide the number of steps ``S``.
        accumulate : bool, default: False
            If ``True``, emit streaming prefixes across blocks.
        starting_point : DenseElem, optional
            Seed ``g`` for streaming/decoration (see behavior above).
        output_starting_point : bool, default: False
            If ``True``, emit the seed as the first item (``g`` if provided, otherwise neutral).
        memory_consumption : {"high","low"}, default: "high"
            Decider on whether to materialize per-step exponentials for "fmexp" (high) or

        Returns
        -------
        DenseElem
            Either a single graded element (if one item emitted) or a graded element
            whose levels carry a new **block axis at `axis`**.

        Notes
        -----
        • Dense backend only; last axis is the flat width ``d**k``.
        • No axis reordering during reduction; only a new block axis is inserted at the end.
        • Uses helper ops: ``tensor_product``, ``tensor_summation``, ``tensor_exponential``.
        """
        N, reduce_op, acc_op, neutral_gen, seed_gen, truncator = self._pre_processing_for_sequential(len(X), op=op,
                                                                                                     trunc=trunc,
                                                                                                     memory_consumption=memory_consumption)
        X = self.tensor_moveaxis(truncator(X), source=axis, destination=0)
        neutral = neutral_gen(X)
        seed = seed_gen(neutral, starting_point)

        # 1) make blocks
        S = X[0].shape[0]
        B = S if (block_size in (None, -1)) else int(block_size)
        q, r = divmod(S, B)
        if r: raise ValueError(f"tensor_abra: block_size={B} must divide S={S}.")
        X_blocks = tuple(L.reshape(q, B, *L.shape[1:]) for L in X)

        # 2) reduce blocks
        associative = not (op == "fmexp" and memory_consumption == "low")
        base_reducer = self._reducer(reduce_op, neutral=neutral, seed=neutral, associative=associative)
        if op == "fmexp" and memory_consumption == "high":
            def reducer(block):
                # Compute exp(step) for every step in the block, then reduce with the product.
                return base_reducer(self.tensor_exponential(block, trunc=N))
        else:
            reducer = base_reducer
        blocks = self._mapper(reducer)(X_blocks)

        if not accumulate:
            if output_starting_point:
                return tuple(self.xp.stack([seed[k], *(blocks[k])], axis=axis) for k in range(N + 1))
            else:
                if q == 1: return tuple(a[0] for a in blocks)
                return self.tensor_moveaxis(blocks, source=0, destination=axis)

        # 3) accumulate (acc_op is assumed to be associative)
        accumulator = self._accumulator(acc_op, neutral=neutral, seed=seed, associative=True)
        stacked = tuple(self.xp.stack([seed[k], *(blocks[k])], axis=0) for k in range(N + 1))
        _, zs = accumulator(stacked)
        if not output_starting_point:
            if q == 1: return tuple(a[1] for a in zs)
            zs = tuple(a[1:] for a in zs)
        return self.tensor_moveaxis(zs, source=0, destination=axis)

    @dummy_jit(static_argnums=0, static_argnames=("axis", "trunc", "block_size", "accumulate", "starting_point",
                                                  "output_starting_point", "memory_consumption"),
               dynamic_batchtime=("X",))
    def tensor_development(
            self,
            X: DenseElemFirstOn,  # [X₁, X₂, …] with steps axis at `axis`
            *,
            axis: int = -2,
            trunc: int,
            block_size: Optional[int] = None,  # None/-1 → one block
            accumulate: bool = True,
            starting_point: Optional[DenseElem] = None,
            output_starting_point: bool = False,
            memory_consumption: Literal["high", "low"] = "low",
    ) -> DenseElem:
        """
        Tensor/free development from a **path whose levels start at degree 1**:
            X = [X₁, X₂, …], each with a steps axis at `axis`.

        Input & convention
        ------------------
        • Xₖ has shape `(..., S_axis, d**k)` with steps on `axis` and width `d**k` last.
        • Work with **increments** along `axis`:
             ΔXₖ = Xₖ[..., 1:, :] − Xₖ[..., :-1, :],  k ≥ 1  (steps S = original_steps−1).
        • Exponentials apply only to degrees ≥ 1 (no scalar level in `X`).

        Implementation (thin wrapper over `tensor_reduce`)
        --------------------------------------------------
        • memory_consumption="high":
            E = tensor_exponential(ΔX, trunc=trunc, output_zero_level=True)
            return tensor_reduce(E, op="product", axis=axis, trunc=trunc, ...)
        • memory_consumption="low":
            return tensor_reduce((zeros_level0, ΔX), op="fmexp", axis=axis, trunc=trunc, ...)

        Output (exactly like `tensor_reduce`)
        -------------------------------------
        • If exactly one item is emitted (one block and no standalone `g`): a single graded
          element (no block axis).
        • Otherwise: each level has a **block axis inserted at `axis`**.
        """
        X = tuple(X)
        if not X:
            raise ValueError("tensor_development: X must contain at least X₁.")
        N = min(trunc, len(X))
        # Increments ΔX along `axis` (degrees 1..N)
        dX = tuple(self.xp.diff(L, axis=axis) for L in X[:N])

        if memory_consumption == "low":
            return self.tensor_abra(
                dX,
                op="fmexp",
                axis=axis,
                trunc=trunc,
                block_size=block_size,
                accumulate=accumulate,
                starting_point=starting_point,
                output_starting_point=output_starting_point,
            )

        # memory_consumption == "high": batched exponential over steps, then product reduce
        E = self.tensor_exponential(dX, trunc=trunc, output_zero_level=True)  # keeps `axis`
        return self.tensor_abra(
            E,
            op="product",
            axis=axis,
            trunc=trunc,
            block_size=block_size,
            accumulate=accumulate,
            starting_point=starting_point,
            output_starting_point=output_starting_point,
        )

    @dummy_jit(static_argnums=0, static_argnames=("axis", "trunc", "block_size", "accumulate", "starting_point",
                                                  "output_starting_point", "memory_consumption"),
               dynamic_batchtime=("X",))
    def tensor_path_signature(
            self,
            x: Array,
            *,
            axis: int = -2,
            trunc: int,
            block_size: Optional[int] = None,  # None/-1 → one block
            accumulate: bool = True,
            starting_point: Optional[DenseElem] = None,
            output_starting_point: bool = False,
            memory_consumption: Literal["high", "low"] = "low",
    ) -> DenseElem:
        """
        Calculate the signature of a given path (X[..., i, ...])_{i in axis} and return as a graded element.
        """
        return self.tensor_development((x,), axis=axis, trunc=trunc, block_size=block_size, accumulate=accumulate,
                                       starting_point=starting_point, output_starting_point=output_starting_point,
                                       memory_consumption=memory_consumption)

    # ----------------------------------------------------------------------
    # API Transformers (from_flat / to_flat / densify)
    # ----------------------------------------------------------------------

    def tensor_densify(self, levels: Elem) -> DenseElem:
        """
        Turn a possibly sparse graded element (levels may be None) into a **dense** one.

        Rules
        -----
        • If a positive degree (k ≥ 1) level is present, infer the base dimension `d`
          from the first such level:  width(level_k) must equal d**k.
          - All other present levels are validated against this `d` and shared batch shape.
        • Missing levels are filled with exact zeros of matching batch shape and width d**k
          (for k=0 the width is 1).
        • If `d` **cannot** be inferred (because the input is empty, all levels are None,
          or only degree-0 is provided), return a canonical zero scalar:
             `(np.array(0.0),)`.

        Parameters
        ----------
        levels : Tuple[Optional[Array], ...]
            Packed levels `(X₀, X₁, ..., X_N)`, where any `X_k` may be `None`.

        Returns
        -------
        Tuple[Array, ...]
            Dense graded element `(Y₀, Y₁, ..., Y_N)` with no `None`s.

        Raises
        ------
        ValueError
            If present levels disagree on batch shape, or a present width at degree k
            does not equal `d**k`.
        """
        levels = tuple(levels)

        # prune trailing Nones
        last_present = -1
        for i in range(len(levels) - 1, -1, -1):
            if levels[i] is not None:
                last_present = i
                break
        if last_present == -1:
            return (self.xp.asarray(0.0),)

        levels = levels[: last_present + 1]  # drop trailing Nones
        present = [(k, L) for k, L in enumerate(levels) if L is not None]

        # infer d from the smallest present positive degree; if none, d=1
        pos = [(k, L) for k, L in present if k > 0]
        if pos:
            k0, L0 = min(pos, key=lambda kv: kv[0])
            width_k0 = L0.shape[-1]
            d = int(round(width_k0 ** (1.0 / k0)))
            if d < 1 or d ** k0 != width_k0:
                raise ValueError(
                    f"tensor_densify: cannot infer integer base d from degree {k0} width {width_k0}."
                )
        else:
            d = 1  # only degree-0 is present

        N = len(levels) - 1  # highest degree we'll output
        ref_max = levels[N] if levels[N] is not None else next(L for _, L in reversed(present))
        batch = ref_max.shape[:-1]
        max_width = ref_max.shape[-1]

        # validate present levels
        for k, L in present:
            exp_w = 1 if k == 0 else d ** k
            if L.shape[:-1] != batch:
                raise ValueError("tensor_densify: batch shapes differ across levels.")
            if L.shape[-1] != exp_w:
                raise ValueError(
                    f"tensor_densify: degree {k} has width {L.shape[-1]} but expected {exp_w}."
                )

        # fill missing levels using slice -> zeros_like (no shape kwarg)
        out = []
        for k in range(N + 1):
            L = levels[k]
            if L is not None:
                out.append(L)
            else:
                w = 1 if k == 0 else d ** k
                # guaranteed w <= max_width because k <= N
                out.append(self.xp.zeros_like(ref_max[..., :w]))
        return tuple(out)

    @dummy_jit(static_argnums=0, static_argnames=("dim", "insert_zero_level"), dynamic_batchtime=("X",))
    def tensor_from_flat(
            self,
            flat: Array,
            dim: int,
            insert_zero_level: Optional[Union[bool, float]] = None,
    ) -> DenseElem:
        """
        Split a flattened graded tensor into a **dense** (X₀, X₁, ..., X_N).

        Layouts supported on the last axis of `flat`:
          • [X₀ | X₁ | ... | X_N]  with widths 1, d, d**2, ...
          • [X₁ | ... | X_N]       with widths d, d**2, ...

        Degree-0 policy (always return a DenseElem):
          - insert_zero_level is None  → if X₀ missing, **insert zeros**; if present, keep.
          - insert_zero_level is True  → set/insert X₀ = **ones**.
          - insert_zero_level is False → set/insert X₀ = **zeros**.
          - insert_zero_level is float → set/insert X₀ = that **scalar**.

        Parameters
        ----------
        flat : Array
            Packed array with the concatenated degrees on the last axis.
        dim : int
            Base dimension `d` so that width(Xₖ) = d**k.
        insert_zero_level : None | bool | float, optional
            Degree-0 policy as described above.

        Returns
        -------
        DenseElem
            A tuple (X₀, X₁, ..., X_N), each with matching batch shape and last dim d**k.
        """
        # Coerce to array without asarray()
        flat = self.xp.asarray(flat)
        total = int(flat.shape[-1])

        def find_N(total_len: int, start: int) -> Optional[int]:
            acc, k = 0, start
            while acc < total_len:
                acc += dim ** k
                if acc == total_len:
                    return k
                k += 1
            return None

        N0 = find_N(total, 0)  # assumes X0 present
        N1 = find_N(total, 1)  # assumes X0 absent
        if N0 is None and N1 is None:
            raise ValueError("tensor_from_flat: last axis is not a valid sum of powers of `dim`.")

        # Prefer layout with X0 if both fit
        start = 0 if N0 is not None else 1
        N = N0 if start == 0 else N1  # type: ignore[assignment]

        parts = []
        off = 0
        X0 = None
        if start == 0:
            # X0 is present in `flat`: slice it directly
            X0 = flat[..., :1]
            off = 1

        # Slice remaining parts: widths dim^1, dim^2, ..., dim^N
        for k in range(1, N + 1):
            w = dim ** k
            parts.append(flat[..., off:off + w])
            off += w

        # Materialize/override X0 using only *_like without shape kwarg (slice to target shape first)
        if isinstance(insert_zero_level, (int, float)):
            X0 = self.xp.ones_like(flat[..., :1]) * float(insert_zero_level)
        elif insert_zero_level is True:
            X0 = self.xp.ones_like(flat[..., :1])
        elif insert_zero_level is False:
            X0 = self.xp.zeros_like(flat[..., :1])
        else:  # None
            if X0 is None:  # missing → insert zeros
                X0 = self.xp.zeros_like(flat[..., :1])

        return (X0,) + tuple(parts)

    @dummy_jit(static_argnums=0, static_argnames=("start_at_level_one",), dynamic_batchtime=("levels",))
    def tensor_to_flat(self, levels: DenseElem, *, start_at_level_one: bool = False) -> Array:
        """
        Concatenate per-degree levels into a single flattened representation.

        Parameters
        ----------
        levels : sequence of ndarray
            ``(X₀, X₁, ..., X_N)`` with matching batch shapes.
        start_at_level_one : bool, optional
            If ``True``, pack only degrees ``1..N`` (drop ``X₀`` in the output).
            Otherwise pack ``0..N``.

        Returns
        -------
        ndarray
            The concatenation along the last axis.
        """
        levels = tuple(levels)
        if not levels:
            return self.xp.asarray([], dtype=float)
        if start_at_level_one:
            levels = levels[1:]
        return self.xp.concat(levels, axis=-1) if levels else self.xp.asarray([], dtype=levels[0].dtype)


    def _index_add(self, target: DenseElem, source: DenseElem, index: DenseElem):
        """
        - Should be private class
        - Example:
        target = [0, 1, 3]
        source = [1, 5, 4]
        index = [0,2,2]
        => Result = [0+1,1+0,3+5+4] = [1,1,12]
        """
        pass


    def index_add(self, a, target_idx):
        """
        Accepts an 2d-array of shape (N_sample, n). For each sample the values are 
        added up based on the indices specified in target_idx.

        Example:

        a = np.array([[1,2,3,4,1,4]]) # shape (1,6)
        target_idx = np.array([1,2,2,1]) # shape (4,) with sum 6
        >>> np.array([1,5,5,4]) # =(1,2+3,4+1,4)
        """
        pass


    def tensor_shuffle_homogeneous(self, Ai, Bi, d):
        """
        sparse einsum 
        at initialization, get d and if wanted, precomoute right away, 
        check whether precomputed shuffle is here or not; if yes, check if dimension d matches.
        check also if precomputed.
        """
        pass


    def tensor_shuffle(self, A, B):
        """
        sparse einsum 
        add optional d: if provided assert that d matches dimension of A and B; if d is not proivided, then 
        from second level of A -> infer dimension d and truncation levels
        if not precoputed for d and n and m then do; 
        """
        pass