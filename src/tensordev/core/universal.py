from __future__ import annotations

import functools
import itertools
from typing import Optional, Sequence, Tuple, Union, Literal, List, TypeVar, Generic, Callable, Protocol, Any

from tensordev.core.utils.annotations import jit as dummy_jit


class _Array(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> Any: ...

    def __getitem__(self, key: Any) -> _Array: ...

    def __add__(self, other: Any) -> _Array: ...

    def __radd__(self, other: Any) -> _Array: ...

    def __mul__(self, other: Any) -> _Array: ...

    def __rmul__(self, other: Any) -> _Array: ...

    def __pow__(self, other: Any) -> _Array: ...

    def sum(self, *args: Any, **kwargs: Any) -> _Array: ...

    def reshape(self, *args: Any, **kwargs: Any) -> _Array: ...


class _ArrayNamespace(Protocol):
    def stack(self, arrays: Sequence[Any], axis: int = 0) -> _Array: ...

    def moveaxis(self, x: _Array, source: Any, destination: Any) -> _Array: ...

    def expand_dims(self, x: _Array, axis: int) -> _Array: ...

    def reshape(self, x: _Array, shape: tuple[int, ...]) -> _Array: ...

    def broadcast_shapes(self, *shapes: tuple[int, ...]) -> tuple[int, ...]: ...

    def broadcast_to(self, x: _Array, shape: tuple[int, ...]) -> _Array: ...

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> _Array: ...

    def zeros_like(self, x: _Array) -> _Array: ...

    def ones_like(self, x: _Array) -> _Array: ...

    def asarray(self, obj: Any, dtype: Any = None) -> _Array: ...

    def diff(self, x: _Array, axis: int) -> _Array: ...

    def concat(self, arrays: Sequence[Any], axis: int = 0) -> _Array: ...


Array = TypeVar("Array", bound=_Array)
Elem = Sequence[Optional[Array]]  # one tensor-algebra element (level-list)
DenseElem = Tuple[Array, ...]  # level k has last dim d**k stating at level k=0; no Nones; shared batch shape
DenseElemFirstOn = Tuple[Array, ...]  # level k has last dim d**k starting at level k=1 no Nones; shared batch shape,


class _TensorSliceProxy:
    """Proxy returned by ``tensor_slice(A)`` to support ``A[key]`` syntax."""
    __slots__ = ("_levels",)

    def __init__(self, levels: tuple) -> None:
        self._levels = levels

    def __getitem__(self, key: Any) -> tuple:
        return tuple(lvl[key] for lvl in self._levels)


class Universal(Generic[Array]):
    def __init__(self, xp: _ArrayNamespace):
        self.xp = xp

    # ----------------------------------------------------------------------
    # Axes iteration and reduction utilites
    # ----------------------------------------------------------------------

    @dummy_jit(static_argnums=0, static_argnames=("axis",), dynamic_batch=("X",))
    def tensor_stack(self, X: List[DenseElem], *, axis: int) -> tuple:
        L = len(X[-1])
        ndim = X[-1][-1].ndim
        stack_axis = axis if axis >= 0 else (ndim + 1 + axis)
        return tuple(self.xp.stack([e[k] for e in X], axis=stack_axis) for k in range(L))

    @dummy_jit(static_argnums=0, static_argnames=("source", "destination"), dynamic_batch=("X",))
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

    @dummy_jit(static_argnums=0, static_argnames=("trunc",), dynamic_batch=("A", "B"))
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

    @dummy_jit(static_argnums=0, dynamic_batch=("Ai", "Bj"))
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

    @dummy_jit(static_argnums=(0, 3), dynamic_batch=("Ai", "v"))
    def tensor_shuffle_vector_homogeneous(self, Ai: Array, v: Array, i: int) -> Array:
        """
        Homogeneous shuffle product ``(A_i ⊔ v)_{i+1}`` where ``v`` is degree-1.

        For each output word ``(l_0, ..., l_i)`` the result is:
            sum_{p=0}^{i} A_i[l_0,...,l_{p-1},l_{p+1},...,l_i] * v[l_p]

        Parameters
        ----------
        Ai : Array, shape ``batch + (d**i,)``
        v : Array, shape ``batch + (d,)`` — batch shapes are broadcast.
        i : int
            Degree of ``Ai``.

        Returns
        -------
        Array, shape ``batch + (d**(i+1),)``
        """
        xp = self.xp
        d = v.shape[-1]
        batch = xp.broadcast_shapes(Ai.shape[:-1], v.shape[:-1])
        A = xp.broadcast_to(Ai, batch + (d ** i,)).reshape(batch + (d,) * i)
        v_ = xp.broadcast_to(v, batch + (d,))
        result = xp.zeros(batch + (d,) * (i + 1), dtype=Ai.dtype)
        for p in range(i + 1):
            A_exp = xp.expand_dims(A, axis=len(batch) + p)
            v_exp = v_.reshape(batch + (1,) * p + (d,) + (1,) * (i - p))
            result = result + A_exp * v_exp
        return result.reshape(batch + (d ** (i + 1),))

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "a_first_on"), dynamic_batch=("A", "v"))
    def tensor_shuffle_vector(
            self,
            A: Union[DenseElem, DenseElemFirstOn],
            v: Array,
            *,
            trunc: Optional[int] = None,
            a_first_on: bool = False,
    ) -> DenseElemFirstOn:
        """
        Graded shuffle product ``C = A ⊔ v`` where ``v`` is degree-1.

        Output is in first-on format (degrees 1, 2, ...):
            C_n = A_{n-1} ⊔ v   for n = 1, ..., min(NA + 1, trunc)

        Parameters
        ----------
        A : DenseElem or DenseElemFirstOn
            Graded element. If ``a_first_on=True``, starts at degree 1.
        v : Array, shape ``batch + (d,)``
            Degree-1 vector; batch shapes are broadcast per level.
        trunc : int, optional
            Maximum output degree.
        a_first_on : bool, default False
            Whether ``A`` starts at degree 1.

        Returns
        -------
        DenseElemFirstOn
            Output levels starting at degree 1.
        """
        A = tuple(A)
        if not A:
            return tuple()
        a0 = 1 if a_first_on else 0
        NA = len(A) + a0 - 1
        N = NA + 1
        if trunc is not None:
            N = min(N, trunc)
        out: List[Array] = []
        for n in range(1, N + 1):
            i = n - 1
            if i < a0 or i > NA:
                continue
            out.append(self.tensor_shuffle_vector_homogeneous(A[i - a0], v, i))
        return tuple(out)

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "a_first_on", "b_first_on"),
               dynamic_batch=("A", "B"))
    def tensor_product(
            self,
            A: Union[DenseElem, DenseElemFirstOn],
            B: Union[DenseElem, DenseElemFirstOn],
            trunc: Optional[int] = None,
            *,
            a_first_on: bool = False,
            b_first_on: bool = False,
    ) -> Union[DenseElem, DenseElemFirstOn]:
        """
        Graded (Cauchy-type) product ``C = A ⊗ B`` in the free tensor algebra.

        For each degree ``n``,
            C_n = ∑_{i+j=n} flatten(tensor_product_homogeneous(A_i, B_j)).

        Parameters
        ----------
        A, B : tuple of ndarray
            Levels of the left/right factors, each of shape ``batch + (d**k,)``.

            By default, inputs are interpreted as
            ``(A_0, A_1, ..., A_N)`` and ``(B_0, B_1, ..., B_M)``.

            If ``a_first_on=True``, then ``A`` is interpreted as
            ``(A_1, A_2, ..., A_N)``.

            If ``b_first_on=True``, then ``B`` is interpreted analogously.

        trunc : int, optional
            If given, truncate the result to degrees up to ``trunc``.

        a_first_on, b_first_on : bool, default=False
            Whether the corresponding input starts at degree ``1`` rather than ``0``.

        Returns
        -------
        tuple of ndarray
            Product levels. If both flags are ``False``, returns
            ``(C_0, C_1, ..., C_N)``. Otherwise returns ``(C_1, C_2, ..., C_N)``.
        """
        A = tuple(A)
        B = tuple(B)

        if len(A) == 0 or len(B) == 0:
            return tuple()

        a0 = 1 if a_first_on else 0
        b0 = 1 if b_first_on else 0
        N = len(A) + len(B) + a0 + b0 - 2
        if trunc is not None:
            N = min(N, trunc)

        out: List[Array] = []
        start = 0

        if a_first_on or b_first_on:
            start = 1
            if a_first_on and b_first_on:
                if N < 1:
                    return tuple()
                out.append(self.xp.zeros_like(A[0]))
                start = 2

        for n in range(start, N + 1):
            i_min = max(a0, n - (len(B) + b0 - 1))
            i_max = min(len(A) + a0 - 1, n - b0)
            i = i_min
            term = self.tensor_product_homogeneous(A[i - a0], B[n - i - b0])
            for i in range(i_min + 1, i_max + 1):
                term = term + self.tensor_product_homogeneous(A[i - a0], B[n - i - b0])
            out.append(term)

        return tuple(out)

    @dummy_jit(static_argnums=0, dynamic_batch=("A",), full_dynamic=("alpha",))
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

    @dummy_jit(static_argnums=0, dynamic_batch=("A",), full_dynamic=("c",))
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

    @dummy_jit(static_argnums=0, dynamic_batch=("Ak", "Bk"))
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

    @dummy_jit(static_argnums=0, dynamic_batch=("A", "B"))
    def tensor_inner_product(
            self,
            A: Union[DenseElem, DenseElemFirstOn],
            B: Union[DenseElem, DenseElemFirstOn],
    ) -> Array:
        """
        Canonical Euclidean inner product, summing level-wise dot products over the
        last axis.

        Parameters
        ----------
        A, B : DenseElem or DenseElemFirstOn
            Graded elements with matching batch shape. If the inputs start at level 1,
            the inner product is taken over the positive levels only.

        Returns
        -------
        Array
            Batch-shaped array with the inner product.
        """
        K = min(len(A), len(B))
        if K == 0:
            return self.xp.asarray(0.0)

        acc = self.tensor_inner_product_homogeneous(A[0], B[0])
        for k in range(1, K):
            acc = acc + self.tensor_inner_product_homogeneous(A[k], B[k])
        return acc

    @dummy_jit(static_argnums=0, dynamic_batch=("Ai", "Yni"))
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

    @dummy_jit(static_argnums=0, dynamic_batch=("Bj", "Ynj"))
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

    def tensor_adjoint_product(
            self,
            W: Union[DenseElem, DenseElemFirstOn],
            Y: Union[DenseElem, DenseElemFirstOn],
            trunc: Optional[int],
            side: Literal["left", "right"],
            *,
            w_first_on: bool = False,
            y_first_on: bool = False,
            first_on_out: bool = False,
    ) -> Union[DenseElem, DenseElemFirstOn]:
        """
        Compute the graded adjoint product of two truncated tensor-algebra elements.

        This is the shared implementation behind the left and right adjoint actions.
        It contracts homogeneous levels of ``W`` against shifted homogeneous levels
        of ``Y`` and sums all contributions that land in the same output degree.

        Degree convention
        -----------------
        Let ``W_i`` denote the degree-``i`` level of ``W`` and ``Y_j`` the degree-``j``
        level of ``Y``. For each output degree ``n``, this routine computes

            Z_n = sum_i Adj(W_i, Y_{n+i}),

        where ``Adj`` is either the left or right homogeneous adjoint contraction,
        depending on ``side``.

        More precisely:
        - for ``side="left"``, use ``tensor_adjoint_left_homogeneous(W_i, Y_{n+i})``;
        - for ``side="right"``, use ``tensor_adjoint_right_homogeneous(W_i, Y_{n+i})``.

        Input representation
        --------------------
        The inputs may be stored either as dense graded elements or as first-on
        graded elements:

        - dense:
            ``(A_0, A_1, ..., A_N)``
        - first-on:
            ``(A_1, A_2, ..., A_N)``

        The flags ``w_first_on`` and ``y_first_on`` specify which convention is used
        for ``W`` and ``Y`` respectively.

        Output representation
        ---------------------
        The returned tuple is intended to be either

        - dense, if ``first_on_out=False``:
            ``(Z_0, Z_1, ..., Z_M)``
        - first-on, if ``first_on_out=True``:
            ``(Z_1, Z_2, ..., Z_M)``

        where ``M`` is the largest output degree allowed by the input ranges and by
        ``trunc``.

        Important
        ---------
        This routine should return output in a canonical graded format:
        - dense output must begin at degree 0,
        - first-on output must begin at degree 1.

        If the lowest nonzero computable degree is higher than that starting degree,
        the missing lower degrees should be represented by zero levels rather than
        being silently skipped. Otherwise the returned tuple no longer has a well-defined
        dense/first-on interpretation.

        Parameters
        ----------
        W :
            Multiplier graded element. Its stored levels are interpreted as starting
            at degree 0 or degree 1 according to ``w_first_on``.
        Y :
            Target graded element. Its stored levels are interpreted as starting
            at degree 0 or degree 1 according to ``y_first_on``.
        trunc :
            Optional maximum output degree. If ``None``, all output degrees permitted
            by the available levels of ``W`` and ``Y`` are produced.
        side :
            Which adjoint action to use:
            - ``"left"``  -> left homogeneous adjoint contraction,
            - ``"right"`` -> right homogeneous adjoint contraction.
        w_first_on :
            Whether ``W`` is stored in first-on format.
        y_first_on :
            Whether ``Y`` is stored in first-on format.
        first_on_out :
            Whether to return the result in first-on format.

        Returns
        -------
        DenseElem or DenseElemFirstOn
            The graded adjoint product, truncated at degree ``trunc`` if requested,
            in the storage convention specified by ``first_on_out``.

        Notes
        -----
        If ``W`` has degrees ``i`` in some range and ``Y`` has degrees ``j`` in some
        range, then the admissible output degrees are those for which at least one
        pair ``(i, j)`` satisfies ``j = n + i``. This determines the natural output
        degree window before truncation is applied.
        """
        W, Y = tuple(W), tuple(Y)

        if len(W) == 0 or len(Y) == 0:
            return tuple()

        w0 = 1 if w_first_on else 0
        y0 = 1 if y_first_on else 0

        w_last = w0 + len(W) - 1
        y_last = y0 + len(Y) - 1

        contract = (
            self.tensor_adjoint_left_homogeneous
            if side == "left"
            else self.tensor_adjoint_right_homogeneous
        )

        start = 1 if first_on_out else 0
        n_min = max(start, y0 - w_last)
        n_max = y_last - w0
        if trunc is not None:
            n_max = min(n_max, trunc)

        if n_max < start:
            return tuple()

        # Find a template for each output degree from the first available contributing term.
        def zero_for_degree(n: int):
            i_min = max(w0, y0 - n)
            i_max = min(w_last, y_last - n)
            if i_min > i_max:
                raise ValueError(f"No contributing term available to infer shape for degree {n}.")
            term = contract(W[i_min - w0], Y[n + i_min - y0])
            return self.xp.zeros_like(term)

        out = []
        for n in range(start, n_max + 1):
            if n < n_min:
                out.append(zero_for_degree(n))
                continue

            i_min = max(w0, y0 - n)
            i_max = min(w_last, y_last - n)

            terms = [contract(W[i - w0], Y[n + i - y0]) for i in range(i_min, i_max + 1)]
            out.append(terms[0] if len(terms) == 1 else sum(terms[1:], terms[0]))

        return tuple(out)

    # ----------------------------------------------------------------------
    # Polynomial / Series Operations
    # ----------------------------------------------------------------------

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "output_zero_level"), dynamic_batch=("g", "X"))
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
        - For the pure degree-1 case (`len(X) == 1`), use the fused Horner scheme
          for `g ⊗ exp(X₁)`.
        - In the general case, build the truncated exponential
              exp(X) = I + X + X^{⊗2}/2! + ... + X^{⊗trunc}/trunc!
          exactly via the truncated tensor power series, and then compute
              Y = g ⊗ exp(X).

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
        if trunc < 0:
            raise ValueError("trunc>=0")
        if trunc == 0 or len(X) == 0:
            out = g[:trunc + 1]
            return out if output_zero_level else out[1:]

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
        # Build E = exp(X) exactly from the truncated power series, then set Y = g ⊗ E.
        zero0 = self.xp.zeros_like(X[0][..., :1])
        one0 = self.xp.ones_like(zero0)

        # Dense version of X with explicit zero level.
        H: DenseElem = (zero0,) + tuple(X[:trunc])

        # E = I + H + H^2/2! + ... + H^trunc/trunc!
        E: DenseElem = (one0,)
        power: DenseElem = (one0,)  # H^0
        inv_fact = 1.0

        for n in range(1, trunc + 1):
            power = self.tensor_product(power, H, trunc=trunc)
            inv_fact /= float(n)
            E = self.tensor_summation(
                E,
                self.tensor_scalar_multiply(power, inv_fact),
                trunc=trunc,
            )

        Y = self.tensor_product(g, E, trunc=trunc)
        return Y if output_zero_level else Y[1:]

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "output_zero_level"), dynamic_batch=("X",))
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

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "output_zero_level"), dynamic_batch=("X",))
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

    def tensor_slice(self, A: DenseElem) -> _TensorSliceProxy:
        """
        Return a proxy that applies an index or slice to every level of ``A``.

        Usage::

            td.tensor_slice(A)[i:j]       # → tuple(lvl[i:j] for lvl in A)
            td.tensor_slice(A)[..., 0]    # → tuple(lvl[..., 0] for lvl in A)
        """
        return _TensorSliceProxy(tuple(A))

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

    @dummy_jit(static_argnums=0, static_argnames=("dim", "insert_zero_level"), dynamic_batch=("X",))
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

    @dummy_jit(static_argnums=0, static_argnames=("start_at_level_one",), dynamic_batch=("levels",))
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

    # ----------------------------------------------------------------------
    # Matrix Tensor Operations
    # ----------------------------------------------------------------------

    def _canonicalize_matrix_axes(
            self,
            X: Array,
            row_axis: int,
            col_axis: int,
    ) -> Tuple[Array, int, int]:
        """
        Move the matrix axes of ``X`` to the canonical positions ``(-3, -2)``.

        Parameters
        ----------
        X : ndarray
            Array with a final tensor-coordinate axis.
        row_axis, col_axis : int
            Positions of the row and column axes.

        Returns
        -------
        Xc : ndarray
            Array with row and column axes moved to ``(-3, -2)``.
        row_axis, col_axis : int
            Normalized original axis positions.
        """
        ndim = X.ndim
        row_axis = row_axis % ndim
        col_axis = col_axis % ndim

        if row_axis == col_axis:
            raise ValueError("row_axis and col_axis must be distinct.")
        if row_axis == ndim - 1 or col_axis == ndim - 1:
            raise ValueError("row_axis and col_axis may not coincide with the tensor axis.")

        Xc = self.xp.moveaxis(X, (row_axis, col_axis), (-3, -2))
        return Xc, row_axis, col_axis

    def _restore_matrix_axes(
            self,
            X: Array,
            row_axis: int,
            col_axis: int,
    ) -> Array:
        """
        Restore canonical matrix axes ``(-3, -2)`` to ``(row_axis, col_axis)``.
        """
        return self.xp.moveaxis(X, (-3, -2), (row_axis, col_axis))

    @dummy_jit(static_argnums=0, static_argnames=("row_axis", "col_axis"), dynamic_batch=("A",),
               full_dynamic=("M",))
    def tensor_matrix_product_right_homogeneous(
            self,
            A: Array,
            M: Array,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> Array:
        """
        Right-multiply a homogeneous matrix-valued tensor-algebra coefficient by
        a numeric matrix.

        The default axis convention is

            ``batch + (T?, n, k, d_r)``,

        where the final axis stores tensor coordinates, an optional time axis
        precedes the matrix axes, and the matrix axes are ``(row_axis, col_axis)``.

        Parameters
        ----------
        A : ndarray
            Homogeneous matrix-valued coefficient with shape
            ``batch + (T?, n, k, d_r)`` up to axis placement.
        M : ndarray
            Numeric matrix with shape ``(..., k, l)``.
        row_axis, col_axis : int, default=(-3, -2)
            Positions of the matrix row and column axes in ``A``.

        Returns
        -------
        ndarray
            Homogeneous coefficient with shape
            ``batch + (T?, n, l, d_r)`` up to axis placement.
        """
        A, row_axis, col_axis = self._canonicalize_matrix_axes(A, row_axis, col_axis)

        a = self.xp.expand_dims(A, axis=-2)  # ... n k 1 a
        m = self.xp.expand_dims(M, axis=-1)  # ... k l 1
        m = self.xp.expand_dims(m, axis=-4)  # ... 1 k l 1
        out = (a * m).sum(axis=-3)  # ... n l a

        return self._restore_matrix_axes(out, row_axis, col_axis)

    @dummy_jit(static_argnums=0, static_argnames=("row_axis", "col_axis"), dynamic_batch=("A",),
               full_dynamic=("M",))
    def tensor_matrix_product_left_homogeneous(
            self,
            M: Array,
            A: Array,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> Array:
        """
        Left-multiply a homogeneous matrix-valued tensor-algebra coefficient by
        a numeric matrix.

        The default axis convention is

            ``batch + (T?, k, l, d_r)``,

        where the final axis stores tensor coordinates, an optional time axis
        precedes the matrix axes, and the matrix axes are ``(row_axis, col_axis)``.

        Parameters
        ----------
        M : ndarray
            Numeric matrix with shape ``(..., n, k)``.
        A : ndarray
            Homogeneous matrix-valued coefficient with shape
            ``batch + (T?, k, l, d_r)`` up to axis placement.
        row_axis, col_axis : int, default=(-3, -2)
            Positions of the matrix row and column axes in ``A``.

        Returns
        -------
        ndarray
            Homogeneous coefficient with shape
            ``batch + (T?, n, l, d_r)`` up to axis placement.
        """
        A, row_axis, col_axis = self._canonicalize_matrix_axes(A, row_axis, col_axis)

        m = self.xp.expand_dims(M, axis=-1)  # ... n k 1
        m = self.xp.expand_dims(m, axis=-1)  # ... n k 1 1
        a = self.xp.expand_dims(A, axis=-4)  # ... 1 k l a
        out = (m * a).sum(axis=-3)  # ... n l a

        return self._restore_matrix_axes(out, row_axis, col_axis)

    @dummy_jit(static_argnums=0, static_argnames=("row_axis", "col_axis"), dynamic_batch=("A", "B"))
    def tensor_matrix_product_homogeneous(
            self,
            A: Array,
            B: Array,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> Array:
        """
        Multiply two homogeneous matrix-valued tensor-algebra coefficients.

        If ``A`` has shape ``batch + (T?, n, k, d_i)`` and ``B`` has shape
        ``batch + (T?, k, l, d_j)`` up to axis placement, then the result has
        shape ``batch + (T?, n, l, d_i d_j)`` up to axis placement.

        Parameters
        ----------
        A, B : ndarray
            Homogeneous matrix-valued coefficients.
        row_axis, col_axis : int, default=(-3, -2)
            Positions of the matrix row and column axes in ``A`` and ``B``.
            Both operands are assumed to follow the same convention.

        Returns
        -------
        ndarray
            Homogeneous matrix-valued coefficient.
        """
        A, row_axis, col_axis = self._canonicalize_matrix_axes(A, row_axis, col_axis)
        B, _, _ = self._canonicalize_matrix_axes(B, row_axis, col_axis)

        x = self.xp.expand_dims(A, axis=-2)  # ... n k 1 a
        x = self.xp.expand_dims(x, axis=-1)  # ... n k 1 a 1
        y = self.xp.expand_dims(B, axis=-4)  # ... 1 k l b
        y = self.xp.expand_dims(y, axis=-2)  # ... 1 k l 1 b
        out = (x * y).sum(axis=-4)  # ... n l a b
        out = self.xp.reshape(out, out.shape[:-2] + (out.shape[-2] * out.shape[-1],))

        return self._restore_matrix_axes(out, row_axis, col_axis)

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "row_axis", "col_axis"), dynamic_batch=("A",),
               full_dynamic=("M",))
    def tensor_matrix_product_right(
            self,
            A: DenseElem,
            M: Array,
            trunc: Optional[int] = None,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> DenseElem:
        """
        Right-multiply a matrix-valued tensor-algebra element by a numeric matrix.

        Each homogeneous level ``A[r]`` is interpreted using the matrix-axis
        convention specified by ``row_axis`` and ``col_axis``. Under the default
        convention, levels have shape

            ``batch + (T?, n, k, d_r)``.

        Parameters
        ----------
        A : tuple of ndarray
            Matrix-valued tensor-algebra element.
        M : ndarray
            Numeric matrix with shape ``(..., k, l)``.
        trunc : int, optional
            If given, keep only degrees ``0, ..., trunc``.
        row_axis, col_axis : int, default=(-3, -2)
            Positions of the matrix row and column axes in each level of ``A``.

        Returns
        -------
        tuple of ndarray
            Matrix-valued tensor-algebra element with column dimension updated
            from ``k`` to ``l``.
        """
        A = tuple(A)
        if not A:
            return A

        N = len(A) - 1
        if trunc is not None:
            N = min(N, trunc)

        return tuple(
            self.tensor_matrix_product_right_homogeneous(
                A[r], M, row_axis=row_axis, col_axis=col_axis
            )
            for r in range(N + 1)
        )

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "row_axis", "col_axis"), dynamic_batch=("A",),
               full_dynamic=("M",))
    def tensor_matrix_product_left(
            self,
            M: Array,
            A: DenseElem,
            trunc: Optional[int] = None,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> DenseElem:
        """
        Left-multiply a matrix-valued tensor-algebra element by a numeric matrix.

        Each homogeneous level ``A[r]`` is interpreted using the matrix-axis
        convention specified by ``row_axis`` and ``col_axis``. Under the default
        convention, levels have shape

            ``batch + (T?, k, l, d_r)``.

        Parameters
        ----------
        M : ndarray
            Numeric matrix with shape ``(..., n, k)``.
        A : tuple of ndarray
            Matrix-valued tensor-algebra element.
        trunc : int, optional
            If given, keep only degrees ``0, ..., trunc``.
        row_axis, col_axis : int, default=(-3, -2)
            Positions of the matrix row and column axes in each level of ``A``.

        Returns
        -------
        tuple of ndarray
            Matrix-valued tensor-algebra element with row dimension updated from
            ``k`` to ``n``.
        """
        A = tuple(A)
        if not A:
            return A

        N = len(A) - 1
        if trunc is not None:
            N = min(N, trunc)

        return tuple(
            self.tensor_matrix_product_left_homogeneous(
                M, A[r], row_axis=row_axis, col_axis=col_axis
            )
            for r in range(N + 1)
        )

    @dummy_jit(static_argnums=0, static_argnames=("trunc", "row_axis", "col_axis"), dynamic_batch=("A", "B"))
    def tensor_matrix_product(
            self,
            A: DenseElem,
            B: DenseElem,
            trunc: Optional[int] = None,
            row_axis: int = -3,
            col_axis: int = -2,
    ) -> DenseElem:
        """
        Multiply two matrix-valued tensor-algebra elements.

        If ``A`` and ``B`` are represented levelwise as
        ``batch + (T?, n, k, d_i)`` and ``batch + (T?, k, l, d_j)``,
        respectively, then this computes the graded Cauchy product

            ``(A * B)[r] = sum_{i+j=r} A[i] @ B[j]``,

        where the homogeneous products contract the inner matrix index and use
        the tensor product on the final tensor-coordinate axis.

        Parameters
        ----------
        A, B : tuple of ndarray
            Matrix-valued tensor-algebra elements.
        trunc : int, optional
            If given, truncate the product to degrees ``0, ..., trunc``.
        row_axis, col_axis : int, default=(-3, -2)
            Positions of the matrix row and column axes in each homogeneous
            level of ``A`` and ``B``. Both operands are assumed to use the
            same convention.

        Returns
        -------
        tuple of ndarray
            Matrix-valued tensor-algebra product.
        """
        A = tuple(A)
        B = tuple(B)

        if len(A) == 0 or len(B) == 0:
            return tuple()

        N = len(A) + len(B) - 2
        if trunc is not None:
            N = min(N, trunc)

        out: List[Array] = []
        for r in range(N + 1):
            i_min = max(0, r - (len(B) - 1))
            i_max = min(len(A) - 1, r)

            term = self.tensor_matrix_product_homogeneous(
                A[i_min], B[r - i_min], row_axis=row_axis, col_axis=col_axis
            )
            for i in range(i_min + 1, i_max + 1):
                term = term + self.tensor_matrix_product_homogeneous(
                    A[i], B[r - i], row_axis=row_axis, col_axis=col_axis
                )
            out.append(term)

        return tuple(out)
