from __future__ import annotations

import functools
import itertools
from typing import Optional, Tuple, Callable, List, TypeVar, Generic

from tensordev.core.universal import _Array, _ArrayNamespace
from tensordev.core.utils.annotations import jit as dummy_jit

Array = TypeVar("Array", bound=_Array)
DenseElem = Tuple[Array, ...]


class SequentialCore(Generic[Array]):
    def __init__(self, xp: _ArrayNamespace, default_time_axis: int = -2):
        self.xp = xp
        self.default_time_axis = default_time_axis

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _stack(self, X: List[DenseElem], *, axis: int) -> DenseElem:
        L = len(X[-1])
        ndim = X[-1][-1].ndim
        stack_axis = axis if axis >= 0 else (ndim + 1 + axis)
        return tuple(self.xp.stack([e[k] for e in X], axis=stack_axis) for k in range(L))

    def _moveaxis(self, X: DenseElem, *, source: int, destination: int) -> DenseElem:
        return tuple(self.xp.moveaxis(a, source, destination) for a in X)

    def _mapper(self, fun: Callable[[DenseElem], DenseElem]) -> Callable[[DenseElem], DenseElem]:
        unstack = lambda x: [tuple(x[k][i] for k in range(len(x))) for i in range(len(x[0]))]
        return lambda seq: self._stack(list(map(fun, unstack(seq))), axis=0)

    def _reducer(
            self,
            fun: Callable[[DenseElem, DenseElem], DenseElem],
            *,
            neutral: DenseElem,
            seed: DenseElem,
            in_tree: bool = False,
    ) -> Callable[[DenseElem], DenseElem]:
        def reduce_fn(X):
            X = tuple(X)
            S = int(X[0].shape[0])
            step = lambda t: tuple(a[t] for a in X)
            return functools.reduce(fun, (step(t) for t in range(S)), seed)
        return reduce_fn

    def _accumulator(
            self,
            fun: Callable[[DenseElem, DenseElem], DenseElem],
            *,
            neutral: DenseElem,
            seed: DenseElem,
            in_tree: bool = False,
    ) -> Callable[[DenseElem], Tuple[DenseElem, DenseElem]]:
        def scan_fn(X):
            X = tuple(X)
            S = int(X[0].shape[0])
            step = lambda t: tuple(a[t] for a in X)
            prefixes = itertools.accumulate(
                itertools.chain([seed], (step(t) for t in range(S))),
                fun,
            )
            ys = list(itertools.islice(prefixes, 1, None))
            return ys[-1], self._stack(ys, axis=0)
        return scan_fn

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    @dummy_jit(static_argnums=0,
               static_argnames=("reduce_op", "axis", "accumulate_in_tree"),
               dynamic_batchtime=("X",))
    def tensor_reduce(
            self,
            X: DenseElem,
            *,
            reduce_op: Callable[[DenseElem, DenseElem], DenseElem],
            neutral: DenseElem,
            seed: Optional[DenseElem] = None,
            axis: Optional[int] = None,
            accumulate_in_tree: bool = False,
    ) -> DenseElem:
        """Left-fold `reduce_op` over `X` along `axis`, starting from `seed` (defaults to `neutral`)."""
        axis = self.default_time_axis if axis is None else axis
        X = self._moveaxis(X, source=axis, destination=0)
        seed_ = seed if seed is not None else neutral
        return self._reducer(reduce_op, neutral=neutral, seed=seed_, in_tree=accumulate_in_tree)(X)

    @dummy_jit(static_argnums=0,
               static_argnames=("reduce_op", "axis", "output_starting_point", "accumulate_in_tree"),
               dynamic_batchtime=("X",))
    def tensor_accumulate(
            self,
            X: DenseElem,
            *,
            reduce_op: Callable[[DenseElem, DenseElem], DenseElem],
            neutral: DenseElem,
            seed: Optional[DenseElem] = None,
            axis: Optional[int] = None,
            output_starting_point: bool = False,
            accumulate_in_tree: bool = False,
    ) -> DenseElem:
        """Inclusive prefix scan of `reduce_op` over `X` along `axis`."""
        axis = self.default_time_axis if axis is None else axis
        X = self._moveaxis(X, source=axis, destination=0)
        seed_ = seed if seed is not None else neutral
        N = len(neutral) - 1
        stacked = tuple(self.xp.stack([seed_[k], *(X[k])], axis=0) for k in range(N + 1))
        _, zs = self._accumulator(reduce_op, neutral=neutral, seed=seed_, in_tree=accumulate_in_tree)(stacked)
        if not output_starting_point:
            zs = tuple(a[1:] for a in zs)
        return self._moveaxis(zs, source=0, destination=axis)

    @dummy_jit(static_argnums=0,
               static_argnames=("reduce_op", "acc_op", "axis", "block_size", "accumulate",
                                "output_starting_point", "first_apply_all",
                                "reduce_in_tree", "accumulate_in_tree"),
               dynamic_batchtime=("X",))
    def tensor_abra(
            self,
            X: DenseElem,
            *,
            reduce_op: Callable[[DenseElem, DenseElem], DenseElem],
            acc_op: Callable[[DenseElem, DenseElem], DenseElem],
            neutral: DenseElem,
            axis: Optional[int] = None,
            block_size: Optional[int] = None,
            accumulate: bool = False,
            seed: Optional[DenseElem] = None,
            output_starting_point: bool = False,
            first_apply_all: bool = False,
            reduce_in_tree: bool = False,
            accumulate_in_tree: bool = False,
    ) -> DenseElem:
        """
        Apply, Block, Reduce and Accumulate over `X` along `axis` using caller-supplied ops.

        Parameters
        ----------
        reduce_op : (carry, step) -> carry
            The sequential left-fold operation applied within each block.

            When ``first_apply_all=False``: used directly as the scan step,
            so ``carry`` and ``step`` may have different types (e.g.
            ``tensor_fmexp`` where carry is a full ``DenseElem`` and step
            is a ``DenseElemFirstOn``).  No algebraic assumptions are made.

            When ``first_apply_all=True``: ``reduce_op(neutral, step)`` acts
            as the implicit **apply** function that lifts each raw step to a
            full algebra element before combining with ``acc_op``.  For
            correctness the identity
            ``reduce_op(carry, step) == acc_op(carry, reduce_op(neutral, step))``
            must hold for all ``carry`` and ``step``.

        acc_op : (elem, elem) -> elem
            Associative combining operation.  Used for across-block
            accumulation and, when ``first_apply_all=True``, for within-block
            combination after the lift.

        neutral : DenseElem
            Identity element: ``acc_op(neutral, x) == x`` for all ``x``.

        seed : DenseElem, optional
            Starting accumulation value; defaults to ``neutral``.

        first_apply_all : bool, default False
            If True, every step is first lifted via ``reduce_op(neutral, step)``
            and the lifted elements are then combined with ``acc_op``.  This
            separates the "apply" and "combine" phases and enables use of
            ``reduce_in_tree=True`` (since ``acc_op`` is associative).
            If False, ``reduce_op`` is streamed directly as a left-fold.

        reduce_in_tree : bool, default False
            If True and ``first_apply_all=False``: uses an associative tree
            scan with ``reduce_op`` inside each block.  **Requires
            ``reduce_op`` to be associative** (same type for both arguments).
            If True and ``first_apply_all=True``: uses an associative tree
            scan with ``acc_op`` inside each block.  Requires ``acc_op`` to
            be associative.

        accumulate_in_tree : bool, default False
            If True, use an associative tree scan for the across-block
            accumulation.  Requires ``acc_op`` to be associative.
        """
        axis = self.default_time_axis if axis is None else axis
        N = len(neutral) - 1
        seed_ = seed if seed is not None else neutral
        X = self._moveaxis(X, source=axis, destination=0)

        S = X[0].shape[0]
        B = S if (block_size in (None, -1)) else int(block_size)
        q, r = divmod(S, B)
        if r:
            raise ValueError(f"tensor_abra: block_size={B} must divide S={S}.")
        X_blocks = tuple(L.reshape(q, B, *L.shape[1:]) for L in X)

        if first_apply_all:
            lift = lambda step: reduce_op(neutral, step)
            lifted_reducer = self._reducer(acc_op, neutral=neutral, seed=neutral, in_tree=reduce_in_tree)
            block_reducer = self._mapper(lambda block: lifted_reducer(self._mapper(lift)(block)))
        else:
            block_reducer = self._mapper(
                self._reducer(reduce_op, neutral=neutral, seed=neutral, in_tree=reduce_in_tree)
            )
        blocks = block_reducer(X_blocks)

        if not accumulate:
            if output_starting_point:
                return tuple(self.xp.stack([seed_[k], *(blocks[k])], axis=axis) for k in range(N + 1))
            if q == 1:
                return tuple(a[0] for a in blocks)
            return self._moveaxis(blocks, source=0, destination=axis)

        accumulator = self._accumulator(acc_op, neutral=neutral, seed=seed_, in_tree=accumulate_in_tree)
        stacked = tuple(self.xp.stack([seed_[k], *(blocks[k])], axis=0) for k in range(N + 1))
        _, zs = accumulator(stacked)
        if not output_starting_point:
            if q == 1:
                return tuple(a[1] for a in zs)
            zs = tuple(a[1:] for a in zs)
        return self._moveaxis(zs, source=0, destination=axis)