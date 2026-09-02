from __future__ import annotations

import functools
import itertools
from numbers import Integral
from typing import Any, Optional, Tuple, Callable, List, TypeVar, Generic

from tensordev.core.universal import _Array, _ArrayNamespace
from tensordev.core.utils.annotations import jit as dummy_jit
from tensordev.core.utils.pytrees import (
    PyTree,
    tree_first_leaf,
    tree_index,
    tree_leaves,
    tree_map,
    tree_moveaxis,
    tree_prepend,
    tree_stack,
)

Array = TypeVar("Array", bound=_Array)
DenseElem = Tuple[Array, ...]


class SequentialCore(Generic[Array]):
    capabilities = frozenset(("map", "scan"))

    def __init__(self, xp: _ArrayNamespace, default_time_axis: int = -2):
        self.xp = xp
        namespace = getattr(xp, "__name__", "")
        self.backend = namespace.partition(".")[0] or None
        self.default_time_axis = default_time_axis

    def supports(self, capability: str) -> bool:
        """Return whether this sequential backend supplies ``capability``."""
        return capability in self.capabilities

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def _first_leaf(X: PyTree) -> Array:
        return tree_first_leaf(X)

    @staticmethod
    def _tree_map(fun: Callable, X: PyTree, *rest: PyTree) -> PyTree:
        return tree_map(fun, X, *rest)

    def _stack(self, X: List[PyTree], *, axis: int) -> PyTree:
        return tree_stack(self.xp, X, axis=axis)

    def _prepend(self, seed: PyTree, X: PyTree, *, axis: int) -> PyTree:
        """Prepend one unstacked element to a stacked PyTree sequence."""
        return tree_prepend(self.xp, seed, X, axis=axis)

    def _moveaxis(self, X: PyTree, *, source: int, destination: int) -> PyTree:
        return tree_moveaxis(self.xp, X, source=source, destination=destination)

    def _index(self, X: PyTree, index: Any) -> PyTree:
        return tree_index(X, index)

    def _mapper(self, fun: Callable[[PyTree], PyTree]) -> Callable[[PyTree], PyTree]:
        def map_fn(seq: PyTree) -> PyTree:
            size = int(self._first_leaf(seq).shape[0])
            return self._stack([fun(self._index(seq, i)) for i in range(size)], axis=0)

        return map_fn

    def _scanner(
            self,
            fun: Callable[[PyTree, PyTree], tuple[PyTree, PyTree]],
            *,
            initial: PyTree,
    ) -> Callable[[PyTree], tuple[PyTree, PyTree]]:
        """Construct an eager leading-axis scan for a heterogeneous output."""

        def scan_fn(X: PyTree) -> tuple[PyTree, PyTree]:
            size = int(self._first_leaf(X).shape[0])
            carry = initial
            outputs = []
            for index in range(size):
                carry, output = fun(carry, self._index(X, index))
                outputs.append(output)
            return carry, self._stack(outputs, axis=0)

        return scan_fn

    def _update_index_leaf(self, array: Array, index: Any, value: Array) -> Array:
        """Backend hook for a functional indexed update of one array leaf."""
        del array, index, value
        raise NotImplementedError(
            f"{type(self).__name__} does not implement functional indexed "
            "updates; its backend must override _update_index_leaf."
        )

    @staticmethod
    def _leading_size(X: PyTree, *, name: str) -> int:
        """Validate a nonempty, consistently leading-axis-stacked PyTree."""
        leaves = tree_leaves(X)
        if not leaves:
            raise ValueError(f"{name} must contain at least one array leaf.")
        sizes = tuple(int(leaf.shape[0]) for leaf in leaves)
        if any(size != sizes[0] for size in sizes[1:]):
            raise ValueError(
                f"all array leaves in {name} must have the same mapped length; "
                f"got leading sizes {sizes}."
            )
        return sizes[0]

    def _reducer(
            self,
            fun: Callable[[PyTree, PyTree], PyTree],
            *,
            neutral: PyTree,
            seed: Optional[PyTree],
            in_tree: bool = False,
    ) -> Callable[[PyTree], PyTree]:
        seed_ = neutral if seed is None else seed

        def reduce_fn(X):
            S = int(self._first_leaf(X).shape[0])
            step = lambda t: self._index(X, t)
            return functools.reduce(fun, (step(t) for t in range(S)), seed_)
        return reduce_fn

    def _accumulator(
            self,
            fun: Callable[[PyTree, PyTree], PyTree],
            *,
            neutral: PyTree,
            seed: Optional[PyTree],
            in_tree: bool = False,
    ) -> Callable[[PyTree], Tuple[PyTree, PyTree]]:
        seed_ = neutral if seed is None else seed

        def scan_fn(X):
            S = int(self._first_leaf(X).shape[0])
            step = lambda t: self._index(X, t)
            prefixes = itertools.accumulate(
                itertools.chain([seed_], (step(t) for t in range(S))),
                fun,
            )
            ys = list(itertools.islice(prefixes, 1, None))
            return ys[-1], self._stack(ys, axis=0)
        return scan_fn

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def tensor_map(
            self,
            inputs: tuple[PyTree, ...],
            *,
            map_op: Callable[..., PyTree],
            in_axes: tuple[int, ...] | int,
            out_axis: int = 0,
    ) -> PyTree:
        """Map ``map_op`` over explicitly selected axes of one or more PyTrees.

        ``inputs`` contains the positional arguments supplied to ``map_op``.
        Each input is normalized independently according to its corresponding
        entry in ``in_axes``.  The mapped output is stacked initially on its
        leading axis, which is then moved leafwise to ``out_axis``.

        This method deliberately remains an un-jitted orchestration wrapper.
        Concrete backends implement the leading-axis map through ``_mapper``.
        """
        if not isinstance(inputs, tuple) or not inputs:
            raise ValueError("tensor_map: inputs must be a nonempty tuple of PyTrees.")

        if isinstance(in_axes, Integral) and not isinstance(in_axes, bool):
            in_axes = (int(in_axes),)
        else:
            in_axes = tuple(in_axes)
        if len(in_axes) != len(inputs):
            raise ValueError(
                "tensor_map: in_axes must contain one axis per input; "
                f"got {len(in_axes)} axes for {len(inputs)} inputs."
            )
        if any(isinstance(axis, bool) or not isinstance(axis, Integral) for axis in in_axes):
            raise TypeError("tensor_map: every entry in in_axes must be an integer.")
        if isinstance(out_axis, bool) or not isinstance(out_axis, Integral):
            raise TypeError("tensor_map: out_axis must be an integer.")

        normalized = tuple(
            self._moveaxis(X, source=int(axis), destination=0)
            for X, axis in zip(inputs, in_axes)
        )
        sizes = tuple(
            self._leading_size(X, name=f"tensor_map input {index}")
            for index, X in enumerate(normalized)
        )
        if any(size != sizes[0] for size in sizes[1:]):
            raise ValueError(
                "tensor_map: all inputs must have the same mapped length; "
                f"got {sizes}."
            )
        if sizes[0] == 0:
            raise ValueError("tensor_map: mapped inputs must be nonempty.")

        mapped = self._mapper(lambda step: map_op(*step))(normalized)
        return self._moveaxis(mapped, source=0, destination=int(out_axis))

    def tensor_scan(
            self,
            X: PyTree,
            *,
            initial: PyTree,
            scan_op: Callable[[PyTree, PyTree], tuple[PyTree, PyTree]],
            axis: Optional[int] = None,
            out_axis: int = 0,
    ) -> tuple[PyTree, PyTree]:
        """Scan ``X`` while allowing carry and emitted output PyTrees to differ.

        ``scan_op`` receives ``(carry, step)`` and returns
        ``(new_carry, emitted_output)``.  ``X`` must be nonempty, and every
        input leaf must have the same length along ``axis``.  Carry and output
        structures, metadata, shapes, and dtypes must remain fixed across
        steps.  Emitted outputs are returned with their sequence axis at
        ``out_axis``.

        This method deliberately remains an un-jitted orchestration wrapper.
        Concrete backends implement the leading-axis scan through ``_scanner``.
        """
        axis = self.default_time_axis if axis is None else axis
        if isinstance(axis, bool) or not isinstance(axis, Integral):
            raise TypeError("tensor_scan: axis must be an integer or None.")
        if isinstance(out_axis, bool) or not isinstance(out_axis, Integral):
            raise TypeError("tensor_scan: out_axis must be an integer.")

        normalized = self._moveaxis(X, source=int(axis), destination=0)
        size = self._leading_size(normalized, name="tensor_scan input")
        if size == 0:
            raise ValueError("tensor_scan: X must be nonempty.")

        final, outputs = self._scanner(scan_op, initial=initial)(normalized)
        return final, self._moveaxis(outputs, source=0, destination=int(out_axis))

    def tensor_update_index(
            self,
            tree: PyTree,
            index: Any,
            values: PyTree,
    ) -> PyTree:
        """Functionally set ``tree[index]`` to ``values`` in every array leaf."""
        return self._tree_map(
            lambda array, value: self._update_index_leaf(array, index, value),
            tree,
            values,
        )

    @dummy_jit(static_argnums=0,
               static_argnames=("reduce_op", "axis", "accumulate_in_tree"),
               dynamic_batchtime=("X",))
    def tensor_reduce(
            self,
            X: PyTree,
            *,
            reduce_op: Callable[[PyTree, PyTree], PyTree],
            neutral: PyTree,
            seed: Optional[PyTree] = None,
            axis: Optional[int] = None,
            accumulate_in_tree: bool = False,
    ) -> PyTree:
        """Left-fold `reduce_op` over `X` along `axis`, starting from `seed` (defaults to `neutral`)."""
        axis = self.default_time_axis if axis is None else axis
        X = self._moveaxis(X, source=axis, destination=0)
        seed_ = seed if seed is not None else neutral
        return self._reducer(
            reduce_op, neutral=neutral, seed=seed_, in_tree=accumulate_in_tree
        )(X)

    @dummy_jit(static_argnums=0,
               static_argnames=("reduce_op", "axis", "output_starting_point", "accumulate_in_tree"),
               dynamic_batchtime=("X",))
    def tensor_accumulate(
            self,
            X: PyTree,
            *,
            reduce_op: Callable[[PyTree, PyTree], PyTree],
            neutral: PyTree,
            seed: Optional[PyTree] = None,
            axis: Optional[int] = None,
            output_starting_point: bool = False,
            accumulate_in_tree: bool = False,
    ) -> PyTree:
        """Inclusive prefix scan of `reduce_op` over `X` along `axis`."""
        axis = self.default_time_axis if axis is None else axis
        X = self._moveaxis(X, source=axis, destination=0)
        seed_ = seed if seed is not None else neutral
        size = int(self._first_leaf(X).shape[0])
        stacked = self._stack(
            [seed_] + [self._index(X, index) for index in range(size)],
            axis=0,
        )
        # The neutral-seed topology gives the preferred XLA lowering.  An
        # explicit seed is already the first element and must not be reapplied.
        accumulator_seed = seed_ if seed is None else None
        _, zs = self._accumulator(
            reduce_op,
            neutral=neutral,
            seed=accumulator_seed,
            in_tree=accumulate_in_tree,
        )(stacked)
        if not output_starting_point:
            zs = self._index(zs, slice(1, None))
        return self._moveaxis(zs, source=0, destination=axis)

    @dummy_jit(static_argnums=0,
               static_argnames=("reduce_op", "acc_op", "axis", "block_size", "accumulate",
                                "output_starting_point", "first_apply_all",
                                "reduce_in_tree", "accumulate_in_tree"),
               dynamic_batchtime=("X",))
    def tensor_abra(
            self,
            X: PyTree,
            *,
            reduce_op: Callable[[PyTree, PyTree], PyTree],
            acc_op: Callable[[PyTree, PyTree], PyTree],
            neutral: PyTree,
            axis: Optional[int] = None,
            block_size: Optional[int] = None,
            accumulate: bool = False,
            seed: Optional[PyTree] = None,
            output_starting_point: bool = False,
            first_apply_all: bool = False,
            reduce_in_tree: bool = False,
            accumulate_in_tree: bool = False,
    ) -> PyTree:
        """
        Apply, Block, Reduce and Accumulate over `X` along `axis` using caller-supplied ops.

        Parameters
        ----------
        reduce_op : (carry, step) -> carry
            The sequential left-fold operation applied within each block.

            When ``first_apply_all=False``: used directly as the scan step,
            so ``carry`` and ``step`` may have different types (e.g.
            ``tensor_fmexp`` where carry is a full tensor element and step
            contains only positive levels).  No algebraic assumptions are made.

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

        neutral : PyTree
            Identity element: ``acc_op(neutral, x) == x`` for all ``x``.

        seed : PyTree, optional
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
        seed_ = seed if seed is not None else neutral
        X = self._moveaxis(X, source=axis, destination=0)

        S = self._first_leaf(X).shape[0]
        B = S if (block_size in (None, -1)) else int(block_size)
        q, r = divmod(S, B)
        if r:
            raise ValueError(f"tensor_abra: block_size={B} must divide S={S}.")
        X_blocks = self._tree_map(
            lambda L: L.reshape(q, B, *L.shape[1:]),
            X,
        )

        if first_apply_all:
            lift = lambda step: reduce_op(neutral, step)
            lifted_reducer = self._reducer(
                acc_op, neutral=neutral, seed=neutral, in_tree=reduce_in_tree
            )
            block_reducer = self._mapper(lambda block: lifted_reducer(self._mapper(lift)(block)))
        else:
            block_reducer = self._mapper(
                self._reducer(
                    reduce_op, neutral=neutral, seed=neutral, in_tree=reduce_in_tree
                )
            )
        blocks = block_reducer(X_blocks)

        if not accumulate:
            if output_starting_point:
                blocks = self._moveaxis(blocks, source=0, destination=axis)
                return self._prepend(seed_, blocks, axis=axis)
            if q == 1:
                return self._index(blocks, 0)
            return self._moveaxis(blocks, source=0, destination=axis)

        stacked = self._stack(
            [seed_] + [self._index(blocks, index) for index in range(q)],
            axis=0,
        )
        # The neutral seed retains the preferred scan topology.  An explicit
        # seed is applied exactly once by the prepend above.
        accumulator_seed = seed_ if seed is None else None
        accumulator = self._accumulator(
            acc_op,
            neutral=neutral,
            seed=accumulator_seed,
            in_tree=accumulate_in_tree,
        )
        _, zs = accumulator(stacked)
        if not output_starting_point:
            if q == 1:
                return self._index(zs, 1)
            zs = self._index(zs, slice(1, None))
        return self._moveaxis(zs, source=0, destination=axis)
