"""Small, backend-aware helpers for tensor elements represented as PyTrees.

The tree structure is handled entirely in Python while tracing.  Numerical
work is delegated to the supplied array namespace, so using these helpers does
not introduce device-side tree traversal or dispatch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from jax import tree_util


PyTree = Any
Leaf = TypeVar("Leaf")
Result = TypeVar("Result")
_NO_INITIALIZER = object()


def tree_map(fun: Callable[..., Result], tree: PyTree, *rest: PyTree) -> PyTree:
    """Map ``fun`` over corresponding leaves while preserving tree metadata."""
    return tree_util.tree_map(fun, tree, *rest)


def tree_leaves(tree: PyTree) -> list[Any]:
    """Return the leaves of ``tree`` in JAX's canonical traversal order."""
    return tree_util.tree_leaves(tree)


def tree_reduce(
        fun: Callable[[Result, Leaf], Result],
        tree: PyTree,
        initializer: Any = _NO_INITIALIZER,
) -> Result:
    """Reduce leaves in canonical order, optionally from ``initializer``."""
    if initializer is _NO_INITIALIZER:
        return tree_util.tree_reduce(fun, tree)
    return tree_util.tree_reduce(fun, tree, initializer=initializer)


def tree_first_leaf(tree: PyTree) -> Any:
    """Return the first array leaf, raising clearly for an empty PyTree."""
    leaves = tree_leaves(tree)
    if not leaves:
        raise ValueError("A tensor element must contain at least one array leaf.")
    return leaves[0]


def tree_stack(xp: Any, trees: Sequence[PyTree], *, axis: int) -> PyTree:
    """Stack equally structured PyTrees leaf-wise along ``axis``."""
    if not trees:
        raise ValueError("Cannot stack an empty sequence of tensor elements.")

    def stack_leaves(*leaves: Any) -> Any:
        ndim = leaves[-1].ndim
        stack_axis = axis if axis >= 0 else (ndim + 1 + axis)
        return xp.stack(leaves, axis=stack_axis)

    return tree_map(stack_leaves, trees[0], *trees[1:])


def tree_prepend(xp: Any, seed: PyTree, sequence: PyTree, *, axis: int) -> PyTree:
    """Prepend one unstacked element to an equally structured stacked tree."""

    def prepend_leaf(seed_leaf: Any, sequence_leaf: Any) -> Any:
        ndim = seed_leaf.ndim
        stack_axis = axis if axis >= 0 else (ndim + 1 + axis)
        expanded_seed = xp.expand_dims(seed_leaf, axis=stack_axis)
        return xp.concat([expanded_seed, sequence_leaf], axis=stack_axis)

    return tree_map(prepend_leaf, seed, sequence)


def tree_moveaxis(
        xp: Any,
        tree: PyTree,
        *,
        source: int,
        destination: int,
) -> PyTree:
    """Move an axis in every array leaf of ``tree``."""
    return tree_map(lambda leaf: xp.moveaxis(leaf, source, destination), tree)


def tree_index(tree: PyTree, index: Any) -> PyTree:
    """Apply the same array index to every leaf of ``tree``."""
    return tree_map(lambda leaf: leaf[index], tree)


def tree_take(
        xp: Any,
        tree: PyTree,
        indices: Any,
        *,
        axis: int,
) -> PyTree:
    """Take ``indices`` along ``axis`` in every array leaf of ``tree``."""
    return tree_map(lambda leaf: xp.take(leaf, indices, axis=axis), tree)


__all__ = [
    "PyTree",
    "tree_first_leaf",
    "tree_index",
    "tree_leaves",
    "tree_map",
    "tree_moveaxis",
    "tree_prepend",
    "tree_reduce",
    "tree_stack",
    "tree_take",
]
