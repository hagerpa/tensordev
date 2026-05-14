"""Packed combinatorial layouts for Volterra coefficient algorithms.

The helpers in this module build graded multi-index layouts that replace tuple-
keyed combinatorics by dense array indexing. The resulting layouts are valid JAX
pytrees and can therefore be passed into jitted coefficient and evaluation
kernels, while still providing a few host-side convenience methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import comb, factorial as int_factorial
from typing import Iterator, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp


Array = jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class MultiIndexLayout:
    """
    Packed graded layout of multi-indices.

    For fixed ``n`` and ``trunc``, this object stores all multi-indices
    ``ell in N^n`` with ``|ell| <= trunc`` in graded order. Within each degree,
    the ordering is deterministic and chosen so that lower-degree blocks are
    prefixes of higher-degree layouts.

    The main purpose of the layout is to replace tuple-keyed combinatorics with
    dense integer indexing:

    - ``idx <-> ell[idx]``
    - degree blocks via ``offsets``
    - successor / predecessor lookups via ``plus`` and ``minus``

    Notes
    -----
    - ``offsets[n]`` is the start of the block with total degree ``n``.
    - ``offsets[trunc + 1]`` is the total number of packed multi-indices.
    - ``plus[idx, r]`` is the packed index of ``ell[idx] + e_r`` if that
      multi-index stays within the truncation, otherwise ``-1``.
    - ``minus[idx, r]`` is the packed index of ``ell[idx] - e_r`` when valid,
      otherwise ``-1``.
    """

    q: int = field(metadata={"static": True})
    trunc: int = field(metadata={"static": True})
    ell: Array
    degree: Array
    offsets: Array
    plus: Array
    minus: Array
    factorial: Array
    inv_factorial: Array
    backward_transition: Array

    @property
    def size(self) -> int:
        """Total number of packed multi-indices in the layout."""
        return int(self.ell.shape[0])

    def degree_slice(self, n: int) -> slice:
        """Return the slice containing exactly the degree-``n`` block.

        This is useful when an algorithm loops degree-by-degree in reverse or
        forward order and wants to work on a contiguous block of packed indices.
        """
        if n < 0 or n > self.trunc:
            raise ValueError(f"degree must satisfy 0 <= n <= {self.trunc}, got {n}.")
        return slice(int(self.offsets[n]), int(self.offsets[n + 1]))

    def prefix_size(self, n: int) -> int:
        """
        Number of multi-indices with total degree at most ``n``.

        For example, if the layout was built with ``trunc=N-1``, then
        ``prefix_size(N-2)`` is the packed size needed by arrays indexed over
        ``|ell| <= N-2``.
        """
        if n < 0:
            return 0
        if n > self.trunc:
            raise ValueError(f"degree must satisfy n <= {self.trunc}, got {n}.")
        return int(self.offsets[n + 1])

    def index_of(self, ell: Sequence[int]) -> int:
        """Return the packed index of a multi-index ``ell``.

        This helper is intended for host-side preprocessing and debugging. Jitted
        numerical kernels should instead use the precomputed navigation arrays
        ``plus`` and ``minus``.
        """
        key = tuple(int(v) for v in ell)
        try:
            return _layout_index_cache(self.q, self.trunc)[key]
        except KeyError as exc:
            raise KeyError(f"multi-index {key} is not present in this layout") from exc

    def contains(self, ell: Sequence[int]) -> bool:
        """Whether the multi-index ``ell`` is present in the layout.

        This is a host-side convenience method and is not meant to be used in
        traced numerical code.
        """
        key = tuple(int(v) for v in ell)
        return key in _layout_index_cache(self.q, self.trunc)

    def ordered_word(self, idx: int) -> Array:
        """
        Ordered representative word corresponding to ``ell[idx]``.

        This returns the canonical word
            1^{ell_1} 2^{ell_2} ... n^{ell_q}
        encoded as a 1D ``int32`` array with alphabet ``{0, ..., n-1}``.
        It is intended for preprocessing only.
        """
        counts = self.ell[int(idx)]
        total = int(self.degree[int(idx)])
        out = jnp.empty((total,), dtype=jnp.int32)
        start = 0
        for r in range(self.q):
            c = int(counts[r])
            if c > 0:
                out = out.at[start:start + c].set(r)
                start += c
        return out


@lru_cache(maxsize=None)
def _layout_index_cache(q: int, trunc: int) -> dict[Tuple[int, ...], int]:
    tuples: list[Tuple[int, ...]] = []
    for n in range(trunc + 1):
        tuples.extend(_compositions_desc(total=n, q=q))
    return {multi: i for i, multi in enumerate(tuples)}


def num_multiindices_exact(q: int, n: int) -> int:
    """Return ``# { ell in N^n : |ell| = n }``."""
    if q <= 0:
        raise ValueError(f"n must be positive, got {q}.")
    if n < 0:
        return 0
    return comb(n + q - 1, q - 1)


def num_multiindices_leq(q: int, trunc: int) -> int:
    """Return ``# { ell in N^n : |ell| <= trunc }``."""
    if q <= 0:
        raise ValueError(f"n must be positive, got {q}.")
    if trunc < 0:
        return 0
    return comb(trunc + q, q)


def build_multiindex_layout(q: int, trunc: int) -> MultiIndexLayout:
    """
    Build the packed graded multi-index layout for ``N^n`` up to degree ``trunc``.

    Parameters
    ----------
    q : int
        Number of kernel components / alphabet size.
    trunc : int
        Maximum total degree.

    Returns
    -------
    MultiIndexLayout
        Layout containing all ``ell in N^n`` with ``|ell| <= trunc``.
    """
    if q <= 0:
        raise ValueError(f"n must be positive, got {q}.")
    if trunc < 0:
        raise ValueError(f"trunc must be non-negative, got {trunc}.")

    tuples: list[Tuple[int, ...]] = []
    degree_blocks: list[int] = [0]
    for n in range(trunc + 1):
        tuples.extend(_compositions_desc(total=n, q=q))
        degree_blocks.append(len(tuples))

    size = len(tuples)
    ell_np = jnp.asarray(tuples, dtype=jnp.int32)
    degree = jnp.sum(ell_np, axis=1, dtype=jnp.int32)
    offsets = jnp.asarray(degree_blocks, dtype=jnp.int32)

    index = _layout_index_cache(q, trunc)

    plus_np = [[-1] * q for _ in range(size)]
    minus_np = [[-1] * q for _ in range(size)]
    for i, multi in enumerate(tuples):
        vals = list(multi)
        total = sum(vals)
        for r in range(q):
            if total < trunc:
                vals[r] += 1
                plus_np[i][r] = index[tuple(vals)]
                vals[r] -= 1
            if vals[r] > 0:
                vals[r] -= 1
                minus_np[i][r] = index[tuple(vals)]
                vals[r] += 1

    factorial_np = []
    for multi in tuples:
        value = 1
        for a in multi:
            value *= int_factorial(int(a))
        factorial_np.append(float(value))

    factorial = jnp.asarray(factorial_np)
    inv_factorial = 1.0 / factorial

    backward_transition_np = []
    for multi in tuples:
        total = sum(multi)
        row = []
        for r in range(q):
            if total < trunc:
                row.append((int(multi[r]) + 1.0) / (total + 1.0))
            else:
                row.append(0.0)
        backward_transition_np.append(row)

    return MultiIndexLayout(
        q=q,
        trunc=trunc,
        ell=ell_np,
        degree=degree,
        offsets=offsets,
        plus=jnp.asarray(plus_np, dtype=jnp.int32),
        minus=jnp.asarray(minus_np, dtype=jnp.int32),
        factorial=factorial,
        inv_factorial=inv_factorial,
        backward_transition=jnp.asarray(backward_transition_np),
    )


def _compositions_desc(total: int, q: int) -> Iterator[Tuple[int, ...]]:
    """
    Enumerate compositions of ``total`` into ``n`` non-negative parts.

    The order is descending in the earliest coordinates, e.g. for ``n=3`` and
    ``total=2``:
        (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)
    """
    if q == 1:
        yield (total,)
        return

    prefix = [0] * q

    def rec(pos: int, remaining: int) -> Iterator[Tuple[int, ...]]:
        if pos == q - 1:
            prefix[pos] = remaining
            yield tuple(prefix)
            return
        for first in range(remaining, -1, -1):
            prefix[pos] = first
            yield from rec(pos + 1, remaining - first)

    yield from rec(0, total)


@lru_cache(maxsize=None)
def multiindex_batched_navigation(
    q: int,
    trunc: int,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[tuple[np.ndarray, ...] | None, ...],
]:
    """Host-side precomputation for the batched (stacked multi-index) recursion.

    Returns ``(global_idx_by_deg, succ_local_by_n_r)`` where

    - ``global_idx_by_deg[n]`` is a NumPy integer array of the global packed
      indices for all multi-indices of total degree ``n``.
    - ``succ_local_by_n_r[n]`` is ``None`` when ``n == trunc``; otherwise a
      tuple of ``n`` NumPy integer arrays, where entry ``r`` gives the *local*
      index (within the degree-``(n+1)`` block) of ``plus[idx][r]`` for each
      ``idx`` in ``global_idx_by_deg[n]``.

    NumPy arrays are used intentionally: they are host-side objects that JAX
    never traces, so they can safely be stored in this ``lru_cache`` and used
    as concrete static indices inside ``jax.jit`` without causing tracer leaks.
    """
    tuples: list[Tuple[int, ...]] = []
    offsets: list[int] = [0]
    for n in range(trunc + 1):
        tuples.extend(_compositions_desc(total=n, q=q))
        offsets.append(len(tuples))

    index = {multi: i for i, multi in enumerate(tuples)}

    raw_global = [list(range(offsets[n], offsets[n + 1])) for n in range(trunc + 1)]

    succ_local_list: list[tuple[np.ndarray, ...] | None] = []
    for n in range(trunc + 1):
        if n == trunc:
            succ_local_list.append(None)
            continue
        block = raw_global[n]
        start_next = offsets[n + 1]
        succ_r: list[list[int]] = [[] for _ in range(q)]
        for idx in block:
            vals = list(tuples[idx])
            for r in range(q):
                vals[r] += 1
                succ_r[r].append(index[tuple(vals)] - start_next)
                vals[r] -= 1
        succ_local_list.append(tuple(np.array(sr, dtype=np.intp) for sr in succ_r))

    global_idx_by_deg = tuple(np.array(blk, dtype=np.intp) for blk in raw_global)
    return global_idx_by_deg, tuple(succ_local_list)


__all__ = [
    "MultiIndexLayout",
    "build_multiindex_layout",
    "multiindex_batched_navigation",
    "num_multiindices_exact",
    "num_multiindices_leq",
]
