"""Factored shuffle plans for standard ordered bidegree coordinates.

The expensive combinatorics in this module are deliberately host-side.  A
``BigradedShufflePlanStore`` is a fixed-capacity companion to a
``BigradedPlanStore`` and reuses the latter's input and output placement
tables.  Runtime kernels form the complete placement-pair outer once, then
loop over static dense-permutation keys; each key gathers and updates the
complete output-placement axis at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from types import MappingProxyType
from typing import Dict, Iterator, Mapping, Tuple

import numba as nb
import numpy as np

from tensordev.core.bigraded.precompute import (
    BigradedPlanStore,
    colex_placements,
)
from tensordev.core.bigraded.types import Bidegree, BigradedSpec, _bidegree
from tensordev.core.shuffle import _interleave_axes
from tensordev.core.utils.precompute import _readonly, _unsigned_index_dtype


# Use the shared minimal unsigned-index policy for placement pairs.
_placement_pair_index_dtype = _unsigned_index_dtype


# Below this size, executing the emitter's Python implementation is faster
# than a cold Numba compilation.  Both paths therefore use exactly the same
# rank-emission algorithm; only its execution mode changes.
_COMPILED_SHUFFLE_ENTRY_THRESHOLD = 100_000


@nb.njit(cache=True, nogil=True)
def _fill_placement_pair_indices(
    output,
    output_prime_positions,
    output_doubleprime_positions,
    prime_keys,
    prime_complement_keys,
    doubleprime_keys,
    doubleprime_complement_keys,
    binomial,
    right_placement_count,
):
    """Fill all key/output placement-pair indices for one grade pair."""
    doubleprime_key_count = doubleprime_keys.shape[0]
    for prime_key_rank in range(prime_keys.shape[0]):
        for doubleprime_key_rank in range(doubleprime_key_count):
            key_rank = (
                prime_key_rank * doubleprime_key_count
                + doubleprime_key_rank
            )
            for output_rank in range(output_prime_positions.shape[0]):
                # A prime letter's local position is its prime index plus the
                # number of selected double-prime letters preceding it. The
                # binomial sum is exactly the colex placement rank.
                left_rank = 0
                for prime_index in range(prime_keys.shape[1]):
                    position = output_prime_positions[
                        output_rank,
                        prime_keys[prime_key_rank, prime_index],
                    ]
                    local_position = prime_index
                    for doubleprime_index in range(doubleprime_keys.shape[1]):
                        if (
                            output_doubleprime_positions[
                                output_rank,
                                doubleprime_keys[
                                    doubleprime_key_rank,
                                    doubleprime_index,
                                ],
                            ]
                            < position
                        ):
                            local_position += 1
                    left_rank += binomial[local_position, prime_index + 1]

                right_rank = 0
                for prime_index in range(prime_complement_keys.shape[1]):
                    position = output_prime_positions[
                        output_rank,
                        prime_complement_keys[
                            prime_key_rank,
                            prime_index,
                        ],
                    ]
                    local_position = prime_index
                    for doubleprime_index in range(
                        doubleprime_complement_keys.shape[1]
                    ):
                        if (
                            output_doubleprime_positions[
                                output_rank,
                                doubleprime_complement_keys[
                                    doubleprime_key_rank,
                                    doubleprime_index,
                                ],
                            ]
                            < position
                        ):
                            local_position += 1
                    right_rank += binomial[local_position, prime_index + 1]

                output[key_rank, output_rank] = (
                    left_rank * right_placement_count + right_rank
                )


def _placement_key_matrix(
    placements: Tuple[Tuple[int, ...], ...],
    subset_size: int,
) -> np.ndarray:
    return np.asarray(placements, dtype=np.int32).reshape(
        len(placements), subset_size
    )


def _complement_key_matrix(
    placements: Tuple[Tuple[int, ...], ...],
    length: int,
    subset_size: int,
) -> np.ndarray:
    return np.asarray(
        [
            tuple(position for position in range(length) if position not in key)
            for key in placements
        ],
        dtype=np.int32,
    ).reshape(len(placements), length - subset_size)


def _placement_pair_indices(
    output_placements: np.ndarray,
    prime_placements: Tuple[Tuple[int, ...], ...],
    doubleprime_placements: Tuple[Tuple[int, ...], ...],
    *,
    n: int,
    m: int,
    n1: int,
    m1: int,
    right_placement_count: int,
    dtype,
    compiled: bool,
) -> np.ndarray:
    """Build one block's rank lists with the shared array emitter."""
    output_prime_positions = np.asarray(output_placements, dtype=np.int32)
    total = n + m
    is_prime = np.zeros(
        (output_prime_positions.shape[0], total),
        dtype=np.bool_,
    )
    if n:
        is_prime[
            np.arange(output_prime_positions.shape[0])[:, None],
            output_prime_positions,
        ] = True
    all_positions = np.broadcast_to(
        np.arange(total, dtype=np.int32),
        is_prime.shape,
    )
    output_doubleprime_positions = all_positions[~is_prime].reshape(
        output_prime_positions.shape[0], m
    )

    prime_keys = _placement_key_matrix(prime_placements, n1)
    prime_complement_keys = _complement_key_matrix(
        prime_placements, n, n1
    )
    doubleprime_keys = _placement_key_matrix(doubleprime_placements, m1)
    doubleprime_complement_keys = _complement_key_matrix(
        doubleprime_placements, m, m1
    )
    binomial = np.zeros((total + 1, total + 1), dtype=np.int64)
    for row in range(total + 1):
        for column in range(row + 1):
            binomial[row, column] = comb(row, column)

    rank_lists = np.empty(
        (
            len(prime_placements) * len(doubleprime_placements),
            output_prime_positions.shape[0],
        ),
        dtype=dtype,
    )
    emitter = (
        _fill_placement_pair_indices
        if compiled
        else getattr(
            _fill_placement_pair_indices,
            "py_func",
            _fill_placement_pair_indices,
        )
    )
    emitter(
        rank_lists,
        output_prime_positions,
        output_doubleprime_positions,
        prime_keys,
        prime_complement_keys,
        doubleprime_keys,
        doubleprime_complement_keys,
        binomial,
        right_placement_count,
    )
    return _readonly(rank_lists)


def _shuffle_scope(scope: object) -> str:
    if not isinstance(scope, str):
        raise TypeError(
            f"scope must be 'full' or 'generator', got {scope!r}."
        )
    if scope not in {"full", "generator"}:
        raise ValueError(
            f"scope must be 'full' or 'generator', got {scope!r}."
        )
    return scope


def _pair_in_scope(
    left_grade: Bidegree,
    right_grade: Bidegree,
    scope: str,
) -> bool:
    return scope == "full" or sum(left_grade) == 1 or sum(right_grade) == 1


def _canonical_grade_pairs(
    grades: Tuple[Bidegree, ...],
    max_truncation: Bidegree,
    scope: str,
) -> Iterator[Tuple[Bidegree, Bidegree]]:
    N, M = max_truncation
    for left_index, left_grade in enumerate(grades):
        for right_grade in grades[: left_index + 1]:
            if not _pair_in_scope(left_grade, right_grade, scope):
                continue
            output = (
                left_grade[0] + right_grade[0],
                left_grade[1] + right_grade[1],
            )
            if output[0] <= N and output[1] <= M:
                yield left_grade, right_grade


def _expected_shuffle_memory_bytes_by_category(
    dims: Bidegree,
    max_truncation: Bidegree,
    scope: str = "full",
) -> Mapping[str, int]:
    """Exact payload categories allocated by ``BigradedShufflePlanStore``."""
    scope = _shuffle_scope(scope)
    spec = BigradedSpec(*dims, max_truncation)
    grades = spec.grades
    rank_list_bytes = 0
    dense_permutation_bytes = 0
    for left, right in _canonical_grade_pairs(
        grades,
        spec.truncation,
        scope,
    ):
        output = left[0] + right[0], left[1] + right[1]
        left_placements = comb(sum(left), left[0])
        right_placements = comb(sum(right), right[0])
        output_placements = comb(sum(output), output[0])
        key_count = comb(output[0], left[0]) * comb(output[1], left[1])
        pair_index_itemsize = np.dtype(
            _unsigned_index_dtype(
                max(left_placements * right_placements - 1, 0)
            )
        ).itemsize
        rank_list_bytes += (
            key_count * output_placements * pair_index_itemsize
        )
        dense_permutation_bytes += (
            key_count * sum(output) * np.dtype(np.intp).itemsize
        )
    return {
        "rank_lists": int(rank_list_bytes),
        "dense_permutations": int(dense_permutation_bytes),
    }


def _dense_axis_permutation(
    left_grade: Bidegree,
    right_grade: Bidegree,
    prime_placement: Tuple[int, ...],
    doubleprime_placement: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Dense-axis permutation for one ``(Q_prime, Q_doubleprime)`` key."""
    n1, m1 = left_grade
    n2, m2 = right_grade

    prime_axes = _interleave_axes(
        range(n1),
        range(n1 + m1, n1 + m1 + n2),
        prime_placement,
    )
    doubleprime_axes = _interleave_axes(
        range(n1, n1 + m1),
        range(n1 + m1 + n2, n1 + m1 + n2 + m2),
        doubleprime_placement,
    )
    return prime_axes + doubleprime_axes


@dataclass(frozen=True, slots=True, eq=False)
class BigradedShuffleKeyPlan:
    """Placement-pair indices and dense permutation for one shuffle key.

    ``placement_pair_indices`` is ordered by increasing output placement rank.
    Each entry selects one row from the block's flattened
    ``(left placement, right placement)`` outer.  The output-rank column is
    therefore implicit and no scatter or segmented reduction is needed at
    runtime.
    """

    prime_key_rank: int
    doubleprime_key_rank: int
    prime_placement: Tuple[int, ...]
    doubleprime_placement: Tuple[int, ...]
    placement_pair_indices: np.ndarray
    dense_axis_permutation: Tuple[int, ...]

    @property
    def row_count(self) -> int:
        return int(self.placement_pair_indices.size)

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        return {
            # The compact list indexes placement pairs jointly.
            "rank_lists": int(self.placement_pair_indices.nbytes),
            # These are immutable Python tuples because transpose axes must be
            # static.  Count their integer payload consistently with an intp
            # plan buffer, excluding ordinary Python-object overhead just as
            # BigradedPlanStore excludes dataclass/dict overhead.
            "dense_permutations": int(
                len(self.dense_axis_permutation) * np.dtype(np.intp).itemsize
            ),
        }

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())


@dataclass(frozen=True, slots=True, eq=False)
class BigradedShuffleBlockPlan:
    """Complete factored shuffle plan for one canonical grade pair."""

    left_grade: Bidegree
    right_grade: Bidegree
    output_grade: Bidegree
    left_placement_count: int
    right_placement_count: int
    output_placement_count: int
    left_dense_shape: Tuple[int, ...]
    right_dense_shape: Tuple[int, ...]
    output_dense_shape: Tuple[int, ...]
    key_plans: Tuple[BigradedShuffleKeyPlan, ...]

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def key_count(self) -> int:
        return len(self.key_plans)

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        rank_lists = 0
        dense_permutations = 0
        for key in self.key_plans:
            categories = key.memory_bytes_by_category()
            rank_lists += categories["rank_lists"]
            dense_permutations += categories["dense_permutations"]
        return {
            "rank_lists": rank_lists,
            "dense_permutations": dense_permutations,
        }

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())

    def apply(self, xp, left, right):
        """Apply this plan using a NumPy-compatible array namespace."""
        return standard_shuffle_block(xp, left, right, self)


def _placement_pair_outer(xp, left, right, plan):
    """Form every placement-pair/dense outer exactly once for one block."""
    left_dense_degree = len(plan.left_dense_shape)
    right_dense_degree = len(plan.right_dense_shape)

    left_view = xp.reshape(
        left,
        left.shape[:-1]
        + (plan.left_placement_count,)
        + plan.left_dense_shape,
    )
    right_view = xp.reshape(
        right,
        right.shape[:-1]
        + (plan.right_placement_count,)
        + plan.right_dense_shape,
    )

    left_outer = xp.reshape(
        left_view,
        left.shape[:-1]
        + (plan.left_placement_count, 1)
        + plan.left_dense_shape
        + (1,) * right_dense_degree,
    )
    right_outer = xp.reshape(
        right_view,
        right.shape[:-1]
        + (1, plan.right_placement_count)
        + (1,) * left_dense_degree
        + plan.right_dense_shape,
    )
    values = left_outer * right_outer

    total_degree = left_dense_degree + right_dense_degree
    batch_ndim = values.ndim - total_degree - 2
    return xp.reshape(
        values,
        values.shape[:batch_ndim]
        + (plan.left_placement_count * plan.right_placement_count,)
        + plan.left_dense_shape
        + plan.right_dense_shape,
    )


def _key_update(xp, placement_pair_outer, plan, key):
    """Gather and permute one key from a shared placement-pair outer."""
    left_dense_degree = len(plan.left_dense_shape)
    right_dense_degree = len(plan.right_dense_shape)
    total_degree = left_dense_degree + right_dense_degree

    # The paired-placement axis precedes all dense axes.  One gather returns
    # every output-placement row for this key; there is no runtime row loop.
    values = xp.take(
        placement_pair_outer,
        key.placement_pair_indices,
        axis=-(total_degree + 1),
    )

    identity = tuple(range(total_degree))
    if key.dense_axis_permutation != identity:
        batch_ndim = values.ndim - total_degree - 1
        axes = (
            tuple(range(batch_ndim))
            + (batch_ndim,)
            + tuple(
                batch_ndim + 1 + axis
                for axis in key.dense_axis_permutation
            )
        )
        values = xp.transpose(values, axes)
    return values


def standard_shuffle_block(
    xp,
    left,
    right,
    plan: BigradedShuffleBlockPlan,
):
    """Apply a standard ordered-bidegree homogeneous shuffle plan.

    The function is backend-neutral and JAX-traceable.  ``plan`` is host-side
    static metadata; JAX turns its immutable NumPy paired-index columns into
    compiled constants when this function is closed over by ``jax.jit``.
    """
    expected_left = plan.left_placement_count * int(
        np.prod(plan.left_dense_shape, dtype=np.int64)
    )
    expected_right = plan.right_placement_count * int(
        np.prod(plan.right_dense_shape, dtype=np.int64)
    )
    if left.shape[-1] != expected_left:
        raise ValueError(
            f"left block for bidegree {plan.left_grade} has width "
            f"{left.shape[-1]}, expected {expected_left}."
        )
    if right.shape[-1] != expected_right:
        raise ValueError(
            f"right block for bidegree {plan.right_grade} has width "
            f"{right.shape[-1]}, expected {expected_right}."
        )

    placement_pair_outer = _placement_pair_outer(xp, left, right, plan)
    first, *remaining = plan.key_plans
    result = _key_update(xp, placement_pair_outer, plan, first)
    for key in remaining:
        result = result + _key_update(xp, placement_pair_outer, plan, key)

    batch_ndim = result.ndim - len(plan.output_dense_shape) - 1
    output_dense_width = int(
        np.prod(plan.output_dense_shape, dtype=np.int64)
    )
    return xp.reshape(
        result,
        result.shape[:batch_ndim]
        + (plan.output_placement_count * output_dense_width,),
    )


class BigradedShufflePlanStore:
    """Fixed, eager standard-shuffle plans sharing a bidegree plan store.

    Only one orientation of each grade pair is stored.  Calls in the opposite
    orientation swap their array arguments before applying that canonical
    plan, using commutativity without duplicating precomputed data.  ``scope``
    is either ``"full"`` or ``"generator"``; the latter stores only pairs with
    a first-level factor.
    """

    def __init__(
        self,
        plan_store: BigradedPlanStore,
        scope: str = "full",
    ) -> None:
        if not isinstance(plan_store, BigradedPlanStore):
            raise TypeError(
                "plan_store must be a BigradedPlanStore, got "
                f"{type(plan_store).__name__}."
            )
        self.plan_store = plan_store
        self.scope = _shuffle_scope(scope)
        self.dims = plan_store.dims
        self.max_truncation = plan_store.max_truncation
        self._block_plans: Dict[
            Tuple[Bidegree, Bidegree], BigradedShuffleBlockPlan
        ] = {}
        self._block_plans_view = MappingProxyType(self._block_plans)
        self._placement_keys: Dict[
            Tuple[int, int], Tuple[Tuple[int, ...], ...]
        ] = {}
        grades = tuple(plan_store.grade_plans)
        self._grade_order = {grade: index for index, grade in enumerate(grades)}
        grade_pairs = tuple(
            _canonical_grade_pairs(
                grades,
                self.max_truncation,
                self.scope,
            )
        )
        rank_entry_count = sum(
            comb(left_grade[0] + right_grade[0], left_grade[0])
            * comb(left_grade[1] + right_grade[1], left_grade[1])
            * comb(
                sum(left_grade) + sum(right_grade),
                left_grade[0] + right_grade[0],
            )
            for left_grade, right_grade in grade_pairs
        )
        self._use_compiled_plan_builder = (
            rank_entry_count >= _COMPILED_SHUFFLE_ENTRY_THRESHOLD
        )
        self._precompute(grade_pairs)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(dims={self.dims}, "
            f"max_truncation={self.max_truncation}, scope={self.scope!r})"
        )

    @property
    def block_plans(
        self,
    ) -> Mapping[Tuple[Bidegree, Bidegree], BigradedShuffleBlockPlan]:
        return self._block_plans_view

    def _placements(
        self,
        length: int,
        subset_size: int,
    ) -> Tuple[Tuple[int, ...], ...]:
        key = length, subset_size
        placements = self._placement_keys.get(key)
        if placements is None:
            placements = colex_placements(length, subset_size)
            self._placement_keys[key] = placements
        return placements

    def _precompute(
        self,
        grade_pairs: Tuple[Tuple[Bidegree, Bidegree], ...],
    ) -> None:
        for left_grade, right_grade in grade_pairs:
            self._block_plans[left_grade, right_grade] = (
                self._build_block_plan(left_grade, right_grade)
            )

    def _build_block_plan(
        self,
        left_grade: Bidegree,
        right_grade: Bidegree,
    ) -> BigradedShuffleBlockPlan:
        n1, m1 = left_grade
        n2, m2 = right_grade
        n, m = n1 + n2, m1 + m2
        output_grade = n, m

        left_plan = self.plan_store.grade_plan(left_grade)
        right_plan = self.plan_store.grade_plan(right_grade)
        output_plan = self.plan_store.grade_plan(output_grade)
        pair_index_dtype = _unsigned_index_dtype(
            max(
                left_plan.placement_count * right_plan.placement_count - 1,
                0,
            )
        )

        prime_placements = self._placements(n, n1)
        doubleprime_placements = self._placements(m, m1)
        rank_lists = _placement_pair_indices(
            output_plan.placements,
            prime_placements,
            doubleprime_placements,
            n=n,
            m=m,
            n1=n1,
            m1=m1,
            right_placement_count=right_plan.placement_count,
            dtype=pair_index_dtype,
            compiled=self._use_compiled_plan_builder,
        )

        key_plans = []
        key_rank = 0
        for prime_key_rank, prime_placement in enumerate(prime_placements):
            for doubleprime_key_rank, doubleprime_placement in enumerate(
                doubleprime_placements
            ):
                placement_pair_indices = rank_lists[key_rank]

                key_plans.append(
                    BigradedShuffleKeyPlan(
                        prime_key_rank=prime_key_rank,
                        doubleprime_key_rank=doubleprime_key_rank,
                        prime_placement=prime_placement,
                        doubleprime_placement=doubleprime_placement,
                        placement_pair_indices=placement_pair_indices,
                        dense_axis_permutation=_dense_axis_permutation(
                            left_grade,
                            right_grade,
                            prime_placement,
                            doubleprime_placement,
                        ),
                    )
                )
                key_rank += 1

        d_prime, d_doubleprime = self.dims
        return BigradedShuffleBlockPlan(
            left_grade=left_grade,
            right_grade=right_grade,
            output_grade=output_grade,
            left_placement_count=left_plan.placement_count,
            right_placement_count=right_plan.placement_count,
            output_placement_count=output_plan.placement_count,
            left_dense_shape=(d_prime,) * n1 + (d_doubleprime,) * m1,
            right_dense_shape=(d_prime,) * n2 + (d_doubleprime,) * m2,
            output_dense_shape=(d_prime,) * n + (d_doubleprime,) * m,
            key_plans=tuple(key_plans),
        )

    def resolve_block_plan(
        self,
        left_grade: object,
        right_grade: object,
    ) -> Tuple[BigradedShuffleBlockPlan, bool]:
        """Return ``(canonical_plan, swap_inputs)`` for an admissible pair."""
        left = _bidegree(left_grade, name="left_grade")
        right = _bidegree(right_grade, name="right_grade")
        try:
            left_order = self._grade_order[left]
            right_order = self._grade_order[right]
        except KeyError as exc:
            raise KeyError(
                f"shuffle input bidegree {exc.args[0]} exceeds plan-store "
                f"capacity {self.max_truncation}."
            ) from exc
        swap_inputs = left_order < right_order
        key = (right, left) if swap_inputs else (left, right)
        try:
            return self._block_plans[key], swap_inputs
        except KeyError as exc:
            output = left[0] + right[0], left[1] + right[1]
            if not _pair_in_scope(left, right, self.scope):
                raise KeyError(
                    f"shuffle input pair {(left, right)} is not available in "
                    f"scope={self.scope!r}."
                ) from exc
            raise KeyError(
                f"shuffle output bidegree {output} exceeds plan-store capacity "
                f"{self.max_truncation}."
            ) from exc

    def tensor_shuffle_product_homogeneous(
        self,
        xp,
        left,
        right,
        left_grade: object,
        right_grade: object,
    ):
        """Apply the standard shuffle to two homogeneous bidegree blocks."""
        plan, swap_inputs = self.resolve_block_plan(left_grade, right_grade)
        if swap_inputs:
            left, right = right, left
        return standard_shuffle_block(xp, left, right, plan)

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        rank_lists = 0
        dense_permutations = 0
        for plan in self._block_plans.values():
            categories = plan.memory_bytes_by_category()
            rank_lists += categories["rank_lists"]
            dense_permutations += categories["dense_permutations"]
        return {
            "rank_lists": rank_lists,
            "dense_permutations": dense_permutations,
        }

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())

    def memory_mb(self) -> float:
        return self.memory_bytes() / 1024**2

    def plan_statistics(self) -> Mapping[str, object]:
        categories = self.memory_bytes_by_category()
        shuffle_memory = sum(categories.values())
        base_memory = self.plan_store.memory_bytes()
        return {
            "dims": self.dims,
            "max_truncation": self.max_truncation,
            "scope": self.scope,
            "block_plan_count": len(self._block_plans),
            "key_plan_count": sum(
                plan.key_count for plan in self._block_plans.values()
            ),
            "bytes_by_category": categories,
            "memory_bytes": shuffle_memory,
            "memory_mb": shuffle_memory / 1024**2,
            "shared_plan_store_memory_bytes": base_memory,
            "total_referenced_memory_bytes": base_memory + shuffle_memory,
        }


__all__ = [
    "BigradedShuffleBlockPlan",
    "BigradedShuffleKeyPlan",
    "BigradedShufflePlanStore",
    "standard_shuffle_block",
]
