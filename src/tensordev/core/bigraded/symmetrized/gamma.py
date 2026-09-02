"""Native shear-shuffle plans for partially symmetrized bidegrees."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from tensordev.core.bigraded.shuffle import _interleave_axes, _placement_pair_outer
from tensordev.core.bigraded.symmetrized._compiled_gamma import (
    PartiallySymmetrizedGammaWorkspace,
    compile_partially_symmetrized_shear_shuffle_support,
)
from tensordev.core.bigraded.symmetrized._precompute import (
    _coefficient_dtype,
)
from tensordev.core.bigraded.symmetrized.combinatorics import (
    multiset_placement_count,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
    _validated_capacity,
)
from tensordev.core.bigraded.types import Bidegree, _bidegree
from tensordev.core.utils.precompute import _unsigned_index_dtype
from tensordev.core.utils.segmented import (
    SegmentedRankPlan,
    apply_segmented_rank_plan,
)


ShuffleScope = Literal["none", "generator", "full"]

_MEMORY_CATEGORIES = (
    "shuffle_target_maps",
    "shuffle_segment_metadata",
    "shuffle_coefficients",
    "shuffle_dense_permutations",
)

_COMPILED_GAMMA_ENTRY_THRESHOLD = 100_000


def _gamma_scope(value: object) -> ShuffleScope:
    if isinstance(value, bool):
        return "full" if value else "none"
    if isinstance(value, str) and value in {"none", "generator", "full"}:
        return value
    raise TypeError(
        "scope must be False, 'none', 'generator', True, or 'full', "
        f"got {value!r}."
    )


def _pair_in_scope(
    left_grade: Bidegree,
    right_grade: Bidegree,
    scope: ShuffleScope,
) -> bool:
    return scope != "none" and (
        scope == "full"
        or sum(left_grade) == 1
        or sum(right_grade) == 1
    )


def _expected_gamma_memory_bytes_by_category(
    dims: Bidegree,
    max_truncation: Bidegree,
    *,
    scope: bool | ShuffleScope = "full",
) -> Mapping[str, int]:
    """Count retained partially symmetrized shuffle data."""
    normalized_dims, _, spec = _validated_capacity(dims, max_truncation)
    d_doubleprime = normalized_dims[1]
    normalized_scope = _gamma_scope(scope)
    grades = tuple(spec.grades)
    rank_counts = {
        grade: multiset_placement_count(d_doubleprime, grade)
        for grade in grades
    }
    memory = {name: 0 for name in _MEMORY_CATEGORIES}
    for left_index, left_grade in enumerate(grades):
        for right_grade in grades[: left_index + 1]:
            output_grade = (
                left_grade[0] + right_grade[0],
                left_grade[1] + right_grade[1],
            )
            if (
                not _pair_in_scope(left_grade, right_grade, normalized_scope)
                or not spec.contains(output_grade)
            ):
                continue
            pair_count = rank_counts[left_grade] * rank_counts[right_grade]
            output_rank_count = rank_counts[output_grade]
            interleaving_count = comb(
                output_grade[0], left_grade[0]
            )
            target_itemsize = np.dtype(
                _unsigned_index_dtype(max(output_rank_count - 1, 0))
            ).itemsize
            maximum_coefficient = comb(
                output_grade[1], left_grade[1]
            )
            coefficient_itemsize = np.dtype(
                _coefficient_dtype(maximum_coefficient)
            ).itemsize
            memory["shuffle_target_maps"] += (
                interleaving_count * pair_count * target_itemsize
            )
            memory["shuffle_coefficients"] += (
                pair_count * coefficient_itemsize
            )
            memory["shuffle_dense_permutations"] += (
                interleaving_count
                * output_grade[0]
                * np.dtype(np.intp).itemsize
            )
    return {name: int(memory[name]) for name in _MEMORY_CATEGORIES}


@dataclass(frozen=True, slots=True, eq=False)
class PartiallySymmetrizedShearShuffleKeyPlan:
    """One prime-skeleton interleaving and its many-to-one rank map."""

    dense_axis_permutation: tuple[int, ...]
    rank_plan: SegmentedRankPlan

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        return {
            "target_maps": int(self.rank_plan.target_ranks.nbytes),
            # Direct scatter-add requires no stored segment metadata.
            "segment_metadata": 0,
            "dense_permutations": int(
                len(self.dense_axis_permutation)
                * np.dtype(np.intp).itemsize
            ),
        }

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())


@dataclass(frozen=True, slots=True, eq=False)
class PartiallySymmetrizedShearShuffleBlockPlan:
    """One partially symmetrized shear-shuffle grade-pair plan."""

    left_grade: Bidegree
    right_grade: Bidegree
    output_grade: Bidegree
    left_rank_count: int
    right_rank_count: int
    output_rank_count: int
    left_dense_shape: tuple[int, ...]
    right_dense_shape: tuple[int, ...]
    output_dense_shape: tuple[int, ...]
    coefficients: np.ndarray
    key_plans: tuple[PartiallySymmetrizedShearShuffleKeyPlan, ...]

    @property
    def pair_count(self) -> int:
        return self.left_rank_count * self.right_rank_count

    # Structural aliases expose the placement-count interface required by the
    # shared paired-rank outer helper.
    @property
    def left_placement_count(self) -> int:
        return self.left_rank_count

    @property
    def right_placement_count(self) -> int:
        return self.right_rank_count

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        target_maps = 0
        segment_metadata = 0
        dense_permutations = 0
        for key in self.key_plans:
            categories = key.memory_bytes_by_category()
            target_maps += categories["target_maps"]
            segment_metadata += categories["segment_metadata"]
            dense_permutations += categories["dense_permutations"]
        return {
            "target_maps": int(target_maps),
            "segment_metadata": int(segment_metadata),
            "coefficients": int(self.coefficients.nbytes),
            "dense_permutations": int(dense_permutations),
        }

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())


def apply_partially_symmetrized_shear_shuffle_block(
    xp: Any,
    left,
    right,
    plan: PartiallySymmetrizedShearShuffleBlockPlan,
    *,
    scatter_add,
):
    """Apply one partially symmetrized shear-shuffle block plan."""

    left = xp.asarray(left)
    right = xp.asarray(right)
    left_width = plan.left_rank_count * int(
        np.prod(plan.left_dense_shape, dtype=np.int64)
    )
    right_width = plan.right_rank_count * int(
        np.prod(plan.right_dense_shape, dtype=np.int64)
    )
    if left.ndim == 0 or left.shape[-1] != left_width:
        raise ValueError(
            f"left block has final width "
            f"{None if left.ndim == 0 else left.shape[-1]}, expected "
            f"{left_width} for grade {plan.left_grade}."
        )
    if right.ndim == 0 or right.shape[-1] != right_width:
        raise ValueError(
            f"right block has final width "
            f"{None if right.ndim == 0 else right.shape[-1]}, expected "
            f"{right_width} for grade {plan.right_grade}."
        )

    pair_outer = _placement_pair_outer(xp, left, right, plan)
    output_degree = len(plan.output_dense_shape)
    dense_width = int(np.prod(plan.output_dense_shape, dtype=np.int64))
    batch_shape = pair_outer.shape[: -(output_degree + 1)]
    coefficients = xp.asarray(plan.coefficients).reshape(
        (1,) * len(batch_shape) + (plan.pair_count, 1)
    )
    output = None
    identity = tuple(range(output_degree))
    for key in plan.key_plans:
        values = pair_outer
        if key.dense_axis_permutation != identity:
            rank_axis = len(batch_shape)
            values = xp.transpose(
                values,
                tuple(range(rank_axis + 1))
                + tuple(
                    rank_axis + 1 + axis
                    for axis in key.dense_axis_permutation
                ),
            )
        values = xp.reshape(
            values,
            batch_shape + (plan.pair_count, dense_width),
        )
        values = values * coefficients
        contribution = apply_segmented_rank_plan(
            xp,
            values,
            key.rank_plan,
            scatter_add=scatter_add,
        )
        output = contribution if output is None else output + contribution
    if output is None:
        raise AssertionError("a shuffle block plan must contain an interleaving")
    return xp.reshape(
        output,
        batch_shape + (plan.output_rank_count * dense_width,),
    )


class PartiallySymmetrizedShearShufflePlanStore:
    """Optional partially symmetrized shear-shuffle plans."""

    def __init__(
        self,
        plan_store: PartiallySymmetrizedPlanStore,
        scope: bool | ShuffleScope = "full",
    ) -> None:
        if not isinstance(plan_store, PartiallySymmetrizedPlanStore):
            raise TypeError(
                "plan_store must be a PartiallySymmetrizedPlanStore."
            )
        self.plan_store = plan_store
        self.scope = _gamma_scope(scope)
        self.dims = plan_store.dims
        self.max_truncation = plan_store.max_truncation
        grades = tuple(plan_store.grade_plans)
        self._grade_order = {grade: index for index, grade in enumerate(grades)}
        grade_pairs = []
        target_entry_count = 0
        for left_index, left_grade in enumerate(grades):
            for right_grade in grades[: left_index + 1]:
                output_grade = (
                    left_grade[0] + right_grade[0],
                    left_grade[1] + right_grade[1],
                )
                if (
                    _pair_in_scope(left_grade, right_grade, self.scope)
                    and output_grade in plan_store.grade_plans
                ):
                    grade_pairs.append((left_grade, right_grade))
                    target_entry_count += (
                        comb(output_grade[0], left_grade[0])
                        * plan_store.grade_plan(left_grade).rank_count
                        * plan_store.grade_plan(right_grade).rank_count
                    )
        self._use_compiled_plan_builder = (
            target_entry_count >= _COMPILED_GAMMA_ENTRY_THRESHOLD
        )
        workspace = PartiallySymmetrizedGammaWorkspace()
        dense_permutation_cache = {}
        block_plans = {}
        for left_grade, right_grade in grade_pairs:
            prime_key = left_grade[0], right_grade[0]
            dense_permutations = dense_permutation_cache.get(prime_key)
            if dense_permutations is None:
                output_prime_degree = sum(prime_key)
                left_positions, _ = workspace.prime_tables(
                    output_prime_degree,
                    left_grade[0],
                )
                left_prime_axes = tuple(range(left_grade[0]))
                right_prime_axes = tuple(
                    range(left_grade[0], output_prime_degree)
                )
                dense_permutations = tuple(
                    _interleave_axes(
                        left_prime_axes,
                        right_prime_axes,
                        positions,
                    )
                    for positions in left_positions
                )
                dense_permutation_cache[prime_key] = dense_permutations
            block_plans[left_grade, right_grade] = self._build(
                left_grade,
                right_grade,
                workspace=workspace,
                dense_permutations=dense_permutations,
            )
        self.block_plans = MappingProxyType(block_plans)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(dims={self.dims}, "
            f"max_truncation={self.max_truncation}, scope={self.scope!r})"
        )

    def _build(
        self,
        left_grade: Bidegree,
        right_grade: Bidegree,
        *,
        workspace: PartiallySymmetrizedGammaWorkspace,
        dense_permutations: tuple[tuple[int, ...], ...],
    ) -> PartiallySymmetrizedShearShuffleBlockPlan:
        n_left, _ = left_grade
        n_right, _ = right_grade
        output_grade = (
            left_grade[0] + right_grade[0],
            left_grade[1] + right_grade[1],
        )
        left_meta = self.plan_store.grade_plan(left_grade)
        right_meta = self.plan_store.grade_plan(right_grade)
        output_meta = self.plan_store.grade_plan(output_grade)
        support = compile_partially_symmetrized_shear_shuffle_support(
            left_meta.placements,
            right_meta.placements,
            left_grade,
            right_grade,
            compiled=self._use_compiled_plan_builder,
            _workspace=workspace,
        )
        output_prime_degree = n_left + n_right
        if len(dense_permutations) != support.key_count:
            raise AssertionError("shuffle dense-permutation count mismatch")
        keys = tuple(
            PartiallySymmetrizedShearShuffleKeyPlan(
                dense_axis_permutation=dense_permutations[key],
                rank_plan=SegmentedRankPlan.from_targets(
                    support.target_ranks[key],
                    output_rank_count=output_meta.rank_count,
                ),
            )
            for key in range(support.key_count)
        )

        d_prime = self.dims[0]
        return PartiallySymmetrizedShearShuffleBlockPlan(
            left_grade=left_grade,
            right_grade=right_grade,
            output_grade=output_grade,
            left_rank_count=left_meta.rank_count,
            right_rank_count=right_meta.rank_count,
            output_rank_count=output_meta.rank_count,
            left_dense_shape=(d_prime,) * n_left,
            right_dense_shape=(d_prime,) * n_right,
            output_dense_shape=(d_prime,) * output_prime_degree,
            coefficients=support.coefficients,
            key_plans=keys,
        )

    def resolve_block_plan(
        self,
        left_grade: object,
        right_grade: object,
    ) -> tuple[PartiallySymmetrizedShearShuffleBlockPlan, bool]:
        left = _bidegree(left_grade, name="left_grade")
        right = _bidegree(right_grade, name="right_grade")
        if left not in self._grade_order or right not in self._grade_order:
            raise KeyError(
                f"shuffle input grade pair {(left, right)} exceeds "
                f"capacity {self.max_truncation}."
            )
        swap = self._grade_order[left] < self._grade_order[right]
        key = (right, left) if swap else (left, right)
        try:
            return self.block_plans[key], swap
        except KeyError as exc:
            if not _pair_in_scope(left, right, self.scope):
                raise KeyError(
                    f"shuffle pair {(left, right)} is unavailable in "
                    f"scope={self.scope!r}."
                ) from exc
            raise KeyError(
                f"shuffle output exceeds capacity "
                f"{self.max_truncation}."
            ) from exc

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        memory = {name: 0 for name in _MEMORY_CATEGORIES}
        for plan in self.block_plans.values():
            categories = plan.memory_bytes_by_category()
            memory["shuffle_target_maps"] += categories["target_maps"]
            memory["shuffle_segment_metadata"] += categories[
                "segment_metadata"
            ]
            memory["shuffle_coefficients"] += categories["coefficients"]
            memory["shuffle_dense_permutations"] += categories[
                "dense_permutations"
            ]
        return {name: int(memory[name]) for name in _MEMORY_CATEGORIES}

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())

    def memory_mb(self) -> float:
        return self.memory_bytes() / 1024**2

    def plan_statistics(self) -> Mapping[str, object]:
        memory = self.memory_bytes_by_category()
        key_plans = tuple(
            key
            for plan in self.block_plans.values()
            for key in plan.key_plans
        )
        collision_count = sum(
            key.rank_plan.source_count
            - np.unique(key.rank_plan.target_ranks).size
            for key in key_plans
        )
        return {
            "dims": self.dims,
            "max_truncation": self.max_truncation,
            "scope": self.scope,
            "block_plan_count": len(self.block_plans),
            "key_plan_count": len(key_plans),
            "pair_term_count": sum(
                key.rank_plan.source_count for key in key_plans
            ),
            "collision_count": int(collision_count),
            "strategy_counts": {"segmented_rank": len(key_plans)},
            "authoritative_memory_bytes": sum(memory.values()),
            "derived_execution_memory_bytes": 0,
            "bytes_by_category": memory,
            "memory_bytes": sum(memory.values()),
            "memory_mb": sum(memory.values()) / 1024**2,
        }


__all__ = [
    "PartiallySymmetrizedShearShuffleBlockPlan",
    "PartiallySymmetrizedShearShuffleKeyPlan",
    "PartiallySymmetrizedShearShufflePlanStore",
    "ShuffleScope",
    "apply_partially_symmetrized_shear_shuffle_block",
]
