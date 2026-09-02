"""Precomputed conversion data for partially symmetrized tensors."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from types import MappingProxyType
from typing import Any, Dict, Mapping

import numpy as np

from tensordev.core.bigraded.precompute import colex_placements
from tensordev.core.bigraded.symmetrized._compiled_bridge import (
    compile_symmetrization_bridge_targets,
)
from tensordev.core.bigraded.symmetrized.combinatorics import (
    multiset_placement_count,
)
from tensordev.core.bigraded.symmetrized.plans import (
    _index_itemsize,
    _validated_capacity,
)
from tensordev.core.bigraded.types import Bidegree, _bidegree
from tensordev.core.utils.segmented import (
    SegmentedRankPlan,
    apply_segmented_rank_plan,
)


@dataclass(frozen=True, slots=True, eq=False)
class SymmetrizationBridgeGradePlan:
    """One factored partial-symmetrization plan for a fixed bidegree.

    The source axis combines only ordinary placement rank and the ordered
    double-prime word.  It deliberately omits the unchanged prime dense axis.
    """

    grade: Bidegree
    rank_plan: SegmentedRankPlan
    ordered_placement_count: int
    ordered_doubleprime_width: int
    dense_prime_width: int
    quotient_rank_count: int

    @property
    def source_count(self) -> int:
        return self.rank_plan.source_count

    @property
    def target_ranks(self):
        return self.rank_plan.target_ranks

    @property
    def ordered_block_width(self) -> int:
        return self.source_count * self.dense_prime_width

    @property
    def quotient_block_width(self) -> int:
        return self.quotient_rank_count * self.dense_prime_width

    @property
    def ordered_dense_shape(self) -> tuple[int, int, int]:
        return (
            self.ordered_placement_count,
            self.dense_prime_width,
            self.ordered_doubleprime_width,
        )

    def target_rank_grid(self):
        return self.target_ranks.reshape(
            self.ordered_placement_count,
            self.ordered_doubleprime_width,
        )

    def memory_bytes(self) -> int:
        return int(self.target_ranks.nbytes)


def _validate_block_width(block, expected: int, *, name: str, grade: Bidegree):
    if block.ndim == 0 or block.shape[-1] != expected:
        width = None if block.ndim == 0 else block.shape[-1]
        raise ValueError(
            f"{name} has final width {width}, expected {expected} at "
            f"bidegree {grade}."
        )


def _ordered_bridge_source_rows(
    xp: Any,
    block,
    plan: SymmetrizationBridgeGradePlan,
):
    """Arrange an ordered block as ``batch + (bridge_source, prime)``."""

    _validate_block_width(
        block,
        plan.ordered_block_width,
        name="ordered block",
        grade=plan.grade,
    )
    batch = block.shape[:-1]
    expanded = block.reshape(batch + plan.ordered_dense_shape)
    batch_ndim = len(batch)
    source_order = xp.transpose(
        expanded,
        tuple(range(batch_ndim))
        + (batch_ndim, batch_ndim + 2, batch_ndim + 1),
    )
    return source_order.reshape(
        batch + (plan.source_count, plan.dense_prime_width)
    )


def partially_symmetrize_block(
    xp: Any,
    block,
    plan: SymmetrizationBridgeGradePlan,
    *,
    scatter_add,
):
    """Combine an ordered block into a partially symmetrized tensor block."""

    source_rows = _ordered_bridge_source_rows(xp, block, plan)
    quotient = apply_segmented_rank_plan(
        xp,
        source_rows,
        plan.rank_plan,
        scatter_add=scatter_add,
    )
    return quotient.reshape(
        block.shape[:-1] + (plan.quotient_block_width,)
    )


def _lift_partially_symmetrized_block(
    xp: Any,
    block,
    plan: SymmetrizationBridgeGradePlan,
):
    """Lift one partially symmetrized word block into ordered coordinates."""

    _validate_block_width(
        block,
        plan.quotient_block_width,
        name="partially symmetrized block",
        grade=plan.grade,
    )
    batch = block.shape[:-1]
    quotient = block.reshape(
        batch + (plan.quotient_rank_count, plan.dense_prime_width)
    )
    source_rows = xp.take(
        quotient,
        xp.asarray(plan.target_ranks),
        axis=-2,
    )
    source_order = source_rows.reshape(
        batch
        + (
            plan.ordered_placement_count,
            plan.ordered_doubleprime_width,
            plan.dense_prime_width,
        )
    )
    batch_ndim = len(batch)
    ordered = xp.transpose(
        source_order,
        tuple(range(batch_ndim))
        + (batch_ndim, batch_ndim + 2, batch_ndim + 1),
    )
    return ordered.reshape(batch + (plan.ordered_block_width,))


def pair_partially_symmetrized_with_ordered_block(
    xp: Any,
    words,
    signature,
    plan: SymmetrizationBridgeGradePlan,
):
    """Pair partially symmetrized words with an ordered signature block.

    The ordered expansion is fused with the bilinear pairing and no complex
    conjugation is applied.
    """

    _validate_block_width(
        words,
        plan.quotient_block_width,
        name="partially symmetrized word block",
        grade=plan.grade,
    )
    _validate_block_width(
        signature,
        plan.ordered_block_width,
        name="ordered signature block",
        grade=plan.grade,
    )
    batch = xp.broadcast_shapes(words.shape[:-1], signature.shape[:-1])
    words = xp.broadcast_to(
        words,
        batch + (plan.quotient_block_width,),
    ).reshape(batch + (plan.quotient_rank_count, plan.dense_prime_width))
    signature_rows = _ordered_bridge_source_rows(
        xp,
        xp.broadcast_to(
            signature,
            batch + (plan.ordered_block_width,),
        ),
        plan,
    )
    lifted_rows = xp.take(
        words,
        xp.asarray(plan.target_ranks),
        axis=-2,
    )
    return xp.sum(lifted_rows * signature_rows, axis=(-2, -1))


def _expected_bridge_memory_bytes_by_category(
    dims: Bidegree,
    max_truncation: Bidegree,
) -> Mapping[str, int]:
    normalized_dims, normalized_truncation, spec = _validated_capacity(
        dims,
        max_truncation,
    )
    del normalized_truncation
    _, d_doubleprime = normalized_dims
    target_bytes = 0
    for n, m in spec.grades:
        quotient_count = multiset_placement_count(
            d_doubleprime,
            (n, m),
        )
        source_count = comb(n + m, n) * d_doubleprime**m
        target_bytes += source_count * _index_itemsize(quotient_count - 1)
    return {"bridge_target_maps": int(target_bytes)}


class SymmetrizationBridgePlanStore:
    """Self-contained fixed-capacity bridge plans.

    Only dimensions and truncation metadata are copied at construction.  No
    ordered or partially symmetrized representation store is retained.
    """

    def __init__(
        self,
        dims: Bidegree,
        max_truncation: Bidegree,
    ) -> None:
        self.dims, self.max_truncation, spec = _validated_capacity(
            dims,
            max_truncation,
        )
        self._grade_plans: Dict[Bidegree, SymmetrizationBridgeGradePlan] = {}
        self._grade_plans_view = MappingProxyType(self._grade_plans)
        for grade in spec.grades:
            self._grade_plans[grade] = self._build_grade_plan(grade)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(dims={self.dims}, "
            f"max_truncation={self.max_truncation})"
        )

    @property
    def grade_plans(self) -> Mapping[Bidegree, SymmetrizationBridgeGradePlan]:
        return self._grade_plans_view

    def _build_grade_plan(
        self,
        grade: Bidegree,
    ) -> SymmetrizationBridgeGradePlan:
        n, m = grade
        d_prime, d_doubleprime = self.dims
        placements = colex_placements(n + m, n)
        quotient_rank_count = multiset_placement_count(d_doubleprime, grade)
        targets = compile_symmetrization_bridge_targets(
            placements,
            d_doubleprime=d_doubleprime,
            grade=grade,
        )
        if np.unique(targets).size != quotient_rank_count:
            raise AssertionError(
                f"bridge target map is not surjective at grade {grade}."
            )
        rank_plan = SegmentedRankPlan.from_targets(
            targets,
            output_rank_count=quotient_rank_count,
        )
        return SymmetrizationBridgeGradePlan(
            grade=grade,
            rank_plan=rank_plan,
            ordered_placement_count=len(placements),
            ordered_doubleprime_width=d_doubleprime**m,
            dense_prime_width=d_prime**n,
            quotient_rank_count=quotient_rank_count,
        )

    def grade_plan(self, grade: object) -> SymmetrizationBridgeGradePlan:
        normalized = _bidegree(grade)
        try:
            return self._grade_plans[normalized]
        except KeyError as exc:
            raise KeyError(
                f"bidegree {normalized} exceeds bridge-store capacity "
                f"{self.max_truncation}."
            ) from exc

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        return {
            "bridge_target_maps": int(
                sum(plan.target_ranks.nbytes for plan in self._grade_plans.values())
            )
        }

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())

    def memory_mb(self) -> float:
        return self.memory_bytes() / 1024**2

    def plan_statistics(self) -> Mapping[str, object]:
        memory = self.memory_bytes_by_category()
        return {
            "dims": self.dims,
            "max_truncation": self.max_truncation,
            "grade_count": len(self._grade_plans),
            "bytes_by_category": memory,
            "memory_bytes": sum(memory.values()),
            "memory_mb": sum(memory.values()) / 1024**2,
        }


__all__ = [
    "SymmetrizationBridgeGradePlan",
    "SymmetrizationBridgePlanStore",
    "pair_partially_symmetrized_with_ordered_block",
    "partially_symmetrize_block",
]
