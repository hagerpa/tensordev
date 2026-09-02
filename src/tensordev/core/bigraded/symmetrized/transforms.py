"""Sparse shear-coordinate transforms for partially symmetrized bidegrees.

The host plans precompute conversion in both directions.  They act only on
the partially symmetrized rank axis; the ordered prime-letter axis remains a
vectorized trailing axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from tensordev.core.bigraded.symmetrized._precompute import (
    _coefficient_dtype,
)
from tensordev.core.bigraded.symmetrized._compiled import (
    PartiallySymmetrizedTransformSupport,
    PartiallySymmetrizedTransformWorkspace,
    _transform_maximum_coefficient,
    _transform_term_count,
)
from tensordev.core.bigraded.symmetrized.combinatorics import (
    multiset_placement_count,
)
from tensordev.core.bigraded.symmetrized.plans import (
    PartiallySymmetrizedPlanStore,
    _validated_capacity,
)
from tensordev.core.bigraded.types import Bidegree, _bidegree
from tensordev.core.utils.precompute import _readonly, _unsigned_index_dtype


TransformOrientation = Literal[
    "forward",
    "inverse",
    "forward_transpose",
    "inverse_transpose",
]

_MEMORY_CATEGORIES = (
    "transform_rank_pairs",
    "transform_coefficients",
    "transform_parities",
    "transform_rank_matrices",
)

_COMPILED_TRANSFORM_ENTRY_THRESHOLD = 100_000


def _expected_transform_memory_bytes_by_category(
    dims: Bidegree,
    max_truncation: Bidegree,
    *,
    dense_rank_threshold: int = 64,
    dense_relative_bytes: float = 2.0,
) -> Mapping[str, int]:
    """Count exact retained transform buffers without constructing plans."""
    normalized_dims, _, spec = _validated_capacity(dims, max_truncation)
    if (
        isinstance(dense_rank_threshold, bool)
        or not isinstance(dense_rank_threshold, int)
        or dense_rank_threshold < 0
    ):
        raise ValueError("dense_rank_threshold must be non-negative.")
    if dense_relative_bytes < 0:
        raise ValueError("dense_relative_bytes must be non-negative.")

    d_prime, d_doubleprime = normalized_dims
    del d_prime
    memory = {name: 0 for name in _MEMORY_CATEGORIES}
    for grade in spec.grades:
        n, m = grade
        rank_count = multiset_placement_count(d_doubleprime, grade)
        memory["transform_parities"] += rank_count * np.dtype(np.int8).itemsize
        for inverse in (False, True):
            term_count = _transform_term_count(
                n,
                m,
                d_doubleprime,
                inverse=inverse,
            )
            maximum_coefficient = _transform_maximum_coefficient(
                n,
                m,
                d_doubleprime,
                inverse=inverse,
            )
            pair_itemsize = np.dtype(
                _unsigned_index_dtype(max(rank_count**2 - 1, 0))
            ).itemsize
            coefficient_itemsize = np.dtype(
                _coefficient_dtype(maximum_coefficient)
            ).itemsize
            sparse_bytes = term_count * (pair_itemsize + coefficient_itemsize)
            matrix_bytes = rank_count**2 * coefficient_itemsize
            memory["transform_rank_pairs"] += term_count * pair_itemsize
            memory["transform_coefficients"] += term_count * coefficient_itemsize
            if (
                rank_count <= dense_rank_threshold
                and matrix_bytes <= dense_relative_bytes * sparse_bytes
            ):
                memory["transform_rank_matrices"] += matrix_bytes
    return {name: int(memory[name]) for name in _MEMORY_CATEGORIES}


@dataclass(frozen=True, slots=True, eq=False)
class PartiallySymmetrizedTransformPlan:
    """One stored partial-symmetry shear-coordinate transform."""

    grade: Bidegree
    rank_count: int
    dense_prime_width: int
    inverse: bool
    encoded_rank_pairs: np.ndarray
    coefficients: np.ndarray
    parities: np.ndarray
    rank_matrix: np.ndarray | None

    @property
    def strategy(self) -> str:
        return "dense" if self.rank_matrix is not None else "sparse"

    @property
    def term_count(self) -> int:
        return int(self.encoded_rank_pairs.size)

    def rank_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.encoded_rank_pairs // self.rank_count,
            self.encoded_rank_pairs % self.rank_count,
        )

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        return {
            "rank_pairs": int(self.encoded_rank_pairs.nbytes),
            "coefficients": int(self.coefficients.nbytes),
            "parities": int(self.parities.nbytes),
            "rank_matrix": int(
                0 if self.rank_matrix is None else self.rank_matrix.nbytes
            ),
        }

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())


def apply_partially_symmetrized_transform_plan(
    xp: Any,
    block,
    plan: PartiallySymmetrizedTransformPlan,
    *,
    transpose: bool = False,
    scatter_add,
):
    """Apply a shear-coordinate transform to a partially symmetrized block.

    ``scatter_add`` has the common signature ``(output, targets, values)``
    and updates the penultimate rank axis.  A dense plan does not call it.
    """

    expected_width = plan.rank_count * plan.dense_prime_width
    if block.ndim == 0 or block.shape[-1] != expected_width:
        raise ValueError(
            f"block has final width "
            f"{None if block.ndim == 0 else block.shape[-1]}, expected "
            f"{expected_width} for grade {plan.grade}."
        )
    source = xp.reshape(
        block,
        block.shape[:-1] + (plan.rank_count, plan.dense_prime_width),
    )
    parity = xp.asarray(plan.parities)
    if plan.inverse:
        source = source * parity.reshape(
            (1,) * (source.ndim - 2) + (plan.rank_count, 1)
        )

    if plan.rank_matrix is not None:
        matrix = xp.asarray(plan.rank_matrix)
        if transpose:
            matrix = xp.swapaxes(matrix, -1, -2)
        output = xp.einsum("oi,...id->...od", matrix, source)
    else:
        output_ranks, input_ranks = plan.rank_pairs()
        if transpose:
            output_ranks, input_ranks = input_ranks, output_ranks
        values = xp.take(source, xp.asarray(input_ranks), axis=-2)
        coefficients = xp.asarray(plan.coefficients).reshape(
            (1,) * (values.ndim - 2) + (plan.term_count, 1)
        )
        values = values * coefficients
        output = xp.zeros(
            source.shape[:-2] + (plan.rank_count, plan.dense_prime_width),
            dtype=values.dtype,
        )
        output = scatter_add(output, xp.asarray(output_ranks), values)

    if plan.inverse:
        output = output * parity.reshape(
            (1,) * (output.ndim - 2) + (plan.rank_count, 1)
        )
    return xp.reshape(output, block.shape[:-1] + (expected_width,))


def apply_partially_symmetrized_transform(
    xp: Any,
    block,
    store: "PartiallySymmetrizedShearPlanStore",
    grade: object,
    *,
    orientation: TransformOrientation,
    scatter_add,
):
    """Apply any of the four static transform orientations."""

    if orientation not in {
        "forward",
        "inverse",
        "forward_transpose",
        "inverse_transpose",
    }:
        raise ValueError(f"unknown transform orientation {orientation!r}.")
    inverse = orientation.startswith("inverse")
    transpose = orientation.endswith("transpose")
    return apply_partially_symmetrized_transform_plan(
        xp,
        block,
        store.transform_plan(grade, inverse=inverse),
        transpose=transpose,
        scatter_add=scatter_add,
    )


class PartiallySymmetrizedShearPlanStore:
    """Shear-coordinate transforms sharing one representation store."""

    def __init__(
        self,
        plan_store: PartiallySymmetrizedPlanStore,
        *,
        dense_rank_threshold: int = 64,
        dense_relative_bytes: float = 2.0,
    ) -> None:
        if not isinstance(plan_store, PartiallySymmetrizedPlanStore):
            raise TypeError("plan_store must be a PartiallySymmetrizedPlanStore.")
        if (
            isinstance(dense_rank_threshold, bool)
            or not isinstance(dense_rank_threshold, int)
            or dense_rank_threshold < 0
        ):
            raise ValueError("dense_rank_threshold must be non-negative.")
        if dense_relative_bytes < 0:
            raise ValueError("dense_relative_bytes must be non-negative.")
        self.plan_store = plan_store
        self.dims = plan_store.dims
        self.max_truncation = plan_store.max_truncation
        self.dense_rank_threshold = dense_rank_threshold
        self.dense_relative_bytes = float(dense_relative_bytes)

        candidate_term_count = sum(
            _transform_term_count(
                grade[0],
                grade[1],
                self.dims[1],
                inverse=inverse,
            )
            for grade in plan_store.grade_plans
            for inverse in (False, True)
        )
        self._use_compiled_plan_builder = (
            candidate_term_count >= _COMPILED_TRANSFORM_ENTRY_THRESHOLD
        )
        grades = tuple(plan_store.grade_plans)
        maximum_rank_count = max(
            meta.rank_count for meta in plan_store.grade_plans.values()
        )
        working_index_dtype = _unsigned_index_dtype(max(maximum_rank_count**2 - 1, 0))
        maximum_coefficient = max(
            _transform_maximum_coefficient(
                grade[0],
                grade[1],
                self.dims[1],
                inverse=inverse,
            )
            for grade in grades
            for inverse in (False, True)
        )
        working_coefficient_dtype = _coefficient_dtype(maximum_coefficient)
        forward = dict.fromkeys(grades)
        inverse = dict.fromkeys(grades)
        for prime_count in range(self.max_truncation[0] + 1):
            workspace = PartiallySymmetrizedTransformWorkspace(
                prime_count,
                self.max_truncation[1],
                self.dims[1],
                compiled=self._use_compiled_plan_builder,
                working_index_dtype=working_index_dtype,
                working_coefficient_dtype=working_coefficient_dtype,
            )
            for grade in grades:
                if grade[0] != prime_count:
                    continue
                meta = plan_store.grade_plan(grade)
                forward_support, inverse_support, parities = workspace.compile_pair(
                    meta.placements, grade
                )
                forward[grade] = self._make_plan(
                    grade,
                    forward_support,
                    parities,
                )
                inverse[grade] = self._make_plan(
                    grade,
                    inverse_support,
                    parities,
                )
            del workspace
        self.forward_plans = MappingProxyType(forward)
        self.inverse_plans = MappingProxyType(inverse)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(dims={self.dims}, "
            f"max_truncation={self.max_truncation})"
        )

    def _make_plan(
        self,
        grade: Bidegree,
        support: PartiallySymmetrizedTransformSupport,
        parities: np.ndarray,
    ) -> PartiallySymmetrizedTransformPlan:
        meta = self.plan_store.grade_plan(grade)
        if support.grade != grade or support.rank_count != meta.rank_count:
            raise AssertionError("quotient transform support metadata mismatch")
        encoded = support.encoded_rank_pairs
        coefficients = support.coefficients
        matrix_bytes = meta.rank_count**2 * coefficients.dtype.itemsize
        sparse_bytes = encoded.nbytes + coefficients.nbytes
        rank_matrix = None
        if (
            meta.rank_count <= self.dense_rank_threshold
            and matrix_bytes <= self.dense_relative_bytes * sparse_bytes
        ):
            matrix = np.zeros(
                (meta.rank_count, meta.rank_count),
                dtype=coefficients.dtype,
            )
            output_ranks = encoded // meta.rank_count
            input_ranks = encoded % meta.rank_count
            matrix[output_ranks, input_ranks] = coefficients
            rank_matrix = _readonly(matrix)
        return PartiallySymmetrizedTransformPlan(
            grade=grade,
            rank_count=meta.rank_count,
            dense_prime_width=self.dims[0] ** grade[0],
            inverse=support.inverse,
            encoded_rank_pairs=encoded,
            coefficients=coefficients,
            parities=parities,
            rank_matrix=rank_matrix,
        )

    def transform_plan(
        self,
        grade: object,
        *,
        inverse: bool = False,
    ) -> PartiallySymmetrizedTransformPlan:
        normalized = _bidegree(grade)
        plans = self.inverse_plans if inverse else self.forward_plans
        try:
            return plans[normalized]
        except KeyError as exc:
            raise KeyError(
                f"bidegree {normalized} exceeds transform-store capacity "
                f"{self.max_truncation}."
            ) from exc

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        plans = tuple(self.forward_plans.values()) + tuple(self.inverse_plans.values())
        memory = {name: 0 for name in _MEMORY_CATEGORIES}
        memory["transform_rank_pairs"] = sum(
            plan.encoded_rank_pairs.nbytes for plan in plans
        )
        memory["transform_coefficients"] = sum(
            plan.coefficients.nbytes for plan in plans
        )
        memory["transform_parities"] = sum(
            plan.parities.nbytes for plan in self.forward_plans.values()
        )
        memory["transform_rank_matrices"] = sum(
            0 if plan.rank_matrix is None else plan.rank_matrix.nbytes for plan in plans
        )
        return {name: int(memory[name]) for name in _MEMORY_CATEGORIES}

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())

    def memory_mb(self) -> float:
        return self.memory_bytes() / 1024**2

    def plan_statistics(self) -> Mapping[str, object]:
        plans = tuple(self.forward_plans.values()) + tuple(self.inverse_plans.values())
        memory = self.memory_bytes_by_category()
        dense_count = sum(plan.strategy == "dense" for plan in plans)
        derived_bytes = memory["transform_rank_matrices"]
        return {
            "dims": self.dims,
            "max_truncation": self.max_truncation,
            "forward_grade_count": len(self.forward_plans),
            "inverse_grade_count": len(self.inverse_plans),
            "term_count": sum(plan.term_count for plan in plans),
            "strategy_counts": {
                "dense": dense_count,
                "sparse": len(plans) - dense_count,
            },
            "authoritative_memory_bytes": sum(memory.values()) - derived_bytes,
            "derived_execution_memory_bytes": derived_bytes,
            "bytes_by_category": memory,
            "memory_bytes": sum(memory.values()),
            "memory_mb": sum(memory.values()) / 1024**2,
        }


__all__ = [
    "PartiallySymmetrizedShearPlanStore",
    "PartiallySymmetrizedTransformPlan",
    "TransformOrientation",
    "apply_partially_symmetrized_transform",
    "apply_partially_symmetrized_transform_plan",
]
