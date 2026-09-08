"""Host plans for partially symmetrized bigraded blocks."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from tensordev.core.bigraded.layout import _build_active_layout
from tensordev.core.bigraded.symmetrized._compiled import _binomial_table
from tensordev.core.bigraded.symmetrized._compiled_plans import (
    compile_concatenation_targets,
    compile_doubleprime_generator_destination,
    compile_doubleprime_generator_targets,
    compile_placement_array,
)
from tensordev.core.bigraded.symmetrized.combinatorics import (
    multiset_placement_count,
    multiset_placements,
)
from tensordev.core.bigraded.types import Bidegree, BigradedSpec, _bidegree
from tensordev.core.utils.precompute import _readonly, _unsigned_index_dtype
from tensordev.core.utils.segmented import (
    DestinationRankPlan,
    SegmentedRankPlan,
    apply_destination_rank_plan,
)


_MEMORY_CATEGORIES = (
    "multiset_placements",
    "symmetrized_grade_metadata",
    "symmetrized_concatenation_target_maps",
    "symmetrized_concatenation_segment_metadata",
    "shared_doubleprime_generator_maps",
)


_COMPILED_PLAN_ENTRY_THRESHOLD = 100_000
# Small actions compile faster as one direct scatter.  Above this scalar-work
# count the destination plan reduces the update set enough to repay its gather.
_DESTINATION_GENERATOR_WORK_THRESHOLD = 128


def _index_dtype(maximum: int):
    if maximum < 0:
        raise ValueError("index maximum must be non-negative")
    if maximum > np.iinfo(np.uint64).max:
        raise OverflowError(f"index maximum {maximum} exceeds uint64")
    return _unsigned_index_dtype(maximum)


def _index_itemsize(maximum: int) -> int:
    return np.dtype(_index_dtype(maximum)).itemsize


def _doubleprime_generator_counts(
    d_doubleprime: int,
    output_grade: Bidegree,
) -> tuple[int, int, int, int]:
    """Return output, source, double-prime-prefix, and edge counts."""
    n, m = output_grade
    if m <= 0:
        raise ValueError("a double-prime generator output must have m > 0")
    output_rank_count = multiset_placement_count(
        d_doubleprime,
        output_grade,
    )
    source_rank_count = multiset_placement_count(
        d_doubleprime,
        (n, m - 1),
    )
    prime_rank_count = (
        multiset_placement_count(d_doubleprime, (n - 1, m))
        if n > 0
        else 0
    )
    doubleprime_rank_count = output_rank_count - prime_rank_count
    edge_count = d_doubleprime * source_rank_count
    if not source_rank_count <= doubleprime_rank_count <= edge_count:
        raise AssertionError("invalid double-prime generator rank counts")
    return (
        output_rank_count,
        source_rank_count,
        doubleprime_rank_count,
        edge_count,
    )


def _destination_generator_memory_bytes(
    *,
    source_rank_count: int,
    doubleprime_rank_count: int,
    edge_count: int,
) -> int:
    selected_count = edge_count - source_rank_count
    collision_count = edge_count - doubleprime_rank_count
    return int(
        selected_count * _index_itemsize(edge_count - 1)
        + collision_count * _index_itemsize(doubleprime_rank_count - 1)
    )


def _use_destination_generator_plan(
    d_doubleprime: int,
    *,
    output_rank_count: int,
    source_rank_count: int,
    doubleprime_rank_count: int,
    edge_count: int,
    dense_prime_width: int,
) -> bool:
    """Choose destination execution where it is structurally favorable."""
    destination_bytes = _destination_generator_memory_bytes(
        source_rank_count=source_rank_count,
        doubleprime_rank_count=doubleprime_rank_count,
        edge_count=edge_count,
    )
    forward_bytes = edge_count * _index_itemsize(output_rank_count - 1)
    return d_doubleprime == 1 or (
        edge_count * dense_prime_width
        >= _DESTINATION_GENERATOR_WORK_THRESHOLD
        and destination_bytes <= forward_bytes
    )


def _validated_capacity(
    dims: Bidegree,
    max_truncation: Bidegree,
) -> tuple[Bidegree, Bidegree, BigradedSpec]:
    normalized_dims = _bidegree(dims, name="dims")
    if normalized_dims[0] <= 0 or normalized_dims[1] <= 0:
        raise ValueError(f"dims must be strictly positive, got {normalized_dims}.")
    normalized_truncation = _bidegree(
        max_truncation,
        name="max_truncation",
    )
    spec = BigradedSpec(
        *normalized_dims,
        normalized_truncation,
        partially_symmetrized=True,
    )
    return normalized_dims, normalized_truncation, spec


def _expected_plan_memory_bytes_by_category(
    dims: Bidegree,
    max_truncation: Bidegree,
) -> Mapping[str, int]:
    """Exact retained NumPy payload without constructing placement arrays."""

    normalized_dims, normalized_truncation, spec = _validated_capacity(
        dims,
        max_truncation,
    )
    del normalized_truncation
    d_prime, d_doubleprime = normalized_dims
    memory = {name: 0 for name in _MEMORY_CATEGORIES}

    rank_counts = {}
    for n, m in spec.grades:
        rank_count = multiset_placement_count(d_doubleprime, (n, m))
        rank_counts[n, m] = rank_count
        memory["multiset_placements"] += (
            rank_count
            * (n + 1)
            * d_doubleprime
            * _index_itemsize(m)
        )
        if m > 0:
            (
                output_count,
                source_count,
                doubleprime_count,
                edge_count,
            ) = _doubleprime_generator_counts(
                d_doubleprime,
                (n, m),
            )
            if _use_destination_generator_plan(
                d_doubleprime,
                output_rank_count=output_count,
                source_rank_count=source_count,
                doubleprime_rank_count=doubleprime_count,
                edge_count=edge_count,
                dense_prime_width=d_prime**n,
            ):
                generator_bytes = _destination_generator_memory_bytes(
                    source_rank_count=source_count,
                    doubleprime_rank_count=doubleprime_count,
                    edge_count=edge_count,
                )
            else:
                generator_bytes = edge_count * _index_itemsize(
                    output_count - 1
                )
            memory["shared_doubleprime_generator_maps"] += generator_bytes

    N, M = spec.truncation
    grades = tuple(spec.grades)
    for left_grade in grades:
        for right_grade in grades:
            output_grade = (
                left_grade[0] + right_grade[0],
                left_grade[1] + right_grade[1],
            )
            if output_grade[0] > N or output_grade[1] > M:
                continue
            memory["symmetrized_concatenation_target_maps"] += (
                rank_counts[left_grade]
                * rank_counts[right_grade]
                * _index_itemsize(rank_counts[output_grade] - 1)
            )
    return {name: int(memory[name]) for name in _MEMORY_CATEGORIES}


@dataclass(frozen=True, slots=True, eq=False)
class PartiallySymmetrizedGradePlan:
    """Immutable multiset-rank metadata for one bidegree."""

    grade: Bidegree
    rank_count: int
    block_width: int
    dense_shape: Tuple[int, int]
    placements: np.ndarray

    @property
    def total_degree(self) -> int:
        return self.grade[0] + self.grade[1]

    def memory_bytes(self) -> int:
        return int(self.placements.nbytes)


@dataclass(frozen=True, slots=True, eq=False)
class PartiallySymmetrizedConcatPlan:
    """Many-to-one rank map and prime-axis layout for one concatenation."""

    left_grade: Bidegree
    right_grade: Bidegree
    output_grade: Bidegree
    rank_plan: SegmentedRankPlan
    dense_axis_permutation: Tuple[int, ...]
    outer_axis_permutation: Tuple[int, ...]
    dense_input_shape: Tuple[int, ...]
    dense_output_shape: Tuple[int, ...]
    left_rank_count: int
    right_rank_count: int
    output_rank_count: int

    def target_ranks(self) -> np.ndarray:
        return self.rank_plan.target_ranks.reshape(
            self.left_rank_count,
            self.right_rank_count,
        )

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        return {
            "target_maps": int(self.rank_plan.target_ranks.nbytes),
            # Direct scatter-add requires neither source ordering nor stored
            # segment offsets.
            "segment_metadata": 0,
        }

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())


@dataclass(frozen=True, slots=True, eq=False)
class DoublePrimeGeneratorPlan:
    """Shared terminal-block append map for both coordinate choices."""

    output_grade: Bidegree
    source_grade: Bidegree
    target_ranks: np.ndarray | None
    destination_plan: DestinationRankPlan | None
    source_rank_count: int
    output_rank_count: int
    doubleprime_rank_count: int
    d_doubleprime: int
    dense_prime_width: int

    def __post_init__(self) -> None:
        if (self.target_ranks is None) == (self.destination_plan is None):
            raise ValueError(
                "exactly one generator execution plan must be supplied"
            )
        if not 0 < self.doubleprime_rank_count <= self.output_rank_count:
            raise ValueError("invalid double-prime output prefix")
        if self.target_ranks is not None:
            expected = (self.d_doubleprime, self.source_rank_count)
            if self.target_ranks.shape != expected:
                raise ValueError(
                    f"target_ranks has shape {self.target_ranks.shape}, "
                    f"expected {expected}."
                )
            if self.target_ranks.size:
                if int(self.target_ranks.max()) >= self.doubleprime_rank_count:
                    raise ValueError(
                        "generator targets must lie in the double-prime prefix"
                    )
        else:
            destination = self.destination_plan
            if destination is None:
                raise AssertionError("missing destination plan")
            if destination.edge_count != (
                self.d_doubleprime * self.source_rank_count
            ):
                raise ValueError("destination edge count is inconsistent")
            if destination.output_rank_count != self.doubleprime_rank_count:
                raise ValueError("destination output count is inconsistent")

    @property
    def uses_destination_order(self) -> bool:
        return self.destination_plan is not None

    def memory_bytes(self) -> int:
        if self.destination_plan is not None:
            return self.destination_plan.memory_bytes()
        if self.target_ranks is None:
            raise AssertionError("missing generator plan")
        return int(self.target_ranks.nbytes)


def apply_doubleprime_generator_prefix(
    xp: Any,
    block,
    generator,
    plan: DoublePrimeGeneratorPlan,
    *,
    scatter_add,
):
    """Apply the double-prime contribution in its compact rank prefix."""

    source_width = plan.source_rank_count * plan.dense_prime_width
    if block.ndim == 0 or block.shape[-1] != source_width:
        raise ValueError(
            f"source block has final width "
            f"{None if block.ndim == 0 else block.shape[-1]}, expected "
            f"{source_width} for grade {plan.source_grade}."
        )
    if generator.ndim == 0 or generator.shape[-1] != plan.d_doubleprime:
        raise ValueError(
            "double-prime generator has final width "
            f"{None if generator.ndim == 0 else generator.shape[-1]}, "
            f"expected {plan.d_doubleprime}."
        )

    batch = xp.broadcast_shapes(block.shape[:-1], generator.shape[:-1])
    source = xp.broadcast_to(block, batch + (source_width,)).reshape(
        batch + (plan.source_rank_count, plan.dense_prime_width)
    )
    generator = xp.broadcast_to(
        generator,
        batch + (plan.d_doubleprime,),
    )
    values = (
        source[..., None, :, :]
        * generator[..., :, None, None]
    ).reshape(
        batch
        + (
            plan.d_doubleprime * plan.source_rank_count,
            plan.dense_prime_width,
        )
    )
    if plan.destination_plan is not None:
        output = apply_destination_rank_plan(
            xp,
            values,
            plan.destination_plan,
            scatter_add=scatter_add,
        )
    else:
        if plan.target_ranks is None:
            raise AssertionError("missing generator target map")
        output = xp.zeros(
            batch
            + (plan.doubleprime_rank_count, plan.dense_prime_width),
            dtype=xp.result_type(block.dtype, generator.dtype),
        )
        output = scatter_add(
            output,
            xp.asarray(plan.target_ranks).reshape(-1),
            values,
        )
    return output.reshape(
        batch + (plan.doubleprime_rank_count * plan.dense_prime_width,)
    )


def apply_doubleprime_generator_block(
    xp: Any,
    block,
    generator,
    plan: DoublePrimeGeneratorPlan,
    *,
    scatter_add,
):
    """Apply the double-prime contribution in the full output layout."""
    output = apply_doubleprime_generator_prefix(
        xp,
        block,
        generator,
        plan,
        scatter_add=scatter_add,
    )
    if plan.doubleprime_rank_count == plan.output_rank_count:
        return output
    batch = output.shape[:-1]
    padding_width = (
        plan.output_rank_count - plan.doubleprime_rank_count
    ) * plan.dense_prime_width
    return xp.concatenate(
        (
            output,
            xp.zeros(batch + (padding_width,), dtype=output.dtype),
        ),
        axis=-1,
    )


class PartiallySymmetrizedPlanStore:
    """Fixed-capacity partial-symmetry data shared by active views."""

    def __init__(
        self,
        dims: Bidegree,
        max_truncation: Bidegree,
    ) -> None:
        self.dims, self.max_truncation, capacity_spec = _validated_capacity(
            dims,
            max_truncation,
        )
        self._grade_plans: Dict[Bidegree, PartiallySymmetrizedGradePlan] = {}
        self._concat_plans: Dict[
            Tuple[Bidegree, Bidegree],
            PartiallySymmetrizedConcatPlan,
        ] = {}
        self._doubleprime_generator_plans: Dict[
            Bidegree,
            DoublePrimeGeneratorPlan,
        ] = {}
        self._grade_plans_view = MappingProxyType(self._grade_plans)
        self._concat_plans_view = MappingProxyType(self._concat_plans)
        self._doubleprime_generator_plans_view = MappingProxyType(
            self._doubleprime_generator_plans
        )
        self._active_layouts = {}
        self._layout_lock = RLock()
        N, M = self.max_truncation
        d_doubleprime = self.dims[1]
        rank_counts = {
            grade: multiset_placement_count(d_doubleprime, grade)
            for grade in capacity_spec.grades
        }
        grades = tuple(capacity_spec.grades)
        entry_count = sum(
            rank_counts[left_grade] * rank_counts[right_grade]
            for left_grade in grades
            for right_grade in grades
            if (
                left_grade[0] + right_grade[0] <= N
                and left_grade[1] + right_grade[1] <= M
            )
        ) + sum(
            d_doubleprime * rank_counts[n, m - 1]
            for n, m in grades
            if m > 0
        )
        self._use_compiled_plan_builder = (
            entry_count >= _COMPILED_PLAN_ENTRY_THRESHOLD
        )
        self._precompute_grades(capacity_spec)
        part_count = (N + 1) * d_doubleprime
        binomial = _binomial_table(
            max(part_count - 2 + M, 0),
            max_column=max(part_count - 1, 0),
            max_complement=M,
        )
        self._precompute_concatenations(binomial)
        self._precompute_doubleprime_generators(binomial)

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
    def grade_plans(self) -> Mapping[Bidegree, PartiallySymmetrizedGradePlan]:
        return self._grade_plans_view

    @property
    def concat_plans(
        self,
    ) -> Mapping[
        Tuple[Bidegree, Bidegree],
        PartiallySymmetrizedConcatPlan,
    ]:
        return self._concat_plans_view

    @property
    def doubleprime_generator_plans(
        self,
    ) -> Mapping[Bidegree, DoublePrimeGeneratorPlan]:
        return self._doubleprime_generator_plans_view

    def _precompute_grades(self, spec: BigradedSpec) -> None:
        d_prime, d_doubleprime = self.dims
        for grade in spec.grades:
            n, m = grade
            rank_count = multiset_placement_count(d_doubleprime, grade)
            dense_prime_width = d_prime**n
            if self._use_compiled_plan_builder:
                placements = compile_placement_array(d_doubleprime, grade)
            else:
                placements = _readonly(
                    np.asarray(
                        multiset_placements(d_doubleprime, grade),
                        dtype=_index_dtype(m),
                    ).reshape(rank_count, n + 1, d_doubleprime)
                )
            self._grade_plans[grade] = PartiallySymmetrizedGradePlan(
                grade=grade,
                rank_count=rank_count,
                block_width=rank_count * dense_prime_width,
                dense_shape=(rank_count, dense_prime_width),
                placements=placements,
            )

    def _precompute_concatenations(
        self,
        binomial: np.ndarray,
    ) -> None:
        N, M = self.max_truncation
        grades = tuple(self._grade_plans)
        for left_grade in grades:
            for right_grade in grades:
                output_grade = (
                    left_grade[0] + right_grade[0],
                    left_grade[1] + right_grade[1],
                )
                if output_grade[0] <= N and output_grade[1] <= M:
                    self._concat_plans[left_grade, right_grade] = (
                        self._build_concat_plan(
                            left_grade,
                            right_grade,
                            binomial=binomial,
                        )
                    )

    def _build_concat_plan(
        self,
        left_grade: Bidegree,
        right_grade: Bidegree,
        *,
        binomial: np.ndarray,
    ) -> PartiallySymmetrizedConcatPlan:
        n1, _ = left_grade
        n2, _ = right_grade
        output_grade = (
            left_grade[0] + right_grade[0],
            left_grade[1] + right_grade[1],
        )
        left_plan = self._grade_plans[left_grade]
        right_plan = self._grade_plans[right_grade]
        output_plan = self._grade_plans[output_grade]

        targets = compile_concatenation_targets(
            left_plan.placements,
            right_plan.placements,
            binomial,
            output_rank_count=output_plan.rank_count,
            compiled=self._use_compiled_plan_builder,
        )
        rank_plan = SegmentedRankPlan.from_targets(
            targets,
            output_rank_count=output_plan.rank_count,
        )

        right_rank_axis = 1 + n1
        outer_axis_permutation = (
            (0, right_rank_axis)
            + tuple(range(1, right_rank_axis))
            + tuple(range(right_rank_axis + 1, right_rank_axis + 1 + n2))
        )
        dense_axis_permutation = tuple(range(n1 + n2))
        d_prime = self.dims[0]
        return PartiallySymmetrizedConcatPlan(
            left_grade=left_grade,
            right_grade=right_grade,
            output_grade=output_grade,
            rank_plan=rank_plan,
            dense_axis_permutation=dense_axis_permutation,
            outer_axis_permutation=outer_axis_permutation,
            dense_input_shape=(d_prime,) * (n1 + n2),
            dense_output_shape=(d_prime,) * (n1 + n2),
            left_rank_count=left_plan.rank_count,
            right_rank_count=right_plan.rank_count,
            output_rank_count=output_plan.rank_count,
        )

    def _precompute_doubleprime_generators(
        self,
        binomial: np.ndarray,
    ) -> None:
        d_prime, d_doubleprime = self.dims
        for output_grade, output_plan in self._grade_plans.items():
            n, m = output_grade
            if m == 0:
                continue
            source_grade = n, m - 1
            source_plan = self._grade_plans[source_grade]
            (
                output_rank_count,
                source_rank_count,
                doubleprime_rank_count,
                edge_count,
            ) = _doubleprime_generator_counts(
                d_doubleprime,
                output_grade,
            )
            if output_plan.rank_count != output_rank_count:
                raise AssertionError("inconsistent output rank count")
            if source_plan.rank_count != source_rank_count:
                raise AssertionError("inconsistent source rank count")

            use_destination = _use_destination_generator_plan(
                d_doubleprime,
                output_rank_count=output_rank_count,
                source_rank_count=source_rank_count,
                doubleprime_rank_count=doubleprime_rank_count,
                edge_count=edge_count,
                dense_prime_width=d_prime**n,
            )
            if use_destination:
                targets = None
                destination_plan = (
                    compile_doubleprime_generator_destination(
                        output_plan.placements,
                        binomial,
                        source_rank_count=source_rank_count,
                        doubleprime_rank_count=doubleprime_rank_count,
                        compiled=self._use_compiled_plan_builder,
                    )
                )
            else:
                targets = compile_doubleprime_generator_targets(
                    source_plan.placements,
                    binomial,
                    output_rank_count=output_rank_count,
                    compiled=self._use_compiled_plan_builder,
                )
                destination_plan = None
                for letter_targets in targets:
                    if np.unique(letter_targets).size != letter_targets.size:
                        raise AssertionError(
                            "double-prime generator targets must be injective "
                            f"at output grade {output_grade}."
                        )
            self._doubleprime_generator_plans[output_grade] = (
                DoublePrimeGeneratorPlan(
                    output_grade=output_grade,
                    source_grade=source_grade,
                    target_ranks=targets,
                    destination_plan=destination_plan,
                    source_rank_count=source_plan.rank_count,
                    output_rank_count=output_plan.rank_count,
                    doubleprime_rank_count=doubleprime_rank_count,
                    d_doubleprime=d_doubleprime,
                    dense_prime_width=d_prime**n,
                )
            )

    def grade_plan(self, grade: object) -> PartiallySymmetrizedGradePlan:
        normalized = _bidegree(grade)
        try:
            return self._grade_plans[normalized]
        except KeyError as exc:
            raise KeyError(
                f"bidegree {normalized} exceeds plan-store capacity "
                f"{self.max_truncation}."
            ) from exc

    def concat_plan(
        self,
        left_grade: object,
        right_grade: object,
    ) -> PartiallySymmetrizedConcatPlan:
        key = _bidegree(left_grade), _bidegree(right_grade)
        try:
            return self._concat_plans[key]
        except KeyError as exc:
            output = (
                key[0][0] + key[1][0],
                key[0][1] + key[1][1],
            )
            raise KeyError(
                f"concatenation output {output} exceeds plan-store capacity "
                f"{self.max_truncation}."
            ) from exc

    def doubleprime_generator_plan(
        self,
        output_grade: object,
    ) -> DoublePrimeGeneratorPlan:
        normalized = _bidegree(output_grade, name="output_grade")
        try:
            return self._doubleprime_generator_plans[normalized]
        except KeyError as exc:
            if normalized in self._grade_plans:
                raise KeyError(
                    f"output grade {normalized} has no double-prime "
                    "predecessor."
                ) from exc
            raise KeyError(
                f"output grade {normalized} exceeds plan-store capacity "
                f"{self.max_truncation}."
            ) from exc

    def resolve(
        self,
        truncation: Bidegree | None = None,
        *,
        include_scalar: bool = True,
        coordinates: str = "standard",
    ):
        """Return a cached active layout without copying plan arrays."""

        if truncation is None:
            truncation = self.max_truncation
        truncation = _bidegree(truncation, name="truncation")
        if (
            truncation[0] > self.max_truncation[0]
            or truncation[1] > self.max_truncation[1]
        ):
            raise ValueError(
                f"active truncation {truncation} exceeds plan-store capacity "
                f"{self.max_truncation}."
            )
        if not isinstance(include_scalar, bool):
            raise TypeError("include_scalar must be a bool.")
        key = truncation, include_scalar, coordinates
        with self._layout_lock:
            cached = self._active_layouts.get(key)
            if cached is not None:
                return cached
            layout = _build_active_layout(
                self,
                truncation=truncation,
                include_scalar=include_scalar,
                coordinates=coordinates,
                partially_symmetrized=True,
            )
            self._active_layouts[key] = layout
            return layout

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        memory = {name: 0 for name in _MEMORY_CATEGORIES}
        memory["multiset_placements"] = sum(
            plan.placements.nbytes for plan in self._grade_plans.values()
        )
        for plan in self._concat_plans.values():
            categories = plan.memory_bytes_by_category()
            memory["symmetrized_concatenation_target_maps"] += categories[
                "target_maps"
            ]
            memory["symmetrized_concatenation_segment_metadata"] += categories[
                "segment_metadata"
            ]
        memory["shared_doubleprime_generator_maps"] = sum(
            plan.memory_bytes()
            for plan in self._doubleprime_generator_plans.values()
        )
        return {name: int(memory[name]) for name in _MEMORY_CATEGORIES}

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
            "concat_plan_count": len(self._concat_plans),
            "doubleprime_generator_plan_count": len(
                self._doubleprime_generator_plans
            ),
            "active_layout_count": len(self._active_layouts),
            "bytes_by_category": memory,
            "memory_bytes": sum(memory.values()),
            "memory_mb": sum(memory.values()) / 1024**2,
        }


__all__ = [
    "DoublePrimeGeneratorPlan",
    "PartiallySymmetrizedConcatPlan",
    "PartiallySymmetrizedGradePlan",
    "PartiallySymmetrizedPlanStore",
    "apply_doubleprime_generator_block",
    "apply_doubleprime_generator_prefix",
]
