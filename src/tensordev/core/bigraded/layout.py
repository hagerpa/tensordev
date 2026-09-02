"""Resolved active layouts for rectangular bidegree truncations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple

from tensordev.core.bigraded.types import Bidegree, BigradedSpec


class _BigradedLayoutStore(Protocol):
    """Structural plan-store surface needed by an active layout."""

    dims: Bidegree

    def grade_plan(self, grade: object) -> Any: ...

    def concat_plan(self, left_grade: object, right_grade: object) -> Any: ...

    def resolve(
        self,
        truncation: Bidegree | None = None,
        *,
        include_scalar: bool = True,
        coordinates: str = "standard",
    ) -> "BigradedLayout": ...


class _BigradedConcatPlan(Protocol):
    output_grade: Bidegree


@dataclass(frozen=True, slots=True, eq=False)
class BigradedLayout:
    """A finite active view into a bounded :class:`BigradedPlanStore`.

    The tuples below contain references to capacity-level plans; resolving a
    smaller rectangle does not copy their NumPy arrays.
    """

    store: _BigradedLayoutStore
    spec: BigradedSpec
    grade_plans: Tuple[Any, ...]
    _product_splits: Tuple[Tuple[Tuple[Bidegree, Bidegree], ...], ...]

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def truncation(self) -> Bidegree:
        return self.spec.truncation

    @property
    def grades(self) -> Tuple[Bidegree, ...]:
        return self.spec.grades

    @property
    def zero_grade(self) -> Bidegree:
        return 0, 0

    @property
    def include_scalar(self) -> bool:
        return self.spec.include_scalar

    @property
    def coordinates(self) -> str:
        return self.spec.coordinates

    @property
    def representation(self) -> str:
        return self.spec.representation

    @property
    def dims(self) -> Bidegree:
        return self.spec.dims

    def index(self, grade: object) -> int:
        return self.spec.index(grade)

    def contains(self, grade: object) -> bool:
        return self.spec.contains(grade)

    def total_degree(self, grade: object) -> int:
        return self.spec.total_degree(grade)

    def block_width(self, grade: object) -> int:
        return self.spec.block_width(grade)

    def rank_count(self, grade: object) -> int:
        return self.spec.rank_count(grade)

    def placement_count(self, grade: object) -> int:
        if self.representation != "ordered":
            raise ValueError(
                "placement_count is only defined for ordered layouts; "
                "use rank_count for representation-neutral code."
            )
        return self.spec.placement_count(grade)

    def grade_plan(self, grade: object) -> Any:
        return self.grade_plans[self.index(grade)]

    def product_splits(
        self,
        output_grade: object,
    ) -> Tuple[Tuple[Bidegree, Bidegree], ...]:
        return self._product_splits[self.index(output_grade)]

    def concat_plan(
        self,
        left_grade: object,
        right_grade: object,
    ) -> _BigradedConcatPlan:
        if not self.contains(left_grade):
            raise KeyError(f"left bidegree {left_grade!r} is not active.")
        if not self.contains(right_grade):
            raise KeyError(f"right bidegree {right_grade!r} is not active.")
        plan = self.store.concat_plan(left_grade, right_grade)
        if not self.contains(plan.output_grade):
            raise KeyError(
                f"output bidegree {plan.output_grade} lies outside active "
                f"truncation {self.truncation}."
            )
        return plan

    def with_scalar(self, include_scalar: bool) -> "BigradedLayout":
        return self.store.resolve(
            self.truncation,
            include_scalar=include_scalar,
            coordinates=self.coordinates,
        )


def _build_active_layout(
    store: _BigradedLayoutStore,
    *,
    truncation: Bidegree,
    include_scalar: bool,
    coordinates: str,
    representation: str,
) -> BigradedLayout:
    """Build representation-neutral active metadata around store-owned plans."""
    spec = BigradedSpec(
        *store.dims,
        truncation,
        coordinates=coordinates,
        include_scalar=include_scalar,
        representation=representation,
    )
    grade_plans = tuple(store.grade_plan(grade) for grade in spec.grades)

    # Product schedules depend only on the grading rectangle.  Scalar-free
    # layouts still need scalar splits for products whose input carries one.
    convolution_grades = spec.with_scalar(True).grades
    convolution_grade_set = set(convolution_grades)
    split_rows = []
    for output in spec.grades:
        splits = []
        for left in convolution_grades:
            right = output[0] - left[0], output[1] - left[1]
            if right in convolution_grade_set:
                splits.append((left, right))
        split_rows.append(tuple(splits))

    return BigradedLayout(
        store=store,
        spec=spec,
        grade_plans=grade_plans,
        _product_splits=tuple(split_rows),
    )
