from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import comb
from numbers import Integral
from types import MappingProxyType
from typing import Literal, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np

from tensordev.core.universal import _Array, _ArrayNamespace
from tensordev.core.utils.precompute import _readonly

Array = TypeVar("Array", bound=_Array)
ShuffleScope = Literal["none", "generator", "full"]


def _normalize_precompute_shuffle(
    value: object,
    *,
    allow_generator: bool,
) -> ShuffleScope:
    """Normalize the shared public shuffle-precomputation convention."""
    if isinstance(value, bool):
        return "full" if value else "none"
    if allow_generator and isinstance(value, str) and value == "generator":
        return "generator"
    expected = "a boolean or 'generator'" if allow_generator else "a boolean"
    raise TypeError(f"precompute_shuffle must be {expected}, got {value!r}.")


def _precompute_shuffle_argument(scope: ShuffleScope) -> bool | Literal["generator"]:
    """Return the public constructor value represented by a shuffle scope."""
    if scope == "none":
        return False
    if scope == "generator":
        return "generator"
    if scope == "full":
        return True
    raise ValueError(f"unknown shuffle scope {scope!r}.")


def _non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer, got {value!r}.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _axis_dtype(maximum: int):
    """Smallest signed NumPy dtype capable of storing ``maximum``."""
    for dtype in (np.int8, np.int16, np.int32):
        if maximum <= np.iinfo(dtype).max:
            return dtype
    return np.int64


def _interleave_axes(
    left_axes: Sequence[int],
    right_axes: Sequence[int],
    left_positions: Sequence[int],
) -> Tuple[int, ...]:
    """Interleave two ordered axis lists at fixed left-axis positions.

    This is the common combinatorial primitive behind ordinary and bigraded
    shuffle permutations. The relative order within both input sequences is
    preserved.
    """
    left_axes = tuple(int(axis) for axis in left_axes)
    right_axes = tuple(int(axis) for axis in right_axes)
    left_positions = tuple(int(position) for position in left_positions)
    total_degree = len(left_axes) + len(right_axes)
    if len(left_positions) != len(left_axes):
        raise ValueError(
            "left_positions must contain one position for every left axis, "
            f"got {len(left_positions)} positions for {len(left_axes)} axes."
        )
    if (
        any(position < 0 or position >= total_degree for position in left_positions)
        or any(a >= b for a, b in zip(left_positions, left_positions[1:]))
    ):
        raise ValueError(
            "left_positions must be strictly increasing positions in "
            f"range({total_degree}), got {left_positions}."
        )

    left_iter = iter(left_axes)
    right_iter = iter(right_axes)
    left_position_set = set(left_positions)
    return tuple(
        next(left_iter) if position in left_position_set else next(right_iter)
        for position in range(total_degree)
    )


@lru_cache(maxsize=None)
def ordinary_axis_permutations(
    left_degree: int,
    right_degree: int,
) -> np.ndarray:
    """All order-preserving shuffle permutations for two tensor degrees.

    Rows describe the source-axis order of a transposed outer product. The
    returned table is compact, immutable, and shared between cores with the
    same degree pair.
    """
    left_degree = _non_negative_int(left_degree, name="left_degree")
    right_degree = _non_negative_int(right_degree, name="right_degree")
    total_degree = left_degree + right_degree
    left_axes = tuple(range(left_degree))
    right_axes = tuple(range(left_degree, total_degree))
    permutations = tuple(
        _interleave_axes(left_axes, right_axes, left_positions)
        for left_positions in combinations(range(total_degree), left_degree)
    )
    table = np.asarray(
        permutations,
        dtype=_axis_dtype(max(total_degree - 1, 0)),
    ).reshape(comb(total_degree, left_degree), total_degree)
    return _readonly(table)


@dataclass(frozen=True, slots=True, eq=False)
class HomogeneousShufflePlan:
    """Compact permutation plan for one canonical homogeneous shuffle.

    ``axis_permutations`` stores one source-axis permutation per shuffle. For
    base dimension one, or when the right input has degree zero, transposing is
    unnecessary; ``direct_multiplicity`` records the complete operation and
    the permutation table is empty.
    """

    dimension: int
    left_degree: int
    right_degree: int
    permutation_count: int
    axis_permutations: np.ndarray
    direct_multiplicity: Optional[int]

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def output_degree(self) -> int:
        return self.left_degree + self.right_degree

    @property
    def left_width(self) -> int:
        return self.dimension**self.left_degree

    @property
    def right_width(self) -> int:
        return self.dimension**self.right_degree

    @property
    def output_width(self) -> int:
        return self.dimension**self.output_degree

    @property
    def uses_direct_scaling(self) -> bool:
        return self.direct_multiplicity is not None

    def memory_bytes(self) -> int:
        """Bytes occupied by the NumPy permutation payload."""
        return int(self.axis_permutations.nbytes)

    def apply(self, xp: _ArrayNamespace, left: Array, right: Array) -> Array:
        """Apply this plan using a NumPy-compatible array namespace."""
        return permutation_shuffle_homogeneous(xp, left, right, self)


def build_homogeneous_shuffle_plan(
    dimension: int,
    left_degree: int,
    right_degree: int,
) -> HomogeneousShufflePlan:
    """Build a compact plan for ``left_degree >= right_degree``."""
    dimension = _non_negative_int(dimension, name="dimension")
    if dimension == 0:
        raise ValueError("dimension must be strictly positive, got 0.")
    left_degree = _non_negative_int(left_degree, name="left_degree")
    right_degree = _non_negative_int(right_degree, name="right_degree")
    if left_degree < right_degree:
        raise ValueError(
            "shuffle plans use canonical degree order left_degree >= "
            f"right_degree, got {(left_degree, right_degree)}."
        )

    output_degree = left_degree + right_degree
    permutation_count = comb(output_degree, left_degree)
    if dimension == 1:
        direct_multiplicity = permutation_count
    elif right_degree == 0:
        direct_multiplicity = 1
    else:
        direct_multiplicity = None

    if direct_multiplicity is None:
        axis_permutations = ordinary_axis_permutations(
            left_degree,
            right_degree,
        )
    else:
        axis_permutations = _readonly(
            np.empty(
                (0, output_degree),
                dtype=_axis_dtype(max(output_degree - 1, 0)),
            )
        )
    return HomogeneousShufflePlan(
        dimension=dimension,
        left_degree=left_degree,
        right_degree=right_degree,
        permutation_count=permutation_count,
        axis_permutations=axis_permutations,
        direct_multiplicity=direct_multiplicity,
    )


def _prepare_homogeneous_shuffle_inputs(
    xp: _ArrayNamespace,
    left: Array,
    right: Array,
    plan: HomogeneousShufflePlan,
):
    """Validate homogeneous levels, broadcast their batches, and return both."""
    if left.ndim == 0 or left.shape[-1] != plan.left_width:
        width = None if left.ndim == 0 else left.shape[-1]
        raise ValueError(
            f"left degree-{plan.left_degree} level has width {width}, "
            f"expected {plan.left_width}."
        )
    if right.ndim == 0 or right.shape[-1] != plan.right_width:
        width = None if right.ndim == 0 else right.shape[-1]
        raise ValueError(
            f"right degree-{plan.right_degree} level has width {width}, "
            f"expected {plan.right_width}."
        )

    batch_shape = np.broadcast_shapes(left.shape[:-1], right.shape[:-1])
    left = xp.broadcast_to(left, batch_shape + (plan.left_width,))
    right = xp.broadcast_to(right, batch_shape + (plan.right_width,))
    return batch_shape, left, right


def _direct_homogeneous_shuffle(
    left: Array,
    right: Array,
    plan: HomogeneousShufflePlan,
) -> Array:
    """Apply a direct plan after input preparation."""
    return left * right * plan.direct_multiplicity


def _homogeneous_outer_product(
    xp: _ArrayNamespace,
    left: Array,
    right: Array,
    batch_shape: Tuple[int, ...],
    plan: HomogeneousShufflePlan,
) -> Array:
    """Construct the dense-axis view shared by permutation kernels."""
    return xp.reshape(
        left[..., :, None] * right[..., None, :],
        batch_shape + (plan.dimension,) * plan.output_degree,
    )


def _sum_homogeneous_axis_permutations(
    xp: _ArrayNamespace,
    outer: Array,
    batch_shape: Tuple[int, ...],
    plan: HomogeneousShufflePlan,
) -> Array:
    """Sum a modest static permutation table and flatten the dense axes."""
    batch_ndim = len(batch_shape)
    batch_axes = tuple(range(batch_ndim))
    identity = tuple(range(plan.output_degree))

    def permuted(permutation):
        permutation = tuple(int(axis) for axis in permutation)
        if permutation == identity:
            return outer
        axes = batch_axes + tuple(batch_ndim + axis for axis in permutation)
        return xp.transpose(outer, axes)

    first, *remaining = plan.axis_permutations
    result = permuted(first)
    for permutation in remaining:
        result = result + permuted(permutation)
    return xp.reshape(result, batch_shape + (plan.output_width,))


def permutation_shuffle_homogeneous(
    xp: _ArrayNamespace,
    left: Array,
    right: Array,
    plan: HomogeneousShufflePlan,
) -> Array:
    """Apply an ordinary homogeneous shuffle plan, preserving batch axes."""
    batch_shape, left, right = _prepare_homogeneous_shuffle_inputs(
        xp,
        left,
        right,
        plan,
    )

    if plan.uses_direct_scaling:
        return _direct_homogeneous_shuffle(left, right, plan)

    outer = _homogeneous_outer_product(
        xp,
        left,
        right,
        batch_shape,
        plan,
    )
    return _sum_homogeneous_axis_permutations(
        xp,
        outer,
        batch_shape,
        plan,
    )


class TotalDegreeShufflePlanStore:
    """Fixed-capacity ordinary shuffle plans for one alphabet dimension.

    The store owns only immutable host-side plan metadata.  Algebra cores own
    the numerical operations and may share one store across active-truncation
    views.  One canonical orientation, ``left_degree >= right_degree``, is
    retained for every admissible pair.
    """

    def __init__(self, d: int, max_trunc: int) -> None:
        self.d = _non_negative_int(d, name="d")
        if self.d == 0:
            raise ValueError("d must be strictly positive, got 0.")
        self.max_truncation = _non_negative_int(
            max_trunc,
            name="max_trunc",
        )
        self._plans: dict[Tuple[int, int], HomogeneousShufflePlan] = {}
        self._plans_view = MappingProxyType(self._plans)
        self._precompute()

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(d={self.d}, "
            f"max_truncation={self.max_truncation})"
        )

    @property
    def plans(self) -> Mapping[Tuple[int, int], HomogeneousShufflePlan]:
        """Read-only canonical degree-pair to plan mapping."""
        return self._plans_view

    def memory_bytes_by_category(self) -> dict[str, int]:
        """Precomputed array payload grouped by representation category."""
        return {
            "axis_permutations": sum(
                plan.memory_bytes() for plan in self._plans.values()
            )
        }

    def memory_bytes(self) -> int:
        """Total bytes occupied by the precomputed permutation payloads."""
        return sum(self.memory_bytes_by_category().values())

    def memory_mb(self) -> float:
        """Precomputed plan memory in megabytes."""
        return self.memory_bytes() / 1024**2

    def plan_statistics(self) -> dict[str, object]:
        """Summary counts and payload size for the fixed-capacity plan store."""
        categories = self.memory_bytes_by_category()
        memory_bytes = sum(categories.values())
        return {
            "d": self.d,
            "max_truncation": self.max_truncation,
            "plan_count": len(self._plans),
            "permutation_count": sum(
                plan.permutation_count for plan in self._plans.values()
            ),
            "direct_plan_count": sum(
                plan.uses_direct_scaling for plan in self._plans.values()
            ),
            "bytes_by_category": categories,
            "memory_bytes": memory_bytes,
            "memory_mb": memory_bytes / 1024**2,
        }

    def _precompute(self) -> None:
        for total_degree in range(self.max_truncation + 1):
            for left_degree in range(total_degree, -1, -1):
                right_degree = total_degree - left_degree
                if right_degree > left_degree:
                    break
                self._plans[(left_degree, right_degree)] = (
                    build_homogeneous_shuffle_plan(
                        self.d,
                        left_degree,
                        right_degree,
                    )
                )

    def plan(
        self,
        left_degree: int,
        right_degree: int,
    ) -> HomogeneousShufflePlan:
        """Return the plan for a canonical pair ``left_degree >= right_degree``."""
        left_degree = _non_negative_int(left_degree, name="left_degree")
        right_degree = _non_negative_int(right_degree, name="right_degree")
        if left_degree < right_degree:
            raise ValueError(
                "shuffle plans require canonical degree order left_degree >= "
                f"right_degree, got {(left_degree, right_degree)}."
            )
        output_degree = left_degree + right_degree
        if output_degree > self.max_truncation:
            raise KeyError(
                f"shuffle output degree {output_degree} exceeds plan-store "
                f"capacity {self.max_truncation}."
            )
        return self._plans[left_degree, right_degree]

    def apply(
        self,
        xp: _ArrayNamespace,
        left: Array,
        right: Array,
        left_degree: int,
        right_degree: int,
    ) -> Array:
        """Apply one canonical plan using a NumPy-compatible namespace."""
        return self.plan(left_degree, right_degree).apply(xp, left, right)
