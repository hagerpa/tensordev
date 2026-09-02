"""Host-side precomputation for bounded ordered bidegree layouts."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from numbers import Integral
from threading import RLock
from types import MappingProxyType
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from tensordev.core.bigraded.types import Bidegree, BigradedSpec, _bidegree
from tensordev.core.utils.precompute import _readonly


def _index_dtype(maximum: int):
    return np.int32 if maximum <= np.iinfo(np.int32).max else np.int64


def _expected_plan_memory_bytes_by_category(
    dims: Bidegree,
    max_truncation: Bidegree,
) -> Mapping[str, int]:
    """Exact NumPy payload categories allocated by ``BigradedPlanStore``.

    This follows the same width and index-dtype rules as plan construction but
    does not allocate any plan arrays.
    """
    spec = BigradedSpec(*dims, max_truncation)
    d_prime, d_doubleprime = spec.dims
    dimension = d_prime + d_doubleprime
    placement_bytes = 0
    conversion_bytes = 0
    concatenation_bytes = 0

    for n, m in spec.grades:
        degree = n + m
        placements = comb(degree, n)
        width = placements * d_prime**n * d_doubleprime**m
        ordinary_width = dimension**degree
        placement_itemsize = np.dtype(_index_dtype(max(degree, 1))).itemsize
        coordinate_itemsize = np.dtype(
            _index_dtype(max(ordinary_width - 1, width - 1, 0))
        ).itemsize
        placement_bytes += placements * n * placement_itemsize
        # A single retained-coordinate -> ordinary-word map supports both
        # gather and scatter conversions.  Its dense inverse would duplicate
        # ``ordinary_width`` entries without serving a runtime operation.
        conversion_bytes += width * coordinate_itemsize

    for left in spec.grades:
        for right in spec.grades:
            output = left[0] + right[0], left[1] + right[1]
            if not spec.contains(output):
                continue
            right_placements = comb(sum(right), right[0])
            output_placements = comb(sum(output), output[0])
            itemsize = np.dtype(
                _index_dtype(max(output_placements - 1, 0))
            ).itemsize
            concatenation_bytes += right_placements * itemsize
    return {
        "placements": int(placement_bytes),
        "conversion": int(conversion_bytes),
        "concatenation": int(concatenation_bytes),
    }


def colex_rank(placement: Sequence[int], *, length: int | None = None) -> int:
    """Rank a zero-based, increasing placement subset in colexicographic order."""
    values = tuple(placement)
    if length is not None:
        if isinstance(length, bool) or not isinstance(length, Integral):
            raise TypeError(f"length must be a non-negative integer, got {length!r}.")
        length = int(length)
        if length < 0:
            raise ValueError(f"length must be non-negative, got {length}.")

    normalized = []
    previous = -1
    for position, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                f"placement[{position}] must be a non-negative integer, got {value!r}."
            )
        value = int(value)
        if value < 0:
            raise ValueError(f"placement entries must be non-negative, got {value}.")
        if value <= previous:
            raise ValueError(f"placement must be strictly increasing, got {values}.")
        if length is not None and value >= length:
            raise ValueError(
                f"placement entry {value} is outside zero-based length {length}."
            )
        normalized.append(value)
        previous = value
    return sum(comb(value, i + 1) for i, value in enumerate(normalized))


def colex_unrank(rank: int, subset_size: int, length: int) -> Tuple[int, ...]:
    """Inverse of :func:`colex_rank` for subsets of ``range(length)``."""
    for name, value in (
        ("rank", rank),
        ("subset_size", subset_size),
        ("length", length),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a non-negative integer, got {value!r}.")
    rank = int(rank)
    subset_size = int(subset_size)
    length = int(length)
    if rank < 0 or subset_size < 0 or length < 0:
        raise ValueError(
            "rank, subset_size, and length must all be non-negative, got "
            f"{(rank, subset_size, length)}."
        )
    if subset_size > length:
        raise ValueError(
            f"subset_size={subset_size} exceeds length={length}."
        )
    count = comb(length, subset_size)
    if rank >= count:
        raise ValueError(
            f"rank={rank} is outside [0, {count}) for subsets of size "
            f"{subset_size} in length {length}."
        )
    if subset_size == 0:
        return ()

    remaining = rank
    out = [0] * subset_size
    upper = length - 1
    for i in range(subset_size, 0, -1):
        value = upper
        while value >= i and comb(value, i) > remaining:
            value -= 1
        # ``value == i - 1`` is always valid because C(i - 1, i) = 0.
        out[i - 1] = value
        remaining -= comb(value, i)
        upper = value - 1
    return tuple(out)


def colex_placements(length: int, subset_size: int) -> Tuple[Tuple[int, ...], ...]:
    """All placements ordered by their consecutive colexicographic ranks."""
    if isinstance(length, bool) or not isinstance(length, Integral):
        raise TypeError(f"length must be a non-negative integer, got {length!r}.")
    if isinstance(subset_size, bool) or not isinstance(subset_size, Integral):
        raise TypeError(
            f"subset_size must be a non-negative integer, got {subset_size!r}."
        )
    length = int(length)
    subset_size = int(subset_size)
    if length < 0 or subset_size < 0:
        raise ValueError(
            f"length and subset_size must be non-negative, got {(length, subset_size)}."
        )
    if subset_size > length:
        raise ValueError(f"subset_size={subset_size} exceeds length={length}.")
    return tuple(
        colex_unrank(rank, subset_size, length)
        for rank in range(comb(length, subset_size))
    )


def _base_digit_matrix(base: int, length: int, dtype) -> np.ndarray:
    """Return lexicographically ordered fixed-length base expansions."""
    count = base**length
    if length == 0:
        return np.empty((1, 0), dtype=dtype)
    powers = np.asarray(
        [base**power for power in range(length - 1, -1, -1)],
        dtype=dtype,
    )
    indices = np.arange(count, dtype=dtype)[:, None]
    return np.ascontiguousarray((indices // powers[None, :]) % base)


@dataclass(frozen=True, slots=True, eq=False)
class BigradedGradePlan:
    """Precomputed placement and ordinary-coordinate maps for one bidegree."""

    grade: Bidegree
    placement_count: int
    block_width: int
    dense_shape: Tuple[int, int, int]
    placements: np.ndarray
    block_to_total_indices: np.ndarray

    @property
    def total_degree(self) -> int:
        return self.grade[0] + self.grade[1]

    def memory_bytes(self) -> int:
        return int(
            self.placements.nbytes
            + self.block_to_total_indices.nbytes
        )


@dataclass(frozen=True, slots=True, eq=False)
class BigradedConcatPlan:
    """Factored placement and dense-axis metadata for one block product."""

    left_grade: Bidegree
    right_grade: Bidegree
    output_grade: Bidegree
    placement_offsets: np.ndarray
    dense_axis_permutation: Tuple[int, ...]
    outer_axis_permutation: Tuple[int, ...]
    dense_input_shape: Tuple[int, ...]
    dense_output_shape: Tuple[int, ...]
    left_placement_count: int
    right_placement_count: int
    output_placement_count: int

    def target_ranks(self) -> np.ndarray:
        """Materialize the vectorized ``j1 + offset[j2]`` target-rank grid."""
        left = np.arange(self.left_placement_count, dtype=self.placement_offsets.dtype)
        return left[:, None] + self.placement_offsets[None, :]

    def memory_bytes(self) -> int:
        return int(self.placement_offsets.nbytes)


class BigradedPlanStore:
    """Fixed-capacity host plan store shared by active bidegree layouts.

    Equality and hashing are identity-based.  This permits core/layout objects
    holding a store to be used safely as static JAX arguments later without
    comparing NumPy arrays structurally.
    """

    def __init__(
        self,
        dims: Bidegree,
        max_truncation: Bidegree,
    ) -> None:
        self.dims = _bidegree(dims, name="dims")
        if self.dims[0] <= 0 or self.dims[1] <= 0:
            raise ValueError(f"dims must be strictly positive, got {self.dims}.")
        self.max_truncation = _bidegree(max_truncation, name="max_truncation")
        self._grade_plans: Dict[Bidegree, BigradedGradePlan] = {}
        self._concat_plans: Dict[Tuple[Bidegree, Bidegree], BigradedConcatPlan] = {}
        self._grade_plans_view = MappingProxyType(self._grade_plans)
        self._concat_plans_view = MappingProxyType(self._concat_plans)
        self._active_layouts = {}
        self._layout_lock = RLock()
        self._precompute_grades()
        self._precompute_concatenations()

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
    def grade_plans(self) -> Mapping[Bidegree, BigradedGradePlan]:
        return self._grade_plans_view

    @property
    def concat_plans(
        self,
    ) -> Mapping[Tuple[Bidegree, Bidegree], BigradedConcatPlan]:
        return self._concat_plans_view

    def _precompute_grades(self) -> None:
        spec = BigradedSpec(*self.dims, self.max_truncation)
        digit_tables = {}
        for grade in spec.grades:
            self._grade_plans[grade] = self._build_grade_plan(
                grade,
                digit_tables=digit_tables,
            )

    def _build_grade_plan(
        self,
        grade: Bidegree,
        *,
        digit_tables: dict[tuple[int, int, np.dtype], np.ndarray] | None = None,
    ) -> BigradedGradePlan:
        n, m = grade
        d_prime, d_doubleprime = self.dims
        total = n + m
        placements_tuple = colex_placements(total, n)
        placement_count = len(placements_tuple)
        dense_prime = d_prime**n
        dense_doubleprime = d_doubleprime**m
        block_width = placement_count * dense_prime * dense_doubleprime
        placement_dtype = _index_dtype(max(total, 1))
        placements = _readonly(
            np.asarray(placements_tuple, dtype=placement_dtype).reshape(
                placement_count,
                n,
            )
        )

        full_dimension = (d_prime + d_doubleprime) ** total
        index_dtype = _index_dtype(max(full_dimension - 1, block_width - 1, 0))
        index_dtype = np.dtype(index_dtype)
        if digit_tables is None:
            digit_tables = {}

        def digits(base: int, length: int) -> np.ndarray:
            key = base, length, index_dtype
            table = digit_tables.get(key)
            if table is None:
                table = _base_digit_matrix(base, length, index_dtype)
                digit_tables[key] = table
            return table

        prime_digits = digits(d_prime, n)
        doubleprime_digits = digits(d_doubleprime, m)
        ordinary_base = d_prime + d_doubleprime
        ordinary_weights = np.asarray(
            [ordinary_base**power for power in range(total - 1, -1, -1)],
            dtype=index_dtype,
        )
        prime_mask = np.zeros((placement_count, total), dtype=np.bool_)
        if n:
            prime_mask[
                np.arange(placement_count)[:, None],
                placements,
            ] = True
        prime_weights = ordinary_weights[None, :] * prime_mask
        doubleprime_weights = ordinary_weights[None, :] * ~prime_mask
        prime_weights = prime_weights[prime_mask].reshape(placement_count, n)
        doubleprime_weights = doubleprime_weights[~prime_mask].reshape(
            placement_count,
            m,
        )
        prime_contributions = prime_weights @ prime_digits.T
        doubleprime_contributions = doubleprime_weights @ (
            d_prime + doubleprime_digits
        ).T
        block_to_total = (
            prime_contributions[:, :, None]
            + doubleprime_contributions[:, None, :]
        ).reshape(block_width)

        return BigradedGradePlan(
            grade=grade,
            placement_count=placement_count,
            block_width=block_width,
            dense_shape=(placement_count, dense_prime, dense_doubleprime),
            placements=placements,
            block_to_total_indices=_readonly(block_to_total),
        )

    def _precompute_concatenations(self) -> None:
        N, M = self.max_truncation
        grades = tuple(self._grade_plans)
        for left_grade in grades:
            for right_grade in grades:
                output = (
                    left_grade[0] + right_grade[0],
                    left_grade[1] + right_grade[1],
                )
                if output[0] <= N and output[1] <= M:
                    self._concat_plans[left_grade, right_grade] = (
                        self._build_concat_plan(left_grade, right_grade)
                    )

    def _build_concat_plan(
        self,
        left_grade: Bidegree,
        right_grade: Bidegree,
    ) -> BigradedConcatPlan:
        n1, m1 = left_grade
        n2, m2 = right_grade
        output_grade = n1 + n2, m1 + m2
        left_plan = self._grade_plans[left_grade]
        right_plan = self._grade_plans[right_grade]
        output_plan = self._grade_plans[output_grade]
        l1 = n1 + m1

        offsets = []
        for placement in right_plan.placements:
            offset = 0
            for s, position in enumerate(placement, start=1):
                offset += comb(l1 + int(position), n1 + s)
            offsets.append(offset)
        offset_dtype = _index_dtype(max(output_plan.placement_count - 1, 0))
        placement_offsets = _readonly(np.asarray(offsets, dtype=offset_dtype))

        # Letter axes after an ordinary outer product are ordered
        # prime1, doubleprime1, prime2, doubleprime2.  The output block groups
        # both prime groups before both double-prime groups.
        prime1 = tuple(range(0, n1))
        double1 = tuple(range(n1, n1 + m1))
        prime2 = tuple(range(n1 + m1, n1 + m1 + n2))
        double2 = tuple(range(n1 + m1 + n2, n1 + m1 + n2 + m2))
        dense_axis_permutation = prime1 + prime2 + double1 + double2

        # Same permutation including the two placement axes in the raw layout
        # P1, prime1, double1, P2, prime2, double2.
        p2_axis = 1 + n1 + m1
        outer_prime1 = tuple(range(1, 1 + n1))
        outer_double1 = tuple(range(1 + n1, p2_axis))
        outer_prime2 = tuple(range(p2_axis + 1, p2_axis + 1 + n2))
        outer_double2 = tuple(
            range(p2_axis + 1 + n2, p2_axis + 1 + n2 + m2)
        )
        outer_axis_permutation = (
            (0, p2_axis)
            + outer_prime1
            + outer_prime2
            + outer_double1
            + outer_double2
        )

        d_prime, d_doubleprime = self.dims
        dense_input_shape = (
            (d_prime,) * n1
            + (d_doubleprime,) * m1
            + (d_prime,) * n2
            + (d_doubleprime,) * m2
        )
        dense_output_shape = (
            (d_prime,) * (n1 + n2) + (d_doubleprime,) * (m1 + m2)
        )
        return BigradedConcatPlan(
            left_grade=left_grade,
            right_grade=right_grade,
            output_grade=output_grade,
            placement_offsets=placement_offsets,
            dense_axis_permutation=dense_axis_permutation,
            outer_axis_permutation=outer_axis_permutation,
            dense_input_shape=dense_input_shape,
            dense_output_shape=dense_output_shape,
            left_placement_count=left_plan.placement_count,
            right_placement_count=right_plan.placement_count,
            output_placement_count=output_plan.placement_count,
        )

    def grade_plan(self, grade: object) -> BigradedGradePlan:
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
    ) -> BigradedConcatPlan:
        key = _bidegree(left_grade), _bidegree(right_grade)
        try:
            return self._concat_plans[key]
        except KeyError as exc:
            output = (key[0][0] + key[1][0], key[0][1] + key[1][1])
            raise KeyError(
                f"concatenation output {output} exceeds plan-store capacity "
                f"{self.max_truncation}."
            ) from exc

    def resolve(
        self,
        truncation: Bidegree | None = None,
        *,
        include_scalar: bool = True,
        coordinates: str = "standard",
    ):
        """Return a cached active layout containing no capacity-only grades."""
        from tensordev.core.bigraded.layout import _build_active_layout

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
                representation="ordered",
            )
            self._active_layouts[key] = layout
            return layout

    def memory_bytes_by_category(self) -> Mapping[str, int]:
        """Host NumPy payload by plan family, excluding Python object overhead."""
        placement_bytes = sum(
            plan.placements.nbytes for plan in self._grade_plans.values()
        )
        conversion_bytes = sum(
            plan.block_to_total_indices.nbytes
            for plan in self._grade_plans.values()
        )
        concatenation_bytes = sum(
            plan.placement_offsets.nbytes for plan in self._concat_plans.values()
        )
        return {
            "placements": int(placement_bytes),
            "conversion": int(conversion_bytes),
            "concatenation": int(concatenation_bytes),
        }

    def memory_bytes(self) -> int:
        return sum(self.memory_bytes_by_category().values())

    def memory_mb(self) -> float:
        return self.memory_bytes() / 1024**2

    def plan_statistics(self) -> Mapping[str, object]:
        categories = self.memory_bytes_by_category()
        return {
            "dims": self.dims,
            "max_truncation": self.max_truncation,
            "grade_count": len(self._grade_plans),
            "concat_plan_count": len(self._concat_plans),
            "active_layout_count": len(self._active_layouts),
            "bytes_by_category": categories,
            "memory_bytes": sum(categories.values()),
            "memory_mb": sum(categories.values()) / 1024**2,
        }
