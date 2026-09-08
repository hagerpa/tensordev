"""PyTree tensor container and static metadata for bidegrees."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb
from numbers import Integral
from typing import Any, Iterator, Optional, Sequence, Tuple

import jax
import numpy as np


Bidegree = Tuple[int, int]


def _integer(value: object, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        qualifier = "positive" if positive else "non-negative"
        raise TypeError(f"{name} must be a {qualifier} integer, got {value!r}.")
    result = int(value)
    lower = 1 if positive else 0
    if result < lower:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}, got {result}.")
    return result


def _bidegree(value: object, *, name: str = "grade") -> Bidegree:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be a pair of non-negative integers, got {value!r}.")
    return (
        _integer(value[0], name=f"{name}[0]"),
        _integer(value[1], name=f"{name}[1]"),
    )


def _prefix_slice_degree(value: slice, *, name: str, maximum: int) -> int:
    """Resolve one upper-exclusive prefix slice to its largest active degree."""
    start = value.start
    if start is not None:
        if isinstance(start, bool) or not isinstance(start, Integral):
            raise TypeError(
                f"{name} slice start must be 0 or None, got {start!r}."
            )
        if int(start) != 0:
            raise ValueError(
                f"{name} slice must start at 0; only rectangular prefixes "
                "such as A[:n, :m] are representable."
            )

    step = value.step
    if step is not None:
        if isinstance(step, bool) or not isinstance(step, Integral):
            raise TypeError(
                f"{name} slice step must be 1 or None, got {step!r}."
            )
        if int(step) != 1:
            raise ValueError(
                f"{name} slice step must be 1; only rectangular prefixes "
                "such as A[:n, :m] are representable."
            )

    stop = value.stop
    if stop is None:
        return maximum
    if isinstance(stop, bool) or not isinstance(stop, Integral):
        raise TypeError(
            f"{name} slice stop must be a positive integer or None, got {stop!r}."
        )
    stop = int(stop)
    if stop <= 0:
        raise ValueError(
            f"{name} slice stop must be positive, got {stop}; "
            "an empty bidegree axis is not representable."
        )
    return min(stop - 1, maximum)


@lru_cache(maxsize=None)
def _canonical_grades(
    truncation: Bidegree,
    include_scalar: bool,
) -> Tuple[Bidegree, ...]:
    """Grades in increasing total degree, then decreasing prime degree."""
    N, M = truncation
    out = []
    for total in range(N + M + 1):
        n_max = min(N, total)
        n_min = max(0, total - M)
        for n in range(n_max, n_min - 1, -1):
            grade = (n, total - n)
            if include_scalar or grade != (0, 0):
                out.append(grade)
    return tuple(out)


@lru_cache(maxsize=None)
def _canonical_grade_indices(
    truncation: Bidegree,
    include_scalar: bool,
) -> dict[Bidegree, int]:
    return {
        grade: index
        for index, grade in enumerate(_canonical_grades(truncation, include_scalar))
    }


@dataclass(frozen=True, slots=True)
class BigradedSpec:
    """Small, hashable metadata describing one active bigraded tensor layout.

    The spec is suitable as JAX PyTree auxiliary data.  It intentionally holds
    no NumPy/JAX arrays and no reference to a capacity-level plan store.
    """

    d_prime: int
    d_doubleprime: int
    truncation: Bidegree
    coordinates: str = "standard"
    include_scalar: bool = True
    partially_symmetrized: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "d_prime",
            _integer(self.d_prime, name="d_prime", positive=True),
        )
        object.__setattr__(
            self,
            "d_doubleprime",
            _integer(self.d_doubleprime, name="d_doubleprime", positive=True),
        )
        object.__setattr__(
            self,
            "truncation",
            _bidegree(self.truncation, name="truncation"),
        )
        if self.coordinates not in {"standard", "shear"}:
            raise ValueError(
                "coordinates must be either 'standard' or 'shear', "
                f"got {self.coordinates!r}."
            )
        if not isinstance(self.include_scalar, bool):
            raise TypeError(
                "include_scalar must be a bool, "
                f"got {type(self.include_scalar).__name__}."
            )
        if not isinstance(self.partially_symmetrized, bool):
            raise TypeError(
                "partially_symmetrized must be a bool, got "
                f"{type(self.partially_symmetrized).__name__}."
            )

    @property
    def dims(self) -> Bidegree:
        return self.d_prime, self.d_doubleprime

    @property
    def grades(self) -> Tuple[Bidegree, ...]:
        return _canonical_grades(self.truncation, self.include_scalar)

    @property
    def size(self) -> int:
        return len(self.grades)

    def contains(self, grade: object) -> bool:
        try:
            n, m = _bidegree(grade)
        except (TypeError, ValueError):
            return False
        N, M = self.truncation
        return n <= N and m <= M and (self.include_scalar or (n, m) != (0, 0))

    def index(self, grade: object) -> int:
        normalized = _bidegree(grade)
        if not self.contains(normalized):
            raise KeyError(
                f"bidegree {normalized} is not present in active truncation "
                f"{self.truncation} (include_scalar={self.include_scalar})."
            )
        # Keep lookup tables in a process-local cache rather than in the spec,
        # so PyTree auxiliary equality/hash stays inexpensive.
        return _canonical_grade_indices(
            self.truncation,
            self.include_scalar,
        )[normalized]

    @staticmethod
    def total_degree(grade: object) -> int:
        n, m = _bidegree(grade)
        return n + m

    @staticmethod
    def placement_count(grade: object) -> int:
        """Number of ordinary prime-letter placements at ``grade``."""
        n, m = _bidegree(grade)
        return comb(n + m, n)

    def rank_count(self, grade: object) -> int:
        """Number of stored rank coordinates at ``grade``."""
        n, m = _bidegree(grade)
        if not self.partially_symmetrized:
            return self.placement_count((n, m))
        parts = (n + 1) * self.d_doubleprime
        return comb(m + parts - 1, m)

    def block_width(self, grade: object) -> int:
        n, m = _bidegree(grade)
        if not self.contains((n, m)):
            raise KeyError(
                f"bidegree {(n, m)} is not present in active truncation "
                f"{self.truncation}."
            )
        dense_width = self.d_prime**n
        if not self.partially_symmetrized:
            dense_width *= self.d_doubleprime**m
        return self.rank_count((n, m)) * dense_width

    def with_scalar(self, include_scalar: bool) -> "BigradedSpec":
        return BigradedSpec(
            self.d_prime,
            self.d_doubleprime,
            self.truncation,
            coordinates=self.coordinates,
            include_scalar=include_scalar,
            partially_symmetrized=self.partially_symmetrized,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True, eq=False, repr=False)
class BigradedTensor:
    """Ragged array container with one dynamic PyTree leaf per bidegree.

    Every block uses the package-wide ``batch + (coordinates,)`` convention.
    The block at ``(n, m)`` has the final width prescribed by ``spec``.
    """

    blocks: Tuple[Any, ...]
    spec: BigradedSpec

    def __post_init__(self) -> None:
        if not isinstance(self.spec, BigradedSpec):
            raise TypeError(f"spec must be a BigradedSpec, got {type(self.spec).__name__}.")
        blocks = tuple(self.blocks)
        object.__setattr__(self, "blocks", blocks)
        if len(blocks) != self.spec.size:
            raise ValueError(
                f"expected {self.spec.size} blocks for grades {self.spec.grades}, "
                f"got {len(blocks)}."
            )

        batch_shape: Optional[Tuple[int, ...]] = None
        dtype: Optional[np.dtype] = None
        for grade, block in zip(self.spec.grades, blocks):
            if not hasattr(block, "shape") or not hasattr(block, "dtype"):
                raise TypeError(
                    f"block {grade} must be an array with shape and dtype, "
                    f"got {type(block).__name__}."
                )
            shape = tuple(block.shape)
            if len(shape) == 0:
                raise ValueError(
                    f"block {grade} must have a trailing coordinate axis, got shape {shape}."
                )
            expected_width = self.spec.block_width(grade)
            if shape[-1] != expected_width:
                raise ValueError(
                    f"block {grade} has final width {shape[-1]}, expected "
                    f"{expected_width} for dimensions {self.spec.dims}."
                )
            current_batch = shape[:-1]
            if batch_shape is None:
                batch_shape = current_batch
            elif current_batch != batch_shape:
                raise ValueError(
                    "all blocks must have the same leading batch shape; "
                    f"expected {batch_shape}, got {current_batch} at grade {grade}."
                )
            current_dtype = np.dtype(block.dtype)
            if dtype is None:
                dtype = current_dtype
            elif current_dtype != dtype:
                raise TypeError(
                    "all blocks must have the same dtype; "
                    f"expected {dtype}, got {current_dtype} at grade {grade}."
                )

    def tree_flatten(self):
        return self.blocks, self.spec

    @classmethod
    def tree_unflatten(
        cls,
        spec: BigradedSpec,
        blocks: Sequence[Any],
    ) -> "BigradedTensor":
        return cls(tuple(blocks), spec)

    def __repr__(self) -> str:
        return (
            f"BigradedTensor(grades={self.grades}, batch_shape={self.batch_shape}, "
            f"dtype={self.dtype})"
        )

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.blocks)

    def __getitem__(self, grade: object) -> Any:
        if isinstance(grade, slice):
            raise TypeError(
                "structural bidegree slicing requires two prefix slices, "
                "for example A[:n, :m]."
            )
        if (
            isinstance(grade, (tuple, list))
            and any(isinstance(component, slice) for component in grade)
        ):
            if len(grade) != 2 or not all(
                isinstance(component, slice) for component in grade
            ):
                raise TypeError(
                    "structural bidegree slicing requires two prefix slices, "
                    "for example A[:n, :m]; use A[n, m] to select one block."
                )
            N, M = self.truncation
            truncation = (
                _prefix_slice_degree(
                    grade[0],
                    name="first bidegree component",
                    maximum=N,
                ),
                _prefix_slice_degree(
                    grade[1],
                    name="second bidegree component",
                    maximum=M,
                ),
            )
            if truncation == self.truncation:
                return self
            spec = BigradedSpec(
                self.spec.d_prime,
                self.spec.d_doubleprime,
                truncation,
                coordinates=self.spec.coordinates,
                include_scalar=self.spec.include_scalar,
                partially_symmetrized=self.spec.partially_symmetrized,
            )
            blocks = tuple(
                self.blocks[self.spec.index(active_grade)]
                for active_grade in spec.grades
            )
            return BigradedTensor(blocks, spec)
        return self.blocks[self.spec.index(grade)]

    def block(self, n: int, m: int) -> Any:
        return self[n, m]

    @property
    def grades(self) -> Tuple[Bidegree, ...]:
        return self.spec.grades

    @property
    def truncation(self) -> Bidegree:
        return self.spec.truncation

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        if not self.blocks:
            return ()
        return tuple(self.blocks[0].shape[:-1])

    @property
    def dtype(self) -> Optional[np.dtype]:
        if not self.blocks:
            return None
        return np.dtype(self.blocks[0].dtype)

    def with_blocks(self, blocks: Sequence[Any]) -> "BigradedTensor":
        return BigradedTensor(tuple(blocks), self.spec)
