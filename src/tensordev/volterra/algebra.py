"""Static grading adapter used by native Volterra implementations.

This module contains no Volterra coefficient or time-stepping mathematics.
It resolves a finite algebra layout once, packages exact static grade
worksets, and delegates all coordinate-dependent numerical actions to the
selected tensor core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from numbers import Integral
from types import MappingProxyType
from typing import Any, Callable, Hashable, Mapping, Sequence

from tensordev._backend import _resolve_seq_core, get_default_core_pair


Grade = Hashable
BlockTuple = tuple[Any, ...]


def resolve_volterra_core_pair(
        core: Any = None,
        seq_core: Any = None,
) -> tuple[Any, Any]:
    """Resolve one coherent JAX algebra/sequential-core pair.

    Volterra entry points share this boundary helper so an explicit
    sequential core is never discarded and no inner numerical routine reads
    the mutable process default again.
    """
    default_core, default_seq_core = get_default_core_pair()
    if core is None:
        core = default_core
        if seq_core is None:
            seq_core = default_seq_core
    elif seq_core is None:
        seq_core = (
            default_seq_core
            if core is default_core
            else _resolve_seq_core(core, None)
        )

    core_backend = getattr(core, "backend", None)
    seq_backend = getattr(seq_core, "backend", None)
    if core_backend != "jax" or seq_backend != "jax":
        raise TypeError(
            "Volterra algorithms require compatible JAX algebra and "
            "sequential cores; got algebra backend "
            f"{core_backend!r} and sequential backend {seq_backend!r}."
        )
    if not callable(getattr(seq_core, "supports", None)):
        raise TypeError(
            f"{type(seq_core).__name__} does not expose the sequential "
            "capability protocol required by Volterra algorithms."
        )
    return core, seq_core


def require_volterra_shuffle(
        algebra: "ResolvedVolterraAlgebra",
        *,
        feature: str,
) -> None:
    """Require native shuffle plans exactly when positive depth exceeds one."""
    if algebra.max_order > 1 and not algebra.supports("shuffle"):
        raise RuntimeError(
            f"{feature} requires shuffle-generator plans above depth one; "
            "configure the selected core with "
            "precompute_shuffle='generator' or precompute_shuffle=True."
        )


@lru_cache(maxsize=None)
def _grade_indices(grades: tuple[Grade, ...]) -> Mapping[Grade, int]:
    return MappingProxyType({grade: index for index, grade in enumerate(grades)})


@dataclass(frozen=True, slots=True)
class GradeWorkset:
    """One exact, immutable static grade schedule with dynamic block values.

    ``GradeWorkset`` itself contains no arrays.  Algorithms carry a separate
    tuple of values in ``grades`` order, so changing a workset never changes a
    runtime PyTree structure.
    """

    grades: tuple[Grade, ...]
    widths: tuple[int, ...]
    offsets: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        grades = tuple(self.grades)
        widths = tuple(int(width) for width in self.widths)
        if len(grades) != len(widths):
            raise ValueError("grades and widths must have equal length.")
        if len(set(grades)) != len(grades):
            raise ValueError("workset grades must be unique.")
        if any(width <= 0 for width in widths):
            raise ValueError("workset block widths must be positive.")
        offsets = [0]
        for width in widths:
            offsets.append(offsets[-1] + width)
        object.__setattr__(self, "grades", grades)
        object.__setattr__(self, "widths", widths)
        object.__setattr__(self, "offsets", tuple(offsets))

    @property
    def index_by_grade(self) -> Mapping[Grade, int]:
        return _grade_indices(self.grades)

    @property
    def size(self) -> int:
        return len(self.grades)

    @property
    def packed_width(self) -> int:
        return self.offsets[-1]

    def contains(self, grade: Grade) -> bool:
        return grade in self.index_by_grade

    def index(self, grade: Grade) -> int:
        try:
            return self.index_by_grade[grade]
        except KeyError:
            raise KeyError(f"grade {grade!r} is not in this workset.") from None

    def validate_values(self, values: Sequence[Any]) -> BlockTuple:
        values = tuple(values)
        if len(values) != self.size:
            raise ValueError(
                f"workset has {self.size} grades but received {len(values)} blocks."
            )
        for grade, width, value in zip(self.grades, self.widths, values):
            if value.shape[-1] != width:
                raise ValueError(
                    f"block {grade!r} has width {value.shape[-1]}, expected {width}."
                )
        return values

    def block(self, values: Sequence[Any], grade: Grade) -> Any:
        values = self.validate_values(values)
        return values[self.index(grade)]

    def map(
            self,
            function: Callable[..., Any],
            values: Sequence[Any],
            *rest: Sequence[Any],
    ) -> BlockTuple:
        values = self.validate_values(values)
        others = tuple(self.validate_values(other) for other in rest)
        return tuple(
            function(values[index], *(other[index] for other in others))
            for index in range(self.size)
        )

    def add(self, left: Sequence[Any], right: Sequence[Any]) -> BlockTuple:
        return self.map(lambda a, b: a + b, left, right)

    def pack(self, xp: Any, values: Sequence[Any]) -> Any:
        """Pack retained native blocks along their coordinate axis.

        A one-block workset returns its input array by identity, preserving
        total-degree lowering semantics.
        """
        values = self.validate_values(values)
        if not values:
            raise ValueError("cannot pack an empty workset.")
        if len(values) == 1:
            return values[0]
        batch = xp.broadcast_shapes(*(value.shape[:-1] for value in values))
        broadcast = tuple(
            xp.broadcast_to(value, batch + (width,))
            for value, width in zip(values, self.widths)
        )
        return xp.concat(broadcast, axis=-1)

    def split(self, packed: Any) -> BlockTuple:
        """Split a packed native diagonal at static coordinate offsets."""
        if packed.shape[-1] != self.packed_width:
            raise ValueError(
                f"packed width {packed.shape[-1]} does not match expected "
                f"{self.packed_width}."
            )
        if self.size == 0:
            raise ValueError("cannot split an empty workset.")
        if self.size == 1:
            return (packed,)
        return tuple(
            packed[..., start:stop]
            for start, stop in zip(self.offsets[:-1], self.offsets[1:])
        )


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedVolterraAlgebra:
    """Finite static view of one compatible tensor core and truncation."""

    core: Any
    truncation: Any
    alphabet_dim: int
    layout: Any
    positive_layout: Any
    max_order: int
    grades: tuple[Grade, ...]
    positive_grades: tuple[Grade, ...]
    grades_by_total_order: tuple[tuple[Grade, ...], ...]
    first_level_grades: tuple[Grade, ...]
    diagonal_offsets_by_total_order: tuple[tuple[int, ...], ...]
    _diagonals: tuple[GradeWorkset, ...] = field(repr=False)

    __hash__ = object.__hash__

    @property
    def zero_grade(self) -> Grade:
        return self.layout.zero_grade

    @property
    def backend(self) -> Any:
        return getattr(self.core, "backend", None)

    def supports(self, capability: str) -> bool:
        provider = getattr(self.core, "supports", None)
        return bool(provider(capability)) if callable(provider) else False

    def require(self, capability: str) -> None:
        if not self.supports(capability):
            raise RuntimeError(
                f"{type(self.core).__name__} does not provide the required "
                f"{capability!r} algebra capability."
            )

    def block_width(self, grade: Grade) -> int:
        return self.core._block_width_for_layout(
            self.layout, grade, alphabet_dim=self.alphabet_dim
        )

    def block(self, element: Any, grade: Grade, *, positive: bool = False) -> Any:
        layout = self.positive_layout if positive else self.layout
        return self.core._element_block(element, grade, layout=layout)

    def assemble(self, blocks: Sequence[Any], *, positive: bool = False) -> Any:
        layout = self.positive_layout if positive else self.layout
        return self.core._assemble_element(tuple(blocks), layout)

    def zero_block(
            self,
            grade: Grade,
            *,
            batch_shape: tuple[int, ...],
            dtype: Any,
    ) -> Any:
        return self.core._zero_block_for_layout(
            self.layout,
            grade,
            batch_shape=tuple(batch_shape),
            dtype=dtype,
            alphabet_dim=self.alphabet_dim,
        )

    def constant(
            self,
            *,
            batch_shape: tuple[int, ...],
            dtype: Any,
            scalar: float = 0.0,
            positive: bool = False,
    ) -> Any:
        layout = self.positive_layout if positive else self.layout
        return self.core._constant_element_for_layout(
            layout,
            batch_shape=tuple(batch_shape),
            dtype=dtype,
            alphabet_dim=self.alphabet_dim,
            scalar=scalar,
        )

    def zero(self, *, batch_shape: tuple[int, ...], dtype: Any) -> Any:
        return self.constant(batch_shape=batch_shape, dtype=dtype, scalar=0.0)

    def unit(self, *, batch_shape: tuple[int, ...], dtype: Any) -> Any:
        return self.constant(batch_shape=batch_shape, dtype=dtype, scalar=1.0)

    def workset(self, grades: Sequence[Grade]) -> GradeWorkset:
        grades = tuple(grades)
        outside = tuple(grade for grade in grades if not self.layout.contains(grade))
        if outside:
            raise ValueError(f"workset contains inactive grades: {outside}.")
        return GradeWorkset(
            grades=grades,
            widths=tuple(self.block_width(grade) for grade in grades),
        )

    def diagonal(self, total_order: int) -> GradeWorkset:
        if isinstance(total_order, bool) or not isinstance(total_order, Integral):
            raise TypeError("total_order must be an integer.")
        total_order = int(total_order)
        if total_order < 0 or total_order > self.max_order:
            raise ValueError(
                f"total_order must lie in 0..{self.max_order}, got {total_order}."
            )
        return self._diagonals[total_order]

    def pack_diagonal(self, total_order: int, values: Sequence[Any]) -> Any:
        return self.diagonal(total_order).pack(self.core.xp, values)

    def split_diagonal(self, total_order: int, packed: Any) -> BlockTuple:
        return self.diagonal(total_order).split(packed)

    def generator_blocks(self, z: Any) -> tuple[tuple[Grade, Any], ...]:
        blocks = tuple(self.core._generator_blocks(z, layout=self.layout))
        block_grades = tuple(grade for grade, _ in blocks)
        if len(set(block_grades)) != len(block_grades):
            raise ValueError("the core returned duplicate generator grades.")
        active = set(self.first_level_grades)
        selected = tuple((grade, block) for grade, block in blocks if grade in active)
        selected_grades = tuple(grade for grade, _ in selected)
        if set(selected_grades) != active:
            raise ValueError(
                "the core generator decomposition does not cover every active "
                "first-level grade."
            )
        for grade, block in selected:
            expected = self.block_width(grade)
            if block.ndim == 0 or block.shape[-1] != expected:
                raise ValueError(
                    f"generator block {grade!r} has width "
                    f"{None if block.ndim == 0 else block.shape[-1]}, "
                    f"expected {expected}."
                )
        return selected

    def generator_splits(self, output_grade: Grade) -> tuple[tuple[Grade, Grade], ...]:
        if not self.layout.contains(output_grade):
            raise KeyError(f"output grade {output_grade!r} is not active.")
        first = set(self.first_level_grades)
        return tuple(
            (source_grade, generator_grade)
            for source_grade, generator_grade in self.layout.product_splits(output_grade)
            if generator_grade in first
        )

    def _generator_action_inputs(
            self,
            source_workset: GradeWorkset,
            source_values: Sequence[Any],
            z: Any,
            output_grade: Grade,
    ) -> tuple[BlockTuple, BlockTuple, tuple[Grade, ...], tuple[Grade, ...]]:
        source_values = source_workset.validate_values(source_values)
        generator_by_grade = dict(self.generator_blocks(z))
        splits = self.generator_splits(output_grade)
        missing = tuple(
            source_grade
            for source_grade, _ in splits
            if not source_workset.contains(source_grade)
        )
        if missing:
            raise KeyError(
                f"source workset is missing generator predecessors {missing} "
                f"for output grade {output_grade!r}."
            )
        predecessor_grades = tuple(source_grade for source_grade, _ in splits)
        generator_grades = tuple(generator_grade for _, generator_grade in splits)
        predecessor_blocks = tuple(
            source_workset.block(source_values, grade)
            for grade in predecessor_grades
        )
        generator_blocks = tuple(generator_by_grade[grade] for grade in generator_grades)
        return (
            predecessor_blocks,
            generator_blocks,
            predecessor_grades,
            generator_grades,
        )

    def right_generator_output_block(
            self,
            source_workset: GradeWorkset,
            source_values: Sequence[Any],
            z: Any,
            output_grade: Grade,
    ) -> Any:
        arrays = self._generator_action_inputs(
            source_workset, source_values, z, output_grade
        )
        return self.core._right_multiply_generator_output_block(
            arrays[0],
            arrays[1],
            predecessor_grades=arrays[2],
            generator_grades=arrays[3],
            output_grade=output_grade,
        )

    def shuffle_generator_output_block(
            self,
            source_workset: GradeWorkset,
            source_values: Sequence[Any],
            z: Any,
            output_grade: Grade,
    ) -> Any:
        self.require("shuffle")
        arrays = self._generator_action_inputs(
            source_workset, source_values, z, output_grade
        )
        return self.core._shuffle_generator_output_block(
            arrays[0],
            arrays[1],
            predecessor_grades=arrays[2],
            generator_grades=arrays[3],
            output_grade=output_grade,
        )

    def right_generator_action(
            self,
            source_workset: GradeWorkset,
            source_values: Sequence[Any],
            z: Any,
            target_workset: GradeWorkset,
    ) -> BlockTuple:
        return tuple(
            self.right_generator_output_block(
                source_workset, source_values, z, output_grade
            )
            for output_grade in target_workset.grades
        )

    def shuffle_generator_action(
            self,
            source_workset: GradeWorkset,
            source_values: Sequence[Any],
            z: Any,
            target_workset: GradeWorkset,
    ) -> BlockTuple:
        return tuple(
            self.shuffle_generator_output_block(
                source_workset, source_values, z, output_grade
            )
            for output_grade in target_workset.grades
        )

    def product_output_block(
            self,
            contributions: Sequence[tuple[Any, Any, Grade, Grade]],
            output_grade: Grade,
    ) -> Any:
        """Combine all graded-product contributions to one output block.

        The core hook is coordinate-native: standard cores evaluate ordinary
        concatenation, while transported-coordinate cores evaluate their
        native group law.  The Universal default preserves deterministic
        homogeneous-product and summation order.
        """
        contributions = tuple(contributions)
        if not contributions:
            raise ValueError(
                f"no product contribution for output grade {output_grade!r}."
            )
        active_splits = set(self.layout.product_splits(output_grade))
        for _left, _right, left_grade, right_grade in contributions:
            if (left_grade, right_grade) not in active_splits:
                raise ValueError(
                    f"grades {left_grade!r} and {right_grade!r} do not form "
                    f"an active split of output grade {output_grade!r}."
                )

        return self.core._product_output_block(contributions, output_grade)

    def embed_positive(self, element: Any) -> Any:
        """Add an explicit zero scalar block to a positive-only element."""
        if not self.positive_grades:
            raise ValueError("cannot infer batch metadata from an empty positive element.")
        first = self.block(element, self.positive_grades[0], positive=True)
        scalar = self.zero_block(
            self.zero_grade,
            batch_shape=tuple(first.shape[:-1]),
            dtype=first.dtype,
        )
        blocks = tuple(
            scalar if grade == self.zero_grade else self.block(element, grade, positive=True)
            for grade in self.grades
        )
        return self.assemble(blocks)


def _build_resolved(core: Any, truncation: Any, alphabet_dim: int) -> ResolvedVolterraAlgebra:
    if getattr(core, "backend", None) != "jax":
        raise TypeError(
            "Volterra algebra resolution requires a JAX tensor core; "
            f"got backend {getattr(core, 'backend', None)!r}."
        )
    for capability in ("concatenation", "generator_action"):
        provider = getattr(core, "supports", None)
        if not callable(provider) or not provider(capability):
            raise TypeError(
                f"{type(core).__name__} lacks required capability {capability!r}."
            )
    if not callable(getattr(core, "_product_output_block", None)):
        raise TypeError(
            f"{type(core).__name__} lacks the required coordinate-native "
            "_product_output_block core hook."
        )
    alphabet_dim = core._validate_alphabet_dim(alphabet_dim)
    layout = core.resolve_layout(truncation, include_scalar=True)
    positive_layout = core.resolve_layout(truncation, include_scalar=False)
    grades = tuple(layout.grades)
    positive_grades = tuple(positive_layout.grades)
    if not grades or grades[0] != layout.zero_grade:
        raise ValueError("the scalar-including layout must begin with its zero grade.")
    if positive_grades != tuple(grade for grade in grades if grade != layout.zero_grade):
        raise ValueError("full and positive layouts disagree on their active grades.")
    max_order = max(layout.total_degree(grade) for grade in grades)
    if max_order <= 0:
        raise ValueError("Volterra truncation must retain at least one positive grade.")
    grades_by_total = tuple(
        tuple(grade for grade in grades if layout.total_degree(grade) == order)
        for order in range(max_order + 1)
    )
    if any(not diagonal for diagonal in grades_by_total):
        raise ValueError("the active grading has a missing total-order diagonal.")
    first_level = grades_by_total[1]

    # Every retained output grade must admit every local total order used by
    # the Volterra FFT source schedule.  This is static and inexpensive.
    for output_grade in grades:
        output_order = layout.total_degree(output_grade)
        splits = tuple(layout.product_splits(output_grade))
        for local_order in range(output_order + 1):
            if not any(
                layout.contains(left)
                and layout.contains(right)
                and layout.total_degree(right) == local_order
                for left, right in splits
            ):
                raise ValueError(
                    f"grade {output_grade!r} has no product split with right "
                    f"total order {local_order}."
                )

    diagonals = tuple(
        GradeWorkset(
            diagonal,
            tuple(
                core._block_width_for_layout(
                    layout, grade, alphabet_dim=alphabet_dim
                )
                for grade in diagonal
            ),
        )
        for diagonal in grades_by_total
    )
    return ResolvedVolterraAlgebra(
        core=core,
        truncation=truncation,
        alphabet_dim=alphabet_dim,
        layout=layout,
        positive_layout=positive_layout,
        max_order=max_order,
        grades=grades,
        positive_grades=positive_grades,
        grades_by_total_order=grades_by_total,
        first_level_grades=first_level,
        diagonal_offsets_by_total_order=tuple(
            diagonal.offsets for diagonal in diagonals
        ),
        _diagonals=diagonals,
    )


def resolve_volterra_algebra(
        core: Any,
        trunc: Any,
        alphabet_dim: int,
) -> ResolvedVolterraAlgebra:
    """Resolve and identity-cache one finite native Volterra algebra view."""
    truncation = core.normalize_truncation(trunc)
    alphabet_dim = core._validate_alphabet_dim(alphabet_dim)
    key = (truncation, alphabet_dim)
    cache = getattr(core, "_resolved_volterra_algebras", None)
    if cache is None:
        try:
            cache = {}
            setattr(core, "_resolved_volterra_algebras", cache)
        except (AttributeError, TypeError):
            cache = None
    if cache is not None:
        try:
            cached = cache.get(key)
        except TypeError:
            cached = None
        if cached is not None:
            return cached
    resolved = _build_resolved(core, truncation, alphabet_dim)
    if cache is not None:
        try:
            cache[key] = resolved
        except TypeError:
            pass
    return resolved


__all__ = [
    "GradeWorkset",
    "ResolvedVolterraAlgebra",
    "resolve_volterra_algebra",
    "resolve_volterra_core_pair",
    "require_volterra_shuffle",
]
