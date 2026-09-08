"""Static grading schedules shared by tensor-algebra cores.

The numerical kernels in :mod:`tensordev.core` deliberately operate on one
homogeneous block at a time.  This module contains the small amount of Python
planning needed to combine those kernels into graded operations.  Grade loops
therefore run while a JAX function is traced; no grade-dispatch arrays or
dynamic conditionals are introduced into compiled programs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Hashable, Iterable, Protocol, TypeVar


Grade = TypeVar("Grade", bound=Hashable)
Block = TypeVar("Block")


@dataclass(frozen=True, slots=True)
class GradedMapSchedule:
    """Static layout adapter for a componentwise graded operation."""

    grades: tuple[Any, ...]
    block: Callable[[Any], Any]
    assemble: Callable[[tuple[Any, ...]], Any]


@dataclass(frozen=True, slots=True)
class GradedSummationSchedule:
    """Static layout adapter consumed by ``graded_summation``."""

    grades: tuple[Any, ...]
    left_contains: Callable[[Any], bool]
    right_contains: Callable[[Any], bool]
    left_block: Callable[[Any], Any]
    right_block: Callable[[Any], Any]
    assemble: Callable[[tuple[Any, ...]], Any]
    add: Callable[[Any, Any, Any], Any] | None = None
    left_only: Callable[[Any, Any], Any] | None = None
    right_only: Callable[[Any, Any], Any] | None = None
    zero_block: Callable[[Any], Any] | None = None


@dataclass(frozen=True, slots=True)
class GradedConvolutionSchedule:
    """Static layout adapter consumed by ``graded_convolution``."""

    grades: tuple[Any, ...]
    splits: Callable[[Any], Iterable[tuple[Any, Any]]]
    left_contains: Callable[[Any], bool]
    right_contains: Callable[[Any], bool]
    left_block: Callable[[Any], Any]
    right_block: Callable[[Any], Any]
    assemble: Callable[[tuple[Any, ...]], Any]
    zero_block: Callable[[Any], Any] | None = None
    product_grade: Callable[[tuple[Any, ...], Any], Any] | None = None


@dataclass(frozen=True, slots=True)
class GradedContractionSchedule:
    """Static layout adapter consumed by ``graded_contraction``."""

    grades: tuple[Any, ...]
    pairs: Callable[[Any], Iterable[tuple[Any, Any]]]
    multiplier_block: Callable[[Any], Any]
    target_block: Callable[[Any], Any]
    assemble: Callable[[tuple[Any, ...]], Any]
    zero_block: Callable[[Any], Any] | None = None
    contract_block: Callable[[Any, Any, Any, Any, Any], Any] | None = None


@dataclass(frozen=True, slots=True)
class GradedInnerProductSchedule:
    """Static layout adapter consumed by ``graded_inner_product``."""

    grades: tuple[Any, ...]
    left_block: Callable[[Any], Any]
    right_block: Callable[[Any], Any]
    zero: Callable[[], Any]


class ResolvedGradingLayout(Protocol[Grade]):
    """Finite, immutable grade schedule consumed by shared algorithms."""

    truncation: object
    grades: tuple[Grade, ...]
    zero_grade: Grade

    def index(self, grade: Grade) -> int: ...

    def contains(self, grade: Grade) -> bool: ...

    def total_degree(self, grade: Grade) -> int: ...

    def block_width(self, grade: Grade) -> int: ...

    def product_splits(self, output_grade: Grade) -> tuple[tuple[Grade, Grade], ...]: ...


@dataclass(frozen=True, slots=True)
class TotalDegreeLayout:
    """Resolved layout for an unbounded total-degree core."""

    truncation: int
    include_scalar: bool = True

    def __post_init__(self) -> None:
        if self.truncation < 0:
            raise ValueError("truncation must be non-negative")

    @property
    def zero_grade(self) -> int:
        return 0

    @property
    def grades(self) -> tuple[int, ...]:
        start = 0 if self.include_scalar else 1
        return tuple(range(start, self.truncation + 1))

    def index(self, grade: int) -> int:
        if not self.contains(grade):
            raise KeyError(grade)
        return grade if self.include_scalar else grade - 1

    def contains(self, grade: int) -> bool:
        start = 0 if self.include_scalar else 1
        return start <= grade <= self.truncation

    def total_degree(self, grade: int) -> int:
        return grade

    def block_width(self, grade: int) -> int:
        # The total-degree core is dimension-free until it sees an element, so
        # there is intentionally no width available from this layout.
        raise NotImplementedError("total-degree block width requires the alphabet dimension")

    def product_splits(self, output_grade: int) -> tuple[tuple[int, int], ...]:
        if not self.contains(output_grade):
            return tuple()
        return tuple((i, output_grade - i) for i in range(output_grade + 1))


@lru_cache(maxsize=None)
def total_degree_layout(truncation: int, include_scalar: bool = True) -> TotalDegreeLayout:
    """Return a cached total-degree schedule.

    The cache contains only tiny Python objects and does not impose a maximum
    truncation on :class:`~tensordev.core.jax.Jax`.
    """

    return TotalDegreeLayout(int(truncation), bool(include_scalar))


def graded_convolution(
    output_grades: Iterable[Grade],
    *,
    splits: Callable[[Grade], Iterable[tuple[Grade, Grade]]],
    left_contains: Callable[[Grade], bool],
    right_contains: Callable[[Grade], bool],
    left_block: Callable[[Grade], Block],
    right_block: Callable[[Grade], Block],
    product_block: Callable[[Block, Block, Grade, Grade, Grade], Block],
    zero_block: Callable[[Grade], Block] | None = None,
    product_grade: Callable[
        [tuple[tuple[Block, Block, Grade, Grade], ...], Grade], Block
    ]
    | None = None,
) -> tuple[Block, ...]:
    """Execute a statically scheduled graded convolution.

    ``splits`` controls both the admissible pairs and their summation order.
    Keeping this driver backend- and grading-agnostic lets total degree and
    bidegree share the public product algorithm without runtime dispatch.
    A grading may supply ``product_grade`` to fuse the statically resolved
    contributions before they are assembled into one output block.
    """

    out: list[Block] = []
    for output_grade in output_grades:
        pairs = tuple(
            (left_grade, right_grade)
            for left_grade, right_grade in splits(output_grade)
            if left_contains(left_grade) and right_contains(right_grade)
        )
        if not pairs:
            if zero_block is None:
                raise ValueError(f"No product contribution for output grade {output_grade!r}.")
            out.append(zero_block(output_grade))
            continue
        contributions = tuple(
            (
                left_block(left_grade),
                right_block(right_grade),
                left_grade,
                right_grade,
            )
            for left_grade, right_grade in pairs
        )
        if product_grade is not None:
            out.append(product_grade(contributions, output_grade))
            continue
        left, right, left_grade, right_grade = contributions[0]
        term = product_block(
            left,
            right,
            left_grade,
            right_grade,
            output_grade,
        )
        for left, right, left_grade, right_grade in contributions[1:]:
            term = term + product_block(
                left,
                right,
                left_grade,
                right_grade,
                output_grade,
            )
        out.append(term)
    return tuple(out)


def graded_contraction(
    output_grades: Iterable[Grade],
    *,
    pairs: Callable[[Grade], Iterable[tuple[Grade, Grade]]],
    multiplier_block: Callable[[Grade], Block],
    target_block: Callable[[Grade], Block],
    contract_block: Callable[[Block, Block, Grade, Grade, Grade], Block],
    zero_block: Callable[[Grade], Block] | None = None,
) -> tuple[Block, ...]:
    """Execute the outer schedule for a graded adjoint/contraction.

    Each pair is ``(multiplier_grade, target_grade)``.  Missing lower grades
    can be represented canonically by supplying ``zero_block``.
    """

    out: list[Block] = []
    for output_grade in output_grades:
        grade_pairs = tuple(pairs(output_grade))
        if not grade_pairs:
            if zero_block is None:
                raise ValueError(f"No contraction contribution for output grade {output_grade!r}.")
            out.append(zero_block(output_grade))
            continue
        multiplier_grade, target_grade = grade_pairs[0]
        term = contract_block(
            multiplier_block(multiplier_grade),
            target_block(target_grade),
            multiplier_grade,
            target_grade,
            output_grade,
        )
        for multiplier_grade, target_grade in grade_pairs[1:]:
            term = term + contract_block(
                multiplier_block(multiplier_grade),
                target_block(target_grade),
                multiplier_grade,
                target_grade,
                output_grade,
            )
        out.append(term)
    return tuple(out)


def map_graded_blocks(
    grades: Iterable[Grade],
    block: Callable[[Grade], Block],
    operation: Callable[[Block, Grade], Block],
) -> tuple[Block, ...]:
    """Apply a grade-aware component operation in canonical grade order."""

    return tuple(operation(block(grade), grade) for grade in grades)


def graded_summation(
    output_grades: Iterable[Grade],
    *,
    left_contains: Callable[[Grade], bool],
    right_contains: Callable[[Grade], bool],
    left_block: Callable[[Grade], Block],
    right_block: Callable[[Grade], Block],
    add: Callable[[Block, Block, Grade], Block] | None = None,
    left_only: Callable[[Block, Grade], Block] | None = None,
    right_only: Callable[[Block, Grade], Block] | None = None,
    zero_block: Callable[[Grade], Block] | None = None,
) -> tuple[Block, ...]:
    """Shared static outer driver for graded addition and zero padding."""

    add = add or (lambda left, right, _grade: left + right)
    left_only = left_only or (lambda block, _grade: block)
    right_only = right_only or (lambda block, _grade: block)
    out: list[Block] = []
    for grade in output_grades:
        has_left = left_contains(grade)
        has_right = right_contains(grade)
        if has_left and has_right:
            out.append(add(left_block(grade), right_block(grade), grade))
        elif has_left:
            out.append(left_only(left_block(grade), grade))
        elif has_right:
            out.append(right_only(right_block(grade), grade))
        elif zero_block is not None:
            out.append(zero_block(grade))
        else:
            raise ValueError(f"No summand is available at output grade {grade!r}.")
    return tuple(out)


def graded_inner_product(
    grades: Iterable[Grade],
    *,
    left_block: Callable[[Grade], Block],
    right_block: Callable[[Grade], Block],
    inner_block: Callable[[Block, Block], Any],
    zero: Callable[[], Any],
):
    """Sum homogeneous inner products in a deterministic grade order."""

    grades = tuple(grades)
    if not grades:
        return zero()
    result = inner_block(left_block(grades[0]), right_block(grades[0]))
    for grade in grades[1:]:
        result = result + inner_block(left_block(grade), right_block(grade))
    return result


def _first_level_generator_map(
        generator_blocks: Iterable[tuple[Grade, Block]],
) -> dict[Grade, Block]:
    generators = tuple(generator_blocks)
    generator_by_grade = dict(generators)
    if len(generator_by_grade) != len(generators):
        raise ValueError("first-level generator grades must be unique")
    return generator_by_grade


def _right_generator_action_block(
        layout: ResolvedGradingLayout[Grade],
        output_grade: Grade,
        *,
        source_contains: Callable[[Grade], bool],
        source_block: Callable[[Grade], Block],
        generator_by_grade: dict[Grade, Block],
        right_generator_output_block: Callable[..., Block],
) -> Block:
    splits = tuple(
        (source_grade, generator_grade)
        for source_grade, generator_grade in layout.product_splits(output_grade)
        if generator_grade in generator_by_grade
    )
    if not splits:
        raise ValueError(
            "no first-level generator predecessor for output grade "
            f"{output_grade!r}"
        )
    missing = tuple(
        source_grade
        for source_grade, _generator_grade in splits
        if not source_contains(source_grade)
    )
    if missing:
        raise ValueError(
            f"missing generator predecessors {missing!r} for output grade "
            f"{output_grade!r}"
        )

    predecessor_grades = tuple(source for source, _ in splits)
    generator_grades = tuple(generator for _, generator in splits)
    return right_generator_output_block(
        tuple(source_block(grade) for grade in predecessor_grades),
        tuple(generator_by_grade[grade] for grade in generator_grades),
        predecessor_grades=predecessor_grades,
        generator_grades=generator_grades,
        output_grade=output_grade,
    )


def graded_right_multiply_first_level(
    layout: ResolvedGradingLayout[Grade],
    *,
    source_contains: Callable[[Grade], bool],
    source_block: Callable[[Grade], Block],
    generator_blocks: Iterable[tuple[Grade, Block]],
    right_generator_output_block: Callable[..., Block],
    assemble: Callable[[tuple[Block, ...]], Element],
) -> Element:
    """Right-multiply an element by a first-level generator grade by grade.

    ``layout`` describes the scalar-omitting output.  The numerical action is
    delegated to the core, so total degree, bidegree, and quotient layouts use
    one schedule without materializing a padded generator tensor.
    """
    if layout.contains(layout.zero_grade):
        raise ValueError(
            "a right first-level product requires a scalar-omitting output layout"
        )
    generator_by_grade = _first_level_generator_map(generator_blocks)
    return assemble(
        tuple(
            _right_generator_action_block(
                layout,
                output_grade,
                source_contains=source_contains,
                source_block=source_block,
                generator_by_grade=generator_by_grade,
                right_generator_output_block=right_generator_output_block,
            )
            for output_grade in layout.grades
        )
    )


def _graded_horner_first_level(
    layout: ResolvedGradingLayout[Grade],
    *,
    max_order: int,
    initial_block: Callable[[Grade, Block | None], Block],
    step_block: Callable[[int, Grade, Block | None], Block],
    action_scale: Callable[[int], Any | None],
    generator_blocks: Iterable[tuple[Grade, Block]],
    right_generator_output_block: Callable[..., Block],
    assemble: Callable[[tuple[Block, ...]], Element],
) -> Element:
    """Run a grade-native Horner recurrence driven by a first-level action."""
    if max_order < 0:
        raise ValueError("max_order must be non-negative")

    generator_by_grade = _first_level_generator_map(generator_blocks)

    zero_grade = layout.zero_grade
    previous: dict[Grade, Block] = {
        zero_grade: initial_block(zero_grade, None)
    }
    if max_order == 0:
        return assemble((previous[zero_grade],))

    for step in range(1, max_order + 1):
        active_order = step
        current: dict[Grade, Block] = {
            zero_grade: step_block(step, zero_grade, None)
        }
        for output_grade in layout.grades:
            output_order = layout.total_degree(output_grade)
            if not (1 <= output_order <= active_order):
                continue

            action = _right_generator_action_block(
                layout,
                output_grade,
                source_contains=previous.__contains__,
                source_block=previous.__getitem__,
                generator_by_grade=generator_by_grade,
                right_generator_output_block=right_generator_output_block,
            )
            scale = action_scale(step)
            scaled_action = action if scale is None else action * scale
            current[output_grade] = (
                step_block(step, output_grade, action) + scaled_action
            )
        previous = current

    return assemble(tuple(previous[grade] for grade in layout.grades))


def graded_horner_first_level(
    layout: ResolvedGradingLayout[Grade],
    *,
    max_order: int,
    base_block: Callable[[Grade, Block | None], Block],
    generator_blocks: Iterable[tuple[Grade, Block]],
    right_generator_output_block: Callable[..., Block],
    assemble: Callable[[tuple[Block, ...]], Element],
) -> Element:
    """Evaluate ``g * exp(z)`` with one shared pruned Horner recurrence.

    ``layout`` supplies only static grade structure.  Numerical details of
    right multiplication by the first-level generator are delegated to the
    core hook, so standard/shear and total/bidegree implementations share the
    recurrence without sharing an unsuitable block layout.

    ``base_block(grade, like)`` returns the corresponding block of ``g`` or a
    zero block when it is absent.  ``like`` is ``None`` only for the scalar
    grade and otherwise contains the freshly computed generator action, which
    lets dimension-free total cores infer a missing block's shape without
    padding the input first.
    """
    return _graded_horner_first_level(
        layout,
        max_order=max_order,
        initial_block=base_block,
        step_block=lambda _step, grade, like: base_block(grade, like),
        action_scale=lambda step: 1.0 / float(max_order - step + 1),
        generator_blocks=generator_blocks,
        right_generator_output_block=right_generator_output_block,
        assemble=assemble,
    )


def graded_polynomial_horner_first_level(
    layout: ResolvedGradingLayout[Grade],
    *,
    max_order: int,
    coefficient_block: Callable[[int, Grade, Block | None], Block],
    generator_blocks: Iterable[tuple[Grade, Block]],
    right_generator_output_block: Callable[..., Block],
    assemble: Callable[[tuple[Block, ...]], Element],
) -> Element:
    """Evaluate a scalar-coefficient polynomial at a first-level tensor.

    ``coefficient_block(index, grade, like)`` supplies the block of the
    coefficient of ``z**index``.  The callback normally returns the scalar
    coefficient at the zero grade and a shape-compatible zero elsewhere.
    The polynomial and exponential Horner algorithms consequently share the
    same grade schedule and native generator kernels.
    """
    return _graded_horner_first_level(
        layout,
        max_order=max_order,
        initial_block=lambda grade, like: coefficient_block(
            max_order, grade, like
        ),
        step_block=lambda step, grade, like: coefficient_block(
            max_order - step, grade, like
        ),
        action_scale=lambda _step: None,
        generator_blocks=generator_blocks,
        right_generator_output_block=right_generator_output_block,
        assemble=assemble,
    )


Element = TypeVar("Element")


def formal_exponential_series(
    argument: Element,
    *,
    identity: Element,
    max_order: int,
    product: Callable[[Element, Element], Element],
    summation: Callable[[Element, Element], Element],
    scalar_multiply: Callable[[Element, float], Element],
) -> Element:
    """Compute a nilpotent algebra exponential from shared primitives.

    ``max_order`` is the nilpotence bound of the resolved layout.  It equals
    the truncation in total degree and ``N + M`` for a rectangular bidegree
    layout.  The function is intentionally storage-agnostic.
    """

    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    result = identity
    power = identity
    inverse_factorial = 1.0
    for order in range(1, max_order + 1):
        power = product(power, argument)
        inverse_factorial /= float(order)
        result = summation(result, scalar_multiply(power, inverse_factorial))
    return result


def formal_logarithm_series(
    argument: Element,
    *,
    zero: Element,
    max_order: int,
    product: Callable[[Element, Element], Element],
    summation: Callable[[Element, Element], Element],
    scalar_multiply: Callable[[Element, float], Element],
) -> Element:
    """Compute ``log(1 + argument)`` from shared algebra primitives."""

    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    result = zero
    term = argument
    for order in range(1, max_order + 1):
        coefficient = (1.0 if order % 2 else -1.0) / float(order)
        result = summation(result, scalar_multiply(term, coefficient))
        if order != max_order:
            term = product(term, argument)
    return result


__all__ = [
    "ResolvedGradingLayout",
    "TotalDegreeLayout",
    "graded_contraction",
    "graded_convolution",
    "graded_horner_first_level",
    "graded_polynomial_horner_first_level",
    "graded_right_multiply_first_level",
    "graded_inner_product",
    "graded_summation",
    "formal_exponential_series",
    "formal_logarithm_series",
    "map_graded_blocks",
    "total_degree_layout",
]
