"""Backend-independent symbolic combinatorics for ordered shear coordinates.

The objects in this module describe supports only.  They contain no numerical
arrays, alphabet dimensions, placement ranks, or backend-specific indexing.
All positions and axis labels are zero based.

An axis permutation follows the convention used by NumPy/JAX ``transpose``:
entry ``k`` is the source axis placed at output axis ``k``.  The
``doubleprime_permutation`` stored by :class:`TransformSupportTerm` instead
uses a row convention: it is the order of the output double-prime labels in
the input word.  :func:`transform_axis_permutation` performs the required
conversion.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from numbers import Integral
from threading import RLock
from typing import Iterable, Sequence, Tuple, TypeVar


Positions = Tuple[int, ...]
AxisPermutation = Tuple[int, ...]
_T = TypeVar("_T")


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer, got {value!r}.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}.")
    return result


def _normalize_positions(
    values: Iterable[int],
    *,
    name: str,
    length: int | None = None,
    expected_size: int | None = None,
) -> Positions:
    try:
        values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integer positions.") from error

    normalized = []
    previous = -1
    for index, value in enumerate(values):
        value = _non_negative_int(value, name=f"{name}[{index}]")
        if value <= previous:
            raise ValueError(f"{name} must be strictly increasing, got {values}.")
        if length is not None and value >= length:
            raise ValueError(
                f"{name}[{index}]={value} is outside zero-based length {length}."
            )
        normalized.append(value)
        previous = value
    if expected_size is not None and len(normalized) != expected_size:
        raise ValueError(
            f"{name} must contain {expected_size} positions, got {len(normalized)}."
        )
    return tuple(normalized)


def _normalize_permutation(
    values: Iterable[int],
    *,
    name: str,
) -> AxisPermutation:
    try:
        values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integer axes.") from error
    normalized = tuple(
        _non_negative_int(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if sorted(normalized) != list(range(len(normalized))):
        raise ValueError(
            f"{name} must be a permutation of range({len(normalized)}), "
            f"got {values}."
        )
    return normalized


@dataclass(frozen=True, slots=True)
class TransformSupportTerm:
    """One nonzero row contribution of ``Psi`` or ``Psi^{-1}``.

    Prime positions refer to the output and input words, respectively.  Prime
    letters retain their order.  ``doubleprime_permutation`` is the order in
    which output double-prime labels occur in the input word, matching the
    row permutation ``sigma``.  The coefficient is one for ``Psi`` and either
    sign for ``Psi^{-1}``.
    """

    output_prime_positions: Positions
    input_prime_positions: Positions
    doubleprime_permutation: AxisPermutation
    coefficient: int

    def __post_init__(self) -> None:
        output = _normalize_positions(
            self.output_prime_positions,
            name="output_prime_positions",
        )
        doubleprime = _normalize_permutation(
            self.doubleprime_permutation,
            name="doubleprime_permutation",
        )
        degree = len(output) + len(doubleprime)
        output = _normalize_positions(
            output,
            name="output_prime_positions",
            length=degree,
        )
        input_positions = _normalize_positions(
            self.input_prime_positions,
            name="input_prime_positions",
            length=degree,
            expected_size=len(output),
        )
        if isinstance(self.coefficient, bool) or not isinstance(
            self.coefficient, Integral
        ):
            raise TypeError(
                "coefficient must be either +1 or -1, "
                f"got {self.coefficient!r}."
            )
        coefficient = int(self.coefficient)
        if coefficient not in (-1, 1):
            raise ValueError(
                "coefficient must be either +1 or -1, "
                f"got {coefficient}."
            )
        object.__setattr__(self, "output_prime_positions", output)
        object.__setattr__(self, "input_prime_positions", input_positions)
        object.__setattr__(self, "doubleprime_permutation", doubleprime)
        object.__setattr__(self, "coefficient", coefficient)

    @property
    def total_degree(self) -> int:
        return len(self.output_prime_positions) + len(
            self.doubleprime_permutation
        )

    @property
    def axis_permutation(self) -> AxisPermutation:
        """Full word-axis permutation suitable for ``transpose``."""
        return transform_axis_permutation(self)


@dataclass(frozen=True, slots=True)
class GeneratorSupportTerm:
    """One term of native right multiplication by a first-level generator.

    ``dense_permutation`` acts on the raw axes of ``source outer generator``.
    A term uses the prime generator exactly when the output has one more prime
    position than the input; otherwise it uses the double-prime generator.
    """

    output_prime_positions: Positions
    input_prime_positions: Positions
    dense_permutation: AxisPermutation

    def __post_init__(self) -> None:
        permutation = _normalize_permutation(
            self.dense_permutation,
            name="dense_permutation",
        )
        degree = len(permutation)
        output = _normalize_positions(
            self.output_prime_positions,
            name="output_prime_positions",
            length=degree,
        )
        input_positions = _normalize_positions(
            self.input_prime_positions,
            name="input_prime_positions",
            length=max(degree - 1, 0),
        )
        difference = len(output) - len(input_positions)
        if difference not in (0, 1):
            raise ValueError(
                "a generator term must add either one prime or one "
                "double-prime letter."
            )
        raw_prime_axes = set(input_positions)
        if difference == 1:
            raw_prime_axes.add(degree - 1)
        realized_output = tuple(
            output_axis
            for output_axis, raw_axis in enumerate(permutation)
            if raw_axis in raw_prime_axes
        )
        if realized_output != output:
            raise ValueError(
                "dense_permutation is incompatible with the input/output "
                "generator letter classes."
            )
        object.__setattr__(self, "output_prime_positions", output)
        object.__setattr__(self, "input_prime_positions", input_positions)
        object.__setattr__(self, "dense_permutation", permutation)

    @property
    def generator_is_prime(self) -> bool:
        return len(self.output_prime_positions) == len(
            self.input_prime_positions
        ) + 1


@dataclass(frozen=True, slots=True)
class GammaShuffleSupportTerm:
    """One coefficient-one term of the ordered Gamma-shuffle.

    The two interleavings record the positions occupied by axes from the left
    input in the output prime and double-prime subsequences.  Together with the
    output prime positions they determine the complete raw-outer permutation.
    """

    left_prime_positions: Positions
    right_prime_positions: Positions
    output_prime_positions: Positions
    prime_interleaving: Positions
    doubleprime_interleaving: Positions

    def __post_init__(self) -> None:
        for name in (
            "left_prime_positions",
            "right_prime_positions",
            "output_prime_positions",
            "prime_interleaving",
            "doubleprime_interleaving",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_positions(getattr(self, name), name=name),
            )


@lru_cache(maxsize=None)
def canonical_placements(
    prime_count: int,
    doubleprime_count: int,
) -> Tuple[Positions, ...]:
    """All zero-based prime placements in deterministic lexicographic order."""
    prime_count = _non_negative_int(prime_count, name="prime_count")
    doubleprime_count = _non_negative_int(
        doubleprime_count,
        name="doubleprime_count",
    )
    return tuple(combinations(range(prime_count + doubleprime_count), prime_count))


def complement_positions(length: int, positions: Iterable[int]) -> Positions:
    """Complement an increasing position tuple inside ``range(length)``."""
    length = _non_negative_int(length, name="length")
    positions = _normalize_positions(positions, name="positions", length=length)
    selected = set(positions)
    return tuple(position for position in range(length) if position not in selected)


@lru_cache(maxsize=None)
def ordinary_shuffle_placements(
    left_size: int,
    right_size: int,
) -> Tuple[Positions, ...]:
    """Positions of the left sequence in every order-preserving shuffle."""
    left_size = _non_negative_int(left_size, name="left_size")
    right_size = _non_negative_int(right_size, name="right_size")
    return tuple(combinations(range(left_size + right_size), left_size))


def interleave(
    left: Sequence[_T],
    right: Sequence[_T],
    left_positions: Iterable[int],
) -> Tuple[_T, ...]:
    """Interleave two sequences while preserving their individual orders."""
    left = tuple(left)
    right = tuple(right)
    total = len(left) + len(right)
    left_positions = _normalize_positions(
        left_positions,
        name="left_positions",
        length=total,
        expected_size=len(left),
    )
    selected = set(left_positions)
    left_iterator = iter(left)
    right_iterator = iter(right)
    return tuple(
        next(left_iterator) if position in selected else next(right_iterator)
        for position in range(total)
    )


def invert_permutation(permutation: Iterable[int]) -> AxisPermutation:
    """Invert an axis permutation."""
    permutation = _normalize_permutation(permutation, name="permutation")
    inverse = [0] * len(permutation)
    for output_axis, input_axis in enumerate(permutation):
        inverse[input_axis] = output_axis
    return tuple(inverse)


def compose_permutations(
    first: Iterable[int],
    second: Iterable[int],
) -> AxisPermutation:
    """Compose two transpose permutations in their application order.

    If ``Y = transpose(X, first)`` and ``Z = transpose(Y, second)``, the
    returned tuple is the axis permutation for obtaining ``Z`` directly from
    ``X``.
    """
    first = _normalize_permutation(first, name="first")
    second = _normalize_permutation(second, name="second")
    if len(first) != len(second):
        raise ValueError(
            "first and second must have equal lengths, got "
            f"{len(first)} and {len(second)}."
        )
    return tuple(first[axis] for axis in second)


def placement_parity(positions: Iterable[int]) -> int:
    """Return ``(-1)**sum(positions)`` for one zero-based placement."""
    positions = _normalize_positions(positions, name="positions")
    return -1 if sum(positions) % 2 else 1


def transform_placement_sign(
    output_prime_positions: Iterable[int],
    input_prime_positions: Iterable[int],
) -> int:
    """Sign of one inverse-transform support term from its placements."""
    output_prime_positions = _normalize_positions(
        output_prime_positions,
        name="output_prime_positions",
    )
    input_prime_positions = _normalize_positions(
        input_prime_positions,
        name="input_prime_positions",
        expected_size=len(output_prime_positions),
    )
    return placement_parity(output_prime_positions) * placement_parity(
        input_prime_positions
    )


def packed_axes_to_word_permutation(
    output_prime_positions: Iterable[int],
    prime_axes: Sequence[int],
    doubleprime_axes: Sequence[int],
) -> AxisPermutation:
    """Place packed prime/double-prime source axes into full word order."""
    prime_axes = tuple(prime_axes)
    doubleprime_axes = tuple(doubleprime_axes)
    degree = len(prime_axes) + len(doubleprime_axes)
    output_prime_positions = _normalize_positions(
        output_prime_positions,
        name="output_prime_positions",
        length=degree,
        expected_size=len(prime_axes),
    )
    permutation = interleave(
        prime_axes,
        doubleprime_axes,
        output_prime_positions,
    )
    return _normalize_permutation(permutation, name="word_axis_permutation")


def transform_axis_permutation(
    term: TransformSupportTerm,
) -> AxisPermutation:
    """Convert one transform row term to a full transpose permutation."""
    if not isinstance(term, TransformSupportTerm):
        raise TypeError(
            "term must be a TransformSupportTerm, got "
            f"{type(term).__name__}."
        )
    degree = term.total_degree
    input_doubleprime_positions = complement_positions(
        degree,
        term.input_prime_positions,
    )
    # The stored row permutation lists output labels in input-axis order.
    # A transpose tuple requires the inverse: the input axis carrying each
    # output label.
    inverse_doubleprime = invert_permutation(term.doubleprime_permutation)
    doubleprime_axes = tuple(
        input_doubleprime_positions[index]
        for index in inverse_doubleprime
    )
    return packed_axes_to_word_permutation(
        term.output_prime_positions,
        term.input_prime_positions,
        doubleprime_axes,
    )


@lru_cache(maxsize=None)
def adjacent_block_axis_permutations(
    prefix_size: int,
    left_size: int,
    right_size: int,
) -> Tuple[AxisPermutation, ...]:
    """Leave a prefix fixed and shuffle two adjacent ordered axis blocks."""
    prefix_size = _non_negative_int(prefix_size, name="prefix_size")
    left_size = _non_negative_int(left_size, name="left_size")
    right_size = _non_negative_int(right_size, name="right_size")
    prefix = tuple(range(prefix_size))
    left = tuple(range(prefix_size, prefix_size + left_size))
    right = tuple(
        range(
            prefix_size + left_size,
            prefix_size + left_size + right_size,
        )
    )
    return tuple(
        prefix + interleave(left, right, placement)
        for placement in ordinary_shuffle_placements(left_size, right_size)
    )


@dataclass(frozen=True, slots=True)
class _Letter:
    is_prime: bool
    label: int


Word = Tuple[_Letter, ...]
_WordTerm = Tuple[Word, int]


def _word_from_prime_positions(
    prime_positions: Positions,
    degree: int,
) -> Word:
    prime_set = set(prime_positions)
    prime_index = 0
    doubleprime_index = 0
    word = []
    for position in range(degree):
        if position in prime_set:
            word.append(_Letter(True, prime_index))
            prime_index += 1
        else:
            word.append(_Letter(False, doubleprime_index))
            doubleprime_index += 1
    return tuple(word)


@lru_cache(maxsize=None)
def _shuffle_words(left: Word, right: Word) -> Tuple[Word, ...]:
    return tuple(
        interleave(left, right, placement)
        for placement in ordinary_shuffle_placements(len(left), len(right))
    )


def _combine_word_terms(terms: Iterable[_WordTerm]) -> Tuple[_WordTerm, ...]:
    combined: dict[Word, int] = {}
    for word, coefficient in terms:
        combined[word] = combined.get(word, 0) + coefficient
    return tuple(
        (word, coefficient)
        for word, coefficient in combined.items()
        if coefficient
    )


def _last_prime_position(word: Word) -> int:
    for position in range(len(word) - 1, -1, -1):
        if word[position].is_prime:
            return position
    return -1


@lru_cache(maxsize=None)
def _psi_row(word: Word) -> Tuple[_WordTerm, ...]:
    last_prime = _last_prime_position(word)
    if last_prime < 0:
        return ((word, 1),)
    prefix = word[:last_prime]
    prime = word[last_prime]
    terminal = word[last_prime + 1 :]
    terms = (
        (shuffled, coefficient)
        for parent, coefficient in _psi_row(prefix)
        for shuffled in _shuffle_words(parent + (prime,), terminal)
    )
    return _combine_word_terms(terms)


def _gamma_units_from_word(word: Word) -> Tuple[Tuple[Word, ...], Word]:
    units = []
    doubleprime_block = []
    for letter in word:
        if letter.is_prime:
            units.append(tuple(doubleprime_block) + (letter,))
            doubleprime_block = []
        else:
            doubleprime_block.append(letter)
    return tuple(units), tuple(doubleprime_block)


@lru_cache(maxsize=None)
def _gamma_words(left: Word, right: Word) -> Tuple[Word, ...]:
    left_units, left_terminal = _gamma_units_from_word(left)
    right_units, right_terminal = _gamma_units_from_word(right)
    terms = []
    for unit_positions in ordinary_shuffle_placements(
        len(left_units),
        len(right_units),
    ):
        units = interleave(left_units, right_units, unit_positions)
        prefix = tuple(letter for unit in units for letter in unit)
        terms.extend(
            prefix + terminal
            for terminal in _shuffle_words(left_terminal, right_terminal)
        )
    # Do not deduplicate: distinct unit/terminal shuffles carry multiplicity,
    # even when a later realization uses equal alphabet letters.
    return tuple(terms)


@lru_cache(maxsize=None)
def _psi_inverse_row(word: Word) -> Tuple[_WordTerm, ...]:
    last_prime = _last_prime_position(word)
    if last_prime < 0:
        return ((word, 1),)
    prefix = word[:last_prime]
    prime = word[last_prime]
    terminal = word[last_prime + 1 :]
    terms = []
    for parent, coefficient in _psi_inverse_row(prefix):
        for peeled in range(len(terminal) + 1):
            reversed_prefix = tuple(reversed(terminal[:peeled]))
            sign = -1 if peeled % 2 else 1
            for gamma_word in _gamma_words(parent, reversed_prefix):
                terms.append(
                    (
                        gamma_word + (prime,) + terminal[peeled:],
                        coefficient * sign,
                    )
                )
    return _combine_word_terms(terms)


def _transform_term(
    output_prime_positions: Positions,
    input_word: Word,
    coefficient: int,
) -> TransformSupportTerm:
    input_prime_positions = tuple(
        position
        for position, letter in enumerate(input_word)
        if letter.is_prime
    )
    doubleprime_permutation = tuple(
        letter.label for letter in input_word if not letter.is_prime
    )
    return TransformSupportTerm(
        output_prime_positions=output_prime_positions,
        input_prime_positions=input_prime_positions,
        doubleprime_permutation=doubleprime_permutation,
        coefficient=coefficient,
    )


def validate_transform_support(
    support: Iterable[TransformSupportTerm],
) -> Tuple[TransformSupportTerm, ...]:
    """Require unique ``(input pattern, permutation)`` pairs per output row."""
    support = tuple(support)
    fibres: dict[Positions, set[tuple[Positions, AxisPermutation]]] = {}
    for index, term in enumerate(support):
        if not isinstance(term, TransformSupportTerm):
            raise TypeError(
                f"support[{index}] must be a TransformSupportTerm, "
                f"got {type(term).__name__}."
            )
        key = term.input_prime_positions, term.doubleprime_permutation
        fibre = fibres.setdefault(term.output_prime_positions, set())
        if key in fibre:
            raise ValueError(
                "transform support contains a duplicate input/permutation "
                f"pair in output row {term.output_prime_positions}."
            )
        fibre.add(key)
    return support


def _normalize_row(
    output_prime_positions: Iterable[int],
    total_degree: int,
) -> tuple[Positions, Word]:
    total_degree = _non_negative_int(total_degree, name="total_degree")
    output_prime_positions = _normalize_positions(
        output_prime_positions,
        name="output_prime_positions",
        length=total_degree,
    )
    return (
        output_prime_positions,
        _word_from_prime_positions(output_prime_positions, total_degree),
    )


def psi_row_support(
    output_prime_positions: Iterable[int],
    total_degree: int,
) -> Tuple[TransformSupportTerm, ...]:
    """Support of one ``Psi`` row with a fixed output letter-class pattern."""
    output_prime_positions, word = _normalize_row(
        output_prime_positions,
        total_degree,
    )
    return validate_transform_support(
        _transform_term(output_prime_positions, input_word, coefficient)
        for input_word, coefficient in _psi_row(word)
    )


def psi_inverse_row_support(
    output_prime_positions: Iterable[int],
    total_degree: int,
) -> Tuple[TransformSupportTerm, ...]:
    """Support of one ``Psi^{-1}`` row with a fixed output class pattern."""
    output_prime_positions, word = _normalize_row(
        output_prime_positions,
        total_degree,
    )
    support = validate_transform_support(
        _transform_term(output_prime_positions, input_word, coefficient)
        for input_word, coefficient in _psi_inverse_row(word)
    )
    for term in support:
        if term.coefficient != transform_placement_sign(
            term.output_prime_positions,
            term.input_prime_positions,
        ):
            raise AssertionError(
                "inverse-transform recursion disagrees with placement parity."
            )
    return support


@lru_cache(maxsize=None)
def psi_support(
    prime_count: int,
    doubleprime_count: int,
) -> Tuple[TransformSupportTerm, ...]:
    """Complete symbolic support of ``Psi`` at one bidegree."""
    placements = canonical_placements(prime_count, doubleprime_count)
    degree = _non_negative_int(prime_count, name="prime_count") + _non_negative_int(
        doubleprime_count,
        name="doubleprime_count",
    )
    return validate_transform_support(
        term
        for placement in placements
        for term in psi_row_support(placement, degree)
    )


@lru_cache(maxsize=None)
def psi_inverse_support(
    prime_count: int,
    doubleprime_count: int,
) -> Tuple[TransformSupportTerm, ...]:
    """Complete symbolic support of ``Psi^{-1}`` at one bidegree."""
    placements = canonical_placements(prime_count, doubleprime_count)
    degree = _non_negative_int(prime_count, name="prime_count") + _non_negative_int(
        doubleprime_count,
        name="doubleprime_count",
    )
    return validate_transform_support(
        term
        for placement in placements
        for term in psi_inverse_row_support(placement, degree)
    )


@lru_cache(maxsize=None)
def psi_total_support(total_degree: int) -> Tuple[TransformSupportTerm, ...]:
    """Complete ``Psi`` support across all class patterns of one level."""
    total_degree = _non_negative_int(total_degree, name="total_degree")
    return validate_transform_support(
        term
        for prime_count in range(total_degree + 1)
        for term in psi_support(prime_count, total_degree - prime_count)
    )


@lru_cache(maxsize=None)
def psi_inverse_total_support(
    total_degree: int,
) -> Tuple[TransformSupportTerm, ...]:
    """Complete ``Psi^{-1}`` support across all patterns of one level."""
    total_degree = _non_negative_int(total_degree, name="total_degree")
    return validate_transform_support(
        term
        for prime_count in range(total_degree + 1)
        for term in psi_inverse_support(prime_count, total_degree - prime_count)
    )


def gamma_units(
    prime_positions: Iterable[int],
    total_degree: int,
) -> Tuple[Tuple[Positions, ...], Positions]:
    """Split word axes into ``double-prime block + prime`` units and a tail."""
    total_degree = _non_negative_int(total_degree, name="total_degree")
    prime_positions = _normalize_positions(
        prime_positions,
        name="prime_positions",
        length=total_degree,
    )
    units = []
    start = 0
    for prime_position in prime_positions:
        units.append(tuple(range(start, prime_position + 1)))
        start = prime_position + 1
    return tuple(units), tuple(range(start, total_degree))


def _normalize_gamma_pattern(
    left_degree: int,
    right_degree: int,
    left_prime_positions: Iterable[int],
    right_prime_positions: Iterable[int],
) -> tuple[int, int, Positions, Positions]:
    left_degree = _non_negative_int(left_degree, name="left_degree")
    right_degree = _non_negative_int(right_degree, name="right_degree")
    left_prime_positions = _normalize_positions(
        left_prime_positions,
        name="left_prime_positions",
        length=left_degree,
    )
    right_prime_positions = _normalize_positions(
        right_prime_positions,
        name="right_prime_positions",
        length=right_degree,
    )
    return (
        left_degree,
        right_degree,
        left_prime_positions,
        right_prime_positions,
    )


def gamma_shuffle_pattern_support(
    left_degree: int,
    right_degree: int,
    left_prime_positions: Iterable[int],
    right_prime_positions: Iterable[int],
) -> Tuple[GammaShuffleSupportTerm, ...]:
    """Gamma-shuffle support for one pair of input class patterns."""
    (
        left_degree,
        right_degree,
        left_prime_positions,
        right_prime_positions,
    ) = _normalize_gamma_pattern(
        left_degree,
        right_degree,
        left_prime_positions,
        right_prime_positions,
    )
    left_prime_set = set(left_prime_positions)
    left = tuple(
        _Letter(position in left_prime_set, position)
        for position in range(left_degree)
    )
    right_prime_set = set(right_prime_positions)
    right = tuple(
        _Letter(
            position in right_prime_set,
            left_degree + position,
        )
        for position in range(right_degree)
    )
    terms = []
    for output_word in _gamma_words(left, right):
        output_prime_positions = tuple(
            position
            for position, letter in enumerate(output_word)
            if letter.is_prime
        )
        prime_axes = tuple(
            letter.label for letter in output_word if letter.is_prime
        )
        doubleprime_axes = tuple(
            letter.label for letter in output_word if not letter.is_prime
        )
        terms.append(
            GammaShuffleSupportTerm(
                left_prime_positions=left_prime_positions,
                right_prime_positions=right_prime_positions,
                output_prime_positions=output_prime_positions,
                prime_interleaving=tuple(
                    index
                    for index, axis in enumerate(prime_axes)
                    if axis < left_degree
                ),
                doubleprime_interleaving=tuple(
                    index
                    for index, axis in enumerate(doubleprime_axes)
                    if axis < left_degree
                ),
            )
        )
    return validate_gamma_shuffle_support(
        terms,
        left_degree=left_degree,
        right_degree=right_degree,
    )


@lru_cache(maxsize=None)
def gamma_shuffle_support(
    left_degree: int,
    right_degree: int,
) -> Tuple[GammaShuffleSupportTerm, ...]:
    """Complete Gamma-shuffle support for one total-degree pair."""
    left_degree = _non_negative_int(left_degree, name="left_degree")
    right_degree = _non_negative_int(right_degree, name="right_degree")
    return tuple(
        term
        for left_prime_count in range(left_degree + 1)
        for left_placement in canonical_placements(
            left_prime_count,
            left_degree - left_prime_count,
        )
        for right_prime_count in range(right_degree + 1)
        for right_placement in canonical_placements(
            right_prime_count,
            right_degree - right_prime_count,
        )
        for term in gamma_shuffle_pattern_support(
            left_degree,
            right_degree,
            left_placement,
            right_placement,
        )
    )


def gamma_axis_permutation(
    term: GammaShuffleSupportTerm,
    left_degree: int,
    right_degree: int,
) -> AxisPermutation:
    """Recover a Gamma term's full raw-outer transpose permutation."""
    if not isinstance(term, GammaShuffleSupportTerm):
        raise TypeError(
            "term must be a GammaShuffleSupportTerm, got "
            f"{type(term).__name__}."
        )
    (
        left_degree,
        right_degree,
        left_prime_positions,
        right_prime_positions,
    ) = _normalize_gamma_pattern(
        left_degree,
        right_degree,
        term.left_prime_positions,
        term.right_prime_positions,
    )
    if (
        left_prime_positions != term.left_prime_positions
        or right_prime_positions != term.right_prime_positions
    ):
        raise AssertionError("normalized Gamma positions changed unexpectedly.")
    left_prime_axes = left_prime_positions
    right_prime_axes = tuple(
        left_degree + position for position in right_prime_positions
    )
    left_doubleprime_axes = complement_positions(
        left_degree,
        left_prime_positions,
    )
    right_doubleprime_axes = tuple(
        left_degree + position
        for position in complement_positions(
            right_degree,
            right_prime_positions,
        )
    )
    prime_axes = interleave(
        left_prime_axes,
        right_prime_axes,
        _normalize_positions(
            term.prime_interleaving,
            name="prime_interleaving",
            length=len(left_prime_axes) + len(right_prime_axes),
            expected_size=len(left_prime_axes),
        ),
    )
    doubleprime_axes = interleave(
        left_doubleprime_axes,
        right_doubleprime_axes,
        _normalize_positions(
            term.doubleprime_interleaving,
            name="doubleprime_interleaving",
            length=len(left_doubleprime_axes) + len(right_doubleprime_axes),
            expected_size=len(left_doubleprime_axes),
        ),
    )
    return packed_axes_to_word_permutation(
        _normalize_positions(
            term.output_prime_positions,
            name="output_prime_positions",
            length=left_degree + right_degree,
            expected_size=len(prime_axes),
        ),
        prime_axes,
        doubleprime_axes,
    )


def validate_gamma_shuffle_support(
    support: Iterable[GammaShuffleSupportTerm],
    *,
    left_degree: int | None = None,
    right_degree: int | None = None,
) -> Tuple[GammaShuffleSupportTerm, ...]:
    """Require symbolic Gamma support to be injective and dimensionally valid."""
    if (left_degree is None) != (right_degree is None):
        raise ValueError(
            "left_degree and right_degree must either both be provided or "
            "both be omitted."
        )
    if left_degree is not None:
        left_degree = _non_negative_int(left_degree, name="left_degree")
        right_degree = _non_negative_int(right_degree, name="right_degree")
    support = tuple(support)
    seen = set()
    for index, term in enumerate(support):
        if not isinstance(term, GammaShuffleSupportTerm):
            raise TypeError(
                f"support[{index}] must be a GammaShuffleSupportTerm, "
                f"got {type(term).__name__}."
            )
        if term in seen:
            raise ValueError(
                "Gamma-shuffle support contains a duplicate symbolic term."
            )
        seen.add(term)
        if left_degree is not None:
            gamma_axis_permutation(term, left_degree, right_degree)
    return support


def right_generator_pattern_support(
    output_prime_positions: Iterable[int],
    output_degree: int,
) -> Tuple[GeneratorSupportTerm, ...]:
    """Native right-generator support for one output class pattern."""
    output_degree = _non_negative_int(output_degree, name="output_degree")
    if output_degree == 0:
        raise ValueError("output_degree must be positive for a generator action.")
    output_prime_positions = _normalize_positions(
        output_prime_positions,
        name="output_prime_positions",
        length=output_degree,
    )
    terms = []

    # Appending a double-prime letter leaves the source pattern unchanged and
    # can contribute exactly when the output word ends in that letter.
    if output_degree - 1 not in output_prime_positions:
        terms.append(
            GeneratorSupportTerm(
                output_prime_positions=output_prime_positions,
                input_prime_positions=output_prime_positions,
                dense_permutation=tuple(range(output_degree)),
            )
        )

    # Appending a prime generator removes the rightmost output prime.  The two
    # adjacent double-prime blocks are shuffled into one terminal source block.
    if output_prime_positions:
        final_prime = output_prime_positions[-1]
        previous_prime = (
            output_prime_positions[-2]
            if len(output_prime_positions) >= 2
            else -1
        )
        prefix = tuple(range(previous_prime + 1))
        left_block = tuple(range(previous_prime + 1, final_prime))
        right_block = tuple(range(final_prime + 1, output_degree))
        separated = prefix + left_block + right_block
        for permutation in adjacent_block_axis_permutations(
            len(prefix),
            len(left_block),
            len(right_block),
        ):
            source_word = tuple(separated[axis] for axis in permutation)
            raw_outer_labels = source_word + (final_prime,)
            dense_permutation = tuple(
                raw_outer_labels.index(output_axis)
                for output_axis in range(output_degree)
            )
            terms.append(
                GeneratorSupportTerm(
                    output_prime_positions=output_prime_positions,
                    input_prime_positions=output_prime_positions[:-1],
                    dense_permutation=dense_permutation,
                )
            )
    return validate_generator_support(terms)


@lru_cache(maxsize=None)
def right_generator_support(
    output_degree: int,
) -> Tuple[GeneratorSupportTerm, ...]:
    """Complete native right-generator support at one output total degree."""
    output_degree = _non_negative_int(output_degree, name="output_degree")
    if output_degree == 0:
        raise ValueError("output_degree must be positive for a generator action.")
    return tuple(
        term
        for prime_count in range(output_degree + 1)
        for placement in canonical_placements(
            prime_count,
            output_degree - prime_count,
        )
        for term in right_generator_pattern_support(
            placement,
            output_degree,
        )
    )


def validate_generator_support(
    support: Iterable[GeneratorSupportTerm],
) -> Tuple[GeneratorSupportTerm, ...]:
    """Require unique symbolic terms in one right-generator support."""
    support = tuple(support)
    seen = set()
    for index, term in enumerate(support):
        if not isinstance(term, GeneratorSupportTerm):
            raise TypeError(
                f"support[{index}] must be a GeneratorSupportTerm, "
                f"got {type(term).__name__}."
            )
        if term in seen:
            raise ValueError(
                "right-generator support contains a duplicate symbolic term."
            )
        seen.add(term)
    return support


_SYMBOLIC_CACHES = (
    canonical_placements,
    ordinary_shuffle_placements,
    adjacent_block_axis_permutations,
    _shuffle_words,
    _psi_row,
    _gamma_words,
    _psi_inverse_row,
    psi_support,
    psi_inverse_support,
    psi_total_support,
    psi_inverse_total_support,
    gamma_shuffle_support,
    right_generator_support,
)
_SYMBOLIC_CACHE_SCOPE_LOCK = RLock()
_SYMBOLIC_CACHE_SCOPE_DEPTH = 0


def clear_symbolic_caches() -> None:
    """Release construction-only symbolic support retained by ``lru_cache``."""
    for cached_function in _SYMBOLIC_CACHES:
        cached_function.cache_clear()


def symbolic_cache_entry_count() -> int:
    """Return the number of currently retained symbolic cache entries."""
    return sum(
        cached_function.cache_info().currsize
        for cached_function in _SYMBOLIC_CACHES
    )


@contextmanager
def symbolic_plan_compilation_scope():
    """Share symbolic caches during nested compilation, then release them.

    Clearing is delayed until the last overlapping scope exits.  Concurrent
    builders may therefore reuse the pure support cache without one builder
    evicting entries still useful to another; eviction can only add work and
    never changes the deterministic plans.
    """
    global _SYMBOLIC_CACHE_SCOPE_DEPTH
    with _SYMBOLIC_CACHE_SCOPE_LOCK:
        _SYMBOLIC_CACHE_SCOPE_DEPTH += 1
    try:
        yield
    finally:
        with _SYMBOLIC_CACHE_SCOPE_LOCK:
            _SYMBOLIC_CACHE_SCOPE_DEPTH -= 1
            if _SYMBOLIC_CACHE_SCOPE_DEPTH == 0:
                clear_symbolic_caches()


__all__ = [
    "AxisPermutation",
    "GeneratorSupportTerm",
    "Positions",
    "TransformSupportTerm",
    "adjacent_block_axis_permutations",
    "canonical_placements",
    "complement_positions",
    "compose_permutations",
    "interleave",
    "invert_permutation",
    "ordinary_shuffle_placements",
    "packed_axes_to_word_permutation",
    "placement_parity",
    "right_generator_pattern_support",
    "right_generator_support",
    "clear_symbolic_caches",
    "symbolic_cache_entry_count",
    "symbolic_plan_compilation_scope",
    "transform_axis_permutation",
    "transform_placement_sign",
    "validate_generator_support",
    "validate_transform_support",
]
