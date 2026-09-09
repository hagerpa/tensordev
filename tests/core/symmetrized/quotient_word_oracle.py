"""Exact quotient-word calculations for small independent tests.

Implements normal forms, partial symmetrization and its adjoint, quotient
concatenation, and shear coordinate maps from ``academia/bidegree/main.tex``.
Exhaustive word enumeration limits this pure-Python reference to small examples.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import combinations, product
from math import comb, factorial


Word = tuple[str, ...]
WordPolynomial = Counter[Word]


def _clean(counter):
    return Counter({key: value for key, value in counter.items() if value})


def _alphabets(prime_alphabet, doubleprime_alphabet):
    prime = tuple(prime_alphabet)
    doubleprime = tuple(doubleprime_alphabet)
    if len(set(prime)) != len(prime):
        raise ValueError("prime_alphabet contains duplicate letters")
    if not doubleprime or len(set(doubleprime)) != len(doubleprime):
        raise ValueError(
            "doubleprime_alphabet must be nonempty and contain no duplicates"
        )
    if set(prime) & set(doubleprime):
        raise ValueError("the two alphabets must be disjoint")
    return prime, doubleprime


@dataclass(frozen=True, slots=True)
class NormalForm:
    """A quotient word ``alpha_0 i_1 ... i_n alpha_n``."""

    blocks: tuple[tuple[int, ...], ...]
    primes: Word

    def __post_init__(self):
        if len(self.blocks) != len(self.primes) + 1:
            raise ValueError("a normal form needs one more block than primes")
        widths = {len(block) for block in self.blocks}
        if len(widths) != 1 or not widths or next(iter(widths)) == 0:
            raise ValueError("all normal-form blocks need one positive width")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for block in self.blocks
            for value in block
        ):
            raise ValueError("normal-form multiplicities must be nonnegative ints")

    @property
    def bidegree(self):
        return len(self.primes), sum(map(sum, self.blocks))


def normal_form(
    word: Sequence[str],
    prime_alphabet: Sequence[str],
    doubleprime_alphabet: Sequence[str],
) -> NormalForm:
    """Return the unique quotient normal form of one ordinary word."""
    prime, doubleprime = _alphabets(prime_alphabet, doubleprime_alphabet)
    prime_set = set(prime)
    doubleprime_index = {
        letter: index for index, letter in enumerate(doubleprime)
    }
    blocks = [[0] * len(doubleprime)]
    primes = []
    for letter in tuple(word):
        if letter in prime_set:
            primes.append(letter)
            blocks.append([0] * len(doubleprime))
        elif letter in doubleprime_index:
            blocks[-1][doubleprime_index[letter]] += 1
        else:
            raise ValueError(f"letter {letter!r} is outside the split alphabet")
    return NormalForm(tuple(map(tuple, blocks)), tuple(primes))


def sym_rank(form: NormalForm) -> int:
    """Colexicographic multiset-placement rank from equation (3.2)."""
    flattened = tuple(value for block in form.blocks for value in block)
    prefix = 0
    rank = 0
    for separator, value in enumerate(flattened[:-1], start=1):
        prefix += value
        rank += comb(separator - 1 + prefix, separator)
    return rank


def canonical_word(
    form: NormalForm,
    doubleprime_alphabet: Sequence[str],
) -> Word:
    """Choose the alphabet-ordered representative of a normal form."""
    doubleprime = tuple(doubleprime_alphabet)
    if any(len(block) != len(doubleprime) for block in form.blocks):
        raise ValueError("normal-form block width disagrees with the alphabet")
    word = []
    for index, block in enumerate(form.blocks):
        for letter, multiplicity in zip(doubleprime, block):
            word.extend((letter,) * multiplicity)
        if index < len(form.primes):
            word.append(form.primes[index])
    return tuple(word)


def _multiset_words(counts, alphabet):
    if sum(counts) == 0:
        yield ()
        return
    for index, (count, letter) in enumerate(zip(counts, alphabet)):
        if count == 0:
            continue
        remainder = list(counts)
        remainder[index] -= 1
        for suffix in _multiset_words(tuple(remainder), alphabet):
            yield (letter,) + suffix


def representative_words(
    form: NormalForm,
    doubleprime_alphabet: Sequence[str],
) -> tuple[Word, ...]:
    """Enumerate all ordinary words in one quotient class."""
    doubleprime = tuple(doubleprime_alphabet)
    block_words = tuple(
        tuple(_multiset_words(block, doubleprime)) for block in form.blocks
    )
    representatives = []
    for selected in product(*block_words):
        word = []
        for index, block_word in enumerate(selected):
            word.extend(block_word)
            if index < len(form.primes):
                word.append(form.primes[index])
        representatives.append(tuple(word))
    return tuple(representatives)


def quotient_class_size(form: NormalForm) -> int:
    """Number of ordinary representatives of a quotient normal form."""
    size = 1
    for block in form.blocks:
        block_size = sum(block)
        size *= factorial(block_size)
        for multiplicity in block:
            size //= factorial(multiplicity)
    return size


def ordinary_words(
    prime_alphabet: Sequence[str],
    doubleprime_alphabet: Sequence[str],
    bidegree: tuple[int, int],
) -> tuple[Word, ...]:
    """Enumerate one tiny ordered-word bidegree block."""
    prime, doubleprime = _alphabets(prime_alphabet, doubleprime_alphabet)
    n, m = bidegree
    if n < 0 or m < 0:
        raise ValueError("bidegrees must be nonnegative")
    prime_set = set(prime)
    return tuple(
        word
        for word in product(prime + doubleprime, repeat=n + m)
        if sum(letter in prime_set for letter in word) == n
    )


def quotient_basis(
    prime_alphabet: Sequence[str],
    doubleprime_alphabet: Sequence[str],
    bidegree: tuple[int, int],
) -> tuple[NormalForm, ...]:
    """Return normal forms in skeleton order and note rank order."""
    forms = {
        normal_form(word, prime_alphabet, doubleprime_alphabet)
        for word in ordinary_words(
            prime_alphabet, doubleprime_alphabet, bidegree
        )
    }
    return tuple(sorted(forms, key=lambda form: (form.primes, sym_rank(form))))


def apply_q(
    polynomial: Mapping[Word, int],
    prime_alphabet: Sequence[str],
    doubleprime_alphabet: Sequence[str],
) -> Counter[NormalForm]:
    """Apply partial symmetrization ``Q`` by accumulating quotient classes."""
    output = Counter()
    for word, coefficient in polynomial.items():
        output[
            normal_form(word, prime_alphabet, doubleprime_alphabet)
        ] += coefficient
    return _clean(output)


def apply_q_transpose(
    polynomial: Mapping[NormalForm, int],
    doubleprime_alphabet: Sequence[str],
) -> WordPolynomial:
    """Apply ``Q.T`` by assigning a class coefficient to every representative."""
    output = Counter()
    for form, coefficient in polynomial.items():
        for word in representative_words(form, doubleprime_alphabet):
            output[word] += coefficient
    return _clean(output)


@dataclass(frozen=True, slots=True)
class QuotientProjection:
    """Dense small-degree matrix view of ``Q`` and ``Q.T``."""

    words: tuple[Word, ...]
    forms: tuple[NormalForm, ...]
    rows: tuple[tuple[int, ...], ...]

    @classmethod
    def build(cls, prime_alphabet, doubleprime_alphabet, bidegree):
        words = ordinary_words(
            prime_alphabet, doubleprime_alphabet, bidegree
        )
        forms = quotient_basis(
            prime_alphabet, doubleprime_alphabet, bidegree
        )
        form_index = {form: index for index, form in enumerate(forms)}
        rows = [[0] * len(words) for _ in forms]
        for word_index, word in enumerate(words):
            rows[
                form_index[
                    normal_form(
                        word, prime_alphabet, doubleprime_alphabet
                    )
                ]
            ][word_index] = 1
        return cls(words, forms, tuple(map(tuple, rows)))

    def apply(self, vector: Sequence[int]) -> tuple[int, ...]:
        if len(vector) != len(self.words):
            raise ValueError("ordinary vector has the wrong width")
        return tuple(
            sum(entry * value for entry, value in zip(row, vector))
            for row in self.rows
        )

    def transpose_apply(self, vector: Sequence[int]) -> tuple[int, ...]:
        if len(vector) != len(self.forms):
            raise ValueError("quotient vector has the wrong width")
        return tuple(
            sum(self.rows[row][column] * vector[row] for row in range(len(vector)))
            for column in range(len(self.words))
        )


@cache
def _shuffle_cached(left: Word, right: Word):
    if not left:
        return ((right, 1),)
    if not right:
        return ((left, 1),)
    output = Counter()
    for word, coefficient in _shuffle_cached(left[:-1], right):
        output[word + left[-1:]] += coefficient
    for word, coefficient in _shuffle_cached(left, right[:-1]):
        output[word + right[-1:]] += coefficient
    return tuple(sorted(_clean(output).items()))


def shuffle_words(left: Word, right: Word) -> WordPolynomial:
    """Ordinary word shuffle, retaining multiplicities exactly."""
    return Counter(dict(_shuffle_cached(tuple(left), tuple(right))))


def _last_prime_split(word: Word, prime_set: frozenset[str]):
    for index in range(len(word) - 1, -1, -1):
        if word[index] in prime_set:
            return word[:index], word[index], word[index + 1 :]
    return None


@cache
def _gamma_cached(left: Word, right: Word, prime_letters: tuple[str, ...]):
    prime_set = frozenset(prime_letters)
    left_split = _last_prime_split(left, prime_set)
    right_split = _last_prime_split(right, prime_set)
    if left_split is None and right_split is None:
        return tuple(sorted(shuffle_words(left, right).items()))
    output = Counter()
    if right_split is None:
        prefix, letter, terminal = left_split
        for suffix, coefficient in shuffle_words(terminal, right).items():
            output[prefix + (letter,) + suffix] += coefficient
    elif left_split is None:
        prefix, letter, terminal = right_split
        for suffix, coefficient in shuffle_words(left, terminal).items():
            output[prefix + (letter,) + suffix] += coefficient
    else:
        left_prefix, left_letter, left_terminal = left_split
        right_prefix, right_letter, right_terminal = right_split
        terminal_shuffle = shuffle_words(left_terminal, right_terminal)
        first_prefix = Counter(
            dict(
                _gamma_cached(
                    left_prefix + (left_letter,),
                    right_prefix,
                    prime_letters,
                )
            )
        )
        second_prefix = Counter(
            dict(
                _gamma_cached(
                    left_prefix,
                    right_prefix + (right_letter,),
                    prime_letters,
                )
            )
        )
        for prefix_word, prefix_coefficient in first_prefix.items():
            for suffix, suffix_coefficient in terminal_shuffle.items():
                output[
                    prefix_word + (right_letter,) + suffix
                ] += prefix_coefficient * suffix_coefficient
        for prefix_word, prefix_coefficient in second_prefix.items():
            for suffix, suffix_coefficient in terminal_shuffle.items():
                output[
                    prefix_word + (left_letter,) + suffix
                ] += prefix_coefficient * suffix_coefficient
    return tuple(sorted(_clean(output).items()))


def gamma_words(
    left: Word,
    right: Word,
    prime_alphabet: Sequence[str],
) -> WordPolynomial:
    """Ordered Gamma product from equation (5.1)."""
    return Counter(
        dict(
            _gamma_cached(
                tuple(left), tuple(right), tuple(prime_alphabet)
            )
        )
    )


@cache
def _psi_row_cached(word: Word, prime_letters: tuple[str, ...], inverse: bool):
    split = _last_prime_split(word, frozenset(prime_letters))
    if split is None:
        return ((word, 1),)
    prefix, letter, terminal = split
    parent = Counter(dict(_psi_row_cached(prefix, prime_letters, inverse)))
    output = Counter()
    if not inverse:
        for parent_word, parent_coefficient in parent.items():
            for shuffled, coefficient in shuffle_words(
                parent_word + (letter,), terminal
            ).items():
                output[shuffled] += parent_coefficient * coefficient
    else:
        for moved_count in range(len(terminal) + 1):
            moved = terminal[:moved_count][::-1]
            retained = terminal[moved_count:]
            sign = -1 if moved_count % 2 else 1
            for parent_word, parent_coefficient in parent.items():
                for transformed, coefficient in gamma_words(
                    parent_word, moved, prime_letters
                ).items():
                    output[
                        transformed + (letter,) + retained
                    ] += sign * parent_coefficient * coefficient
    return tuple(sorted(_clean(output).items()))


def psi_row(
    word: Word,
    prime_alphabet: Sequence[str],
    *,
    inverse: bool = False,
) -> WordPolynomial:
    """One ordinary graded row of Psi or Psi-inverse."""
    return Counter(
        dict(
            _psi_row_cached(
                tuple(word), tuple(prime_alphabet), bool(inverse)
            )
        )
    )


def hat_psi_image_of_representative(
    representative: Word,
    prime_alphabet: Sequence[str],
    doubleprime_alphabet: Sequence[str],
    bidegree: tuple[int, int],
    *,
    inverse: bool = False,
) -> Counter[NormalForm]:
    """Compute ``Q Psi e_w`` (or inverse) from the ordinary graded rows."""
    output = Counter()
    for output_word in ordinary_words(
        prime_alphabet, doubleprime_alphabet, bidegree
    ):
        coefficient = psi_row(
            output_word, prime_alphabet, inverse=inverse
        ).get(tuple(representative), 0)
        if coefficient:
            output[
                normal_form(
                    output_word, prime_alphabet, doubleprime_alphabet
                )
            ] += coefficient
    return _clean(output)


@dataclass(frozen=True, slots=True)
class QuotientLinearMap:
    """Exact integer matrix on one quotient bidegree block."""

    basis: tuple[NormalForm, ...]
    rows: tuple[tuple[int, ...], ...]

    def apply(self, polynomial: Mapping[NormalForm, int]):
        vector = tuple(polynomial.get(form, 0) for form in self.basis)
        return _clean(
            Counter(
                {
                    form: sum(
                        entry * value
                        for entry, value in zip(row, vector)
                    )
                    for form, row in zip(self.basis, self.rows)
                }
            )
        )

    def transpose_apply(self, polynomial: Mapping[NormalForm, int]):
        vector = tuple(polynomial.get(form, 0) for form in self.basis)
        return _clean(
            Counter(
                {
                    self.basis[column]: sum(
                        self.rows[row][column] * vector[row]
                        for row in range(len(self.basis))
                    )
                    for column in range(len(self.basis))
                }
            )
        )


def hat_psi_map(
    prime_alphabet: Sequence[str],
    doubleprime_alphabet: Sequence[str],
    bidegree: tuple[int, int],
    *,
    inverse: bool = False,
) -> QuotientLinearMap:
    """Construct the induced hat-Psi map from ``Q Psi = hat-Psi Q``."""
    basis = quotient_basis(
        prime_alphabet, doubleprime_alphabet, bidegree
    )
    rows = [[0] * len(basis) for _ in basis]
    index = {form: position for position, form in enumerate(basis)}
    for column, form in enumerate(basis):
        image = hat_psi_image_of_representative(
            canonical_word(form, doubleprime_alphabet),
            prime_alphabet,
            doubleprime_alphabet,
            bidegree,
            inverse=inverse,
        )
        for output_form, coefficient in image.items():
            rows[index[output_form]][column] = coefficient
    return QuotientLinearMap(basis, tuple(map(tuple, rows)))


def concatenate_forms(left: NormalForm, right: NormalForm) -> NormalForm:
    """Boundary-block merge ``alpha dot beta`` from Section 4.2."""
    if len(left.blocks[0]) != len(right.blocks[0]):
        raise ValueError("normal forms use different double-prime alphabets")
    boundary = tuple(
        a + b for a, b in zip(left.blocks[-1], right.blocks[0])
    )
    return NormalForm(
        left.blocks[:-1] + (boundary,) + right.blocks[1:],
        left.primes + right.primes,
    )


def quotient_concatenate(
    left: Mapping[NormalForm, int],
    right: Mapping[NormalForm, int],
) -> Counter[NormalForm]:
    """Bilinear standard-coordinate quotient concatenation."""
    output = Counter()
    for left_form, left_coefficient in left.items():
        for right_form, right_coefficient in right.items():
            output[
                concatenate_forms(left_form, right_form)
            ] += left_coefficient * right_coefficient
    return _clean(output)


def quotient_shuffle(
    left: Mapping[NormalForm, int],
    right: Mapping[NormalForm, int],
    prime_alphabet: Sequence[str],
    doubleprime_alphabet: Sequence[str],
) -> Counter[NormalForm]:
    """Tiny quotient shuffle oracle obtained by shuffling representatives."""
    output = Counter()
    for left_form, left_coefficient in left.items():
        for right_form, right_coefficient in right.items():
            shuffled = shuffle_words(
                canonical_word(left_form, doubleprime_alphabet),
                canonical_word(right_form, doubleprime_alphabet),
            )
            for form, coefficient in apply_q(
                shuffled, prime_alphabet, doubleprime_alphabet
            ).items():
                output[form] += (
                    left_coefficient * right_coefficient * coefficient
                )
    return _clean(output)


def _interleave(left, right, left_positions):
    left_positions = frozenset(left_positions)
    left_iterator = iter(left)
    right_iterator = iter(right)
    return tuple(
        next(left_iterator) if position in left_positions
        else next(right_iterator)
        for position in range(len(left) + len(right))
    )


def _multibinomial_merge(left, right):
    coefficient = 1
    for left_count, right_count in zip(left, right):
        coefficient *= comb(left_count + right_count, left_count)
    return coefficient


def hat_gamma(
    left: Mapping[NormalForm, int],
    right: Mapping[NormalForm, int],
) -> Counter[NormalForm]:
    """Normal-form hat-Gamma from Lemma 5.8."""
    output = Counter()
    for alpha, alpha_coefficient in left.items():
        for beta, beta_coefficient in right.items():
            if len(alpha.blocks[0]) != len(beta.blocks[0]):
                raise ValueError(
                    "normal forms use different double-prime alphabets"
                )
            n_left = len(alpha.primes)
            n_right = len(beta.primes)
            terminal = tuple(
                a + b for a, b in zip(alpha.blocks[-1], beta.blocks[-1])
            )
            coefficient = (
                alpha_coefficient
                * beta_coefficient
                * _multibinomial_merge(
                    alpha.blocks[-1], beta.blocks[-1]
                )
            )
            for left_positions in combinations(
                range(n_left + n_right), n_left
            ):
                blocks = _interleave(
                    alpha.blocks[:-1], beta.blocks[:-1], left_positions
                ) + (terminal,)
                primes = _interleave(
                    alpha.primes, beta.primes, left_positions
                )
                output[NormalForm(blocks, primes)] += coefficient
    return _clean(output)


def polynomial_concatenate(
    left: Mapping[Word, int], right: Mapping[Word, int]
) -> WordPolynomial:
    """Bilinear ordinary concatenation used only for quotient checks."""
    output = Counter()
    for left_word, left_coefficient in left.items():
        for right_word, right_coefficient in right.items():
            output[left_word + right_word] += (
                left_coefficient * right_coefficient
            )
    return _clean(output)


__all__ = [
    "NormalForm",
    "QuotientLinearMap",
    "QuotientProjection",
    "apply_q",
    "apply_q_transpose",
    "canonical_word",
    "concatenate_forms",
    "gamma_words",
    "hat_gamma",
    "hat_psi_image_of_representative",
    "hat_psi_map",
    "normal_form",
    "ordinary_words",
    "polynomial_concatenate",
    "psi_row",
    "quotient_basis",
    "quotient_class_size",
    "quotient_concatenate",
    "quotient_shuffle",
    "representative_words",
    "shuffle_words",
    "sym_rank",
]
