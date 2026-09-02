"""Focused tests for backend-independent shear support combinatorics."""

from __future__ import annotations

from collections import Counter
from itertools import combinations, product

import numpy as np
import pytest

from tensordev.core.shear.symbolic import (
    GammaShuffleSupportTerm,
    TransformSupportTerm,
    adjacent_block_axis_permutations,
    canonical_placements,
    complement_positions,
    compose_permutations,
    gamma_axis_permutation,
    gamma_shuffle_pattern_support,
    gamma_units,
    interleave,
    invert_permutation,
    ordinary_shuffle_placements,
    psi_inverse_row_support,
    psi_inverse_support,
    psi_inverse_total_support,
    psi_row_support,
    psi_support,
    psi_total_support,
    right_generator_pattern_support,
    right_generator_support,
    transform_placement_sign,
    validate_gamma_shuffle_support,
    validate_generator_support,
    validate_transform_support,
)


def _class_words(prime_count, doubleprime_count, d_prime=2, d_doubleprime=2):
    degree = prime_count + doubleprime_count
    words = []
    for placement in canonical_placements(prime_count, doubleprime_count):
        prime_set = set(placement)
        alphabets = tuple(
            range(d_prime)
            if position in prime_set
            else range(d_prime, d_prime + d_doubleprime)
            for position in range(degree)
        )
        words.extend(product(*alphabets))
    return tuple(words)


def _input_word_from_transform_term(output_word, term, d_prime):
    output_prime_values = tuple(
        value for value in output_word if value < d_prime
    )
    output_doubleprime_values = tuple(
        value for value in output_word if value >= d_prime
    )
    input_doubleprime_values = tuple(
        output_doubleprime_values[label]
        for label in term.doubleprime_permutation
    )
    return interleave(
        output_prime_values,
        input_doubleprime_values,
        term.input_prime_positions,
    )


def _transform_matrix(
    prime_count,
    doubleprime_count,
    support,
    d_prime=2,
    d_doubleprime=2,
):
    words = _class_words(
        prime_count,
        doubleprime_count,
        d_prime=d_prime,
        d_doubleprime=d_doubleprime,
    )
    indices = {word: index for index, word in enumerate(words)}
    terms_by_output = {}
    for term in support:
        terms_by_output.setdefault(term.output_prime_positions, []).append(term)
    matrix = np.zeros((len(words), len(words)), dtype=np.int64)
    for output_index, output_word in enumerate(words):
        output_pattern = tuple(
            position
            for position, value in enumerate(output_word)
            if value < d_prime
        )
        for term in terms_by_output[output_pattern]:
            input_word = _input_word_from_transform_term(
                output_word,
                term,
                d_prime,
            )
            matrix[output_index, indices[input_word]] += term.coefficient
    return matrix


def _ordinary_word_shuffle(left, right):
    return tuple(
        interleave(left, right, placement)
        for placement in combinations(range(len(left) + len(right)), len(left))
    )


def _last_prime(word):
    for position in range(len(word) - 1, -1, -1):
        if word[position][0] == "P":
            return position
    return -1


def _recursive_gamma_oracle(left, right):
    """Independent implementation of the note's four-case recursion."""
    left_prime = _last_prime(left)
    right_prime = _last_prime(right)
    if left_prime < 0 and right_prime < 0:
        return _ordinary_word_shuffle(left, right)
    if right_prime < 0:
        prefix = left[: left_prime + 1]
        terminal = left[left_prime + 1 :]
        return tuple(
            prefix + shuffled
            for shuffled in _ordinary_word_shuffle(terminal, right)
        )
    if left_prime < 0:
        prefix = right[: right_prime + 1]
        terminal = right[right_prime + 1 :]
        return tuple(
            prefix + shuffled
            for shuffled in _ordinary_word_shuffle(left, terminal)
        )

    left_prefix = left[:left_prime]
    left_letter = left[left_prime]
    left_terminal = left[left_prime + 1 :]
    right_prefix = right[:right_prime]
    right_letter = right[right_prime]
    right_terminal = right[right_prime + 1 :]
    terminal_shuffles = _ordinary_word_shuffle(
        left_terminal,
        right_terminal,
    )
    terms = []
    for parent in _recursive_gamma_oracle(
        left_prefix + (left_letter,),
        right_prefix,
    ):
        terms.extend(
            parent + (right_letter,) + terminal
            for terminal in terminal_shuffles
        )
    for parent in _recursive_gamma_oracle(
        left_prefix,
        right_prefix + (right_letter,),
    ):
        terms.extend(
            parent + (left_letter,) + terminal
            for terminal in terminal_shuffles
        )
    return tuple(terms)


def test_basic_placement_interleaving_and_permutation_helpers():
    assert canonical_placements(2, 2) == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )
    assert complement_positions(5, (0, 3)) == (1, 2, 4)
    assert ordinary_shuffle_placements(2, 1) == ((0, 1), (0, 2), (1, 2))
    assert interleave(("a", "b"), ("x",), (0, 2)) == ("a", "x", "b")

    permutation = (1, 2, 0)
    inverse = invert_permutation(permutation)
    assert inverse == (2, 0, 1)
    assert compose_permutations(permutation, inverse) == (0, 1, 2)
    assert adjacent_block_axis_permutations(1, 1, 2) == (
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 2, 3, 1),
    )


def test_gamma_units_keep_each_preprime_block_attached():
    units, terminal = gamma_units((1, 4), 6)
    assert units == ((0, 1), (2, 3, 4))
    assert terminal == (5,)


def test_degree_two_transform_rows_match_the_note():
    assert psi_row_support((0,), 2) == (
        TransformSupportTerm((0,), (0,), (0,), 1),
        TransformSupportTerm((0,), (1,), (0,), 1),
    )
    assert psi_row_support((1,), 2) == (
        TransformSupportTerm((1,), (1,), (0,), 1),
    )
    assert psi_inverse_row_support((0,), 2) == (
        TransformSupportTerm((0,), (0,), (0,), 1),
        TransformSupportTerm((0,), (1,), (0,), -1),
    )
    assert psi_inverse_row_support((1,), 2) == (
        TransformSupportTerm((1,), (1,), (0,), 1),
    )


def test_inverse_peeling_reverses_the_removed_doubleprime_prefix():
    terms = psi_inverse_row_support((0,), 3)
    assert terms == (
        TransformSupportTerm((0,), (0,), (0, 1), 1),
        TransformSupportTerm((0,), (1,), (0, 1), -1),
        TransformSupportTerm((0,), (2,), (1, 0), 1),
    )


def test_noninvolutive_row_permutation_converts_to_transpose_axes():
    term = TransformSupportTerm(
        output_prime_positions=(1,),
        input_prime_positions=(3,),
        doubleprime_permutation=(1, 2, 0),
        coefficient=1,
    )
    # The row convention lists output D labels in input order.  Transpose axes
    # require its inverse before prime and D axes are placed in word order.
    assert term.axis_permutation == (2, 3, 0, 1)


@pytest.mark.parametrize(
    (
        "degree",
        "forward_terms",
        "forward_permutations",
        "inverse_terms",
        "inverse_permutations",
    ),
    (
        (0, 1, 1, 1, 1),
        (1, 2, 1, 2, 1),
        (2, 5, 2, 5, 2),
        (3, 15, 5, 14, 5),
        (4, 52, 15, 41, 13),
        (5, 203, 52, 122, 34),
        (6, 877, 203, 365, 89),
    ),
)
def test_transform_support_counts_and_distinct_permutations(
    degree,
    forward_terms,
    forward_permutations,
    inverse_terms,
    inverse_permutations,
):
    forward = psi_total_support(degree)
    inverse = psi_inverse_total_support(degree)
    assert len(forward) == forward_terms
    assert len({term.axis_permutation for term in forward}) == forward_permutations
    assert len(inverse) == inverse_terms
    assert len({term.axis_permutation for term in inverse}) == inverse_permutations


@pytest.mark.parametrize(
    ("prime_count", "doubleprime_count"),
    tuple(
        (prime_count, degree - prime_count)
        for degree in range(5)
        for prime_count in range(degree + 1)
    ),
)
def test_forward_and_inverse_symbolic_matrices_compose_exactly(
    prime_count,
    doubleprime_count,
):
    forward = _transform_matrix(
        prime_count,
        doubleprime_count,
        psi_support(prime_count, doubleprime_count),
    )
    inverse = _transform_matrix(
        prime_count,
        doubleprime_count,
        psi_inverse_support(prime_count, doubleprime_count),
    )
    np.testing.assert_array_equal(inverse @ forward, np.eye(len(forward), dtype=int))
    np.testing.assert_array_equal(forward @ inverse, np.eye(len(forward), dtype=int))


def test_transform_fibres_are_unique_and_inverse_sign_is_placement_parity():
    for prime_count in range(5):
        for doubleprime_count in range(5 - prime_count):
            forward = psi_support(prime_count, doubleprime_count)
            inverse = psi_inverse_support(prime_count, doubleprime_count)
            for support in (forward, inverse):
                fibres = {}
                for term in support:
                    key = term.output_prime_positions
                    fibre_key = (
                        term.input_prime_positions,
                        term.doubleprime_permutation,
                    )
                    assert fibre_key not in fibres.setdefault(key, set())
                    fibres[key].add(fibre_key)
            assert all(term.coefficient == 1 for term in forward)
            assert all(
                term.coefficient
                == transform_placement_sign(
                    term.output_prime_positions,
                    term.input_prime_positions,
                )
                for term in inverse
            )


def test_distinct_support_terms_preserve_realized_multiplicity():
    forward = _transform_matrix(1, 2, psi_support(1, 2), 1, 1)
    inverse = _transform_matrix(1, 2, psi_inverse_support(1, 2), 1, 1)
    # Pattern order is PDD, DPD, DDP.  The DPD row has two distinct terms
    # feeding DDP, even though a one-dimensional D alphabet realizes the same
    # coordinate for both permutations.
    assert forward[1, 2] == 2
    assert inverse[1, 2] == -2


def test_gamma_unit_interleavings_and_terminal_shuffle_match_note_example():
    terms = gamma_shuffle_pattern_support(3, 4, (1,), (2,))
    assert len(terms) == 4
    assert all(isinstance(term, GammaShuffleSupportTerm) for term in terms)
    assert tuple(gamma_axis_permutation(term, 3, 4) for term in terms) == (
        (0, 1, 3, 4, 5, 2, 6),
        (0, 1, 3, 4, 5, 6, 2),
        (3, 4, 5, 0, 1, 2, 6),
        (3, 4, 5, 0, 1, 6, 2),
    )


def test_gamma_pure_classes_retain_shuffle_multiplicity():
    doubleprime = gamma_shuffle_pattern_support(1, 1, (), ())
    prime = gamma_shuffle_pattern_support(1, 1, (0,), (0,))
    assert len(doubleprime) == 2
    assert {gamma_axis_permutation(term, 1, 1) for term in doubleprime} == {
        (0, 1),
        (1, 0),
    }
    assert len(prime) == 2
    assert {gamma_axis_permutation(term, 1, 1) for term in prime} == {
        (0, 1),
        (1, 0),
    }


def test_gamma_unit_builder_matches_four_case_recursion_exhaustively():
    for left_degree in range(4):
        for right_degree in range(4):
            if left_degree + right_degree > 5:
                continue
            for left_mask in range(1 << left_degree):
                left_prime_positions = tuple(
                    position
                    for position in range(left_degree)
                    if left_mask & (1 << position)
                )
                left_word = tuple(
                    (
                        "P" if position in left_prime_positions else "D",
                        position,
                    )
                    for position in range(left_degree)
                )
                for right_mask in range(1 << right_degree):
                    right_prime_positions = tuple(
                        position
                        for position in range(right_degree)
                        if right_mask & (1 << position)
                    )
                    right_word = tuple(
                        (
                            "P" if position in right_prime_positions else "D",
                            left_degree + position,
                        )
                        for position in range(right_degree)
                    )
                    expected = Counter(
                        tuple(letter[1] for letter in word)
                        for word in _recursive_gamma_oracle(
                            left_word,
                            right_word,
                        )
                    )
                    support = gamma_shuffle_pattern_support(
                        left_degree,
                        right_degree,
                        left_prime_positions,
                        right_prime_positions,
                    )
                    actual = Counter(
                        gamma_axis_permutation(
                            term,
                            left_degree,
                            right_degree,
                        )
                        for term in support
                    )
                    assert actual == expected


@pytest.mark.parametrize(
    ("output_degree", "term_count"),
    ((1, 2), (2, 5), (3, 12), (4, 28), (5, 64), (6, 144), (7, 320)),
)
def test_right_generator_support_counts(output_degree, term_count):
    support = right_generator_support(output_degree)
    assert len(support) == term_count
    assert all(
        sorted(term.dense_permutation) == list(range(output_degree))
        for term in support
    )


def test_right_generator_shuffles_only_blocks_adjacent_to_last_prime():
    terms = right_generator_pattern_support((1,), 3)
    assert tuple(
        (
            term.generator_is_prime,
            term.input_prime_positions,
            term.dense_permutation,
        )
        for term in terms
    ) == (
        (False, (1,), (0, 1, 2)),
        (True, (), (0, 2, 1)),
        (True, (), (1, 2, 0)),
    )
    assert right_generator_pattern_support((0,), 3)[-1].dense_permutation == (
        2,
        0,
        1,
    )
    assert right_generator_pattern_support((2,), 3)[0].dense_permutation == (
        0,
        1,
        2,
    )


def test_symbolic_validation_rejects_invalid_positions_and_permutations():
    with pytest.raises(ValueError, match="strictly increasing"):
        complement_positions(3, (1, 1))
    with pytest.raises(ValueError, match="outside"):
        gamma_units((3,), 3)
    with pytest.raises(ValueError, match="permutation"):
        invert_permutation((0, 0))
    with pytest.raises(ValueError, match="positive"):
        right_generator_support(0)


def test_support_uniqueness_validators_reject_duplicate_terms():
    transform = TransformSupportTerm((0,), (0,), (0,), 1)
    with pytest.raises(ValueError, match="duplicate"):
        validate_transform_support((transform, transform))

    gamma = gamma_shuffle_pattern_support(1, 0, (0,), ())[0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_gamma_shuffle_support((gamma, gamma))

    generator = right_generator_pattern_support((0,), 1)[0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_generator_support((generator, generator))
