"""Self-consistency checks for the independent quotient-word oracle."""

from __future__ import annotations

from collections import Counter

import pytest

from quotient_word_oracle import (
    NormalForm,
    QuotientProjection,
    apply_q,
    apply_q_transpose,
    canonical_word,
    concatenate_forms,
    hat_gamma,
    hat_psi_image_of_representative,
    hat_psi_map,
    normal_form,
    polynomial_concatenate,
    quotient_basis,
    quotient_class_size,
    quotient_concatenate,
    quotient_shuffle,
    representative_words,
    sym_rank,
)


PRIME = ("p",)
DOUBLE = ("x", "y")


def _matrix_product(left, right):
    return tuple(
        tuple(
            sum(
                left[row][inner] * right[inner][column]
                for inner in range(len(right))
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _identity(size):
    return tuple(
        tuple(int(row == column) for column in range(size))
        for row in range(size)
    )


def test_normal_form_representatives_and_note_rank_are_exact():
    word = ("y", "x", "p", "y", "p", "x", "x")
    form = normal_form(word, PRIME, DOUBLE)

    assert form == NormalForm(
        blocks=((1, 1), (0, 1), (2, 0)),
        primes=("p", "p"),
    )
    assert form.bidegree == (2, 5)
    assert canonical_word(form, DOUBLE) == (
        "x",
        "y",
        "p",
        "y",
        "p",
        "x",
        "x",
    )
    assert set(representative_words(form, DOUBLE)) == {
        word,
        ("x", "y", "p", "y", "p", "x", "x"),
    }
    assert quotient_class_size(form) == 2
    assert sym_rank(form) == 149

    basis = quotient_basis(PRIME, DOUBLE, (1, 2))
    assert tuple(sym_rank(item) for item in basis) == tuple(range(10))


def test_q_and_q_transpose_are_incidence_transposes():
    projection = QuotientProjection.build(PRIME, DOUBLE, (1, 2))
    assert all(sum(column) == 1 for column in zip(*projection.rows))

    quotient_vector = tuple(index - 3 for index in range(len(projection.forms)))
    ordinary_vector = projection.transpose_apply(quotient_vector)
    expected_ordinary = apply_q_transpose(
        dict(zip(projection.forms, quotient_vector)), DOUBLE
    )
    assert ordinary_vector == tuple(
        expected_ordinary[word] for word in projection.words
    )

    projected = projection.apply(ordinary_vector)
    assert projected == tuple(
        quotient_class_size(form) * value
        for form, value in zip(projection.forms, quotient_vector)
    )

    arbitrary = tuple((3 * index + 1) % 7 - 2 for index in range(len(projection.words)))
    q_arbitrary = projection.apply(arbitrary)
    assert sum(a * b for a, b in zip(q_arbitrary, quotient_vector)) == sum(
        a * b for a, b in zip(arbitrary, ordinary_vector)
    )


def test_standard_quotient_concatenation_merges_only_the_boundary_blocks():
    left_form = normal_form(("x", "p", "y"), PRIME, DOUBLE)
    right_form = normal_form(("y", "p", "x"), PRIME, DOUBLE)
    expected = NormalForm(
        blocks=((1, 0), (0, 2), (1, 0)),
        primes=("p", "p"),
    )
    assert concatenate_forms(left_form, right_form) == expected

    left_words = Counter(
        {
            ("x", "y", "p"): 2,
            ("y", "x", "p"): -1,
            ("p", "x", "y"): 3,
        }
    )
    right_words = Counter({("y", "p"): 2, ("p", "x"): -2})
    ordinary_then_q = apply_q(
        polynomial_concatenate(left_words, right_words), PRIME, DOUBLE
    )
    quotient_product = quotient_concatenate(
        apply_q(left_words, PRIME, DOUBLE),
        apply_q(right_words, PRIME, DOUBLE),
    )
    assert quotient_product == ordinary_then_q

    third = Counter(
        {normal_form(("x", "p"), PRIME, DOUBLE): 2}
    )
    left_associated = quotient_concatenate(quotient_product, third)
    right_associated = quotient_concatenate(
        apply_q(left_words, PRIME, DOUBLE),
        quotient_concatenate(
            apply_q(right_words, PRIME, DOUBLE), third
        ),
    )
    assert left_associated == right_associated


@pytest.mark.parametrize("bidegree", ((0, 2), (1, 1), (1, 2), (2, 1)))
def test_hat_psi_and_inverse_are_induced_and_mutually_inverse(bidegree):
    forward = hat_psi_map(PRIME, DOUBLE, bidegree)
    inverse = hat_psi_map(PRIME, DOUBLE, bidegree, inverse=True)
    identity = _identity(len(forward.basis))

    assert forward.basis == inverse.basis
    assert _matrix_product(forward.rows, inverse.rows) == identity
    assert _matrix_product(inverse.rows, forward.rows) == identity

    for form in forward.basis:
        images = {
            tuple(
                sorted(
                    hat_psi_image_of_representative(
                        representative,
                        PRIME,
                        DOUBLE,
                        bidegree,
                    ).items(),
                    key=repr,
                )
            )
            for representative in representative_words(form, DOUBLE)
        }
        assert len(images) == 1


def test_degree_one_prime_one_double_hat_psi_matches_the_row_recursions():
    prime = ("p",)
    double = ("x",)
    terminal = normal_form(("p", "x"), prime, double)
    initial = normal_form(("x", "p"), prime, double)
    forward = hat_psi_map(prime, double, (1, 1))
    inverse = hat_psi_map(prime, double, (1, 1), inverse=True)

    assert forward.basis == (terminal, initial)
    assert forward.rows == ((1, 1), (0, 1))
    assert inverse.rows == ((1, -1), (0, 1))
    assert forward.apply({initial: 1}) == Counter(
        {terminal: 1, initial: 1}
    )
    assert inverse.apply({initial: 1}) == Counter(
        {terminal: -1, initial: 1}
    )


def test_note_low_degree_standard_shuffle_and_hat_gamma_examples():
    prime = ("k", "l")
    double = ("x", "y")
    xk = normal_form(("x", "k"), prime, double)
    yl = normal_form(("y", "l"), prime, double)
    xkyl = normal_form(("x", "k", "y", "l"), prime, double)
    central_kl = normal_form(("x", "y", "k", "l"), prime, double)
    central_lk = normal_form(("x", "y", "l", "k"), prime, double)
    ylxk = normal_form(("y", "l", "x", "k"), prime, double)

    standard = quotient_shuffle(
        Counter({xk: 1}), Counter({yl: 1}), prime, double
    )
    assert standard == Counter(
        {
            xkyl: 1,
            central_kl: 2,
            central_lk: 2,
            ylxk: 1,
        }
    )
    assert hat_gamma(Counter({xk: 1}), Counter({yl: 1})) == Counter(
        {xkyl: 1, ylxk: 1}
    )


def test_hat_gamma_terminal_multibinomials_and_algebra_laws():
    unit = NormalForm(blocks=((0, 0),), primes=())
    terminal_x = NormalForm(blocks=((1, 0),), primes=())
    terminal_y = NormalForm(blocks=((0, 1),), primes=())
    terminal_xx = NormalForm(blocks=((2, 0),), primes=())
    terminal_xy = NormalForm(blocks=((1, 1),), primes=())

    assert hat_gamma({terminal_x: 1}, {terminal_x: 1}) == Counter(
        {terminal_xx: 2}
    )
    assert hat_gamma({terminal_x: 1}, {terminal_y: 1}) == Counter(
        {terminal_xy: 1}
    )

    a = Counter(
        {
            normal_form(("x", "p", "x"), PRIME, DOUBLE): 2,
            terminal_y: -1,
        }
    )
    b = Counter(
        {
            normal_form(("y", "p"), PRIME, DOUBLE): 1,
            terminal_x: 3,
        }
    )
    c = Counter(
        {
            normal_form(("p", "x"), PRIME, DOUBLE): -2,
            terminal_xy: 1,
        }
    )
    identity = Counter({unit: 1})

    assert hat_gamma(identity, a) == a
    assert hat_gamma(a, identity) == a
    assert hat_gamma(a, b) == hat_gamma(b, a)
    assert hat_gamma(hat_gamma(a, b), c) == hat_gamma(
        a, hat_gamma(b, c)
    )
    for form in hat_gamma(a, b):
        assert form.bidegree in {(2, 3), (1, 3), (1, 2), (0, 2)}
