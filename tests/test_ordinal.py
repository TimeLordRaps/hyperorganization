"""Ordinal phases below epsilon_0: Cantor normal form, and a convergence that outruns arithmetic.

`viability` gives every state a rank in omega+1. That is all a finite game can have. But a
descent can be longer than any bound a theory can name, and then the ordinal that certifies it
is bigger. The standard example is Goodstein's process: write a number in hereditary base b,
replace b by b+1 everywhere, subtract one, repeat. It always reaches zero, and Peano arithmetic
cannot prove that (Kirby and Paris), because the proof is a strictly descending chain of
ordinals below epsilon_0 and epsilon_0 is the proof-theoretic ordinal of Peano arithmetic.

What is checked
    * order: comparison of normal forms agrees with an independent valuation (replace omega by
      a number larger than every coefficient; an ordinal below omega^omega^... maps to an
      integer and the map is order-preserving); ordinal addition satisfies its laws and the
      absorption 1 + omega = omega that makes it non-commutative; the natural (Hessenberg) sum
      is the valuation's sum;
    * fundamental sequences have the standard values (omega[n] = n, omega^2[n] = omega n,
      omega^omega[n] = omega^n) and climb strictly toward their limit;
    * Goodstein: the sequences for 1, 2, 3 are the known ones, the first terms for 4 are worked
      by hand, the ordinal attached to each term evaluates back to it, and the ordinals descend
      strictly for thousands of terms of 4, 5 and 6 (whose lengths are 10^8-digit numbers and
      beyond, so the descent is certified, not run to the end);
    * a descent checker that reports the first step that fails to descend.
"""

from __future__ import annotations

import random

import pytest

from hyperoptimization import ordinal as O
from hyperoptimization.ground import OMEGA, Status, rank_key

Ord = O.Ordinal
ZERO, ONE, W = Ord.zero(), Ord.of(1), Ord.omega()


def w_pow(e):
    return Ord.omega_power(e)


def random_ordinal(rng, depth=2, width=3, coeff=4):
    if depth == 0 or rng.random() < 0.3:
        return Ord.of(rng.randint(0, coeff))
    terms = {}
    for _ in range(rng.randint(1, width)):
        terms[random_ordinal(rng, depth - 1, width, coeff)] = rng.randint(1, coeff)
    ordered = sorted(terms.items(), key=lambda kv: kv[0], reverse=True)
    return Ord.from_terms(ordered)


BASE = 16  # larger than every coefficient the generator makes (and than any sum of two of them)


# --- construction and printing ------------------------------------------------------------------------


def test_the_small_ordinals_print_the_way_they_are_written():
    assert str(ZERO) == "0" and str(Ord.of(7)) == "7" and str(W) == "ω"
    assert str(W + ONE) == "ω+1" and str(Ord.from_terms([(ONE, 2), (ZERO, 5)])) == "ω·2+5"
    assert str(W * 2) == "ω·2" and str(w_pow(Ord.of(2)) * 3) == "ω^2·3"
    assert str(w_pow(Ord.of(2))) == "ω^2" and str(w_pow(W)) == "ω^ω"
    assert str(w_pow(W + ONE) + W) == "ω^(ω+1)+ω"


def test_multiplying_by_a_natural_number_repeats_the_ordinal():
    assert W * 1 == W and W * 0 == ZERO and Ord.of(4) * 3 == Ord.of(12)
    assert (W + ONE) * 2 == W * 2 + ONE  # (w + 1) + (w + 1) = w*2 + 1
    rng = random.Random(80)
    for _ in range(100):
        a = random_ordinal(rng, depth=2)
        assert a * 3 == a + a + a
    with pytest.raises(TypeError):
        W * 1.5


def test_terms_must_descend_and_coefficients_be_positive():
    with pytest.raises(ValueError, match="descending"):
        Ord.from_terms([(ZERO, 1), (ONE, 1)])
    with pytest.raises(ValueError, match="descending"):
        Ord.from_terms([(ONE, 1), (ONE, 2)])
    with pytest.raises(ValueError, match="positive"):
        Ord.from_terms([(ONE, 0)])
    with pytest.raises(ValueError, match="natural"):
        Ord.of(-1)
    with pytest.raises(TypeError):
        Ord.of(1.5)
    with pytest.raises(TypeError):
        Ord.of(True)


def test_zero_successor_and_limit_are_told_apart():
    assert ZERO.is_zero and not ZERO.is_limit and not ZERO.is_successor
    assert Ord.of(3).is_successor and not Ord.of(3).is_limit
    assert W.is_limit and not W.is_successor
    assert (W + ONE).is_successor and (w_pow(W) + W).is_limit


def test_finite_ordinals_convert_back_to_integers_and_infinite_ones_refuse():
    assert Ord.of(12).to_int() == 12 and ZERO.to_int() == 0
    with pytest.raises(ValueError, match="finite"):
        W.to_int()


# --- order, against an independent valuation ---------------------------------------------------------------------


def test_comparison_agrees_with_the_valuation_and_is_a_total_order():
    rng = random.Random(81)
    pool = [random_ordinal(rng) for _ in range(150)]
    values = [a.evaluate(BASE) for a in pool]
    for a, va in zip(pool, values):
        for b, vb in zip(pool, values):
            assert (a < b) == (va < vb) and (a == b) == (va == vb) and (a > b) == (va > vb)
            assert (a <= b) == (va <= vb) and (a >= b) == (va >= vb)
    ordered = sorted(pool)
    assert all(x <= y for x, y in zip(ordered, ordered[1:]))
    assert len(set(values)) > 60  # the pool is not trivially all the same few ordinals


def test_a_handful_of_known_inequalities():
    assert ZERO < ONE < Ord.of(2) < W < W + ONE < W * 2
    assert Ord.of(10**9) < W
    assert W + Ord.of(10**9) < w_pow(Ord.of(2))
    assert w_pow(Ord.of(10**6)) < w_pow(W) < w_pow(W + ONE) < w_pow(w_pow(W))
    assert not (W < W) and W <= W and W >= W


def test_evaluate_rejects_a_base_too_small_to_separate_the_coefficients():
    with pytest.raises(ValueError, match="base"):
        Ord.of(5).evaluate(5)
    with pytest.raises(ValueError, match="base"):
        W.evaluate(1)


# --- addition: ordinary and natural ---------------------------------------------------------------------------------


def test_ordinary_addition_is_associative_and_absorbs_on_the_left():
    assert ONE + W == W and Ord.of(5) + W == W
    assert W + ONE != W and W + ONE > W
    assert W + W == Ord.from_terms([(ONE, 2)])
    assert (W + ONE) + (W + ONE) == Ord.from_terms([(ONE, 2), (ZERO, 1)])
    rng = random.Random(82)
    for _ in range(300):
        a, b, c = (random_ordinal(rng, depth=2) for _ in range(3))
        assert (a + b) + c == a + (b + c)
        assert a + ZERO == a == ZERO + a
        assert a + b >= a and a + b >= b
        if b > ZERO:
            assert a + b > a


def test_ordinary_addition_cancels_on_the_left_and_is_monotone_on_the_right():
    rng = random.Random(83)
    for _ in range(300):
        a, b, c = (random_ordinal(rng, depth=2) for _ in range(3))
        if b < c:
            assert a + b < a + c
        if a + b == a + c:
            assert b == c


def test_natural_sum_is_commutative_associative_monotone_and_the_valuations_sum():
    rng = random.Random(84)
    for _ in range(300):
        a, b, c = (random_ordinal(rng, depth=2) for _ in range(3))
        assert a.natural_sum(b) == b.natural_sum(a)
        assert a.natural_sum(b).natural_sum(c) == a.natural_sum(b.natural_sum(c))
        assert a.natural_sum(b).evaluate(BASE) == a.evaluate(BASE) + b.evaluate(BASE)
        if b < c:
            assert a.natural_sum(b) < a.natural_sum(c)
        assert a + b <= a.natural_sum(b)  # the natural sum is the larger of the two


def test_successor_and_predecessor():
    assert W.successor() == W + ONE and Ord.of(4).successor() == Ord.of(5)
    assert (W + Ord.of(3)).predecessor() == W + Ord.of(2)
    assert (W + ONE).predecessor() == W and Ord.of(1).predecessor() == ZERO
    for limit in (W, w_pow(W), w_pow(Ord.of(2)) + W):
        with pytest.raises(ValueError, match="no predecessor"):
            limit.predecessor()
    with pytest.raises(ValueError, match="no predecessor"):
        ZERO.predecessor()


# --- fundamental sequences ------------------------------------------------------------------------------------------------


def test_fundamental_sequences_have_the_standard_values():
    for n in range(1, 6):
        assert W.fundamental(n) == Ord.of(n)
        assert w_pow(Ord.of(2)).fundamental(n) == W * n  # omega^2[n] = omega n
        assert w_pow(W).fundamental(n) == w_pow(Ord.of(n))  # omega^omega[n] = omega^n
        assert (w_pow(W) + W).fundamental(n) == w_pow(W) + Ord.of(n)
        assert w_pow(w_pow(W)).fundamental(n) == w_pow(w_pow(Ord.of(n)))
    assert w_pow(W + ONE).fundamental(3) == Ord.from_terms([(W, 3)])  # omega^(omega+1)[3] = omega^omega 3
    # the zeroth term, by the same rules
    assert W.fundamental(0) == ZERO and w_pow(Ord.of(2)).fundamental(0) == ZERO
    assert w_pow(W).fundamental(0) == ONE


def test_fundamental_sequences_climb_strictly_and_stay_below_their_limit():
    rng = random.Random(85)
    checked = 0
    while checked < 120:
        a = random_ordinal(rng, depth=3)
        if not a.is_limit:
            continue
        previous = a.fundamental(1)
        for n in range(2, 6):
            nxt = a.fundamental(n)
            assert previous < nxt < a, (a, n)
            previous = nxt
        checked += 1


def test_only_a_limit_has_a_fundamental_sequence():
    with pytest.raises(ValueError, match="limit"):
        (W + ONE).fundamental(3)
    with pytest.raises(ValueError, match="limit"):
        ZERO.fundamental(3)
    with pytest.raises(ValueError, match="natural"):
        W.fundamental(-1)


# --- ranks as ordinals ----------------------------------------------------------------------------------------------------------


def test_a_viability_rank_is_an_ordinal_and_the_order_is_preserved():
    ranks = [0, 1, 7, OMEGA, 3]
    ordinals = [O.rank_ordinal(r) for r in ranks]
    assert ordinals == [ZERO, ONE, Ord.of(7), W, Ord.of(3)]
    assert sorted(range(5), key=lambda i: rank_key(ranks[i])) == sorted(range(5), key=lambda i: ordinals[i])
    with pytest.raises(TypeError):
        O.rank_ordinal(-1)


# --- Goodstein -----------------------------------------------------------------------------------------------------------------------


def test_the_hereditary_ordinal_of_small_numbers():
    assert O.hereditary_ordinal(0, 2) == ZERO
    assert O.hereditary_ordinal(1, 2) == ONE
    assert O.hereditary_ordinal(2, 2) == W  # 2 = 2^1
    assert O.hereditary_ordinal(3, 2) == W + ONE
    assert O.hereditary_ordinal(4, 2) == w_pow(W)  # 4 = 2^2 = 2^(2^1)
    assert O.hereditary_ordinal(10, 2) == w_pow(W + ONE) + W  # 10 = 2^3 + 2 = 2^(2+1) + 2^1
    assert O.hereditary_ordinal(26, 3) == Ord.from_terms([(Ord.of(2), 2), (ONE, 2), (ZERO, 2)])
    for n in range(0, 200):
        for base in (2, 3, 5, 10):
            assert O.hereditary_ordinal(n, base).evaluate(base) == n


def test_the_hereditary_ordinal_needs_a_base_of_at_least_two_and_a_natural_number():
    with pytest.raises(ValueError):
        O.hereditary_ordinal(5, 1)
    with pytest.raises(ValueError):
        O.hereditary_ordinal(-1, 2)


def test_bumping_the_base_is_hereditary():
    assert O.bump(4, 2) == 27  # 2^2 -> 3^3
    assert O.bump(3, 2) == 4  # 2 + 1 -> 3 + 1
    assert O.bump(26, 3) == 2 * 4**2 + 2 * 4 + 2 + 0  # digits unchanged, base 3 -> 4
    assert O.bump(0, 5) == 0


def values(n, limit=None):
    return [value for value, _base, _ord in O.goodstein_sequence(n, limit=limit)]


def test_the_sequences_that_end_quickly_are_the_known_ones():
    assert values(1) == [1, 0]
    assert values(2) == [2, 2, 1, 0]
    assert values(3) == [3, 3, 3, 2, 1, 0]


def test_the_first_terms_of_the_sequence_from_four_are_worked_by_hand():
    # 4 = 2^2 -> 3^3 - 1 = 26 -> 2*3^2+2*3+2 -> 2*4^2+2*4+2 - 1 = 41 -> ... 60, 83
    assert values(4, limit=5) == [4, 26, 41, 60, 83, 109]


def test_every_term_carries_an_ordinal_that_evaluates_back_to_it_and_the_ordinals_descend():
    for n in (1, 2, 3, 4, 5, 6, 7):
        previous = None
        for step, (value, base, ordinal) in enumerate(O.goodstein_sequence(n, limit=1500)):
            assert base == step + 2
            assert ordinal.evaluate(base) == value
            if previous is not None:
                assert ordinal < previous, (n, step)
            previous = ordinal


def test_the_starting_ordinals_of_the_first_few_numbers():
    start = {n: next(iter(O.goodstein_sequence(n)))[2] for n in range(1, 8)}
    assert start[1] == ONE and start[2] == W and start[3] == W + ONE
    assert start[4] == w_pow(W) and start[5] == w_pow(W) + ONE
    assert start[6] == w_pow(W) + W and start[7] == w_pow(W) + W + ONE


def test_the_descent_checker_accepts_a_descent_and_names_the_first_step_that_is_not_one():
    sequence = [ordinal for _v, _b, ordinal in O.goodstein_sequence(3)]
    assert O.check_descent(sequence).status is Status.HOLDS_IN_MODEL
    broken = [W + ONE, W, W, Ord.of(3)]  # the third step stands still
    verdict = O.check_descent(broken)
    assert verdict.status is Status.FAILS_IN_MODEL and verdict.witness == ("step 2: ω !> ω",)
    rising = [Ord.of(3), Ord.of(4)]
    assert "step 1: 3 !> 4" in O.check_descent(rising).witness
    assert O.check_descent([W]).status is Status.HOLDS_IN_MODEL  # a single term descends vacuously
    assert O.check_descent([]).status is Status.HOLDS_IN_MODEL


def test_the_descent_checker_catches_a_tampered_goodstein_trace():
    sequence = [ordinal for _v, _b, ordinal in O.goodstein_sequence(4, limit=200)]
    sequence[57] = sequence[56]  # a stall
    verdict = O.check_descent(sequence)
    assert verdict.status is Status.FAILS_IN_MODEL and verdict.witness[0].startswith("step 57")


def test_a_goodstein_step_needs_a_natural_number_and_a_limit_that_is_one():
    with pytest.raises(ValueError):
        list(O.goodstein_sequence(-1))
    with pytest.raises(ValueError):
        list(O.goodstein_sequence(3, limit=-1))
