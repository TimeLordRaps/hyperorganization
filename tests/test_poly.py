"""Exact polynomial arithmetic and Sturm root counting, held to independent oracles.

The quorum theorems lean on one claim about a polynomial of degree m: that it has
exactly one root in (0, 1). A float grid cannot establish that, so the count is done by
Sturm's theorem in rational arithmetic. These tests do not check Sturm against itself.
They build polynomials whose roots are known by construction (a product of chosen
linear factors, with multiplicities, times factors with no real root) and count those
roots directly, then check the arithmetic by evaluating at points, which is a
homomorphism and so does not share code with the coefficient routines.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

import pytest

from hyperoptimization import poly as P


def test_zero_polynomial_is_the_empty_tuple_and_trailing_zeros_vanish():
    assert P.poly() == () and P.poly(0, 0) == ()
    assert P.poly(1, 2, 0, 0) == (F(1), F(2))
    assert P.degree(()) == -1 and P.degree(P.poly(5)) == 0 and P.degree(P.poly(0, 0, 3)) == 2


def test_coefficients_are_converted_exactly():
    assert P.poly(0.5, 0.25) == (F(1, 2), F(1, 4))
    assert P.poly(F(1, 3)) == (F(1, 3),)
    with pytest.raises(TypeError):
        P.poly("1")
    with pytest.raises(TypeError):
        P.poly(True)


def random_poly(rng, max_degree=6):
    return P.poly(*[F(rng.randint(-9, 9), rng.randint(1, 5)) for _ in range(rng.randint(0, max_degree + 1))])


def test_evaluation_is_a_ring_homomorphism():
    rng = random.Random(11)
    points = [F(0), F(1), F(-2), F(1, 2), F(7, 3), F(-5, 4)]
    for _ in range(200):
        a, b = random_poly(rng), random_poly(rng)
        for x in points:
            assert P.peval(P.padd(a, b), x) == P.peval(a, x) + P.peval(b, x)
            assert P.peval(P.psub(a, b), x) == P.peval(a, x) - P.peval(b, x)
            assert P.peval(P.pmul(a, b), x) == P.peval(a, x) * P.peval(b, x)
            assert P.peval(P.pscale(a, F(3, 7)), x) == F(3, 7) * P.peval(a, x)


def test_evaluation_matches_the_naive_power_sum():
    rng = random.Random(12)
    for _ in range(100):
        a = random_poly(rng)
        x = F(rng.randint(-5, 5), rng.randint(1, 4))
        assert P.peval(a, x) == sum(c * x**i for i, c in enumerate(a))


def test_division_reassembles_and_the_remainder_is_smaller():
    rng = random.Random(13)
    for _ in range(300):
        a, b = random_poly(rng, 8), random_poly(rng, 4)
        if not b:
            continue
        q, r = P.pdivmod(a, b)
        assert P.padd(P.pmul(q, b), r) == a
        assert P.degree(r) < P.degree(b)


def test_division_by_zero_is_refused():
    with pytest.raises(ZeroDivisionError):
        P.pdivmod(P.poly(1, 1), ())


def test_derivative_of_a_power_and_the_product_rule():
    for n in range(0, 8):
        monomial = P.poly(*([0] * n + [1]))
        expected = P.poly(*([0] * (n - 1) + [n])) if n else ()
        assert P.pdiff(monomial) == expected
    rng = random.Random(14)
    for _ in range(200):
        a, b = random_poly(rng), random_poly(rng)
        left = P.pdiff(P.pmul(a, b))
        right = P.padd(P.pmul(P.pdiff(a), b), P.pmul(a, P.pdiff(b)))
        assert left == right


# --- Sturm counting against roots known by construction -----------------------------------


def build(rng):
    """A polynomial with chosen rational roots (with multiplicity) and no-real-root factors."""
    roots = {}
    for _ in range(rng.randint(0, 5)):
        roots[F(rng.randint(-12, 12), rng.randint(1, 6))] = rng.randint(1, 3)
    p = P.poly(rng.choice([1, 2, -3, F(5, 7)]))
    for r, mult in roots.items():
        for _ in range(mult):
            p = P.pmul(p, P.poly(-r, 1))
    for _ in range(rng.randint(0, 2)):
        shift, sq = F(rng.randint(-3, 3)), F(rng.randint(1, 4))
        p = P.pmul(p, P.poly(shift * shift + sq, -2 * shift, 1))  # (x - s)^2 + c, no real root
    return p, roots


def test_sturm_counts_distinct_real_roots_known_by_construction():
    rng = random.Random(15)
    checked = 0
    while checked < 300:
        p, roots = build(rng)
        a, b = sorted([F(rng.randint(-14, 14), rng.randint(1, 6)) for _ in range(2)])
        if a == b or a in roots or b in roots:
            continue
        want = sum(1 for r in roots if a < r < b)
        assert P.count_distinct_roots(p, a, b) == want, (p, a, b, roots)
        checked += 1


def test_a_repeated_root_counts_once():
    p = P.pmul(P.pmul(P.poly(F(-1, 2), 1), P.poly(F(-1, 2), 1)), P.poly(F(-3, 4), 1))
    assert P.count_distinct_roots(p, F(0), F(1)) == 2


def test_sturm_refuses_an_endpoint_that_is_a_root_and_the_zero_polynomial():
    p = P.poly(-1, 1)  # x - 1
    with pytest.raises(ValueError, match="endpoint"):
        P.count_distinct_roots(p, F(0), F(1))
    with pytest.raises(ValueError, match="zero polynomial"):
        P.count_distinct_roots((), F(0), F(1))
    with pytest.raises(ValueError, match="a < b"):
        P.count_distinct_roots(p, F(3), F(2))


def test_a_nonzero_constant_has_no_roots():
    assert P.count_distinct_roots(P.poly(4), F(-5), F(5)) == 0


def test_bracket_root_isolates_an_irrational_root_by_exact_signs():
    p = P.poly(-2, 0, 1)  # x^2 - 2
    lo, hi = P.bracket_root(p, F(1), F(2), width=F(1, 10**9))
    assert hi - lo <= F(1, 10**9)
    assert P.peval(p, lo) < 0 < P.peval(p, hi)
    assert lo * lo < 2 < hi * hi


def test_bracket_root_returns_a_point_when_it_lands_on_the_root():
    p = P.poly(-1, 4)  # 4x - 1, root 1/4 is hit by bisection of (0, 1/2)
    assert P.bracket_root(p, F(0), F(1, 2), width=F(1, 100)) == (F(1, 4), F(1, 4))


def test_bracket_root_demands_a_sign_change():
    with pytest.raises(ValueError, match="sign change"):
        P.bracket_root(P.poly(1, 0, 1), F(-1), F(1), width=F(1, 10))


def test_isolate_roots_separates_every_distinct_root_known_by_construction():
    rng = random.Random(16)
    checked = 0
    while checked < 200:
        p, roots = build(rng)
        a, b = sorted([F(rng.randint(-14, 14), rng.randint(1, 6)) for _ in range(2)])
        if a == b or a in roots or b in roots:
            continue
        want = sorted(r for r in roots if a < r < b)
        boxes = P.isolate_roots(p, a, b)
        assert len(boxes) == len(want), (p, a, b, roots)
        for (lo, hi), root in zip(boxes, want):
            assert a <= lo < root < hi <= b
        assert all(first[1] <= second[0] for first, second in zip(boxes, boxes[1:]))
        checked += 1


def test_isolate_roots_steps_around_a_split_point_that_is_a_root():
    p = P.pmul(P.poly(F(-1, 2), 1), P.poly(F(-1, 4), 1))  # roots at 1/4 and 1/2; 1/2 is the first midpoint
    boxes = P.isolate_roots(p, F(0), F(1))
    assert len(boxes) == 2
    assert boxes[0][0] < F(1, 4) < boxes[0][1] <= boxes[1][0] < F(1, 2) < boxes[1][1]


def test_the_zero_polynomial_has_no_sturm_chain():
    with pytest.raises(ValueError, match="zero polynomial"):
        P.sturm_chain(())


def test_bracket_root_returns_an_endpoint_that_is_itself_a_root():
    p = P.poly(-1, 1)  # x - 1
    assert P.bracket_root(p, F(1), F(2), width=F(1, 10)) == (F(1), F(1))
    assert P.bracket_root(p, F(0), F(1), width=F(1, 10)) == (F(1), F(1))
