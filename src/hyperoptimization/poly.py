"""Exact polynomial arithmetic over the rationals, and Sturm's theorem.

The quorum theorems ask how many roots a polynomial of degree m has in (0, 1). A float
grid can miss two roots that sit closer together than its spacing, so the count is done
exactly: coefficients are `Fraction`, division is exact, and Sturm's theorem counts the
distinct real roots in an interval from sign changes along a chain of remainders.

A polynomial is a tuple of Fractions in ASCENDING order of degree, with no trailing zero.
The zero polynomial is the empty tuple and has degree -1. Nothing here is fast; it is
exact, and the degrees that occur (at most a few dozen) are small.
"""

from __future__ import annotations

from fractions import Fraction

Poly = tuple[Fraction, ...]


def _coefficient(value: object) -> Fraction:
    if type(value) is bool or not isinstance(value, (int, float, Fraction)):
        raise TypeError(f"a coefficient is an int, float or Fraction, not {value!r}")
    return Fraction(value)


def _trim(coefficients: list[Fraction]) -> Poly:
    while coefficients and coefficients[-1] == 0:
        coefficients.pop()
    return tuple(coefficients)


def poly(*coefficients: object) -> Poly:
    """The polynomial with these coefficients, constant term first. Floats are read exactly."""
    return _trim([_coefficient(c) for c in coefficients])


def degree(p: Poly) -> int:
    return len(p) - 1


def padd(a: Poly, b: Poly) -> Poly:
    n = max(len(a), len(b))
    return _trim([(a[i] if i < len(a) else Fraction(0)) + (b[i] if i < len(b) else Fraction(0))
                  for i in range(n)])


def pscale(a: Poly, factor: object) -> Poly:
    k = _coefficient(factor)
    return _trim([c * k for c in a])


def psub(a: Poly, b: Poly) -> Poly:
    return padd(a, pscale(b, -1))


def pmul(a: Poly, b: Poly) -> Poly:
    if not a or not b:
        return ()
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return _trim(out)


def pdiff(a: Poly) -> Poly:
    return _trim([i * a[i] for i in range(1, len(a))])


def peval(a: Poly, x: object) -> Fraction:
    point = _coefficient(x)
    value = Fraction(0)
    for c in reversed(a):
        value = value * point + c
    return value


def pdivmod(a: Poly, b: Poly) -> tuple[Poly, Poly]:
    """Quotient and remainder, so that a = q*b + r and deg r < deg b."""
    if not b:
        raise ZeroDivisionError("division by the zero polynomial")
    remainder = list(a)
    quotient = [Fraction(0)] * max(len(a) - len(b) + 1, 1)
    lead = b[-1]
    while len(remainder) >= len(b):
        shift = len(remainder) - len(b)
        factor = remainder[-1] / lead
        quotient[shift] = factor
        for i, c in enumerate(b):
            remainder[i + shift] -= factor * c
        remainder.pop()  # the leading term is now exactly zero
        while remainder and remainder[-1] == 0:
            remainder.pop()
    return _trim(quotient), _trim(remainder)


# ---------------------------------------------------------------------------
# Sturm's theorem.
# ---------------------------------------------------------------------------


def sturm_chain(p: Poly) -> list[Poly]:
    """p, p', then the negated remainders, until the remainder is zero (the last is gcd(p, p'))."""
    if not p:
        raise ValueError("the zero polynomial has no Sturm chain")
    chain = [p]
    derivative = pdiff(p)
    if not derivative:
        return chain
    chain.append(derivative)
    while True:
        _, remainder = pdivmod(chain[-2], chain[-1])
        if not remainder:
            return chain
        chain.append(pscale(remainder, -1))


def _variations(values: list[Fraction]) -> int:
    signs = [v > 0 for v in values if v != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def count_distinct_roots(p: Poly, a: object, b: object) -> int:
    """The number of distinct real roots of `p` strictly between `a` and `b`.

    Sturm's theorem: the sign variations of the chain at `a` less those at `b`. A repeated
    root counts once. Neither endpoint may be a root, because the theorem counts the
    half-open interval and a root at an endpoint would make the count depend on which end.
    """
    if not p:
        raise ValueError("the zero polynomial has every number as a root")
    lo, hi = _coefficient(a), _coefficient(b)
    if not lo < hi:
        raise ValueError("an interval needs a < b")
    if peval(p, lo) == 0 or peval(p, hi) == 0:
        raise ValueError("an endpoint is a root; move it or divide the root out first")
    chain = sturm_chain(p)
    return (_variations([peval(c, lo) for c in chain])
            - _variations([peval(c, hi) for c in chain]))


def bracket_root(p: Poly, lo: object, hi: object, *, width: object) -> tuple[Fraction, Fraction]:
    """A bracket of at most `width` around a root, found by exact bisection.

    `p` must change sign on [lo, hi]. The result is (lo', hi') with the sign change kept
    inside it, or (x, x) if bisection lands on a root exactly. Every comparison is on exact
    values, so the bracket is a proof that a root lies in it, not an estimate.
    """
    left, right, target = _coefficient(lo), _coefficient(hi), _coefficient(width)
    f_left, f_right = peval(p, left), peval(p, right)
    if f_left == 0:
        return (left, left)
    if f_right == 0:
        return (right, right)
    if (f_left > 0) == (f_right > 0):
        raise ValueError("bracket_root needs a sign change on the interval")
    while right - left > target:
        mid = (left + right) / 2
        f_mid = peval(p, mid)
        if f_mid == 0:
            return (mid, mid)
        if (f_mid > 0) == (f_left > 0):
            left, f_left = mid, f_mid
        else:
            right, f_right = mid, f_mid
    return (left, right)


def isolate_roots(p: Poly, a: object, b: object) -> list[tuple[Fraction, Fraction]]:
    """Disjoint intervals, in order, each holding exactly one distinct root of `p` in (a, b).

    Found by bisecting until Sturm's count on every piece is zero or one. A split point that
    happens to be a root is stepped aside by a thousandth of the interval, so every root is
    strictly inside its interval. Endpoints may not be roots (as for `count_distinct_roots`).
    """
    lo, hi = _coefficient(a), _coefficient(b)
    boxes: list[tuple[Fraction, Fraction]] = []

    def split(left: Fraction, right: Fraction, count: int) -> None:
        if count == 0:
            return
        if count == 1:
            boxes.append((left, right))
            return
        mid = (left + right) / 2
        while peval(p, mid) == 0:
            mid += (right - left) / 1000
        split(left, mid, count_distinct_roots(p, left, mid))
        split(mid, right, count_distinct_roots(p, mid, right))

    split(lo, hi, count_distinct_roots(p, lo, hi))
    return boxes
