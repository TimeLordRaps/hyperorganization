"""The quorum map, its fixed points, and the flow it generates.

Setting
    A body has m seats. Each seat is compromised independently with probability x. The
    body is CAPTURED when at least q of its seats are. The probability of capture is

        T(x) = P(Bin(m, x) >= q) = sum_{j >= q} C(m, j) x^j (1 - x)^(m - j).

    If the seats of the next body up are themselves bodies of this kind (a council of
    councils), the compromise probability one level up is T(x), and the levels are the
    iterates T, T o T, ... That is a renormalization-group flow in the ordinary sense: a
    one-dimensional map with fixed points, a stability for each, and a critical point
    between two phases.

What is proved here (FORM)
    * For 2 <= q <= m-1, T has exactly one fixed point in (0, 1), it is unstable, and 0 and
      1 attract. For every other q there is none (q = 1: capture by any one seat; q = m:
      by all; both are monotone). Proof: T' is proportional to x^(q-1) (1-x)^(m-q), so T is
      convex then concave with one inflection; f(x) = T(x) - x starts at 0 with slope -1 and
      ends at 0 with slope -1, so a convex stretch followed by a concave one crosses zero
      exactly once. `interior_fixed_points` also counts it exactly, by Sturm's theorem.
    * Duality: T_{m,q}(x) + T_{m,m-q+1}(1-x) = 1. A threshold that makes capture easy makes
      blocking hard by exactly the complementary amount. The odd majority q = (m+1)/2 is its
      own dual, so it is critical at exactly x = 1/2.
    * Below the critical point T(x) <= C(m, q) x^q (a union bound), so capture falls
      doubly exponentially with depth: each level squares-or-better the last.

What is observed, not proved (FRAME)
    Near the critical point the slope is T'(x_c) ~ A(theta) sqrt(m) with theta = q/m and
    A(theta) = phi(Phi^-1(theta)) / sqrt(theta (1 - theta)), phi and Phi the standard normal
    density and distribution. This is the de Moivre-Laplace limit read at the critical point.
    `critical_slope_float` is checked against it to three digits; it is not proved here.

Everything exact is a `Fraction`. Exact flow is limited to shallow depth because the
denominator of an iterate has degree m**depth; `flow_float` goes deeper, with ordinary
rounding.
"""

from __future__ import annotations

import math
from enum import Enum
from fractions import Fraction
from statistics import NormalDist

from . import poly as P
from .ground import OMEGA, Omega

_EXACT_FLOW_LIMIT = 10**6


class Fate(str, Enum):
    """Where the flow takes a starting compromise probability."""

    HEALTHY = "healthy"  # compromise decays to 0
    CAPTURED = "captured"  # compromise grows to 1
    CRITICAL = "critical"  # exactly on the unstable fixed point


def _size(m: object, q: object) -> None:
    if type(m) is not int or type(q) is not int:
        raise TypeError("m and q must be ints")
    if m < 1:
        raise ValueError("a body needs at least one seat")
    if q < 0:
        raise ValueError("a quorum cannot be negative")


def _probability(x: object) -> Fraction:
    if type(x) is bool or not isinstance(x, (int, float, Fraction)):
        raise TypeError(f"a probability is an int, float or Fraction, not {x!r}")
    value = Fraction(x)
    if not 0 <= value <= 1:
        raise ValueError(f"a probability lies in [0, 1], not {x!r}")
    return value


# ---------------------------------------------------------------------------
# The map.
# ---------------------------------------------------------------------------


def tail(m: int, q: int, x: object) -> Fraction:
    """T(x) = P(at least q of m independent seats are compromised), exactly."""
    _size(m, q)
    p = _probability(x)
    if q == 0:
        return Fraction(1)
    if q > m:
        return Fraction(0)
    rest = 1 - p
    return sum((math.comb(m, j) * p**j * rest ** (m - j) for j in range(q, m + 1)), Fraction(0))


def tail_poly(m: int, q: int) -> P.Poly:
    """T as a polynomial in x, constant term first."""
    _size(m, q)
    if q == 0:
        return P.poly(1)
    if q > m:
        return ()
    coefficients = [Fraction(0)] * (m + 1)
    for j in range(q, m + 1):
        for k in range(j, m + 1):  # (1 - x)^(m-j) contributes x^(k-j)
            coefficients[k] += math.comb(m, j) * math.comb(m - j, k - j) * (-1) ** (k - j)
    return P.poly(*coefficients)


def tail_slope(m: int, q: int, x: object) -> Fraction:
    """T'(x) = m C(m-1, q-1) x^(q-1) (1-x)^(m-q): the density of the q-th smallest seat."""
    _size(m, q)
    p = _probability(x)
    if q < 1 or q > m:
        return Fraction(0)
    return m * math.comb(m - 1, q - 1) * p ** (q - 1) * (1 - p) ** (m - q)


def dual(m: int, q: int) -> int:
    """The blocking threshold: the number of compromised seats that stops a body acting."""
    _size(m, q)
    return m - q + 1


def is_self_dual(m: int, q: int) -> bool:
    return dual(m, q) == q


# ---------------------------------------------------------------------------
# Fixed points.
# ---------------------------------------------------------------------------


def interior_fixed_points(m: int, q: int) -> int:
    """The number of fixed points of T strictly between 0 and 1, counted exactly.

    f = T - x vanishes at 0 and 1 (simply, for every real quorum except the one-seat body,
    which is the identity), so it is divided by x(x - 1) and Sturm's theorem counts the
    roots of what is left in (0, 1).
    """
    _size(m, q)
    if m == 1 and q == 1:
        raise ValueError("a one-seat body is the identity map: every point is a fixed point")
    if q == 0 or q > m:
        return 0  # T is constant 1 or 0, and x = 1 or 0 is not interior
    f = P.psub(tail_poly(m, q), P.poly(0, 1))
    g, remainder = P.pdivmod(f, P.poly(0, 1))
    assert not remainder, "f(0) = 0 for every real quorum"
    g, remainder = P.pdivmod(g, P.poly(-1, 1))
    assert not remainder, "f(1) = 0 for every real quorum"
    if P.degree(g) < 1:
        return 0
    assert P.peval(g, 0) != 0 and P.peval(g, 1) != 0, "the end fixed points are simple"
    return P.count_distinct_roots(g, 0, 1)


def critical_point(m: int, q: int, width: object = Fraction(1, 10**9)) -> tuple[Fraction, Fraction]:
    """The unstable fixed point, bracketed by exact signs of T(x) - x.

    f = T - x is negative to the left of the point and positive to the right (proved in the
    module docstring), so the sign of f at a rational says which side of the point it is on.
    Returns (lo, hi) with f(lo) <= 0 <= f(hi) and hi - lo <= width, or (x, x) on an exact hit.
    """
    if interior_fixed_points(m, q) == 0:
        raise ValueError(f"({m}, {q}) has no interior fixed point to be critical")
    target = Fraction(width)
    lo, hi = Fraction(0), Fraction(1)
    while hi - lo > target:
        mid = (lo + hi) / 2
        sign = tail(m, q, mid) - mid
        if sign == 0:
            return (mid, mid)
        if sign < 0:
            lo = mid
        else:
            hi = mid
    return (lo, hi)


def critical_slope(m: int, q: int) -> float:
    """T'(x_c), read at the middle of the default bracket."""
    lo, hi = critical_point(m, q)
    return float(tail_slope(m, q, (lo + hi) / 2))


# ---------------------------------------------------------------------------
# The flow.
# ---------------------------------------------------------------------------


def flow(m: int, q: int, x0: object, depth: int) -> list[Fraction]:
    """[x0, T(x0), ..., T^depth(x0)], exactly."""
    _size(m, q)
    if type(depth) is not int or depth < 0:
        raise ValueError("depth is a natural number")
    if m**depth > _EXACT_FLOW_LIMIT:
        raise ValueError(
            "exact flow grows doubly exponentially in depth (the denominator of an iterate has "
            f"degree m**depth = {m}**{depth}); use flow_float"
        )
    orbit = [_probability(x0)]
    for _ in range(depth):
        orbit.append(tail(m, q, orbit[-1]))
    return orbit


def flow_float(m: int, q: int, x0: float, depth: int) -> list[float]:
    """The same orbit in floats, for depths exact arithmetic cannot reach."""
    _size(m, q)
    if type(depth) is not int or depth < 0:
        raise ValueError("depth is a natural number")
    orbit = [float(_probability(x0))]
    for _ in range(depth):
        orbit.append(tail_float(m, q, orbit[-1]))
    return orbit


def fate(m: int, q: int, x0: object) -> Fate:
    """Which phase a starting probability flows to, from the exact sign of T(x0) - x0.

    On (0, 1) the sign of T - x is constant for an any- or all-quorum and changes once, from
    negative to positive, at the critical point otherwise. The ends are fixed points and are
    reported as the phase they belong to.
    """
    start = _probability(x0)
    if start == 0:
        return Fate.HEALTHY
    if start == 1:
        return Fate.CAPTURED
    sign = tail(m, q, start) - start
    if sign < 0:
        return Fate.HEALTHY
    if sign > 0:
        return Fate.CAPTURED
    return Fate.CRITICAL


def stage_count(m: int, q: int, x0: object, band: tuple[object, object], *,
                cap: int = 10**6) -> int | Omega:
    """The number of levels the flow takes to leave an open band around the critical point.

    This is the ordinal phase of a starting point: a natural number for every start the flow
    drives out of the band, `OMEGA` for a start that stays in it for ever, which is exactly a
    rational start that is itself a fixed point. The count diverges as the start approaches
    the critical point (about log(1/distance) / log(T'(x_c)) levels) and takes the value
    omega only on it. The count is computed in floats, so very small distances saturate at
    the float resolution; the fixed-point test is exact.
    """
    a, b = _probability(band[0]), _probability(band[1])
    if not 0 < a < b < 1:
        raise ValueError("the band must lie strictly inside (0, 1) so that it excludes both attractors")
    start = _probability(x0)
    if not a < start < b:
        raise ValueError("the start must lie inside the band")
    if tail(m, q, start) == start:
        return OMEGA
    low, high = float(a), float(b)
    x = float(start)
    count = 0
    while low < x < high:
        x = tail_float(m, q, x)
        count += 1
        if count > cap:
            raise RuntimeError(f"the orbit did not leave the band in {cap} levels; is the start "
                               "a float fixed point?")
    return count


# ---------------------------------------------------------------------------
# Floats and the large-m scaling.
# ---------------------------------------------------------------------------


def _log_binomial(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def tail_float(m: int, q: int, x: float) -> float:
    """T(x) in floats, summed in log space so that m in the thousands does not overflow."""
    _size(m, q)
    if q == 0:
        return 1.0
    if q > m or x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_x, log_rest = math.log(x), math.log1p(-x)
    return min(1.0, math.fsum(
        math.exp(_log_binomial(m, j) + j * log_x + (m - j) * log_rest) for j in range(q, m + 1)
    ))


def critical_point_float(m: int, q: int) -> float:
    """x_c in floats by bisection on the sign of T(x) - x."""
    _size(m, q)
    if not 2 <= q <= m - 1:
        raise ValueError(f"({m}, {q}) has no interior fixed point to be critical")
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if tail_float(m, q, mid) - mid < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def tail_slope_float(m: int, q: int, x: float) -> float:
    """T'(x) in floats, from the density form, for m too large for exact arithmetic."""
    _size(m, q)
    if q < 1 or q > m:
        return 0.0
    if x <= 0.0:
        return float(m) if q == 1 else 0.0
    if x >= 1.0:
        return float(m) if q == m else 0.0
    return m * math.exp(
        _log_binomial(m - 1, q - 1) + (q - 1) * math.log(x) + (m - q) * math.log1p(-x)
    )


def critical_slope_float(m: int, q: int) -> float:
    """T'(x_c) in floats, for m too large for exact arithmetic."""
    return tail_slope_float(m, q, critical_point_float(m, q))


def amplitude(theta: float) -> float:
    """phi(Phi^-1(theta)) / sqrt(theta (1 - theta)): the predicted T'(x_c) / sqrt(m)."""
    if not 0.0 < theta < 1.0:
        raise ValueError("theta lies strictly between 0 and 1")
    normal = NormalDist()
    return normal.pdf(normal.inv_cdf(theta)) / math.sqrt(theta * (1.0 - theta))
