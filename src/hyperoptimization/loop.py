"""The closed loop of checks: a body whose overseers are bodies of its own kind.

L2 (hyperethics) shows that a finite governance system with no unchecked apex must close
its checks into a loop. This module asks what the loop costs and when it holds, in the
mean-field model of `quorum`: every body is compromised with the same probability p, and a
body is overseen by a quorum body of m seats that is captured when q of them are. With pi
the probability that a body is attacked directly, there are two kinds of link between a body
and its overseers, and so two self-consistency equations.

    COMMAND   the overseers can command the body, so capturing them captures it:
                  p = G(p) = pi + (1 - pi) T(p).
              Compromise propagates. p = 1 is always a fixed point and is where the loop
              ends up when it fails.
    VETO      the overseers can only block the body, so a direct attack succeeds only if
              they fail as well:
                  p = G(p) = pi T(p).
              Compromise needs both. p = 0 is always a fixed point and is where the loop
              ends up when it holds.

One equation in two variables
    With s = 1 - p, the probability a body is sound, the COMMAND loop is
    s = (1 - pi) T*(s) where T* = T_{m, m-q+1} is the dual quorum: it is the VETO loop with
    pi replaced by 1 - pi. The identity 1 - G_command(1 - s) = G_veto*(s) is exact and is
    tested as such. A captured checker that commands is the mirror image of one that only
    blocks, with "compromised" and "sound" swapped and the quorum dualised.

The tower and the loop
    A hierarchy of depth L ending in an unchecked apex has the compromise probability of
    its bottom body given by L steps of the map from p_0 = pi. The folded loop, which obeys
    the no-apex rule, is the limit of that iteration. For COMMAND the iterates increase to
    the least fixed point, so a finite hierarchy is never worse than the loop it folds into;
    `closing_price` is the gap, and it shrinks at the slope of the map at the limit.

Thresholds (FRAME: mean-field, independent seats, every body alike)
    The loop has a healthy state iff pi is below a critical pressure pi*, and the failure is
    discontinuous: a saddle-node, where the healthy fixed point meets the unstable one and
    both vanish (COMMAND) or appear (VETO). With Lambda(m, q) = sup_c T(c)/c, attained where
    the tangent from the origin touches T,
        pi*_veto    = 1 / Lambda(m, q),
        pi*_command = 1 - 1 / Lambda(m, m - q + 1).
    For a self-dual quorum the two sum to exactly one. As m grows both close on q/m, with a
    gap of order sqrt(ln(m) / m) (checked numerically).
    Just past the threshold the loop still looks healthy for a long time: the iteration
    crawls through the ghost of the vanished fixed point and the number of levels before it
    crosses diverges like (pi - pi*)^(-1/2).

What this is not
    The model has no geometry. A real ring or graph of bodies has correlated neighbours, and
    mean-field theory is known to misplace thresholds there. This module says nothing about
    that, and the decision ledger records it as a conceptual block.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from . import poly as P
from . import quorum as Q


class Link(str, Enum):
    """What an overseer can do to the body it oversees."""

    COMMAND = "command"
    VETO = "veto"


class Outcome(str, Enum):
    """Where the tower started at the direct-attack level ends up."""

    HEALTHY = "healthy"
    COLLAPSED = "collapsed"


class Kind(str, Enum):
    STABLE = "stable"
    UNSTABLE = "unstable"
    SEMISTABLE = "semistable"  # a double root: the saddle-node itself


@dataclass(frozen=True, slots=True)
class FixedPoint:
    """An interior fixed point, bracketed by exact rational endpoints."""

    lo: Fraction
    hi: Fraction
    kind: Kind

    @property
    def stable(self) -> bool:
        return self.kind is Kind.STABLE


_EXACT_TOWER_LIMIT = 10**6


def _link(link: object) -> Link:
    if type(link) is not Link:
        raise ValueError(f"link must be a Link, not {link!r}")
    return link


def _pi(pi: object) -> Fraction:
    if type(pi) is bool or not isinstance(pi, (int, float, Fraction)):
        raise TypeError(f"pi must be an int, float or Fraction, not {pi!r}")
    value = Fraction(pi)
    if not 0 < value < 1:
        raise ValueError(f"pi is a probability strictly inside (0, 1), not {pi!r}")
    return value


def step(link: Link, m: int, q: int, pi: object, p: object) -> Fraction:
    """One level: the compromise probability of a body given that of its overseers' seats."""
    link, direct = _link(link), _pi(pi)
    captured = Q.tail(m, q, p)
    if link is Link.COMMAND:
        return direct + (1 - direct) * captured
    return direct * captured


def step_float(link: Link, m: int, q: int, pi: float, p: float) -> float:
    link = _link(link)
    captured = Q.tail_float(m, q, p)
    return pi + (1.0 - pi) * captured if link is Link.COMMAND else pi * captured


def tower(link: Link, m: int, q: int, pi: object, depth: int) -> list[Fraction]:
    """[p_0, ..., p_depth] with p_0 = pi, exactly. Exact depth is limited as in `quorum.flow`."""
    if type(depth) is not int or depth < 0:
        raise ValueError("depth is a natural number")
    if m**depth > _EXACT_TOWER_LIMIT:
        raise ValueError(
            "exact towers grow doubly exponentially in depth (the denominator of a level has "
            f"degree m**depth = {m}**{depth}); use tower_float"
        )
    direct = _pi(pi)
    orbit = [direct]
    for _ in range(depth):
        orbit.append(step(link, m, q, direct, orbit[-1]))
    return orbit


def tower_float(link: Link, m: int, q: int, pi: float, depth: int) -> list[float]:
    if type(depth) is not int or depth < 0:
        raise ValueError("depth is a natural number")
    direct = float(_pi(pi))
    orbit = [direct]
    for _ in range(depth):
        orbit.append(step_float(link, m, q, direct, orbit[-1]))
    return orbit


# ---------------------------------------------------------------------------
# Fixed points, exactly.
# ---------------------------------------------------------------------------


def fixed_point_poly(link: Link, m: int, q: int, pi: object) -> P.Poly:
    """G(p) - p as a polynomial: its roots are the loop's fixed points."""
    link, direct = _link(link), _pi(pi)
    t = Q.tail_poly(m, q)
    if link is Link.COMMAND:
        g = P.padd(P.pscale(t, 1 - direct), P.poly(direct))
    else:
        g = P.pscale(t, direct)
    return P.psub(g, P.poly(0, 1))


def _divide_out(f: P.Poly, root: int) -> P.Poly:
    factor = P.poly(-root, 1)
    while P.peval(f, root) == 0:
        f, remainder = P.pdivmod(f, factor)
        assert not remainder
    return f


def _interior_part(link: Link, m: int, q: int, pi: object) -> P.Poly:
    """The fixed-point polynomial with its end roots divided out (1 for COMMAND, 0 for VETO)."""
    f = fixed_point_poly(link, m, q, pi)
    return _divide_out(f, 1 if link is Link.COMMAND else 0)


def count_fixed_points(link: Link, m: int, q: int, pi: object) -> int:
    """The number of distinct fixed points strictly between 0 and 1, counted exactly.

    The end fixed point (1 for COMMAND, 0 for VETO) always exists and is divided out; the
    other end is never a fixed point for pi strictly inside (0, 1).
    """
    g = _interior_part(link, m, q, pi)
    if P.degree(g) < 1:
        return 0
    other = 0 if link is Link.COMMAND else 1
    assert P.peval(g, other) != 0, "the far end is not a fixed point"
    assert P.peval(g, 1 - other) != 0, "the end root was divided out completely"
    return P.count_distinct_roots(g, 0, 1)


def fixed_points(link: Link, m: int, q: int, pi: object,
                 width: object = Fraction(1, 10**9)) -> list[FixedPoint]:
    """The interior fixed points in increasing order, each bracketed and classified exactly.

    A fixed point is STABLE if G(p) - p is positive just left of it and negative just right
    (the map pulls toward it), UNSTABLE if the signs are the other way, and SEMISTABLE if the
    sign does not change (a double root, which happens only exactly at the threshold).
    """
    g = _interior_part(link, m, q, pi)
    if P.degree(g) < 1:
        return []
    target = Fraction(width)
    found: list[FixedPoint] = []
    f = fixed_point_poly(link, m, q, pi)
    for box_lo, box_hi in P.isolate_roots(g, 0, 1):
        # The box holds exactly one distinct root, so the sign of f = G - p at its ends says how
        # the map behaves around it. Reading f itself, not g, keeps the divided-out end factor
        # from flipping the answer.
        left, right = P.peval(f, box_lo), P.peval(f, box_hi)
        if left > 0 > right:
            kind = Kind.STABLE
        elif left < 0 < right:
            kind = Kind.UNSTABLE
        else:
            kind = Kind.SEMISTABLE
        if kind is Kind.SEMISTABLE:
            # a double root has no sign change to bisect on, so narrow the box by Sturm counts
            lo, hi = box_lo, box_hi
            while hi - lo > target:
                mid = (lo + hi) / 2
                if P.peval(g, mid) == 0:
                    lo = hi = mid
                elif P.count_distinct_roots(g, lo, mid) == 1:
                    hi = mid
                else:
                    lo = mid
            found.append(FixedPoint(lo, hi, kind))
        else:
            lo, hi = P.bracket_root(g, box_lo, box_hi, width=target)
            found.append(FixedPoint(lo, hi, kind))
    return found


def outcome(link: Link, m: int, q: int, pi: object) -> Outcome:
    """Whether the tower started at p_0 = pi settles in the healthy state, exactly.

    COMMAND: the iterates rise to the least fixed point, which is interior iff any interior
    fixed point exists. VETO: they fall to the greatest, which is interior iff one exists.
    """
    interior = count_fixed_points(link, m, q, pi) > 0
    if _link(link) is Link.COMMAND:
        return Outcome.HEALTHY if interior else Outcome.COLLAPSED
    return Outcome.COLLAPSED if interior else Outcome.HEALTHY


# ---------------------------------------------------------------------------
# Thresholds.
# ---------------------------------------------------------------------------


def capture_gain(m: int, q: int) -> float:
    """Lambda(m, q) = sup over c in (0, 1] of T(c) / c.

    T/c has the sign of c T'(c) - T(c) for its derivative; that function is positive and then
    negative once (T is convex then concave), so the supremum is where it crosses zero. For
    q = 1 the supremum is the limit m as c -> 0; for q = m it is 1, reached as c -> 1.
    """
    Q._size(m, q)
    if q == 1:
        return float(m)
    if q == m:
        return 1.0
    if not 1 <= q <= m:
        raise ValueError("a real quorum has 1 <= q <= m")
    lo, hi = 1e-15, 1.0 - 1e-15
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid * Q.tail_slope_float(m, q, mid) - Q.tail_float(m, q, mid) > 0:
            lo = mid
        else:
            hi = mid
    c = (lo + hi) / 2
    return Q.tail_float(m, q, c) / c


def threshold(link: Link, m: int, q: int) -> float:
    """The critical direct-attack probability pi* at which the loop's healthy state disappears."""
    if _link(link) is Link.VETO:
        return 1.0 / capture_gain(m, q)
    return 1.0 - 1.0 / capture_gain(m, Q.dual(m, q))


# ---------------------------------------------------------------------------
# The tower in floats: limits, prices and delays.
# ---------------------------------------------------------------------------


def limit(link: Link, m: int, q: int, pi: object, *, cap: int = 10**6) -> float:
    """Where the tower started at p_0 = pi comes to rest, by iterating until it stops moving."""
    direct = float(_pi(pi))
    p = direct
    for _ in range(cap):
        nxt = step_float(link, m, q, direct, p)
        if nxt == p or abs(nxt - p) <= 1e-16 * max(1.0, abs(p)):
            return nxt
        p = nxt
    raise RuntimeError(f"the tower did not converge within {cap} levels; is pi at the threshold?")


def closing_price(link: Link, m: int, q: int, pi: object, depth: int) -> float:
    """|p_limit - p_depth|: what folding a depth-L hierarchy into a loop changes."""
    return abs(limit(link, m, q, pi) - tower_float(link, m, q, float(_pi(pi)), depth)[-1])


def contraction_rate(link: Link, m: int, q: int, pi: object) -> float:
    """The slope of the map at the state the tower rests in: how fast the price shrinks."""
    direct = float(_pi(pi))
    p = limit(link, m, q, direct)
    slope = Q.tail_slope_float(m, q, p)
    return (1.0 - direct) * slope if _link(link) is Link.COMMAND else direct * slope


def levels_to_cross(link: Link, m: int, q: int, pi: object, level: float,
                    *, cap: int = 10**7) -> int | None:
    """The first level at which the tower passes `level`, counting p_0 as level 0.

    Upward for COMMAND (collapse) and downward for VETO (settling to health). `None` if the
    tower comes to rest without passing it.
    """
    link = _link(link)
    direct = float(_pi(pi))
    p = direct
    crosses = (lambda x: x > level) if link is Link.COMMAND else (lambda x: x < level)
    for k in range(cap):
        if crosses(p):
            return k
        nxt = step_float(link, m, q, direct, p)
        if nxt == p:
            return None
        p = nxt
    raise RuntimeError(f"no crossing and no rest within {cap} levels")


__all__ = [
    "Link", "Outcome", "Kind", "FixedPoint", "step", "step_float", "tower", "tower_float",
    "fixed_point_poly", "count_fixed_points", "fixed_points", "outcome", "capture_gain",
    "threshold", "limit", "closing_price", "contraction_rate", "levels_to_cross",
]
