#!/usr/bin/env python3
# PROVENANCE. Written on 2026-10-11 by a subagent (the owner's standing rule: subagents use the smallest model tier) from a written specification alone: it never saw
# this repository's implementation of the same functions. Nothing but this comment and the shebang line was
# added. It is an UNTRUSTED oracle: agreement between it and the package shows that implementation slips are
# caught, not that the definitions are right (TECHNICAL_DEBT.md, HO-024). Run by differential_oracles.py.
# It replaced an earlier oracle that solved the same linear programs with sympy's simplex, which was dropped
# because it hangs on degenerate programs for some hash seeds (HO-025). This one uses exact Fourier-Motzkin
# elimination on fractions, imports only the standard library and answers the same under every hash seed.
"""Exact price-invisibility oracle (standard library only).

Options are equal-length tuples of non-negative integers, bigger is better in
every coordinate.  Equal options do not dominate each other.

supported(points, i): True iff some strictly positive price vector w (w[k] > 0
for every k) gives w.points[i] >= w.points[j] for every j (ties count).
The system is homogeneous: dividing a strictly positive solution by its
smallest coordinate gives w[k] >= 1 for all k, and conversely w[k] >= 1 implies
w[k] > 0.  So it suffices to decide feasibility of
    w[k] >= 1 for all k,        w.(points[i] - points[j]) >= 0 for all j,
exactly over Fraction, by Fourier-Motzkin elimination.  A row (c, b) stands for
c.w + b >= 0.  Eliminating variable v pairs every row with c[v] > 0 against
every row with c[v] < 0; the two positive multipliers cancel the v term, rows
with c[v] == 0 are kept.  Infeasible iff a row with all-zero c and b < 0 ever
appears.

efficient(points): ascending indices of options that no option dominates.
price_invisible(points): ascending indices of efficient options that are not
supported.
"""
from fractions import Fraction


def dominates(a, b):
    """True iff a != b and a[k] >= b[k] for every coordinate k."""
    return a != b and all(x >= y for x, y in zip(a, b))


def efficient(points):
    pts = [tuple(p) for p in points]
    return [i for i in range(len(pts)) if not any(dominates(q, pts[i]) for q in pts)]


def _normalise_and_merge(rows):
    """Scale each row by a positive factor so its first nonzero coefficient is
    +-1, drop trivially true rows, and merge rows with equal coefficients.
    Merging keeps the smallest constant: c.w + b >= 0 with smaller b is the
    stronger row, and it implies the others.  Returns None on a contradiction.
    """
    scaled = []
    for c, b in rows:
        if not any(c):
            if b < 0:
                return None
            continue
        lead = next(abs(x) for x in c if x != 0)
        scaled.append((tuple(Fraction(x) / lead for x in c), Fraction(b) / lead))
    # Sorting by plain tuple order keeps the result independent of PYTHONHASHSEED
    # (no sets or dicts are iterated); within equal coefficients the smallest b comes first.
    scaled.sort()
    kept = []
    for row in scaled:
        if not kept or kept[-1][0] != row[0]:
            kept.append(row)
    return kept


def _pair_count(rows, v):
    """Number of rows that eliminating v would combine."""
    return sum(1 for c, _ in rows if c[v] > 0) * sum(1 for c, _ in rows if c[v] < 0)


def _feasible(rows, d):
    rows = _normalise_and_merge(rows)
    if rows is None:
        return False
    todo = list(range(d))
    while todo:
        v = min(todo, key=lambda u: (_pair_count(rows, u), u))
        todo.remove(v)
        nxt = [r for r in rows if r[0][v] == 0]
        pos = [r for r in rows if r[0][v] > 0]
        neg = [r for r in rows if r[0][v] < 0]
        for cp, bp in pos:
            for cn, bn in neg:
                m_neg, m_pos = -cn[v], cp[v]  # both strictly positive
                nxt.append((tuple(m_neg * x + m_pos * y for x, y in zip(cp, cn)),
                            m_neg * bp + m_pos * bn))
        rows = _normalise_and_merge(nxt)
        if rows is None:
            return False
    return True


def supported(points, i):
    pts = [tuple(p) for p in points]
    p = pts[i]
    d = len(p)
    # Early exit, valid for strictly positive w: if q >= p and q != p then
    # w.q > w.p, so the constraint for q fails.  Elimination reaches the same answer.
    if any(dominates(q, p) for q in pts):
        return False
    rows = [(tuple(1 if t == k else 0 for t in range(d)), -1) for k in range(d)]  # w[k] >= 1
    rows += [(tuple(p[t] - q[t] for t in range(d)), 0) for q in pts]  # w.(p - q) >= 0
    return _feasible(rows, d)


def price_invisible(points):
    return [i for i in efficient(points) if not supported(points, i)]


if __name__ == "__main__":
    CASES = [
        ([(10, 0), (0, 10), (4, 4)], [True, True, False], [0, 1, 2], [2]),
        ([(0, 1), (1, 0), (1, 1)], [False, False, True], [2], []),
        ([(2, 0), (0, 2), (1, 1)], [True, True, True], [0, 1, 2], []),
        ([(5,)], [True], [0], []),
        ([(3, 3), (3, 3), (1, 5)], [True, True, True], [0, 1, 2], []),
        ([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)], [False, False, True, True], [2, 3], []),
    ]
    for pts, sup, eff, inv in CASES:
        got = ([supported(pts, i) for i in range(len(pts))], efficient(pts), price_invisible(pts))
        if got != (sup, eff, inv):
            raise SystemExit(f"self-check failed on {pts}: got {got}, want {(sup, eff, inv)}")
    tie = [(0, 2, 1, 0), (1, 1, 1, 0), (1, 1, 1, 0), (0, 0, 1, 2), (2, 0, 1, 0), (1, 0, 2, 0)]
    if supported(tie, 0) is not True:
        raise SystemExit("self-check failed on the all-ties example")
    print("OK")
