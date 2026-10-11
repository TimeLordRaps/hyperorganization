#!/usr/bin/env python3
# PROVENANCE. Written on 2026-10-11 by a subagent (the owner's standing rule: subagents use the smallest model tier) from a written specification alone: it never saw
# this repository's implementation of the same functions. Besides this comment,
# the only other change is one blank line removed after the imports (isort).
# It is an UNTRUSTED oracle: agreement between it and the package shows that implementation slips are
# caught, not that the definitions are right (TECHNICAL_DEBT.md, HO-024). Run by differential_oracles.py.
"""Brute-force reference oracle for reliance footprints and group coverage.

Standard library only. Every group-level question is answered by explicitly
enumerating groups of people (all subsets, via itertools.combinations).
"""

import itertools

# ---------------------------------------------------------------------------
# 1. footprint
# ---------------------------------------------------------------------------

def _reverse_graph(depends_on):
    rev = {}
    for x, ys in depends_on.items():
        for y in ys:
            rev.setdefault(y, []).append(x)
    return rev


def _reachable_in_one_or_more(adj, starts):
    """Every node reachable from `starts` by following adj in one or more steps."""
    seen = set()
    stack = []
    for s in starts:
        stack.extend(adj.get(s, ()))
    while stack:
        y = stack.pop()
        if y in seen:
            continue
        seen.add(y)
        stack.extend(adj.get(y, ()))
    return seen


def footprint(depends_on, targets):
    targets = list(targets)
    # material: targets plus every node some target relies on (path of >= 1 step)
    material = set(targets) | _reachable_in_one_or_more(depends_on, targets)
    # impact: targets plus every node that relies on some target (path of >= 1 step)
    impact = set(targets) | _reachable_in_one_or_more(_reverse_graph(depends_on), targets)
    return material, impact


# ---------------------------------------------------------------------------
# 2-4. groups of people, enumerated explicitly
# ---------------------------------------------------------------------------

def _prepare(knowledge, nodes):
    """Index the people (keys of knowledge) and the needed nodes.

    masks[i] has bit b set iff person i understands needed node b.
    full has every needed bit set.
    """
    people = list(knowledge.keys())
    need = list(set(nodes))
    bit = {node: b for b, node in enumerate(need)}
    full = (1 << len(need)) - 1
    masks = []
    for p in people:
        m = 0
        for node in knowledge[p]:
            if node in bit:
                m |= 1 << bit[node]
        masks.append(m)
    return people, masks, full


def _group_understands(masks, full, group):
    """True iff every needed node is understood by at least one member of group."""
    acc = 0
    for i in group:
        acc |= masks[i]
    return acc == full


def cover_number(knowledge, nodes):
    people, masks, full = _prepare(knowledge, nodes)
    n = len(people)
    for size in range(n + 1):
        for group in itertools.combinations(range(n), size):
            if _group_understands(masks, full, group):
                return size
    return None


def robust_quorum(knowledge, nodes):
    people, masks, full = _prepare(knowledge, nodes)
    n = len(people)
    for q in range(n + 1):
        if all(_group_understands(masks, full, g) for g in itertools.combinations(range(n), q)):
            return q
    return None


def necessary(knowledge, nodes):
    people, masks, full = _prepare(knowledge, nodes)
    n = len(people)
    common = None  # intersection of all understanding groups, as index sets
    for size in range(n + 1):
        for group in itertools.combinations(range(n), size):
            if _group_understands(masks, full, group):
                if common is None:
                    common = set(group)
                else:
                    common &= set(group)
    if common is None:
        return set()
    return {people[i] for i in common}


# ---------------------------------------------------------------------------
# self-test: worked examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # E1
    k1 = {"p": ["a", "b"], "q": ["b", "c"], "r": ["b"]}
    assert cover_number(k1, ["a", "b", "c"]) == 2
    assert robust_quorum(k1, ["a", "b", "c"]) == 3
    assert necessary(k1, ["a", "b", "c"]) == {"p", "q"}
    # E2
    assert cover_number(k1, ["a", "b", "c", "d"]) is None
    assert robust_quorum(k1, ["a", "b", "c", "d"]) is None
    assert necessary(k1, ["a", "b", "c", "d"]) == set()
    # E3
    d3 = {"a": [], "b": ["a"], "c": ["b"], "d": ["b"], "e": []}
    mat, imp = footprint(d3, ["b"])
    assert mat == {"a", "b"}
    assert imp == {"b", "c", "d"}
    # E4
    d4 = {"x": ["y", "v"], "y": ["z"], "z": ["x"], "w": ["x"], "v": []}
    mat, imp = footprint(d4, ["z"])
    assert mat == {"z", "x", "y", "v"}
    assert imp == {"z", "y", "x", "w"}
    # E5
    k5 = {"p": ["a"]}
    assert cover_number(k5, []) == 0
    assert robust_quorum(k5, []) == 0
    assert necessary(k5, []) == set()
    print("OK")
