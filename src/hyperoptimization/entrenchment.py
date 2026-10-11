"""Who can change the rule that says who can change the rule: protection, authority and the loop.

A RULE is governed by another rule, the one that says how it may be changed, or by nothing if it
cannot be changed at all. Every rule also carries the price in bits of the PROCEDURE it defines:
what whoever wants to change a rule it governs has to pay.

Protection (FORM)
    To change rule u you may satisfy the procedure of its governor, or first change the governor
    and then change u for nothing. So the cheapest way is

        protection(u) = min over every rule v above u (governor, governor's governor, ...)
                        of the price of v's procedure,

    the WEAKEST LINK in the chain of authority. Around a cycle every member is above every
    other, so every member has the same protection: the cheapest procedure on the cycle. This is
    Ross's paradox: an entrenched clause that an ordinary majority can amend is protected by an
    ordinary majority, whatever the clause says of itself. The tests check the formula against
    an independent shortest-path search over sets of already-changed rules.

Authority (FORM)
    The HEIGHT of a rule is the number of governors above it before the chain ends. It is
    finite iff the chain ends at a rule that nothing governs: an unamendable apex, a sovereign in
    hyperethics' sense. If every rule has a governor then in a finite set every chain returns to
    itself (pigeonhole) and every height is omega. A constitution that obeys the no-apex rule has
    no finite rank for validity at all; its authority is a loop. The tests check this over every
    functional graph on up to five rules.

What this does and does not settle
    It settles the arithmetic of amendment. It does not say how a loop of rules can be stable: a
    loop's protection is its cheapest procedure, so a loop is as strong as its weakest member
    and no stronger, which is why the price of each procedure (a quorum, in `surprisal`, with a
    price from `quorum`) is what has to be large. And it cannot choose between a grounded
    hierarchy (finite ranks, a sovereign at the top) and a loop (no finite ranks, no sovereign):
    that choice is the rule hyperethics adopts, and L* in `viability` is how a system can be
    grounded without an apex, by DERIVATION and not by authorisation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .ground import OMEGA, Omega


@dataclass(frozen=True, slots=True)
class Rule:
    """A rule: the price in bits of the procedure it defines, and the rule that governs it."""

    name: str
    procedure_bits: float
    governed_by: str | None = None

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("a rule needs a name")
        bits = self.procedure_bits
        if type(bits) is bool or not isinstance(bits, (int, float)):
            raise TypeError(f"procedure_bits is a number of bits, not {bits!r}")
        if math.isnan(bits) or bits < 0:
            raise ValueError("procedure_bits is a nonnegative number of bits")
        if self.governed_by is not None and type(self.governed_by) is not str:
            raise TypeError("governed_by is a rule name or None")


def _index(rules: Iterable[Rule]) -> dict[str, Rule]:
    table: dict[str, Rule] = {}
    for rule in rules:
        if rule.name in table:
            raise ValueError(f"the rule {rule.name!r} is listed twice; names must not repeat")
        table[rule.name] = rule
    if not table:
        raise ValueError("a constitution needs at least one rule")
    for rule in table.values():
        if rule.governed_by is not None and rule.governed_by not in table:
            raise ValueError(f"the governor {rule.governed_by!r} of {rule.name!r} is not a rule")
    return table


def _chain(table: dict[str, Rule], name: str) -> tuple[list[str], bool]:
    """The rules above `name` in order, and whether the chain ended (at an ungoverned rule)."""
    above: list[str] = []
    seen = {name}
    node = table[name].governed_by
    while node is not None:
        above.append(node)
        if node in seen:
            return above, False
        seen.add(node)
        node = table[node].governed_by
    return above, True


def apex(rules: Iterable[Rule]) -> tuple[str, ...]:
    """The rules nothing governs: unamendable, so sovereign."""
    return tuple(r.name for r in _index(rules).values() if r.governed_by is None)


def cycles(rules: Iterable[Rule]) -> list[tuple[str, ...]]:
    """Every cycle of governance, each listed once from its first rule, following governors."""
    table = _index(rules)
    order = {name: position for position, name in enumerate(table)}
    found: list[tuple[str, ...]] = []
    covered: set[str] = set()
    for name in table:
        if name in covered:
            continue
        above, ended = _chain(table, name)
        if ended:
            continue
        repeat = above[-1]  # the first rule met twice lies on the cycle
        cycle = [repeat]
        node = table[repeat].governed_by
        while node != repeat:
            cycle.append(node)
            node = table[node].governed_by
        covered |= set(cycle)
        first = min(range(len(cycle)), key=lambda i: order[cycle[i]])
        found.append(tuple(cycle[first:] + cycle[:first]))
    return found


def authority_height(rules: Iterable[Rule]) -> dict[str, int | Omega]:
    """Governors above each rule before the chain ends: a number, or OMEGA if it never does."""
    table = _index(rules)
    heights: dict[str, int | Omega] = {}
    for name in table:
        above, ended = _chain(table, name)
        heights[name] = len(above) if ended else OMEGA
    return heights


def weakest_link(rules: Iterable[Rule], name: str) -> tuple[str | None, float]:
    """The rule above `name` whose procedure is cheapest, and its price. (None, inf) if none."""
    table = _index(rules)
    if name not in table:
        raise KeyError(f"{name!r} is not a rule")
    above, _ = _chain(table, name)
    best: tuple[str | None, float] = (None, math.inf)
    for node in above:
        if table[node].procedure_bits < best[1]:
            best = (node, table[node].procedure_bits)
    return best


def protection(rules: Iterable[Rule]) -> dict[str, float]:
    """Each rule's protection in bits: the cheapest way to change it. inf if it cannot be changed."""
    table = _index(rules)
    return {name: weakest_link(table.values(), name)[1] for name in table}


# ---------------------------------------------------------------------------
# Many governors: the phase of a dependency graph.
# ---------------------------------------------------------------------------


def _dependencies(depends_on: Mapping[str, Iterable[str]]) -> dict[str, frozenset[str]]:
    graph: dict[str, frozenset[str]] = {}
    for name, deps in depends_on.items():
        if type(name) is not str or not name:
            raise ValueError("every node needs a name")
        if isinstance(deps, str):
            raise TypeError(f"the dependencies of {name!r} are a collection of names, not one string")
        graph[name] = frozenset(deps)
    if not graph:
        raise ValueError("a graph needs at least one node")
    for name, deps in graph.items():
        for dep in deps:
            if dep not in graph:
                raise ValueError(f"{name!r} relies on {dep!r}, which is not a node")
    return graph


def phases(depends_on: Mapping[str, Iterable[str]]) -> dict[str, int | Omega]:
    """The ordinal phase of each node of a finite graph of reliance: 0 if it relies on nothing,
    one more than the largest phase among what it relies on otherwise, and OMEGA if its reliance
    never bottoms out (it lies on a cycle, or relies, however indirectly, on something that does).

    This is `authority_height` for nodes with any number of governors. A finite phase exists exactly
    when nothing the node reaches lies on a cycle: round k takes every node whose dependencies were
    all taken in earlier rounds, and a node never taken relies on a cycle.
    """
    graph = _dependencies(depends_on)
    taken: dict[str, int] = {}
    layer = 0
    while True:
        ready = [name for name, deps in graph.items() if name not in taken and deps.issubset(taken)]
        if not ready:
            break
        for name in ready:
            taken[name] = layer
        layer += 1
    return {name: taken.get(name, OMEGA) for name in graph}


def cores(depends_on: Mapping[str, Iterable[str]]) -> list[tuple[str, ...]]:
    """The groups of nodes that rely on one another, directly or through others: the strongly
    connected groups of two or more nodes, and each node that relies on itself, alone. Each group is
    sorted and the groups are listed in the order of their first member.

    A core has no foundation inside it. Every node of a core has phase OMEGA, and so does every
    node that relies on one.
    """
    graph = _dependencies(depends_on)
    reached = _reached(graph)
    groups: list[tuple[str, ...]] = []
    placed: set[str] = set()
    for name in sorted(graph):
        if name in placed or name not in reached[name]:
            continue
        group = tuple(sorted(other for other in graph if other in reached[name] and name in reached[other]))
        placed.update(group)
        groups.append(group)
    return groups


def _reached(graph: Mapping[str, frozenset[str]]) -> dict[str, set[str]]:
    """For each node, every node it reaches along one or more reliances."""
    reached: dict[str, set[str]] = {}
    for start in graph:
        seen: set[str] = set()
        frontier = list(graph[start])
        while frontier:
            node = frontier.pop()
            if node not in seen:
                seen.add(node)
                frontier.extend(graph[node])
        reached[start] = seen
    return reached


def incomparable_pairs(depends_on: Mapping[str, Iterable[str]]) -> list[tuple[str, str]]:
    """The pairs of nodes that neither relies on, directly or through others, sorted.

    Dependence is a partial order (a preorder, around a core): where this list is not empty it
    leaves some pairs unordered, and a rule of priority that must settle every conflict needs more
    than the dependence it is read from. A chain leaves nothing; two branches leave the pair of
    their tops.
    """
    graph = _dependencies(depends_on)
    reached = _reached(graph)
    names = sorted(graph)
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
            if b not in reached[a] and a not in reached[b]]


__all__ = ["Rule", "apex", "authority_height", "cores", "cycles", "incomparable_pairs", "phases", "protection",
           "weakest_link"]
