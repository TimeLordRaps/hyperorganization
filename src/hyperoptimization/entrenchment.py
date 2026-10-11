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
from collections.abc import Iterable
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


__all__ = ["Rule", "apex", "authority_height", "cycles", "protection", "weakest_link"]
