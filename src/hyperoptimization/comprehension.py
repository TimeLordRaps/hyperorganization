"""Understanding as the gate on deciding: who may decide, and how many must.

A decision is makeable only if the people making it understand its material and its impacts. Take that
as the whole premise and let the model say what it implies, with nothing assumed beyond who understands
what.

The footprint
    A decision acts on some TARGETS in a graph of reliance (node relies on node: repositories, rules,
    states of a game). Its MATERIAL is what the targets rest on, to any depth, because changing a thing
    soundly needs knowing what it stands on. Its IMPACT is what rests on the targets, to any depth,
    because the decision reaches whatever depends on what it changes. The FOOTPRINT is both. A footprint
    is a union of whole cores: you cannot understand part of a circle of mutual reliance, so a decision
    that touches one needs people who between them span it.

Who is needed
    Each person understands some nodes (the model takes this as given: see below). Then:

    cover         the fewest people who between them understand the whole footprint (exact, by search);
    robust        the fewest attendees such that WHOEVER attends understands it. If the least-understood
                  part is understood by only w of the m people then a group of m - w can miss it, so
                  the robust quorum is m - w + 1. It is never below the cover. w is the BUS FACTOR;
    necessary     the people no decision can be made without are exactly those who ALONE understand some
                  part of the footprint. When the footprint includes what rests on the target, the people
                  affected who alone know their own stake are among them; the necessity lapses as soon as
                  someone else understands that stake (a delegate can then stand in);
    gap           a part nobody understands: the decision is not makeable until someone learns it.

The barrier
    To enter the decision a person must understand at least a share of the footprint, the BARRIER. Raising
    it shrinks the pool, raises the cover and can make the decision unmakeable, never the reverse. The
    highest barrier a decision can bear is the last at which the people left still understand all of it
    (`max_barrier`); that is one way for the barrier to be DETERMINED by the decision and not chosen, and
    a rule that fixes the barrier is the other. A newcomer's entry cost is the number of parts of the
    footprint still to be understood.

The price in capture resistance
    The required quorum is the larger of a balanced majority of the pool and the robust quorum. The pool
    is what the barrier leaves, and a smaller body is cheaper to capture (Condorcet's jury theorem, in
    the bits of `surprisal`). So every seat the barrier removes is paid for in capture resistance:
    comprehension and fault tolerance pull against each other, and `capture_bits` says by how much.

What this does not show
    That understanding can be measured. Here it is a set handed in. A test of understanding that cannot
    be gamed (say, predicting the effects of a change on a model held back from the examinee) is the
    missing instrument, and without one the barrier invites the capture it was meant to prevent: whoever
    certifies understanding holds the gate. That is a conceptual block, recorded as D29.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from . import entrenchment, surprisal
from .viability import Game


@dataclass(frozen=True, slots=True)
class Footprint:
    """What a decision rests on and what rests on it."""

    targets: frozenset[str]
    material: frozenset[str]
    impact: frozenset[str]

    @property
    def nodes(self) -> frozenset[str]:
        return self.material | self.impact


def _names(items: Iterable[str], label: str) -> frozenset[str]:
    if isinstance(items, str):
        raise TypeError(f"{label} is a collection of names, not one string")
    found = frozenset(items)
    for name in found:
        if type(name) is not str or not name:
            raise ValueError(f"every name in {label} must be a nonempty string")
    return found


def _footprint(closure: Mapping[str, frozenset[str]], chosen: frozenset[str]) -> Footprint:
    material = chosen.union(*(closure[t] for t in chosen))
    impact = chosen | {m for m, found in closure.items() if found & chosen}
    return Footprint(chosen, frozenset(material), frozenset(impact))


def footprint(depends_on: Mapping[str, Iterable[str]], targets: Iterable[str]) -> Footprint:
    """The material and the impact of a decision on `targets` in a graph of reliance."""
    chosen = _names(targets, "targets")
    if not chosen:
        raise ValueError("a decision needs at least one target")
    closure = entrenchment.reliance_closure(depends_on)
    for target in chosen:
        if target not in closure:
            raise ValueError(f"{target!r} is not a node of the graph")
    return _footprint(closure, chosen)


def loads(depends_on: Mapping[str, Iterable[str]]) -> dict[str, Footprint]:
    """The footprint of a decision on each single node: the comprehension load of changing it."""
    closure = entrenchment.reliance_closure(depends_on)
    return {name: _footprint(closure, frozenset({name})) for name in closure}


def reliance_of_game(game: Game) -> dict[str, frozenset[str]]:
    """A game's states as a graph of reliance: a state relies on every state it can move to.

    The footprint of a permission at a state is then the closure of that state: the states it can lead
    to (material) and the states that can lead to it (impact).
    """
    return {s: frozenset(game.target(s, a, b) for a in game.allowed(s) for b in game.nature)
            for s in game.states}


# ---------------------------------------------------------------------------
# Who understands what.
# ---------------------------------------------------------------------------


def _knowledge(knowledge: Mapping[str, Iterable[str]]) -> dict[str, frozenset[str]]:
    if not isinstance(knowledge, Mapping):
        raise TypeError("knowledge maps each person to the nodes they understand")
    held: dict[str, frozenset[str]] = {}
    for person, nodes in knowledge.items():
        if type(person) is not str or not person:
            raise ValueError("every person needs a name")
        held[person] = _names(nodes, f"what {person!r} understands")
    return held


def _nodes(nodes: Footprint | Iterable[str]) -> frozenset[str]:
    return nodes.nodes if isinstance(nodes, Footprint) else _names(nodes, "the footprint")


def _barrier(value: object) -> Fraction:
    if type(value) not in (int, Fraction):
        raise TypeError("a barrier is an int or a Fraction between 0 and 1")
    barrier = Fraction(value)
    if not 0 <= barrier <= 1:
        raise ValueError("a barrier is a share of the footprint, between 0 and 1")
    return barrier


def _share(understood: frozenset[str], target: frozenset[str]) -> Fraction:
    return Fraction(len(understood & target), len(target)) if target else Fraction(1)


def coverage(knowledge: Mapping[str, Iterable[str]], person: str, nodes: Footprint | Iterable[str]) -> Fraction:
    """The share of the footprint `person` understands. Someone with no entry understands nothing yet."""
    held, target = _knowledge(knowledge), _nodes(nodes)
    return _share(held.get(person, frozenset()), target)


def eligible(knowledge: Mapping[str, Iterable[str]], nodes: Footprint | Iterable[str],
             barrier: object = 0) -> tuple[str, ...]:
    """Who may take part: the people who understand at least `barrier` of the footprint, sorted."""
    held, target, floor = _knowledge(knowledge), _nodes(nodes), _barrier(barrier)
    return tuple(sorted(p for p, understood in held.items() if _share(understood, target) >= floor))


def _pool(held: Mapping[str, frozenset[str]], pool: Iterable[str] | None) -> list[str]:
    chosen = sorted(held) if pool is None else sorted(set(pool))
    for person in chosen:
        if person not in held:
            raise ValueError(f"{person!r} is not a person with knowledge")
    return chosen


def witnesses(knowledge: Mapping[str, Iterable[str]], nodes: Footprint | Iterable[str],
              pool: Iterable[str] | None = None) -> dict[str, frozenset[str]]:
    """For each part of the footprint, the people in the pool who understand it."""
    held, target = _knowledge(knowledge), _nodes(nodes)
    people = _pool(held, pool)
    return {n: frozenset(p for p in people if n in held[p]) for n in sorted(target)}


def gaps(knowledge: Mapping[str, Iterable[str]], nodes: Footprint | Iterable[str],
         pool: Iterable[str] | None = None) -> frozenset[str]:
    """The parts of the footprint nobody in the pool understands."""
    return frozenset(n for n, who in witnesses(knowledge, nodes, pool).items() if not who)


def bus_factor(knowledge: Mapping[str, Iterable[str]], nodes: Footprint | Iterable[str],
               pool: Iterable[str] | None = None) -> int:
    """The fewest people who understand the least-understood part: the fewest whose loss leaves
    something not understood. 0 if some part is already not understood. An empty footprint cannot be
    lost, so it is as large as the pool."""
    held = _knowledge(knowledge)
    people = _pool(held, pool)
    who = witnesses(held, nodes, people)
    return min((len(w) for w in who.values()), default=len(people))


def necessary_voices(knowledge: Mapping[str, Iterable[str]], nodes: Footprint | Iterable[str],
                     pool: Iterable[str] | None = None) -> frozenset[str]:
    """The people who alone understand some part of the footprint. When the decision is makeable, every
    group that understands the whole contains them and no one else is in every such group. When some part
    is not understood by anyone there is no such group, and this is the sole witnesses of the parts that are."""
    return frozenset().union(*(w for w in witnesses(knowledge, nodes, pool).values() if len(w) == 1))


def covers(knowledge: Mapping[str, Iterable[str]], nodes: Footprint | Iterable[str],
           pool: Iterable[str] | None = None) -> list[tuple[str, ...]]:
    """Every smallest group in the pool that between them understands the whole footprint, in a fixed
    order. [] if there is none; [()] if the footprint is empty. Exhaustive, so for pools of a dozen or
    so people."""
    held, target = _knowledge(knowledge), _nodes(nodes)
    people = _pool(held, pool)
    if not target:
        return [()]
    for size in range(1, len(people) + 1):
        found = [group for group in itertools.combinations(people, size)
                 if target <= frozenset().union(*(held[p] for p in group))]
        if found:
            return found
    return []


def cover_number(knowledge: Mapping[str, Iterable[str]], nodes: Footprint | Iterable[str],
                 pool: Iterable[str] | None = None) -> int | None:
    """The fewest people who between them understand the whole footprint; None if no group does."""
    found = covers(knowledge, nodes, pool)
    return len(found[0]) if found else None


def robust_quorum(knowledge: Mapping[str, Iterable[str]], nodes: Footprint | Iterable[str],
                  pool: Iterable[str] | None = None) -> int | None:
    """The fewest attendees such that whoever attends understands the whole footprint: m - w + 1 for
    m people and bus factor w. None if some part is not understood by anyone."""
    held = _knowledge(knowledge)
    people = _pool(held, pool)
    if not _nodes(nodes):
        return 0
    weakest = bus_factor(held, nodes, people)
    return None if weakest == 0 else len(people) - weakest + 1


def entry_cost(knowledge: Mapping[str, Iterable[str]], person: str, nodes: Footprint | Iterable[str],
               barrier: object) -> int:
    """How many more parts of the footprint `person` must come to understand to be eligible."""
    held, target, floor = _knowledge(knowledge), _nodes(nodes), _barrier(barrier)
    needed = math.ceil(floor * len(target))
    return max(0, needed - len(held.get(person, frozenset()) & target))


def max_barrier(knowledge: Mapping[str, Iterable[str]], nodes: Footprint | Iterable[str]) -> Fraction | None:
    """The highest barrier at which the people still eligible between them understand the whole
    footprint; None if not even everyone does."""
    held, target = _knowledge(knowledge), _nodes(nodes)
    levels = sorted({_share(understood, target) for understood in held.values()} | {Fraction(0)},
                    reverse=True)
    for level in levels:
        if not gaps(held, target, eligible(held, target, level)):
            return level
    return None


# ---------------------------------------------------------------------------
# The requirement, and what it costs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Requirement:
    """What making one decision asks, at one barrier."""

    barrier: Fraction
    pool: tuple[str, ...]
    makeable: bool
    gaps: frozenset[str]
    cover: int | None
    robust: int | None
    balanced: int
    required: int | None
    bus_factor: int
    necessary: frozenset[str]


def requirement(knowledge: Mapping[str, Iterable[str]], nodes: Footprint | Iterable[str],
                barrier: object = 0) -> Requirement:
    """The pool, the cover, the robust quorum and the quorum required at one barrier.

    `required` is the larger of a strict majority of the pool (the balanced quorum of `quorum`) and the
    robust quorum, and is None if the decision is not makeable.
    """
    held, target, floor = _knowledge(knowledge), _nodes(nodes), _barrier(barrier)
    pool = eligible(held, target, floor)
    missing = gaps(held, target, pool)
    balanced = len(pool) // 2 + 1 if pool else 0
    robust = robust_quorum(held, target, pool)
    return Requirement(
        barrier=floor, pool=pool, makeable=not missing, gaps=missing,
        cover=cover_number(held, target, pool), robust=robust, balanced=balanced,
        required=None if robust is None else max(balanced, robust),
        bus_factor=bus_factor(held, target, pool),
        necessary=necessary_voices(held, target, pool),
    )


def sweep(knowledge: Mapping[str, Iterable[str]], nodes: Footprint | Iterable[str],
          barriers: Iterable[object]) -> list[Requirement]:
    """The requirement at each of a rising list of barriers."""
    levels = [_barrier(b) for b in barriers]
    if any(a > b for a, b in zip(levels, levels[1:])):
        raise ValueError("barriers must not fall")
    return [requirement(knowledge, nodes, level) for level in levels]


def capture_bits(needs: Requirement, x: object) -> float | None:
    """The price in bits of capturing the deciding body, when each seat is compromised independently
    with probability `x`: the pool is the body and the required quorum is what captures it. None if the
    decision is not makeable."""
    if needs.required is None or not needs.pool:
        return None
    return surprisal.council_bits(len(needs.pool), needs.required, x)


__all__ = ["Footprint", "Requirement", "bus_factor", "capture_bits", "cover_number", "coverage", "covers",
           "eligible", "entry_cost", "footprint", "gaps", "loads", "max_barrier", "necessary_voices",
           "reliance_of_game", "requirement", "robust_quorum", "sweep", "witnesses"]
