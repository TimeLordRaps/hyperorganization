"""Deals without money: an opening of the law is valued by a vector, and the vector is not a price.

An OPENING takes a law `before` to a law `after`. It is valued by

    allowed       the permissions it opens, counted along each dimension of freedom (by default the
                  action; a mapping can group actions): the value of the freedom allowed;
    enabled       the states it makes reachable from where the parties are, and the permissions the law
                  grants there: the degree to which those freedoms allow other freedoms;
    coherence     four yes-or-no dimensions along which it coheres with the system's lawful existence:
                  EXISTS (the system survives for ever from where it is), CORRIGIBLE (and stays where an
                  overseer can bring it home), LAWFUL (nothing is permitted anywhere, reachable or not,
                  that the greatest corrigible law forbids: no latent exemption), MONOTONE (nobody loses
                  a permission they had).

What the model shows, and `tests/test_deals.py` checks, mostly by brute force over every law of small
games:

    closure        a freedom must come with the freedoms it enables. A door opened onto a room that
                   permits nothing leaves the system stuck in the doorway: not coherent. So coherent
                   openings are closed under what they enable (a single permission is coherent only
                   when everything it leads to is already permitted).
    no trade-off   among coherent openings the greatest corrigible law is best in every coordinate at
                   once. Given non-rival permissions, nothing within lawful existence competes for a
                   price to settle: freedom of this kind is not scarce. The scarce things are knowledge (`knowledge`), attention and
                   understanding (`comprehension`).
    admission      an opening is admitted only if it is coherent AND the people deciding understand its
                   footprint. A coalition that understands the material of an opening can certify it
                   exactly, so the two conditions meet.
    no price       where options DO compete (a menu, a capacity), the order that matters is Pareto
                   dominance, which is partial. A price is a linear weighting of the coordinates, and a
                   linear weighting never picks an efficient option that sits strictly inside the blend of
                   others: `price_invisible` finds them exactly, and gives the blend that hides each. By
                   the theorem of the alternative every option is either picked by some strictly positive
                   price (`supporting_weights`) or strictly beaten by a blend (`dominating_combination`),
                   never both, and both are certificates a reader can check without trusting the solver.
    no currency    coherence cannot be bought. Put lawfulness in the price at any finite weight and a
                   game exists (`trap_game`) in which the lawless opening outscores the lawful one:
                   freedoms can be made as numerous as you like, a finite price cannot outweigh them.
                   Coherence is a constraint, not a currency.

What this does not show: that the scalar is useless everywhere (it is a fine summary of a chain), or that
real freedoms are non-rival. A permission to use something that runs out is rival, and then granting it
to one party takes it from another; this model has only non-rival permissions, and the exchange of rival
ones without money (top trading cycles is the standard answer) is not built. "Win" here means a
permission nobody could certify before, not a payoff.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from . import comprehension, viability
from .ground import Status
from .viability import Game

COHERENCE = ("exists", "corrigible", "lawful", "monotone")

Law = dict[str, frozenset[str]]
Number = int | Fraction


@dataclass(frozen=True, slots=True)
class Value:
    """The value of an opening: three kinds of coordinate and no scalar."""

    allowed: tuple[tuple[str, int], ...]
    new_states: int
    new_permissions: int
    coherence: tuple[tuple[str, bool], ...]


def _law(game: Game, law: Mapping[str, Iterable[str]], label: str) -> Law:
    if not isinstance(law, Mapping):
        raise TypeError(f"{label} maps each state to the actions it permits there")
    out: Law = {}
    for state, actions in law.items():
        if state not in game.states:
            raise ValueError(f"{label} names {state!r}, which is not a state of this game")
        if isinstance(actions, str):
            raise TypeError(f"{label} permits a collection of actions at {state!r}, not one string")
        chosen = frozenset(actions)
        if not chosen <= game.allowed(state):
            raise ValueError(f"{label} permits an action at {state!r} that is not allowed there")
        out[state] = chosen
    return out


def _starts(game: Game, starts: Iterable[str]) -> tuple[str, ...]:
    if isinstance(starts, str):
        raise TypeError("starts is a collection of state names, not one string")
    found = tuple(dict.fromkeys(starts))
    if not found:
        raise ValueError("an opening is valued from at least one starting state")
    for state in found:
        if state not in game.states:
            raise ValueError(f"{state!r} is not a state of this game")
    return found


def _below(small: Mapping[str, frozenset[str]], large: Mapping[str, frozenset[str]]) -> bool:
    return all(actions <= large.get(state, frozenset()) for state, actions in small.items())


def _reach(game: Game, law: Law, starts: Iterable[str]) -> frozenset[str]:
    return frozenset().union(*(viability.reachable(game, law, s) for s in starts))


def _opened(before: Law, after: Law) -> list[tuple[str, str]]:
    return sorted((s, a) for s, actions in after.items() for a in actions - before.get(s, frozenset()))


def opening_value(game: Game, before: Mapping[str, Iterable[str]], after: Mapping[str, Iterable[str]],
                  starts: Iterable[str], *, dimension: Mapping[str, str] | None = None) -> Value:
    """The value of taking the law from `before` to `after` for a system that is in `starts`."""
    old, new = _law(game, before, "before"), _law(game, after, "after")
    where = _starts(game, starts)
    label = (lambda action: action) if dimension is None else (lambda action: dimension.get(action, action))
    counts: dict[str, int] = {}
    for _state, action in _opened(old, new):
        counts[label(action)] = counts.get(label(action), 0) + 1
    fresh = _reach(game, new, where) - _reach(game, old, where)
    greatest = viability.greatest_law(game, corrigible=True)
    flags = {
        "exists": all(viability.sustaining(game, new, s, corrigible=False).status is Status.HOLDS_IN_MODEL
                      for s in where),
        "corrigible": all(viability.sustaining(game, new, s, corrigible=True).status is Status.HOLDS_IN_MODEL
                          for s in where),
        "lawful": _below(new, greatest),
        "monotone": _below(old, new),
    }
    return Value(allowed=tuple(sorted(counts.items())), new_states=len(fresh),
                 new_permissions=sum(len(new.get(s, ())) for s in fresh),
                 coherence=tuple((name, flags[name]) for name in COHERENCE))


def admissible(value: Value) -> bool:
    """Whether the opening coheres along every dimension."""
    return all(holds for _name, holds in value.coherence)


# ---------------------------------------------------------------------------
# Who gains, and who may decide.
# ---------------------------------------------------------------------------


def party_values(game: Game, before: Mapping[str, Iterable[str]], after: Mapping[str, Iterable[str]],
                 starts: Iterable[str], owner: Mapping[str, str], *,
                 dimension: Mapping[str, str] | None = None) -> dict[str, Value]:
    """What the opening is worth to each party: the permissions opened at the states it owns, the new
    states it owns, and the permissions at them. The coherence of an opening is a fact about the whole
    system, so every party is shown the same flags."""
    if not isinstance(owner, Mapping) or set(owner) != set(game.states):
        raise ValueError("owner must name every state of the game, and only those")
    old, new = _law(game, before, "before"), _law(game, after, "after")
    where = _starts(game, starts)
    whole = opening_value(game, old, new, where, dimension=dimension)
    label = (lambda action: action) if dimension is None else (lambda action: dimension.get(action, action))
    fresh = _reach(game, new, where) - _reach(game, old, where)
    shares: dict[str, Value] = {}
    for party in sorted(set(owner.values())):
        counts: dict[str, int] = {}
        for state, action in _opened(old, new):
            if owner[state] == party:
                counts[label(action)] = counts.get(label(action), 0) + 1
        mine = {s for s in fresh if owner[s] == party}
        shares[party] = Value(tuple(sorted(counts.items())), len(mine),
                              sum(len(new.get(s, ())) for s in mine), whole.coherence)
    return shares


def _gains(value: Value) -> bool:
    return bool(value.allowed) or value.new_states > 0 or value.new_permissions > 0


def win_win(game: Game, before: Mapping[str, Iterable[str]], after: Mapping[str, Iterable[str]],
            starts: Iterable[str], owner: Mapping[str, str], *,
            dimension: Mapping[str, str] | None = None) -> bool:
    """The opening is coherent along every dimension and every party that owns a state gains."""
    shares = party_values(game, before, after, starts, owner, dimension=dimension)
    return len(shares) >= 1 and all(admissible(v) and _gains(v) for v in shares.values())


@dataclass(frozen=True, slots=True)
class Admission:
    """An opening, its footprint, what making the decision asks, and whether it is admitted."""

    value: Value
    footprint: comprehension.Footprint
    requirement: comprehension.Requirement
    coherent: bool
    makeable: bool
    admissible: bool


def admit(game: Game, before: Mapping[str, Iterable[str]], after: Mapping[str, Iterable[str]],
          starts: Iterable[str], knowledge: Mapping[str, Iterable[str]], barrier: object = 0, *,
          dimension: Mapping[str, str] | None = None) -> Admission:
    """Whether the opening may be made: coherent, and understood by enough people.

    The footprint is that of the states where something is opened, in the game's graph of reliance
    (a state relies on every state it can move to). `knowledge` maps each person to the states they
    understand; the barrier is the share of the footprint a person must understand to take part.
    """
    old, new = _law(game, before, "before"), _law(game, after, "after")
    value = opening_value(game, old, new, starts, dimension=dimension)
    targets = {state for state, _action in _opened(old, new)}
    if not targets:
        raise ValueError("nothing is opened: the two laws permit the same")
    footprint = comprehension.footprint(comprehension.reliance_of_game(game), targets)
    needs = comprehension.requirement(knowledge, footprint, barrier)
    coherent = admissible(value)
    return Admission(value, footprint, needs, coherent, needs.makeable, coherent and needs.makeable)


# ---------------------------------------------------------------------------
# Vectors, dominance and what a price can see.
# ---------------------------------------------------------------------------


def dimensions(*values: Value) -> tuple[str, ...]:
    """Every dimension of freedom any of the values counts, sorted."""
    return tuple(sorted({d for v in values for d, _count in v.allowed}))


def vector(value: Value, dims: Iterable[str] | None = None) -> tuple[int, ...]:
    """The value as numbers: allowed along each dimension, new states, new permissions, and each
    coherence flag as 0 or 1."""
    order = dimensions(value) if dims is None else tuple(dims)
    counts = dict(value.allowed)
    return (*(counts.get(d, 0) for d in order), value.new_states, value.new_permissions,
            *(int(holds) for _name, holds in value.coherence))


def _vectors(items: Iterable[object]) -> list[tuple[Number, ...]]:
    listed = list(items)
    if listed and all(isinstance(v, Value) for v in listed):
        dims = dimensions(*listed)  # type: ignore[arg-type]
        return [vector(v, dims) for v in listed]  # type: ignore[arg-type]
    return [tuple(p) for p in listed]  # type: ignore[arg-type]


def at_least(a: Value | tuple[Number, ...], b: Value | tuple[Number, ...]) -> bool:
    """a is at least b in every coordinate."""
    x, y = _vectors([a, b])
    return len(x) == len(y) and all(p >= q for p, q in zip(x, y))


def dominates(a: Value | tuple[Number, ...], b: Value | tuple[Number, ...]) -> bool:
    """a is at least b in every coordinate and differs from it: Pareto dominance, a strict partial
    order. Two options can be incomparable, which a scalar price cannot say."""
    x, y = _vectors([a, b])
    return at_least(x, y) and x != y


def efficient(items: Iterable[Value | tuple[Number, ...]]) -> list[int]:
    """The indices of the options no other option dominates."""
    points = _vectors(items)
    return [i for i, p in enumerate(points) if not any(dominates(q, p) for q in points)]


def priced(point: tuple[Number, ...], weights: tuple[Number, ...]) -> Number:
    """The scalar a linear price gives an option."""
    if len(point) != len(weights):
        raise ValueError("a price weights every coordinate")
    return sum((w * x for w, x in zip(weights, point)), 0)


def select_by_price(points: Iterable[tuple[Number, ...]], weights: tuple[Number, ...]) -> list[int]:
    """The options a strictly positive linear price rates highest (all of them, on a tie)."""
    listed = _vectors(points)
    if any(w <= 0 for w in weights):
        raise ValueError("a price is one strictly positive weight per coordinate")
    scores = [priced(p, weights) for p in listed]
    top = max(scores)
    return [i for i, s in enumerate(scores) if s == top]


def _solve(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction] | None:
    """Exact Gaussian elimination; None if the system is singular."""
    size = len(rhs)
    rows = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = next((r for r in range(col, size) if rows[r][col] != 0), None)
        if pivot is None:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        inverse = 1 / rows[col][col]
        rows[col] = [x * inverse for x in rows[col]]
        for r in range(size):
            if r != col and rows[r][col] != 0:
                factor = rows[r][col]
                rows[r] = [x - factor * y for x, y in zip(rows[r], rows[col])]
    return [rows[i][size] for i in range(size)]


def dominating_combination(points: Iterable[tuple[Number, ...]], i: int) -> list[Fraction] | None:
    """Weights on the options (summing to 1, none negative, none on option i) whose blend is at least
    option i in every coordinate and differs from it; None if there is no such blend.

    A blend like that beats option i at every strictly positive price, since the price of a blend is
    the blend of the prices and some option's price is at least that. It is found exactly, as the best
    vertex of a small linear program, by listing the basic solutions; the blend returned is a vertex
    and so not unique when several are equally good. Only the options other than i, and only the
    coordinates that are not the same for every option, take part.
    """
    listed = [tuple(Fraction(x) for x in p) for p in _vectors(points)]
    if not 0 <= i < len(listed):
        raise ValueError("option index out of range")
    beaten_by = next((j for j, p in enumerate(listed) if dominates(p, listed[i])), None)
    if beaten_by is not None:  # a single option already beats it: that option is the blend
        return [Fraction(int(j == beaten_by)) for j in range(len(listed))]
    kept = [k for k in range(len(listed[0])) if len({p[k] for p in listed}) > 1]
    front = set(efficient(listed))  # a blend of options can always be improved by blending the efficient ones above them
    others = [j for j in range(len(listed)) if j != i and j in front]
    if not kept or not others:
        return None
    # variables: one weight per other option, then one slack per kept coordinate (blend minus option i)
    n, d = len(others), len(kept)
    rows = [[listed[j][k] for j in others] + [Fraction(-1) if position == slot else Fraction(0) for slot in range(d)]
            for position, k in enumerate(kept)] + [[Fraction(1)] * n + [Fraction(0)] * d]
    rhs = [listed[i][k] for k in kept] + [Fraction(1)]
    best: tuple[Fraction, list[Fraction]] | None = None
    for basis in itertools.combinations(range(n + d), d + 1):
        solution = _solve([[row[c] for c in basis] for row in rows], rhs[:])
        if solution is None or any(x < 0 for x in solution):
            continue
        full = [Fraction(0)] * (n + d)
        for column, value in zip(basis, solution):
            full[column] = value
        gain = sum(full[n:])
        if gain > 0 and (best is None or gain > best[0]):
            best = (gain, full[:n])
    if best is None:
        return None
    weights = [Fraction(0)] * len(listed)
    for j, w in zip(others, best[1]):
        weights[j] = w
    return weights


def supporting_weights(points: Iterable[tuple[Number, ...]], i: int) -> list[Fraction] | None:
    """A strictly positive price (every weight at least 1) at which option i is rated at least as high
    as every other option; None if no strictly positive price does that.

    Found exactly as a vertex of the cone of prices that favour option i. Exactly one of this and
    `dominating_combination` exists for every option: that is the theorem of the alternative.
    """
    listed = [tuple(Fraction(x) for x in p) for p in _vectors(points)]
    if not 0 <= i < len(listed):
        raise ValueError("option index out of range")
    front = set(efficient(listed))
    if i not in front:  # an option something beats outright is rated lower by every strictly positive price
        return None
    d = len(listed[0])
    rows = [[Fraction(1) if k == c else Fraction(0) for c in range(d)] for k in range(d)]
    bounds = [Fraction(1)] * d
    for j, p in enumerate(listed):
        difference = [a - b for a, b in zip(listed[i], p)]
        if j != i and j in front and any(difference):
            rows.append(difference)
            bounds.append(Fraction(0))
    for active in itertools.combinations(range(len(rows)), d):
        candidate = _solve([rows[r][:] for r in active], [bounds[r] for r in active])
        if candidate is None:
            continue
        if all(sum(a * w for a, w in zip(row, candidate)) >= bound for row, bound in zip(rows, bounds)):
            return candidate
    return None


def supported(points: Iterable[tuple[Number, ...]], i: int) -> bool:
    """Whether some strictly positive price rates option i as high as any other."""
    return dominating_combination(points, i) is None


def price_invisible(points: Iterable[tuple[Number, ...]]) -> list[int]:
    """The efficient options that no strictly positive price will ever choose."""
    listed = _vectors(points)
    return [i for i in efficient(listed) if not supported(listed, i)]


# ---------------------------------------------------------------------------
# Two games that carry the arguments.
# ---------------------------------------------------------------------------


def trap_game(rooms: int) -> Game:
    """A hub with a safe place to stay or pause and a door into a run of rooms that look free and are
    not: in a flood every move in a room ends in the abyss. Freedom there is as large as `rooms` makes
    it, and none of it is lawful."""
    if type(rooms) is not int or rooms < 1:
        raise ValueError("a trap has at least one room")
    names = [f"r{i}" for i in range(1, rooms + 1)]
    table: dict[str, dict[str, tuple[str, ...]]] = {
        "hub": {"stay": ("hub", "hub"), "pause": ("hub", "hub"), "enter": ("r1", "r1")},
        "abyss": {"stay": ("abyss", "abyss")},
    }
    for i, name in enumerate(names):
        following = names[min(i + 1, rooms - 1)]
        table[name] = {action: (following, "abyss") for action in ("a", "b", "c")}
    return Game.from_table(nature=("calm", "flood"), table=table, safe=["hub", *names])


def trap_opening(rooms: int) -> Law:
    """The law that opens the door into the trap and every move inside it."""
    law: Law = {"hub": frozenset({"stay", "enter"})}
    law.update({f"r{i}": frozenset({"a", "b", "c"}) for i in range(1, rooms + 1)})
    return law


def three_doors() -> tuple[Game, Law, list[Law], dict[str, str]]:
    """A hub with three doors, all safe, only one of which can be taken. One opens onto ten rooms of
    x-freedoms, one onto ten of y-freedoms, one onto four rooms with both. Returns the game, the law
    before, the three openings and the grouping of actions into the dimensions x, y and rest."""
    table: dict[str, dict[str, tuple[str, ...]]] = {
        "hub": {"stay": ("hub",), "door_x": ("x1",), "door_y": ("y1",), "door_b": ("b1",)}}
    for i in range(1, 11):
        table[f"x{i}"] = {"x": (f"x{min(i + 1, 10)}",)}
        table[f"y{i}"] = {"y": (f"y{min(i + 1, 10)}",)}
    for i in range(1, 5):
        table[f"b{i}"] = {"x": (f"b{min(i + 1, 4)}",), "y": (f"b{min(i + 1, 4)}",)}
    game = Game.from_table(nature=("n",), table=table)
    base: Law = {"hub": frozenset({"stay"})}
    openings: list[Law] = [
        {"hub": frozenset({"stay", "door_x"}), **{f"x{i}": frozenset({"x"}) for i in range(1, 11)}},
        {"hub": frozenset({"stay", "door_y"}), **{f"y{i}": frozenset({"y"}) for i in range(1, 11)}},
        {"hub": frozenset({"stay", "door_b"}), **{f"b{i}": frozenset({"x", "y"}) for i in range(1, 5)}},
    ]
    grouping = {"door_x": "x", "door_y": "y", "door_b": "x", "x": "x", "y": "y", "stay": "rest"}
    return game, base, openings, grouping


__all__ = ["Admission", "COHERENCE", "Value", "admissible", "admit", "at_least", "dimensions",
           "dominates", "dominating_combination", "efficient", "opening_value", "party_values",
           "price_invisible", "priced", "select_by_price", "supported", "supporting_weights",
           "three_doors", "trap_game", "trap_opening", "vector", "win_win"]
