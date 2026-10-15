"""Standing: what a start gives, and what can move from it without money.

`rival` exchanges indivisible goods by top trading cycles from a starting allocation it does not derive. The
owner's statement of 2026-10-11 (`PROVENANCE.md`) is that the start cannot be drawn from a uniform distribution:
it is the present world, which money built, and the way to a future without money has to be extrapolated from it.
This module says what can be said exactly about ANY start, whoever made it.

There are n agents and n goods; the *start* (the endowment) gives each agent one good; each agent has a strict
ranking of all the goods. Four things follow from these definitions, and each is checked on every market of two
and three agents, on seeded larger ones, and against an independent reference (`tests/test_standing.py`, decision
D32):

  envy         agent i *envies* agent j when i ranks j's good strictly above its own. The *envy graph* has an
               edge from i to j for each such pair. A *loop* of envy is a cycle in it: i envies j, who envies ...,
               who envies i.
  trade        top trading cycles moves someone exactly when envy has a loop, which is exactly when the start is
               not Pareto efficient. (A loop can be traded round: each member gets a good it ranks higher. In
               the other direction, the first round of top trading cycles that moves anyone is a loop of envy.)
  ladder       a *pecking order* of a start is an ordering of the agents in which each holds its favourite among
               the goods held by it and by every agent after it. A start has one exactly when envy has no loop,
               which is the same as being Pareto efficient. It is a ladder: the first holds its first choice, the
               next its first choice among what is left, and so on. Envy points up it, never down. The result of
               top trading cycles is always a ladder, so after one exchange nothing moves again.
  wealth start the start that results when the wealthiest agent picks first, the next wealthiest second, and so
               on (serial dictatorship in the order of decreasing wealth). It is a ladder whose pecking order is
               the wealth order, and top trading cycles leaves it where it is. This is the exact cost of
               abolishing money while keeping holdings: a ladder made by money is frozen under an exchange that
               uses only rankings. It is a stylisation (a market is not a serial dictatorship), and it stands for
               the most that "the wealthy choose first" can mean in a model with rankings and no prices.

What thaws a ladder is the last part, the *thaw*: a dial t from 0 to 1 that moves every agent's ranking from the
one a common price induces to its own. Goods have vectors of values (as in `deals`), the price is one weight per
coordinate, and at setting t an agent scores a good by the exact dot product of its vector with
(1 - t) * price + t * its own weights, goods with equal scores listed in the order `goods` lists them
(`rival.rank_by_weights`). Because a score difference between two goods is a straight line in t, no ranking can
change between two settings at which some agent scores two goods equally (the *breakpoints*, exact rationals), so
the dial is computed exactly as the points 0, each breakpoint and 1, and the open stretch between each neighbouring
pair, each with its top trading cycles outcome. What it shows, and the tests check:

  * at t = 0 every agent ranks alike, so nothing moves from any start (a price is one ranking for everybody);
  * with ONE coordinate nothing ever moves, at any setting: a single number cannot rank two agents apart;
  * weights that are positive multiples of the price never move anything either, nor do weights that are all
    equal; and that is all: SOME goods and SOME start trade at SOME setting exactly when the weights are neither
    all equal nor all along the price (`can_thaw`, with a constructive `thaw_witness`). Trade without money needs
    a difference in the DIRECTION of what agents value, not in how strongly they value it;
  * the setting at which anyone first moves is an exact rational (`thaw_threshold`), attained or not, and it can
    be 0 (two goods of equal worth at the price can be swapped by any divergence at all);
  * the dial is NOT a one-way door: trade can open and close again as t rises (`thaw_reverses`), because one
    agent's ranking can come back to the common one while another's has not yet left it.

What it does not show. Real status is not envy of a good: tastes that depend on who else holds what are outside
this model (`rival`, HO-028), and the guarantees of top trading cycles fail or change there. The mix (1 - t) *
price + t * own is a choice, one path among many from a common ranking to private ones. A frozen ladder is shown
to be frozen; how to unfreeze one by anything other than the divergence of tastes (making goods less rival,
rotating who picks first) is not modelled, and a transfer from the top of a ladder is not an exchange in which
nobody loses (HO-030).

What it refuses. Numbers are checked, because a float would give a wrong answer quietly: coordinates, wealth and the
dial are integers or fractions (a float or a bool is a TypeError), the price and the weights are strictly positive,
vectors are as long as the price, the dial lies in [0, 1], and the goods and agents of a market match (a ValueError).
Containers are not checked: a non-mapping where a mapping goes raises an exception (AttributeError, TypeError or
ValueError, as `rival` does) and never gives an answer. `tests/test_standing.py` checks both boundaries: the numbers
against a rule written from these requirements, and the containers by putting a non-mapping in every slot.
"""

from __future__ import annotations

import functools
import heapq
import itertools
import math
import random
from collections.abc import Hashable, Mapping, Sequence
from fractions import Fraction
from typing import NamedTuple

from . import rival

__all__ = [
    "Piece", "Threshold", "Witness", "can_thaw", "dial_survey", "envy_graph", "envy_loop", "frozen",
    "is_pecking_order", "pecking_order", "serial_dictatorship", "survey", "thaw_breakpoints", "thaw_outcome",
    "thaw_pieces", "thaw_rankings", "thaw_reverses", "thaw_threshold", "thaw_weights", "thaw_witness",
    "wealth_start",
]

Agent = Hashable
Good = Hashable
Number = int | Fraction


class Piece(NamedTuple):
    """One stretch of the dial: the single setting `low` (when `low == high`) or the open stretch between two
    neighbouring settings, with the outcome of top trading cycles on it and the agents it moves."""

    low: Fraction
    high: Fraction
    allocation: dict
    movers: tuple


class Witness(NamedTuple):
    """A market of two agents, two goods and a setting of the dial at which each strictly prefers the other's good,
    so that top trading cycles swaps them (`thaw_witness`)."""

    lam: Fraction
    agents: tuple
    endowment: dict
    goods: dict


class Threshold(NamedTuple):
    """The least setting at which anyone moves: `lam`, and whether anyone moves AT it (`attained`) or only just
    above it."""

    lam: Fraction
    attained: bool


# --- envy, loops and ladders -----------------------------------------------------------------------------------


def _places(prefs: Mapping[Agent, Sequence[Good]]) -> dict[Agent, dict[Good, int]]:
    return {agent: {good: place for place, good in enumerate(ranking)} for agent, ranking in prefs.items()}


def envy_graph(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]]) -> dict[Agent, tuple]:
    """For each agent, the agents it envies (it ranks their good strictly above its own), in the endowment's
    order. Raises ValueError unless this is a strict market (`rival.check_market`)."""
    rival.check_market(endowment, prefs)
    place = _places(prefs)
    return {i: tuple(j for j in endowment if j != i and place[i][endowment[j]] < place[i][endowment[i]])
            for i in endowment}


def envy_loop(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]]) -> tuple | None:
    """One loop of envy (each member envies the next and the last envies the first), rotated to begin with its
    earliest member in the endowment's order, or None if envy has no loop. Depth-first from each agent in the
    endowment's order, so the loop found is the same on every run."""
    graph = envy_graph(endowment, prefs)
    order = {agent: place for place, agent in enumerate(endowment)}
    state: dict[Agent, int] = {}  # 1: on the walk in progress, 2: finished
    for root in endowment:
        if root in state:
            continue
        state[root] = 1
        walk = [root]
        edges = [iter(graph[root])]
        while walk:
            for target in edges[-1]:
                if state.get(target) == 1:  # the walk met itself: the part of it from `target` is a loop
                    loop = walk[walk.index(target):]
                    first = min(range(len(loop)), key=lambda k: order[loop[k]])
                    return tuple(loop[first:] + loop[:first])
                if target not in state:
                    state[target] = 1
                    walk.append(target)
                    edges.append(iter(graph[target]))
                    break
            else:
                state[walk.pop()] = 2
                edges.pop()
    return None


def frozen(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]]) -> bool:
    """Whether nothing can move from this start: envy has no loop, so top trading cycles returns it."""
    return envy_loop(endowment, prefs) is None


def pecking_order(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]]) -> tuple | None:
    """A pecking order of the start (each agent holds its favourite among the goods held by it and by every agent
    after it), or None if envy has a loop. When several exist, the one that always takes the earliest agent
    (in the endowment's order) that envies nobody still left."""
    graph = envy_graph(endowment, prefs)
    order = {agent: place for place, agent in enumerate(endowment)}
    envied_by: dict[Agent, list] = {agent: [] for agent in endowment}
    waiting = {agent: len(targets) for agent, targets in graph.items()}  # agents each envies that are still left
    ready: list[tuple[int, Agent]] = []
    for agent, targets in graph.items():
        for target in targets:
            envied_by[target].append(agent)
        if not targets:
            heapq.heappush(ready, (order[agent], agent))
    picked = []
    while ready:
        _, agent = heapq.heappop(ready)
        picked.append(agent)
        for envier in envied_by[agent]:
            waiting[envier] -= 1
            if waiting[envier] == 0:
                heapq.heappush(ready, (order[envier], envier))
    return tuple(picked) if len(picked) == len(endowment) else None


def is_pecking_order(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]],
                     order: Sequence[Agent]) -> bool:
    """Whether `order` (every agent once) is a pecking order of the start: each agent holds its favourite among
    the goods held by it and by every agent after it. Raises ValueError unless `order` lists each agent once."""
    rival.check_market(endowment, prefs)
    agents = list(order)
    if len(agents) != len(endowment) or set(agents) != set(endowment):
        raise ValueError("the order must list each agent exactly once")
    place = _places(prefs)
    left = {endowment[agent] for agent in agents}  # the goods held by this agent and everyone after it
    for agent in agents:
        mine = place[agent]
        if any(mine[good] < mine[endowment[agent]] for good in left):
            return False
        left.discard(endowment[agent])
    return True


def serial_dictatorship(order: Sequence[Agent], prefs: Mapping[Agent, Sequence[Good]]) -> dict:
    """Each agent in `order` takes its favourite good among those not yet taken; the result lists the agents in
    that order. Raises ValueError unless `order` lists each agent who ranks exactly once and every agent ranks
    the same goods, once each, as many goods as agents."""
    agents = list(order)
    if len(set(agents)) != len(agents) or set(agents) != set(prefs):
        raise ValueError("the order must list each agent who ranks exactly once")
    goods = list(next(iter(prefs.values()), ()))
    if len(goods) != len(agents) or any(len(r) != len(goods) or set(r) != set(goods) for r in prefs.values()):
        raise ValueError("every agent must rank the same goods, once each, and there must be as many goods as agents")
    left = set(goods)
    taken: dict[Agent, Good] = {}
    for agent in agents:
        pick = next(good for good in prefs[agent] if good in left)
        taken[agent] = pick
        left.discard(pick)
    return taken


def wealth_start(wealth: Mapping[Agent, Number], prefs: Mapping[Agent, Sequence[Good]]) -> dict:
    """The start in which the wealthiest agent picks first, the next wealthiest second, and so on: serial
    dictatorship in the order of decreasing wealth, listed in the order of `prefs`. Wealth is an integer or a
    fraction (never a float) for exactly the agents who rank, and no two agents have equal wealth: equal wealth
    has no pick order, and the tie is for whoever breaks it to decide, not for this function to guess."""
    if set(wealth) != set(prefs):
        raise ValueError("wealth is needed for exactly the agents who rank")
    if any(type(w) not in (int, Fraction) for w in wealth.values()):
        raise TypeError("wealth is an integer or a fraction, not a float")
    if len(set(wealth.values())) != len(wealth):
        raise ValueError("equal wealth has no pick order: break the tie first")
    picks = serial_dictatorship(sorted(prefs, key=lambda agent: -wealth[agent]), prefs)
    return {agent: picks[agent] for agent in prefs}


# --- the thaw ------------------------------------------------------------------------------------------------


def _fractions(label: str, values: Sequence[Number], dimension: int) -> tuple[Fraction, ...]:
    if len(values) != dimension:
        raise ValueError(f"{label} has {len(values)} coordinates and the price has {dimension}")
    if any(type(v) not in (int, Fraction) for v in values):
        raise TypeError(f"{label}: coordinates are integers or fractions, not floats")
    return tuple(Fraction(v) for v in values)


def _dial(goods: Mapping[Good, Sequence[Number]], price: Sequence[Number],
          weights: Mapping[Agent, Sequence[Number]]) -> tuple[dict, tuple, dict]:
    """The goods, the price and each agent's weights, checked and as exact fractions: strictly positive weights
    (a price is a strictly positive linear weighting, `deals`), every vector as long as the price."""
    dimension = len(price)
    if dimension < 1:
        raise ValueError("the price needs at least one coordinate")
    cost = _fractions("the price", price, dimension)
    if any(c <= 0 for c in cost):
        raise ValueError("the price must be strictly positive")
    own = {}
    for agent, w in weights.items():
        own[agent] = _fractions(f"agent {agent!r}'s weights", w, dimension)
        if any(c <= 0 for c in own[agent]):
            raise ValueError(f"agent {agent!r}: the weights must be strictly positive")
    vectors = {good: _fractions(f"good {good!r}", v, dimension) for good, v in goods.items()}
    return vectors, cost, own


def _linked(endowment: Mapping[Agent, Good], goods: Mapping[Good, Sequence[Number]],
            weights: Mapping[Agent, Sequence[Number]]) -> None:
    if set(goods) != set(endowment.values()):
        raise ValueError("the goods must be exactly the goods the endowment gives out")
    if set(weights) != set(endowment):
        raise ValueError("weights are needed for exactly the agents who own goods")


def _setting(lam: Number) -> Fraction:
    if type(lam) not in (int, Fraction):
        raise TypeError("the dial is an integer or a fraction, not a float")
    if not 0 <= lam <= 1:
        raise ValueError("the dial lies between 0 and 1")
    return Fraction(lam)


def thaw_weights(price: Sequence[Number], weights: Sequence[Number], lam: Number) -> tuple[Fraction, ...]:
    """The weights an agent with `weights` scores by at setting `lam`: (1 - lam) * price + lam * weights, exact."""
    lam = _setting(lam)
    cost = _fractions("the price", price, len(price))
    own = _fractions("the weights", weights, len(price))
    return tuple((1 - lam) * c + lam * w for c, w in zip(cost, own))


def _rankings(goods: Mapping, cost: tuple, own: Mapping, lam: Fraction) -> dict:
    return {agent: rival.rank_by_weights(goods, tuple((1 - lam) * c + lam * w for c, w in zip(cost, ws)))
            for agent, ws in own.items()}


def thaw_rankings(goods: Mapping[Good, Sequence[Number]], price: Sequence[Number],
                  weights: Mapping[Agent, Sequence[Number]], lam: Number) -> dict:
    """Each agent's strict ranking of the goods at setting `lam`: by the exact weighted sum under
    `thaw_weights`, goods with equal sums in the order `goods` lists them (`rival.rank_by_weights`)."""
    lam = _setting(lam)
    vectors, cost, own = _dial(goods, price, weights)
    return _rankings(dict(vectors), cost, own, lam)


def thaw_outcome(endowment: Mapping[Agent, Good], goods: Mapping[Good, Sequence[Number]],
                 price: Sequence[Number], weights: Mapping[Agent, Sequence[Number]], lam: Number) -> dict:
    """The top trading cycles outcome from the start when every agent ranks as `thaw_rankings` says at `lam`."""
    lam = _setting(lam)
    _linked(endowment, goods, weights)
    vectors, cost, own = _dial(goods, price, weights)
    return rival.top_trading_cycles(endowment, _rankings(dict(vectors), cost, own, lam))


def thaw_breakpoints(goods: Mapping[Good, Sequence[Number]], price: Sequence[Number],
                     weights: Mapping[Agent, Sequence[Number]]) -> tuple[Fraction, ...]:
    """Every setting strictly between 0 and 1 at which some agent scores two goods equally and its ranking of
    them can therefore change, in increasing order, as exact fractions. For goods g and h with difference vector
    d, an agent with weights w scores g - h as price.d + t * (w.d - price.d): a straight line in t, so it crosses
    zero at most once, at t = -price.d / (w.d - price.d) when that lies strictly between 0 and 1."""
    vectors, cost, own = _dial(goods, price, weights)
    found: set[Fraction] = set()
    for ws in own.values():
        for g, h in itertools.combinations(vectors, 2):
            difference = [a - b for a, b in zip(vectors[g], vectors[h])]
            at_price = sum(c * x for c, x in zip(cost, difference))
            slope = sum(w * x for w, x in zip(ws, difference)) - at_price
            if slope != 0 and 0 < -at_price / slope < 1:
                found.add(-at_price / slope)
    return tuple(sorted(found))


def thaw_pieces(endowment: Mapping[Agent, Good], goods: Mapping[Good, Sequence[Number]],
                price: Sequence[Number], weights: Mapping[Agent, Sequence[Number]]) -> list[Piece]:
    """The whole dial, exactly: 0, each breakpoint and 1 as single settings, and the open stretch between each
    neighbouring pair, in increasing order, each with the top trading cycles outcome on it and the agents it moves.
    No agent's ranking changes inside an open stretch, so one outcome stands for all of it.

    Cost: at most n * n(n - 1) / 2 breakpoints for n agents and, between them, one top trading cycles run per piece;
    most pairs of goods cross nowhere between 0 and 1, so there are far fewer in practice (20 agents with three
    coordinates, 117 breakpoints, 237 pieces, about a second and a half). It is meant for models of a few dozen agents."""
    _linked(endowment, goods, weights)
    vectors, cost, own = _dial(goods, price, weights)
    edges = [Fraction(0), *thaw_breakpoints(goods, price, weights), Fraction(1)]
    listed = dict(vectors)
    pieces: list[Piece] = []

    def piece(low: Fraction, high: Fraction, at: Fraction) -> Piece:
        outcome = rival.top_trading_cycles(endowment, _rankings(listed, cost, own, at))
        return Piece(low, high, outcome, rival.movers(endowment, outcome))

    for k, low in enumerate(edges):
        pieces.append(piece(low, low, low))
        if k + 1 < len(edges):
            pieces.append(piece(low, edges[k + 1], (low + edges[k + 1]) / 2))
    return pieces


def thaw_threshold(endowment: Mapping[Agent, Good], goods: Mapping[Good, Sequence[Number]],
                   price: Sequence[Number], weights: Mapping[Agent, Sequence[Number]]) -> Threshold | None:
    """The least setting at which anyone moves, or None if nobody moves at any setting. If anyone moves at that
    setting itself `attained` is True; if they move only just above it (the setting is a tie broken against
    trade, or the first piece that moves is an open stretch) it is False."""
    for piece in thaw_pieces(endowment, goods, price, weights):
        if piece.movers:
            return Threshold(piece.low, piece.low == piece.high)
    return None


def thaw_reverses(endowment: Mapping[Agent, Good], goods: Mapping[Good, Sequence[Number]],
                  price: Sequence[Number], weights: Mapping[Agent, Sequence[Number]]) -> bool:
    """Whether trade opens and then closes again as the dial rises: some piece with movers is followed by one
    without. The dial is not a one-way door."""
    seen = False
    for piece in thaw_pieces(endowment, goods, price, weights):
        if piece.movers:
            seen = True
        elif seen:
            return True
    return False


def _along(a: Sequence[Fraction], b: Sequence[Fraction]) -> bool:
    """Whether two strictly positive vectors point the same way: one is a positive multiple of the other."""
    return all(a[k] * b[m] == a[m] * b[k] for k in range(len(a)) for m in range(k + 1, len(a)))


def can_thaw(price: Sequence[Number], weights: Mapping[Agent, Sequence[Number]]) -> bool:
    """Whether SOME goods and SOME start trade at SOME setting of the dial: exactly when the agents' weights are
    neither all equal nor all positive multiples of the price. If they are all equal, or all along the price, every
    agent's effective weights point the same way at every setting, so every agent ranks alike and top trading cycles
    moves nobody, whatever the goods and the start (with one coordinate every weight is along the price). Otherwise two
    agents' effective weights point different ways at some setting, and `thaw_witness` builds goods that each of them
    ranks the other's above its own. A single agent, or none, has no one to trade with."""
    _goods, cost, own = _dial({}, price, weights)
    held = list(own.values())  # with none, or one, both tests below hold of every agent: the answer is no
    return not (all(w == held[0] for w in held) or all(_along(w, cost) for w in held))


def thaw_witness(price: Sequence[Number], weights: Mapping[Agent, Sequence[Number]]) -> Witness | None:
    """Two agents, two goods and a setting at which they swap, whenever `can_thaw` says some goods and some start trade;
    None otherwise. Take two agents whose effective weights e and f point different ways at the setting (setting 1 if
    their own weights do; setting 1/2 if their weights are parallel but unequal and not along the price). The
    vectors e and f are positive, so e.f > 0, and by the strict Cauchy-Schwarz inequality the interval from
    (e.f)/(f.f) to (e.e)/(e.f) is not empty; for t in it, u = e - t f has e.u > 0 and f.u < 0. Let the first agent hold
    the goods' negative part of u and the second the positive part: each ranks the other's good strictly higher. The
    vectors are scaled to coprime integers (a ranking does not change under scaling)."""
    _goods, cost, own = _dial({}, price, weights)
    for first, second in itertools.combinations(own, 2):
        one, other = own[first], own[second]
        if not _along(one, other):
            lam = Fraction(1)
        elif one != other and not _along(one, cost):
            lam = Fraction(1, 2)
        else:
            continue
        e = tuple((1 - lam) * c + lam * x for c, x in zip(cost, one))
        f = tuple((1 - lam) * c + lam * x for c, x in zip(cost, other))
        dot = sum(a * b for a, b in zip(e, f))
        low, high = dot / sum(b * b for b in f), sum(a * a for a in e) / dot
        t = (low + high) / 2
        u = [a - t * b for a, b in zip(e, f)]
        scale = math.lcm(*(x.denominator for x in u))
        whole = [int(x * scale) for x in u]
        divisor = math.gcd(*whole)
        whole = [x // divisor for x in whole]
        wanted, given_up = tuple(max(x, 0) for x in whole), tuple(max(-x, 0) for x in whole)
        return Witness(lam, (first, second), {first: "x", second: "y"}, {"x": given_up, "y": wanted})
    return None


# --- counts -------------------------------------------------------------------------------------------------------


def survey(n: int = 3) -> dict[str, int]:
    """Counts over EVERY profile of strict rankings of n agents with the identity start, which stands for every
    start up to the names of the goods: how many are ladders (`ladders`, envy has no loop) and how many have a
    loop of envy (`loops`); and over every wealth order of every profile, how many wealth starts there are
    (`wealth_starts`) and how many of them are ladders (`wealth_frozen`). The ladders and loops are the profiles
    in which `rival.survey` finds nobody and somebody trading. Refused beyond three agents."""
    if n < 1:
        raise ValueError("a market has at least one agent")
    if n > 3:
        raise ValueError("the survey is refused beyond 3 agents: at 4 there are 331,776 profiles")
    agents, goods = list(range(n)), [f"g{i}" for i in range(n)]
    start = dict(zip(agents, goods))
    counts = {"profiles": 0, "ladders": 0, "loops": 0, "wealth_starts": 0, "wealth_frozen": 0}
    for choice in itertools.product(itertools.permutations(goods), repeat=n):
        prefs = dict(zip(agents, choice))
        counts["profiles"] += 1
        counts["ladders" if frozen(start, prefs) else "loops"] += 1
        for ranked in itertools.permutations(agents):  # ranked[0] is the wealthiest
            counts["wealth_starts"] += 1
            counts["wealth_frozen"] += frozen(wealth_start({a: n - place for place, a in enumerate(ranked)}, prefs),
                                              prefs)
    return counts


def dial_survey(count: int = 300, seed: int = 77) -> dict[str, int]:
    """Counts over `count` seeded random instances of the dial (three or four agents, two or three coordinates,
    goods with coordinates 0 to 6, price and weights 1 to 6, a random start): how many trade at some setting
    (`trade`) and, of those, how many trade at one setting and not at a higher one (`reverse`). These are
    samples of an arbitrary distribution, not a rate in any population. The counts are a pure function of the
    arguments, so the last few are remembered (the report asks for the same ones many times)."""
    return dict(_dial_counts(count, seed))


@functools.lru_cache(maxsize=8)
def _dial_counts(count: int, seed: int) -> tuple[tuple[str, int], ...]:
    rng = random.Random(seed)
    counts = {"instances": 0, "trade": 0, "reverse": 0}
    for _ in range(count):
        n, d = rng.choice((3, 4)), rng.choice((2, 3))
        names = [f"g{k}" for k in range(n)]
        goods = {g: tuple(rng.randint(0, 6) for _ in range(d)) for g in names}
        price = tuple(rng.randint(1, 6) for _ in range(d))
        weights = {a: tuple(rng.randint(1, 6) for _ in range(d)) for a in range(n)}
        held = names[:]
        rng.shuffle(held)
        moved = [bool(piece.movers) for piece in thaw_pieces(dict(zip(range(n), held)), goods, price, weights)]
        counts["instances"] += 1
        counts["trade"] += any(moved)
        counts["reverse"] += any(moved) and not all(moved[moved.index(True):])
    return tuple(counts.items())
