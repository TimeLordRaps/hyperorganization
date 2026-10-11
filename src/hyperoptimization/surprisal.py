"""Price without money: the cost of compromising something is the surprisal of doing it.

hyperethics prices a party by what an attacker pays to bring it under control, and a gate by
the cheapest total price of a set of parties that together control it. Nothing in that
machinery needs money. It needs a quantity that ADDS across things an attacker must do
together, and under which a cheaper way beats a dearer one. Surprisal does both when the
events are independent:

    bits(p) = -log2 p          P(A and B) = P(A) P(B)   <=>   bits(A and B) = bits(A) + bits(B)

so the cheapest cover of a gate, read in bits, is the MOST PROBABLE way in. Read this way a
price is a measurement of how improbable a compromise is, and a "stake" is how many
independent attempts a prize attracts (an attacker facing a cover of c bits succeeds by luck
once in 2**c attempts, so a prize that draws 2**s attempts makes covers of up to s bits
affordable). No currency appears.

What the number means (FORM, exact rationals)
    For a gate that needs a set of parties, with parties independently compromised at their
    own probabilities and control passing down command edges, let C be the cost in bits of the
    cheapest cover and N the number of minimal covers. Then

        2**-C  <=  P(the gate is taken)  <=  min(1, N * 2**-C).

    The left bound is the cheapest cover alone; the right is the union bound over minimal
    covers. `sandwich_gate` returns both with the exact probability between them.

Where additivity breaks
    The sum of marginals overstates the cost whenever compromise events are positively
    associated. A common principal that commands several seats is the case hyperethics calls
    the `common-principal` cause; here it is `additivity_gap`, the bits a naive sum
    over-prices a set of seats by. It is also why `gate_size` counts roots and not names.

A council of m seats needing q is itself a party, priced at -log2 T(x): when seats are dear
that is about q bits per seat less log2 C(m, q), the number of ways to choose the q.

The cover search enumerates subsets of parties. It is exact and refuses more than
`PARTY_LIMIT` of them rather than guess.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from . import quorum as Q

PARTY_LIMIT = 16
Edge = tuple[str, str]


def _probability(value: object) -> Fraction:
    if type(value) is bool or not isinstance(value, (int, float, Fraction)):
        raise TypeError(f"a probability is an int, float or Fraction, not {value!r}")
    p = Fraction(value)
    if not 0 <= p <= 1:
        raise ValueError(f"a probability lies in [0, 1], not {value!r}")
    return p


def bits(p: object) -> float:
    """-log2 p. Infinity for an impossible event, zero for a certain one.

    Computed from the numerator and denominator separately, so probabilities far too small
    for a float (1 in 2**2000) still have an exact surprisal.
    """
    prob = _probability(p)
    if prob == 0:
        return math.inf
    if prob == 1:
        return 0.0
    return math.log2(prob.denominator) - math.log2(prob.numerator)


def probability(surprisal: float) -> float:
    """2**-b: the probability of an event that is b bits surprising."""
    if type(surprisal) is bool or not isinstance(surprisal, (int, float)):
        raise TypeError(f"a surprisal is a real number, not {surprisal!r}")
    if math.isnan(surprisal) or surprisal < 0:
        raise ValueError("a surprisal is a nonnegative number of bits")
    return 2.0**-surprisal


def council_bits(m: int, q: int, x: object) -> float:
    """The price in bits of capturing a council of m seats needing q, seats compromised at x."""
    return bits(Q.tail(m, q, x))


# ---------------------------------------------------------------------------
# Gates: parties, command edges and a set of parties whose control passes the gate.
# ---------------------------------------------------------------------------


def _check_net(names: Iterable[str], controls: Iterable[Edge], need: Iterable[str]) -> tuple[
        tuple[str, ...], frozenset[Edge], frozenset[str]]:
    party_names = tuple(sorted(set(names)))
    if len(party_names) > PARTY_LIMIT:
        raise ValueError(f"the exact cover search enumerates subsets (limit {PARTY_LIMIT} "
                         f"parties); group parties into roles first")
    known = set(party_names)
    edges = frozenset((a, b) for a, b in controls)
    for a, b in edges:
        if a not in known or b not in known:
            raise ValueError(f"the command edge {(a, b)!r} mentions an unknown party")
    goal = frozenset(need)
    if not goal <= known:
        raise ValueError(f"the gate needs unknown parties {sorted(goal - known)}")
    return party_names, edges, goal


def closure(controls: Iterable[Edge], roots: Iterable[str]) -> frozenset[str]:
    """Every party that acts on the instruction of some root, the roots included."""
    onward: dict[str, list[str]] = {}
    for a, b in controls:
        onward.setdefault(a, []).append(b)
    reached = set(roots)
    frontier = list(reached)
    while frontier:
        for target in onward.get(frontier.pop(), ()):
            if target not in reached:
                reached.add(target)
                frontier.append(target)
    return frozenset(reached)


def minimal_covers(names: Iterable[str], controls: Iterable[Edge],
                   need: Iterable[str]) -> list[frozenset[str]]:
    """Every smallest-by-inclusion set of roots whose command closure contains the gate's need."""
    party_names, edges, goal = _check_net(names, controls, need)
    found: list[frozenset[str]] = []
    for size in range(len(party_names) + 1):
        for roots in combinations(party_names, size):
            candidate = frozenset(roots)
            if any(c <= candidate for c in found):
                continue
            if goal <= closure(edges, candidate):
                found.append(candidate)
    return found


def _bit_prices(prices: Mapping[str, float]) -> dict[str, float]:
    out = {}
    for name, price in prices.items():
        if type(price) is bool or not isinstance(price, (int, float)):
            raise TypeError(f"the price of {name!r} is a number of bits, not {price!r}")
        if math.isnan(price) or price < 0:
            raise ValueError(f"the price of {name!r} is a nonnegative number of bits")
        out[name] = price
    return out


def cheapest_cover(prices: Mapping[str, float], controls: Iterable[Edge],
                   need: Iterable[str]) -> tuple[float, frozenset[str]]:
    """The least total bits of any set of roots that passes the gate, and one such set.

    Ties go to the set with fewer roots and then to the lexicographically smallest, so the
    answer is stable. Infinity (and no roots) if every cover includes a party that cannot be
    bought.
    """
    table = _bit_prices(prices)
    covers = minimal_covers(table, controls, need)
    best: tuple[float, int, tuple[str, ...]] | None = None
    for cover in covers:
        cost = sum(table[r] for r in cover)
        if cost == math.inf:
            continue
        key = (cost, len(cover), tuple(sorted(cover)))
        if best is None or key < best:
            best = key
    if best is None:
        return (math.inf, frozenset())
    return (best[0], frozenset(best[2]))


def gate_bits(prices: Mapping[str, float], controls: Iterable[Edge], need: Iterable[str]) -> float:
    """hyperethics' `gate_cost` with prices read as bits."""
    return cheapest_cover(prices, controls, need)[0]


def capture_probability_gate(probs: Mapping[str, object], controls: Iterable[Edge],
                             need: Iterable[str]) -> Fraction:
    """P(the gate is taken) when party i is independently compromised with probability probs[i].

    Exact, by summing over every set of directly compromised parties and following command.
    """
    chances = {name: _probability(p) for name, p in probs.items()}
    party_names, edges, goal = _check_net(chances, controls, need)
    total = Fraction(0)
    for mask in range(1 << len(party_names)):
        weight = Fraction(1)
        chosen = []
        for i, name in enumerate(party_names):
            if mask >> i & 1:
                weight *= chances[name]
                chosen.append(name)
            else:
                weight *= 1 - chances[name]
            if weight == 0:
                break
        if weight and goal <= closure(edges, chosen):
            total += weight
    return total


def sandwich_gate(probs: Mapping[str, object], controls: Iterable[Edge],
                  need: Iterable[str]) -> tuple[Fraction, Fraction, Fraction]:
    """(lower, exact, upper) for the probability of taking a gate.

    lower is the most probable minimal cover alone (2**-cost when probabilities are powers of
    a half); upper is the union bound over minimal covers, capped at one.
    """
    chances = {name: _probability(p) for name, p in probs.items()}
    covers = minimal_covers(chances, controls, need)
    terms = []
    for cover in covers:
        term = Fraction(1)
        for root in cover:
            term *= chances[root]
        terms.append(term)
    lower = max(terms, default=Fraction(0))
    upper = min(Fraction(1), sum(terms, Fraction(0)))
    return lower, capture_probability_gate(chances, controls, need), upper


# ---------------------------------------------------------------------------
# Councils as blocks: independent groups of seats that count toward a quorum together.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Block:
    """An independent source of compromise carrying `weight` seats toward the quorum."""

    name: str
    probability: Fraction
    weight: int = 1

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("a block needs a name")
        object.__setattr__(self, "probability", _probability(self.probability))
        if type(self.weight) is not int or self.weight < 1:
            raise ValueError("a block carries at least one seat")


def _quorum(q: object) -> int:
    if type(q) is not int or q < 0:
        raise ValueError("a quorum is a natural number")
    return q


def capture_probability(blocks: Iterable[Block], q: int) -> Fraction:
    """P(the compromised blocks carry at least q seats between them), exactly."""
    target = _quorum(q)
    if target == 0:
        return Fraction(1)
    distribution = {0: Fraction(1)}
    for block in blocks:
        step: dict[int, Fraction] = {}
        for carried, mass in distribution.items():
            up = min(target, carried + block.weight)
            step[up] = step.get(up, Fraction(0)) + mass * block.probability
            step[carried] = step.get(carried, Fraction(0)) + mass * (1 - block.probability)
        distribution = step
    return distribution.get(target, Fraction(0))


def cheapest_capture(blocks: Iterable[Block], q: int) -> tuple[float, tuple[str, ...]]:
    """The least total bits of any set of blocks carrying at least q seats, and one such set.

    A 0/1 knapsack on weight capped at q. (math.inf, ()) if the blocks cannot reach q.
    """
    target = _quorum(q)
    best: list[tuple[float, tuple[str, ...]]] = [(math.inf, ())] * (target + 1)
    best[0] = (0.0, ())
    for block in blocks:
        price = bits(block.probability)
        for carried in range(target, -1, -1):
            cost, names = best[carried]
            if cost == math.inf:
                continue
            reach = min(target, carried + block.weight)
            candidate = cost + price
            if candidate < best[reach][0] or (
                    candidate == best[reach][0] and len(names) + 1 < len(best[reach][1])):
                best[reach] = (candidate, names + (block.name,))
    cost, names = best[target]
    return (math.inf, ()) if cost == math.inf else (cost, tuple(sorted(names)))


def minimal_coalitions(blocks: Iterable[Block], q: int) -> list[tuple[str, ...]]:
    """Every inclusion-minimal set of blocks carrying at least q seats."""
    target = _quorum(q)
    members = sorted(blocks, key=lambda b: b.name)
    if len(members) > PARTY_LIMIT:
        raise ValueError(f"at most {PARTY_LIMIT} blocks for the exact search")
    found: list[frozenset[str]] = []
    weights = {b.name: b.weight for b in members}
    for size in range(len(members) + 1):
        for chosen in combinations(members, size):
            names = frozenset(b.name for b in chosen)
            if any(c <= names for c in found):
                continue
            if sum(weights[n] for n in names) >= target:
                found.append(names)
    return [tuple(sorted(c)) for c in found]


def sandwich(blocks: Iterable[Block], q: int) -> tuple[Fraction, Fraction, Fraction]:
    """(lower, exact, upper): the most probable minimal coalition, the truth, the union bound."""
    members = list(blocks)
    by_name = {b.name: b.probability for b in members}
    terms = []
    for coalition in minimal_coalitions(members, q):
        term = Fraction(1)
        for name in coalition:
            term *= by_name[name]
        terms.append(term)
    lower = max(terms, default=Fraction(0))
    upper = min(Fraction(1), sum(terms, Fraction(0)))
    return lower, capture_probability(members, q), upper


# ---------------------------------------------------------------------------
# Where additivity breaks.
# ---------------------------------------------------------------------------


def common_principal_joint(principal: object, own: Iterable[object]) -> tuple[Fraction, Fraction]:
    """(P(all seats compromised), the product of their marginals) under a shared principal.

    Each seat is compromised if the principal is (probability `principal`, one event for all
    of them) or, independently, by its own cause with probability own[i]. The first number is
    the truth; the second is what a table of per-seat prices adds up to.
    """
    x = _probability(principal)
    causes = [_probability(e) for e in own]
    if not causes:
        raise ValueError("a principal commands at least one seat")
    all_by_own = Fraction(1)
    independent = Fraction(1)
    for e in causes:
        all_by_own *= e
        independent *= x + (1 - x) * e
    return x + (1 - x) * all_by_own, independent


def additivity_gap(principal: object, own: Iterable[object]) -> float:
    """Bits by which summing the seats' marginal surprisals over-prices taking all of them."""
    joint, independent = common_principal_joint(principal, own)
    if joint == 0:
        return 0.0
    return bits(independent) - bits(joint)


__all__ = [
    "PARTY_LIMIT", "Block", "additivity_gap", "bits", "capture_probability",
    "capture_probability_gate", "cheapest_capture", "cheapest_cover", "closure",
    "common_principal_joint", "council_bits", "gate_bits", "minimal_coalitions",
    "minimal_covers", "probability", "sandwich", "sandwich_gate",
]
