"""Rival freedoms: an exchange of indivisible goods without money.

A freedom that is not rival costs nobody anything to grant (`knowledge`, `deals`): granting one leaves every
other holder's as it was, and "win" means a permission nobody could certify before. A rival freedom, a slot or a
seat or a unit that runs out, is held by one party at a time, so a gain for one can be a loss for another, and
a rule is needed that makes the exchange win-win in the only sense still available: nobody ends below where
they started.

The model is Shapley and Scarf's housing market. There are n agents and n goods; the *endowment* gives each
agent one good; each agent has a strict ranking of all the goods (no ties) that only it knows. An *allocation* is
any rearrangement of the goods among the agents. The rule is Gale's *top trading cycles* (TTC): every remaining
agent points at the owner of the good it likes best among those still on the market; the pointers form at least
one cycle; the agents on each cycle trade along it and leave; repeat until nobody is left.

Definitions, all checked here by exhaustion and not assumed from the theorems:

  individually rational   no agent ends with a good ranked below the one it started with;
  Pareto efficient        no other allocation leaves every agent no worse off and some agent better off;
  in the (strict) core    no coalition S can trade the goods it started with among its own members so that
                          every member is no worse off and some member is better off than in the allocation
                          (a *blocking coalition*);
  strategy-proof          no agent can end with a good it ranks higher by reporting a ranking other than its
                          true one, whatever the others report (here: no profitable misreport, found by trying
                          every ranking for the agent against the others' truthful reports).

The theorems the tests meet on the sampled markets: TTC is individually rational, Pareto efficient and in the
core (Shapley and Scarf 1974, credited to Gale); with strict rankings it is the only allocation in the core
(Roth and Postlewaite 1977); and it is strategy-proof (Roth 1982), so private rankings can be asked for without
a price. What this module adds is checked, not cited: `audit`, by which an agent verifies the outcome from its own
ranking and the public record of the cycles alone; `endowment_spread`, which says how far the starting allocation
decides what each agent gets; and `movers` and `improvement`, which show trade happening exactly when the
starting allocation is not Pareto efficient.

What it does not show: preferences with ties, goods held in several units, agents who want several goods, and
any preference over *who else* gets a good are outside the model. The brute-force checkers refuse more than
`MAX_BRUTE` agents. And the endowment is a starting allocation nobody here derives: who owns what at the start
is exactly the question money used to settle, and the exchange does not answer it (`endowment_spread`).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from fractions import Fraction

__all__ = [
    "MAX_BRUTE", "all_records", "allocation_of", "audit", "blocking_coalition", "boston", "check_allocation",
    "check_market", "core_allocations", "endowment_spread", "improvement", "in_core",
    "individually_rational", "movers", "pareto_efficient", "profitable_misreports", "rank_by_weights",
    "rounds", "survey", "top_trading_cycles",
]

Agent = Hashable
Good = Hashable
Allocation = Mapping[Agent, Good]
Cycle = tuple[Agent, ...]
Record = list[list[Cycle]]  # the public record: the cycles that traded, round by round

MAX_BRUTE = 8  # the brute-force checkers enumerate n! (or sum of C(n,k) k!) cases


def check_market(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]]) -> None:
    """Raise ValueError unless this is a strict housing market: distinct endowed goods, and for exactly the
    agents who own goods, a ranking that lists every good exactly once."""
    goods = list(endowment.values())
    if len(set(goods)) != len(goods):
        raise ValueError("the endowment must give each agent a distinct good")
    if set(prefs) != set(endowment):
        raise ValueError("rankings are needed for exactly the agents who own goods")
    for agent, ranking in prefs.items():
        if len(ranking) != len(goods) or set(ranking) != set(goods):
            raise ValueError(f"agent {agent!r}: the ranking must rank every good exactly once")


def check_allocation(endowment: Mapping[Agent, Good], allocation: Allocation) -> None:
    """Raise ValueError unless the allocation is a rearrangement of the endowed goods among the same agents."""
    if set(allocation) != set(endowment) or sorted(map(repr, allocation.values())) != sorted(
            map(repr, endowment.values())):
        raise ValueError("an allocation must be a rearrangement of the endowed goods among the same agents")


def _ranks(prefs: Mapping[Agent, Sequence[Good]]) -> dict[Agent, dict[Good, int]]:
    return {agent: {good: place for place, good in enumerate(ranking)} for agent, ranking in prefs.items()}


def _cycles(points: Mapping[Agent, Agent], order: Mapping[Agent, int]) -> list[Cycle]:
    """Every cycle of the pointing graph (each agent points at exactly one agent), each rotated to begin with its
    earliest member and listed by that member's place in `order`."""
    state: dict[Agent, int] = {}  # 1: on the walk in progress, 2: finished
    found: list[Cycle] = []
    for start in points:
        if start in state:
            continue
        walk: list[Agent] = []
        agent = start
        while agent not in state:
            state[agent] = 1
            walk.append(agent)
            agent = points[agent]
        if state[agent] == 1:  # the walk met itself: that part of it is a cycle
            loop = walk[walk.index(agent):]
            first = min(range(len(loop)), key=lambda k: order[loop[k]])
            found.append(tuple(loop[first:] + loop[:first]))
        for visited in walk:
            state[visited] = 2
    return sorted(found, key=lambda cycle: order[cycle[0]])


def _run(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]]) -> tuple[dict, Record]:
    check_market(endowment, prefs)
    order = {agent: place for place, agent in enumerate(endowment)}
    owner = {good: agent for agent, good in endowment.items()}
    left = dict.fromkeys(endowment)  # the agents still in the market, in the endowment's order
    cursor = dict.fromkeys(endowment, 0)  # each agent's place in its ranking: goods only ever leave the market
    allocation: dict[Agent, Good] = {}
    record: Record = []
    while left:
        points = {}
        for agent in left:
            ranking = prefs[agent]
            while owner[ranking[cursor[agent]]] not in left:
                cursor[agent] += 1
            points[agent] = owner[ranking[cursor[agent]]]
        cycles = _cycles(points, order)
        record.append(cycles)
        for cycle in cycles:
            for agent, next_agent in zip(cycle, cycle[1:] + cycle[:1]):
                allocation[agent] = endowment[next_agent]  # you receive the good of the agent you point at
            for agent in cycle:
                del left[agent]
    return {agent: allocation[agent] for agent in endowment}, record


def top_trading_cycles(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]]) -> dict:
    """The top trading cycles allocation: agent -> the good it ends with. It does not depend on the order the
    agents are listed in, nor on which top cycles are executed first."""
    return _run(endowment, prefs)[0]


def boston(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]]) -> dict:
    """The Boston mechanism, a common rule for assigning goods, with the order the agents are listed in as their
    priority. It ignores ownership (it is a house-allocation rule, not an exchange): round by round every agent
    still without a good applies for its favourite good not yet taken, and each good goes at once, and for good, to its
    highest-priority applicant. It is here as the control for `profitable_misreports`: it is NOT strategy-proof, and
    that check finds the lie that shows it."""
    check_market(endowment, prefs)
    order = {agent: place for place, agent in enumerate(endowment)}
    taken: set = set()
    held: dict = {}
    pending = list(endowment)
    while pending:
        applications: dict = {}
        for agent in pending:
            applications.setdefault(next(good for good in prefs[agent] if good not in taken), []).append(agent)
        for good, applicants in applications.items():
            winner = min(applicants, key=order.__getitem__)
            held[winner] = good
            taken.add(good)
        pending = [agent for agent in pending if agent not in held]
    return {agent: held[agent] for agent in endowment}


def rounds(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]]) -> Record:
    """The public record of the run: for each round, the cycles that traded (each a tuple of agents in pointing
    order, beginning with its earliest member in the endowment's order). A one-agent cycle is an agent that keeps
    its own good because it is the best it can still reach."""
    return _run(endowment, prefs)[1]


def rank_by_weights(goods: Mapping[Good, Sequence[int | Fraction]], weights: Sequence[int | Fraction]) -> tuple:
    """A strict ranking of the goods, best first, by one agent's private weights on the coordinates of each good's
    vector (`deals`: freedom allowed, freedom enabled, and so on): the exact weighted sum, higher first, goods with
    equal sums in the order `goods` lists them. Floats are refused, as everywhere in this package.

    This is how a ranking can come from the vector value of a deal without a price shared by everyone: each agent
    weighs the coordinates its own way, and nobody else sees the weights. A common price is one set of weights for all;
    total value under it is the same before and after any exchange of the goods, so it cannot register a swap as a gain
    (`tests/test_rival.py`)."""
    scale = list(weights)
    if any(type(w) not in (int, Fraction) for w in scale):
        raise TypeError("weights are integers or fractions, not floats")
    totals = {}
    for good, vector in goods.items():
        if len(vector) != len(scale):
            raise ValueError(f"good {good!r} has {len(vector)} coordinates and there are {len(scale)} weights")
        if any(type(x) not in (int, Fraction) for x in vector):
            raise TypeError(f"good {good!r}: coordinates are integers or fractions, not floats")
        totals[good] = sum(Fraction(w) * Fraction(x) for w, x in zip(scale, vector))
    listed = {good: place for place, good in enumerate(goods)}
    return tuple(sorted(goods, key=lambda good: (-totals[good], listed[good])))


# --- the definitions, checked by exhaustion ----------------------------------------------------------------------------


def _brute_guard(endowment: Mapping[Agent, Good]) -> None:
    if len(endowment) > MAX_BRUTE:
        raise ValueError(f"the brute-force check enumerates every rearrangement; refused beyond {MAX_BRUTE} agents")


def _allocations(endowment: Mapping[Agent, Good]) -> Iterator[dict]:
    agents, goods = list(endowment), list(endowment.values())
    for arrangement in itertools.permutations(goods):
        yield dict(zip(agents, arrangement))


def individually_rational(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]],
                          allocation: Allocation) -> bool:
    """No agent ends with a good it ranks below the one it started with."""
    check_market(endowment, prefs)
    check_allocation(endowment, allocation)
    rank = _ranks(prefs)
    return all(rank[a][allocation[a]] <= rank[a][endowment[a]] for a in endowment)


def _dominates(rank: dict, better: Allocation, worse: Allocation) -> bool:
    """`better` leaves every agent no worse off than `worse` and some agent strictly better off."""
    return all(rank[a][better[a]] <= rank[a][worse[a]] for a in worse) and any(
        rank[a][better[a]] < rank[a][worse[a]] for a in worse)


def pareto_efficient(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]],
                     allocation: Allocation) -> bool:
    """No rearrangement of the goods leaves every agent no worse off and some agent better off."""
    check_market(endowment, prefs)
    check_allocation(endowment, allocation)
    _brute_guard(endowment)
    rank = _ranks(prefs)
    return not any(_dominates(rank, other, allocation) for other in _allocations(endowment))


def blocking_coalition(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]],
                       allocation: Allocation) -> tuple[Cycle, dict] | None:
    """The first coalition that blocks the allocation, smallest first, with the trade among its own endowed
    goods that leaves every member no worse off than in the allocation and some member better off; None if there
    is none (the allocation is in the strict core)."""
    check_market(endowment, prefs)
    check_allocation(endowment, allocation)
    _brute_guard(endowment)
    rank = _ranks(prefs)
    agents = list(endowment)
    for size in range(1, len(agents) + 1):
        for members in itertools.combinations(agents, size):
            for arrangement in itertools.permutations([endowment[a] for a in members]):
                trade = dict(zip(members, arrangement))
                if all(rank[a][trade[a]] <= rank[a][allocation[a]] for a in members) and any(
                        rank[a][trade[a]] < rank[a][allocation[a]] for a in members):
                    return members, trade
    return None


def in_core(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]], allocation: Allocation) -> bool:
    """No coalition blocks the allocation."""
    return blocking_coalition(endowment, prefs, allocation) is None


def core_allocations(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]]) -> list[dict]:
    """Every allocation in the strict core, found by trying all of them (the goods listed in the endowment's
    order, so the result is reproducible)."""
    check_market(endowment, prefs)
    _brute_guard(endowment)
    return [a for a in _allocations(endowment) if in_core(endowment, prefs, a)]


def profitable_misreports(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]],
                          mechanism: Callable[[Mapping, Mapping], Allocation] = top_trading_cycles,
                          agents: Sequence[Agent] | None = None) -> list[tuple[Agent, tuple, Good]]:
    """Every (agent, reported ranking, good received) in which the agent, reporting that ranking while the others
    tell the truth, receives a good it truly ranks strictly above the one it receives by telling the truth. An
    empty list means no agent among `agents` (default: all) can gain by lying about its ranking."""
    check_market(endowment, prefs)
    _brute_guard(endowment)
    rank = _ranks(prefs)
    truthful = mechanism(endowment, prefs)
    found = []
    for agent in (list(endowment) if agents is None else agents):
        for lie in itertools.permutations(prefs[agent]):  # the truthful report is among them and never gains
            got = mechanism(endowment, {**prefs, agent: lie})[agent]
            if rank[agent][got] < rank[agent][truthful[agent]]:
                found.append((agent, lie, got))
    return found


# --- what the exchange did, and how much the start decided ------------------------------------------------------------


def movers(endowment: Mapping[Agent, Good], allocation: Allocation) -> tuple[Agent, ...]:
    """The agents who end with a good other than the one they started with."""
    check_allocation(endowment, allocation)
    return tuple(a for a in endowment if allocation[a] != endowment[a])


def improvement(endowment: Mapping[Agent, Good], prefs: Mapping[Agent, Sequence[Good]],
                allocation: Allocation) -> dict:
    """For each agent, how many goods it ranks above the one it started with that it has climbed past:
    its starting place in its ranking minus its final place (negative if it ended lower)."""
    check_market(endowment, prefs)
    check_allocation(endowment, allocation)
    rank = _ranks(prefs)
    return {a: rank[a][endowment[a]] - rank[a][allocation[a]] for a in endowment}


def endowment_spread(prefs: Mapping[Agent, Sequence[Good]]) -> dict:
    """For each agent, the best and the worst place (0 is the favourite) of the good it receives over EVERY way of
    giving the goods out as the starting allocation. The wider the range, the more the start, which no exchange
    here derives, decides what the agent gets."""
    agents = list(prefs)
    if len(agents) > MAX_BRUTE:
        raise ValueError(f"every starting allocation is tried; refused beyond {MAX_BRUTE} agents")
    if not agents:
        return {}
    goods = list(prefs[agents[0]])
    rank = _ranks(prefs)
    best = dict.fromkeys(agents, len(goods))
    worst = dict.fromkeys(agents, -1)
    for held in itertools.permutations(goods):
        outcome = top_trading_cycles(dict(zip(agents, held)), prefs)
        for a in agents:
            place = rank[a][outcome[a]]
            best[a], worst[a] = min(best[a], place), max(worst[a], place)
    return {a: (best[a], worst[a]) for a in agents}


def survey(n: int = 3) -> dict[str, int]:
    """Counts over EVERY profile of strict rankings of n agents (n! to the power n of them) with the identity
    endowment, which stands for every endowment up to the names of the goods: how many leave every agent where it began
    (`still`) and how many trade (`trade`); in how many an agent can gain by misreporting under top trading cycles
    (`gameable_ttc`, none, by the theorem) and under the Boston mechanism (`gameable_boston`). Refused beyond three
    agents: four have 331,776 profiles, which `scripts/check_rival_exhaustive.py` runs in parallel."""
    if n < 1:
        raise ValueError("a market has at least one agent")
    if n > 3:
        raise ValueError("the survey is refused beyond 3 agents: at 4 there are 331,776 profiles")
    agents, goods = list(range(n)), [f"g{i}" for i in range(n)]
    endowment = dict(zip(agents, goods))
    counts = {"profiles": 0, "trade": 0, "still": 0, "gameable_ttc": 0, "gameable_boston": 0}
    for choice in itertools.product(itertools.permutations(goods), repeat=n):
        prefs = dict(zip(agents, choice))
        counts["profiles"] += 1
        counts["trade" if movers(endowment, top_trading_cycles(endowment, prefs)) else "still"] += 1
        counts["gameable_ttc"] += bool(profitable_misreports(endowment, prefs))
        counts["gameable_boston"] += bool(profitable_misreports(endowment, prefs, mechanism=boston))
    return counts


# --- the audit ---------------------------------------------------------------------------------------------------------


def audit(agent: Agent, ranking: Sequence[Good], endowment: Mapping[Agent, Good], record: Record) -> bool:
    """Whether the record is honest toward this agent, from its own ranking and the public record alone: in the
    round it traded, the good it received is its favourite among the goods still on the market (those endowed to
    agents who trade in that round or later). The endowment and the record are public; no other agent's ranking is
    read. Raises ValueError unless the agent is in exactly one cycle of the record.

    If every agent's audit passes, the record describes the top trading cycles outcome and no other (the first
    round's cycles are cycles of the first round's pointers, and the argument repeats on the market that is left).
    The converse holds for the honest record only: the same final allocation reached in another order of rounds can
    fail the audit of an agent who is given less than its favourite on the market at that time."""
    where = [(r, cycle) for r, cycles in enumerate(record) for cycle in cycles for member in cycle if member == agent]
    if len(where) != 1:
        raise ValueError("the agent must appear exactly once in the record")
    r, cycle = where[0]
    received = endowment[cycle[(cycle.index(agent) + 1) % len(cycle)]]
    on_market = {endowment[a] for cycles in record[r:] for c in cycles for a in c}
    return received == next(good for good in ranking if good in on_market)


def allocation_of(endowment: Mapping[Agent, Good], record: Record) -> dict:
    """The allocation a record describes: each agent receives the good of the next agent in its cycle."""
    allocation = {}
    for cycles in record:
        for cycle in cycles:
            for agent, next_agent in zip(cycle, cycle[1:] + cycle[:1]):
                allocation[agent] = endowment[next_agent]
    return {a: allocation[a] for a in endowment}


def _cyclic_blocks(items: list) -> Iterator[list[Cycle]]:
    """Every way to split `items` into cycles (cyclic orders, each rotated to begin with its earliest item)."""
    if not items:
        yield []
        return
    head, rest = items[0], items[1:]
    for size in range(len(rest) + 1):
        for others in itertools.combinations(rest, size):
            remaining = [x for x in rest if x not in others]
            for arrangement in itertools.permutations(others):
                for tail in _cyclic_blocks(remaining):
                    yield [(head, *arrangement)] + tail


def all_records(agents: list) -> Iterator[Record]:
    """Every record that puts each agent in exactly one cycle: a split of the agents into cycles, and an ordered
    division of those cycles into rounds. For n agents there are 1, 4, 24, 192 for n = 1, 2, 3, 4."""
    place = {agent: k for k, agent in enumerate(agents)}
    for cycles in _cyclic_blocks(list(agents)):
        count = len(cycles)
        for assignment in itertools.product(range(count), repeat=count):
            used = sorted(set(assignment))
            if used != list(range(len(used))):
                continue  # rounds are 0, 1, ... with none empty
            record: Record = [[] for _ in used]
            for cycle, round_number in zip(cycles, assignment):
                record[round_number].append(cycle)
            for cycles_in_round in record:
                cycles_in_round.sort(key=lambda cycle: place[cycle[0]])
            yield record
