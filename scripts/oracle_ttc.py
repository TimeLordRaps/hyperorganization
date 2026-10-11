#!/usr/bin/env python3
# PROVENANCE. Written on 2026-10-11 by a subagent (the owner's standing rule: subagents use the smallest model
# tier) from a written specification alone: it never saw this repository's implementation of the same
# functions. Nothing but this comment and the shebang line was added. It is an UNTRUSTED oracle: agreement
# between it and the package shows that implementation slips are caught, not that the definitions are right
# (TECHNICAL_DEBT.md, HO-024). It executes one top cycle at a time, from the smallest remaining agent, where
# the package executes every cycle of a round together; the outcome is the same by a standard argument and is
# compared, not assumed. Run by differential_oracles.py.
"""Reference oracle for the Shapley-Scarf housing market (top trading cycles).

endowment and allocation map agent -> good and are bijections; prefs maps agent ->
list of all goods, most preferred first, no ties.

in_core uses the strict core: a coalition S blocks an allocation x when re-assigning
the goods endowed to S among S makes every member of S weakly better off than under x
and at least one member strictly better off.
"""
from itertools import combinations, permutations


def _ranks(prefs):
    return {a: {g: i for i, g in enumerate(goods)} for a, goods in prefs.items()}


def _blocks(change):
    """change[i] is member i's rank under the new good minus its rank under the old
    one, so a negative entry is an improvement. Blocks iff nobody gets worse and
    somebody gets better."""
    return max(change) <= 0 and min(change) < 0


def ttc(endowment, prefs):
    """Top trading cycles. Each remaining agent points at the owner of its favourite
    remaining good. The cycle reached from the smallest remaining agent trades and
    leaves the market; repeat until no agent remains."""
    owner = {g: a for a, g in endowment.items()}
    remaining = sorted(endowment)
    allocation = {}
    while remaining:
        left = {endowment[a] for a in remaining}
        point = {a: next(owner[g] for g in prefs[a] if g in left) for a in remaining}
        walk = [remaining[0]]
        while point[walk[-1]] not in walk:
            walk.append(point[walk[-1]])
        cycle = walk[walk.index(point[walk[-1]]):]
        for a in cycle:
            allocation[a] = endowment[point[a]]
        remaining = [a for a in remaining if a not in cycle]
    return allocation


def individually_rational(endowment, prefs, allocation):
    rank = _ranks(prefs)
    return all(rank[a][allocation[a]] <= rank[a][endowment[a]] for a in endowment)


def pareto_efficient(prefs, allocation):
    rank = _ranks(prefs)
    agents = sorted(allocation)
    goods = [allocation[a] for a in agents]
    for new in permutations(goods):
        change = [rank[a][g] - rank[a][allocation[a]] for a, g in zip(agents, new)]
        if _blocks(change):
            return False
    return True


def in_core(endowment, prefs, allocation):
    rank = _ranks(prefs)
    agents = sorted(endowment)
    for size in range(1, len(agents) + 1):
        for members in combinations(agents, size):
            for new in permutations([endowment[a] for a in members]):
                change = [rank[a][g] - rank[a][allocation[a]] for a, g in zip(members, new)]
                if _blocks(change):
                    return False
    return True


def strategy_proof(endowment, prefs):
    """Every profitable single-agent misreport, as (agent, reported_list)."""
    rank = _ranks(prefs)
    truthful = ttc(endowment, prefs)
    goods = sorted(endowment.values())
    found = []
    for a in sorted(endowment):
        for report in permutations(goods):
            misreport = {**prefs, a: list(report)}
            if rank[a][ttc(endowment, misreport)[a]] < rank[a][truthful[a]]:
                found.append((a, list(report)))
    return found


def core_allocations(endowment, prefs):
    agents = sorted(endowment)
    goods = sorted(endowment.values())
    found = []
    for new in permutations(goods):
        allocation = dict(zip(agents, new))
        if in_core(endowment, prefs, allocation):
            found.append(allocation)
    return sorted(found, key=lambda x: tuple(sorted(x.items())))


def check_profile(endowment, prefs):
    """Item (d): TTC is individually rational, Pareto efficient and in the strict core;
    the strict core is exactly {TTC}; no agent has a profitable misreport."""
    x = ttc(endowment, prefs)
    assert individually_rational(endowment, prefs, x), (endowment, prefs)
    assert pareto_efficient(prefs, x), (endowment, prefs)
    assert in_core(endowment, prefs, x), (endowment, prefs)
    assert core_allocations(endowment, prefs) == [x], (endowment, prefs)
    assert strategy_proof(endowment, prefs) == [], (endowment, prefs)


if __name__ == "__main__":
    e3 = {1: 'a', 2: 'b', 3: 'c'}
    e2 = {1: 'a', 2: 'b'}

    # (a)-(c): hand-computed TTC outcomes
    p_swap = {1: ['b', 'c', 'a'], 2: ['a', 'c', 'b'], 3: ['a', 'b', 'c']}
    assert ttc(e3, p_swap) == {1: 'b', 2: 'a', 3: 'c'}
    p_cycle = {1: ['b', 'a', 'c'], 2: ['c', 'b', 'a'], 3: ['a', 'c', 'b']}
    assert ttc(e3, p_cycle) == {1: 'b', 2: 'c', 3: 'a'}
    p_own = {1: ['a', 'b'], 2: ['b', 'a']}
    assert ttc(e2, p_own) == {1: 'a', 2: 'b'}

    # (d): full battery on each of those profiles
    for endowment, prefs in [(e3, p_swap), (e3, p_cycle), (e2, p_own)]:
        check_profile(endowment, prefs)

    # (e): each checker must reject a bad allocation
    # IR and PE, but agents 1 and 2 both gain by swapping their goods b and a
    p_blocked = {1: ['b', 'c', 'a'], 2: ['a', 'b', 'c'], 3: ['a', 'c', 'b']}
    not_core = {1: 'c', 2: 'b', 3: 'a'}
    assert individually_rational(e3, p_blocked, not_core)
    assert pareto_efficient(p_blocked, not_core)
    assert not in_core(e3, p_blocked, not_core)

    # agent 1 receives c, which it ranks below its own endowment a
    p_ir = {1: ['b', 'a', 'c'], 2: ['a', 'b', 'c'], 3: ['b', 'c', 'a']}
    not_ir = {1: 'c', 2: 'a', 3: 'b'}
    assert not individually_rational(e3, p_ir, not_ir)

    # identity is IR, but agents 1 and 2 both gain by swapping
    p_pe = {1: ['b', 'a', 'c'], 2: ['a', 'b', 'c'], 3: ['a', 'b', 'c']}
    not_pe = {1: 'a', 2: 'b', 3: 'c'}
    assert individually_rational(e3, p_pe, not_pe)
    assert not pareto_efficient(p_pe, not_pe)

    print("OK")
