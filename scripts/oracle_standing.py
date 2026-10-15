#!/usr/bin/env python3
# PROVENANCE. Written on 2026-10-11 by a subagent (the owner's standing rule: subagents use the smallest model
# tier) from a written specification alone: it never saw this repository's implementation of the same
# functions. Added by the assistant: this comment, the shebang line, the function-name prefix in the worked
# examples below (they were the subagent's separate self-test file, moved here so that the test suite runs them
# the way it runs the other oracles), and the closing `print("OK")`. It is an UNTRUSTED oracle: agreement between
# it and the package shows that implementation slips are caught, not that the definitions are right
# (TECHNICAL_DEBT.md, HO-024). It finds envy loops by trying every ordered selection of agents, pecking orders by
# trying every ordering, and the top trading cycles by walking pointers, where the package searches depth-first,
# peels sinks and executes a round at a time. Run by differential_oracles.py.
"""Reference implementation of the "standing" specification (`validation/oracle-standing-specification.md`).

Standard library only. Exact arithmetic throughout: fractions.Fraction, no floats.
"""

import itertools
from fractions import Fraction


def envy_edges(endowment, prefs):
    """Pairs (i, j), i != j, where agent i ranks j's endowed good strictly above its own."""
    edges = set()
    for i in endowment:
        for j in endowment:
            if i != j and prefs[i].index(endowment[j]) < prefs[i].index(endowment[i]):
                edges.add((i, j))
    return edges


def has_envy_loop(endowment, prefs):
    """True iff some sequence of k >= 2 distinct agents has every consecutive pair
    (cyclically) an envy edge. Brute force: every ordered selection, for k = 2..n."""
    edges = envy_edges(endowment, prefs)
    agents = list(endowment)
    for k in range(2, len(agents) + 1):
        for seq in itertools.permutations(agents, k):
            if all((seq[t], seq[(t + 1) % k]) in edges for t in range(k)):
                return True
    return False


def pecking_orders(endowment, prefs):
    """Every agent ordering where, at each position k, the agent's endowed good is its
    favourite among the goods endowed to the agents at positions k..n-1 (itself and after)."""
    found = []
    for order in itertools.permutations(list(endowment)):
        ok = True
        for k, agent in enumerate(order):
            pool = [endowment[b] for b in order[k:]]
            if min(pool, key=prefs[agent].index) != endowment[agent]:
                ok = False
                break
        if ok:
            found.append(order)
    return found


def serial_dictatorship(order, prefs):
    """Agents pick in the given order; each takes its favourite good not yet taken."""
    taken = set()
    result = {}
    for agent in order:
        good = next(g for g in prefs[agent] if g not in taken)
        result[agent] = good
        taken.add(good)
    return result


def wealth_start(wealth, prefs):
    """Serial dictatorship in decreasing order of wealth; wealth values must be distinct."""
    values = list(wealth.values())
    if len(set(values)) != len(values):
        raise ValueError("two agents have equal wealth")
    order = tuple(sorted(wealth, key=lambda agent: wealth[agent], reverse=True))
    picks = serial_dictatorship(order, prefs)
    return {agent: picks[agent] for agent in prefs}


def top_trading_cycles(endowment, prefs):
    """Gale's top trading cycles (SPEC.md item 6). Returns agent -> good received,
    keys in endowment order."""
    owner = {good: agent for agent, good in endowment.items()}
    remaining = list(endowment)
    received = {}
    while remaining:
        live_goods = {endowment[a] for a in remaining}
        points_at = {}
        for a in remaining:
            favourite = next(g for g in prefs[a] if g in live_goods)
            points_at[a] = owner[favourite]
        # a is on a cycle iff following the pointers from a returns to a; a cycle through a
        # has length at most len(remaining), so that many steps are enough to find it
        on_cycle = []
        for a in remaining:
            x = points_at[a]
            for _ in range(len(remaining)):
                if x == a:
                    on_cycle.append(a)
                    break
                x = points_at[x]
        for a in on_cycle:
            received[a] = endowment[points_at[a]]
        remaining = [a for a in remaining if a not in on_cycle]
    return {a: received[a] for a in endowment}


def thaw_ranking(goods, price, weights, lam):
    """Goods best first, by score = dot((1 - lam) * price + lam * weights, good's vector);
    goods with equal scores keep listing order."""
    lam = Fraction(lam)
    price = [Fraction(p) for p in price]
    weights = [Fraction(w) for w in weights]
    if len(weights) != len(price):
        raise ValueError("price and weights must have the same dimension")
    effective = [(1 - lam) * price[c] + lam * weights[c] for c in range(len(price))]
    scored = []
    for index, good in enumerate(goods):  # dict key order is the listing order
        vector = [Fraction(v) for v in goods[good]]
        if len(vector) != len(effective):
            raise ValueError("every good vector must have the same dimension as price")
        score = sum((effective[c] * vector[c] for c in range(len(effective))), Fraction(0))
        scored.append((-score, index, good))
    scored.sort(key=lambda item: (item[0], item[1]))
    return tuple(good for _, _, good in scored)


def thaw_allocation(endowment, goods, price, all_weights, lam):
    """Each agent ranks goods with thaw_ranking; the allocation is the top trading cycles
    of the endowment under those rankings."""
    rankings = {agent: thaw_ranking(goods, price, all_weights[agent], lam) for agent in endowment}
    return top_trading_cycles(endowment, rankings)


def movers(endowment, allocation):
    """Agents, in endowment order, whose allocated good differs from their endowed good."""
    return tuple(agent for agent in endowment if allocation[agent] != endowment[agent])


if __name__ == "__main__":
    def raises(exc_type, fn, *args):
        try:
            fn(*args)
        except exc_type:
            return True
        return False


    # Example A
    endA = {"A": "g1", "B": "g2"}
    prefA = {"A": ["g2", "g1"], "B": ["g1", "g2"]}
    assert envy_edges(endA, prefA) == {("A", "B"), ("B", "A")}
    assert has_envy_loop(endA, prefA) is True
    assert pecking_orders(endA, prefA) == []
    assert top_trading_cycles(endA, prefA) == {"A": "g2", "B": "g1"}

    # Example B
    endB = {"A": "g2", "B": "g1"}
    prefB = {"A": ["g1", "g2"], "B": ["g1", "g2"]}
    assert envy_edges(endB, prefB) == {("A", "B")}
    assert has_envy_loop(endB, prefB) is False
    assert pecking_orders(endB, prefB) == [("B", "A")]
    ttcB = top_trading_cycles(endB, prefB)
    assert ttcB == {"A": "g2", "B": "g1"}
    assert movers(endB, ttcB) == ()

    # Example C
    prefC = {"A": ["g1", "g2"], "B": ["g1", "g2"]}
    sd = serial_dictatorship(("B", "A"), prefC)
    assert sd == {"B": "g1", "A": "g2"} and list(sd) == ["B", "A"]
    ws = wealth_start({"A": 1, "B": 5}, prefC)
    assert ws == {"A": "g2", "B": "g1"} and list(ws) == ["A", "B"]

    # Example D: the price-blind swap
    goodsD = {"y": (1, 5), "x": (5, 1)}  # listing order y, x
    priceD = (1, 1)
    weightsD = {"A": (3, 1), "B": (1, 3)}
    endD = {"A": "y", "B": "x"}
    lam0, lamHalf, lam1 = Fraction(0), Fraction(1, 2), Fraction(1)
    assert thaw_ranking(goodsD, priceD, weightsD["A"], lam0) == ("y", "x")
    assert thaw_ranking(goodsD, priceD, weightsD["B"], lam0) == ("y", "x")
    assert thaw_allocation(endD, goodsD, priceD, weightsD, lam0) == {"A": "y", "B": "x"}
    assert movers(endD, thaw_allocation(endD, goodsD, priceD, weightsD, lam0)) == ()
    assert thaw_ranking(goodsD, priceD, weightsD["A"], lamHalf) == ("x", "y")
    assert thaw_ranking(goodsD, priceD, weightsD["B"], lamHalf) == ("y", "x")
    assert thaw_allocation(endD, goodsD, priceD, weightsD, lamHalf) == {"A": "x", "B": "y"}
    assert thaw_allocation(endD, goodsD, priceD, weightsD, lam1) == {"A": "x", "B": "y"}

    # Example E
    endE = {1: "g3", 2: "g1", 3: "g2"}
    prefE = {1: ["g1", "g2", "g3"], 2: ["g1", "g2", "g3"], 3: ["g1", "g2", "g3"]}
    assert envy_edges(endE, prefE) == {(1, 2), (1, 3), (3, 2)}
    assert has_envy_loop(endE, prefE) is False
    assert pecking_orders(endE, prefE) == [(2, 3, 1)]
    assert top_trading_cycles(endE, prefE) == endE

    # Extra 1 (hand-derived): a 3-cycle of envy, and a 3-cycle of trade
    end3 = {1: "g1", 2: "g2", 3: "g3"}
    pref3 = {1: ["g2", "g1", "g3"], 2: ["g3", "g2", "g1"], 3: ["g1", "g3", "g2"]}
    assert envy_edges(end3, pref3) == {(1, 2), (2, 3), (3, 1)}
    assert has_envy_loop(end3, pref3) is True
    assert pecking_orders(end3, pref3) == []
    assert top_trading_cycles(end3, pref3) == {1: "g2", 2: "g3", 3: "g1"}

    # Extra 2 (hand-derived): agent 2 points into the 1-3 cycle and keeps its own good
    endT = {1: "g1", 2: "g2", 3: "g3"}
    prefT = {1: ["g3", "g1", "g2"], 2: ["g1", "g2", "g3"], 3: ["g1", "g3", "g2"]}
    assert top_trading_cycles(endT, prefT) == {1: "g3", 2: "g2", 3: "g1"}

    # Guard rails
    assert raises(ValueError, wealth_start, {"A": 3, "B": 3}, prefC)
    assert raises(ValueError, thaw_ranking, {"y": (1, 5, 2)}, priceD, weightsD["A"], lam0)

    print("OK")
