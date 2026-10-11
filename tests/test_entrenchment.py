"""Who can change the rule that says who can change the rule: protection, authority and the loop.

A RULE is governed by another rule, the one that says how it may be changed (or by nothing, if
it cannot be changed at all). Every rule also has a PROCEDURE with a price in bits: what it
takes to satisfy the procedure it defines, which is what whoever wants to change a rule it
governs has to pay.

Protection. To change rule u you may satisfy the procedure of its governor, or first change the
governor and then change u for nothing. So the cheapest way is

    protection(u) = min over every rule v above u (governor, governor's governor, ...) of
                    the price of v's procedure,

the WEAKEST LINK of the chain of authority. On a cycle every member is above every other, so
every member of a cycle has the same protection, the cheapest procedure on the cycle. This is
Ross's old paradox: an entrenched clause amendable by an ordinary majority is protected by
an ordinary majority, whatever it says about itself.

Authority. The HEIGHT of a rule is the number of steps up the chain of governors before it ends.
It is finite iff the chain ends at a rule nothing governs: an unamendable apex, which is a
sovereign in hyperethics' sense. If no rule is ungoverned, every chain in a finite set of rules
returns to itself (pigeonhole) and every height is omega: there is no finite rank for validity.
A constitution that obeys the no-apex rule has no grounded top, and its authority is a loop.

Checked here
    * cycles and heights against a direct walk, over EVERY functional graph on up to five rules;
    * protection against an independent shortest-path search over sets of already-changed
      rules, for random rule sets with prices 0, finite and infinite;
    * the weakest-link and cycle-equality theorems as properties of those same graphs;
    * the finite-rank / unamendable-apex equivalence, exhaustively.
"""

from __future__ import annotations

import heapq
import itertools
import math
import random

import pytest

from hyperoptimization import entrenchment as E
from hyperoptimization.ground import OMEGA


def rules_from(governors, prices):
    names = [f"r{i}" for i in range(len(governors))]
    return [
        E.Rule(names[i], prices[i], None if g is None else names[g])
        for i, g in enumerate(governors)
    ]


def all_governor_choices(n):
    return itertools.product([None, *range(n)], repeat=n)


# --- construction --------------------------------------------------------------------------------------


def test_a_rule_has_a_name_a_nonnegative_price_and_a_governor_that_exists():
    E.Rule("a", 3, None)
    E.Rule("a", math.inf, "b")
    for bad in (("", 1, None), (3, 1, None), ("a", -1, None), ("a", "x", None), ("a", True, None),
                ("a", math.nan, None), ("a", 1, 7)):
        with pytest.raises((ValueError, TypeError)):
            E.Rule(*bad)


def test_a_constitution_names_each_rule_once_and_every_governor_must_be_in_it():
    with pytest.raises(ValueError, match="repeat"):
        E.protection([E.Rule("a", 1, None), E.Rule("a", 2, None)])
    with pytest.raises(ValueError, match="governor"):
        E.protection([E.Rule("a", 1, "ghost")])
    with pytest.raises(ValueError, match="at least one"):
        E.protection([])


# --- cycles and heights, over every functional graph -----------------------------------------------------------


def walk_up(governors, start):
    """Indices above `start` in order, ending with None or a repeat."""
    seen, node = [], start
    while node is not None and node not in seen:
        seen.append(node)
        node = governors[node]
    return seen, node


def test_cycles_are_found_exactly_over_every_functional_graph_on_up_to_five_rules():
    count = 0
    for n in range(1, 6):
        for governors in all_governor_choices(n):
            rules = rules_from(governors, [1] * n)
            on_cycle = set()
            for i in range(n):
                _seen, end = walk_up(governors, i)
                if end is not None:  # the walk met a repeat: i leads into a cycle
                    cycle, node = [end], governors[end]
                    while node != end:
                        cycle.append(node)
                        node = governors[node]
                    on_cycle |= set(cycle)
            found = {name for cyc in E.cycles(rules) for name in cyc}
            assert found == {f"r{i}" for i in on_cycle}, governors
            count += 1
    assert count == sum((k + 1) ** k for k in range(1, 6))


def test_a_cycle_is_listed_once_in_the_order_of_authority():
    rules = rules_from([1, 2, 0, None], [1, 1, 1, 1])
    assert E.cycles(rules) == [("r0", "r1", "r2")]  # r0 is governed by r1, r1 by r2, r2 by r0
    assert E.cycles(rules_from([None, 0], [1, 1])) == []
    assert E.cycles(rules_from([0], [1])) == [("r0",)]  # a rule that governs itself


def test_authority_height_counts_steps_to_the_apex_and_is_omega_if_there_is_none():
    count = 0
    for n in range(1, 6):
        for governors in all_governor_choices(n):
            heights = E.authority_height(rules_from(governors, [1] * n))
            for i in range(n):
                visited, end = walk_up(governors, i)
                # the walk ends at None (a chain that reaches an apex: height = governors passed)
                # or at a repeat (a chain that never ends: omega)
                want = len(visited) - 1 if end is None else OMEGA
                assert heights[f"r{i}"] == want, (governors, i)
            count += 1
    assert count == sum((k + 1) ** k for k in range(1, 6))


def test_no_finite_rank_exists_unless_something_is_unamendable():
    # If every rule has a governor, a finite set of rules must close into a loop: all heights omega.
    for n in range(1, 6):
        for governors in itertools.product(range(n), repeat=n):
            heights = E.authority_height(rules_from(governors, [1] * n))
            assert all(h is OMEGA for h in heights.values()), governors
    # and the converse direction: a finite height anywhere means an ungoverned rule is above it
    for n in range(1, 6):
        for governors in all_governor_choices(n):
            heights = E.authority_height(rules_from(governors, [1] * n))
            if any(h is not OMEGA for h in heights.values()):
                assert None in governors


def test_the_apex_is_the_set_of_ungoverned_rules():
    rules = rules_from([None, 0, 1, None], [1, 1, 1, 1])
    assert E.apex(rules) == ("r0", "r3")
    assert E.apex(rules_from([1, 0], [1, 1])) == ()


# --- protection, against an independent search ---------------------------------------------------------------------------


def cheapest_change(rules, target):
    """Dijkstra over sets of changed rules. Changing u costs the price of its governor's
    procedure unless the governor has already been changed, when the procedure is free.
    A rule nothing governs cannot be changed."""
    n = len(rules)
    index = {r.name: i for i, r in enumerate(rules)}
    gov = [None if r.governed_by is None else index[r.governed_by] for r in rules]
    price = [r.procedure_bits for r in rules]
    goal = index[target]
    start = 0
    best = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        cost, mask = heapq.heappop(heap)
        if cost > best.get(mask, math.inf):
            continue
        if mask >> goal & 1:
            return cost
        for u in range(n):
            if mask >> u & 1 or gov[u] is None:
                continue
            step = 0.0 if mask >> gov[u] & 1 else price[gov[u]]
            if step == math.inf:
                continue
            nxt = mask | 1 << u
            if cost + step < best.get(nxt, math.inf):
                best[nxt] = cost + step
                heapq.heappush(heap, (cost + step, nxt))
    return math.inf


def test_protection_is_the_weakest_link_by_independent_shortest_path():
    rng = random.Random(91)
    for _ in range(400):
        n = rng.randint(1, 7)
        governors = [rng.choice([None, *range(n)]) for _ in range(n)]
        prices = [rng.choice([0, 1, 2, 5, 9, 30, math.inf]) for _ in range(n)]
        rules = rules_from(governors, prices)
        got = E.protection(rules)
        for r in rules:
            assert got[r.name] == cheapest_change(rules, r.name), (governors, prices, r.name)


def test_ross_an_entrenched_clause_is_only_as_strong_as_the_clause_that_amends_it():
    rules = [
        E.Rule("free speech", 100, "amendment clause"),  # a procedure worth 100 bits ...
        E.Rule("amendment clause", 5, "amendment clause"),  # ... amendable by a 5-bit procedure
    ]
    protection = E.protection(rules)
    assert protection["free speech"] == 5 and protection["amendment clause"] == 5


def test_protection_never_exceeds_that_of_the_governor_and_is_equal_around_a_cycle():
    rng = random.Random(92)
    for _ in range(400):
        n = rng.randint(1, 7)
        governors = [rng.choice([None, *range(n)]) for _ in range(n)]
        prices = [rng.choice([0, 1, 3, 8, math.inf]) for _ in range(n)]
        rules = rules_from(governors, prices)
        protection = E.protection(rules)
        for i, g in enumerate(governors):
            if g is not None:
                assert protection[f"r{i}"] <= protection[f"r{g}"]
        for cycle in E.cycles(rules):
            values = {protection[name] for name in cycle}
            assert len(values) == 1
            assert values == {min(prices[int(name[1:])] for name in cycle)}


def test_an_ungoverned_rule_cannot_be_changed_at_any_price():
    rules = [E.Rule("root", 3, None), E.Rule("child", 7, "root")]
    protection = E.protection(rules)
    assert protection["root"] == math.inf and protection["child"] == 3


def test_the_weakest_link_names_the_rule_whose_procedure_sets_the_protection():
    rules = [
        E.Rule("statute", 1, "code"),
        E.Rule("code", 50, "constitution"),
        E.Rule("constitution", 9, "convention"),
        E.Rule("convention", 200, "constitution"),
    ]
    name, bits = E.weakest_link(rules, "statute")
    assert (name, bits) == ("constitution", 9)
    assert E.protection(rules)["statute"] == 9
    assert E.weakest_link(rules, "constitution") == ("constitution", 9)  # around the cycle
    with pytest.raises(KeyError):
        E.weakest_link(rules, "nowhere")


def test_a_rule_with_no_governor_has_no_weakest_link_to_report():
    assert E.weakest_link([E.Rule("root", 3, None)], "root") == (None, math.inf)
