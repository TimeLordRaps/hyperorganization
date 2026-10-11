"""Standing: what a start gives, and what can move from it without money.

`rival` exchanges indivisible goods by top trading cycles from a starting allocation it does not derive. The
owner's statement of 2026-10-11 (PROVENANCE.md) is that the start cannot be drawn from a uniform distribution:
it is the present world, which money built, and the way to a future without money has to be extrapolated from
it. `standing` is the first thing that can be said exactly about any start:

  * *envy*: an agent envies another when it ranks the other's good above its own. Trade is possible exactly when
    envy has a loop, which is exactly when the start is not Pareto efficient;
  * *pecking order*: an ordering of the agents in which each holds its favourite of what it and those after it
    hold. A start has one exactly when envy has no loop: the shape of a ladder;
  * *wealth start*: the start that results when the wealthiest agent picks first, and so on down. It is a ladder,
    and top trading cycles leaves it where it is, so abolishing money while keeping holdings freezes it wherever
    the rankings agree;
  * *the thaw*: a dial from the rankings a common price induces to private rankings, exact in rational
    arithmetic, that shows what unfreezes a ladder: a difference in the DIRECTION of what agents value, not in
    its magnitude, and so more than one coordinate.

Everything is checked from the definitions: every strict market of two and three agents (the endowment ranges
over every endowment, not only the identity), seeded markets of four and five, hand-derived worked examples, and
doctored inputs the properties must reject. The theorems these meet are not cited from memory as proof.
"""

from __future__ import annotations

import itertools
import os
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from hyperoptimization import rival, standing

ROOT = Path(__file__).resolve().parent.parent


def markets(n):
    """Every (endowment, rankings) of n agents: every way to give out the goods and every strict profile."""
    agents, goods = list(range(n)), [f"g{i}" for i in range(n)]
    for held in itertools.permutations(goods):
        for choice in itertools.product(itertools.permutations(goods), repeat=n):
            yield dict(zip(agents, held)), dict(zip(agents, choice))


def seeded_markets(n, count, seed):
    rng = random.Random(seed)
    agents, goods = list(range(n)), [f"g{i}" for i in range(n)]
    for _ in range(count):
        held = goods[:]
        rng.shuffle(held)
        yield dict(zip(agents, held)), {a: tuple(rng.sample(goods, n)) for a in agents}


def efficient(endowment, prefs):
    return rival.pareto_efficient(endowment, prefs, endowment)


def holds_a_pecking_order(endowment, prefs, order):
    """The definition, written out here and not taken from the module: at each place the agent's good is its
    favourite among the goods held by it and by every agent after it."""
    for place, agent in enumerate(order):
        pool = [endowment[b] for b in order[place:]]
        if min(pool, key=prefs[agent].index) != endowment[agent]:
            return False
    return True


# --- worked examples, derived by hand ---------------------------------------------------------------------------


def test_two_agents_who_each_want_the_other_s_good_have_a_loop_and_swap():
    endowment, prefs = {"A": "g1", "B": "g2"}, {"A": ["g2", "g1"], "B": ["g1", "g2"]}
    assert standing.envy_graph(endowment, prefs) == {"A": ("B",), "B": ("A",)}
    assert standing.envy_loop(endowment, prefs) == ("A", "B")
    assert not standing.frozen(endowment, prefs)
    assert standing.pecking_order(endowment, prefs) is None
    assert rival.top_trading_cycles(endowment, prefs) == {"A": "g2", "B": "g1"}


def test_two_agents_who_want_the_same_good_are_a_ladder_and_nothing_moves():
    endowment, prefs = {"A": "g2", "B": "g1"}, {"A": ["g1", "g2"], "B": ["g1", "g2"]}
    assert standing.envy_graph(endowment, prefs) == {"A": ("B",), "B": ()}
    assert standing.envy_loop(endowment, prefs) is None
    assert standing.frozen(endowment, prefs)
    assert standing.pecking_order(endowment, prefs) == ("B", "A")
    assert not standing.is_pecking_order(endowment, prefs, ("A", "B"))
    assert rival.top_trading_cycles(endowment, prefs) == endowment


def test_a_ladder_of_three_has_one_pecking_order_and_envy_points_up_it():
    endowment, prefs = {1: "g3", 2: "g1", 3: "g2"}, {a: ["g1", "g2", "g3"] for a in (1, 2, 3)}
    assert standing.envy_graph(endowment, prefs) == {1: (2, 3), 2: (), 3: (2,)}
    assert standing.pecking_order(endowment, prefs) == (2, 3, 1)
    assert [o for o in itertools.permutations(endowment) if holds_a_pecking_order(endowment, prefs, o)] == [
        (2, 3, 1)]
    assert rival.top_trading_cycles(endowment, prefs) == endowment


def test_when_every_agent_holds_its_own_favourite_every_order_is_a_pecking_order_and_the_first_is_chosen():
    endowment = {1: "g1", 2: "g2", 3: "g3"}
    prefs = {1: ["g1", "g2", "g3"], 2: ["g2", "g3", "g1"], 3: ["g3", "g1", "g2"]}
    assert standing.envy_graph(endowment, prefs) == {1: (), 2: (), 3: ()}
    assert all(standing.is_pecking_order(endowment, prefs, o) for o in itertools.permutations(endowment))
    assert standing.pecking_order(endowment, prefs) == (1, 2, 3)  # the earliest agent that envies nobody left


def test_a_loop_is_a_loop_of_three_when_each_envies_the_next():
    endowment = {1: "g1", 2: "g2", 3: "g3"}
    prefs = {1: ["g2", "g3", "g1"], 2: ["g3", "g1", "g2"], 3: ["g1", "g2", "g3"]}
    assert standing.envy_graph(endowment, prefs) == {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    loop = standing.envy_loop(endowment, prefs)
    assert loop is not None and loop[0] == 1 and set(loop) <= {1, 2, 3} and len(loop) in (2, 3)
    assert rival.top_trading_cycles(endowment, prefs) == {1: "g2", 2: "g3", 3: "g1"}


def test_one_agent_is_a_ladder_of_one():
    endowment, prefs = {"A": "g"}, {"A": ["g"]}
    assert standing.envy_graph(endowment, prefs) == {"A": ()}
    assert standing.envy_loop(endowment, prefs) is None
    assert standing.pecking_order(endowment, prefs) == ("A",)
    assert standing.wealth_start({"A": 3}, prefs) == {"A": "g"}


def test_the_empty_market_is_frozen_and_has_the_empty_pecking_order():
    assert standing.envy_graph({}, {}) == {}
    assert standing.frozen({}, {})
    assert standing.pecking_order({}, {}) == ()
    assert standing.serial_dictatorship((), {}) == {}


# --- trade, envy loops and Pareto efficiency are one fact -----------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3])
def test_trade_happens_exactly_when_envy_has_a_loop_which_is_exactly_when_the_start_is_inefficient(n):
    count = 0
    for endowment, prefs in markets(n):
        count += 1
        loop = standing.envy_loop(endowment, prefs)
        moved = bool(rival.movers(endowment, rival.top_trading_cycles(endowment, prefs)))
        assert (loop is not None) == moved == (not efficient(endowment, prefs)), (endowment, prefs, loop)
        assert standing.frozen(endowment, prefs) == (not moved)
    assert count == {2: 8, 3: 1296}[n]


@pytest.mark.parametrize(("n", "count"), [(4, 250), (5, 40)])
def test_the_same_on_seeded_markets_of_four_and_five_agents(n, count):
    loops = 0
    for endowment, prefs in seeded_markets(n, count, seed=1000 + n):
        loop = standing.envy_loop(endowment, prefs)
        moved = bool(rival.movers(endowment, rival.top_trading_cycles(endowment, prefs)))
        assert (loop is not None) == moved == (not efficient(endowment, prefs)), (endowment, prefs)
        loops += loop is not None
    assert 0 < loops < count  # both kinds of start occur, so the equivalence is not met by one side only


@pytest.mark.parametrize("n", [2, 3])
def test_the_envy_loop_returned_is_a_loop_of_distinct_agents_beginning_with_its_earliest_member(n):
    seen = 0
    for endowment, prefs in markets(n):
        loop = standing.envy_loop(endowment, prefs)
        if loop is None:
            continue
        seen += 1
        graph = standing.envy_graph(endowment, prefs)
        assert len(loop) >= 2 and len(set(loop)) == len(loop)
        assert all(loop[(k + 1) % len(loop)] in graph[loop[k]] for k in range(len(loop)))
        assert loop[0] == min(loop, key=list(endowment).index)
    assert seen > 0


@pytest.mark.parametrize("n", [2, 3])
def test_a_pecking_order_exists_exactly_when_the_start_is_efficient_and_the_module_finds_it(n):
    for endowment, prefs in markets(n):
        orders = [o for o in itertools.permutations(endowment) if holds_a_pecking_order(endowment, prefs, o)]
        found = standing.pecking_order(endowment, prefs)
        assert bool(orders) == efficient(endowment, prefs), (endowment, prefs)
        assert (found is not None) == bool(orders)
        if found is not None:
            assert found in orders
        for order in itertools.permutations(endowment):  # the module's own test agrees with the definition
            assert standing.is_pecking_order(endowment, prefs, order) == holds_a_pecking_order(
                endowment, prefs, order)


def test_a_pecking_order_exists_exactly_when_the_start_is_efficient_on_seeded_markets_of_four_and_five():
    for n, count in ((4, 120), (5, 25)):
        for endowment, prefs in seeded_markets(n, count, seed=2000 + n):
            found = standing.pecking_order(endowment, prefs)
            assert (found is not None) == efficient(endowment, prefs)
            if found is not None:
                assert holds_a_pecking_order(endowment, prefs, found)


@pytest.mark.parametrize("n", [2, 3])
def test_after_the_exchange_envy_has_no_loop_and_nothing_moves_again(n):
    for endowment, prefs in markets(n):
        outcome = rival.top_trading_cycles(endowment, prefs)
        assert standing.frozen(outcome, prefs)
        assert standing.pecking_order(outcome, prefs) is not None
        assert rival.top_trading_cycles(outcome, prefs) == outcome


def test_the_survey_of_three_agents_splits_into_ladders_and_loops_as_the_exchange_does():
    endowment = {0: "g0", 1: "g1", 2: "g2"}
    counts = {"frozen": 0, "loop": 0}
    for choice in itertools.product(itertools.permutations(["g0", "g1", "g2"]), repeat=3):
        prefs = dict(zip(range(3), choice))
        counts["frozen" if standing.frozen(endowment, prefs) else "loop"] += 1
    assert counts == {"frozen": rival.survey(3)["still"], "loop": rival.survey(3)["trade"]}
    assert counts == {"frozen": 98, "loop": 118}


# --- the ladder of a start made by wealth ---------------------------------------------------------------------


def test_serial_dictatorship_gives_each_agent_its_favourite_of_what_is_left():
    prefs = {"A": ["g1", "g2"], "B": ["g1", "g2"]}
    assert standing.serial_dictatorship(("B", "A"), prefs) == {"B": "g1", "A": "g2"}
    assert list(standing.serial_dictatorship(("B", "A"), prefs)) == ["B", "A"]
    assert standing.wealth_start({"A": 1, "B": 5}, prefs) == {"A": "g2", "B": "g1"}
    assert list(standing.wealth_start({"A": 1, "B": 5}, prefs)) == ["A", "B"]  # listed as the rankings are


@pytest.mark.parametrize("n", [2, 3])
def test_a_start_made_by_wealth_is_efficient_frozen_and_envy_points_up_the_wealth_order(n):
    agents, goods = list(range(n)), [f"g{i}" for i in range(n)]
    pairs = 0
    for choice in itertools.product(itertools.permutations(goods), repeat=n):
        prefs = dict(zip(agents, choice))
        for ranked in itertools.permutations(agents):  # ranked[0] is the wealthiest
            wealth = {a: Fraction(n - place) for place, a in enumerate(ranked)}
            start = standing.wealth_start(wealth, prefs)
            pairs += 1
            assert sorted(start.values()) == goods  # a rearrangement of the goods, one each
            assert efficient(start, prefs)
            assert standing.frozen(start, prefs)
            assert rival.top_trading_cycles(start, prefs) == start
            assert holds_a_pecking_order(start, prefs, ranked)
            assert standing.is_pecking_order(start, prefs, ranked)
            assert start[ranked[0]] == prefs[ranked[0]][0]  # the wealthiest holds its first choice
            for envious, targets in standing.envy_graph(start, prefs).items():
                assert all(wealth[t] > wealth[envious] for t in targets)  # nobody envies a poorer agent
    assert pairs == {2: 4 * 2, 3: 216 * 6}[n]


@pytest.mark.parametrize(("n", "count"), [(4, 60), (6, 40), (9, 25), (30, 6)])
def test_the_same_on_seeded_wealth_starts_of_larger_markets(n, count):
    rng = random.Random(3000 + n)
    agents, goods = list(range(n)), [f"g{i}" for i in range(n)]
    for _ in range(count):
        prefs = {a: tuple(rng.sample(goods, n)) for a in agents}
        wealth = dict(zip(agents, rng.sample(range(10 * n), n)))
        start = standing.wealth_start(wealth, prefs)
        ranked = sorted(agents, key=lambda a: -wealth[a])
        assert sorted(start.values()) == sorted(goods)
        assert standing.frozen(start, prefs)
        assert holds_a_pecking_order(start, prefs, ranked)
        assert rival.top_trading_cycles(start, prefs) == start
        if n <= 6:
            assert efficient(start, prefs)


def test_a_ladder_is_not_the_only_efficient_start_so_a_pecking_order_need_not_follow_wealth():
    # wealth puts A above B, but this efficient start has B picking first: the wealth order is not forced
    endowment, prefs = {"A": "g2", "B": "g1"}, {"A": ["g2", "g1"], "B": ["g1", "g2"]}
    assert standing.frozen(endowment, prefs)
    assert standing.pecking_order(endowment, prefs) == ("A", "B")
    assert holds_a_pecking_order(endowment, prefs, ("B", "A"))  # both orders work: each holds its first choice
    assert standing.wealth_start({"A": 1, "B": 9}, prefs) == {"A": "g2", "B": "g1"}  # and wealth reaches it too


def test_equal_wealth_has_no_pick_order_and_is_refused():
    prefs = {"A": ["g1", "g2"], "B": ["g1", "g2"]}
    with pytest.raises(ValueError, match="equal wealth"):
        standing.wealth_start({"A": 3, "B": 3}, prefs)
    with pytest.raises(ValueError, match="equal wealth"):
        standing.wealth_start({"A": Fraction(6, 2), "B": 3}, prefs)


@pytest.mark.parametrize("wealth", [{"A": 1.5, "B": 2}, {"A": True, "B": 2}, {"A": "1", "B": 2}])
def test_wealth_is_an_integer_or_a_fraction_never_a_float(wealth):
    with pytest.raises(TypeError):
        standing.wealth_start(wealth, {"A": ["g1", "g2"], "B": ["g1", "g2"]})


def test_wealth_must_belong_to_exactly_the_agents_who_rank():
    prefs = {"A": ["g1", "g2"], "B": ["g1", "g2"]}
    for wealth in ({"A": 1}, {"A": 1, "B": 2, "C": 3}, {"A": 1, "C": 2}):
        with pytest.raises(ValueError, match="wealth"):
            standing.wealth_start(wealth, prefs)


def test_serial_dictatorship_refuses_an_order_or_rankings_that_do_not_describe_a_market():
    prefs = {"A": ["g1", "g2"], "B": ["g1", "g2"]}
    for order in (("A",), ("A", "A"), ("A", "B", "C"), ("A", "C")):
        with pytest.raises(ValueError, match="order"):
            standing.serial_dictatorship(order, prefs)
    with pytest.raises(ValueError, match="rank"):
        standing.serial_dictatorship(("A", "B"), {"A": ["g1", "g2"], "B": ["g1", "g3"]})
    with pytest.raises(ValueError, match="rank"):
        standing.serial_dictatorship(("A", "B"), {"A": ["g1", "g2", "g3"], "B": ["g1", "g2", "g3"]})
    with pytest.raises(ValueError, match="rank"):
        standing.serial_dictatorship(("A", "B"), {"A": ["g1", "g1"], "B": ["g1", "g2"]})


def test_is_pecking_order_refuses_an_order_that_is_not_a_rearrangement_of_the_agents():
    endowment, prefs = {"A": "g2", "B": "g1"}, {"A": ["g1", "g2"], "B": ["g1", "g2"]}
    for order in (("B",), ("B", "B"), ("B", "A", "C"), ("B", "C")):
        with pytest.raises(ValueError, match="order"):
            standing.is_pecking_order(endowment, prefs, order)


def test_the_envy_functions_refuse_what_is_not_a_strict_market():
    with pytest.raises(ValueError):
        standing.envy_graph({"A": "g1", "B": "g1"}, {"A": ["g1"], "B": ["g1"]})
    with pytest.raises(ValueError):
        standing.envy_graph({"A": "g1", "B": "g2"}, {"A": ["g1", "g2"]})
    with pytest.raises(ValueError):
        standing.pecking_order({"A": "g1", "B": "g2"}, {"A": ["g1", "g2"], "B": ["g1"]})


def test_a_ladder_of_hundreds_is_found_in_the_time_it_takes_to_write_the_graph():
    n = 300
    same = {a: tuple(range(n)) for a in range(n)}  # everybody ranks good 0 best, then 1, ...
    ladder = {a: a for a in range(n)}
    assert standing.pecking_order(ladder, same) == tuple(range(n))
    assert standing.frozen(ladder, same)
    upside_down = {a: n - 1 - a for a in range(n)}
    assert standing.pecking_order(upside_down, same) == tuple(range(n - 1, -1, -1))
    assert standing.wealth_start({a: a for a in range(n)}, same) == {a: n - 1 - a for a in range(n)}


# --- the thaw: from the rankings a price induces to private rankings -------------------------------------------


def swap_instance():
    """The price-blind swap of rival: (5, 1) and (1, 5) have equal worth at the price (1, 1); A holds the
    y-rich one and weighs x three times as much as y, B the reverse."""
    goods = {"y": (1, 5), "x": (5, 1)}
    return {"A": "y", "B": "x"}, goods, (1, 1), {"A": (3, 1), "B": (1, 3)}


def test_the_effective_weights_are_the_exact_mix_of_the_price_and_the_agent_s_own():
    assert standing.thaw_weights((1, 1), (3, 1), Fraction(1, 2)) == (2, 1)
    assert standing.thaw_weights((1, 1), (3, 1), 0) == (1, 1)
    assert standing.thaw_weights((1, 1), (3, 1), 1) == (3, 1)
    assert standing.thaw_weights((1, 2), (1, 4), Fraction(1, 3)) == (1, Fraction(8, 3))
    assert all(type(w) is Fraction for w in standing.thaw_weights((1, 1), (3, 1), Fraction(1, 2)))


def test_the_price_alone_ranks_every_agent_alike_and_the_private_weights_alone_rank_them_apart():
    endowment, goods, price, weights = swap_instance()
    alike = standing.thaw_rankings(goods, price, weights, 0)
    assert alike == {"A": ("y", "x"), "B": ("y", "x")}  # equal scores 6 and 6: the listing order breaks the tie
    apart = standing.thaw_rankings(goods, price, weights, 1)
    assert apart == {"A": ("x", "y"), "B": ("y", "x")}
    assert apart == {a: rival.rank_by_weights(goods, w) for a, w in weights.items()}
    assert standing.thaw_rankings(goods, price, weights, Fraction(1, 2)) == apart


def test_the_swap_at_equal_worth_needs_some_divergence_and_no_more():
    endowment, goods, price, weights = swap_instance()
    assert standing.thaw_outcome(endowment, goods, price, weights, 0) == endowment
    for lam in (Fraction(1, 1000), Fraction(1, 2), 1):
        assert standing.thaw_outcome(endowment, goods, price, weights, lam) == {"A": "x", "B": "y"}
    assert standing.thaw_breakpoints(goods, price, weights) == ()
    assert standing.thaw_threshold(endowment, goods, price, weights) == standing.Threshold(Fraction(0), False)
    pieces = standing.thaw_pieces(endowment, goods, price, weights)
    assert [(p.low, p.high, p.movers) for p in pieces] == [
        (0, 0, ()), (0, 1, ("A", "B")), (1, 1, ("A", "B"))]
    assert not standing.thaw_reverses(endowment, goods, price, weights)


def gap_instance(listing=("G1", "G2")):
    """Goods worth 4 and 3 at the price (1, 1): B holds the dearer one and A the cheaper. A wants the dearer one
    at every setting; B wants the cheaper one once the dial passes 1/3 (derived by hand in the tests below)."""
    vectors = {"G1": (3, 1), "G2": (1, 2)}
    return ({"A": "G2", "B": "G1"}, {g: vectors[g] for g in listing}, (1, 1), {"A": (3, 1), "B": (1, 4)})


def test_a_gap_in_worth_is_crossed_at_an_exact_rational_point_and_a_tie_there_is_broken_by_the_listing():
    endowment, goods, price, weights = gap_instance()
    # at the price G1 is worth 4 and G2 is worth 3; B's own weights (1, 4) close the gap of 1 at t = 1/3 exactly
    # (B's margin of G2 over G1 is -1 + 3t), while A's margin of G1 over G2 is 1 + 4t and never closes
    assert standing.thaw_breakpoints(goods, price, weights) == (Fraction(1, 3),)
    below, above = Fraction(1, 4), Fraction(1, 2)
    assert standing.thaw_outcome(endowment, goods, price, weights, below) == endowment
    assert standing.thaw_outcome(endowment, goods, price, weights, above) == {"A": "G1", "B": "G2"}
    # at 1/3 B scores the two goods equally; the one listed first is ranked first
    assert standing.thaw_outcome(endowment, goods, price, weights, Fraction(1, 3)) == endowment
    assert standing.thaw_threshold(endowment, goods, price, weights) == standing.Threshold(Fraction(1, 3), False)
    flipped = gap_instance(("G2", "G1"))
    assert standing.thaw_outcome(*flipped[:1], flipped[1], flipped[2], flipped[3], Fraction(1, 3)) == {
        "A": "G1", "B": "G2"}
    assert standing.thaw_threshold(*flipped) == standing.Threshold(Fraction(1, 3), True)


def reversal_instance():
    """A wants B's dearer good until the dial passes 1/2, B wants A's cheaper good once it passes 1/3: they trade
    on the stretch between and not outside it (derived by hand: A's margin is 1 - 2t, B's is -1 + 3t)."""
    vectors = {"G1": (3, 1), "G2": (1, 2)}
    return {"A": "G2", "B": "G1"}, vectors, (1, 1), {"A": (1, 3), "B": (1, 4)}


def test_the_dial_is_not_a_one_way_door_trade_opens_and_closes():
    endowment, goods, price, weights = reversal_instance()
    assert standing.thaw_breakpoints(goods, price, weights) == (Fraction(1, 3), Fraction(1, 2))
    pieces = standing.thaw_pieces(endowment, goods, price, weights)
    assert [(p.low, p.high, bool(p.movers)) for p in pieces] == [
        (0, 0, False), (0, Fraction(1, 3), False), (Fraction(1, 3), Fraction(1, 3), False),
        (Fraction(1, 3), Fraction(1, 2), True), (Fraction(1, 2), Fraction(1, 2), True),
        (Fraction(1, 2), 1, False), (1, 1, False)]
    assert standing.thaw_threshold(endowment, goods, price, weights) == standing.Threshold(Fraction(1, 3), False)
    assert standing.thaw_reverses(endowment, goods, price, weights)


def test_the_dial_reverses_in_a_real_share_of_random_instances_and_never_in_none():
    reversing = trading = 0
    for endowment, goods, price, weights in random_instances(seed=77, count=300, ns=(3, 4), ds=(2, 3)):
        pieces = standing.thaw_pieces(endowment, goods, price, weights)
        if any(p.movers for p in pieces):
            trading += 1
            reversing += standing.thaw_reverses(endowment, goods, price, weights)
    assert (trading, reversing) == (95, 26)  # of 300 seeded instances, which trade at some setting, which reverse


def random_instances(seed, count, ns, ds, high=6):
    rng = random.Random(seed)
    for _ in range(count):
        n, d = rng.choice(ns), rng.choice(ds)
        names = [f"g{k}" for k in range(n)]
        goods = {g: tuple(rng.randint(0, high) for _ in range(d)) for g in names}
        price = tuple(rng.randint(1, high) for _ in range(d))
        weights = {a: tuple(rng.randint(1, high) for _ in range(d)) for a in range(n)}
        held = names[:]
        rng.shuffle(held)
        yield dict(zip(range(n), held)), goods, price, weights


def test_at_the_price_alone_nobody_trades_whatever_the_start_and_the_weights():
    count = 0
    for endowment, goods, price, weights in random_instances(seed=11, count=250, ns=(2, 3, 4, 5), ds=(1, 2, 3)):
        count += 1
        assert standing.thaw_outcome(endowment, goods, price, weights, 0) == endowment
        assert standing.thaw_pieces(endowment, goods, price, weights)[0].movers == ()
    assert count == 250


def test_with_one_coordinate_nobody_ever_trades_the_one_dimensional_price_is_all_there_is():
    for endowment, goods, price, weights in random_instances(seed=12, count=250, ns=(2, 3, 4, 5), ds=(1,)):
        assert all(p.movers == () for p in standing.thaw_pieces(endowment, goods, price, weights))
        assert standing.thaw_threshold(endowment, goods, price, weights) is None


def test_weights_that_are_positive_multiples_of_the_price_never_trade_at_any_setting():
    rng = random.Random(13)
    for endowment, goods, price, _weights in random_instances(seed=13, count=200, ns=(3, 4, 5), ds=(2, 3)):
        stretched = {}
        for agent in endowment:
            factor = Fraction(rng.randint(1, 9), rng.randint(1, 4))  # one strength per agent, along the price
            stretched[agent] = tuple(factor * p for p in price)
        assert all(p.movers == () for p in standing.thaw_pieces(endowment, goods, price, stretched))


def test_weights_parallel_to_each_other_but_not_to_the_price_trade_in_between_and_not_at_the_ends():
    # The price is (1, 2); both agents' own weights point along (1, 1), B's three times as strong. At 0 they rank
    # alike (the price), at 1 they rank alike (the same direction), yet between the effective weights point in
    # different directions: A's is (1, 2 - t) and B's is (1 + 2t, 2 + t). With G = (1, 6) and H = (8, 1) the
    # margin of G over H is 3 - 5t for A and 3 - 9t for B (by hand), so A (holding H) wants G while t < 3/5
    # and B (holding G) wants H once t > 1/3: they swap on the stretch between.
    vectors = {"G": (1, 6), "H": (8, 1)}
    endowment, price, weights = {"A": "H", "B": "G"}, (1, 2), {"A": (1, 1), "B": (3, 3)}
    assert standing.thaw_breakpoints(vectors, price, weights) == (Fraction(1, 3), Fraction(3, 5))
    pieces = standing.thaw_pieces(endowment, vectors, price, weights)
    assert [(p.low, p.high, bool(p.movers)) for p in pieces] == [
        (0, 0, False), (0, Fraction(1, 3), False), (Fraction(1, 3), Fraction(1, 3), False),
        (Fraction(1, 3), Fraction(3, 5), True), (Fraction(3, 5), Fraction(3, 5), True),
        (Fraction(3, 5), 1, False), (1, 1, False)]
    assert pieces[3].allocation == {"A": "G", "B": "H"}


def test_between_two_breakpoints_nothing_changes_and_the_pieces_agree_with_direct_evaluation():
    checked = 0
    for endowment, goods, price, weights in random_instances(seed=21, count=70, ns=(3, 4, 5), ds=(2, 3)):
        pieces = standing.thaw_pieces(endowment, goods, price, weights)
        bounds = [p.low for p in pieces if p.low == p.high]
        assert bounds == sorted(set(bounds)) and bounds[0] == 0 and bounds[-1] == 1
        assert len(pieces) == 2 * len(bounds) - 1
        for k in range(1, 48):
            lam = Fraction(k, 48)
            piece = next(p for p in pieces if p.low < lam < p.high or p.low == p.high == lam)
            assert standing.thaw_outcome(endowment, goods, price, weights, lam) == piece.allocation
            checked += 1
        for piece in pieces:
            assert piece.movers == rival.movers(endowment, piece.allocation)
    assert checked == 70 * 47


def test_the_threshold_is_the_least_setting_at_which_anyone_moves():
    firsts = 0
    for endowment, goods, price, weights in random_instances(seed=22, count=120, ns=(3, 4), ds=(2, 3)):
        found = standing.thaw_threshold(endowment, goods, price, weights)
        for k in range(0, 61):
            lam = Fraction(k, 60)
            moves = bool(rival.movers(endowment, standing.thaw_outcome(endowment, goods, price, weights, lam)))
            if found is None or lam < found.lam:
                assert not moves, (lam, found)
            if found is not None and lam == found.lam and found.attained:
                assert moves
        firsts += found is not None
    assert firsts == 41  # of 120 seeded instances, those in which anyone ever moves


def test_scaling_the_price_and_every_weight_together_changes_nothing():
    for endowment, goods, price, weights in random_instances(seed=23, count=60, ns=(3, 4), ds=(2, 3)):
        scaled_price = tuple(7 * p for p in price)
        scaled_weights = {a: tuple(Fraction(7, 1) * w for w in ws) for a, ws in weights.items()}
        assert standing.thaw_pieces(endowment, goods, price, weights) == standing.thaw_pieces(
            endowment, goods, scaled_price, scaled_weights)


def test_listing_the_agents_in_another_order_moves_the_same_agents_at_every_setting():
    for endowment, goods, price, weights in random_instances(seed=24, count=60, ns=(3, 4, 5), ds=(2, 3)):
        reversed_endowment = dict(reversed(list(endowment.items())))
        before = standing.thaw_pieces(endowment, goods, price, weights)
        after = standing.thaw_pieces(reversed_endowment, goods, price, weights)
        assert [(p.low, p.high, frozenset(p.movers)) for p in before] == [
            (p.low, p.high, frozenset(p.movers)) for p in after]


def test_the_thaw_refuses_what_it_cannot_mean():
    endowment, goods, price, weights = swap_instance()
    with pytest.raises(TypeError):
        standing.thaw_outcome(endowment, goods, price, weights, 0.5)
    for lam in (Fraction(3, 2), Fraction(-1, 2), 2, -1):
        with pytest.raises(ValueError, match="between 0 and 1"):
            standing.thaw_outcome(endowment, goods, price, weights, lam)
    for bad_price in ((0, 1), (1, -1), (1,), (1, 1, 1)):
        with pytest.raises(ValueError):
            standing.thaw_pieces(endowment, goods, bad_price, weights)
    with pytest.raises(TypeError):
        standing.thaw_pieces(endowment, goods, (1.0, 1), weights)
    with pytest.raises(ValueError):
        standing.thaw_pieces(endowment, goods, price, {"A": (3, 1), "B": (0, 3)})
    with pytest.raises(ValueError):
        standing.thaw_pieces(endowment, goods, price, {"A": (3, 1)})
    with pytest.raises(ValueError):
        standing.thaw_pieces(endowment, goods, price, {"A": (3, 1), "B": (1, 3), "C": (1, 1)})
    with pytest.raises(ValueError, match="goods"):
        standing.thaw_pieces({"A": "y", "B": "z"}, goods, price, weights)
    with pytest.raises(ValueError):
        standing.thaw_pieces(endowment, {"y": (1, 5), "x": (5,)}, price, weights)
    with pytest.raises(TypeError):
        standing.thaw_pieces(endowment, {"y": (1, 5), "x": (5, 1.5)}, price, weights)


# --- the counts ------------------------------------------------------------------------------------------------


def test_the_survey_counts_ladders_loops_and_wealth_starts_over_every_profile():
    assert standing.survey(1) == {"profiles": 1, "ladders": 1, "loops": 0, "wealth_starts": 1, "wealth_frozen": 1}
    assert standing.survey(2) == {"profiles": 4, "ladders": 3, "loops": 1, "wealth_starts": 8, "wealth_frozen": 8}
    assert standing.survey(3) == {"profiles": 216, "ladders": 98, "loops": 118, "wealth_starts": 1296,
                                  "wealth_frozen": 1296}
    assert (standing.survey(3)["ladders"], standing.survey(3)["loops"]) == (rival.survey(3)["still"],
                                                                         rival.survey(3)["trade"])
    for n in (0, 4):
        with pytest.raises(ValueError):
            standing.survey(n)


def test_the_dial_survey_counts_what_the_independent_generator_above_counts():
    assert standing.dial_survey() == {"instances": 300, "trade": 95, "reverse": 26}
    assert standing.dial_survey(0) == {"instances": 0, "trade": 0, "reverse": 0}
    trading = reversing = 0
    for endowment, goods, price, weights in random_instances(seed=77, count=300, ns=(3, 4), ds=(2, 3)):
        if any(p.movers for p in standing.thaw_pieces(endowment, goods, price, weights)):
            trading += 1
            reversing += standing.thaw_reverses(endowment, goods, price, weights)
    assert (trading, reversing) == (95, 26)


# --- the same answers whatever the string hash seed ------------------------------------------------------------------

SCRIPT = """
import itertools
from fractions import Fraction
from hyperoptimization import standing
goods = ['g%d' % i for i in range(4)]
agents = ['a%d' % i for i in range(4)]
out = []
for k, perm in enumerate(itertools.islice(itertools.permutations(goods), 0, 24, 5)):
    held = dict(zip(agents, perm))
    prefs = {a: tuple(goods[(i + j) % 4] for j in range(4)) for i, a in enumerate(agents)}
    out.append((standing.envy_graph(held, prefs), standing.envy_loop(held, prefs), standing.pecking_order(held, prefs)))
    out.append(standing.wealth_start({a: Fraction(i + 1) for i, a in enumerate(agents)}, prefs))
vectors = {'G1': (3, 1), 'G2': (1, 2), 'G3': (2, 2), 'G4': (0, 3)}
held = {'p': 'G2', 'q': 'G1', 'r': 'G4', 's': 'G3'}
weights = {'p': (3, 1), 'q': (1, 4), 'r': (2, 5), 's': (5, 2)}
out.append(standing.thaw_pieces(held, vectors, (1, 1), weights))
print(repr(out))
"""


def test_the_answers_do_not_depend_on_the_string_hash_seed():
    outputs = set()
    for seed in ("0", "1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT / "src")}
        done = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True,
                              encoding="utf-8", timeout=120, check=True)
        outputs.add(done.stdout)
    assert len(outputs) == 1
