"""Rival freedoms: an exchange of indivisible goods without money (top trading cycles).

A freedom that is not rival costs nobody anything to grant (`knowledge`, `deals`). A rival one, a slot or a seat
that one party holds at a time, can only be exchanged, and exchange without money needs a rule. The rule here is
Shapley and Scarf's housing market solved by Gale's top trading cycles: every remaining party points at the owner
of the good it likes best among those left; the cycles that form trade; repeat.

What is established by checking definitions on every strict preference profile of two and three parties, and on
seeded random profiles of four, five and six:
  * the outcome is individually rational (nobody ends below what they started with), Pareto efficient, in the
    strict core (no coalition can trade among itself to leave every member no worse and one better), and it is the
    ONLY allocation in the strict core (Roth and Postlewaite 1977);
  * truthful reporting is a best reply whatever the others report (Roth 1982), so private preferences can be
    asked for without a price; a popular rule (the Boston mechanism) is NOT, which shows the checker can fail;
  * each party can audit the outcome with its own preference and the public record of the cycles alone;
  * trade happens exactly when the starting allocation is not Pareto efficient; and the starting allocation, the
    endowment, decides everything else: with identical preferences nothing moves.
The theorems are cited; what is tested is that this implementation meets the definitions on the sampled markets.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import pytest

from hyperoptimization import rival

LARGER_MARKETS = [(4, 120), (5, 25), (6, 6)]  # (agents, seeded random markets) checked against every definition
RECORD_MARKETS = 60  # random markets of two to four agents whose every record of cycles is audited


def profiles(goods, agents):
    """Every profile of strict preferences: each agent ranks all goods."""
    rankings = list(itertools.permutations(goods))
    for choice in itertools.product(rankings, repeat=len(agents)):
        yield dict(zip(agents, choice))


def random_market(rng, n, shuffled=True):
    agents = list(range(n))
    goods = [f"g{i}" for i in range(n)]
    held = goods[:]
    if shuffled:
        rng.shuffle(held)
    endowment = dict(zip(agents, held))
    prefs = {a: tuple(rng.sample(goods, n)) for a in agents}
    return endowment, prefs


# --- the worked examples ----------------------------------------------------------------------------------------------


def test_two_agents_who_want_each_others_goods_swap_in_one_round():
    endowment = {1: "a", 2: "b", 3: "c"}
    prefs = {1: ("b", "c", "a"), 2: ("a", "c", "b"), 3: ("a", "b", "c")}
    assert rival.top_trading_cycles(endowment, prefs) == {1: "b", 2: "a", 3: "c"}
    assert rival.rounds(endowment, prefs) == [[(1, 2)], [(3,)]]  # agent 3 wants a, but it left in the first round


def test_a_ring_of_three_trades_in_one_cycle():
    endowment = {1: "a", 2: "b", 3: "c"}
    prefs = {1: ("b", "a", "c"), 2: ("c", "b", "a"), 3: ("a", "c", "b")}
    assert rival.top_trading_cycles(endowment, prefs) == {1: "b", 2: "c", 3: "a"}
    assert rival.rounds(endowment, prefs) == [[(1, 2, 3)]]


def test_agents_who_already_hold_their_favourite_keep_it():
    endowment = {1: "a", 2: "b"}
    prefs = {1: ("a", "b"), 2: ("b", "a")}
    assert rival.top_trading_cycles(endowment, prefs) == endowment
    assert rival.rounds(endowment, prefs) == [[(1,), (2,)]]


def test_a_later_round_can_reach_what_the_first_round_left():
    # round 1: 1 -> 2 -> 1 swap. Agent 3 points at agent 1's good, which leaves; then 3 and 4 trade.
    endowment = {1: "a", 2: "b", 3: "c", 4: "d"}
    prefs = {1: ("b", "a", "c", "d"), 2: ("a", "b", "c", "d"), 3: ("a", "d", "c", "b"), 4: ("c", "d", "a", "b")}
    assert rival.top_trading_cycles(endowment, prefs) == {1: "b", 2: "a", 3: "d", 4: "c"}
    assert rival.rounds(endowment, prefs) == [[(1, 2)], [(3, 4)]]


def test_a_cycle_is_listed_from_its_earliest_member_even_when_the_walk_reaches_it_from_outside():
    # agent 1 points into the cycle (2, 3) at agent 3, and the cycle is found by walking from 1
    endowment = {1: "a", 2: "b", 3: "c"}
    prefs = {1: ("c", "a", "b"), 2: ("c", "b", "a"), 3: ("b", "c", "a")}
    assert rival.rounds(endowment, prefs) == [[(2, 3)], [(1,)]]
    assert rival.top_trading_cycles(endowment, prefs) == {1: "a", 2: "c", 3: "b"}


def test_the_cycles_of_a_round_are_listed_by_their_earliest_member_not_by_the_order_they_were_found():
    # walking from agent 1 finds the cycle (3, 4) before the self-cycle (2,), but 2 comes first in the market
    endowment = {1: "a", 2: "b", 3: "c", 4: "d"}
    prefs = {1: ("c", "a", "b", "d"), 2: ("b", "a", "c", "d"), 3: ("d", "c", "a", "b"), 4: ("c", "d", "a", "b")}
    assert rival.rounds(endowment, prefs) == [[(2,), (3, 4)], [(1,)]]


def test_the_result_does_not_depend_on_the_order_the_agents_are_listed():
    rng = random.Random(7)
    for _ in range(60):
        endowment, prefs = random_market(rng, 5)
        expected = rival.top_trading_cycles(endowment, prefs)
        order = list(endowment)
        rng.shuffle(order)
        again = rival.top_trading_cycles({a: endowment[a] for a in order}, {a: prefs[a] for a in order})
        assert again == expected


# --- inputs are checked ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("endowment, prefs, message", [
    ({1: "a", 2: "a"}, {1: ("a",), 2: ("a",)}, "distinct"),
    ({1: "a", 2: "b"}, {1: ("a", "b")}, "exactly the agents"),
    ({1: "a", 2: "b"}, {1: ("a", "b"), 2: ("a", "b"), 3: ("a", "b")}, "exactly the agents"),
    ({1: "a", 2: "b"}, {1: ("a", "b"), 2: ("a",)}, "rank every good exactly once"),
    ({1: "a", 2: "b"}, {1: ("a", "b"), 2: ("a", "a")}, "rank every good exactly once"),
    ({1: "a", 2: "b"}, {1: ("a", "b"), 2: ("a", "z")}, "rank every good exactly once"),
    ({1: "a", 2: "b"}, {1: ("a", "b"), 2: ("a", "b", "a")}, "rank every good exactly once"),  # right goods, plus a repeat
])
def test_a_market_that_is_not_a_strict_housing_market_is_refused(endowment, prefs, message):
    with pytest.raises(ValueError, match=message):
        rival.top_trading_cycles(endowment, prefs)


@pytest.mark.parametrize("allocation", [{1: "a"}, {1: "a", 2: "a"}, {1: "a", 2: "z"}, {1: "a", 2: "b", 3: "c"},
                                        {1: "a", 3: "b"}])  # the last has the right goods and the wrong agents
def test_an_allocation_that_is_not_a_rearrangement_of_the_goods_is_refused(allocation):
    endowment = {1: "a", 2: "b"}
    prefs = {1: ("a", "b"), 2: ("a", "b")}
    for check in (rival.individually_rational, rival.in_core):
        with pytest.raises(ValueError, match="rearrangement"):
            check(endowment, prefs, allocation)
    with pytest.raises(ValueError, match="rearrangement"):
        rival.pareto_efficient(endowment, prefs, allocation)


# --- the definitions, on every market of two and three agents ---------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3])
def test_the_outcome_meets_every_definition_on_every_strict_profile(n):
    agents = list(range(n))
    goods = [f"g{i}" for i in range(n)]
    endowment = dict(zip(agents, goods))
    checked = 0
    for prefs in profiles(goods, agents):
        outcome = rival.top_trading_cycles(endowment, prefs)
        assert rival.individually_rational(endowment, prefs, outcome)
        assert rival.pareto_efficient(endowment, prefs, outcome)
        assert rival.in_core(endowment, prefs, outcome)
        assert rival.core_allocations(endowment, prefs) == [outcome]  # and it is the only one
        assert rival.profitable_misreports(endowment, prefs) == []
        checked += 1
    assert checked == {2: 4, 3: 216}[n]


@pytest.mark.parametrize("n, count", LARGER_MARKETS)
def test_the_outcome_meets_every_definition_on_random_larger_markets(n, count):
    rng = random.Random(1000 * n)
    for _ in range(count):
        endowment, prefs = random_market(rng, n)
        outcome = rival.top_trading_cycles(endowment, prefs)
        assert sorted(outcome.values()) == sorted(endowment.values())
        assert rival.individually_rational(endowment, prefs, outcome)
        assert rival.pareto_efficient(endowment, prefs, outcome)
        assert rival.in_core(endowment, prefs, outcome)
        if n <= 5:
            assert rival.core_allocations(endowment, prefs) == [outcome]
            assert rival.profitable_misreports(endowment, prefs) == []
        else:  # six agents: all 720 lies each is the same enumeration, for one agent in turn
            assert rival.profitable_misreports(endowment, prefs, agents=[rng.choice(list(endowment))]) == []


def test_large_markets_run_in_polynomial_time_and_stay_individually_rational():
    rng = random.Random(5)
    endowment, prefs = random_market(rng, 300)
    outcome = rival.top_trading_cycles(endowment, prefs)
    assert rival.individually_rational(endowment, prefs, outcome)
    assert sorted(outcome.values()) == sorted(endowment.values())
    flat = [cycle for one in rival.rounds(endowment, prefs) for cycle in one]
    assert sorted(a for cycle in flat for a in cycle) == sorted(endowment)  # every agent trades in exactly one cycle


# --- the checkers can fail --------------------------------------------------------------------------------------------


def test_each_checker_rejects_an_allocation_that_breaks_its_definition():
    endowment = {1: "a", 2: "b", 3: "c"}
    prefs = {1: ("b", "a", "c"), 2: ("a", "b", "c"), 3: ("a", "b", "c")}
    swap = {1: "b", 2: "a", 3: "c"}
    assert rival.top_trading_cycles(endowment, prefs) == swap
    # not individually rational: agent 1 is given its worst good, below the 'a' it started with
    worse = {1: "c", 2: "b", 3: "a"}
    assert not rival.individually_rational(endowment, prefs, worse)
    # individually rational and still not efficient: nobody trades although 1 and 2 would both gain
    assert rival.individually_rational(endowment, prefs, endowment)
    assert not rival.pareto_efficient(endowment, prefs, endowment)
    # blocked: 1 and 2 swap among themselves and leave both better off than in the allocation
    blocker = rival.blocking_coalition(endowment, prefs, endowment)
    assert blocker == ((1, 2), {1: "b", 2: "a"})
    assert not rival.in_core(endowment, prefs, endowment)
    # the explanation really blocks: weakly better for every member, strictly for one
    members, trade = blocker
    for a in members:
        assert prefs[a].index(trade[a]) <= prefs[a].index(endowment[a])
    assert any(prefs[a].index(trade[a]) < prefs[a].index(endowment[a]) for a in members)
    assert rival.blocking_coalition(endowment, prefs, swap) is None


def test_the_boston_mechanism_can_be_gamed_and_the_checker_finds_how():
    endowment = {1: "a", 2: "b", 3: "c"}
    prefs = {1: ("a", "b", "c"), 2: ("a", "b", "c"), 3: ("b", "a", "c")}
    assert rival.boston(endowment, prefs) == {1: "a", 3: "b", 2: "c"}  # agent 2 ends with its worst good by telling the truth
    lies = rival.profitable_misreports(endowment, prefs, mechanism=rival.boston)
    assert lies and all(a == 2 for a, _lie, _got in lies)
    agent, lie, got = lies[0]
    assert lie[0] == "b" and got == "b"  # applying for its second choice first gets it the second choice
    assert prefs[2].index(got) < prefs[2].index(rival.boston(endowment, prefs)[2])  # strictly better by its true ranking


def test_the_boston_mechanism_is_gameable_in_a_fixed_share_of_small_markets_and_ttc_in_none():
    agents, goods = [0, 1, 2], ["g0", "g1", "g2"]
    endowment = dict(zip(agents, goods))
    gameable = 0
    for prefs in profiles(goods, agents):
        gameable += bool(rival.profitable_misreports(endowment, prefs, mechanism=rival.boston))
    assert gameable == 36  # of the 216 profiles of three agents, counted with the checker above


# --- trade happens exactly when the endowment is not efficient -------------------------------------------------------


def test_trade_happens_exactly_when_the_starting_allocation_is_not_pareto_efficient():
    n = 3
    agents, goods = list(range(n)), [f"g{i}" for i in range(n)]
    endowment = dict(zip(agents, goods))
    traded = still = 0
    for prefs in profiles(goods, agents):
        outcome = rival.top_trading_cycles(endowment, prefs)
        moved = rival.movers(endowment, outcome)
        assert bool(moved) == (not rival.pareto_efficient(endowment, prefs, endowment))
        gain = rival.improvement(endowment, prefs, outcome)
        assert all(g >= 0 for g in gain.values())  # nobody is worse off
        assert (sum(gain.values()) > 0) == bool(moved)  # and someone is better off exactly when someone moved
        traded += bool(moved)
        still += not moved
    assert (traded, still) == (118, 98)  # of the 216 profiles of three agents, counted


# --- the starting allocation decides -----------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4, 5])
def test_with_identical_preferences_nothing_moves_whatever_the_endowment(n):
    goods = [f"g{i}" for i in range(n)]
    agents = list(range(n))
    prefs = {a: tuple(goods) for a in agents}
    for held in itertools.permutations(goods):
        endowment = dict(zip(agents, held))
        assert rival.top_trading_cycles(endowment, prefs) == endowment  # the owner of the best good keeps it


def test_the_welfare_of_each_agent_spans_the_whole_range_when_all_want_the_same():
    n = 4
    goods = [f"g{i}" for i in range(n)]
    prefs = {a: tuple(goods) for a in range(n)}
    spread = rival.endowment_spread(prefs)
    assert spread == {a: (0, n - 1) for a in range(n)}  # best rank 0 under one endowment, worst n-1 under another


def test_when_tastes_differ_enough_the_endowment_matters_less():
    n = 4
    goods = [f"g{i}" for i in range(n)]
    # each agent's favourite is a different good, and ranks the rest in order: a perfect matching of tastes
    prefs = {a: (goods[a],) + tuple(g for g in goods if g != goods[a]) for a in range(n)}
    spread = rival.endowment_spread(prefs)
    assert all(best == worst == 0 for best, worst in spread.values())  # everybody gets their favourite from any start


# --- the audit ---------------------------------------------------------------------------------------------------------


def test_each_agent_can_audit_the_outcome_with_its_own_preference_and_the_public_record():
    rng = random.Random(11)
    for _ in range(80):
        endowment, prefs = random_market(rng, rng.randint(2, 6))
        record = rival.rounds(endowment, prefs)
        for a in endowment:
            assert rival.audit(a, prefs[a], endowment, record)


def test_an_audit_fails_for_an_agent_given_less_than_its_favourite_among_what_was_left():
    endowment = {1: "a", 2: "b", 3: "c"}
    prefs = {1: ("b", "c", "a"), 2: ("a", "c", "b"), 3: ("a", "b", "c")}
    honest = rival.rounds(endowment, prefs)
    assert honest == [[(1, 2)], [(3,)]]
    forged = [[(1,)], [(2, 3)]]  # claims 1 kept 'a', then 2 and 3 swapped
    assert not rival.audit(1, prefs[1], endowment, forged)  # 1 preferred 'b', which was still on the market
    assert rival.audit(2, prefs[2], endowment, forged) and rival.audit(3, prefs[3], endowment, forged)
    # (agents 2 and 3 cannot tell: only the agent who was wronged can, and each can check for itself)
    # an agent that the record never mentions, or mentions twice, cannot be audited at all
    for agent, bad in ((1, [[(1,)], [(1, 2)]]), (3, [[(1, 2)]]), (3, [[(1, 2)], [(3,)], [(3,)]]),
                       (1, [[(1, 1, 2)], [(3,)]])):  # the last: listed twice inside one cycle
        with pytest.raises(ValueError, match="exactly once"):
            rival.audit(agent, prefs[agent], endowment, bad)


def test_a_record_that_every_agent_accepts_describes_the_ttc_outcome_and_the_honest_one_is_accepted():
    """Every record of cycles that covers each agent once, on random markets of up to four agents. Soundness: if
    all the audits pass, the allocation the record describes is the top trading cycles outcome (so a record cannot
    cheat everyone it does not wrong). Completeness is only for the honest record: the same allocation reached in a
    different order of rounds can fail an audit, because an agent who could have had its favourite is given less."""
    rng = random.Random(3)
    accepted = rejected = same_outcome_rejected = 0
    for _ in range(RECORD_MARKETS):
        endowment, prefs = random_market(rng, rng.randint(2, 4))
        expected = rival.top_trading_cycles(endowment, prefs)
        honest = rival.rounds(endowment, prefs)
        assert all(rival.audit(a, prefs[a], endowment, honest) for a in endowment)
        for record in rival.all_records(list(endowment)):
            allocation = rival.allocation_of(endowment, record)
            passed = all(rival.audit(a, prefs[a], endowment, record) for a in endowment)
            if passed:
                assert allocation == expected, (endowment, prefs, record)
                accepted += 1
            else:
                rejected += 1
                same_outcome_rejected += allocation == expected
    assert accepted >= 60 and rejected > 1000  # many records, few accepted
    assert same_outcome_rejected > 0  # and some reach the right allocation in a way an agent can see is not top-first


def test_the_honest_record_is_one_of_the_records_the_enumeration_lists():
    rng = random.Random(8)
    for _ in range(80):
        endowment, prefs = random_market(rng, rng.randint(1, 4))
        assert rival.rounds(endowment, prefs) in list(rival.all_records(list(endowment)))


def test_the_records_of_four_agents_are_the_ordered_partitions_of_their_cycles():
    counts = {n: sum(1 for _ in rival.all_records(list(range(n)))) for n in (1, 2, 3, 4)}
    assert counts == {1: 1, 2: 1 + 3, 3: 2 + 3 * 3 + 13, 4: 6 + 11 * 3 + 6 * 13 + 75}  # 1, 4, 24, 192


# --- rankings from private weights on the vector value of a good -------------------------------------------------------


def test_a_ranking_comes_from_an_agents_private_weights_on_the_coordinates_of_each_good():
    goods = {"x-rich": (5, 1), "y-rich": (1, 5), "plain": (2, 2)}
    assert rival.rank_by_weights(goods, (3, 1)) == ("x-rich", "y-rich", "plain")  # 16, 8, 8: the tie in listing order
    assert rival.rank_by_weights(goods, (1, 3)) == ("y-rich", "x-rich", "plain")  # 16, 8, 8
    assert rival.rank_by_weights(goods, (1, 1)) == ("x-rich", "y-rich", "plain")  # 6, 6, 4
    assert rival.rank_by_weights(goods, (Fraction(1, 2), 0)) == ("x-rich", "plain", "y-rich")  # fractions are exact
    assert rival.rank_by_weights({}, ()) == ()


@pytest.mark.parametrize("goods, weights, error", [
    ({"a": (1, 2)}, (1.5, 1), TypeError),
    ({"a": (1.0, 2)}, (1, 1), TypeError),
    ({"a": (1, 2)}, (True, 1), TypeError),
    ({"a": (1, 2)}, (1,), ValueError),
    ({"a": (1, 2, 3)}, (1, 1), ValueError),
])
def test_weights_and_coordinates_are_exact_and_must_agree_in_number(goods, weights, error):
    with pytest.raises(error):
        rival.rank_by_weights(goods, weights)


def test_a_swap_that_no_common_price_can_see_is_made_when_agents_weigh_the_coordinates_their_own_way():
    """Two slots of equal worth under a common price, (5, 1) and (1, 5) in freedom of x and of y. A holds the y-rich
    slot but weighs x three times as much as y; B holds the x-rich one and weighs y three times as much. By their own
    weights each doubles its worth by swapping. Under ANY common price the goods held are worth the same before and
    after (a swap only permutes them), so no price registers a gain, and a price-taker would not make the swap or
    would demand money. Top trading cycles makes it with no money, from the two private rankings alone."""
    vectors = {"x-rich": (5, 1), "y-rich": (1, 5)}
    weights = {"A": (3, 1), "B": (1, 3)}
    endowment = {"A": "y-rich", "B": "x-rich"}
    prefs = {a: rival.rank_by_weights(vectors, w) for a, w in weights.items()}
    assert prefs == {"A": ("x-rich", "y-rich"), "B": ("y-rich", "x-rich")}
    outcome = rival.top_trading_cycles(endowment, prefs)
    assert outcome == {"A": "x-rich", "B": "y-rich"} and rival.movers(endowment, outcome) == ("A", "B")

    def worth(agent, good):
        return sum(w * x for w, x in zip(weights[agent], vectors[good]))

    assert [worth(a, endowment[a]) for a in endowment] == [8, 8] and [worth(a, outcome[a]) for a in outcome] == [16, 16]
    for px, py in itertools.product(range(1, 7), repeat=2):  # any common price list
        value = {g: px * v[0] + py * v[1] for g, v in vectors.items()}
        assert sum(value[g] for g in endowment.values()) == sum(value[g] for g in outcome.values())
    assert 1 * 5 + 1 * 1 == 1 * 1 + 1 * 5  # and at the price (1, 1) the two goods are worth exactly the same


def test_private_weights_make_trade_possible_exactly_when_rankings_disagree_enough():
    """Over every pair of weights from a small grid and every way of giving two goods out, the exchange moves the goods
    exactly when each agent prefers the other's good, and then both strictly gain by their own weights."""
    vectors = {"x-rich": (5, 1), "y-rich": (1, 5)}
    grid = list(itertools.product(range(0, 4), repeat=2))
    moved = held = 0
    for wa, wb in itertools.product(grid, repeat=2):
        if wa == (0, 0) or wb == (0, 0):
            continue  # all-zero weights rank by listing order; that is a convention, not a taste
        for a_holds in vectors:
            endowment = {"A": a_holds, "B": next(g for g in vectors if g != a_holds)}
            prefs = {"A": rival.rank_by_weights(vectors, wa), "B": rival.rank_by_weights(vectors, wb)}
            outcome = rival.top_trading_cycles(endowment, prefs)
            wants_swap = all(prefs[a].index(endowment[b]) < prefs[a].index(endowment[a])
                             for a, b in (("A", "B"), ("B", "A")))
            assert bool(rival.movers(endowment, outcome)) == wants_swap
            moved += wants_swap
            held += not wants_swap
    assert moved > 0 and held > 0


# --- edges ------------------------------------------------------------------------------------------------------------


def test_an_empty_market_is_a_market_and_trades_nothing():
    assert rival.top_trading_cycles({}, {}) == {} and rival.rounds({}, {}) == []
    assert rival.endowment_spread({}) == {} and list(rival.all_records([])) == [[]]
    assert rival.core_allocations({}, {}) == [{}] and rival.movers({}, {}) == ()


def test_the_brute_force_checks_refuse_a_market_too_large_to_enumerate():
    agents = list(range(rival.MAX_BRUTE + 1))
    endowment = {a: f"g{a}" for a in agents}
    prefs = {a: tuple(endowment.values()) for a in agents}
    for call in (lambda: rival.pareto_efficient(endowment, prefs, endowment),
                 lambda: rival.blocking_coalition(endowment, prefs, endowment),
                 lambda: rival.core_allocations(endowment, prefs),
                 lambda: rival.profitable_misreports(endowment, prefs),
                 lambda: rival.endowment_spread(prefs)):
        with pytest.raises(ValueError, match="refused beyond"):
            call()
    assert rival.individually_rational(endowment, prefs, endowment)  # the cheap checks still run
    assert rival.top_trading_cycles(endowment, prefs) == endowment


# --- the survey the report and the ledger both quote --------------------------------------------------------------------


def test_the_survey_counts_every_strict_profile_of_small_markets_with_the_identity_endowment():
    assert rival.survey(1) == {"profiles": 1, "trade": 0, "still": 1, "gameable_ttc": 0, "gameable_boston": 0}
    assert rival.survey(2) == {"profiles": 4, "trade": 1, "still": 3, "gameable_ttc": 0, "gameable_boston": 0}
    assert rival.survey(3) == {"profiles": 216, "trade": 118, "still": 98, "gameable_ttc": 0, "gameable_boston": 36}
    with pytest.raises(ValueError, match="refused beyond 3"):
        rival.survey(4)
    with pytest.raises(ValueError, match="at least one"):
        rival.survey(0)


def test_the_best_place_an_agent_can_reach_over_all_starting_allocations_is_always_its_first_choice():
    """An agent that starts holding its favourite keeps it (individual rationality), so some starting allocation
    always gives it place 0. This is also why the starting value of the running minimum in `endowment_spread`
    cannot matter, and why the mutant that changes it is equivalent."""
    rng = random.Random(21)
    for _ in range(40):
        n = rng.randint(1, 5)
        goods = [f"g{i}" for i in range(n)]
        prefs = {a: tuple(rng.sample(goods, n)) for a in range(n)}
        assert all(best == 0 for best, _worst in rival.endowment_spread(prefs).values())
