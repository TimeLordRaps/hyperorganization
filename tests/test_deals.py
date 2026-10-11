"""A deal is a vector, not a price: the freedom it allows, the freedom that freedom enables, and where
it coheres with the system's lawful existence.

An OPENING takes a law `before` to a law `after` that permits more. It is valued by

    allowed      the permissions it opens, counted along each dimension of freedom (by default the
                 action; a mapping can group actions);
    enabled      the states it makes reachable that were not, and the permissions the law grants there:
                 the degree to which those freedoms allow others;
    coherence    four yes-or-no dimensions: EXISTS (the system survives for ever from where it is),
                 CORRIGIBLE (and stays where an overseer can bring it home), LAWFUL (nothing is permitted,
                 reachable or not, that the greatest corrigible law forbids), MONOTONE (nobody loses a
                 permission they had).

What follows is checked here, mostly by brute force over every law of small games:

    closure        a freedom must come with the freedoms it enables: a door opened onto a room that permits
                   nothing is not coherent (the system is stuck there);
    no trade-off   among coherent openings the greatest corrigible law dominates all of them in every
                   coordinate, so within lawful existence there is nothing to price: the deal-language
                   form of the unique greatest law;
    comprehension  a coalition that understands the material of an opening can certify it exactly;
    no price       where options DO compete (a menu, a capacity), a price is a linear weighting, and a
                   linear weighting can never select an efficient option that sits strictly inside the
                   convex hull of others: the balanced deal is invisible to every price;
    no currency    coherence cannot be bought: for any finite price on lawfulness a game exists in which
                   the lawless opening outscores the lawful one.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import pytest
from randomgames import exit_times, reach_within

from hyperoptimization import comprehension, deals, knowledge, viability
from hyperoptimization.ground import OMEGA
from hyperoptimization.viability import Game, random_game


def hamlets() -> Game:
    return Game.from_table(nature=("calm", "storm"),
                           table={"a": {"hold": ("a", "b")}, "b": {"hold": ("b", "a")}})


def corridor() -> Game:
    return Game.from_table(nature=("n",), table={
        "h": {"stay": ("h",), "go": ("d",)}, "d": {"on": ("r",)}, "r": {"stay": ("r",)}})


def region(game, corrigible):
    return game.safe & viability.reach_home(game) if corrigible else game.safe


def top(game, corrigible=True):
    """The greatest law by value iteration on integers: no code shared with the descent of sets."""
    seen = region(game, corrigible)
    inside = {s for s, w in exit_times(game, seen).items() if w is OMEGA}
    return {s: frozenset(a for a in game.allowed(s) if all(game.target(s, a, b) in inside for b in game.nature))
            for s in sorted(inside)}


def survives(game, law, start, corrigible):
    reg = region(game, corrigible)
    seen = reach_within(game, law, start, reg)
    return all(s in reg and law.get(s) for s in seen)


def below(small, large):
    return all(set(small.get(s, ())) <= set(large.get(s, ())) for s in set(small) | set(large))


def sub_laws(game, rng, density=0.5):
    return {s: frozenset(a for a in sorted(game.allowed(s)) if rng.random() < density) for s in game.states}


# --- valuing an opening -----------------------------------------------------------------------------------------------


def test_the_hamlets_open_two_permissions_enable_nothing_and_cohere_along_every_dimension():
    game = hamlets()
    value = deals.opening_value(game, {}, {"a": {"hold"}, "b": {"hold"}}, ["a", "b"])
    assert value.allowed == (("hold", 2),) and value.new_states == 0 and value.new_permissions == 0
    assert dict(value.coherence) == {"exists": True, "corrigible": True, "lawful": True, "monotone": True}
    assert deals.admissible(value)


def test_one_hamlet_opening_alone_leaves_the_other_stuck_so_it_is_not_coherent():
    game = hamlets()
    value = deals.opening_value(game, {}, {"a": {"hold"}}, ["a", "b"])
    assert dict(value.coherence) == {"exists": False, "corrigible": False, "lawful": True, "monotone": True}
    assert not deals.admissible(value)


def test_a_freedom_must_come_with_the_freedoms_it_enables_or_the_system_is_stuck_in_the_doorway():
    game = corridor()
    base = {"h": {"stay"}}
    door_only = deals.opening_value(game, base, {"h": {"stay", "go"}}, ["h"])
    assert door_only.allowed == (("go", 1),) and door_only.new_states == 1 and door_only.new_permissions == 0
    assert dict(door_only.coherence)["exists"] is False  # at d nothing is permitted
    through = deals.opening_value(game, base, {"h": {"stay", "go"}, "d": {"on"}, "r": {"stay"}}, ["h"])
    assert through.allowed == (("go", 1), ("on", 1), ("stay", 1))
    assert through.new_states == 2 and through.new_permissions == 2 and deals.admissible(through)
    # and the smallest coherent opening that contains the door is exactly the door plus what is behind it
    keys = [("h", "go"), ("d", "on"), ("r", "stay")]
    coherent = []
    for chosen in itertools.chain.from_iterable(itertools.combinations(keys, r) for r in range(1, 4)):
        after = {"h": {"stay"} | {a for s, a in chosen if s == "h"}, "d": {a for s, a in chosen if s == "d"},
                 "r": {a for s, a in chosen if s == "r"}}
        if ("h", "go") in chosen and deals.admissible(deals.opening_value(game, base, after, ["h"])):
            coherent.append(set(chosen))
    assert coherent == [set(keys)]


def test_the_dimensions_of_freedom_can_be_grouped_and_the_start_decides_what_is_enabled():
    game = corridor()
    after = {"h": {"stay", "go"}, "d": {"on"}, "r": {"stay"}}
    grouped = deals.opening_value(game, {"h": {"stay"}}, after, ["h"], dimension={"go": "motion", "on": "motion", "stay": "rest"})
    assert grouped.allowed == (("motion", 2), ("rest", 1))
    from_the_room = deals.opening_value(game, {"h": {"stay"}}, after, ["r"])
    assert from_the_room.new_states == 0  # already there: nothing new to reach
    both = deals.opening_value(game, {"h": {"stay"}}, after, ["h", "d"])
    assert both.new_states == 1  # d is already reachable, only r is new


def test_the_coherence_flags_agree_with_independent_checks_on_random_games_and_laws():
    flags = {"exists": [0, 0], "corrigible": [0, 0], "lawful": [0, 0], "monotone": [0, 0]}
    for seed in range(400):
        rng = random.Random(11000 + seed)
        game = random_game(rng, n=4, home=True)
        greatest = top(game, True)
        before = sub_laws(game, rng, 0.3)
        extra = sub_laws(game, rng, 0.3)
        after = {s: frozenset(before[s]) | extra[s] for s in game.states} if rng.random() < 0.5 else sub_laws(game, rng, 0.5)
        starts = rng.sample(game.states, rng.randint(1, 2))
        value = dict(deals.opening_value(game, before, after, starts).coherence)
        expected = {
            "exists": all(survives(game, after, s, False) for s in starts),
            "corrigible": all(survives(game, after, s, True) for s in starts),
            "lawful": below(after, greatest),
            "monotone": below(before, after),
        }
        assert value == expected, (seed, value, expected)
        for name, holds in expected.items():
            flags[name][holds] += 1
    assert all(no > 10 and yes > 10 for no, yes in flags.values()), flags  # each flag both holds and fails in the sample


def test_the_value_of_an_opening_only_grows_as_more_is_opened():
    checked = 0
    for seed in range(200):
        rng = random.Random(12000 + seed)
        game = random_game(rng, n=4, home=False)
        before = sub_laws(game, rng, 0.25)
        middle = {s: frozenset(before[s]) | sub_laws(game, rng, 0.3)[s] for s in game.states}
        after = {s: frozenset(middle[s]) | sub_laws(game, rng, 0.3)[s] for s in game.states}
        starts = [game.states[0]]
        small = deals.opening_value(game, before, middle, starts)
        large = deals.opening_value(game, before, after, starts)
        assert dict(small.allowed).keys() <= dict(large.allowed).keys()
        for dimension, count in small.allowed:
            assert dict(large.allowed)[dimension] >= count
        assert small.new_states <= large.new_states and small.new_permissions <= large.new_permissions
        checked += 1
    assert checked == 200


def test_a_law_that_is_not_one_and_a_start_that_is_not_a_state_are_refused():
    game = hamlets()
    with pytest.raises(ValueError):
        deals.opening_value(game, {}, {"nowhere": {"hold"}}, ["a"])
    with pytest.raises(ValueError):
        deals.opening_value(game, {}, {"a": {"fly"}}, ["a"])
    with pytest.raises(ValueError):
        deals.opening_value(game, {}, {"a": {"hold"}}, [])
    with pytest.raises(ValueError):
        deals.opening_value(game, {}, {"a": {"hold"}}, ["nowhere"])
    with pytest.raises(TypeError):
        deals.opening_value(game, {}, {"a": "hold"}, ["a"])  # a string is not a set of actions
    with pytest.raises(TypeError):
        deals.opening_value(game, {}, {"a": {"hold"}}, "a")


# --- within lawful existence there is nothing to trade -----------------------------------------------------------------


def test_among_coherent_openings_the_greatest_corrigible_law_is_best_in_every_coordinate():
    games = informative = admissible_total = 0
    for seed in range(150):
        rng = random.Random(13000 + seed)
        game = random_game(rng, n=3, na=2, home=True)
        greatest = top(game, True)
        if not greatest:
            continue
        starts = [sorted(greatest)[0]]
        before = {s: frozenset(p) & greatest.get(s, frozenset()) for s, p in sub_laws(game, rng, 0.3).items()}
        best = deals.opening_value(game, before, greatest, starts)
        assert deals.admissible(best), seed
        values = []
        for after in viability.all_laws(game):
            value = deals.opening_value(game, before, after, starts)
            if deals.admissible(value):
                values.append(value)
                assert deals.at_least(best, value), (seed, after)
        front = deals.efficient(values)
        assert front and all(values[i] == best for i in front)  # the only efficient coherent opening is the best one
        admissible_total += len(values)
        informative += len(set(values)) > 1
        games += 1
    assert games > 40 and informative > 20 and admissible_total > 200


def test_an_unlawful_starting_law_leaves_no_coherent_opening_at_all():
    found = 0
    for seed in range(60):
        rng = random.Random(14000 + seed)
        game = random_game(rng, n=3, na=2, home=True)
        greatest = top(game, True)
        before = sub_laws(game, rng, 0.8)
        if below(before, greatest):
            continue
        starts = [game.states[0]]
        found += 1
        assert not any(deals.admissible(deals.opening_value(game, before, after, starts))
                       for after in viability.all_laws(game))  # lawful and monotone cannot both hold
    assert found > 20


# --- understanding the material is enough to certify -----------------------------------------------------------------


def test_a_coalition_that_understands_the_material_of_an_opening_can_certify_it_exactly():
    checked = certified_some = refused_some = 0
    for seed in range(300):
        rng = random.Random(15000 + seed)
        game = random_game(rng, n=4, home=True)
        graph = comprehension.reliance_of_game(game)
        targets = rng.sample(game.states, rng.randint(1, 2))
        material = comprehension.footprint(graph, targets).material
        extra = frozenset(s for s in game.states if rng.random() < 0.4)
        known = material | extra
        for corrigible in (False, True):
            true_law = viability.greatest_law(game, corrigible=corrigible)
            seen = knowledge.certified(game, known, corrigible=corrigible)
            for s in targets:
                for a in game.allowed(s):
                    permitted = a in true_law.get(s, ())
                    assert ((s, a) in seen) == permitted  # certifiable exactly when it is lawful
                    checked += 1
                    certified_some += permitted
                    refused_some += not permitted
    assert checked > 1000 and certified_some > 200 and refused_some > 200


def test_a_deal_is_admissible_only_if_it_is_coherent_and_the_people_deciding_understand_its_footprint():
    game = hamlets()
    both = {"a": {"hold"}, "b": {"hold"}}
    full = deals.admit(game, {}, both, ["a", "b"], {"A": ["a"], "B": ["b"]})
    assert full.coherent and full.makeable and full.admissible
    assert full.footprint.nodes == frozenset({"a", "b"}) and full.requirement.cover == 2
    assert full.requirement.required == 2 and full.requirement.necessary == frozenset({"A", "B"})
    blind = deals.admit(game, {}, both, ["a", "b"], {"A": ["a"]})
    assert blind.coherent and not blind.makeable and not blind.admissible
    assert blind.requirement.gaps == frozenset({"b"})
    lone = deals.admit(game, {}, {"a": {"hold"}}, ["a", "b"], {"A": ["a", "b"]})
    assert lone.makeable and not lone.coherent and not lone.admissible  # understood, and still not lawful
    with pytest.raises(ValueError):
        deals.admit(game, both, both, ["a"], {"A": ["a"]})  # nothing is opened


def test_the_hamlets_can_only_win_together_and_each_gains():
    game = hamlets()
    owner = {"a": "A", "b": "B"}
    both = {"a": {"hold"}, "b": {"hold"}}
    shares = deals.party_values(game, {}, both, ["a", "b"], owner)
    assert shares["A"].allowed == (("hold", 1),) and shares["B"].allowed == (("hold", 1),)
    assert deals.win_win(game, {}, both, ["a", "b"], owner)
    for single in ({"a": {"hold"}}, {"b": {"hold"}}):
        assert not deals.win_win(game, {}, single, ["a", "b"], owner)  # incoherent for the other, and a gain for one only
    gift = Game.from_table(nature=("calm", "storm"), table={"a": {"hold": ("a", "b")}, "b": {"hold": ("b", "b")}})
    assert deals.win_win(gift, {}, both, ["a", "b"], owner)  # still both gain a permission
    assert not deals.win_win(gift, {"b": {"hold"}}, both, ["a", "b"], owner)  # b already had it: only a gains


def test_each_party_is_shown_what_it_gains_and_nothing_it_does_not():
    ring = Game.from_table(nature=("n",), table={"a1": {"hold": ("a2",)}, "a2": {"hold": ("b",)}, "b": {"hold": ("a1",)}})
    owner = {"a1": "A", "a2": "A", "b": "B"}
    shares = deals.party_values(ring, {}, {"a1": {"hold"}, "a2": {"hold"}, "b": {"hold"}}, ["a1"], owner)
    assert shares["A"].allowed == (("hold", 2),) and shares["B"].allowed == (("hold", 1),)  # asymmetric
    assert (shares["A"].new_states, shares["A"].new_permissions) == (1, 1)  # a2 becomes reachable, and it is A's
    assert (shares["B"].new_states, shares["B"].new_permissions) == (1, 1)  # b becomes reachable, and it is B's
    assert shares["A"].coherence == shares["B"].coherence
    # the corridor: the room behind the door is all one party's, and only that party is shown it
    owner = {"h": "A", "d": "B", "r": "B"}
    shares = deals.party_values(corridor(), {"h": {"stay"}}, {"h": {"stay", "go"}, "d": {"on"}, "r": {"stay"}}, ["h"], owner)
    assert shares["A"].allowed == (("go", 1),) and (shares["A"].new_states, shares["A"].new_permissions) == (0, 0)
    assert shares["B"].allowed == (("on", 1), ("stay", 1)) and (shares["B"].new_states, shares["B"].new_permissions) == (2, 2)


def test_a_party_can_gain_by_what_is_enabled_alone_but_an_unlawful_deal_is_not_a_win_for_anyone():
    game = corridor()
    # b's room already permits staying; it only needs the door and the corridor to be opened by a
    owner = {"h": "A", "d": "A", "r": "B"}
    before, after = {"h": {"stay"}, "r": {"stay"}}, {"h": {"stay", "go"}, "d": {"on"}, "r": {"stay"}}
    shares = deals.party_values(game, before, after, ["h"], owner)
    assert shares["B"].allowed == () and (shares["B"].new_states, shares["B"].new_permissions) == (1, 1)
    assert deals.win_win(game, before, after, ["h"], owner)  # B gains the room it could not reach
    trap, hub_owner = deals.trap_game(2), {"hub": "A", "r1": "B", "r2": "B", "abyss": "B"}
    opening = deals.trap_opening(2)
    shares = deals.party_values(trap, {"hub": {"stay"}}, opening, ["hub"], hub_owner)
    assert shares["A"].allowed and shares["B"].allowed  # each party is offered something
    assert not deals.win_win(trap, {"hub": {"stay"}}, opening, ["hub"], hub_owner)  # both would gain, and it ends everything
    with pytest.raises(ValueError):
        deals.party_values(game, before, after, ["h"], {"h": "A"})


def test_the_footprint_of_a_deal_is_that_of_the_states_where_something_is_opened():
    sealed = Game.from_table(nature=("n",), table={
        "h": {"stay": ("h",), "go": ("d",)}, "d": {"on": ("r",)}, "r": {"stay": ("r",)}, "z": {"stay": ("z",)}})
    admission = deals.admit(sealed, {}, {"z": {"stay"}}, ["z"], {"P": ["z"], "Q": ["h", "d", "r"]})
    assert admission.footprint.nodes == frozenset({"z"}) and admission.makeable and admission.admissible
    assert admission.requirement.cover == 1 and admission.requirement.necessary == frozenset({"P"})  # only P understands z


# --- Pareto order and what a price can see ------------------------------------------------------------------------------


def test_dominance_is_a_strict_partial_order_and_the_efficient_points_are_the_maximal_ones():
    rng = random.Random(16000)
    for _ in range(300):
        points = [tuple(rng.randint(0, 4) for _ in range(3)) for _ in range(rng.randint(2, 7))]
        for a, b, c in itertools.product(points, repeat=3):
            assert not deals.dominates(a, a)
            if deals.dominates(a, b):
                assert not deals.dominates(b, a)
            if deals.dominates(a, b) and deals.dominates(b, c):
                assert deals.dominates(a, c)
        maximal = [i for i, p in enumerate(points) if not any(deals.dominates(q, p) for q in points)]
        assert deals.efficient(points) == maximal
    assert not deals.dominates((1, 2), (1, 2)) and deals.dominates((2, 2), (1, 2))
    assert not deals.dominates((2, 0), (0, 2)) and not deals.dominates((0, 2), (2, 0))  # incomparable
    assert deals.efficient([(1, 1), (1, 1), (0, 2)]) == [0, 1, 2]  # equal options do not beat one another
    assert deals.at_least((2, 2), (2, 2)) and not deals.at_least((1, 2), (1,)) and not deals.at_least((1,), (1, 2))


def interval_supported(points, i):
    """Two dimensions, exactly, by an interval of slopes: some weight (1, t) with t > 0 makes point i a maximizer."""
    low, high = Fraction(0), None
    for j, p in enumerate(points):
        dx, dy = points[i][0] - p[0], points[i][1] - p[1]
        if dy > 0:
            low = max(low, Fraction(-dx, dy))
        elif dy < 0:
            bound = Fraction(dx, -dy)
            high = bound if high is None else min(high, bound)
        elif dx < 0:
            return False
    return high is None or (high > 0 and low <= high)


def test_every_point_is_either_picked_by_some_positive_price_or_strictly_beaten_by_a_blend_never_both():
    both = 0
    for seed in range(120):
        rng = random.Random(17000 + seed)
        for dims in (2, 3, 4):
            points = list(dict.fromkeys(tuple(rng.randint(0, 6) for _ in range(dims))
                                        for _ in range(rng.randint(2, 7))))
            for i, point in enumerate(points):
                weights = deals.supporting_weights(points, i)
                blend = deals.dominating_combination(points, i)
                assert (weights is None) != (blend is None), (points, i)  # exactly one of the two
                if weights is not None:
                    assert all(w > 0 for w in weights)
                    scale = sum(w * x for w, x in zip(weights, point))
                    assert all(sum(w * x for w, x in zip(weights, q)) <= scale for q in points)
                    assert deals.supported(points, i)
                else:
                    assert len(blend) == len(points) and blend[i] == 0 and sum(blend) == 1 and all(x >= 0 for x in blend)
                    mixed = [sum(share * q[k] for share, q in zip(blend, points)) for k in range(dims)]
                    assert all(m >= x for m, x in zip(mixed, point)) and mixed != [Fraction(x) for x in point]
                    assert not deals.supported(points, i)
                    both += deals.efficient(points).count(i)
                if dims == 2:
                    assert deals.supported(points, i) == interval_supported(points, i)
    assert both > 20  # efficient points that no price ever picks really occur among random vectors


def test_a_price_is_a_linear_weighting_and_picks_what_it_maximizes():
    points = [(10, 0), (0, 10), (4, 4)]
    assert deals.select_by_price(points, (1, 1)) == [0, 1]
    assert deals.select_by_price(points, (3, 1)) == [0]
    assert deals.select_by_price(points, (1, 3)) == [1]
    assert deals.priced(points[2], (1, 1)) == 8
    for bad in ((0, 1), (-1, 1), (1,), (1, 1, 1)):
        with pytest.raises(ValueError):
            deals.select_by_price(points, bad)
    with pytest.raises(ValueError):
        deals.priced((1, 2), (1,))


def test_the_balanced_deal_is_efficient_and_coherent_and_no_price_will_ever_choose_it():
    game, base, openings, grouping = deals.three_doors()
    values = [deals.opening_value(game, base, after, ["hub"], dimension=grouping) for after in openings]
    assert all(deals.admissible(v) for v in values)
    assert [dict(v.allowed) for v in values] == [{"x": 11}, {"y": 11}, {"x": 5, "y": 4}]
    assert [(v.new_states, v.new_permissions) for v in values] == [(10, 10), (10, 10), (4, 8)]
    dims = deals.dimensions(*values)
    vectors = [deals.vector(v, dims) for v in values]
    assert deals.efficient(vectors) == [0, 1, 2]  # none dominates another: all three are on the menu
    assert deals.price_invisible(vectors) == [2]  # and the balanced one is below the blend of the other two
    blend = deals.dominating_combination(vectors, 2)
    assert blend is not None and blend[2] == 0 and sum(blend) == 1 and blend[0] >= Fraction(5, 11) and blend[1] >= Fraction(4, 11)
    mixed = [sum(w * v[k] for w, v in zip(blend, vectors)) for k in range(len(vectors[2]))]
    assert all(m >= x for m, x in zip(mixed, vectors[2])) and mixed != list(vectors[2])
    assert vectors[2][0] + vectors[2][1] == 9 < 11  # the balanced deal gives 9 in all, the others 11
    for wx in range(1, 40):
        for wy in range(1, 40):  # any x-price against any y-price, with every other coordinate priced as well
            assert 2 not in deals.select_by_price(vectors, (wx, wy) + (1,) * (len(vectors[0]) - 2))
    assert deals.supporting_weights(vectors, 2) is None and deals.supporting_weights(vectors, 0) is not None


def test_in_random_games_a_menu_of_competing_options_is_the_exception_and_a_hidden_option_rarer_still():
    # the honest base rate: coalitions of one or two people choosing among the freedoms they can certify
    menus = multiple = hidden = 0
    for seed in range(500):
        rng = random.Random(18000 + seed)
        game = random_game(rng, n=5, na=3, home=False)
        people = {f"p{i}": frozenset(s for s in game.states if rng.random() < 0.5) for i in range(5)}
        options = []
        for size in (1, 2):
            for group in itertools.combinations(sorted(people), size):
                known = frozenset().union(*(people[p] for p in group))
                law = viability.greatest_law_within(game, known, corrigible=False)
                options.append(tuple(sum(1 for permitted in law.values() if a in permitted) for a in game.actions))
        points = list(dict.fromkeys(options))
        if len(points) < 3:
            continue
        menus += 1
        front = deals.efficient(points)
        multiple += len(front) > 1
        lost = deals.price_invisible(points)
        hidden += bool(lost)
        assert set(lost) <= set(front)
        if lost:
            assert len(front) >= 3  # two efficient options cannot hide one another: a blend of one is itself
    assert menus > 150 and hidden <= multiple <= menus


# --- coherence is not a currency --------------------------------------------------------------------------------------------


def test_the_trap_has_more_of_everything_a_price_can_see_and_none_of_what_it_cannot():
    for rooms in (1, 3, 8):
        game = deals.trap_game(rooms)
        base = {"hub": {"stay"}}
        trap = deals.opening_value(game, base, deals.trap_opening(rooms), ["hub"])
        safe = deals.opening_value(game, base, {"hub": {"stay", "pause"}}, ["hub"])
        assert deals.admissible(safe) and not deals.admissible(trap)
        assert dict(trap.coherence) == {"exists": False, "corrigible": False, "lawful": False, "monotone": True}
        assert trap.new_states == rooms + 1 and trap.new_permissions == 3 * rooms  # the abyss is reachable too
        assert sum(count for _d, count in trap.allowed) == 1 + 3 * rooms
        assert safe.new_states == 0 and sum(count for _d, count in safe.allowed) == 1


def test_for_any_finite_price_on_lawfulness_a_game_exists_where_the_lawless_opening_wins():
    for price in (1, 5, 40, 300):
        smallest = None
        for rooms in range(1, 400):
            game = deals.trap_game(rooms)
            base = {"hub": {"stay"}}
            trap = deals.opening_value(game, base, deals.trap_opening(rooms), ["hub"])
            safe = deals.opening_value(game, base, {"hub": {"stay", "pause"}}, ["hub"])
            dims = deals.dimensions(trap, safe)
            weights = (1,) * len(dims) + (1, 1) + (price,) * 4  # freedom and enabling at 1, each coherence flag at `price`
            if deals.priced(deals.vector(trap, dims), weights) > deals.priced(deals.vector(safe, dims), weights):
                smallest = rooms
                break
        assert smallest == (3 * price - 1) // 7 + 1  # exactly: the trap scores 7k + 2 + price, the safe opening 1 + 4 price
        assert not deals.admissible(deals.opening_value(deals.trap_game(smallest), {"hub": {"stay"}},
                                                        deals.trap_opening(smallest), ["hub"]))
