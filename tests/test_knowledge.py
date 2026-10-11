"""What a coalition can lawfully permit is what its pooled knowledge supports itself on.

A participant sees part of the game. Whatever it cannot see it must treat as unsafe, since an unknown
state may be the one that ends everything: ignorance is pessimism. From that one rule four things
follow, and each is checked here on every subset of every sampled game:

    soundness   what a view certifies is inside the true kernel and the true greatest law;
    monotone    knowing more never certifies less;
    exactness   a view closed under the moves it can make certifies exactly;
    coinduction a state is certified exactly when the view holds a set that supports itself around it
                (the certificate of R6), so that is the knowledge a permission needs.

Then deals. Pool the knowledge of two parties that each hold a territory: whatever either certified
alone is still certified, sometimes strictly more is, and the strictly-more can only exist where
the territories touch. Two hamlets that each shelter in the other in a storm are the smallest case:
neither can certify its own safe move alone, together they certify both.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import pytest
from randomgames import exit_times

from hyperoptimization import knowledge, viability
from hyperoptimization.ground import OMEGA
from hyperoptimization.viability import Game, kernel, kernel_within, random_game


def subsets(items):
    items = sorted(items)
    for size in range(len(items) + 1):
        for chosen in itertools.combinations(items, size):
            yield frozenset(chosen)


def forward_closed(game, known):
    return all(game.target(s, a, b) in known for s in known for a in game.allowed(s) for b in game.nature)


def hamlets() -> Game:
    """Each hamlet is safe in calm weather alone; in a storm its people shelter in the other."""
    return Game.from_table(nature=("calm", "storm"),
                           table={"a": {"hold": ("a", "b")}, "b": {"hold": ("b", "a")}})


def refuge() -> Game:
    """b needs no one; a shelters in b in a storm."""
    return Game.from_table(nature=("calm", "storm"),
                           table={"a": {"hold": ("a", "b")}, "b": {"hold": ("b", "b")}})


def games(count=60, seed=31000):
    """(corrigible, game) for `count` reproducible random games on four states, each with and without
    corrigibility. Every game has a home set, so the corrigible region differs from the safe set."""
    for corrigible in (False, True):
        for i in range(count):
            yield corrigible, random_game(random.Random(seed + i), n=4, home=True)


# --- a view of part of the game ----------------------------------------------------------------------------------------


def test_a_view_that_sees_everything_certifies_the_whole_kernel_and_the_greatest_law():
    informative = 0
    for corrigible, game in games():
        everything = game.states
        assert kernel_within(game, everything, corrigible=corrigible) == kernel(game, corrigible=corrigible)
        assert viability.greatest_law_within(game, everything, corrigible=corrigible) == \
            viability.greatest_law(game, corrigible=corrigible)
        informative += 0 < len(kernel(game, corrigible=corrigible)) < len(game.states)
    assert informative > 20


def test_a_view_that_sees_nothing_certifies_nothing():
    for corrigible, game in games(20):
        assert kernel_within(game, (), corrigible=corrigible) == frozenset()
        assert viability.greatest_law_within(game, (), corrigible=corrigible) == {}


def test_ignorance_is_pessimism_unknown_states_are_unsafe_and_nothing_else_changes():
    # the same answer by a different road: build the game whose safe set is only what the view can see,
    # and rank it by value iteration on integers, which shares no code with the descent of sets
    checked = 0
    for corrigible, game in games(40):
        base = game.safe & viability.reach_home(game) if corrigible else game.safe
        for known in subsets(game.states):
            by_value = {s for s, w in exit_times(game, base & known).items() if w is OMEGA}
            assert kernel_within(game, known, corrigible=corrigible) == by_value, (sorted(known), corrigible)
            checked += 1
    assert checked > 1200


def test_what_a_view_certifies_is_sound_and_a_part_of_what_it_sees():
    checked = 0
    for corrigible, game in games():
        true_kernel = kernel(game, corrigible=corrigible)
        true_law = viability.greatest_law(game, corrigible=corrigible)
        for known in subsets(game.states):
            inside = kernel_within(game, known, corrigible=corrigible)
            assert inside <= true_kernel and inside <= known
            law = viability.greatest_law_within(game, known, corrigible=corrigible)
            assert set(law) == set(inside)
            for s, permitted in law.items():
                assert permitted <= true_law[s]
                assert permitted, "a certified state keeps at least one certified move"
                for a in permitted:  # and each certified move lands inside what is certified
                    assert all(game.target(s, a, b) in inside for b in game.nature)
            checked += 1
    assert checked == 2 * 60 * 2**4  # every subset of every sampled game


def test_knowing_more_never_certifies_less():
    pairs = 0
    for corrigible, game in games(30):
        everything = list(subsets(game.states))
        kernels = {known: kernel_within(game, known, corrigible=corrigible) for known in everything}
        for small, large in itertools.product(everything, repeat=2):
            if small <= large:
                assert kernels[small] <= kernels[large]
                pairs += 1
    assert pairs == 2 * 30 * 3**4  # each pair small <= large of subsets of four states: 3^4


def test_a_view_closed_under_the_moves_it_can_make_certifies_exactly():
    closed = strict = 0
    for corrigible, game in games():
        true_kernel = kernel(game, corrigible=corrigible)
        true_law = viability.greatest_law(game, corrigible=corrigible)
        for known in subsets(game.states):
            if forward_closed(game, known):
                closed += 1
                assert kernel_within(game, known, corrigible=corrigible) == true_kernel & known
                law = viability.greatest_law_within(game, known, corrigible=corrigible)
                assert law == {s: p for s, p in true_law.items() if s in known}
            else:
                strict += kernel_within(game, known, corrigible=corrigible) != true_kernel & known
    assert closed > 300 and strict > 100  # exactness is not vacuous, and an open view really loses


def test_a_state_is_certified_exactly_when_the_view_holds_a_set_that_supports_itself_around_it():
    checked = 0
    for corrigible, game in games(30):
        for known in subsets(game.states):
            by_certificate = frozenset().union(*(y for y in subsets(known)
                                                  if viability.maintains(game, y, corrigible=corrigible)))
            assert kernel_within(game, known, corrigible=corrigible) == by_certificate
            checked += 1
    assert checked == 2 * 30 * 2**4


def test_names_that_are_not_states_are_refused():
    game = hamlets()
    with pytest.raises(ValueError):
        kernel_within(game, ("a", "nowhere"))
    with pytest.raises(ValueError):
        viability.greatest_law_within(game, ["nowhere"])


# --- pooling ---------------------------------------------------------------------------------------------------------------------


def test_two_hamlets_neither_certifies_its_own_move_alone_and_together_they_certify_both():
    game = hamlets()
    assert kernel(game) == frozenset({"a", "b"})
    assert knowledge.certified(game, {"a"}) == frozenset()
    assert knowledge.certified(game, {"b"}) == frozenset()
    assert knowledge.pooled(game, {"A": {"a"}, "B": {"b"}}) == frozenset({("a", "hold"), ("b", "hold")})
    result = knowledge.surplus(game, {"A": {"a"}, "B": {"b"}}, {"a": "A", "b": "B"})
    assert result.alone == {"A": frozenset(), "B": frozenset()}
    assert result.gains == {"A": frozenset({("a", "hold")}), "B": frozenset({("b", "hold")})}
    assert result.mutual and result.together == frozenset({("a", "hold"), ("b", "hold")})
    assert knowledge.crossings(game, {"a": "A", "b": "B"}) == frozenset({("a", "hold", "b"), ("b", "hold", "a")})


def test_when_only_one_hamlet_needs_the_other_the_deal_is_a_gift_and_costs_the_giver_nothing():
    game = refuge()
    result = knowledge.surplus(game, {"A": {"a"}, "B": {"b"}}, {"a": "A", "b": "B"})
    assert result.alone["B"] == frozenset({("b", "hold")})  # b already stands on its own
    assert result.gains == {"A": frozenset({("a", "hold")}), "B": frozenset()}
    assert not result.mutual
    assert result.together == frozenset({("a", "hold"), ("b", "hold")})


def test_pooled_knowledge_certifies_at_least_what_each_part_does_and_sometimes_strictly_more():
    trials = strict = 0
    for corrigible, game in games(40):
        for left in subsets(game.states):
            for right in subsets(frozenset(game.states) - left):
                one = knowledge.certified(game, left, corrigible=corrigible)
                other = knowledge.certified(game, right, corrigible=corrigible)
                both = knowledge.pooled(game, {"x": left, "y": right}, corrigible=corrigible)
                assert one | other <= both
                trials += 1
                strict += one | other != both
    assert trials == 2 * 40 * 3**4 and strict > 300  # complementarity is common, not an artefact


def test_pooling_every_territory_certifies_the_whole_greatest_law():
    for index, (corrigible, game) in enumerate(games()):
        rng = random.Random(index)
        owner = {s: rng.choice("AB") for s in game.states}
        territory = {p: {s for s in game.states if owner[s] == p} for p in set(owner.values())}
        everything = knowledge.pooled(game, territory, corrigible=corrigible)
        law = viability.greatest_law(game, corrigible=corrigible)
        assert everything == frozenset((s, a) for s, permitted in law.items() for a in permitted)


def territorial(rng, leak):
    """Six states, three to each of two parties. Every move stays in its own territory, except that
    with probability `leak` its outcome is any state at all. Returns the game and who owns what."""
    states = [f"s{i}" for i in range(6)]
    owner = {s: "AB"[i // 3] for i, s in enumerate(states)}
    own = {p: [s for s in states if owner[s] == p] for p in "AB"}
    step = {}
    for s in states:
        for a in ("x", "y"):
            for b in ("n0", "n1"):
                step[(s, a, b)] = rng.choice(states) if rng.random() < leak else rng.choice(own[owner[s]])
    safe = [s for s in states if rng.random() < 0.8] or states[:1]
    return Game(states, ("x", "y"), ("n0", "n1"), step, safe=safe), owner


def test_without_a_crossing_between_territories_pooling_gains_nobody_anything():
    sealed = informative = leaky = gained = 0
    for corrigible in (False, True):
        for seed in range(150):
            for leak in (0, Fraction(1, 12)):
                game, owner = territorial(random.Random(seed), float(leak))
                territory = {p: {s for s in game.states if owner[s] == p} for p in "AB"}
                result = knowledge.surplus(game, territory, owner, corrigible=corrigible)
                if not knowledge.crossings(game, owner):
                    sealed += 1
                    assert all(not gain for gain in result.gains.values())  # nothing to gain without a border
                    assert result.together == result.alone["A"] | result.alone["B"]
                    informative += bool(result.together)
                else:
                    leaky += 1
                if any(result.gains.values()):  # the contrapositive: a gain needs a crossing
                    assert knowledge.crossings(game, owner)
                    gained += 1
    assert sealed > 250 and informative > 100  # sealed territories exist and are not all dead
    assert leaky > 100 and gained > 20  # and with a leak, gains appear


def test_no_party_ever_certifies_less_by_pooling():
    for index, (corrigible, game) in enumerate(games(40)):
        rng = random.Random(100 + index)
        owner = {s: rng.choice("AB") for s in game.states}
        if len(set(owner.values())) < 2:
            continue
        territory = {p: {s for s in game.states if owner[s] == p} for p in "AB"}
        result = knowledge.surplus(game, territory, owner, corrigible=corrigible)
        for party in "AB":
            assert result.alone[party] <= result.together
            held = {(s, a) for (s, a) in result.together if owner[s] == party}
            assert result.alone[party] | result.gains[party] == held


def test_gain_is_mutual_in_most_splits_one_sided_in_some_and_absent_in_others():
    counts = {"mutual": 0, "one-sided": 0, "none": 0}
    for corrigible, game in games(150):
        if not viability.greatest_law(game, corrigible=corrigible):
            continue
        for assignment in itertools.product("AB", repeat=len(game.states)):
            if len(set(assignment)) < 2 or assignment[0] != "A":
                continue
            owner = dict(zip(game.states, assignment))
            territory = {p: {s for s in game.states if owner[s] == p} for p in "AB"}
            gains = knowledge.surplus(game, territory, owner, corrigible=corrigible).gains
            held = sum(bool(g) for g in gains.values())
            counts["mutual" if held == 2 else "one-sided" if held == 1 else "none"] += 1
    assert all(counts.values()), counts
    assert counts["mutual"] > counts["one-sided"]  # these random games depend on each other more often than not


# --- the knowledge a permission needs ---------------------------------------------------------------------------------------


def test_the_knowledge_a_permission_needs_is_a_smallest_set_that_certifies_it():
    checked = 0
    for corrigible, game in games(30):
        law = viability.greatest_law(game, corrigible=corrigible)
        for s, permitted in law.items():
            for a in permitted:
                found = knowledge.minimal_knowledge(game, (s, a), corrigible=corrigible)
                certifying = [k for k in subsets(game.states)
                              if (s, a) in knowledge.certified(game, k, corrigible=corrigible)]
                smallest = min(len(k) for k in certifying)
                assert found is not None and len(found) == smallest and (s, a) in knowledge.certified(
                    game, found, corrigible=corrigible)
                assert set(knowledge.minimal_knowledge_all(game, (s, a), corrigible=corrigible)) == {
                    k for k in certifying if len(k) == smallest}
                checked += 1
    assert checked > 100


def test_a_permission_outside_the_greatest_law_can_never_be_certified():
    game = hamlets()
    assert knowledge.minimal_knowledge(game, ("a", "no-such-move")) is None
    assert knowledge.minimal_knowledge_all(game, ("a", "no-such-move")) == []
    unsafe = Game.from_table(nature=("n",), table={"ok": {"stay": ("ok",), "leap": ("pit",)},
                                                   "pit": {"stay": ("pit",)}}, safe={"ok"})
    assert knowledge.minimal_knowledge(unsafe, ("ok", "stay")) == frozenset({"ok"})
    assert knowledge.minimal_knowledge(unsafe, ("ok", "leap")) is None  # a freedom that ends existence is never certifiable


def test_the_smallest_sets_come_in_a_fixed_order_and_the_first_is_the_one_returned():
    ordered = checked = 0
    for corrigible, game in games(30):
        for s, permitted in viability.greatest_law(game, corrigible=corrigible).items():
            for a in permitted:
                every = knowledge.minimal_knowledge_all(game, (s, a), corrigible=corrigible)
                assert every == sorted(every, key=sorted)
                assert knowledge.minimal_knowledge(game, (s, a), corrigible=corrigible) == every[0]
                ordered += len(every) > 1
                checked += 1
    assert checked > 100 and ordered >= 1  # a tie between smallest sets does occur among random games, though rarely


def test_when_two_different_sets_certify_equally_well_both_are_listed_and_the_first_is_returned():
    # s goes to m, and from m either of two safe rooms will do: knowing one room is enough, either one
    game = Game.from_table(nature=("n",), table={
        "s": {"go": ("m",)}, "m": {"left": ("t1",), "right": ("t2",)},
        "t1": {"stay": ("t1",)}, "t2": {"stay": ("t2",)}})
    assert knowledge.minimal_knowledge_all(game, ("s", "go")) == [frozenset({"m", "s", "t1"}),
                                                                  frozenset({"m", "s", "t2"})]
    assert knowledge.minimal_knowledge(game, ("s", "go")) == frozenset({"m", "s", "t1"})
    assert knowledge.minimal_knowledge_all(game, ("m", "right")) == [frozenset({"m", "t2"})]


def test_a_single_party_cannot_have_a_mutual_gain():
    game = hamlets()
    alone = knowledge.surplus(game, {"A": {"a", "b"}}, {"a": "A", "b": "A"})
    assert alone.gains == {"A": frozenset()} and not alone.mutual  # nothing to pool with
    helped = knowledge.surplus(game, {"A": {"a"}, "C": {"b"}}, {"a": "A", "b": "A"})
    assert helped.gains["A"] == frozenset({("a", "hold"), ("b", "hold")}) and not helped.mutual  # one owner, helped by an outsider
    watcher = knowledge.surplus(game, {"A": {"a"}, "B": {"b"}, "C": {"a", "b"}}, {"a": "A", "b": "B"})
    assert watcher.gains["C"] == frozenset() and watcher.mutual  # an observer owns nothing and gains nothing
    assert watcher.together == frozenset({("a", "hold"), ("b", "hold")})


def test_the_hamlets_need_each_others_knowledge_for_each_permission():
    game = hamlets()
    assert knowledge.minimal_knowledge(game, ("a", "hold")) == frozenset({"a", "b"})
    assert knowledge.minimal_knowledge_all(game, ("b", "hold")) == [frozenset({"a", "b"})]


# --- inputs --------------------------------------------------------------------------------------------------------------------------------


def test_the_inputs_are_checked():
    game = hamlets()
    with pytest.raises(ValueError):
        knowledge.surplus(game, {"A": {"a"}, "B": {"b"}}, {"a": "A"})  # owner must cover every state
    with pytest.raises(ValueError):
        knowledge.surplus(game, {"A": {"a"}}, {"a": "A", "b": "B"})  # B has no knowledge entry
    with pytest.raises(ValueError):
        knowledge.surplus(game, {"A": {"a"}, "B": {"b"}}, {"a": "A", "b": "B", "c": "B"})  # c is not a state
    with pytest.raises(ValueError):
        knowledge.pooled(game, {"A": {"a"}}, parties=["Z"])
    with pytest.raises(ValueError):
        knowledge.minimal_knowledge(game, ("nowhere", "hold"))
    with pytest.raises(TypeError):
        knowledge.crossings(game, {"a": "A", "b": "B"}.items())  # a mapping, not its items
