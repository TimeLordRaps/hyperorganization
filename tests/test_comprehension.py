"""A decision is makeable only if its material and its impacts are understood, and that sets who may
decide and how many must.

The footprint of a decision is what it rests on (everything its targets rely on, to any depth) together
with what rests on it (everything that relies on its targets, to any depth). Participants understand
parts of the world. From that, with nothing assumed beyond the sets, these follow and are checked here
by brute force over every subset of people on small instances:

    cover       the fewest people who between them understand the whole footprint;
    robust      the fewest attendees such that WHOEVER attends understands it: m - w + 1, where w is the
                fewest people who understand the least-understood part (the bus factor);
    necessary   the people no decision can be made without are exactly those who alone understand
                something in the footprint;
    barrier     eligibility is a threshold on how much of the footprint a person understands; raising it
                shrinks the pool, raises the cover and can make the decision unmakeable, never the reverse;
    cores       a footprint is a union of whole cores: you cannot understand part of a circle of mutual
                reliance, so a decision that touches one needs a quorum that spans it.

And one tension with the quorum theory of `quorum` and `surprisal`: a smaller electorate is cheaper to
capture (Condorcet's jury theorem, in bits), so every seat the barrier removes is paid for in capture
resistance.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import pytest

from hyperoptimization import comprehension, entrenchment, family, quorum, surprisal, viability


def chain():
    """c relies on b relies on a; d relies on b; e stands alone."""
    return {"a": [], "b": ["a"], "c": ["b"], "d": ["b"], "e": []}


def ring():
    """x, y, z rely on one another; w relies on the ring; v is relied on by the ring."""
    return {"x": ["y", "v"], "y": ["z"], "z": ["x"], "w": ["x"], "v": []}


def random_graph(rng, n=None):
    n = n or rng.randint(2, 7)
    names = [f"n{i}" for i in range(n)]
    return {name: [other for other in names if other != name and rng.random() < 0.25] for name in names}


def random_knowledge(rng, names, people=None):
    people = people or rng.randint(1, 6)
    density = rng.choice([0.3, 0.5, 0.7])
    return {f"p{i}": [n for n in names if rng.random() < density] for i in range(people)}


def brute_closure(graph, start):
    seen, frontier = set(), list(graph[start])
    while frontier:
        n = frontier.pop()
        if n not in seen:
            seen.add(n)
            frontier.extend(graph[n])
    return seen


def covering(knowledge, nodes, group):
    held = set().union(set(), *(set(knowledge[p]) for p in group))
    return set(nodes) <= held


# --- the footprint ----------------------------------------------------------------------------------------------------


def test_the_reliance_closure_puts_a_node_in_its_own_closure_exactly_when_it_lies_on_a_cycle():
    closure = entrenchment.reliance_closure(chain())
    assert closure == {"a": frozenset(), "b": frozenset({"a"}), "c": frozenset({"a", "b"}),
                       "d": frozenset({"a", "b"}), "e": frozenset()}
    around = entrenchment.reliance_closure(ring())
    assert all(n in around[n] for n in "xyz") and "w" not in around["w"] and "v" not in around["v"]
    assert around["w"] == frozenset({"x", "y", "z", "v"})
    for seed in range(200):  # and it agrees with the cores
        graph = random_graph(random.Random(seed), n=6)
        on_a_cycle = {n for n, found in entrenchment.reliance_closure(graph).items() if n in found}
        assert on_a_cycle == {n for group in entrenchment.cores(graph) for n in group}


def test_the_footprint_of_a_node_is_what_it_rests_on_and_what_rests_on_it():
    fp = comprehension.footprint(chain(), ["b"])
    assert fp.material == frozenset({"b", "a"}) and fp.impact == frozenset({"b", "c", "d"})
    assert fp.nodes == frozenset({"a", "b", "c", "d"}) and fp.targets == frozenset({"b"})
    alone = comprehension.footprint(chain(), ["e"])
    assert alone.nodes == frozenset({"e"})  # nothing rests on it and it rests on nothing: a decision of one thing


def test_the_footprint_agrees_with_an_independent_closure_on_random_graphs():
    checked = 0
    for seed in range(300):
        rng = random.Random(seed)
        graph = random_graph(rng)
        reverse = {n: [m for m in graph if n in graph[m]] for n in graph}
        for target in graph:
            fp = comprehension.footprint(graph, [target])
            assert fp.material == {target} | brute_closure(graph, target)
            assert fp.impact == {target} | brute_closure(reverse, target)
            checked += 1
    assert checked > 1000


def test_the_footprint_of_several_targets_is_the_union_of_their_footprints_and_grows_with_them():
    for seed in range(200):
        rng = random.Random(1000 + seed)
        graph = random_graph(rng)
        names = list(graph)
        a, b = rng.sample(names, 2) if len(names) > 1 else (names[0], names[0])
        one = comprehension.footprint(graph, [a]).nodes
        two = comprehension.footprint(graph, [b]).nodes
        both = comprehension.footprint(graph, [a, b])
        assert both.nodes == one | two
        assert one <= both.nodes and two <= both.nodes


def test_you_cannot_understand_part_of_a_circle_a_footprint_holds_whole_cores():
    touching = 0
    for seed in range(400):
        rng = random.Random(2000 + seed)
        graph = random_graph(rng, n=rng.randint(3, 7))
        groups = entrenchment.cores(graph)
        for target in graph:
            nodes = comprehension.footprint(graph, [target]).nodes
            for group in groups:
                if set(group) & nodes:
                    assert set(group) <= nodes
                    touching += 1
    assert touching > 200 and any(len(g) > 1 for g in entrenchment.cores(ring()))
    fp = comprehension.footprint(ring(), ["z"])
    assert {"x", "y", "z"} <= fp.nodes and fp.nodes == frozenset({"x", "y", "z", "w", "v"})


def test_the_family_load_of_a_change_is_the_footprint_of_its_repository():
    graph = family.depends_on("formal")
    loads = comprehension.loads(graph)
    assert set(loads) == set(graph)
    for name, fp in loads.items():
        reverse = {n: [m for m in graph if n in graph[m]] for n in graph}
        assert fp.nodes == {name} | brute_closure(graph, name) | brute_closure(reverse, name)
    core = {"hyperethics", "hyperphysics", "metamathethicology"}
    for name in core:  # the circle of mutual reliance in the formal view: every member's footprint holds all three
        assert core <= loads[name].nodes and len(loads[name].nodes) == 8
    assert min(len(fp.nodes) for fp in loads.values()) == 1 and max(len(fp.nodes) for fp in loads.values()) == 8
    assert len(loads["hyperlogic"].nodes) == 8 and loads["hyperlogic"].impact == {"hyperlogic"}  # relies on all, nothing on it


def test_a_games_states_rely_on_their_successors_so_a_permissions_footprint_is_its_state_closure():
    game = viability.Game.from_table(
        nature=("n",), table={"s": {"go": ("m",)}, "m": {"left": ("t1",), "right": ("t2",)},
                              "t1": {"stay": ("t1",)}, "t2": {"stay": ("t2",)}})
    graph = comprehension.reliance_of_game(game)
    assert graph == {"s": frozenset({"m"}), "m": frozenset({"t1", "t2"}), "t1": frozenset({"t1"}),
                     "t2": frozenset({"t2"})}
    assert comprehension.footprint(graph, ["m"]).nodes == frozenset({"s", "m", "t1", "t2"})
    assert comprehension.footprint(graph, ["t1"]).nodes == frozenset({"s", "m", "t1"})  # t2 is not touched by t1
    storm = viability.Game.from_table(nature=("calm", "storm"),
                                      table={"a": {"hold": ("a", "b")}, "b": {"hold": ("b", "a")}})
    assert comprehension.reliance_of_game(storm) == {"a": frozenset({"a", "b"}), "b": frozenset({"a", "b"})}
    two_ways = viability.Game.from_table(nature=("n0", "n1"), table={"p": {"x": ("p", "q"), "y": ("q", "q")},
                                                                     "q": {"x": ("q", "q"), "y": ("q", "q")}})
    assert comprehension.reliance_of_game(two_ways)["p"] == frozenset({"p", "q"})  # every action, every nature choice


# --- who understands what ------------------------------------------------------------------------------------------------------


def test_coverage_is_the_share_of_the_footprint_a_person_understands():
    know = {"p": ["a", "b"], "q": ["b"], "r": []}
    nodes = ["a", "b", "c", "d"]
    assert comprehension.coverage(know, "p", nodes) == Fraction(1, 2)
    assert comprehension.coverage(know, "q", nodes) == Fraction(1, 4)
    assert comprehension.coverage(know, "r", nodes) == 0
    assert comprehension.coverage(know, "stranger", nodes) == 0  # someone who knows nothing yet


def test_witnesses_gaps_bus_factor_and_necessary_voices_on_a_worked_example():
    know = {"p": ["a", "b"], "q": ["b", "c"], "r": ["b"]}
    nodes = ["a", "b", "c", "d"]
    assert comprehension.witnesses(know, nodes) == {
        "a": frozenset({"p"}), "b": frozenset({"p", "q", "r"}), "c": frozenset({"q"}), "d": frozenset()}
    assert list(comprehension.witnesses(know, nodes)) == ["a", "b", "c", "d"]  # in a fixed order, not set order
    assert comprehension.gaps(know, nodes) == frozenset({"d"}) and comprehension.bus_factor(know, nodes) == 0
    assert comprehension.cover_number(know, nodes) is None and comprehension.robust_quorum(know, nodes) is None
    assert comprehension.covers(know, nodes) == []
    understood = ["a", "b", "c"]
    assert comprehension.gaps(know, understood) == frozenset()
    assert comprehension.bus_factor(know, understood) == 1
    assert comprehension.necessary_voices(know, understood) == frozenset({"p", "q"})  # the sole witnesses of a and c
    assert comprehension.covers(know, understood) == [("p", "q")]
    assert comprehension.cover_number(know, understood) == 2
    assert comprehension.robust_quorum(know, understood) == 3  # all three must come: any two could miss a or c


def test_an_empty_footprint_needs_no_one():
    know = {"p": ["a"], "q": []}
    assert comprehension.cover_number(know, []) == 0 and comprehension.robust_quorum(know, []) == 0
    assert comprehension.covers(know, []) == [()]
    assert comprehension.bus_factor(know, []) == 2 and comprehension.necessary_voices(know, []) == frozenset()
    assert comprehension.coverage(know, "q", []) == 1  # everyone understands all of nothing
    assert comprehension.eligible(know, [], 1) == ("p", "q")
    assert comprehension.max_barrier({}, []) == 0 and comprehension.max_barrier({}, ["a"]) is None  # nobody, nothing to understand
    req = comprehension.requirement(know, [], 1)
    assert req.makeable and req.required == req.balanced == 2 and req.robust == 0 and req.cover == 0


def test_witnesses_can_be_asked_of_a_pool_and_the_pool_must_be_people_who_have_knowledge():
    know = {"p": ["a", "b"], "q": ["b"], "r": ["a"]}
    assert comprehension.witnesses(know, ["a", "b"], pool=["q", "r"]) == {"a": frozenset({"r"}), "b": frozenset({"q"})}
    assert comprehension.gaps(know, ["a", "b"], pool=["q"]) == frozenset({"a"})
    assert comprehension.cover_number(know, ["a", "b"], pool=["q", "r"]) == 2
    with pytest.raises(ValueError):
        comprehension.witnesses(know, ["a"], pool=["ghost"])


# --- the identities, by brute force over every group of people --------------------------------------------------------


def test_the_cover_number_is_the_size_of_the_smallest_group_that_understands_everything():
    checked = unmakeable = 0
    for seed in range(1500):
        rng = random.Random(3000 + seed)
        names = [f"e{i}" for i in range(rng.randint(1, 6))]
        know = random_knowledge(rng, names)
        nodes = rng.sample(names, rng.randint(1, len(names)))
        groups = [g for r in range(len(know) + 1) for g in itertools.combinations(sorted(know), r)
                  if covering(know, nodes, g)]
        if not groups:
            assert comprehension.cover_number(know, nodes) is None and comprehension.covers(know, nodes) == []
            unmakeable += 1
            continue
        smallest = min(len(g) for g in groups)
        assert comprehension.cover_number(know, nodes) == smallest
        assert comprehension.covers(know, nodes) == sorted(g for g in groups if len(g) == smallest)
        checked += 1
    assert checked > 700 and unmakeable > 100


def test_the_robust_quorum_is_exactly_the_attendance_at_which_whoever_comes_understands():
    mixed = 0
    for seed in range(1500):
        rng = random.Random(4000 + seed)
        names = [f"e{i}" for i in range(rng.randint(1, 6))]
        know = random_knowledge(rng, names)
        nodes = rng.sample(names, rng.randint(1, len(names)))
        people = sorted(know)
        robust = comprehension.robust_quorum(know, nodes)
        cover = comprehension.cover_number(know, nodes)
        every_attendance = {q: all(covering(know, nodes, g) for g in itertools.combinations(people, q))
                            for q in range(len(people) + 1)}
        if robust is None:
            assert cover is None and not any(every_attendance.values())
            continue
        assert every_attendance[robust] and (robust == 0 or not every_attendance[robust - 1])
        assert all(every_attendance[q] for q in range(robust, len(people) + 1))  # and it stays true as more attend
        assert cover is not None and cover <= robust <= len(people)
        mixed += cover < robust
    assert mixed > 200  # the two really differ: a cover can be found, attendance cannot be relied on


def test_the_bus_factor_is_the_fewest_people_whose_loss_leaves_something_not_understood():
    for seed in range(1000):
        rng = random.Random(5000 + seed)
        names = [f"e{i}" for i in range(rng.randint(1, 5))]
        know = random_knowledge(rng, names)
        nodes = rng.sample(names, rng.randint(1, len(names)))
        people = sorted(know)
        losses = [len(lost) for r in range(len(people) + 1) for lost in itertools.combinations(people, r)
                  if not covering(know, nodes, [q for q in people if q not in lost])]
        assert losses  # losing everyone always leaves something not understood
        assert comprehension.bus_factor(know, nodes) == min(losses)


def test_the_people_no_decision_can_be_made_without_are_exactly_the_sole_witnesses():
    informative = 0
    for seed in range(1500):
        rng = random.Random(6000 + seed)
        names = [f"e{i}" for i in range(rng.randint(1, 6))]
        know = random_knowledge(rng, names)
        nodes = rng.sample(names, rng.randint(1, len(names)))
        groups = [set(g) for r in range(len(know) + 1) for g in itertools.combinations(sorted(know), r)
                  if covering(know, nodes, g)]
        if not groups:
            continue
        everywhere = set.intersection(*groups)  # in every group that understands the footprint
        assert comprehension.necessary_voices(know, nodes) == everywhere
        informative += bool(everywhere)
    assert informative > 200


def test_the_people_a_decision_needs_are_those_affected_when_only_they_know_their_own_stake():
    # the footprint includes what rests on the target; a person who alone understands their own place in it
    # cannot be left out, and stops being necessary the moment someone else understands that place too
    graph = {"law": [], "farm": ["law"], "mill": ["law"], "market": ["farm", "mill"]}
    fp = comprehension.footprint(graph, ["law"])
    assert fp.nodes == frozenset({"law", "farm", "mill", "market"})
    know = {"drafter": ["law"], "farmer": ["farm"], "miller": ["mill"], "trader": ["market"]}
    assert comprehension.necessary_voices(know, fp.nodes) == frozenset(know)
    know["trader"] = ["market", "farm", "mill"]
    assert comprehension.necessary_voices(know, fp.nodes) == frozenset({"drafter", "trader"})


# --- the barrier ----------------------------------------------------------------------------------------------------------------------


def test_eligibility_is_a_threshold_on_coverage_and_the_pool_shrinks_as_the_threshold_rises():
    know = {"p": ["a", "b", "c", "d"], "q": ["a", "b", "c"], "r": ["a", "b"], "s": ["a"], "t": []}
    nodes = ["a", "b", "c", "d"]
    assert comprehension.eligible(know, nodes) == ("p", "q", "r", "s", "t")
    assert comprehension.eligible(know, nodes, Fraction(1, 2)) == ("p", "q", "r")
    assert comprehension.eligible(know, nodes, Fraction(3, 4)) == ("p", "q")
    assert comprehension.eligible(know, nodes, 1) == ("p",)
    for lower, higher in zip((0, Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)), (Fraction(1, 4), Fraction(1, 2),
                                                                                    Fraction(3, 4), 1)):
        assert set(comprehension.eligible(know, nodes, higher)) <= set(comprehension.eligible(know, nodes, lower))


def test_raising_the_barrier_never_lowers_the_cover_and_never_makes_an_unmakeable_decision_makeable():
    steps = [Fraction(k, 6) for k in range(7)]
    flips = 0
    for seed in range(800):
        rng = random.Random(7000 + seed)
        names = [f"e{i}" for i in range(rng.randint(2, 6))]
        know = random_knowledge(rng, names, people=rng.randint(2, 6))
        nodes = rng.sample(names, rng.randint(2, len(names)))
        rows = comprehension.sweep(know, nodes, steps)
        assert [r.barrier for r in rows] == steps
        for lower, higher in zip(rows, rows[1:]):
            assert set(higher.pool) <= set(lower.pool)
            if not lower.makeable:
                assert not higher.makeable
            if lower.cover is not None and higher.cover is not None:
                assert lower.cover <= higher.cover
            if lower.makeable and not higher.makeable:
                flips += 1
                assert higher.cover is None and higher.required is None and higher.gaps
    assert flips > 100  # decisions really do become unmakeable as the barrier rises


def test_a_higher_barrier_can_force_a_larger_cover_and_then_make_the_decision_unmakeable():
    # the person who completes the shortest cover understands little, so a barrier that removes them costs a third person
    nodes = [str(i) for i in range(1, 9)]
    know = {"H1": [str(i) for i in (1, 2, 3, 4, 5, 6)], "H2": [str(i) for i in (1, 2, 3, 4, 5, 7)],
            "H3": ["2", "3", "8"], "L": ["7", "8"]}
    rows = comprehension.sweep(know, nodes, [0, Fraction(3, 8), Fraction(7, 8)])
    assert [r.cover for r in rows] == [2, 3, None]
    assert [r.pool for r in rows] == [("H1", "H2", "H3", "L"), ("H1", "H2", "H3"), ()]
    assert [r.makeable for r in rows] == [True, True, False]
    assert comprehension.covers(know, nodes) == [("H1", "L")]
    assert comprehension.max_barrier(know, nodes) == Fraction(3, 8)  # the highest barrier that still leaves a cover


def test_the_highest_barrier_a_decision_can_bear_is_the_last_one_at_which_the_pool_still_understands_it():
    found = 0
    for seed in range(1000):
        rng = random.Random(8000 + seed)
        names = [f"e{i}" for i in range(rng.randint(1, 6))]
        know = random_knowledge(rng, names)
        nodes = rng.sample(names, rng.randint(1, len(names)))
        top = comprehension.max_barrier(know, nodes)
        values = sorted({comprehension.coverage(know, p, nodes) for p in know} | {Fraction(0)})
        makeable = [v for v in values if comprehension.requirement(know, nodes, v).makeable]
        if not comprehension.requirement(know, nodes, 0).makeable:
            assert top is None and not makeable
            continue
        found += 1
        assert top == max(makeable)
        assert comprehension.requirement(know, nodes, top).makeable
        assert all(not comprehension.requirement(know, nodes, v).makeable for v in values if v > top)
    assert found > 600


def test_the_cost_of_entry_is_the_number_of_parts_of_the_footprint_still_to_be_understood():
    nodes = ["a", "b", "c", "d", "e"]
    know = {"p": ["a", "b"]}
    assert comprehension.entry_cost(know, "p", nodes, Fraction(1, 2)) == 1  # needs 3 of 5, has 2
    assert comprehension.entry_cost(know, "p", nodes, Fraction(2, 5)) == 0
    assert comprehension.entry_cost(know, "p", nodes, 1) == 3
    assert comprehension.entry_cost(know, "newcomer", nodes, Fraction(3, 5)) == 3
    for barrier in (Fraction(1, 5), Fraction(1, 2), Fraction(4, 5)):
        for person, held in {"p": ["a", "b"], "newcomer": []}.items():
            cost = comprehension.entry_cost({"p": ["a", "b"]}, person, nodes, barrier)
            learned = list(held) + [n for n in nodes if n not in held][:cost]
            assert Fraction(len(learned), len(nodes)) >= barrier
            if cost:
                assert Fraction(len(learned) - 1, len(nodes)) < barrier  # one fewer is not enough


# --- the requirement and its price in capture resistance -------------------------------------------------------------


def test_the_required_quorum_is_the_larger_of_a_balanced_majority_and_the_robust_attendance():
    informative = 0
    for seed in range(600):
        rng = random.Random(9000 + seed)
        names = [f"e{i}" for i in range(rng.randint(1, 6))]
        know = random_knowledge(rng, names, people=rng.randint(1, 7))
        nodes = rng.sample(names, rng.randint(1, len(names)))
        req = comprehension.requirement(know, nodes, Fraction(1, 3))
        pool = list(req.pool)
        assert req.balanced == (len(pool) // 2 + 1 if pool else 0)
        if not req.makeable:
            assert req.required is None and req.cover is None and req.robust is None
            continue
        assert req.required == max(req.balanced, req.robust) <= len(pool)
        assert req.cover <= req.robust and req.bus_factor >= 1
        for attending in itertools.combinations(pool, req.required):  # whoever comes at the required size understands it
            assert covering(know, nodes, attending)
        informative += req.robust > req.balanced
    assert informative > 30  # comprehension sometimes asks for more than a majority


def test_an_unmakeable_decision_says_what_nobody_understands():
    know = {"p": ["a"], "q": ["a", "b"]}
    req = comprehension.requirement(know, ["a", "b", "c"], 0)
    assert not req.makeable and req.gaps == frozenset({"c"}) and req.required is None
    assert comprehension.capture_bits(req, Fraction(1, 4)) is None


def test_a_smaller_electorate_is_cheaper_to_capture_condorcets_jury_theorem_in_bits():
    for x in (Fraction(1, 10), Fraction(1, 4), Fraction(2, 5)):
        prices = [surprisal.council_bits(m, (m + 1) // 2, x) for m in range(1, 42, 2)]
        assert all(a < b for a, b in zip(prices, prices[1:])), x  # strictly dearer with every pair of seats
        tails = [quorum.tail(m, (m + 1) // 2, x) for m in range(1, 42, 2)]
        assert all(a > b for a, b in zip(tails, tails[1:]))  # exact: the chance of capture falls
    # and the barrier is what removes seats: seven people, three of whom understand all three parts and four only one
    know = {f"p{i}": ["a", "b", "c"] for i in range(3)} | {f"q{i}": ["a"] for i in range(4)}
    open_door = comprehension.requirement(know, ["a", "b", "c"], 0)
    raised = comprehension.requirement(know, ["a", "b", "c"], 1)
    assert len(open_door.pool) == 7 and open_door.bus_factor == 3
    assert open_door.balanced == 4 and open_door.robust == 5 and open_door.required == 5  # any four could all be q's
    assert len(raised.pool) == 3 and raised.robust == 1 and raised.required == raised.balanced == 2
    x = Fraction(1, 4)
    assert comprehension.capture_bits(open_door, x) == pytest.approx(surprisal.council_bits(7, 5, x))
    assert comprehension.capture_bits(raised, x) == pytest.approx(surprisal.council_bits(3, 2, x))
    assert comprehension.capture_bits(raised, x) < 3 < 6 < comprehension.capture_bits(open_door, x)


def test_the_inputs_are_checked():
    with pytest.raises(ValueError):
        comprehension.footprint(chain(), [])
    with pytest.raises(ValueError):
        comprehension.footprint(chain(), ["nowhere"])
    with pytest.raises(TypeError):
        comprehension.footprint(chain(), "b")
    with pytest.raises(ValueError):
        comprehension.footprint({"a": ["b"]}, ["a"])  # b is relied on and is not a node
    with pytest.raises(ValueError):
        comprehension.eligible({"p": ["a"]}, ["a"], 2)
    with pytest.raises(ValueError):
        comprehension.eligible({"p": ["a"]}, ["a"], Fraction(-1, 2))
    with pytest.raises(TypeError):
        comprehension.eligible({"p": ["a"]}, ["a"], 0.5)  # a float is not an exact share
    with pytest.raises(TypeError):
        comprehension.eligible({"p": ["a"]}, ["a"], True)
    with pytest.raises(TypeError):
        comprehension.eligible({"p": "a"}, ["a"])  # a string is not a collection of names
    with pytest.raises(TypeError):
        comprehension.requirement([("p", ["a"])], ["a"])  # knowledge is a mapping
    with pytest.raises(TypeError):
        comprehension.requirement({"p": ["a"]}, "a")  # nodes are a collection, not one string
    with pytest.raises(ValueError):
        comprehension.sweep({"p": ["a"]}, ["a"], [Fraction(1, 2), Fraction(1, 4)])  # barriers rise
    assert len(comprehension.sweep({"p": ["a"]}, ["a"], [Fraction(1, 2), Fraction(1, 2)])) == 2  # and may repeat
