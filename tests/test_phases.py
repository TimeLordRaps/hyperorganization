"""The ordinal phase of a dependency graph: authority height for rules with many governors.

`authority_height` handles rules with at most one governor. A field, a repository or a clause usually
relies on several. The phase of a node is 0 if it relies on nothing, one more than the largest phase
among what it relies on otherwise, and ω if its reliance never bottoms out: it lies on a cycle or
relies, however indirectly, on a node that does. A finite graph has a finite phase at a node exactly
when nothing it reaches lies on a cycle (the pigeonhole step of E2, for graphs).

Oracles. The implementation peels the graph layer by layer. The tests do not: they compute the same
quantity as the length of the longest walk (a walk of n edges in a graph of n nodes repeats a node,
so it proves a cycle), and the cores from mutual reachability by Floyd and Warshall's closure. Every
digraph on up to four nodes, self-loops included, is checked, and random graphs beyond that.
"""

from __future__ import annotations

import itertools
import random

import pytest

from hyperoptimization import entrenchment as E
from hyperoptimization.ground import OMEGA, rank_key


def graph(*edges, nodes=()):
    """{node: dependencies} from (node, dependency) pairs, plus any isolated nodes."""
    names = set(nodes) | {a for a, _ in edges} | {b for _, b in edges}
    return {n: {b for a, b in edges if a == n} for n in sorted(names)}


def longest_walk_phase(deps):
    """The phase by definition: the length of the longest walk from the node, or OMEGA if a walk is as
    long as the graph has nodes (a repeated node is a cycle)."""
    names = sorted(deps)
    n = len(names)
    has_walk = [set(names)]  # has_walk[k] = the nodes from which a walk of k edges starts
    for _ in range(n):
        prior = has_walk[-1]
        has_walk.append({v for v in names if any(w in prior for w in deps[v])})
    out = {}
    for v in names:
        if v in has_walk[n]:
            out[v] = OMEGA
        else:
            out[v] = max(k for k in range(n + 1) if v in has_walk[k])
    return out


def mutual_groups(deps):
    """Groups of nodes on a common cycle, by Floyd-Warshall closure over paths of one or more edges."""
    names = sorted(deps)
    reach = {(a, b): b in deps[a] for a in names for b in names}
    for k in names:
        for a in names:
            for b in names:
                if reach[(a, k)] and reach[(k, b)]:
                    reach[(a, b)] = True
    groups, placed = [], set()
    for a in names:
        if a in placed or not reach[(a, a)]:
            continue
        group = tuple(b for b in names if reach[(a, b)] and reach[(b, a)])
        placed.update(group)
        groups.append(group)
    return groups


def all_digraphs(n):
    names = [f"n{i}" for i in range(n)]
    pairs = [(a, b) for a in names for b in names]  # self-loops included
    for mask in range(1 << len(pairs)):
        yield graph(*[p for i, p in enumerate(pairs) if mask >> i & 1], nodes=names)


# --- by hand -----------------------------------------------------------------------------------------------------------


def test_a_chain_a_diamond_and_an_uneven_tree_have_the_phases_worked_by_hand():
    assert E.phases(graph(("a", "b"), ("b", "c"))) == {"a": 2, "b": 1, "c": 0}
    assert E.phases(graph(("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"))) == {"a": 2, "b": 1, "c": 1, "d": 0}
    assert E.phases(graph(("a", "b"), ("a", "d"), ("b", "c"))) == {"a": 2, "b": 1, "c": 0, "d": 0}
    assert E.phases(graph(nodes=["x"])) == {"x": 0}


def test_a_node_on_a_cycle_or_above_one_has_no_finite_phase_and_a_node_beside_it_does():
    assert E.phases(graph(("a", "a"))) == {"a": OMEGA}  # relies on itself
    assert E.phases(graph(("a", "b"), ("b", "a"), ("c", "a"), ("c", "d"), ("e", "d"))) == {
        "a": OMEGA, "b": OMEGA, "c": OMEGA, "d": 0, "e": 1}
    assert E.cores(graph(("a", "b"), ("b", "a"), ("c", "a"), ("c", "d"), ("e", "d"))) == [("a", "b")]


def test_cores_are_listed_once_each_sorted_and_a_self_reliant_node_is_a_core_of_one():
    g = graph(("m", "n"), ("n", "o"), ("o", "m"), ("x", "y"), ("y", "x"), ("z", "z"), ("p", "m"))
    assert E.cores(g) == [("m", "n", "o"), ("x", "y"), ("z",)]
    shuffled = dict(reversed(list(g.items())))
    assert E.cores(shuffled) == E.cores(g)  # the order of the input does not matter


# --- against the definitions ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3])
def test_every_digraph_on_up_to_three_nodes_agrees_with_the_walk_definition(n):
    count = 0
    for g in all_digraphs(n):
        assert E.phases(g) == longest_walk_phase(g), g
        assert E.cores(g) == mutual_groups(g), g
        count += 1
    assert count == 2 ** (n * n)


def test_every_digraph_on_four_nodes_agrees_with_the_walk_definition():
    count = finite = 0
    for g in all_digraphs(4):
        got = E.phases(g)
        assert got == longest_walk_phase(g), g
        assert E.cores(g) == mutual_groups(g), g
        finite += all(p is not OMEGA for p in got.values())
        count += 1
    assert count == 65536 and finite == 543  # 543 acyclic digraphs on 4 labelled nodes, no loops allowed


def test_random_larger_graphs_agree_with_the_walk_definition():
    rng = random.Random(41000)
    seen_finite = seen_omega = 0
    for _ in range(600):
        n = rng.randint(5, 9)
        names = [f"v{i}" for i in range(n)]
        density = rng.choice([0.08, 0.15, 0.3])
        g = {a: {b for b in names if rng.random() < density} for a in names}
        got = E.phases(g)
        assert got == longest_walk_phase(g), g
        assert E.cores(g) == mutual_groups(g), g
        seen_finite += any(p is not OMEGA for p in got.values())
        seen_omega += any(p is OMEGA for p in got.values())
    assert seen_finite >= 300 and seen_omega >= 300


# --- what the phase means ------------------------------------------------------------------------------------------------


def test_a_phase_satisfies_its_local_equation_and_omega_marks_exactly_the_nodes_that_reach_a_cycle():
    rng = random.Random(41001)
    for _ in range(400):
        names = [f"v{i}" for i in range(rng.randint(2, 8))]
        g = {a: {b for b in names if rng.random() < 0.2} for a in names}
        got = E.phases(g)
        for v, deps in g.items():
            if got[v] is OMEGA:
                continue
            assert got[v] == (0 if not deps else 1 + max(got[d] for d in deps)), (g, v)  # finite means equation
        core_nodes = {n for group in E.cores(g) for n in group}
        reaches = {v: set() for v in g}
        for v in g:
            frontier = list(g[v])
            while frontier:
                w = frontier.pop()
                if w not in reaches[v]:
                    reaches[v].add(w)
                    frontier.extend(g[w])
        for v in g:
            assert (got[v] is OMEGA) == bool(({v} | reaches[v]) & core_nodes), (g, v)


def test_adding_a_dependency_never_lowers_a_phase():
    rng = random.Random(41002)
    for _ in range(400):
        names = [f"v{i}" for i in range(rng.randint(2, 7))]
        g = {a: {b for b in names if rng.random() < 0.15} for a in names}
        before = E.phases(g)
        a, b = rng.choice(names), rng.choice(names)
        g[a] = g[a] | {b}
        after = E.phases(g)
        assert all(rank_key(after[v]) >= rank_key(before[v]) for v in names), (g, a, b)


def test_with_at_most_one_governor_the_phase_is_the_authority_height_of_the_same_rules():
    checked = 0
    for n in range(1, 6):
        names = [f"r{i}" for i in range(n)]
        for governors in itertools.product([None, *range(n)], repeat=n):
            rules = [E.Rule(names[i], 1, None if g is None else names[g]) for i, g in enumerate(governors)]
            deps = {names[i]: set() if g is None else {names[g]} for i, g in enumerate(governors)}
            assert E.phases(deps) == E.authority_height(rules), governors
            checked += 1
    assert checked == sum((n + 1) ** n for n in range(1, 6))


def test_a_graph_that_obeys_the_no_apex_rule_has_no_finite_phase_anywhere():
    # every node relies on something (no apex): in a finite graph every walk returns to itself
    rng = random.Random(41003)
    for _ in range(200):
        names = [f"v{i}" for i in range(rng.randint(1, 8))]
        g = {a: set(rng.sample(names, rng.randint(1, len(names)))) for a in names}
        assert all(p is OMEGA for p in E.phases(g).values()), g


# --- comparability: the order that dependence leaves undecided ----------------------------------------------------


def closure_incomparable(deps):
    """Pairs of nodes with no path either way, by Floyd and Warshall's closure."""
    names = sorted(deps)
    reach = {(a, b): b in deps[a] for a in names for b in names}
    for k in names:
        for a in names:
            for b in names:
                if reach[(a, k)] and reach[(k, b)]:
                    reach[(a, b)] = True
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
            if not reach[(a, b)] and not reach[(b, a)]]


@pytest.mark.parametrize("n", [1, 2, 3])
def test_incomparable_pairs_agree_with_the_closure_on_every_digraph_up_to_three_nodes(n):
    for g in all_digraphs(n):
        assert E.incomparable_pairs(g) == closure_incomparable(g), g


def test_incomparable_pairs_agree_with_the_closure_on_every_digraph_on_four_nodes_and_on_random_larger_ones():
    for g in all_digraphs(4):
        assert E.incomparable_pairs(g) == closure_incomparable(g), g
    rng = random.Random(41004)
    seen = 0
    for _ in range(300):
        names = [f"v{i}" for i in range(rng.randint(5, 9))]
        g = {a: {b for b in names if rng.random() < 0.15} for a in names}
        assert E.incomparable_pairs(g) == closure_incomparable(g), g
        seen += bool(E.incomparable_pairs(g))
    assert seen >= 150


def test_a_total_chain_leaves_nothing_undecided_and_two_branches_leave_the_pair_between_them():
    assert E.incomparable_pairs(graph(("c", "b"), ("b", "a"))) == []
    assert E.incomparable_pairs(graph(("b", "a"), ("c", "a"))) == [("b", "c")]
    assert E.incomparable_pairs(graph(("a", "b"), ("b", "a"))) == []  # a cycle is one group, mutually comparable


def test_the_ranking_of_laws_in_hyperphysics_by_dependence_has_phases_that_are_not_its_tier_numbers():
    # hyperphysics docs/research/LAW_RANKING.md, column "Depends on" (2026-10-06): consistency requirements
    # (tier 0) rely on nothing; symmetry principles (1) on tier 0; conservation laws (2) on tier 1; dynamical
    # laws (3) on tiers 0 and 1; statistical and effective laws (4) on tier 3
    laws = {
        "consistency": set(), "symmetry": {"consistency"}, "conservation": {"symmetry"},
        "dynamics": {"consistency", "symmetry"}, "statistics": {"dynamics"},
    }
    assert E.phases(laws) == {"consistency": 0, "symmetry": 1, "conservation": 2, "dynamics": 2, "statistics": 3}
    # dependence is a partial order: conservation and dynamics, and conservation and statistics, are not
    # ordered by it, so "priority in resolution" cannot be the dependence order alone
    assert E.incomparable_pairs(laws) == [("conservation", "dynamics"), ("conservation", "statistics")]


# --- refusals --------------------------------------------------------------------------------------------------------------


def test_a_dependency_that_is_not_a_node_or_a_malformed_graph_is_refused():
    with pytest.raises(ValueError, match="not a node"):
        E.phases({"a": {"ghost"}})
    with pytest.raises(ValueError, match="at least one node"):
        E.phases({})
    with pytest.raises(ValueError, match="needs a name"):
        E.phases({"": set()})
    with pytest.raises(ValueError, match="needs a name"):
        E.cores({3: set()})
    with pytest.raises(TypeError, match="not one string"):
        E.phases({"a": "a"})
