"""Self-maintenance is coinductive: a set that supports itself is a certificate, and the kernel is the union of them.

hyperethics records, as the principal open problem of its first layer, that a realm's maintenance of
its own consistency appears to presuppose the consistency being maintained. In the viability game the
same shape is exact. "Maintained" is the greatest fixed point of F(Y) = the states of the region from
which some allowed action sends every nature choice into Y. Two statements follow, and both are checked
here against definitions that share no code with `kernel`:

  * COINDUCTION. Y supports itself when Y is inside F(Y). The kernel is the union of every set that
    supports itself, so to show a state is maintained it is enough to exhibit one such set holding it.
    That is a finite certificate a verifier checks in one pass and never has to trust.
  * NO INDUCTIVE READING. The least fixed point of the same F, built upward from the empty set, is empty
    for every game, because with at least one nature choice nothing is supported by nothing. Maintenance
    has no base case to start from; it has only the coinductive reading.

What this does not show: that a realm's kernel is non-empty, or that logical consistency is a safe set.
"""

from __future__ import annotations

import itertools
import random

import pytest

from hyperoptimization import viability as V
from hyperoptimization.ground import Status


def subsets(items):
    items = sorted(items)
    for size in range(len(items) + 1):
        for chosen in itertools.combinations(items, size):
            yield frozenset(chosen)


def games(count=80, **kwargs):
    for seed in range(count):
        yield seed, V.random_game(random.Random(31000 + seed), **kwargs)


def lone_state() -> V.Game:
    """One safe state that keeps itself safe: the smallest realm that maintains itself."""
    return V.Game.from_table(nature=("n",), table={"a": {"hold": ("a",)}}, safe={"a"})


# --- the kernel is the union of the sets that support themselves -------------------------------------------


@pytest.mark.parametrize("corrigible", [False, True])
def test_the_kernel_is_the_union_of_every_set_that_supports_itself(corrigible):
    informative = whole = 0
    for seed, game in games(120, n=4, home=True):
        supporting = [y for y in subsets(game.states) if V.maintains(game, y, corrigible=corrigible)]
        union = frozenset().union(*supporting)
        kernel = V.kernel(game, corrigible=corrigible)
        assert union == kernel, (seed, sorted(union), sorted(kernel))
        assert V.maintains(game, kernel, corrigible=corrigible), seed  # the kernel supports itself
        assert all(y <= kernel for y in supporting), seed
        informative += 0 < len(kernel) < len(game.states)
        whole += len(kernel) == len(game.states)
    # not a sample of empty or full kernels (seeds are fixed: observed 35 partial and 38 full, 35 and 19 corrigible)
    assert informative >= 25 and whole >= 10, (informative, whole)


def test_the_empty_set_supports_itself_vacuously_and_a_set_with_one_state_outside_the_kernel_does_not():
    for seed, game in games(60, n=4):
        assert V.maintains(game, [], corrigible=False)
        kernel = V.kernel(game, corrigible=False)
        for outsider in sorted(set(game.states) - kernel):
            assert not V.maintains(game, kernel | {outsider}, corrigible=False), (seed, outsider)


def test_maintains_refuses_what_is_not_a_set_of_states_of_the_game():
    game = lone_state()
    with pytest.raises(ValueError, match="not a state"):
        V.maintains(game, ["zzz"])
    with pytest.raises(ValueError, match="not a state"):
        V.maintains(game, [3])
    assert V.maintains(game, ["a"]) and V.maintains(game, iter(["a", "a"]))


# --- a set that supports itself is a certificate of existence for each of its members ---------------------------


def test_exhibiting_a_set_that_supports_itself_certifies_existence_by_an_independent_replay():
    certified = 0
    for seed, game in games(60, n=4):
        for y in subsets(game.states):
            if not y or not V.maintains(game, y, corrigible=False):
                continue
            # the law that permits, inside y, every action that keeps all nature choices inside y
            law = {s: frozenset(a for a in game.allowed(s)
                                if all(game.target(s, a, b) in y for b in game.nature)) for s in y}
            for start in sorted(y):
                assert V.sustaining(game, law, start, corrigible=False).status is Status.HOLDS_IN_MODEL
                assert V.survival_probability(game, law, start, corrigible=False) == 1
                certified += 1
    assert certified >= 100


# --- the inductive reading is empty -----------------------------------------------------------------------------------


@pytest.mark.parametrize("corrigible", [False, True])
def test_the_least_fixed_point_of_the_same_operator_is_empty_for_every_game(corrigible):
    for seed, game in games(120, home=True):
        assert V.inductive_kernel(game, corrigible=corrigible) == frozenset(), seed


def test_a_realm_that_maintains_itself_has_a_kernel_and_no_inductive_reading():
    game = lone_state()
    assert V.kernel(game, corrigible=False) == frozenset({"a"})
    assert V.maintains(game, ["a"], corrigible=False)
    assert V.inductive_kernel(game, corrigible=False) == frozenset()


def test_the_inductive_reading_is_empty_because_nature_has_a_choice_and_not_otherwise():
    # negative control, outside the model: the constructor refuses a game with no nature choice, but
    # one can be made by hand. With nothing for nature to do, F(empty) is not empty, the iteration
    # climbs, and the least fixed point is the states that have a safe move; so the emptiness above
    # is the nature choice at work and not a property of the code
    game = lone_state()
    game.nature = ()
    assert V.inductive_kernel(game, corrigible=False) == frozenset({"a"})


def test_the_inductive_iteration_never_leaves_the_empty_set_by_hand():
    # F(empty) = states with an action sending every nature choice into the empty set = none, since
    # a nature choice exists; so the iteration is empty at every stage and stops at once.
    game = V.Game.from_table(nature=("n", "m"), table={"a": {"x": ("a", "a"), "y": ("a", "b")},
                                                      "b": {"x": ("b", "b")}}, safe={"a", "b"})
    assert V.kernel(game, corrigible=False) == frozenset({"a", "b"})
    assert V.inductive_kernel(game, corrigible=False) == frozenset()
