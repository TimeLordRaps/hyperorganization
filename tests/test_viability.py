"""Existence, freedom and corrigibility as one finite game, and the law that falls out of it.

The game. A system sits in a state s. Each step it chooses an action a from those allowed
there, nature chooses b, and the state moves to step(s, a, b). Some states are SAFE. The
system EXISTS from s if it can choose actions that keep it in the safe set for ever whatever
nature does; the set of such states is the VIABILITY KERNEL, the greatest fixed point of
"safe, and some action keeps every successor in the set".

What is claimed, each checked against an independent oracle
    * ranks. A state outside the kernel is lost after a definite number of steps (the stage
      at which the descending iteration drops it): a natural number. A kernel state is never
      lost: the rank omega. Checked against minimax exit times by value iteration.
    * THE GREATEST LAW. A law assigns each state a set of permitted actions. Call it
      SUSTAINING from s if, whichever permitted action is taken and whatever nature does, the
      system stays safe for ever and is never left with nothing permitted. Then
          a law is sustaining from s  <=>  everywhere it can reach from s it permits only
                                           actions that keep every successor in the kernel.
      So there is one pointwise-greatest sustaining law, L*: it forbids exactly the actions
      that leave the kernel and permits everything else. It is the least restriction that
      preserves existence, and it is unique. Checked by enumerating EVERY law on random
      games and EVERY start state.
    * FREE WILL, as full support. If every permitted action is taken with positive
      probability (the law constrains which actions are possible, not which is taken) and so
      is every nature choice, survival with probability one is exactly the sustaining
      condition. Checked with exact absorption probabilities, bracketed by an independent
      step-by-step distribution.
    * EXEMPTION IS FATAL. Add to L* a single permission outside it at a state the system
      reaches, and the result is not sustaining and survival has probability below one.
    * COROLLARY, CORRIGIBILITY. Require the system to stay where an overseer can still bring
      it home (`reach_home`). That is the same game on a smaller safe set. The corrigible
      kernel is inside the kernel, strictly so for a system that survives only by passing
      through states from which it cannot be brought back. Moves that leave the corrigible
      region are the irreversible moves.
    * ANONYMITY. L* does not depend on the names of states, actions or nature choices, and
      an automorphism of the game maps L* to itself.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

import pytest
from randomgames import all_laws, exit_times, own_reach, random_game, reach_within

from hyperoptimization import viability as V
from hyperoptimization.ground import OMEGA, Status

# --- a game you can check by hand ------------------------------------------------------------------


def cliff():
    """x is unsafe; c always falls into x; b always falls into c; a can stay at a for ever."""
    return V.Game.from_table(
        nature=("n",),
        table={
            "a": {"stay": ("a",), "go": ("b",)},
            "b": {"stay": ("c",), "go": ("c",)},
            "c": {"stay": ("x",), "go": ("x",)},
            "x": {"stay": ("x",), "go": ("x",)},
        },
        safe={"a", "b", "c"},
    )


def test_ranks_of_a_game_worked_by_hand():
    ranks = V.ranks(cliff())
    assert ranks == {"x": 0, "c": 1, "b": 2, "a": OMEGA}


def test_the_kernel_of_the_cliff_is_the_one_state_with_a_self_loop():
    assert V.kernel(cliff()) == frozenset({"a"})


def test_the_greatest_law_of_the_cliff_forbids_only_the_step_that_leaves_the_kernel():
    law = V.greatest_law(cliff())
    assert law == {"a": frozenset({"stay"})}
    assert V.forced_moves(law) == {"a": "stay"}
    assert V.freedom(cliff(), law) == F(1, 2)  # one of the two allowed actions is permitted


def nature_game():
    """Nature picks which of two rooms the system lands in; only one room is a trap."""
    return V.Game.from_table(
        nature=("left", "right"),
        table={
            "hub": {"safe-bet": ("ok", "ok"), "gamble": ("ok", "trap")},
            "ok": {"rest": ("hub", "hub")},
            "trap": {"rest": ("dead", "dead")},
            "dead": {"rest": ("dead", "dead")},
        },
        safe={"hub", "ok", "trap"},
    )


def test_nature_is_an_adversary_so_a_gamble_with_any_losing_branch_is_forbidden():
    law = V.greatest_law(nature_game())
    assert law["hub"] == frozenset({"safe-bet"})
    assert V.ranks(nature_game())["trap"] == 1 and V.ranks(nature_game())["hub"] == OMEGA


# --- ranks against the minimax oracle ---------------------------------------------------------------------


def test_ranks_agree_with_minimax_exit_times_on_random_games():
    rng = random.Random(61)
    omegas = finite = 0
    for _ in range(400):
        game = random_game(rng)
        want = exit_times(game, set(game.safe))
        got = V.ranks(game, corrigible=False)
        assert got == want, (game.states, got, want)
        omegas += sum(1 for r in got.values() if r is OMEGA)
        finite += sum(1 for r in got.values() if r is not OMEGA)
    assert omegas > 200 and finite > 200  # the sample has both kinds of state in quantity


def test_a_rank_is_zero_off_the_safe_set_and_omega_exactly_on_the_kernel():
    rng = random.Random(62)
    for _ in range(200):
        game = random_game(rng)
        ranks = V.ranks(game, corrigible=False)
        kernel = V.kernel(game, corrigible=False)
        for s in game.states:
            assert (ranks[s] is OMEGA) == (s in kernel)
            if s not in game.safe:
                assert ranks[s] == 0


# --- the greatest law: every law, every start ----------------------------------------------------------------


def test_a_law_sustains_existence_exactly_when_it_permits_nothing_outside_the_greatest_law():
    rng = random.Random(63)
    sustaining_seen = lost_seen = 0
    for _ in range(120):
        game = random_game(rng, n=rng.choice([3, 4]), na=2)
        kernel = V.kernel(game, corrigible=False)
        star = V.greatest_law(game, corrigible=False)
        for law in all_laws(game):
            for start in game.states:
                reach = own_reach(game, law, start)
                deadlock = any(not law[s] for s in reach)
                independent = reach <= set(game.safe) and not deadlock
                contained = (not deadlock and reach <= set(kernel)
                             and all(law[s] <= star[s] for s in reach))
                verdict = V.sustaining(game, law, start, corrigible=False)
                assert (verdict.status is Status.HOLDS_IN_MODEL) == independent == contained, (
                    game.states, law, start)
                sustaining_seen += independent
                lost_seen += not independent
    assert sustaining_seen > 5000 and lost_seen > 5000


def test_the_greatest_law_is_itself_sustaining_from_every_kernel_state_and_never_empty_there():
    rng = random.Random(64)
    for _ in range(300):
        game = random_game(rng)
        kernel = V.kernel(game, corrigible=False)
        star = V.greatest_law(game, corrigible=False)
        assert set(star) == set(kernel)
        for s in kernel:
            assert star[s]
            assert V.sustaining(game, star, s, corrigible=False).status is Status.HOLDS_IN_MODEL


def test_every_sustaining_law_is_no_freer_than_the_greatest():
    rng = random.Random(65)
    for _ in range(60):
        game = random_game(rng, n=3, na=2)
        star = V.greatest_law(game, corrigible=False)
        for law in all_laws(game):
            for start in game.safe:
                if V.sustaining(game, law, start, corrigible=False).status is Status.HOLDS_IN_MODEL:
                    reach = own_reach(game, law, start)
                    assert sum(len(law[s]) for s in reach) <= sum(len(star[s]) for s in reach)


def test_a_failing_verdict_names_where_the_law_leaves_the_safe_set_or_runs_dry():
    game = cliff()
    letting_go = {"a": frozenset({"go"}), "b": frozenset({"stay"}), "c": frozenset({"go"}),
                  "x": frozenset({"stay"})}
    verdict = V.sustaining(game, letting_go, "a", corrigible=False)
    assert verdict.status is Status.FAILS_IN_MODEL and "exits:x" in verdict.witness
    stuck = {"a": frozenset(), "b": frozenset(), "c": frozenset(), "x": frozenset()}
    assert "deadlock:a" in V.sustaining(game, stuck, "a", corrigible=False).witness
    assert V.sustaining(game, {"a": frozenset({"stay"})}, "a", corrigible=False).status \
        is Status.HOLDS_IN_MODEL


# --- free will as full support: exact probabilities -------------------------------------------------------------


def stepwise_bounds(game, law, start, region, rounds=120):
    """(lower, upper) on P(eventually leave the region): mass already out, and that plus the mass
    still in a state from which leaving is possible. By distribution stepping; no linear solve."""
    states = sorted(own_reach(game, law, start) | set(game.states))
    can_leave = {s for s in states if s not in region}
    changed = True
    while changed:
        changed = False
        for s in states:
            if s in can_leave or not law.get(s):
                continue
            if any(game.target(s, a, b) in can_leave for a in law[s] for b in game.nature):
                can_leave.add(s)
                changed = True
    mass = {start: F(1)}
    out = F(0)
    for _ in range(rounds):
        nxt = {}
        for s, m in mass.items():
            if s not in region:
                out += m
                continue
            moves = [(a, b) for a in sorted(law[s]) for b in game.nature]
            for a, b in moves:
                t = game.target(s, a, b)
                nxt[t] = nxt.get(t, F(0)) + m / len(moves)
        mass = nxt
    pending = sum((m for s, m in mass.items() if s in can_leave and s in region), F(0))
    out += sum((m for s, m in mass.items() if s not in region), F(0))
    return out, out + pending


def test_survival_probability_is_one_for_the_greatest_law_and_matches_stepwise_bounds_otherwise():
    rng = random.Random(66)
    below_one = at_one = 0
    for _ in range(150):
        game = random_game(rng, n=rng.choice([3, 4]), na=2, nb=rng.choice([1, 2]))
        region = set(game.safe)
        star = V.greatest_law(game, corrigible=False)
        for start in game.safe:
            if start in star:
                assert V.survival_probability(game, star, start, corrigible=False) == 1
        for law in list(all_laws(game))[:40]:
            for start in sorted(game.safe)[:2]:
                reach = reach_within(game, law, start, region)
                if any(not law[s] for s in reach if s in region):
                    with pytest.raises(ValueError, match="permits nothing"):
                        V.survival_probability(game, law, start, corrigible=False)
                    continue
                p = V.survival_probability(game, law, start, corrigible=False)
                lower, upper = stepwise_bounds(game, law, start, region)
                assert lower <= 1 - p <= upper, (law, start, p, lower, upper)
                at_one += p == 1
                below_one += p < 1
                assert (p == 1) == (V.sustaining(game, law, start, corrigible=False).status
                                    is Status.HOLDS_IN_MODEL)
    assert at_one > 100 and below_one > 100


def test_a_known_survival_probability_worked_by_hand():
    # With both actions permitted the system "risks" with probability 1/2 at every step, so it
    # dies with probability one; permit only "stay" and it lives for ever.
    game = V.Game.from_table(
        nature=("n",),
        table={"s": {"stay": ("s",), "risk": ("dead",)}, "dead": {"stay": ("dead",)}},
        safe={"s"},
    )
    law = {"s": frozenset({"stay", "risk"}), "dead": frozenset({"stay"})}
    assert V.survival_probability(game, law, "s", corrigible=False) == 0  # it will risk eventually
    law_safe = {"s": frozenset({"stay"})}
    assert V.survival_probability(game, law_safe, "s", corrigible=False) == 1


# --- exemption is fatal ------------------------------------------------------------------------------------------


def test_one_permission_outside_the_greatest_law_at_a_reachable_state_ends_existence():
    rng = random.Random(67)
    broken = 0
    for _ in range(250):
        game = random_game(rng)
        kernel = V.kernel(game, corrigible=False)
        star = V.greatest_law(game, corrigible=False)
        for start in sorted(kernel)[:2]:
            for s in sorted(own_reach(game, star, start)):
                for a in sorted(game.allowed(s) - star[s]):
                    exempt = {t: set(star.get(t, game.allowed(t))) for t in game.states}
                    exempt[s].add(a)
                    exempt = {t: frozenset(v) for t, v in exempt.items()}
                    assert V.sustaining(game, exempt, start, corrigible=False).status \
                        is Status.FAILS_IN_MODEL
                    assert V.survival_probability(game, exempt, start, corrigible=False) < 1
                    broken += 1
    assert broken > 150  # non-vacuity: 217 instances at this seed


# --- corrigibility ---------------------------------------------------------------------------------------------


def culdesac():
    """Home is p. From p the system can wander into q, a pleasant dead end with no way back."""
    return V.Game.from_table(
        nature=("n",),
        table={
            "p": {"stay": ("p",), "wander": ("q",)},
            "q": {"linger": ("q",)},
            "r": {"hold": ("r",), "dive": ("p",)},
        },
        safe={"p", "q", "r"},
        home={"p"},
        correct={("r", "p")},
    )


def test_a_system_can_survive_for_ever_and_still_be_beyond_correction():
    game = culdesac()
    assert V.kernel(game, corrigible=False) == frozenset({"p", "q", "r"})
    assert V.reach_home(game) == frozenset({"p", "r"})
    assert V.kernel(game, corrigible=True) == frozenset({"p", "r"})
    assert V.ranks(game, corrigible=True)["q"] == 0  # outside the corrigible region


def test_the_corrigible_law_forbids_exactly_the_irreversible_move():
    law = V.greatest_law(culdesac(), corrigible=True)
    assert law["p"] == frozenset({"stay"})  # "wander" reaches q, from which nobody can bring it home
    plain = V.greatest_law(culdesac(), corrigible=False)
    assert plain["p"] == frozenset({"stay", "wander"})
    assert V.sustaining(culdesac(), plain, "p", corrigible=True).status is Status.FAILS_IN_MODEL
    assert V.sustaining(culdesac(), law, "p", corrigible=True).status is Status.HOLDS_IN_MODEL


def test_corrigibility_costs_freedom_and_never_gains_it():
    rng = random.Random(68)
    strict = 0
    for _ in range(300):
        game = random_game(rng, home=True)
        plain, corrigible = V.kernel(game, corrigible=False), V.kernel(game, corrigible=True)
        assert corrigible <= plain
        strict += corrigible < plain
        law_plain = V.greatest_law(game, corrigible=False)
        law_corr = V.greatest_law(game, corrigible=True)
        for s in corrigible:
            assert law_corr[s] <= law_plain[s]
    assert strict > 20  # corrigibility really does cost something in a good share of games


def test_no_corrigible_law_ever_leaves_the_set_from_which_home_is_reachable():
    rng = random.Random(69)
    for _ in range(300):
        game = random_game(rng, home=True)
        # reach_home from the definition: iterate "has a correction move into the set"
        home = set(game.home)
        changed = True
        while changed:
            changed = False
            for s, t in game.correct:
                if t in home and s not in home:
                    home.add(s)
                    changed = True
        assert V.reach_home(game) == frozenset(home)
        law = V.greatest_law(game, corrigible=True)
        for s, permitted in law.items():
            for a in permitted:
                assert all(game.target(s, a, b) in home for b in game.nature)


def test_without_a_home_corrigibility_changes_nothing():
    rng = random.Random(70)
    for _ in range(100):
        game = random_game(rng)
        assert game.home is None
        assert V.kernel(game, corrigible=True) == V.kernel(game, corrigible=False)
        assert V.greatest_law(game, corrigible=True) == V.greatest_law(game, corrigible=False)


# --- anonymity ----------------------------------------------------------------------------------------------------


def renamed(game, rng):
    perm = {s: f"S{i}" for i, s in enumerate(rng.sample(list(game.states), len(game.states)))}
    act = {a: f"A{i}" for i, a in enumerate(rng.sample(list(game.actions), len(game.actions)))}
    nat = {b: f"N{i}" for i, b in enumerate(rng.sample(list(game.nature), len(game.nature)))}
    twin = V.Game(
        tuple(perm.values()), tuple(act.values()), tuple(nat.values()),
        {(perm[s], act[a], nat[b]): perm[game.target(s, a, b)]
         for s in game.states for a in game.allowed(s) for b in game.nature},
        allowed={perm[s]: frozenset(act[a] for a in game.allowed(s)) for s in game.states},
        safe=frozenset(perm[s] for s in game.safe),
    )
    return twin, perm, act


def test_the_greatest_law_does_not_depend_on_the_names_things_are_given():
    rng = random.Random(71)
    for _ in range(150):
        game = random_game(rng)
        twin, perm, act = renamed(game, rng)
        star, twin_star = V.greatest_law(game, corrigible=False), V.greatest_law(twin, corrigible=False)
        assert {perm[s]: frozenset(act[a] for a in permitted) for s, permitted in star.items()} \
            == twin_star
        assert {perm[s]: r for s, r in V.ranks(game, corrigible=False).items()} \
            == V.ranks(twin, corrigible=False)


def test_an_automorphism_maps_the_greatest_law_to_itself():
    rng = random.Random(72)
    for _ in range(100):
        game = random_game(rng, n=rng.choice([3, 4]))
        left = {s: f"L{s}" for s in game.states}
        right = {s: f"R{s}" for s in game.states}
        both = V.Game(
            tuple(left.values()) + tuple(right.values()), game.actions, game.nature,
            {**{(left[s], a, b): left[game.target(s, a, b)]
                for s in game.states for a in game.allowed(s) for b in game.nature},
             **{(right[s], a, b): right[game.target(s, a, b)]
                for s in game.states for a in game.allowed(s) for b in game.nature}},
            allowed={**{left[s]: game.allowed(s) for s in game.states},
                     **{right[s]: game.allowed(s) for s in game.states}},
            safe=frozenset(left[s] for s in game.safe) | frozenset(right[s] for s in game.safe),
        )
        star = V.greatest_law(both, corrigible=False)
        for s in game.states:
            assert star.get(left[s]) == star.get(right[s])  # swapping the two copies fixes the law


# --- repair as a game: the worst-case counterpart of the renewal chain --------------------------------------------------


def worst_case_rank(n, q, attack, repair):
    """Simulate optimal play: repair as much as possible, attack as hard as possible."""
    steps = 0
    while n < q:
        n = n - min(repair, n) + attack
        steps += 1
        if steps > 10_000:
            return OMEGA
    return steps


@pytest.mark.parametrize("attack,repair", [(1, 1), (1, 2), (2, 1), (3, 1), (2, 2), (4, 1), (5, 2)])
def test_a_repair_game_is_lost_after_the_number_of_steps_the_worst_case_takes(attack, repair):
    m, q = 12, 7
    game = V.repair_game(m, q, attack=attack, repair=repair)
    ranks = V.ranks(game, corrigible=False)
    for n in range(q):
        assert ranks[str(n)] == worst_case_rank(n, q, attack, repair), (n, attack, repair)
    for n in range(q, m + 1):
        assert ranks[str(n)] == 0


def test_repair_that_keeps_up_with_attack_survives_and_repair_that_falls_behind_does_not():
    keeps_up = V.ranks(V.repair_game(12, 7, attack=2, repair=2), corrigible=False)
    assert all(r is OMEGA for n, r in keeps_up.items() if int(n) < 7)
    falls_behind = V.ranks(V.repair_game(12, 7, attack=3, repair=2), corrigible=False)
    assert all(r is not OMEGA for r in falls_behind.values())
    # one attack more than the quorum can absorb in a single step loses at once
    instant = V.ranks(V.repair_game(12, 3, attack=3, repair=3), corrigible=False)
    assert all(r == 1 for n, r in instant.items() if int(n) < 3)


def test_in_a_surviving_repair_game_the_law_obliges_repair_when_idling_would_be_fatal():
    game = V.repair_game(6, 4, attack=1, repair=1)
    law = V.greatest_law(game, corrigible=False)
    # at 3 compromised (one short of the quorum) the only safe action is to repair one
    assert law["3"] == frozenset({"repair-1"})
    assert V.forced_moves(law)["3"] == "repair-1"
    # with room to spare the system may idle
    assert "repair-0" in law["0"]


# --- validation -----------------------------------------------------------------------------------------------------------


def test_a_game_must_say_where_every_allowed_move_goes():
    with pytest.raises(ValueError, match="missing"):
        V.Game(("a", "b"), ("x",), ("n",), {("a", "x", "n"): "a"})
    with pytest.raises(ValueError, match="unknown"):
        V.Game(("a",), ("x",), ("n",), {("a", "x", "n"): "z"})
    with pytest.raises(ValueError, match="not allowed"):
        V.Game(("a",), ("x", "y"), ("n",), {("a", "x", "n"): "a", ("a", "y", "n"): "a"},
               allowed={"a": frozenset({"x"})})


def test_a_state_must_allow_something_and_home_must_be_safe():
    with pytest.raises(ValueError, match="allows no action"):
        V.Game(("a",), ("x",), ("n",), {}, allowed={"a": frozenset()})
    with pytest.raises(ValueError, match="home"):
        V.Game(("a", "b"), ("x",), ("n",), {("a", "x", "n"): "a", ("b", "x", "n"): "b"},
               safe={"a"}, home={"b"})


def test_names_may_not_repeat_or_be_empty():
    with pytest.raises(ValueError, match="repeat"):
        V.Game(("a", "a"), ("x",), ("n",), {("a", "x", "n"): "a"})
    with pytest.raises(ValueError):
        V.Game(("",), ("x",), ("n",), {("", "x", "n"): ""})


def test_games_with_the_same_content_are_equal():
    assert cliff() == cliff() and hash(cliff()) == hash(cliff())
    assert cliff() != nature_game()


# --- found by the first coverage audit ------------------------------------------------------------------


def test_without_a_home_every_state_counts_as_correctable():
    assert V.reach_home(cliff()) == frozenset(cliff().states)


def test_a_start_must_be_a_state_of_the_game():
    with pytest.raises(ValueError, match="not a state"):
        V.reachable(cliff(), {}, "nowhere")
    with pytest.raises(ValueError, match="not a state"):
        V.survival_probability(cliff(), {}, "nowhere")


def test_a_system_that_starts_outside_the_region_has_already_lost():
    assert V.survival_probability(cliff(), {}, "x", corrigible=False) == 0
    assert V.freedom(cliff(), {}) == 0


def test_survival_is_exactly_zero_when_every_reachable_state_can_reach_an_unsafe_one():
    rng = random.Random(73)
    zero = positive = 0
    for _ in range(120):
        game = random_game(rng, n=rng.choice([3, 4]), na=2, nb=rng.choice([1, 2]))
        region = set(game.safe)
        for law in list(all_laws(game))[:50]:
            for start in sorted(region)[:2]:
                seen = reach_within(game, law, start, region)
                live = {s for s in seen if s in region}
                if any(not law[s] for s in live):
                    continue
                can_die = {s for s in seen if s not in region}
                changed = True
                while changed:  # backward closure written here, not taken from the module
                    changed = False
                    for s in live - can_die:
                        if any(game.target(s, a, b) in can_die for a in law[s] for b in game.nature):
                            can_die.add(s)
                            changed = True
                p = V.survival_probability(game, law, start, corrigible=False)
                if live <= can_die and seen - live:
                    assert p == 0
                    zero += 1
                elif p > 0:
                    positive += 1
    assert zero > 50 and positive > 50


def test_repair_game_arguments_are_checked():
    for kwargs in ({"m": 0, "q": 1}, {"m": 3, "q": 0}, {"m": 3, "q": 4}, {"m": 3.0, "q": 2}):
        with pytest.raises(ValueError):
            V.repair_game(attack=1, repair=1, **kwargs)
    with pytest.raises(ValueError):
        V.repair_game(3, 2, attack=-1, repair=1)
    with pytest.raises(ValueError):
        V.repair_game(3, 2, attack=1, repair=1.5)


def test_game_inputs_are_checked_and_a_game_describes_itself():
    ok = {("a", "x", "n"): "a"}
    with pytest.raises(ValueError, match="must not be empty"):
        V.Game((), ("x",), ("n",), {})
    with pytest.raises(ValueError, match="allowed must name"):
        V.Game(("a",), ("x",), ("n",), ok, allowed={})
    with pytest.raises(ValueError, match="unknown action"):
        V.Game(("a",), ("x",), ("n",), ok, allowed={"a": frozenset({"zzz"})})
    with pytest.raises(ValueError, match="unknown state, action or nature"):
        V.Game(("a",), ("x",), ("n",), {("a", "x", "m"): "a"})
    with pytest.raises(ValueError, match="safe names"):
        V.Game(("a",), ("x",), ("n",), ok, safe={"zzz"})
    with pytest.raises(ValueError, match="correction move"):
        V.Game(("a",), ("x",), ("n",), ok, correct=[("a", "zzz")])
    with pytest.raises(ValueError, match="one successor per nature choice"):
        V.Game.from_table(nature=("n1", "n2"), table={"a": {"x": ("a",)}})
    assert "4 states" in repr(cliff()) and cliff() != 3
