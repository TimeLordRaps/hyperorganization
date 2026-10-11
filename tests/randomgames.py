"""The oracles that check `viability`, and the generator they are run on.

`random_game` lives in the package (and is re-exported here) so that a user can test their own
claims on the same games. The oracles do not: they are written from the definitions with no
reference to the module under test, so a test that compares the two is comparing two
implementations, not one implementation with itself:

    own_reach          breadth-first reachability under a law, against every nature choice
    exit_times         the minimax time to leave the safe set, by value iteration on
                       integers (the module computes the same thing as a descending chain of
                       sets; these two share no code)
"""

from __future__ import annotations

from hyperoptimization import viability as V
from hyperoptimization.viability import all_laws, random_game  # noqa: F401  (re-exported)


def own_reach(game, law, start):
    seen, frontier = {start}, [start]
    while frontier:
        s = frontier.pop()
        for a in law.get(s, ()):
            for b in game.nature:
                t = game.target(s, a, b)
                if t not in seen:
                    seen.add(t)
                    frontier.append(t)
    return seen


def reach_within(game, law, start, region):
    """Like `own_reach` but a state outside the region is recorded and not expanded: once the
    system has left, what happens after is not the question."""
    seen, frontier = {start}, [start]
    while frontier:
        s = frontier.pop()
        if s not in region:
            continue
        for a in law.get(s, ()):
            for b in game.nature:
                t = game.target(s, a, b)
                if t not in seen:
                    seen.add(t)
                    frontier.append(t)
    return seen


def exit_times(game, region):
    """rank(s) by value iteration: W_0 = 0; W_{k+1}(s) = 0 off the region, else 1 + max_a min_b W_k.

    After |S| + 1 rounds a state whose W equals |S| + 1 survives for ever; every other state's
    W is its rank (the number of steps after which nature can force the system out).
    """
    n = len(game.states)
    w = {s: 0 for s in game.states}
    for _ in range(n + 1):
        w = {
            s: 0 if s not in region else 1 + max(
                min(w[game.target(s, a, b)] for b in game.nature) for a in game.allowed(s))
            for s in game.states
        }
    return {s: (V.OMEGA if w[s] >= n + 1 else w[s]) for s in game.states}
