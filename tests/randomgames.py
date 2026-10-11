"""Random finite games for property tests of `viability`, and the oracles that check them.

Nothing here is imported by the package. The oracles are written from the definitions with
no reference to the module under test, so a test that compares the two is comparing two
implementations, not one implementation with itself:

    own_reach          breadth-first reachability under a law, against every nature choice
    exit_times         the minimax time to leave the safe set, by value iteration on
                       integers (the module computes the same thing as a descending chain of
                       sets; these two share no code)
    all_laws           every assignment of a subset of its allowed actions to every state
"""

from __future__ import annotations

import itertools
import random

from hyperoptimization import viability as V


def random_game(rng: random.Random, *, n=None, na=None, nb=None, safe_rate=0.75, home=False):
    n = n or rng.choice([3, 4, 4, 5])
    na = na or rng.choice([2, 2, 3])
    nb = nb or rng.choice([1, 2, 2, 3])
    states = tuple(f"s{i}" for i in range(n))
    actions = tuple(f"a{i}" for i in range(na))
    nature = tuple(f"n{i}" for i in range(nb))
    allowed = {s: frozenset(a for a in actions if rng.random() < 0.85) or frozenset({actions[0]})
               for s in states}
    step = {(s, a, b): rng.choice(states) for s in states for a in allowed[s] for b in nature}
    safe = frozenset(s for s in states if rng.random() < safe_rate) or frozenset({states[0]})
    kwargs = {}
    if home:
        kwargs["home"] = frozenset(rng.sample(sorted(safe), rng.randint(1, len(safe))))
        kwargs["correct"] = frozenset((rng.choice(states), rng.choice(states)) for _ in range(n))
    return V.Game(states, actions, nature, step, allowed=allowed, safe=safe, **kwargs)


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


def all_laws(game):
    """Every assignment of a subset of the allowed actions to every state."""
    per_state = []
    for s in game.states:
        acts = sorted(game.allowed(s))
        per_state.append([frozenset(c) for r in range(len(acts) + 1)
                          for c in itertools.combinations(acts, r)])
    for choice in itertools.product(*per_state):
        yield {s: choice[i] for i, s in enumerate(game.states)}
