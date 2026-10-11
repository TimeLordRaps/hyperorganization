"""Existence, freedom and corrigibility as one finite game, and the law that falls out of it.

The game
    A system sits in a state s. Each step it chooses an action a from those allowed there,
    nature chooses b, and the state moves to step(s, a, b). Some states are SAFE. The system
    EXISTS from s if it has a way of choosing actions that keeps it in the safe set for ever
    whatever nature does. The states from which it exists are the VIABILITY KERNEL: the
    greatest fixed point of "safe, and some allowed action sends every nature choice back into
    the set". (Aubin's viability kernel; here it is the winning region of a safety game.)

Ranks: the ordinal phase of a state
    Compute the kernel from above: V_0 = safe, V_{k+1} = safe and "some action keeps every
    successor in V_k". The chain descends and, on a finite game, stops. A state's RANK is the
    first stage that drops it, a natural number: nature can force it out after that many
    steps and not before. A state in the kernel is never dropped and has rank omega. So the
    rank is the number of steps of grace, and omega marks the states with unbounded grace.
    (On an infinite game the ranks run through the countable ordinals; nothing here needs
    that, and `ordinal` is where it lives.)

A rank is a certificate you can check in one pass
    The ranks are the unique solution of a LOCAL equation: a state outside the region has rank
    0, and a state inside has rank 1 + max over actions of (min over nature choices of the
    successor's rank), with 1 + omega = omega. (Proof: by induction on j, {s : rank(s) > j} is
    the j-th stage of the descent, because "rank > j + 1" unfolds to "in the region and some
    action sends every nature choice to rank > j".) So `check_ranks` verifies a claimed ranking
    without recomputing the fixed point, and any single wrong entry is caught. The verifier
    never needs to trust the prover.

The greatest sustaining law (FORM, finite games; brute-force checked in the tests)
    A LAW assigns every state a set of permitted actions. Call it SUSTAINING from s if, however
    the permitted actions are chosen and whatever nature does, the system never leaves the safe
    set and never reaches a state where nothing is permitted. Let X be the kernel and
    L*(s) = { a allowed at s : every nature choice leads back into X } for s in X. Then

        a law is sustaining from s   <=>   everywhere it can reach from s it permits only
                                            actions of L*, and what it reaches lies in X.

    Proof. (=>) Let R be what the law reaches from s. R is safe, closed under every permitted
    action and every nature choice, and nothing is stuck in it. A set with those properties is
    contained in the greatest fixed point, so R is inside X; and each permitted action keeps
    its successors in R, hence in X, hence belongs to L*. (<=) The law keeps successors in X,
    where L* is never empty, and X is safe. In particular L* is itself sustaining from every
    state of X, and it contains every other sustaining law, state by state. It is the unique
    pointwise-greatest sustaining law: the LEAST RESTRICTION that preserves existence. It
    forbids exactly the actions that leave the kernel and permits everything else.

    So "which laws follow from wanting to exist, and not otherwise to be told what to do" has
    one answer, and it is canonical: it mentions no name, no weight and no vote, so it cannot
    favour a participant the game does not already single out. `renamed` and automorphism
    tests pin that. Where L* permits exactly one action the law is an OBLIGATION (`forced_moves`):
    a duty to repair is a prohibition on every alternative.

Free will, read as full support (FRAME)
    "Free will" here is the assumption that a law constrains which actions are PERMITTED and
    not which permitted action is TAKEN: every permitted action is taken with positive
    probability, and so is every nature choice. Then the system survives with probability one
    exactly when the law is sustaining (`survival_probability`, exact rationals). Precisely:
    (a) the survival probability is 1 iff the law is sustaining; (b) otherwise it is strictly
    below 1, because a reachable state outside the kernel lets nature force the system down its
    ranks with positive probability; (c) it is exactly 0 when every state the law reaches can
    reach an unsafe one. This is the sense in which an EXEMPTION is fatal: a single permission
    outside L* at a state the system reaches (an author who is not bound by what it authors)
    puts a state outside the kernel within reach, so (b) applies. That is the L2 clause
    `exempt-reach`, derived instead of adopted, inside this model.

Corrigibility (FRAME)
    Give the game a HOME set (where an overseer wants the system to be able to return) and
    CORRECTION moves (what the overseer can do). `reach_home` is the set from which the overseer
    can bring the system home. Requiring the system to stay inside it is the same game on a
    smaller safe set, so everything above holds again with the corrigible kernel and the
    corrigible greatest law. A move that leaves the corrigible region is an IRREVERSIBLE move:
    after it nobody can correct the system, though it may survive for ever. The corrigible
    kernel is inside the kernel, strictly so when survival can be bought by passing through such
    states, which is the price of corrigibility, and `corrigible=False` shows it.

What is not here
    A kernel is persistence, not goodness: a stable, uncorrupted regime that exploits some of
    its members has a kernel too. Nothing in this module says persistence is the right thing to
    want. That is the family's open GC-5, and the decision ledger records it as a conceptual
    block rather than hiding it inside the word "optimal".
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Iterator, Mapping
from fractions import Fraction

from .ground import OMEGA, Omega, Status, Verdict, rank_key

Law = dict[str, frozenset[str]]


def _names(value: Iterable[str], label: str) -> tuple[str, ...]:
    names = tuple(value)
    if not names:
        raise ValueError(f"{label} must not be empty")
    for name in names:
        if type(name) is not str or not name:
            raise ValueError(f"every name in {label} must be a nonempty string")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must not repeat a name")
    return names


class Game:
    """A finite safety game. Immutable by convention: nothing in the package mutates one."""

    __slots__ = ("states", "actions", "nature", "safe", "home", "correct",
                 "_allowed", "_step", "_key")

    def __init__(
        self,
        states: Iterable[str],
        actions: Iterable[str],
        nature: Iterable[str],
        step: Mapping[tuple[str, str, str], str],
        *,
        allowed: Mapping[str, Iterable[str]] | None = None,
        safe: Iterable[str] | None = None,
        home: Iterable[str] | None = None,
        correct: Iterable[tuple[str, str]] = (),
    ) -> None:
        self.states = _names(states, "states")
        self.actions = _names(actions, "actions")
        self.nature = _names(nature, "nature")
        known_states, known_actions = set(self.states), set(self.actions)

        if allowed is None:
            table = {s: frozenset(self.actions) for s in self.states}
        else:
            if set(allowed) != known_states:
                raise ValueError("allowed must name every state, and only states")
            table = {s: frozenset(allowed[s]) for s in self.states}
            for s, acts in table.items():
                if not acts:
                    raise ValueError(f"state {s!r} allows no action")
                if not acts <= known_actions:
                    raise ValueError(f"state {s!r} allows an unknown action")
        self._allowed = table

        moves: dict[tuple[str, str, str], str] = {}
        for (s, a, b), t in step.items():
            if s not in known_states or a not in known_actions or b not in self.nature:
                raise ValueError(f"the move {(s, a, b)!r} names an unknown state, action or "
                                 f"nature choice")
            if a not in table[s]:
                raise ValueError(f"the move {(s, a, b)!r} uses an action that is not allowed "
                                 f"in {s!r}")
            if t not in known_states:
                raise ValueError(f"the move {(s, a, b)!r} leads to the unknown state {t!r}")
            moves[(s, a, b)] = t
        for s in self.states:
            for a in table[s]:
                for b in self.nature:
                    if (s, a, b) not in moves:
                        raise ValueError(f"the move {(s, a, b)!r} is missing")
        self._step = moves

        self.safe = frozenset(self.states) if safe is None else frozenset(safe)
        if not self.safe <= known_states:
            raise ValueError("safe names an unknown state")
        if home is None:
            self.home = None
        else:
            self.home = frozenset(home)
            if not self.home <= self.safe:
                raise ValueError("home must lie within the safe set")
        self.correct = frozenset((s, t) for s, t in correct)
        for s, t in self.correct:
            if s not in known_states or t not in known_states:
                raise ValueError("a correction move names an unknown state")

        self._key = (
            self.states, self.actions, self.nature, tuple(sorted(moves.items())),
            tuple(sorted((s, tuple(sorted(a))) for s, a in table.items())),
            tuple(sorted(self.safe)),
            None if self.home is None else tuple(sorted(self.home)),
            tuple(sorted(self.correct)),
        )

    @classmethod
    def from_table(cls, *, nature: Iterable[str], table: Mapping[str, Mapping[str, Iterable[str]]],
                   safe: Iterable[str] | None = None, home: Iterable[str] | None = None,
                   correct: Iterable[tuple[str, str]] = ()) -> Game:
        """table[state][action] is the tuple of successors, one per nature choice in order."""
        choices = tuple(nature)
        actions: list[str] = []
        step: dict[tuple[str, str, str], str] = {}
        allowed: dict[str, frozenset[str]] = {}
        for s, row in table.items():
            allowed[s] = frozenset(row)
            for a, targets in row.items():
                if a not in actions:
                    actions.append(a)
                targets = tuple(targets)
                if len(targets) != len(choices):
                    raise ValueError(f"{s!r}/{a!r} needs one successor per nature choice")
                for b, t in zip(choices, targets):
                    step[(s, a, b)] = t
        return cls(tuple(table), tuple(actions), choices, step, allowed=allowed, safe=safe,
                   home=home, correct=correct)

    def relabel(self, *, states: Mapping[str, str] | None = None,
                actions: Mapping[str, str] | None = None,
                nature: Mapping[str, str] | None = None) -> Game:
        """The same game under new names. Each mapping that is given must rename every name, once."""

        def table(old: tuple[str, ...], given: Mapping[str, str] | None, label: str) -> dict[str, str]:
            if given is None:
                return {name: name for name in old}
            if set(given) != set(old):
                raise ValueError(f"the {label} renaming must cover exactly the {label}")
            if len(set(given.values())) != len(given):
                raise ValueError(f"the {label} renaming must not merge two names")
            return dict(given)

        ns, na, nn = (table(self.states, states, "states"), table(self.actions, actions, "actions"),
                      table(self.nature, nature, "nature"))
        return Game(
            tuple(ns[s] for s in self.states), tuple(na[a] for a in self.actions),
            tuple(nn[b] for b in self.nature),
            {(ns[s], na[a], nn[b]): ns[t] for (s, a, b), t in self._step.items()},
            allowed={ns[s]: frozenset(na[a] for a in acts) for s, acts in self._allowed.items()},
            safe={ns[s] for s in self.safe},
            home=None if self.home is None else {ns[s] for s in self.home},
            correct={(ns[s], ns[t]) for s, t in self.correct},
        )

    def allowed(self, state: str) -> frozenset[str]:
        return self._allowed[state]

    def target(self, state: str, action: str, nature: str) -> str:
        return self._step[(state, action, nature)]

    def __eq__(self, other: object) -> bool:
        return self._key == other._key if isinstance(other, Game) else NotImplemented

    def __hash__(self) -> int:
        return hash(self._key)

    def __repr__(self) -> str:
        return (f"Game({len(self.states)} states, {len(self.actions)} actions, "
                f"{len(self.nature)} nature choices, {len(self.safe)} safe)")


# ---------------------------------------------------------------------------
# Kernel and ranks.
# ---------------------------------------------------------------------------


def reach_home(game: Game) -> frozenset[str]:
    """The states from which the overseer can bring the system home. Everything if there is no home."""
    if game.home is None:
        return frozenset(game.states)
    reached = set(game.home)
    changed = True
    while changed:
        changed = False
        for s, t in game.correct:
            if t in reached and s not in reached:
                reached.add(s)
                changed = True
    return frozenset(reached)


def _region(game: Game, corrigible: bool) -> frozenset[str]:
    if corrigible and game.home is not None:
        return game.safe & reach_home(game)
    return game.safe


def _supported(game: Game, candidates: Iterable[str], within: frozenset[str],
               permitted: Mapping[str, frozenset[str]] | None = None) -> frozenset[str]:
    """F(within): the candidates with a permitted action that sends every nature choice into `within`.

    The one operator behind the kernel, the ranks and the greatest law. Its greatest fixed point
    inside the region is the kernel; its least fixed point is the empty set.
    """
    return frozenset(
        s for s in candidates
        if any(all(game.target(s, a, b) in within for b in game.nature)
               for a in (game.allowed(s) if permitted is None else permitted.get(s, frozenset())))
    )


def _stages(game: Game, region: frozenset[str],
            permitted: Mapping[str, frozenset[str]] | None = None) -> list[frozenset[str]]:
    current = region
    stages = [current]
    while True:
        following = _supported(game, current, current, permitted)
        if following == current:
            return stages
        stages.append(following)
        current = following


def kernel(game: Game, *, corrigible: bool = True) -> frozenset[str]:
    """The states from which the system can stay in the region for ever against any nature."""
    return _stages(game, _region(game, corrigible))[-1]


def kernel_within(game: Game, known: Iterable[str], *, corrigible: bool = True) -> frozenset[str]:
    """The states a participant who can see only `known` can be sure exist for ever.

    Whatever it cannot see it must treat as unsafe, because an unknown state may be the one that
    ends everything: ignorance is pessimism. So this is the kernel of the same game with the safe
    set cut down to what is known. It lies inside the kernel (soundness), grows with what is
    known (monotone), is the whole kernel when everything is known, and is the kernel restricted
    to `known` when `known` is closed under the moves it can make (exactness). A state is in it
    exactly when `known` holds a set that supports itself around that state (`maintains`): the
    knowledge a certificate needs is the certificate.
    """
    if isinstance(known, str):
        raise TypeError("known is a collection of state names, not one string")
    seen = frozenset(known)
    for state in seen:
        _check_state(game, state)
    return _stages(game, _region(game, corrigible) & seen)[-1]


def maintains(game: Game, held: Iterable[str], *, corrigible: bool = True) -> bool:
    """Whether `held` supports itself: it lies in the region, and each of its states has an allowed
    action that sends every nature choice back into it.

    Such a set is a finite certificate that every state in it exists for ever (the coinduction
    principle). The kernel is the union of all of them, and is itself one. The empty set supports
    itself vacuously.
    """
    inside = frozenset(held)
    for state in inside:
        _check_state(game, state)
    return inside <= _region(game, corrigible) and _supported(game, inside, inside) == inside


def inductive_kernel(game: Game, *, corrigible: bool = True) -> frozenset[str]:
    """The LEAST fixed point of the operator whose greatest fixed point is the kernel, built upward
    from the empty set.

    It is empty for every game: a game has at least one nature choice (the constructor insists), so
    no state is supported by nothing, and the iteration stops where it starts. Maintenance has no
    base case to be built up from; the kernel is its only reading, and that reading is coinductive.
    (The loop is for the case the constructor excludes: with no nature choice F(empty) is not empty,
    the iteration climbs, and the least fixed point is not empty either. The tests use that as a
    negative control.)
    """
    region = _region(game, corrigible)
    current: frozenset[str] = frozenset()
    while True:
        following = _supported(game, region, current)
        if following == current:
            return current
        current = following


def ranks(game: Game, *, corrigible: bool = True) -> dict[str, int | Omega]:
    """Each state's stage of elimination: 0 off the region, a natural number, or OMEGA in the kernel."""
    stages = _stages(game, _region(game, corrigible))
    out: dict[str, int | Omega] = {}
    for s in game.states:
        rank: int | Omega = OMEGA
        for k, stage in enumerate(stages):
            if s not in stage:
                rank = k
                break
        out[s] = rank
    return out


def check_ranks(game: Game, claimed: Mapping[str, int | Omega], *, corrigible: bool = True) -> Verdict:
    """Whether `claimed` satisfies the local rank equations, which only the true ranks do.

    Fails with one line per state whose claimed rank differs from what the equation gives at
    that state. A claimed rank must be a natural number or OMEGA for every state of the game.
    """
    if set(claimed) != set(game.states):
        raise ValueError("a claimed ranking must name every state of the game, and only those")
    for rank in claimed.values():
        rank_key(rank)  # refuses a negative number, a float, a bool
    region = _region(game, corrigible)
    problems = []
    for s in game.states:
        if s not in region:
            expected: int | Omega = 0
        else:
            best = max(
                (min((claimed[game.target(s, a, b)] for b in game.nature), key=rank_key)
                 for a in game.allowed(s)),
                key=rank_key,
            )
            expected = OMEGA if best is OMEGA else best + 1
        if claimed[s] != expected:
            problems.append(f"{s}: claimed {claimed[s]}, the equation gives {expected}")
    claim = "the claimed ranks satisfy the local rank equations"
    if problems:
        return Verdict(claim, Status.FAILS_IN_MODEL, tuple(problems))
    return Verdict(claim, Status.HOLDS_IN_MODEL, evidence=(f"{len(game.states)} states checked",))


# ---------------------------------------------------------------------------
# Laws.
# ---------------------------------------------------------------------------


def greatest_law(game: Game, *, corrigible: bool = True) -> Law:
    """L*: at each kernel state, every allowed action that keeps all nature choices in the kernel."""
    inside = kernel(game, corrigible=corrigible)
    return {
        s: frozenset(a for a in game.allowed(s)
                     if all(game.target(s, a, b) in inside for b in game.nature))
        for s in sorted(inside)
    }


def greatest_law_within(game: Game, known: Iterable[str], *, corrigible: bool = True) -> Law:
    """What a participant who can see only `known` can certify as permitted: at each state of
    `kernel_within`, every allowed action whose every outcome lands back inside it. Never above the
    greatest law, and equal to it, on the states known, when `known` is closed under the moves."""
    inside = kernel_within(game, known, corrigible=corrigible)
    return {
        s: frozenset(a for a in game.allowed(s)
                     if all(game.target(s, a, b) in inside for b in game.nature))
        for s in sorted(inside)
    }


def restricted_kernel(game: Game, law: Mapping[str, frozenset[str]], *,
                      corrigible: bool = True) -> frozenset[str]:
    """The states from which the system survives for ever using only what `law` permits.

    A state where the law permits nothing is lost. This is the kernel of the same game with
    the allowed actions cut down to the law's, so it lies inside the true kernel.
    """
    permitted = {s: frozenset(law.get(s, frozenset())) & game.allowed(s) for s in game.states}
    return _stages(game, _region(game, corrigible), permitted)[-1]


def settle(game: Game, law: Mapping[str, frozenset[str]], *, corrigible: bool = True) -> Law:
    """Prune what cannot survive, then permit everything that stays inside what does.

    The result is the greatest law that the survivors of `law` can vouch for. It never lies
    above the greatest law (it is built inside a smaller kernel), is monotone in `law`, and is
    unchanged by settling again. It is also built UPWARD from `law`'s own survivors, so a law
    that forbids the only way to reach some self-sustaining region cannot discover it:
    greatest fixed points are found from above, and a start that is too small is stuck.
    """
    inside = restricted_kernel(game, law, corrigible=corrigible)
    return {
        s: frozenset(a for a in game.allowed(s)
                     if all(game.target(s, a, b) in inside for b in game.nature))
        for s in sorted(inside)
    }


def random_law(game: Game, rng: random.Random, density: float = 0.5) -> Law:
    """A law permitting each allowed action independently with probability `density`."""
    if not 0.0 <= density <= 1.0:
        raise ValueError("density is a probability")
    return {
        s: frozenset(a for a in sorted(game.allowed(s)) if rng.random() < density)
        for s in game.states
    }


def converge(game: Game, rng: random.Random, population: int, *, density: float = 0.5,
             corrigible: bool = True) -> tuple[Law, int | None]:
    """Settle `population` random laws and pool what they permit.

    Returns the pooled law and the number of laws it took before the pool equalled the greatest
    law (None if it never did). The pool never lies above the greatest law; whether it reaches
    it depends on the starts together covering the survivors, which is why diversity of starts
    is what converges, not any single start.
    """
    if type(population) is not int or population < 0:
        raise ValueError("population is a natural number")
    target = greatest_law(game, corrigible=corrigible)
    pooled: Law = {s: frozenset() for s in target}
    reached: int | None = 0 if pooled == target else None
    for count in range(1, population + 1):
        for s, permitted in settle(game, random_law(game, rng, density), corrigible=corrigible).items():
            pooled[s] = pooled[s] | permitted
        if reached is None and pooled == target:
            reached = count
    return pooled, reached


def forced_moves(law: Mapping[str, frozenset[str]]) -> dict[str, str]:
    """The states where exactly one action is permitted: there the law is an obligation."""
    return {s: next(iter(permitted)) for s, permitted in law.items() if len(permitted) == 1}


def freedom(game: Game, law: Mapping[str, frozenset[str]]) -> Fraction:
    """The share of allowed actions the law permits, over the states it speaks about."""
    allowed = sum(len(game.allowed(s)) for s in law)
    return Fraction(sum(len(p) for p in law.values()), allowed) if allowed else Fraction(0)


def _check_state(game: Game, state: str) -> None:
    if state not in game.states:
        raise ValueError(f"{state!r} is not a state of this game")


def reachable(game: Game, law: Mapping[str, frozenset[str]], start: str, *,
              within: frozenset[str] | None = None) -> frozenset[str]:
    """Every state reachable from `start` when any permitted action and any nature choice may occur.

    With `within`, states outside it are recorded but not expanded: the first exit ends the walk
    along that branch, which is all that survival and sustaining need.
    """
    _check_state(game, start)
    seen = {start}
    frontier = [start]
    while frontier:
        s = frontier.pop()
        if within is not None and s not in within:
            continue
        for a in law.get(s, ()):
            for b in game.nature:
                t = game.target(s, a, b)
                if t not in seen:
                    seen.add(t)
                    frontier.append(t)
    return frozenset(seen)


def sustaining(game: Game, law: Mapping[str, frozenset[str]], start: str, *,
               corrigible: bool = True) -> Verdict:
    """Whether `law` keeps the system in the region for ever from `start`, with a witness if not.

    Fails with `exits:<state>` for each reachable state outside the region and `deadlock:<state>`
    for each reachable region state where the law permits nothing.
    """
    region = _region(game, corrigible)
    seen = reachable(game, law, start, within=region)
    exits = sorted(s for s in seen if s not in region)
    stuck = sorted(s for s in seen if s in region and not law.get(s))
    claim = f"the law sustains existence from {start}"
    if exits or stuck:
        witness = tuple(f"exits:{s}" for s in exits) + tuple(f"deadlock:{s}" for s in stuck)
        return Verdict(claim, Status.FAILS_IN_MODEL, witness)
    return Verdict(claim, Status.HOLDS_IN_MODEL, evidence=(f"reachable states: {len(seen)}",))


def _solve(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
    """Exact Gaussian elimination for a nonsingular system."""
    size = len(rhs)
    for col in range(size):
        pivot = next(r for r in range(col, size) if matrix[r][col] != 0)
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        inverse = 1 / matrix[col][col]
        matrix[col] = [x * inverse for x in matrix[col]]
        rhs[col] *= inverse
        for row in range(size):
            if row != col and matrix[row][col] != 0:
                factor = matrix[row][col]
                matrix[row] = [x - factor * y for x, y in zip(matrix[row], matrix[col])]
                rhs[row] -= factor * rhs[col]
    return rhs


def survival_probability(game: Game, law: Mapping[str, frozenset[str]], start: str, *,
                         corrigible: bool = True) -> Fraction:
    """P(the system never leaves the region), when every permitted action and every nature choice
    is taken with equal probability at each step: full support, exactly.

    Solved as an absorption problem in exact rationals. Refuses a law that permits nothing at a
    reachable region state, because a system with nothing it may do has no will to be free.
    """
    _check_state(game, start)
    region = _region(game, corrigible)
    if start not in region:
        return Fraction(0)
    seen = reachable(game, law, start, within=region)
    stuck = sorted(s for s in seen if s in region and not law.get(s))
    if stuck:
        raise ValueError(f"the law permits nothing at {stuck[0]!r}, which the system reaches")
    dead = {s for s in seen if s not in region}
    if not dead:
        return Fraction(1)

    live = sorted(s for s in seen if s in region)

    def moves(s: str) -> list[tuple[str, str]]:
        return [(a, b) for a in sorted(law[s]) for b in game.nature]

    can_die = set(dead)
    changed = True
    while changed:
        changed = False
        for s in live:
            if s not in can_die and any(game.target(s, a, b) in can_die for a, b in moves(s)):
                can_die.add(s)
                changed = True
    unknown = [s for s in live if s in can_die]  # includes start: it reaches a dead state
    index = {s: i for i, s in enumerate(unknown)}
    matrix = [[Fraction(0)] * len(unknown) for _ in unknown]
    rhs = [Fraction(0)] * len(unknown)
    for s in unknown:
        i = index[s]
        weight = Fraction(1, len(moves(s)))
        matrix[i][i] += 1
        for a, b in moves(s):
            t = game.target(s, a, b)
            if t in dead:
                rhs[i] += weight
            elif t in index:
                matrix[i][index[t]] -= weight
    return 1 - _solve(matrix, rhs)[index[start]]


# ---------------------------------------------------------------------------
# Brute force over every law.
# ---------------------------------------------------------------------------


def all_laws(game: Game) -> Iterator[Law]:
    """Every law on the game: each state is given every subset of its allowed actions.

    There are the product over states of 2**(number allowed) of them, so this is for games of a
    handful of states, where it lets a claim about ALL laws be checked rather than sampled.
    """
    options = []
    for s in game.states:
        acts = sorted(game.allowed(s))
        options.append([frozenset(c) for r in range(len(acts) + 1)
                        for c in itertools.combinations(acts, r)])
    for choice in itertools.product(*options):
        yield {s: choice[i] for i, s in enumerate(game.states)}


# ---------------------------------------------------------------------------
# Random games, for property tests.
# ---------------------------------------------------------------------------


def random_game(rng: random.Random, *, n: int | None = None, na: int | None = None,
                nb: int | None = None, safe_rate: float = 0.75, home: bool = False) -> Game:
    """A random finite game, reproducible from the generator's seed alone.

    Nothing here iterates a set in an order that depends on the string-hash seed: an earlier
    version drew its transitions in `frozenset` order and so produced different games, from the
    same seed, in different processes. That made a failing property test impossible to rerun.
    """
    n = n or rng.choice([3, 4, 4, 5])
    na = na or rng.choice([2, 2, 3])
    nb = nb or rng.choice([1, 2, 2, 3])
    states = tuple(f"s{i}" for i in range(n))
    actions = tuple(f"a{i}" for i in range(na))
    nature = tuple(f"n{i}" for i in range(nb))
    allowed = {s: frozenset(a for a in actions if rng.random() < 0.85) or frozenset({actions[0]})
               for s in states}
    step = {(s, a, b): rng.choice(states)
            for s in states for a in sorted(allowed[s]) for b in nature}
    safe = frozenset(s for s in states if rng.random() < safe_rate) or frozenset({states[0]})
    kwargs: dict = {}
    if home:
        kwargs["home"] = frozenset(rng.sample(sorted(safe), rng.randint(1, len(safe))))
        kwargs["correct"] = frozenset((rng.choice(states), rng.choice(states)) for _ in range(n))
    return Game(states, actions, nature, step, allowed=allowed, safe=safe, **kwargs)


# ---------------------------------------------------------------------------
# Repair as a game: the worst-case counterpart of the renewal chain.
# ---------------------------------------------------------------------------


def repair_game(m: int, q: int, *, attack: int, repair: int) -> Game:
    """n compromised seats among m; nature compromises up to `attack` more per step, the system
    repairs up to `repair`; the system exists while n < q.

    The adversarial counterpart of `renewal`: a worst-case nature in place of independent
    exponential clocks. The state "n" is a string. A state's rank is the number of steps the
    optimal worst-case attack needs; the kernel is non-empty exactly when repair keeps up with
    attack and one step of attack cannot reach the quorum.
    """
    for label, value, low in (("m", m, 1), ("q", q, 1), ("attack", attack, 0), ("repair", repair, 0)):
        if type(value) is not int or value < low:
            raise ValueError(f"{label} must be an int of at least {low}")
    if q > m:
        raise ValueError("the quorum cannot exceed the number of seats")
    states = tuple(str(n) for n in range(m + 1))
    actions = tuple(f"repair-{k}" for k in range(repair + 1))
    nature = tuple(f"attack-{j}" for j in range(attack + 1))
    step = {
        (str(n), f"repair-{k}", f"attack-{j}"): str(min(m, n - min(k, n) + j))
        for n in range(m + 1) for k in range(repair + 1) for j in range(attack + 1)
    }
    return Game(states, actions, nature, step, safe={str(n) for n in range(q)})


__all__ = [
    "Game", "Law", "OMEGA", "all_laws", "check_ranks", "forced_moves", "freedom", "greatest_law",
    "greatest_law_within", "inductive_kernel", "kernel", "kernel_within", "maintains", "ranks",
    "random_game", "random_law", "reach_home", "reachable", "repair_game", "restricted_kernel",
    "settle", "converge", "survival_probability", "sustaining",
]
