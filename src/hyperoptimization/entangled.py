"""Entangled decisions, exactly: what a shared quantum state buys parties who cannot talk, and what it does not.

Read the user's phrase as a finite question with an exact answer. Parties each hold a private input (what
only they know) and must each give an output with no communication. If every party wins or loses together
the game is COMMON-INTEREST: an exclusively win-win deal. The CLASSICAL VALUE is the best win probability of
any strategy that uses only the party's own input and shared randomness. The QUANTUM VALUE is the win
probability when the parties measure their own parts of a shared entangled state, the measurement chosen
by the input. Everything is computed in exact rationals, over the Gaussian rationals (a + bi with a and b
rational), because the measurements are Pauli measurements or rotations by rational points on the circle:
no floating point, no sampling.

Three witnesses, all exact:

    GHZ           three parties; classical value 3/4; the three-qubit GHZ state wins with certainty;
    magic square  two parties; classical value 8/9; two entangled pairs win with certainty (and there is
                  no classical ledger of nine signs that satisfies every row and column at all);
    CHSH          two parties; classical value 3/4; a rational strategy on a Bell pair wins with 577/676.
                  The best quantum value is (2 + sqrt 2)/4 (Tsirelson, recalled, not proved here), which
                  is irrational; the rational angles used here reach it to within about 3 millionths.

And the limits, checked as exactly as the gains:

    entanglement is the resource   with the same measurements a product state never beats the classical
                                   value, and for a GHZ-type state cos t |000> + sin t |111> the value is
                                   exactly 1/2 + (sin 2t)/2, which exceeds the classical 3/4 exactly when
                                   sin 2t > 1/2: the value of the state is a formula in its degree of
                                   entanglement, and below a threshold it is worth nothing;
    nothing is communicated        in every strategy here each party's output distribution depends only
                                   on its own input (no-signalling), so the shared state cannot replace
                                   disclosure; it only helps where the parties cannot or will not
                                   communicate;
    it is consumed                 after the measurements the state is a product state, and a second round
                                   on it is worth no more than a classical one;
    no-signalling is not enough    a PR box is no-signalling and wins CHSH with certainty, which no
                                   quantum strategy can, so what entanglement achieves is strictly
                                   between what local strategies and what any no-signalling rule allows.

What this does not show: that human or institutional decisions can be entangled (that is a proposal with no
witness, D28); anything about games whose parties have opposed interests (the claim that entanglement
dissolves the prisoner's dilemma depends on the strategy set allowed, recalled from the literature and not
modelled); or that the shared state can be distributed and kept at a cost an institution could pay.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


class Gauss:
    """A Gaussian rational, a + bi with a and b rational: exact complex arithmetic."""

    __slots__ = ("re", "im")

    def __init__(self, re: int | Fraction = 0, im: int | Fraction = 0) -> None:
        if type(re) is float or type(im) is float:
            raise TypeError("a Gaussian rational is built from integers and fractions, not floats")
        self.re = Fraction(re)
        self.im = Fraction(im)

    @staticmethod
    def _of(other: object) -> Gauss | None:
        if isinstance(other, Gauss):
            return other
        if isinstance(other, (int, Fraction)) and not isinstance(other, bool):
            return Gauss(other)
        return None

    def __add__(self, other: object) -> Gauss:
        o = self._of(other)
        return NotImplemented if o is None else Gauss(self.re + o.re, self.im + o.im)

    __radd__ = __add__

    def __sub__(self, other: object) -> Gauss:
        o = self._of(other)
        return NotImplemented if o is None else Gauss(self.re - o.re, self.im - o.im)

    def __rsub__(self, other: object) -> Gauss:
        o = self._of(other)
        return NotImplemented if o is None else Gauss(o.re - self.re, o.im - self.im)

    def __mul__(self, other: object) -> Gauss:
        o = self._of(other)
        if o is None:
            return NotImplemented
        return Gauss(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)

    __rmul__ = __mul__

    def __neg__(self) -> Gauss:
        return Gauss(-self.re, -self.im)

    def conj(self) -> Gauss:
        return Gauss(self.re, -self.im)

    def __eq__(self, other: object) -> bool:
        o = self._of(other)
        return NotImplemented if o is None else (self.re == o.re and self.im == o.im)

    def __hash__(self) -> int:
        return hash((self.re, self.im))

    def __bool__(self) -> bool:
        return bool(self.re) or bool(self.im)

    def __repr__(self) -> str:
        return f"({self.re}{'+' if self.im >= 0 else '-'}{abs(self.im)}i)"


Matrix = tuple[tuple[Gauss, Gauss], tuple[Gauss, Gauss]]
State = tuple[Gauss, ...]

I2: Matrix = ((Gauss(1), Gauss(0)), (Gauss(0), Gauss(1)))
X: Matrix = ((Gauss(0), Gauss(1)), (Gauss(1), Gauss(0)))
Y: Matrix = ((Gauss(0), Gauss(0, -1)), (Gauss(0, 1), Gauss(0)))
Z: Matrix = ((Gauss(1), Gauss(0)), (Gauss(0), Gauss(-1)))
_PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


@dataclass(frozen=True, slots=True)
class Observable:
    """A tensor product of 2 x 2 matrices on some qubits (the identity elsewhere), times a sign.

    With eigenvalues +1 and -1 it is a yes-or-no measurement; the caller supplies observables that
    square to the identity, and `table` checks that they do and that those measured together commute.
    """

    factors: tuple[tuple[int, Matrix], ...]
    sign: int = 1


def pauli(letters: str, first: int = 0, sign: int = 1) -> Observable:
    """Pauli letters on consecutive qubits from `first` (I leaves a qubit alone)."""
    if sign not in (1, -1):
        raise ValueError("the sign of an observable is +1 or -1")
    factors = []
    for offset, letter in enumerate(letters):
        if letter not in _PAULI:
            raise ValueError(f"{letter!r} is not a Pauli letter (I, X, Y, Z)")
        if letter != "I":
            factors.append((first + offset, _PAULI[letter]))
    return Observable(tuple(factors), sign)


def rotated(qubit: int, cos: int | Fraction, sin: int | Fraction) -> Observable:
    """cos Z + sin X on one qubit: the measurement along an angle in the XZ plane. (cos, sin) must be a
    point on the unit circle, exactly, so that the observable squares to the identity."""
    c, s = Fraction(cos), Fraction(sin)
    if c * c + s * s != 1:
        raise ValueError("(cos, sin) must satisfy cos^2 + sin^2 = 1 exactly")
    return Observable(((qubit, ((Gauss(c), Gauss(s)), (Gauss(s), Gauss(-c)))),))


def shift(observable: Observable, by: int) -> Observable:
    """The same observable acting `by` qubits further along."""
    return Observable(tuple((q + by, m) for q, m in observable.factors), observable.sign)


def _qubits(state: Sequence[Gauss]) -> int:
    n = len(state).bit_length() - 1
    if len(state) < 2 or 2**n != len(state):
        raise ValueError("a state of n qubits has 2^n amplitudes")
    return n


def apply(state: State, qubit: int, matrix: Matrix) -> State:
    """The 2 x 2 matrix applied to one qubit of the state. Qubit 0 is the most significant."""
    n = _qubits(state)
    if not 0 <= qubit < n:
        raise ValueError(f"qubit {qubit} is not one of the {n} qubits")
    shift_by = n - 1 - qubit
    out = [Gauss(0)] * len(state)
    for index, amplitude in enumerate(state):
        if not amplitude:
            continue
        bit = (index >> shift_by) & 1
        for new in (0, 1):
            entry = matrix[new][bit]
            if entry:
                target = (index & ~(1 << shift_by)) | (new << shift_by)
                out[target] = out[target] + entry * amplitude
    return tuple(out)


def act(observable: Observable, state: State) -> State:
    for qubit, matrix in observable.factors:
        state = apply(state, qubit, matrix)
    return tuple(observable.sign * a for a in state)


def inner(a: State, b: State) -> Gauss:
    return sum((x.conj() * y for x, y in zip(a, b)), Gauss(0))


def collapse(state: State, observables: Sequence[Observable], outcomes: Sequence[int]) -> State:
    """The (unnormalised) state after measuring the observables and getting the outcomes."""
    _qubits(state)
    if len(observables) != len(outcomes):
        raise ValueError("one outcome per observable")
    vector = tuple(state)
    for observable, outcome in zip(observables, outcomes):
        if outcome not in (1, -1) or isinstance(outcome, bool):
            raise ValueError("an outcome is +1 or -1")
        moved = act(observable, vector)
        vector = tuple((v + outcome * m) * Fraction(1, 2) for v, m in zip(vector, moved))
    return vector


def probability(state: State, observables: Sequence[Observable], outcomes: Sequence[int]) -> Fraction:
    """The probability of getting these outcomes when the commuting observables are measured together."""
    if not any(state):
        raise ValueError("the zero vector is not a state")
    after = collapse(state, observables, outcomes)
    numerator, norm = inner(state, after), inner(state, state)
    if numerator.im != 0:
        raise ValueError("the observables do not commute: the probability is not real")
    return numerator.re / norm.re


def product_state(parts: Sequence[tuple[int | Fraction | Gauss, int | Fraction | Gauss]]) -> State:
    """The tensor product of single-qubit states (a, b) = a|0> + b|1>, unnormalised."""
    out: State = (Gauss(1),)
    for a, b in parts:
        pair = (a if isinstance(a, Gauss) else Gauss(a), b if isinstance(b, Gauss) else Gauss(b))
        out = tuple(x * y for x in out for y in pair)
    return out


# ---------------------------------------------------------------------------
# Games, strategies and what they win.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CommonInterestGame:
    """Parties with private inputs who win or lose together, and cannot communicate.

    inputs[p] are party p's possible inputs; prior gives each profile of inputs its probability;
    allowed(p, x) are the outputs party p may give on input x; wins(profile, outputs) says whether the
    outputs win on that profile. An output outside the allowed set loses.
    """

    name: str
    players: int
    inputs: tuple[tuple[Any, ...], ...]
    prior: Mapping[tuple[Any, ...], Fraction]
    allowed: Callable[[int, Any], Sequence[Any]]
    wins: Callable[[tuple[Any, ...], tuple[Any, ...]], bool]

    def __post_init__(self) -> None:
        if len(self.inputs) != self.players:
            raise ValueError("one set of inputs per player")
        if sum(self.prior.values()) != 1:
            raise ValueError("the prior over input profiles must sum to 1")
        for profile in self.prior:
            if len(profile) != self.players or any(x not in self.inputs[p] for p, x in enumerate(profile)):
                raise ValueError(f"{profile!r} is not a profile of inputs of this game")


def classical_value(game: CommonInterestGame) -> Fraction:
    """The best win probability over every deterministic strategy: each party's output a function of its
    own input only. Shared randomness mixes deterministic strategies and the value is linear in the mix,
    so it cannot do better than the best of them."""
    strategies = []
    for p in range(game.players):
        choices = [list(game.allowed(p, x)) for x in game.inputs[p]]
        strategies.append(list(itertools.product(*choices)))
    slot = [{x: k for k, x in enumerate(game.inputs[p])} for p in range(game.players)]
    best = Fraction(0)
    for combination in itertools.product(*strategies):
        total = Fraction(0)
        for profile, weight in game.prior.items():
            outputs = tuple(combination[p][slot[p][profile[p]]] for p in range(game.players))
            if game.wins(profile, outputs):
                total += weight
        best = max(best, total)
    return best


@dataclass(frozen=True, slots=True)
class Strategy:
    """A shared state and, for each party and input, the commuting observables it measures and how its
    outcomes become an output. `players`, when given, is checked against the game."""

    state: State
    measure: Callable[[int, Any], Sequence[Observable]]
    decode: Callable[[int, Any, Sequence[int]], Any]
    players: int | None = None


Table = dict[tuple[Any, ...], dict[tuple[Any, ...], Fraction]]


def _commute(a: Observable, b: Observable, n: int) -> bool:
    if {q for q, _ in a.factors}.isdisjoint({q for q, _ in b.factors}):
        return True
    for index in range(2**n):
        basis = tuple(Gauss(int(i == index)) for i in range(2**n))
        if act(a, act(b, basis)) != act(b, act(a, basis)):
            return False
    return True


def _squares_to_one(a: Observable, n: int) -> bool:
    for index in range(2**n):
        basis = tuple(Gauss(int(i == index)) for i in range(2**n))
        if act(a, act(a, basis)) != basis:
            return False
    return True


def table(game: CommonInterestGame, strategy: Strategy) -> Table:
    """For every profile of inputs (not only those the prior reaches), the probability of each profile
    of outputs, from measuring the shared state."""
    if strategy.players is not None and strategy.players != game.players:
        raise ValueError(f"the strategy is for {strategy.players} players and the game has {game.players}")
    n = _qubits(strategy.state)
    checked_one: set[Observable] = set()
    result: Table = {}
    for profile in itertools.product(*game.inputs):
        observables: list[Observable] = []
        spans: list[tuple[int, int]] = []
        for p in range(game.players):
            chosen = list(strategy.measure(p, profile[p]))
            spans.append((len(observables), len(observables) + len(chosen)))
            observables.extend(chosen)
        for observable in observables:
            if observable not in checked_one:
                if not _squares_to_one(observable, n):
                    raise ValueError("an observable does not square to the identity")
                checked_one.add(observable)
        for a, b in itertools.combinations(observables, 2):
            if not _commute(a, b, n):
                raise ValueError("observables measured together must commute")
        distribution: dict[tuple[Any, ...], Fraction] = {}
        for outcomes in itertools.product((1, -1), repeat=len(observables)):
            mass = probability(strategy.state, observables, outcomes)
            if mass:
                outputs = tuple(strategy.decode(p, profile[p], outcomes[lo:hi])
                                for p, (lo, hi) in enumerate(spans))
                distribution[outputs] = distribution.get(outputs, Fraction(0)) + mass
        result[profile] = distribution
    return result


def _checked_table(game: CommonInterestGame, given: Mapping[tuple[Any, ...], Mapping[tuple[Any, ...], Fraction]]
                   ) -> None:
    if set(given) != set(itertools.product(*game.inputs)):
        raise ValueError("a table gives a distribution for every profile of inputs")
    for profile, distribution in given.items():
        if sum(distribution.values()) != 1 or any(m < 0 for m in distribution.values()):
            raise ValueError(f"the row for {profile!r} is not a probability distribution")


def table_value(game: CommonInterestGame,
                given: Mapping[tuple[Any, ...], Mapping[tuple[Any, ...], Fraction]]) -> Fraction:
    """The win probability of a table of output distributions."""
    _checked_table(game, given)
    total = Fraction(0)
    for profile, weight in game.prior.items():
        for outputs, mass in given[profile].items():
            if all(outputs[p] in game.allowed(p, profile[p]) for p in range(game.players)) \
                    and game.wins(profile, outputs):
                total += weight * mass
    return total


def quantum_value(game: CommonInterestGame, strategy: Strategy) -> Fraction:
    """The win probability of the strategy on the game, exactly."""
    return table_value(game, table(game, strategy))


def is_no_signalling(game: CommonInterestGame,
                     given: Mapping[tuple[Any, ...], Mapping[tuple[Any, ...], Fraction]]) -> bool:
    """Whether each party's output distribution depends only on its own input."""
    _checked_table(game, given)
    for p in range(game.players):
        seen: dict[Any, dict[Any, Fraction]] = {}
        for profile, distribution in given.items():
            marginal: dict[Any, Fraction] = {}
            for outputs, mass in distribution.items():
                marginal[outputs[p]] = marginal.get(outputs[p], Fraction(0)) + mass
            if seen.setdefault(profile[p], marginal) != marginal:
                return False
    return True


# ---------------------------------------------------------------------------
# The three games and their strategies.
# ---------------------------------------------------------------------------


def chsh_game() -> CommonInterestGame:
    """Each party gets a bit and gives a bit; they win when the bits they give differ exactly when both
    received a one."""
    inputs = ((0, 1), (0, 1))
    return CommonInterestGame(
        "CHSH", 2, inputs, {(x, y): Fraction(1, 4) for x in (0, 1) for y in (0, 1)},
        lambda p, x: (0, 1), lambda i, o: (o[0] ^ o[1]) == (i[0] & i[1]))


def chsh_strategy(a0: tuple[Any, Any], a1: tuple[Any, Any], b0: tuple[Any, Any], b1: tuple[Any, Any],
                  state: State | None = None) -> Strategy:
    """Alice measures along the angle (cos, sin) a0 or a1 by her input, Bob along b0 or b1; the shared
    state is a Bell pair unless another is given."""
    alice, bob = (a0, a1), (b0, b1)
    bell: State = (Gauss(1), Gauss(0), Gauss(0), Gauss(1)) if state is None else state

    def measure(p: int, x: int) -> list[Observable]:
        c, s = (alice if p == 0 else bob)[x]
        return [rotated(p, c, s)]

    return Strategy(bell, measure, lambda p, x, outcomes: 0 if outcomes[0] == 1 else 1, players=2)


def ghz_game() -> CommonInterestGame:
    """Three parties, inputs of even parity; they win when the parity of their bits is the OR of their inputs."""
    inputs = ((0, 1),) * 3
    prior = {profile: Fraction(1, 4) for profile in ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))}
    return CommonInterestGame(
        "GHZ", 3, inputs, prior, lambda p, x: (0, 1),
        lambda i, o: (o[0] ^ o[1] ^ o[2]) == int(any(i)))


def ghz_strategy(alpha: int | Fraction = 1, beta: int | Fraction = 1) -> Strategy:
    """Measure X on input 0 and Y on input 1, on the state alpha|000> + beta|111>."""
    state = [Gauss(0)] * 8
    state[0], state[7] = Gauss(alpha), Gauss(beta)

    def measure(p: int, x: int) -> list[Observable]:
        return [pauli("I" * p + ("X" if x == 0 else "Y") + "I" * (2 - p))]

    return Strategy(tuple(state), measure, lambda p, x, outcomes: 0 if outcomes[0] == 1 else 1, players=3)


MAGIC_SQUARE: tuple[tuple[Observable, ...], ...] = (
    (pauli("IX", sign=-1), pauli("XI", sign=-1), pauli("XX")),
    (pauli("ZI"), pauli("IZ"), pauli("ZZ")),
    (pauli("ZX"), pauli("XZ"), pauli("YY")),
)  # each row multiplies to +I, each column to -I; the cells in a line commute


def _even(bits: tuple[int, ...]) -> bool:
    return sum(bits) % 2 == 0


def magic_square_game() -> CommonInterestGame:
    """Alice is given a row and Bob a column of a 3 x 3 table. Alice fills her row with three bits of even
    parity, Bob fills his column with three bits of odd parity, and they win when they agree on the cell
    where the row and column meet."""
    triples = list(itertools.product((0, 1), repeat=3))
    even = tuple(t for t in triples if _even(t))
    odd = tuple(t for t in triples if not _even(t))
    return CommonInterestGame(
        "magic square", 2, ((0, 1, 2), (0, 1, 2)),
        {(r, c): Fraction(1, 9) for r in range(3) for c in range(3)},
        lambda p, x: even if p == 0 else odd, lambda i, o: o[0][i[1]] == o[1][i[0]])


def magic_square_strategy() -> Strategy:
    """Two entangled pairs, Alice holding one qubit of each; she measures her row of commuting observables,
    Bob his column, and the shared cell always agrees."""
    state = [Gauss(0)] * 16
    for a1, a2 in itertools.product((0, 1), repeat=2):
        state[(a1 << 3) | (a2 << 2) | (a1 << 1) | a2] = Gauss(1)

    def measure(p: int, x: int) -> list[Observable]:
        if p == 0:
            return [MAGIC_SQUARE[x][c] for c in range(3)]
        return [shift(MAGIC_SQUARE[r][x], 2) for r in range(3)]

    return Strategy(tuple(state), measure,
                    lambda p, x, outcomes: tuple(0 if o == 1 else 1 for o in outcomes), players=2)


def pr_box() -> Table:
    """The Popescu-Rohrlich box: outputs that differ exactly when both inputs are one, each output alone
    uniform. No-signalling, and it wins CHSH every time, which no quantum strategy can."""
    return {(x, y): {(a, a ^ (x & y)): Fraction(1, 2) for a in (0, 1)} for x in (0, 1) for y in (0, 1)}


__all__ = ["CommonInterestGame", "Gauss", "I2", "MAGIC_SQUARE", "Observable", "Strategy", "X", "Y", "Z",
           "act", "apply", "chsh_game", "chsh_strategy", "classical_value", "collapse", "ghz_game",
           "ghz_strategy", "inner", "is_no_signalling", "magic_square_game", "magic_square_strategy",
           "pauli", "pr_box", "probability", "product_state", "quantum_value", "rotated", "shift", "table",
           "table_value"]
