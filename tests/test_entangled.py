"""Entangled decisions, exactly: a shared state lets parties who cannot talk win a common-interest game
more often than any classical strategy can, and it moves no information.

A COMMON-INTEREST GAME is one in which every party wins or loses together: an exclusively win-win deal,
with private inputs (what each party alone knows) and no communication. The CLASSICAL VALUE is the best
win probability over every strategy that uses only the party's own input and shared randomness; shared
randomness does not help because the value is linear in the mixture, so the maximum is at a deterministic
strategy, found here by enumerating all of them. The QUANTUM VALUE is the win probability of explicit
local measurements on a shared state, computed in exact rationals over the Gaussian rationals (no
floating point anywhere).

Three games, three witnesses:

    GHZ           three parties; classical value 3/4; the entangled state wins with certainty;
    magic square  two parties; classical value 8/9; the entangled state wins with certainty;
    CHSH          two parties; classical value 3/4; a rational strategy on a Bell pair wins with 577/676.

And the limits, checked as exactly as the gains: nothing here beats a classical strategy without
entanglement (a product state never does), the advantage grows with the degree of entanglement and is
a formula, the correlations carry no message (no party's output distribution depends on another's
input), one use consumes the resource (after the measurements the state is a product state), and
no-signalling alone does not characterise what entanglement can do: a PR box is no-signalling and wins
CHSH with certainty, which no quantum strategy can.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import pytest

from hyperoptimization import entangled as E
from hyperoptimization.entangled import Gauss


def rational_angle(rng):
    """(cos, sin) of a random angle, both rational: the rational parametrisation of the circle."""
    t = Fraction(rng.randint(-9, 9), rng.randint(1, 9))
    return (1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)


# --- the arithmetic and the engine, against a dense matrix oracle ---------------------------------------------------------


def test_gaussian_rationals_are_a_field_with_i_squared_minus_one_and_a_conjugate():
    i = Gauss(0, 1)
    assert i * i == Gauss(-1) == -1 and Gauss(3, 4) * Gauss(3, -4) == 25
    assert Gauss(1, 2) + Gauss(3, -5) == Gauss(4, -3) and Gauss(1, 2) - Gauss(1, 2) == 0
    assert Gauss(2, 3).conj() == Gauss(2, -3) and Gauss(Fraction(1, 2), 0) * 2 == 1
    assert hash(Gauss(1, 2)) == hash(Gauss(1, 2)) and Gauss(0, 0) == 0 and not Gauss(0, 0) and Gauss(0, 1)
    assert Gauss(1, 1) != Gauss(1, 2) and Gauss(1, 1) != Gauss(2, 1) and Gauss(1) != "one" and Gauss(0, 1) != 0
    assert 1 - Gauss(0, 1) == Gauss(1, -1) and 2 + Gauss(0, 1) == Gauss(2, 1) and 3 * Gauss(1, 1) == Gauss(3, 3)
    assert Gauss(0, 1) * Gauss(0, 1) * Gauss(0, 1) == Gauss(0, -1)
    with pytest.raises(TypeError):
        Gauss(0.5)
    with pytest.raises(TypeError):
        Gauss(1, 0.5)
    with pytest.raises(TypeError):
        Gauss(1) + 0.5  # nothing inexact is mixed in
    with pytest.raises(TypeError):
        Gauss(1) + True  # and a truth value is not a number here
    with pytest.raises(TypeError):
        Gauss(1) * "x"
    assert repr(Gauss(1, -2)) == "(1-2i)" and repr(Gauss(Fraction(1, 2), 3)) == "(1/2+3i)"
    rng = random.Random(1)
    for _ in range(200):
        a, b, c = (Gauss(Fraction(rng.randint(-9, 9), rng.randint(1, 5)), Fraction(rng.randint(-9, 9), rng.randint(1, 5)))
                   for _ in range(3))
        assert a * (b + c) == a * b + a * c and (a * b).conj() == a.conj() * b.conj() and a * b == b * a


def kron(matrices):
    out = [[Gauss(1)]]
    for m in matrices:
        out = [[x * y for x in row for y in other] for row in out for other in m]
    return out


def dense(observable, n):
    """The observable as a full 2^n x 2^n matrix, by Kronecker products: another route to the same operator."""
    parts = {q: m for q, m in observable.factors}
    full = kron([parts.get(q, E.I2) for q in range(n)])
    return [[observable.sign * entry for entry in row] for row in full]


def matvec(matrix, vector):
    return tuple(sum((m * v for m, v in zip(row, vector)), Gauss(0)) for row in matrix)


def dense_probability(state, observables, outcomes):
    n = len(state).bit_length() - 1
    vector = tuple(state)
    for obs, outcome in zip(observables, outcomes):
        moved = matvec(dense(obs, n), vector)
        vector = tuple((v + outcome * m) * Fraction(1, 2) for v, m in zip(vector, moved))
    num = sum((a.conj() * b for a, b in zip(state, vector)), Gauss(0))
    den = sum((a.conj() * a for a in state), Gauss(0))
    assert num.im == 0 and den.im == 0
    return num.re / den.re


def random_state(rng, n):
    return tuple(Gauss(rng.randint(-3, 3), rng.randint(-3, 3)) for _ in range(2**n))


def test_the_engine_agrees_with_a_dense_matrix_computation_on_random_states_and_observables():
    rng = random.Random(2)
    checked = 0
    for _ in range(150):
        n = rng.randint(2, 4)
        state = random_state(rng, n)
        if all(a == 0 for a in state):
            continue
        letters = [rng.choice("XYZ") for _ in range(n)]
        observables = [E.pauli("".join(letters[q] if q == k else "I" for q in range(n)), sign=rng.choice((1, -1)))
                       for k in range(rng.randint(1, n))]
        outcomes = [rng.choice((1, -1)) for _ in observables]
        assert E.probability(state, observables, outcomes) == dense_probability(state, observables, outcomes)
        c, s = rational_angle(rng)
        rotated = [E.rotated(rng.randrange(n), c, s)]
        assert E.probability(state, rotated, [1]) == dense_probability(state, rotated, [1])
        checked += 1
    assert checked > 100


def test_probabilities_over_all_outcomes_of_commuting_observables_sum_to_one():
    rng = random.Random(3)
    for _ in range(40):
        state = random_state(rng, 3)
        if all(a == 0 for a in state):
            continue
        observables = [E.pauli("XII"), E.pauli("IYI"), E.pauli("IIZ")]
        total = sum(E.probability(state, observables, outcomes)
                    for outcomes in itertools.product((1, -1), repeat=3))
        assert total == 1


def test_the_walk_down_the_outcomes_gives_the_probability_of_every_outcome_and_only_those():
    rng = random.Random(6)
    checked = pruned = 0
    for _ in range(80):
        n = rng.randint(2, 4)
        state = random_state(rng, n)
        if all(a == 0 for a in state):
            continue
        observables = [E.pauli("".join(rng.choice("XYZ") if q == k else "I" for q in range(n)), sign=rng.choice((1, -1)))
                       for k in range(n)]
        walked = E.distribution(state, observables)
        for outcomes in itertools.product((1, -1), repeat=len(observables)):
            expected = E.probability(state, observables, outcomes)
            assert walked.get(outcomes, Fraction(0)) == expected, outcomes
            pruned += expected == 0
        assert sum(walked.values()) == 1 and all(mass > 0 for mass in walked.values())
        checked += 1
    bell = [E.pauli("XX"), E.pauli("ZZ"), E.pauli("YY", sign=-1)]  # XX times ZZ is -YY: the third outcome is fixed by the first two
    for _ in range(40):
        state = random_state(rng, 2)
        if all(a == 0 for a in state):
            continue
        walked = E.distribution(state, bell)
        assert all(c == a * b for (a, b, c) in walked)  # only the outcomes the algebra allows are ever visited
        for outcomes in itertools.product((1, -1), repeat=3):
            expected = E.probability(state, bell, outcomes)
            assert walked.get(outcomes, Fraction(0)) == expected
            pruned += expected == 0
        checked += 1
    assert checked > 80 and pruned > 100  # and many outcomes really were impossible, so pruning was exercised
    with pytest.raises(ValueError):
        E.distribution((Gauss(0),) * 2, [E.pauli("X")])
    with pytest.raises(ValueError):
        E.distribution((Gauss(1), Gauss(0)), [E.pauli("X"), E.pauli("Y")])  # X and Y together: not commuting


def test_the_walk_does_no_work_on_outcomes_the_state_cannot_give(monkeypatch):
    # one profile of the magic square (row 0, column 0): six observables, 64 outcomes of which 8 are possible.
    # Alice's first two cells are free and her third is fixed by the parity; Bob's first cell is the shared
    # one, fixed by hers, his second is free and his third fixed. One operator is applied per node visited:
    # 1 + 2 + 4 + 4 + 4 + 8 = 23. A walk that pruned nothing would visit 1 + 2 + 4 + 8 + 16 + 32 = 63.
    strategy = E.magic_square_strategy()
    observables = [*strategy.measure(0, 0), *strategy.measure(1, 0)]
    calls = []
    real = E.act
    monkeypatch.setattr(E, "act", lambda observable, state: calls.append(1) or real(observable, state))
    walked = E.distribution(strategy.state, observables)
    assert len(walked) == 8 and sum(walked.values()) == 1 and len(calls) == 23


def test_a_rotation_must_be_a_point_on_the_circle_and_an_empty_state_has_no_probabilities():
    with pytest.raises(ValueError):
        E.rotated(0, Fraction(1, 2), Fraction(1, 2))
    with pytest.raises(ValueError):
        E.probability((Gauss(0),) * 4, [E.pauli("XI")], [1])
    with pytest.raises(ValueError):
        E.probability((Gauss(1), Gauss(0), Gauss(0)), [E.pauli("X")], [1])  # three amplitudes are not qubits
    with pytest.raises(ValueError):
        E.probability((Gauss(1), Gauss(0)), [E.pauli("X")], [0])  # an outcome is +1 or -1
    with pytest.raises(ValueError):
        E.pauli("XQ")


def test_observables_that_do_not_commute_or_do_not_square_to_one_are_refused():
    zero_state = (Gauss(1), Gauss(0))
    assert E.probability(zero_state, [E.pauli("X")], [1]) == Fraction(1, 2)
    with pytest.raises(ValueError):
        E.probability(zero_state, [E.pauli("X"), E.pauli("Y")], [1, 1])  # X and Y on one qubit: (1 + i)/4, not a probability
    game = E.chsh_game()
    state = (Gauss(1), Gauss(0), Gauss(0), Gauss(1))
    clash = E.Strategy(state, lambda p, x: [E.pauli("XI"), E.pauli("ZI")] if p == 0 else [E.pauli("IZ")],
                       lambda p, x, o: 0, players=2)
    with pytest.raises(ValueError):
        E.table(game, clash)  # one party measuring X and Z of the same qubit at once
    double = Fraction(2)
    unsquared = E.Observable(((0, ((Gauss(double), Gauss(0)), (Gauss(0), Gauss(-double)))),))  # 2 Z: eigenvalues +-2
    big = E.Strategy(state, lambda p, x: [unsquared] if p == 0 else [E.pauli("IZ")], lambda p, x, o: 0, players=2)
    with pytest.raises(ValueError):
        E.table(game, big)
    with pytest.raises(ValueError):
        E.pauli("X", sign=2)
    with pytest.raises(ValueError, match="not one of the"):
        E.apply(zero_state, 1, E.X)  # there is only qubit 0
    with pytest.raises(ValueError, match="one outcome per observable"):
        E.collapse(zero_state, [E.pauli("X")], [1, -1])
    assert E.shift(E.pauli("XZ"), 3).factors == ((3, E.X), (4, E.Z))
    assert E.product_state([(1, 0), (0, 1)]) == (Gauss(0), Gauss(1), Gauss(0), Gauss(0))  # |0>|1> is the basis state 01


def test_the_magic_square_rows_multiply_to_the_identity_and_the_columns_to_minus_it():
    square = E.MAGIC_SQUARE
    identity = kron([E.I2, E.I2])
    for r in range(3):
        product = identity
        for c in range(3):
            product = matmul(product, dense(square[r][c], 2))
        assert product == identity
    for c in range(3):
        product = identity
        for r in range(3):
            product = matmul(product, dense(square[r][c], 2))
        assert product == [[-entry for entry in row] for row in identity]
    for r, c in itertools.product(range(3), repeat=2):  # every line commutes with itself, and each cell squares to one
        cell = dense(square[r][c], 2)
        assert matmul(cell, cell) == identity
        for other in range(3):
            assert matmul(cell, dense(square[r][other], 2)) == matmul(dense(square[r][other], 2), cell)
            assert matmul(cell, dense(square[other][c], 2)) == matmul(dense(square[other][c], 2), cell)


def matmul(a, b):
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), Gauss(0)) for j in range(len(b[0]))] for i in range(len(a))]


# --- GHZ ----------------------------------------------------------------------------------------------------------------------------------------


def test_ghz_classical_value_is_three_quarters_by_enumerating_every_deterministic_strategy():
    game = E.ghz_game()
    assert E.classical_value(game) == Fraction(3, 4)
    assert sum(game.prior.values()) == 1 and len(game.prior) == 4 and all(sum(i) % 2 == 0 for i in game.prior)


def test_the_ghz_state_wins_with_certainty_which_no_classical_strategy_can():
    game = E.ghz_game()
    assert E.quantum_value(game, E.ghz_strategy(1, 1)) == 1 > E.classical_value(game)


GHZ_TYPE_STATES = [(1, 1), (3, 4), (5, 12), (8, 15), (1, 0), (0, 1), (7, 24), (1, 2), (2, 3), (4, 3)]  # (alpha, beta)


def test_the_value_of_a_ghz_type_state_is_one_half_plus_its_degree_of_entanglement_halved():
    game = E.ghz_game()
    classical = E.classical_value(game)
    for a, b in GHZ_TYPE_STATES:
        degree = Fraction(2 * a * b, a * a + b * b)  # sin 2t for the state cos t |000> + sin t |111>
        value = E.quantum_value(game, E.ghz_strategy(a, b))
        assert value == Fraction(1, 2) + degree / 2, (a, b)
        assert (value > classical) == (degree > Fraction(1, 2))  # entanglement pays exactly beyond a threshold
    # a product state is worth a coin toss with these measurements, and the best classical strategy is better
    assert E.quantum_value(game, E.ghz_strategy(1, 0)) == Fraction(1, 2) < classical


def test_nothing_with_a_product_state_beats_the_classical_value_it_is_a_classical_strategy():
    rng = random.Random(4)
    ghz, chsh = E.ghz_game(), E.chsh_game()
    for _ in range(25):
        parts = [(Gauss(rng.randint(-3, 3), rng.randint(-3, 3)), Gauss(rng.randint(-3, 3), rng.randint(-3, 3)))
                 for _ in range(3)]
        if any(a == 0 and b == 0 for a, b in parts):
            continue
        state = E.product_state(parts)
        assert E.quantum_value(ghz, E.Strategy(state, E.ghz_strategy(1, 1).measure, E.ghz_strategy(1, 1).decode)) \
            <= Fraction(3, 4)
    for _ in range(40):
        angles = [rational_angle(rng) for _ in range(4)]
        parts = [(Gauss(rng.randint(-3, 3)), Gauss(rng.randint(-3, 3))) for _ in range(2)]
        if any(a == 0 and b == 0 for a, b in parts):
            continue
        strategy = E.chsh_strategy(*angles, state=E.product_state(parts))
        assert E.quantum_value(chsh, strategy) <= Fraction(3, 4)


def test_once_the_parties_have_measured_the_state_is_a_product_and_a_second_round_is_only_classical():
    game = E.ghz_game()
    strategy = E.ghz_strategy(1, 1)
    first = [E.pauli("XII"), E.pauli("IXI"), E.pauli("IIX")]
    for outcomes in itertools.product((1, -1), repeat=3):
        after = E.collapse(strategy.state, first, outcomes)
        if all(a == 0 for a in after):
            continue  # an outcome the entangled state cannot give
        assert E.quantum_value(game, E.Strategy(after, strategy.measure, strategy.decode)) <= Fraction(3, 4)
        assert E.quantum_value(game, E.Strategy(after, strategy.measure, strategy.decode)) < 1


# --- the magic square ------------------------------------------------------------------------------------------------------------------


def test_there_is_no_classical_ledger_of_nine_entries_that_satisfies_every_row_and_every_column():
    solutions = 0
    for cells in itertools.product((1, -1), repeat=9):
        grid = [cells[0:3], cells[3:6], cells[6:9]]
        rows = all(grid[r][0] * grid[r][1] * grid[r][2] == 1 for r in range(3))
        cols = all(grid[0][c] * grid[1][c] * grid[2][c] == -1 for c in range(3))
        solutions += rows and cols
    assert solutions == 0  # the product of all nine is +1 by rows and -1 by columns
    assert E.magic_square_ledgers() == 0


def test_magic_square_classical_value_is_eight_ninths_and_the_entangled_pair_wins_every_time():
    game = E.magic_square_game()
    assert sum(game.prior.values()) == 1 and len(game.prior) == 9
    assert E.classical_value(game) == Fraction(8, 9)
    assert E.quantum_value(game, E.magic_square_strategy()) == 1


def test_the_magic_square_pair_never_gives_an_answer_the_referee_rejects():
    game = E.magic_square_game()
    strategy = E.magic_square_strategy()
    table = E.table(game, strategy)
    for profile, distribution in table.items():
        assert sum(distribution.values()) == 1
        for outputs, mass in distribution.items():
            assert all(outputs[p] in game.allowed(p, profile[p]) for p in range(2)), (profile, outputs, mass)


# --- CHSH ---------------------------------------------------------------------------------------------------------------------------------------


def test_chsh_classical_value_is_three_quarters_and_a_rational_strategy_on_a_bell_pair_beats_it():
    game = E.chsh_game()
    assert E.classical_value(game) == Fraction(3, 4)
    strategy = E.chsh_strategy((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)),
                               (Fraction(119, 169), Fraction(120, 169)), (Fraction(119, 169), Fraction(-120, 169)))
    value = E.quantum_value(game, strategy)
    assert value == Fraction(577, 676) > Fraction(3, 4)
    assert abs(float(value) - (2 + 2**0.5) / 4) < 4e-6  # within a few millionths of the Tsirelson value (recalled, not proved here)
    assert value < 1


def test_outcomes_that_decode_to_the_same_output_have_their_probabilities_added():
    """A party may measure more than it reports. Here Alice also measures a spare qubit that is in an equal
    superposition, and her output ignores that outcome: two outcomes decode to one output, and `table` must add
    their probabilities. Every shipped strategy decodes one outcome to one output, so only this exercises it.
    The table must equal the plain strategy's, with each profile's probabilities adding to one."""
    angles = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)),
              (Fraction(119, 169), Fraction(120, 169)), (Fraction(119, 169), Fraction(-120, 169)))
    plain = E.chsh_strategy(*angles)
    one, zero = Gauss(1), Gauss(0)
    bell_and_spare = (one, one, zero, zero, zero, zero, one, one)  # qubits 0 and 1 a Bell pair, qubit 2 in |0> + |1>

    def measure(p, x):
        return [*plain.measure(p, x), E.pauli("Z", 2)] if p == 0 else plain.measure(p, x)

    wider = E.Strategy(bell_and_spare, measure, plain.decode, players=2)
    game = E.chsh_game()
    expected, got = E.table(game, plain), E.table(game, wider)
    assert got == expected
    for profile, masses in got.items():
        assert sum(masses.values()) == 1, profile
    assert E.quantum_value(game, wider) == Fraction(577, 676)


def test_the_commutation_caches_are_bounded_so_a_long_run_cannot_grow_without_limit():
    """`_commute` and `_squares_to_one` remember an answer per observable. A cache with no bound keeps every
    observable ever checked alive, which a process that tries many strategies would feel (audit 5)."""
    for cached in (E._commute, E._squares_to_one):
        size = cached.cache_info().maxsize
        assert size is not None and 0 < size <= 1 << 16, size
    limit = E._squares_to_one.cache_info().maxsize
    E._squares_to_one.cache_clear()
    for step in range(1, limit + 300):  # more distinct observables than the cache can hold
        t = Fraction(step, 7)
        observable = E.rotated(0, (1 - t * t) / (1 + t * t), 2 * t / (1 + t * t))
        assert E._squares_to_one(observable, 1)
    assert E._squares_to_one.cache_info().currsize == limit  # full, not growing
    again = E.rotated(0, Fraction(3, 5), Fraction(4, 5))  # and still correct after the oldest entries were dropped
    assert E._squares_to_one(again, 1) and E._squares_to_one(again, 1)
    assert E._squares_to_one.cache_info().hits >= 1


def test_the_engine_reproduces_the_analytic_chsh_value_and_no_sampled_strategy_exceeds_tsirelsons_bound():
    rng = random.Random(5)
    game = E.chsh_game()
    best = Fraction(0)
    for _ in range(150):
        (c0, s0), (c1, s1), (d0, t0), (d1, t1) = [rational_angle(rng) for _ in range(4)]
        value = E.quantum_value(game, E.chsh_strategy((c0, s0), (c1, s1), (d0, t0), (d1, t1)))

        def corr(c, s, d, t):
            return c * d + s * t  # cos(a - b) for a Bell pair measured along those angles

        s_value = corr(c0, s0, d0, t0) + corr(c0, s0, d1, t1) + corr(c1, s1, d0, t0) - corr(c1, s1, d1, t1)
        assert value == Fraction(1, 2) + s_value / 8  # the analytic formula, by another route
        assert s_value**2 <= 8  # Tsirelson's bound, exactly, on every sample
        best = max(best, value)
    assert best > Fraction(3, 4)  # and some random strategies do beat the classical value


# --- what entanglement does not do ---------------------------------------------------------------------------------------------


def test_no_party_output_distribution_depends_on_another_partys_input():
    for game, strategy in ((E.ghz_game(), E.ghz_strategy(1, 1)), (E.ghz_game(), E.ghz_strategy(3, 4)),
                           (E.magic_square_game(), E.magic_square_strategy()),
                           (E.chsh_game(), E.chsh_strategy((1, 0), (0, 1), (Fraction(3, 5), Fraction(4, 5)),
                                                           (Fraction(3, 5), Fraction(-4, 5))))):
        assert E.is_no_signalling(game, E.table(game, strategy)), game.name


def test_a_pr_box_is_no_signalling_and_wins_chsh_with_certainty_which_no_quantum_strategy_does():
    game = E.chsh_game()
    box = E.pr_box()
    assert E.is_no_signalling(game, box) and E.table_value(game, box) == 1
    assert E.table_value(game, box) > Fraction(577, 676) > E.classical_value(game)  # local < quantum < no-signalling


def test_a_table_where_one_party_is_told_the_others_input_is_caught_and_communication_wins_chsh_outright():
    game = E.chsh_game()
    told = {(x, y): {(0, x & y): Fraction(1)} for x in (0, 1) for y in (0, 1)}  # Bob's output uses Alice's input
    assert not E.is_no_signalling(game, told)
    assert E.table_value(game, told) == 1  # one message wins what no entangled pair can
    leaking = {(x, y): {(y, 0): Fraction(1)} for x in (0, 1) for y in (0, 1)}  # Alice's output is Bob's input
    assert not E.is_no_signalling(game, leaking)
    with pytest.raises(ValueError):
        E.table_value(game, {(0, 0): {(0, 0): Fraction(1)}})  # a row for every profile
    with pytest.raises(ValueError):
        E.table_value(game, {k: {(0, 0): Fraction(1, 2)} for k in itertools.product((0, 1), repeat=2)})


def test_an_output_the_referee_does_not_allow_loses_and_the_bit_convention_is_plus_one_for_zero():
    game = E.chsh_game()
    refused = {(x, y): {(2, 2): Fraction(1)} for x in (0, 1) for y in (0, 1)}  # 2 xor 2 = 0 would "win" three inputs of four
    assert E.table_value(game, refused) == 0
    z = (Fraction(1), Fraction(0))
    on_zero = E.chsh_strategy(z, z, z, z, state=E.product_state([(1, 0), (1, 0)]))  # both measure Z on |00>: outcome +1
    assert E.table(game, on_zero)[(0, 0)] == {(0, 0): Fraction(1)}  # +1 is the bit 0
    on_one = E.chsh_strategy(z, z, z, z, state=E.product_state([(0, 1), (0, 1)]))  # on |11> the outcome is -1
    assert E.table(game, on_one)[(0, 0)] == {(1, 1): Fraction(1)}


def test_the_games_are_checked_for_their_inputs():
    with pytest.raises(ValueError):
        E.CommonInterestGame("bad", 2, ((0, 1), (0, 1), (0, 1)), {(0, 0): Fraction(1)}, lambda p, x: (0, 1),
                             lambda i, o: True)  # three sets of inputs for two players
    with pytest.raises(ValueError):
        E.CommonInterestGame("bad", 2, ((0, 1), (0, 1)), {(0, 0): Fraction(1, 2)}, lambda p, x: (0, 1), lambda i, o: True)
    with pytest.raises(ValueError):
        E.CommonInterestGame("bad", 2, ((0, 1), (0, 1)), {(0, 5): Fraction(1)}, lambda p, x: (0, 1), lambda i, o: True)
    with pytest.raises(ValueError):
        E.quantum_value(E.chsh_game(), E.ghz_strategy(1, 1))  # three qubits for a two-party game
