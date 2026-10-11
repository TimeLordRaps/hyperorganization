#!/usr/bin/env python3
# PROVENANCE. Written on 2026-10-11 by a subagent (the owner's standing rule: subagents use the smallest model tier) from a written specification alone: it never saw
# this repository's implementation of the same functions. Besides this comment,
# the only other change is the variable `l` renamed `col` in `_kron` (E741).
# It is an UNTRUSTED oracle: agreement between it and the package shows that implementation slips are
# caught, not that the definitions are right (TECHNICAL_DEBT.md, HO-024). Run by differential_oracles.py.
"""Exact reference oracle for three quantum nonlocal games.

Games: GHZ (3 parties), Mermin-Peres magic square (2 parties, 4 qubits) and CHSH
(2 parties, 2 qubits).  Every win probability is computed from explicit projectors
P(o) = (I + o*A)/2 of observables A with A*A = I, as

    Pr(o_1, ..., o_k) = <psi| P_1(o_1) ... P_k(o_k) |psi> / <psi|psi>,

where the bra-ket is evaluated by applying the projectors to the state vector.
All arithmetic is exact (sympy.Rational and sympy.I); no floating point is used.
"""

import itertools
from fractions import Fraction

import sympy
from sympy import Matrix, Rational, S, eye, zeros

OUTCOMES = (1, -1)      # outcome +1 is output bit 0, outcome -1 is output bit 1
HALF = Rational(1, 2)

PAULI = {
    "I": Matrix([[1, 0], [0, 1]]),
    "X": Matrix([[0, 1], [1, 0]]),
    "Y": Matrix([[0, -sympy.I], [sympy.I, 0]]),
    "Z": Matrix([[1, 0], [0, -1]]),
}


# ---------------------------------------------------------------------------
# exact linear-algebra helpers
# ---------------------------------------------------------------------------

def _exact(x):
    """Return x as an exact sympy rational. Floats (and other inexact input) are refused."""
    if isinstance(x, (float, bool)):
        raise TypeError("inexact input rejected: %r" % (x,))
    if isinstance(x, Fraction):
        return Rational(x.numerator, x.denominator)
    val = sympy.sympify(x)
    if not getattr(val, "is_Rational", False):
        raise TypeError("expected an exact rational, got %r" % (x,))
    return Rational(val)


def _as_rational(expr):
    """Expand and simplify an exact scalar; it must come out as a rational number."""
    val = sympy.simplify(sympy.expand(sympy.sympify(expr)))
    if not getattr(val, "is_Rational", False):
        raise ArithmeticError("expected a rational value, got %s" % (val,))
    return Rational(val)


def _dagger(M):
    return M.conjugate().T


def _is_zero_matrix(M):
    return all(sympy.expand(M[i, j]) == 0 for i in range(M.rows) for j in range(M.cols))


def _kron(A, B):
    """Kronecker product A (x) B; the left factor carries the more significant index."""
    ar, ac = A.shape
    br, bc = B.shape
    out = zeros(ar * br, ac * bc)
    for i, j in itertools.product(range(ar), range(ac)):
        for k, col in itertools.product(range(br), range(bc)):  # lint: E741, was `l`
            out[i * br + k, j * bc + col] = A[i, j] * B[k, col]
    return out


def _pauli_on(n, start, letters):
    """n-qubit operator: Pauli `letters` on qubits start, start+1, ... and identity elsewhere.
    Qubit 0 is the leftmost tensor factor."""
    at = {start + i: ch for i, ch in enumerate(letters)}
    out = Matrix([[1]])
    for q in range(n):
        out = _kron(out, PAULI[at.get(q, "I")])
    return out


def _apply(F, v):
    """Matrix-vector product F v, summing only over the nonzero entries of F."""
    out = zeros(F.rows, 1)
    for i in range(F.rows):
        acc = S.Zero
        for j in range(F.cols):
            entry = F[i, j]
            if entry != 0:
                acc += entry * v[j, 0]
        out[i, 0] = acc
    return out


def _expect_product(psi, factors):
    """<psi| F_1 F_2 ... F_k |psi> (unnormalized); F_k is applied to the state first."""
    v = psi
    for F in reversed(factors):
        v = _apply(F, v)
    return sympy.expand((_dagger(psi) * v)[0, 0])


def _norm_sq(psi):
    return sympy.expand((_dagger(psi) * psi)[0, 0])


def _observable(obs):
    """Check that obs is a +/-1 observable: Hermitian with obs*obs = I."""
    n = obs.rows
    assert _is_zero_matrix(obs * obs - eye(n)), "observable is not an involution"
    assert _is_zero_matrix(_dagger(obs) - obs), "observable is not Hermitian"
    return obs


def _commute(a, b):
    assert _is_zero_matrix(a * b - b * a), "observables do not commute"


def _projector(obs, outcome):
    """P(o) = (I + o*A)/2."""
    return (eye(obs.rows) + outcome * obs) * HALF


def _bit(outcome):
    """Outcome +1 -> output bit 0; outcome -1 -> output bit 1."""
    return 0 if outcome == 1 else 1


# ---------------------------------------------------------------------------
# Game 1: GHZ (3 parties)
# ---------------------------------------------------------------------------

GHZ_PROFILES = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))


def ghz_value(alpha, beta):
    """Exact win probability of the 3-party GHZ game for |psi> = alpha|000> + beta|111>."""
    alpha = _exact(alpha)
    beta = _exact(beta)
    n = 3
    dim = 2 ** n
    psi = zeros(dim, 1)
    psi[0, 0] = alpha           # |000>
    psi[dim - 1, 0] = beta      # |111>
    norm = _norm_sq(psi)
    if norm == 0:
        raise ValueError("state has zero norm")

    total = S.Zero
    for x in GHZ_PROFILES:
        # party p measures X on its own qubit if x_p = 0, and Y if x_p = 1
        proj = []
        for p in range(n):
            obs = _observable(_pauli_on(n, p, "Y" if x[p] else "X"))
            proj.append({o: _projector(obs, o) for o in OUTCOMES})

        target = 1 if any(x) else 0     # XOR of the outputs must equal OR of the inputs
        win = S.Zero
        mass = S.Zero
        for outs in itertools.product(OUTCOMES, repeat=n):
            prob = _expect_product(psi, [proj[p][outs[p]] for p in range(n)]) / norm
            mass += prob
            if (_bit(outs[0]) ^ _bit(outs[1]) ^ _bit(outs[2])) == target:
                win += prob
        assert _as_rational(mass) == 1, "outcome probabilities do not sum to 1"
        total += win

    return _as_rational(total / len(GHZ_PROFILES))


# ---------------------------------------------------------------------------
# Game 2: Mermin-Peres magic square (2 parties, 4 qubits)
# ---------------------------------------------------------------------------

# cell (row, column) -> (sign, two-qubit Pauli string)
MAGIC_TABLE = (
    ((-1, "IX"), (-1, "XI"), (+1, "XX")),
    ((+1, "ZI"), (+1, "IZ"), (+1, "ZZ")),
    ((+1, "ZX"), (+1, "XZ"), (+1, "YY")),
)


def magic_value():
    """Exact win probability of the Mermin-Peres magic square game with the Bell-pair state."""
    n = 4
    dim = 2 ** n
    # amplitude 1 on |b0 b1 b2 b3> whenever b0 == b2 and b1 == b3 (qubit 0 most significant)
    psi = zeros(dim, 1)
    for idx in range(dim):
        b = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
        if b[0] == b[2] and b[1] == b[3]:
            psi[idx, 0] = 1
    norm = _norm_sq(psi)
    assert norm == 4

    # cell observables as operators on one pair of qubits
    cell = [[None] * 3 for _ in range(3)]
    for r in range(3):
        for c in range(3):
            sign, letters = MAGIC_TABLE[r][c]
            cell[r][c] = _observable(sign * _pauli_on(2, 0, letters))
    for r in range(3):              # each row is a commuting triple
        for j1, j2 in ((0, 1), (0, 2), (1, 2)):
            _commute(cell[r][j1], cell[r][j2])
    for c in range(3):              # each column is a commuting triple
        for i1, i2 in ((0, 1), (0, 2), (1, 2)):
            _commute(cell[i1][c], cell[i2][c])

    # Alice acts on qubits (0,1): her operator for cell (r,j) is cell[r][j] (x) I.
    # Bob acts on qubits (2,3): his operator for cell (i,c) is I (x) cell[i][c].
    alice = {}
    bob = {}
    for r in range(3):
        for j in range(3):
            obs_a = _observable(_kron(cell[r][j], eye(4)))
            alice[(r, j)] = {o: _projector(obs_a, o) for o in OUTCOMES}
    for i in range(3):
        for c in range(3):
            obs_b = _observable(_kron(eye(4), cell[i][c]))
            bob[(i, c)] = {o: _projector(obs_b, o) for o in OUTCOMES}

    total = S.Zero
    for r in range(3):              # Alice's input: row
        for c in range(3):          # Bob's input: column
            win = S.Zero
            mass = S.Zero
            invalid = S.Zero
            for a_out in itertools.product(OUTCOMES, repeat=3):
                for b_out in itertools.product(OUTCOMES, repeat=3):
                    # six mutually commuting projectors: Alice's row r and Bob's column c
                    factors = [alice[(r, j)][a_out[j]] for j in range(3)]
                    factors += [bob[(i, c)][b_out[i]] for i in range(3)]
                    prob = _expect_product(psi, factors) / norm
                    mass += prob

                    bits_a = [_bit(o) for o in a_out]   # cells (r,0), (r,1), (r,2)
                    bits_b = [_bit(o) for o in b_out]   # cells (0,c), (1,c), (2,c)
                    if sum(bits_a) % 2 != 0 or sum(bits_b) % 2 != 1:
                        invalid += prob                 # invalid answers lose
                        continue
                    # Alice's bit for cell (r,c) is her index c; Bob's is his index r
                    if bits_a[c] == bits_b[r]:
                        win += prob

            assert _as_rational(mass) == 1, "outcome probabilities do not sum to 1"
            assert _as_rational(invalid) == 0, "probability mass on invalid answers"
            total += win

    return _as_rational(total / 9)


# ---------------------------------------------------------------------------
# Game 3: CHSH (2 parties, 2 qubits)
# ---------------------------------------------------------------------------

def chsh_value(c0, s0, c1, s1, d0, t0, d1, t1):
    """Exact CHSH win probability for |psi> = |00> + |11>.

    Alice (qubit 0), input x, measures c_x Z + s_x X; Bob (qubit 1), input y, measures
    d_y Z + t_y X.  They win iff (output_A XOR output_B) == (x AND y).
    """
    alice_coef = [(_exact(c0), _exact(s0)), (_exact(c1), _exact(s1))]
    bob_coef = [(_exact(d0), _exact(t0)), (_exact(d1), _exact(t1))]
    for u, v in alice_coef + bob_coef:
        if sympy.expand(u * u + v * v) != 1:
            raise ValueError("(%s, %s) is not on the unit circle" % (u, v))

    psi = zeros(4, 1)
    psi[0, 0] = 1               # |00>
    psi[3, 0] = 1               # |11>
    norm = _norm_sq(psi)

    total = S.Zero
    for x in (0, 1):
        cx, sx = alice_coef[x]
        A = _observable(cx * _pauli_on(2, 0, "Z") + sx * _pauli_on(2, 0, "X"))
        PA = {o: _projector(A, o) for o in OUTCOMES}
        for y in (0, 1):
            dy, ty = bob_coef[y]
            B = _observable(dy * _pauli_on(2, 1, "Z") + ty * _pauli_on(2, 1, "X"))
            PB = {o: _projector(B, o) for o in OUTCOMES}
            win = S.Zero
            mass = S.Zero
            for oa, ob in itertools.product(OUTCOMES, repeat=2):
                prob = _expect_product(psi, [PA[oa], PB[ob]]) / norm
                mass += prob
                if (_bit(oa) ^ _bit(ob)) == (x & y):
                    win += prob
            assert _as_rational(mass) == 1, "outcome probabilities do not sum to 1"
            total += win

    return _as_rational(total / 4)


if __name__ == "__main__":
    R = Rational
    assert ghz_value(1, 1) == 1
    assert ghz_value(3, 4) == R(49, 50)
    assert ghz_value(1, 0) == R(1, 2)
    assert magic_value() == 1
    assert chsh_value(1, 0, 0, 1, R(119, 169), R(120, 169),
                      R(119, 169), R(-120, 169)) == R(577, 676)
    print("OK")
