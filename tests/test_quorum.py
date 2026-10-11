"""The quorum map T(x) = P(Bin(m, x) >= q), its fixed points, and the flow it generates.

x is the probability that one seat is compromised. A body of m seats is captured when at
least q of them are. If the seats of the next body are themselves bodies of this kind
(a council of councils), the compromise probability one level up is T(x), and the levels
are the iterates of T. That iteration is a renormalization-group flow in the ordinary
sense: it has fixed points, a stability for each, and a critical point between two
phases.

What is claimed, and where it is checked
    * T equals the probability of at least q successes, by enumerating every outcome of
      m coins (not by the formula the module uses);
    * for 2 <= q <= m-1 the map has exactly one interior fixed point and it is unstable;
      for every other q it has none. Counted exactly by Sturm's theorem, and against an
      independent float grid that counts sign changes;
    * quorum duality: T_{m,q}(x) + T_{m,m-q+1}(1-x) = 1, exactly. The odd majority is its
      own dual and so is critical at exactly 1/2;
    * below the critical point capture falls doubly exponentially: T(x) <= C(m,q) x^q;
    * near the critical point the slope grows like sqrt(m) with the amplitude
      phi(Phi^-1(theta)) / sqrt(theta (1-theta)) (checked numerically, not proved here).
"""

from __future__ import annotations

import math
import random
from fractions import Fraction as F
from itertools import product

import pytest

from hyperoptimization import poly as P
from hyperoptimization import quorum as Q
from hyperoptimization.ground import OMEGA


def enumerated_tail(m, q, x):
    """P(at least q successes in m independent trials), summing over every outcome."""
    x = F(x)
    total = F(0)
    for outcome in product((0, 1), repeat=m):
        weight = F(1)
        for bit in outcome:
            weight *= x if bit else 1 - x
        if sum(outcome) >= q:
            total += weight
    return total


def grid_tail(m, q, x):
    return sum(math.comb(m, j) * x**j * (1 - x) ** (m - j) for j in range(q, m + 1))


# --- the map itself --------------------------------------------------------------------------


@pytest.mark.parametrize("m", range(1, 8))
def test_tail_is_the_probability_of_at_least_q_successes_by_enumeration(m):
    for q in range(0, m + 2):
        for x in (F(0), F(1, 3), F(1, 2), F(2, 3), F(1), F(3, 7), F(1, 10)):
            assert Q.tail(m, q, x) == enumerated_tail(m, q, x), (m, q, x)


def test_tail_reads_floats_as_the_exact_binary_rationals_they_are():
    assert Q.tail(3, 2, 0.5) == F(1, 2)
    assert Q.tail(5, 3, 0.1) == Q.tail(5, 3, F(0.1))
    assert Q.tail(5, 3, 0.1) != Q.tail(5, 3, F(1, 10))


@pytest.mark.parametrize("args", [(0, 1, 0.5), (-1, 1, 0.5), (3, -1, 0.5), (3, 2, -0.1), (3, 2, 1.1),
                                  (3.0, 2, 0.5), (3, 2.0, 0.5), (True, 1, 0.5), (3, 2, "half")])
def test_tail_refuses_what_is_not_a_quorum_or_a_probability(args):
    with pytest.raises((ValueError, TypeError)):
        Q.tail(*args)


def test_edge_quorums_are_constant_or_the_obvious_powers():
    for x in (F(1, 5), F(1, 2), F(9, 10)):
        assert Q.tail(6, 0, x) == 1  # nothing needed: always captured
        assert Q.tail(6, 7, x) == 0  # more than the body has: never captured
        assert Q.tail(6, 6, x) == x**6  # all of them
        assert Q.tail(6, 1, x) == 1 - (1 - x) ** 6  # any one of them


def test_tail_is_zero_at_zero_and_one_at_one_for_every_real_quorum():
    for m in range(1, 12):
        for q in range(1, m + 1):
            assert Q.tail(m, q, 0) == 0 and Q.tail(m, q, 1) == 1


def test_the_polynomial_form_agrees_with_the_direct_form():
    rng = random.Random(21)
    for _ in range(120):
        m = rng.randint(1, 14)
        q = rng.randint(0, m + 1)
        x = F(rng.randint(0, 20), 20)
        assert P.peval(Q.tail_poly(m, q), x) == Q.tail(m, q, x)


def test_the_slope_formula_is_the_derivative_of_the_polynomial():
    rng = random.Random(22)
    for _ in range(120):
        m = rng.randint(1, 14)
        q = rng.randint(1, m)
        x = F(rng.randint(0, 20), 20)
        assert Q.tail_slope(m, q, x) == P.peval(P.pdiff(Q.tail_poly(m, q)), x)


def test_every_real_quorum_map_is_strictly_increasing_inside_the_unit_interval():
    for m in range(1, 13):
        for q in range(1, m + 1):
            for i in range(1, 20):
                assert Q.tail_slope(m, q, F(i, 20)) > 0


def test_the_end_slopes_vanish_when_both_phases_attract():
    for m in range(3, 14):
        for q in range(2, m):
            assert Q.tail_slope(m, q, 0) == 0 and Q.tail_slope(m, q, 1) == 0
    assert Q.tail_slope(6, 1, 0) == 6  # q = 1: the origin repels
    assert Q.tail_slope(6, 6, 1) == 6  # q = m: the other end repels


# --- duality ----------------------------------------------------------------------------------


def test_dual_quorum_is_the_blocking_threshold_and_duality_is_exact():
    assert Q.dual(7, 3) == 5 and Q.dual(7, 7) == 1 and Q.dual(7, 1) == 7
    for m in range(1, 10):
        for q in range(1, m + 1):
            assert Q.dual(m, Q.dual(m, q)) == q
            for i in range(0, 13):
                x = F(i, 12)
                assert Q.tail(m, q, x) + Q.tail(m, Q.dual(m, q), 1 - x) == 1, (m, q, x)


def test_the_odd_majority_is_its_own_dual_and_is_critical_at_exactly_one_half():
    for m in (3, 5, 7, 9, 11, 13):
        q = (m + 1) // 2
        assert Q.is_self_dual(m, q)
        assert Q.tail(m, q, F(1, 2)) == F(1, 2)
        assert Q.critical_point(m, q) == (F(1, 2), F(1, 2))
    assert not Q.is_self_dual(5, 2) and not Q.is_self_dual(4, 2)


# --- fixed points -------------------------------------------------------------------------------


def sign_changes_of_f(m, q, cells=1023):
    previous = None
    changes = 0
    for i in range(1, cells + 1):
        x = i / (cells + 1)
        value = grid_tail(m, q, x) - x
        if previous is not None and value != 0 and previous != 0 and (value > 0) != (previous > 0):
            changes += 1
        if value != 0:
            previous = value
    return changes


def test_exactly_one_interior_fixed_point_iff_the_quorum_is_neither_any_nor_all():
    for m in range(2, 23):
        for q in range(0, m + 2):
            want = 1 if 2 <= q <= m - 1 else 0
            assert Q.interior_fixed_points(m, q) == want, (m, q)


def test_a_one_seat_body_is_the_identity_map_and_every_point_is_fixed():
    with pytest.raises(ValueError, match="identity"):
        Q.interior_fixed_points(1, 1)
    assert Q.tail(1, 1, F(3, 7)) == F(3, 7)


def test_the_exact_count_agrees_with_an_independent_float_grid():
    for m in range(2, 21):
        for q in range(0, m + 2):
            assert Q.interior_fixed_points(m, q) == sign_changes_of_f(m, q), (m, q)


def test_critical_point_is_bracketed_by_exact_signs():
    width = F(1, 10**9)
    for m, q in [(4, 2), (4, 3), (5, 2), (5, 4), (6, 3), (9, 3), (9, 7), (12, 8), (20, 5), (21, 11)]:
        lo, hi = Q.critical_point(m, q, width)
        assert hi - lo <= width and 0 < lo <= hi < 1
        below, above = Q.tail(m, q, lo) - lo, Q.tail(m, q, hi) - hi
        assert below <= 0 <= above
        if lo < hi:
            assert below < 0 < above


def test_no_critical_point_exists_for_an_any_or_all_quorum():
    for m, q in [(5, 1), (5, 5), (5, 0), (5, 6)]:
        with pytest.raises(ValueError, match="no interior fixed point"):
            Q.critical_point(m, q)


def test_the_critical_point_is_unstable_and_the_ends_attract():
    for m, q in [(4, 2), (5, 2), (5, 3), (6, 5), (9, 4), (15, 8)]:
        assert Q.critical_slope(m, q) > 1
        lo, _ = Q.critical_point(m, q)
        assert Q.tail(m, q, lo / 2) < lo / 2  # to the left the map pulls toward 0
        assert Q.tail(m, q, (1 + lo) / 2) > (1 + lo) / 2  # to the right toward 1


def test_the_critical_slope_is_the_exact_slope_to_within_the_bracket_it_was_read_from():
    # critical_slope reads T' at the middle of a bracket of width 1e-9 that contains x_c, so it
    # differs from the slope at x_c by at most (max |T''|) * 1e-9, and |T''| <= m (m - 1) on [0, 1].
    for m, q in [(5, 3), (7, 2), (11, 6), (9, 8)]:
        lo, hi = Q.critical_point(m, q, F(1, 10**12))
        exact = float(Q.tail_slope(m, q, (lo + hi) / 2))
        assert abs(Q.critical_slope(m, q) - exact) <= m * (m - 1) * 1e-9, (m, q)


# --- the flow -------------------------------------------------------------------------------------


def test_flow_is_the_iterate_of_the_map_computed_exactly():
    orbit = Q.flow(5, 3, F(1, 4), 3)
    assert orbit[0] == F(1, 4)
    for before, after in zip(orbit, orbit[1:]):
        assert after == Q.tail(5, 3, before)
    assert len(orbit) == 4


def test_each_side_of_the_critical_point_flows_to_its_own_attractor():
    for x0 in (F(1, 5), F(2, 5), F(9, 20)):
        orbit = Q.flow(5, 3, x0, 5)
        assert all(a > b for a, b in zip(orbit, orbit[1:])), orbit
    for x0 in (F(11, 20), F(3, 5), F(4, 5)):
        orbit = Q.flow(5, 3, x0, 5)
        assert all(a < b for a, b in zip(orbit, orbit[1:])), orbit


def test_the_critical_point_is_a_fixed_point_of_the_flow_when_it_is_rational():
    assert Q.flow(5, 3, F(1, 2), 6) == [F(1, 2)] * 7


def test_exact_flow_refuses_depths_that_would_not_fit_in_memory():
    with pytest.raises(ValueError, match="doubly exponentially"):
        Q.flow(5, 3, F(1, 3), 40)


def test_below_the_critical_point_capture_obeys_the_union_bound_at_every_level():
    for m, q in [(5, 3), (6, 3), (7, 4), (9, 5)]:
        bound = math.comb(m, q)
        orbit = Q.flow(m, q, F(1, 10), 4)
        for before, after in zip(orbit, orbit[1:]):
            assert after <= bound * before**q, (m, q, before, after)
    rng = random.Random(23)
    for _ in range(300):
        m = rng.randint(2, 12)
        q = rng.randint(1, m)
        x = F(rng.randint(0, 40), 40)
        assert Q.tail(m, q, x) <= math.comb(m, q) * x**q


def test_fate_reads_the_exact_sign_of_the_map_minus_the_identity():
    assert Q.fate(5, 3, F(2, 5)) is Q.Fate.HEALTHY
    assert Q.fate(5, 3, F(3, 5)) is Q.Fate.CAPTURED
    assert Q.fate(5, 3, F(1, 2)) is Q.Fate.CRITICAL
    assert Q.fate(5, 3, 0) is Q.Fate.HEALTHY and Q.fate(5, 3, 1) is Q.Fate.CAPTURED
    # an any-quorum captures from every positive start, an all-quorum from none short of 1
    assert Q.fate(5, 1, F(1, 1000)) is Q.Fate.CAPTURED
    assert Q.fate(5, 5, F(999, 1000)) is Q.Fate.HEALTHY
    assert Q.fate(5, 0, F(1, 1000)) is Q.Fate.CAPTURED and Q.fate(5, 6, F(999, 1000)) is Q.Fate.HEALTHY


def test_fate_agrees_with_long_iteration_away_from_the_critical_point():
    rng = random.Random(24)
    for _ in range(200):
        m = rng.randint(3, 15)
        q = rng.randint(2, m - 1)
        lo, hi = Q.critical_point(m, q, F(1, 10**6))
        x0 = F(rng.randint(1, 999), 1000)
        if lo - F(1, 100) < x0 < hi + F(1, 100):
            continue
        x = float(x0)
        for _ in range(400):
            x = Q.tail_float(m, q, x)
        assert (x > 0.5) == (Q.fate(m, q, x0) is Q.Fate.CAPTURED), (m, q, x0, x)


# --- stage counts: the ordinal phase of a starting point ----------------------------------------------


def test_stage_count_is_omega_exactly_at_a_rational_critical_point():
    assert Q.stage_count(3, 2, F(1, 2), (F(1, 4), F(3, 4))) is OMEGA
    assert Q.stage_count(5, 3, F(1, 2), (F(1, 10), F(9, 10))) is OMEGA


def test_stage_count_grows_without_bound_as_the_start_approaches_the_critical_point():
    band = (F(1, 4), F(3, 4))
    counts = [Q.stage_count(3, 2, F(1, 2) + F(1, 2**k), band) for k in range(3, 41)]
    assert all(isinstance(c, int) for c in counts)
    assert all(a <= b for a, b in zip(counts, counts[1:]))
    assert counts[-1] >= counts[0] + 40  # slope 3/2 per step: ~1.7 steps per halving
    assert counts[-1] < 100
    # symmetric by self-duality (the float orbit may differ from its mirror by one step)
    mirror = [Q.stage_count(3, 2, F(1, 2) - F(1, 2**k), band) for k in range(3, 41)]
    assert all(abs(a - b) <= 1 for a, b in zip(counts, mirror))


def test_the_stage_count_is_the_logarithm_of_the_distance_over_the_log_of_the_slope():
    band = (F(1, 4), F(3, 4))
    slope = float(Q.tail_slope(3, 2, F(1, 2)))
    for k in (10, 20, 30):
        delta = 2.0**-k
        predicted = math.log(0.25 / delta) / math.log(slope)
        got = Q.stage_count(3, 2, F(1, 2) + F(1, 2**k), band)
        assert abs(got - predicted) <= 3, (k, got, predicted)


def test_stage_count_refuses_a_band_that_contains_an_attractor():
    with pytest.raises(ValueError, match="strictly inside"):
        Q.stage_count(3, 2, F(1, 3), (F(0), F(3, 4)))
    with pytest.raises(ValueError, match="strictly inside"):
        Q.stage_count(3, 2, F(1, 3), (F(1, 4), F(1)))
    with pytest.raises(ValueError, match="start"):
        Q.stage_count(3, 2, F(9, 10), (F(1, 4), F(3, 4)))


# --- floats and the large-m scaling --------------------------------------------------------------------


def test_float_tail_agrees_with_the_exact_tail():
    rng = random.Random(25)
    for _ in range(150):
        m = rng.randint(1, 80)
        q = rng.randint(0, m + 1)
        x = rng.random()
        want = float(Q.tail(m, q, x))
        got = Q.tail_float(m, q, x)
        assert got == pytest.approx(want, rel=1e-9, abs=1e-14), (m, q, x)


def test_float_tail_handles_the_corners():
    assert Q.tail_float(10, 0, 0.3) == 1.0 and Q.tail_float(10, 11, 0.3) == 0.0
    assert Q.tail_float(10, 4, 0.0) == 0.0 and Q.tail_float(10, 4, 1.0) == 1.0
    assert 0.0 <= Q.tail_float(5000, 2500, 0.5) <= 1.0
    assert Q.tail_float(5000, 2500, 0.5) == pytest.approx(0.5, abs=0.01)


def test_float_critical_point_agrees_with_the_exact_bracket():
    for m, q in [(7, 3), (21, 11), (30, 9), (41, 30)]:
        lo, hi = Q.critical_point(m, q, F(1, 10**12))
        assert Q.critical_point_float(m, q) == pytest.approx(float((lo + hi) / 2), abs=1e-9)


@pytest.mark.parametrize("theta", [0.3, 0.5, 0.7])
def test_critical_slope_grows_like_root_m_with_the_universal_amplitude(theta):
    for m in (101, 401, 1601):
        q = round(theta * m) if theta != 0.5 else (m + 1) // 2
        ratio = Q.critical_slope_float(m, q) / math.sqrt(m)
        # The amplitude is the leading term of an expansion in 1/sqrt(m), so the error is
        # O(1/sqrt(m)). Observed |ratio - A| at m = 101 / 401 / 1601: 2e-3 / 5e-4 / 1.3e-4
        # (theta .5), 5.3e-3 / ? / 1.1e-3 (theta .3), 5e-4 / 1.2e-3 / 8.5e-4 (theta .7); the
        # bound 0.06 / sqrt(m) clears every observation by at least 13% and is not monotone in
        # m because q = round(theta m) moves.
        assert abs(ratio - Q.amplitude(q / m)) < 0.06 / math.sqrt(m), (m, ratio)
    # the bound has teeth: an amplitude wrong by 1% fails it at the largest size
    q = round(theta * 1601) if theta != 0.5 else 801
    ratio = Q.critical_slope_float(1601, q) / math.sqrt(1601)
    assert abs(ratio - 1.01 * Q.amplitude(q / 1601)) > 0.06 / math.sqrt(1601)


def test_the_amplitude_at_one_half_is_the_square_root_of_two_over_pi():
    assert Q.amplitude(0.5) == pytest.approx(math.sqrt(2 / math.pi), rel=1e-12)


def test_amplitude_needs_a_fraction_strictly_between_zero_and_one():
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError):
            Q.amplitude(bad)


def test_the_critical_point_converges_to_the_quorum_fraction_at_the_root_m_rate():
    # x_c - q/m ~ Phi^-1(theta) sqrt(theta (1-theta)/m): at most 0.25/sqrt(m) for theta in [.3, .7]
    for theta in (0.3, 0.5, 0.7):
        for m in (101, 401, 1601):
            q = round(theta * m)
            assert abs(Q.critical_point_float(m, q) - q / m) <= 0.5 / math.sqrt(m) + 1 / m


# --- found by the first coverage audit: public paths no test had run --------------------------------


def test_float_flow_follows_the_exact_flow_and_then_goes_where_the_phase_says():
    exact = Q.flow(5, 3, F(3, 10), 4)
    assert Q.flow_float(5, 3, 0.3, 4) == pytest.approx([float(x) for x in exact], rel=1e-12)
    assert Q.flow_float(5, 3, 0.3, 40)[-1] == 0.0  # healthy: doubly exponential decay underflows
    assert Q.flow_float(5, 3, 0.7, 40)[-1] == 1.0  # captured side
    with pytest.raises(ValueError, match="natural"):
        Q.flow_float(5, 3, 0.3, -1)
    with pytest.raises(ValueError, match="natural"):
        Q.flow(5, 3, F(3, 10), -1)


def test_a_constant_map_has_zero_slope():
    assert Q.tail_slope(5, 0, F(1, 2)) == 0 and Q.tail_slope(5, 6, F(1, 2)) == 0


def test_float_slope_agrees_with_the_exact_slope_and_handles_the_corners():
    for m, q, x in [(5, 3, 0.3), (7, 2, 0.6), (9, 8, 0.9)]:
        assert Q.tail_slope_float(m, q, x) == pytest.approx(float(Q.tail_slope(m, q, x)), rel=1e-12)
    assert Q.tail_slope_float(5, 1, 0.0) == 5.0 and Q.tail_slope_float(5, 5, 1.0) == 5.0
    assert Q.tail_slope_float(5, 3, 0.0) == 0.0 and Q.tail_slope_float(5, 3, 1.0) == 0.0
    assert Q.tail_slope_float(5, 0, 0.5) == 0.0 and Q.tail_slope_float(5, 6, 0.5) == 0.0


def test_the_float_critical_point_refuses_an_any_or_all_quorum():
    for q in (0, 1, 5, 6):
        with pytest.raises(ValueError, match="no interior fixed point"):
            Q.critical_point_float(5, q)


def test_stage_count_gives_up_rather_than_loop_for_ever():
    nearest = F(Q.critical_point_float(4, 2))  # the float closest to an irrational critical point
    assert Q.tail(4, 2, nearest) != nearest  # so it is not an exact fixed point ...
    with pytest.raises(RuntimeError, match="did not leave the band"):
        Q.stage_count(4, 2, nearest, (F(1, 10), F(9, 10)), cap=5)  # ... but it sits there for a while
