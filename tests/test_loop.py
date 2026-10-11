"""The closed loop of checks: a body whose overseers are bodies of its own kind.

L2 shows a finite governance system with no unchecked apex must close its checks into a
loop. This module asks what a loop costs and when it holds, in the mean-field model of
`quorum`: every body is compromised with the same probability p, and a body is overseen by
a quorum body of m seats, captured when q of them are. Two kinds of link between a body and
its overseers give two self-consistency equations. With pi the probability that a body is
attacked directly:

    COMMAND   the overseers can command the body, so capturing them captures it too:
                  p = pi + (1 - pi) T(p)             (compromise propagates)
    VETO      the overseers can only block it, so a direct attack succeeds only if they fail:
                  p = pi T(p)                        (compromise needs both)

These are one equation in two variables. Put s = 1 - p, the probability a body is sound:
the COMMAND loop becomes s = (1 - pi) T*(s) where T* is the dual quorum (m - q + 1), which
is the VETO loop with pi -> 1 - pi. `test_the_duality_identity_is_exact` pins that exactly.

A loop of depth L ends in an unchecked apex; the folded loop is the limit L -> infinity of
the same iteration started from p_0 = pi. That makes the iteration the ordinal picture of the
loop: finite stages, then the limit.

Oracles. Fixed points are counted exactly by Sturm and, independently, by a float scan.
Thresholds come from a tangency condition in the module and are checked here against a
direct max/min scan of the defining ratio, which shares no formula with it, and then
certified by an exact count of fixed points on either side.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction as F

import pytest

from hyperoptimization import loop as L
from hyperoptimization import poly as P
from hyperoptimization import quorum as Q

C, V = L.Link.COMMAND, L.Link.VETO


def rational_near(x, scale=10**9):
    return F(round(x * scale), scale)


# --- the two equations ---------------------------------------------------------------------------


def test_the_steps_are_the_two_equations():
    for m, q, pi, p in [(5, 3, F(1, 10), F(1, 3)), (7, 2, F(1, 5), F(2, 3)), (4, 4, F(1, 2), F(1, 2))]:
        t = Q.tail(m, q, p)
        assert L.step(C, m, q, pi, p) == pi + (1 - pi) * t
        assert L.step(V, m, q, pi, p) == pi * t


def test_the_duality_identity_is_exact():
    for m in range(1, 8):
        for q in range(1, m + 1):
            for pi in (F(1, 10), F(1, 3), F(7, 10)):
                for s in (F(0), F(1, 4), F(1, 2), F(5, 6), F(1)):
                    lhs = 1 - L.step(C, m, q, pi, 1 - s)
                    rhs = L.step(V, m, Q.dual(m, q), 1 - pi, s)
                    assert lhs == rhs, (m, q, pi, s)


@pytest.mark.parametrize("pi", [0, 1, -F(1, 2), F(3, 2), "x", True])
def test_the_direct_attack_probability_is_strictly_inside_the_unit_interval(pi):
    with pytest.raises((ValueError, TypeError)):
        L.step(C, 5, 3, pi, F(1, 2))


def test_step_refuses_an_unknown_link_and_a_probability_outside_the_interval():
    with pytest.raises(ValueError):
        L.step("command", 5, 3, F(1, 10), F(1, 2))
    with pytest.raises(ValueError):
        L.step(C, 5, 3, F(1, 10), F(3, 2))


def test_exact_and_float_towers_agree():
    for link in (C, V):
        exact = L.tower(link, 5, 3, F(1, 10), 4)
        approx = L.tower_float(link, 5, 3, 0.1, 4)
        assert len(exact) == len(approx) == 5
        assert exact[0] == F(1, 10)
        assert [float(x) for x in exact] == pytest.approx(approx, rel=1e-12)


def test_exact_tower_refuses_depths_that_would_not_fit():
    with pytest.raises(ValueError, match="doubly exponentially"):
        L.tower(C, 5, 3, F(1, 10), 30)


# --- fixed points, counted two ways ---------------------------------------------------------------


def scan_count(link, m, q, pi, cells=4000):
    """Interior sign changes of G(p) - p on a grid, in floats."""
    pi = float(pi)
    previous, changes = None, 0
    for i in range(1, cells):
        p = i / cells
        t = Q.tail_float(m, q, p)
        value = (pi + (1 - pi) * t if link is C else pi * t) - p
        if previous is not None and value != 0 and previous != 0 and (value > 0) != (previous > 0):
            changes += 1
        if value != 0:
            previous = value
    return changes


def test_sturm_count_agrees_with_an_independent_scan():
    rng = random.Random(41)
    checked = 0
    while checked < 120:
        m = rng.randint(2, 15)
        q = rng.randint(1, m)
        pi = F(rng.randint(1, 99), 100)
        for link in (C, V):
            near = abs(float(pi) - L.threshold(link, m, q)) < 1e-3  # too close to tell by a grid
            if near:
                continue
            assert L.count_fixed_points(link, m, q, pi) == scan_count(link, m, q, pi), (link, m, q, pi)
        checked += 1


def test_command_loop_has_a_bistable_window_and_veto_loop_the_mirror():
    # m = 5, q = 3: thresholds are about 0.165 (command) and 0.835 (veto).
    assert L.count_fixed_points(C, 5, 3, F(1, 10)) == 2
    assert L.count_fixed_points(C, 5, 3, F(1, 5)) == 0
    assert L.count_fixed_points(V, 5, 3, F(4, 5)) == 0
    assert L.count_fixed_points(V, 5, 3, F(9, 10)) == 2


def test_fixed_points_come_with_exact_brackets_and_a_stability():
    found = L.fixed_points(C, 5, 3, F(1, 10))
    assert len(found) == 2
    healthy, separatrix = found
    assert healthy.stable and not separatrix.stable
    assert 0.10 < float(healthy.lo) <= float(healthy.hi) < 0.12
    assert 0.42 < float(separatrix.lo) <= float(separatrix.hi) < 0.48
    for point in found:
        f_lo = L.step(C, 5, 3, F(1, 10), point.lo) - point.lo
        f_hi = L.step(C, 5, 3, F(1, 10), point.hi) - point.hi
        assert (f_lo > 0 > f_hi) if point.stable else (f_lo < 0 < f_hi)
        assert point.hi - point.lo <= F(1, 10**9)
    veto = L.fixed_points(V, 5, 3, F(9, 10))
    assert [p.stable for p in veto] == [False, True]


def test_no_interior_fixed_point_means_an_empty_list():
    assert L.fixed_points(C, 5, 3, F(1, 5)) == []


# --- thresholds ------------------------------------------------------------------------------------------


def scanned_threshold(link, m, q, cells=40000):
    """The defining extremum, by brute force: max (p - T)/(1 - T) for COMMAND, min p/T for VETO."""
    best = None
    for i in range(1, cells):
        p = i / cells
        t = Q.tail_float(m, q, p)
        if link is C:
            if t >= 1.0:
                continue
            value = (p - t) / (1 - t)
            best = value if best is None else max(best, value)
        else:
            if t <= 0.0:
                continue
            value = p / t
            best = value if best is None else min(best, value)
    return best


@pytest.mark.parametrize("link", [C, V])
@pytest.mark.parametrize("m,q", [(3, 2), (5, 3), (7, 2), (9, 6), (11, 6), (15, 4), (21, 11)])
def test_the_tangency_threshold_matches_the_scan_of_the_defining_ratio(link, m, q):
    assert L.threshold(link, m, q) == pytest.approx(scanned_threshold(link, m, q), abs=2e-6)


@pytest.mark.parametrize("link", [C, V])
@pytest.mark.parametrize("m,q", [(3, 2), (5, 3), (7, 2), (9, 6), (11, 6), (15, 4)])
def test_the_threshold_is_exactly_where_the_fixed_points_appear_or_vanish(link, m, q):
    t = L.threshold(link, m, q)
    below, above = rational_near(t * (1 - 1e-6)), rational_near(t * (1 + 1e-6))
    if link is C:
        assert L.count_fixed_points(C, m, q, below) == 2  # the healthy point and the separatrix
        assert L.count_fixed_points(C, m, q, above) == 0
    else:
        assert L.count_fixed_points(V, m, q, below) == 0
        assert L.count_fixed_points(V, m, q, above) == 2  # the separatrix and the captured point


def test_the_two_thresholds_of_a_self_dual_quorum_sum_to_exactly_one():
    for m in (3, 5, 7, 9, 11, 21, 51, 101):
        q = (m + 1) // 2
        assert L.threshold(C, m, q) + L.threshold(V, m, q) == pytest.approx(1.0, abs=1e-9)


def test_a_veto_only_check_outlasts_a_commanding_one_at_every_majority_size_we_try():
    for m in (3, 5, 7, 11, 21, 51, 101, 401):
        q = (m + 1) // 2
        assert L.threshold(V, m, q) > L.threshold(C, m, q)
    # at m = 5 the difference is a factor of five
    assert L.threshold(V, 5, 3) / L.threshold(C, 5, 3) > 5


def test_both_thresholds_close_on_the_quorum_fraction_as_the_body_grows():
    # (theta - pi*) sqrt(m) grows slowly, like sqrt(ln m): the Gaussian tail's tangent point sits
    # sqrt(2 ln sqrt(m)) standard deviations below theta. Observed 0.96 at m = 3 up to 1.39 at
    # m = 1601; the grid below stays inside 2 / sqrt(m).
    for theta in (0.3, 0.5, 0.7):
        for m in (101, 401, 1601):
            q = round(theta * m) if theta != 0.5 else (m + 1) // 2
            for link in (C, V):
                assert abs(L.threshold(link, m, q) - q / m) <= 2 / math.sqrt(m), (link, theta, m)
    gaps = [(0.5 - L.threshold(C, m, (m + 1) // 2)) * math.sqrt(m) for m in (101, 401, 1601, 6401)]
    assert all(a < b for a, b in zip(gaps, gaps[1:])), gaps  # the sqrt(ln m) drift is real


def test_the_any_and_all_quorums_have_the_thresholds_the_limits_say():
    for m in (2, 5, 12):
        assert L.threshold(V, m, 1) == pytest.approx(1 / m)  # one compromised overseer suffices
        assert L.threshold(V, m, m) == pytest.approx(1.0)  # all of them: never
        assert L.threshold(C, m, 1) == pytest.approx(0.0)
        assert L.threshold(C, m, m) == pytest.approx(1 - 1 / m)


def test_capture_gain_is_the_largest_ratio_of_capture_to_compromise():
    for m, q in [(5, 3), (9, 4), (21, 11)]:
        grid = max(Q.tail_float(m, q, i / 20000) / (i / 20000) for i in range(1, 20000))
        assert L.capture_gain(m, q) == pytest.approx(grid, rel=1e-8)
        assert L.capture_gain(m, q) > 1


# --- outcome, and the tower that reaches it ------------------------------------------------------------------


def iterate_to_rest(link, m, q, pi, steps=60000):
    p = pi
    for _ in range(steps):
        nxt = pi + (1 - pi) * Q.tail_float(m, q, p) if link is C else pi * Q.tail_float(m, q, p)
        if nxt == p:
            break
        p = nxt
    return p


def test_outcome_is_exact_and_agrees_with_long_iteration_away_from_the_threshold():
    rng = random.Random(42)
    checked = 0
    while checked < 150:
        m = rng.randint(2, 25)
        q = rng.randint(1, m)
        link = rng.choice([C, V])
        pi = F(rng.randint(1, 99), 100)
        t = L.threshold(link, m, q)
        if abs(float(pi) - t) < 0.02 * max(t, 0.05):
            continue
        settled = iterate_to_rest(link, m, q, float(pi))
        # a healthy COMMAND loop can rest above 1/2 (q = m = 2 at pi = .45 rests near .82), so the
        # test is where it rests: at 1 (captured) or not; a VETO loop is healthy only at 0
        collapsed = settled > 1 - 1e-9 if link is C else settled > 1e-9
        want = L.Outcome.COLLAPSED if collapsed else L.Outcome.HEALTHY
        assert L.outcome(link, m, q, pi) is want, (link, m, q, pi, settled)
        checked += 1


def test_the_tower_moves_monotonically_toward_its_limit():
    for pi in (F(1, 20), F(1, 10), F(3, 20)):  # command, below threshold: rises to the healthy point
        orbit = L.tower(C, 5, 3, pi, 5)
        assert all(a < b for a, b in zip(orbit, orbit[1:]))
        assert orbit[-1] <= L.fixed_points(C, 5, 3, pi)[0].hi
    for pi in (F(1, 2), F(7, 10), F(4, 5)):  # veto, below threshold: falls to 0
        orbit = L.tower(V, 5, 3, pi, 5)
        assert all(a > b for a, b in zip(orbit, orbit[1:]))


def test_a_finite_hierarchy_is_never_worse_than_the_loop_it_folds_into_when_commanded():
    # depth-L oversight with an unchecked apex has apex compromise p_L <= the loop's p_inf, and the
    # gap closes geometrically: the price of obeying the no-apex rule is small while the loop is healthy.
    pi = F(1, 10)
    limit = L.limit(C, 5, 3, pi)
    last = None
    for depth in range(1, 8):
        price = L.closing_price(C, 5, 3, pi, depth)
        assert price >= 0 and limit - L.tower_float(C, 5, 3, float(pi), depth)[-1] == pytest.approx(price)
        if last is not None and last > 1e-14:
            assert price < last
        last = price
    assert L.closing_price(C, 5, 3, pi, 1) < 0.02  # one level of oversight already gets within 2%


def test_the_closing_price_shrinks_at_the_slope_of_the_map_at_the_limit():
    pi = F(1, 10)
    rate = L.contraction_rate(C, 5, 3, pi)
    assert 0 < rate < 1
    a, b = L.closing_price(C, 5, 3, pi, 3), L.closing_price(C, 5, 3, pi, 4)
    assert b / a == pytest.approx(rate, rel=0.15)


def test_a_veto_tower_converges_faster_than_geometrically_when_q_is_at_least_two():
    assert L.contraction_rate(V, 5, 3, F(1, 2)) == pytest.approx(0.0, abs=1e-12)
    # q = 1 is the exception: the slope of the map at 0 is pi m = 0.25, a plain geometric rate
    assert L.contraction_rate(V, 5, 1, F(1, 20)) == pytest.approx(0.25, rel=1e-6)


def test_limit_refuses_to_report_a_value_it_did_not_reach():
    just_past = rational_near(L.threshold(C, 51, 26) * (1 + 1e-6))
    with pytest.raises(RuntimeError, match="did not converge"):
        L.limit(C, 51, 26, just_past, cap=50)


# --- the ghost: crossing time diverges at the saddle-node --------------------------------------------------------


def test_just_past_the_threshold_the_collapse_is_slow_and_the_delay_diverges_as_inverse_root():
    m, q = 51, 26
    t = L.threshold(C, m, q)
    levels = {eps: L.levels_to_cross(C, m, q, t * (1 + eps), 0.5) for eps in (1e-3, 1e-4, 1e-5, 1e-6)}
    assert all(isinstance(v, int) for v in levels.values())
    assert levels[1e-3] < levels[1e-4] < levels[1e-5] < levels[1e-6]
    for fine, coarse in [(1e-5, 1e-3), (1e-6, 1e-4)]:  # 100x closer, about 10x longer
        assert 9.0 <= levels[fine] / levels[coarse] <= 11.5, levels


def test_below_the_threshold_the_tower_never_crosses_and_the_function_says_so():
    m, q = 51, 26
    t = L.threshold(C, m, q)
    assert L.levels_to_cross(C, m, q, t * 0.9, 0.5) is None
    assert L.levels_to_cross(V, m, q, L.threshold(V, m, q) * 0.9, 0.001) is not None  # veto falls


def test_levels_to_cross_counts_from_the_start_and_agrees_with_the_tower():
    orbit = L.tower_float(C, 5, 3, 0.2, 12)  # above threshold: it collapses (at level 7)
    first = next(k for k, p in enumerate(orbit) if p > 0.5)
    assert L.levels_to_cross(C, 5, 3, 0.2, 0.5) == first


# --- found by the first coverage audit ------------------------------------------------------------------


def test_a_one_seat_overseer_who_commands_is_an_apex_in_disguise():
    for pi in (F(1, 100), F(1, 2), F(99, 100)):
        assert L.count_fixed_points(C, 1, 1, pi) == 0 and L.fixed_points(C, 1, 1, pi) == []
        assert L.count_fixed_points(V, 1, 1, pi) == 0 and L.fixed_points(V, 1, 1, pi) == []
        # compromise travels up a single chain of command to certainty; one who can only block
        # leaves p = pi p, so nothing is ever compromised
        assert L.outcome(C, 1, 1, pi) is L.Outcome.COLLAPSED
        assert L.outcome(V, 1, 1, pi) is L.Outcome.HEALTHY
    assert L.threshold(C, 1, 1) == pytest.approx(0.0) and L.threshold(V, 1, 1) == pytest.approx(1.0)
    assert L.limit(C, 1, 1, F(1, 2)) == pytest.approx(1.0)


def test_exact_thresholds_at_a_three_seat_majority_are_where_the_roots_merge():
    # T = 3c^2 - 2c^3 so T/c = 3c - 2c^2, largest at c = 3/4 with value 9/8: veto 8/9, and by
    # duality command 1/9. At exactly those pressures the two fixed points are one double root.
    assert L.threshold(V, 3, 2) == pytest.approx(8 / 9, abs=1e-12)
    assert L.threshold(C, 3, 2) == pytest.approx(1 / 9, abs=1e-12)
    [point] = L.fixed_points(V, 3, 2, F(8, 9))
    assert point.kind is L.Kind.SEMISTABLE and not point.stable
    assert point.lo <= F(3, 4) <= point.hi and point.hi - point.lo <= F(1, 10**9)
    [point] = L.fixed_points(C, 3, 2, F(1, 9))
    assert point.kind is L.Kind.SEMISTABLE
    assert point.lo <= F(1, 4) <= point.hi and point.hi - point.lo <= F(1, 10**9)
    hair = F(1, 10**6)
    assert L.count_fixed_points(V, 3, 2, F(8, 9) - hair) == 0
    assert L.count_fixed_points(V, 3, 2, F(8, 9)) == 1
    assert L.count_fixed_points(V, 3, 2, F(8, 9) + hair) == 2
    assert L.count_fixed_points(C, 3, 2, F(1, 9) - hair) == 2
    assert L.count_fixed_points(C, 3, 2, F(1, 9)) == 1
    assert L.count_fixed_points(C, 3, 2, F(1, 9) + hair) == 0


@pytest.mark.parametrize("m", [2, 3, 4, 7, 12])
def test_when_every_seat_must_fall_the_command_loop_has_a_double_end_root_exactly_at_its_threshold(m):
    # COMMAND with q = m: G(p) - p = pi + (1 - pi) p^m - p, and p = 1 is always a root. Its slope
    # there is (1 - pi) m - 1 and its curvature (1 - pi) m (m - 1). At pi = 1 - 1/m the slope is
    # zero and the curvature is m - 1 > 0, so the end root is exactly double: the interior root
    # that lived just below has been swallowed by it. Dividing the end root out once would leave
    # it standing and make the interior count wrong.
    pi = 1 - F(1, m)
    f = L.fixed_point_poly(C, m, m, pi)
    for x in (F(0), F(1, 3), F(1, 2), F(2, 3), F(1), F(3, 2), F(-2)):
        assert P.peval(f, x) == pi + (1 - pi) * x**m - x
    assert (1 - pi) * m - 1 == 0 and (1 - pi) * m * (m - 1) == m - 1
    hair = F(1, 10**6)
    assert L.count_fixed_points(C, m, m, pi - hair) == 1
    assert L.count_fixed_points(C, m, m, pi) == 0
    assert L.count_fixed_points(C, m, m, pi + hair) == 0
    assert L.fixed_points(C, m, m, pi) == []
    assert L.outcome(C, m, m, pi - hair) is L.Outcome.HEALTHY
    assert L.outcome(C, m, m, pi) is L.Outcome.COLLAPSED
    assert L.threshold(C, m, m) == pytest.approx(float(pi), abs=1e-12)


@pytest.mark.parametrize("m", [2, 3, 4, 7, 12])
def test_when_one_seat_suffices_the_veto_loop_has_a_double_end_root_exactly_at_its_threshold(m):
    # VETO with q = 1: G(p) - p = pi (1 - (1 - p)^m) - p, and p = 0 is always a root. Its slope
    # there is pi m - 1 and its curvature -pi m (m - 1); at pi = 1/m the slope vanishes and the
    # curvature is -(m - 1) < 0, an exactly double root, with no interior root until pi passes 1/m.
    pi = F(1, m)
    f = L.fixed_point_poly(V, m, 1, pi)
    for x in (F(0), F(1, 3), F(1, 2), F(2, 3), F(1), F(3, 2), F(-2)):
        assert P.peval(f, x) == pi * (1 - (1 - x) ** m) - x
    assert pi * m - 1 == 0 and -pi * m * (m - 1) == -(m - 1)
    hair = F(1, 10**6)
    assert L.count_fixed_points(V, m, 1, pi - hair) == 0
    assert L.count_fixed_points(V, m, 1, pi) == 0
    assert L.count_fixed_points(V, m, 1, pi + hair) == 1
    assert L.fixed_points(V, m, 1, pi) == []
    assert L.outcome(V, m, 1, pi) is L.Outcome.HEALTHY
    assert L.outcome(V, m, 1, pi + hair) is L.Outcome.COLLAPSED
    assert L.threshold(V, m, 1) == pytest.approx(float(pi), abs=1e-12)


def test_capture_gain_refuses_a_quorum_that_is_no_quorum():
    for q in (0, 6):
        with pytest.raises(ValueError):
            L.capture_gain(5, q)


def test_levels_to_cross_gives_up_at_its_cap():
    slow = L.threshold(C, 51, 26) * 0.9  # converging, not crossing, and not at rest after 3 levels
    with pytest.raises(RuntimeError, match="no crossing and no rest"):
        L.levels_to_cross(C, 51, 26, slow, 0.5, cap=3)


def test_depth_must_be_a_natural_number():
    with pytest.raises(ValueError, match="natural"):
        L.tower(C, 5, 3, F(1, 10), -1)
    with pytest.raises(ValueError, match="natural"):
        L.tower_float(C, 5, 3, 0.1, -1)
