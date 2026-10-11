"""Renewal: how long a body of m seats lasts when seats are attacked and repaired.

The chain. n is the number of compromised seats. A sound seat is compromised at rate
`attack` (lambda); a compromised seat is repaired at rate `repair` (nu) but ONLY while the
body still holds, that is while n < q. Repair is performed by the body itself, so once q
seats are compromised nothing can repair them and the capture is permanent. The lifetime is
the time from n = 0 to n = q.

Three oracles check the same quantity by different mathematics:
    1. the closed form for a birth-death chain, evaluated in exact rationals;
    2. the linear system for expected hitting times, solved by exact Gaussian elimination;
    3. the uniformization series, in floats (the chain is turned into a discrete-time chain
       and the expected lifetime is a sum of survival probabilities).
A mistake in the closed form that the linear system happens to share would have to also fool
a power series that shares no code with either.

Three regimes, with theta = q/m and rho = lambda/(lambda + nu), the stationary fraction of
seats that are compromised while the body holds:
    rho < theta   lifetime grows like exp(m D(theta || rho)) / sqrt(m), D the Bernoulli
                  Kullback-Leibler divergence: redundancy buys exponential life;
    rho > theta   lifetime tends to ln(rho / (rho - theta)) / (lambda + nu), whatever m is:
                  you cannot out-vote decay;
    rho = theta   lifetime grows like (1/2) ln(m) / (lambda + nu), slower than any power.
The two limits are checked numerically below. They are the standard large-deviation and
fluid limits of this chain; they are not proved in this package.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction as F

import pytest

from hyperoptimization import renewal as R


def chain_of(m, q, attack, repair):
    return R.Chain(m, q, F(attack), F(repair))


# --- construction ------------------------------------------------------------------------------


def test_a_chain_reads_its_rates_exactly_and_exposes_the_two_fractions():
    c = R.Chain(10, 4, 0.5, F(3, 2))
    assert c.attack == F(1, 2) and c.repair == F(3, 2)
    assert c.rho == F(1, 4) and c.theta == F(2, 5) and c.total_rate == 2


@pytest.mark.parametrize("args", [(0, 1, 1, 1), (3, 0, 1, 1), (3, 4, 1, 1), (3, 2, -1, 1), (3, 2, 1, -1),
                                  (3, 2, 0, 0), (3.0, 2, 1, 1), (3, 2.0, 1, 1), (True, 1, 1, 1),
                                  (3, 2, "fast", 1)])
def test_a_chain_refuses_what_is_not_a_body_with_rates(args):
    with pytest.raises((ValueError, TypeError)):
        R.Chain(*args)


# --- the three oracles --------------------------------------------------------------------------


def random_chain(rng, max_m=9):
    m = rng.randint(1, max_m)
    q = rng.randint(1, m)
    return chain_of(m, q, F(rng.randint(1, 9), rng.randint(1, 5)), F(rng.randint(1, 9), rng.randint(1, 5)))


def test_closed_form_and_linear_system_agree_exactly():
    rng = random.Random(31)
    for _ in range(300):
        c = random_chain(rng)
        for start in range(c.q):
            assert R.mean_lifetime(c, start) == R.mean_lifetime_linear(c, start), (c, start)


def uniformized_lifetime(c, start, tolerance=1e-13):
    """E[T] by the uniformization series; shares no code with the module."""
    lam, nu, m, q = float(c.attack), float(c.repair), c.m, c.q
    rate = m * (lam + nu)  # at least every state's total jump rate
    mass = [0.0] * q
    mass[start] = 1.0
    total = 0.0
    for _ in range(2_000_000):
        survivors = sum(mass)
        if survivors < tolerance:
            break
        total += survivors
        nxt = [0.0] * q
        for n, w in enumerate(mass):
            if w == 0.0:
                continue
            up, down = (m - n) * lam, n * nu
            stay = rate - up - down
            nxt[n] += w * stay / rate
            if n + 1 < q:
                nxt[n + 1] += w * up / rate
            if n >= 1:
                nxt[n - 1] += w * down / rate
        mass = nxt
    return total / rate


def test_the_uniformization_series_reproduces_the_exact_lifetime():
    rng = random.Random(32)
    checked = 0
    while checked < 60:
        c = random_chain(rng, 7)
        exact = R.mean_lifetime(c, 0)
        if exact > 40:  # the series is slow on very long lives
            continue
        assert uniformized_lifetime(c, 0) == pytest.approx(float(exact), rel=1e-9), c
        checked += 1


def test_the_series_oracle_can_tell_a_wrong_chain_from_the_right_one():
    c = chain_of(5, 3, 1, 2)
    wrong = chain_of(5, 3, 2, 1)
    assert uniformized_lifetime(c, 0) != pytest.approx(float(R.mean_lifetime(wrong, 0)), rel=1e-3)


# --- closed forms and exact identities ----------------------------------------------------------


def test_with_no_repair_the_lifetime_is_a_harmonic_difference():
    for m in range(1, 9):
        for q in range(1, m + 1):
            c = chain_of(m, q, 2, 0)
            want = sum(F(1, (m - j) * 2) for j in range(q))
            assert R.mean_lifetime(c, 0) == want
            assert R.mean_lifetime_linear(c, 0) == want


def test_any_single_compromise_captures_when_q_is_one():
    for m in range(1, 8):
        for repair in (0, 1, 5):
            assert R.mean_lifetime(chain_of(m, 1, 3, repair), 0) == F(1, 3 * m)


def test_a_body_already_at_or_past_its_quorum_has_no_life_left():
    c = chain_of(5, 3, 1, 1)
    assert R.mean_lifetime(c, 3) == 0 and R.mean_lifetime(c, 5) == 0
    with pytest.raises(ValueError):
        R.mean_lifetime(c, -1)


def test_no_attack_means_an_unending_life():
    assert R.mean_lifetime(chain_of(5, 3, 0, 1), 0) == math.inf
    assert R.log_mean_lifetime(chain_of(5, 3, 0, 1)) == math.inf


def test_time_scales_inversely_with_the_rates():
    rng = random.Random(33)
    for _ in range(80):
        c = random_chain(rng)
        k = F(rng.randint(2, 9))
        faster = R.Chain(c.m, c.q, c.attack * k, c.repair * k)
        assert R.mean_lifetime(faster, 0) == R.mean_lifetime(c, 0) / k


def test_lifetime_falls_with_attack_rises_with_repair_and_falls_with_more_seats_at_fixed_q():
    for m, q in [(6, 3), (8, 4), (7, 2)]:
        slower = chain_of(m, q, 1, 2)
        assert R.mean_lifetime(chain_of(m, q, F(3, 2), 2), 0) < R.mean_lifetime(slower, 0)
        assert R.mean_lifetime(chain_of(m, q, 1, 3), 0) > R.mean_lifetime(slower, 0)
    for q in (2, 3):
        lives = [R.mean_lifetime(chain_of(m, q, 1, 2), 0) for m in range(q, q + 6)]
        assert all(a > b for a, b in zip(lives, lives[1:])), lives


def test_the_stationary_weights_satisfy_detailed_balance():
    c = chain_of(9, 5, 2, 3)
    weights = R.stationary_weights(c)
    assert len(weights) == c.m + 1
    for n in range(c.m):
        up = (c.m - n) * c.attack
        down = (n + 1) * c.repair
        assert weights[n] * up == weights[n + 1] * down
    assert weights[0] == 1


def test_the_log_form_agrees_with_the_exact_form_until_the_exact_form_is_slow():
    rng = random.Random(34)
    for _ in range(40):
        m = rng.randint(2, 120)
        q = rng.randint(1, m)
        c = chain_of(m, q, F(rng.randint(1, 9), 4), F(rng.randint(1, 9), 4))
        exact = R.mean_lifetime(c, 0)
        assert R.log_mean_lifetime(c) == pytest.approx(math.log(exact), rel=1e-9, abs=1e-9), (m, q)


# --- the three regimes --------------------------------------------------------------------------


def test_regime_is_decided_by_comparing_rho_with_theta():
    assert R.regime(F(1, 2), F(1, 5)) is R.Regime.EXPONENTIAL
    assert R.regime(F(1, 2), F(1, 2)) is R.Regime.CRITICAL
    assert R.regime(F(1, 2), F(4, 5)) is R.Regime.IMMEDIATE
    c = chain_of(100, 50, 1, 4)  # rho = 1/5
    assert R.regime_of(c) is R.Regime.EXPONENTIAL


def test_the_minimum_repair_ratio_is_the_boundary_between_the_regimes():
    for theta in (F(1, 4), F(1, 2), F(3, 5), F(9, 10)):  # 1000 theta is an integer
        ratio = R.min_repair_ratio(theta)
        assert ratio == (1 - theta) / theta
        on_the_line = R.Chain(1000, int(theta * 1000), 1, ratio)
        assert R.regime_of(on_the_line) is R.Regime.CRITICAL
        above = R.Chain(1000, int(theta * 1000), 1, ratio * F(11, 10))
        below = R.Chain(1000, int(theta * 1000), 1, ratio * F(9, 10))
        assert R.regime_of(above) is R.Regime.EXPONENTIAL
        assert R.regime_of(below) is R.Regime.IMMEDIATE


def test_the_exponent_is_the_bernoulli_kullback_leibler_divergence():
    assert R.exponent(0.5, 0.5) == 0.0
    assert R.exponent(0.5, 0.2) == pytest.approx(0.5 * math.log(0.5 / 0.2) + 0.5 * math.log(0.5 / 0.8))
    assert R.exponent(0.5, 0.2) > 0
    assert R.exponent(0.7, 0.3) == pytest.approx(0.7 * math.log(0.7 / 0.3) + 0.3 * math.log(0.3 / 0.7))
    with pytest.raises(ValueError):
        R.exponent(0.0, 0.5)
    with pytest.raises(ValueError):
        R.exponent(0.5, 1.0)


@pytest.mark.parametrize("theta,rho", [(0.5, 0.2), (0.7, 0.3), (0.9, 0.5)])
def test_redundancy_buys_exponential_life_with_the_kullback_leibler_rate(theta, rho):
    lam, nu = F(rho).limit_denominator(1000), 1 - F(rho).limit_denominator(1000)
    D = R.exponent(theta, float(lam))

    def residual(m, rate=D):
        c = R.Chain(m, round(theta * m), lam, nu)
        return R.log_mean_lifetime(c) - m * rate + 0.5 * math.log(m)

    # ln E[T] = m D - (1/2) ln m + const + o(1): the residual stops moving
    assert abs(residual(3200) - residual(1600)) < 0.003, (residual(1600), residual(3200))
    # and it is the right D that makes it stop: 2% off and the residual runs away
    assert abs(residual(3200, 1.02 * D) - residual(1600, 1.02 * D)) > 10 * 0.003


@pytest.mark.parametrize("theta,rho", [(0.5, 0.8), (0.5, 0.7), (0.3, 0.6)])
def test_past_the_margin_no_number_of_seats_helps(theta, rho):
    lam, nu = F(rho).limit_denominator(1000), 1 - F(rho).limit_denominator(1000)
    want = R.immediate_limit(theta, float(lam), float(lam + nu))
    lives = []
    for m in (400, 1600, 3200):
        c = R.Chain(m, round(theta * m), lam, nu)
        lives.append(math.exp(R.log_mean_lifetime(c)))
    assert lives[-1] == pytest.approx(want, rel=6e-3), (lives, want)
    assert abs(lives[-1] - want) <= abs(lives[0] - want) + 1e-12  # still converging
    assert lives[-1] < 1.01 * lives[-2] + 1e-9  # and not growing with m


def test_the_fluid_limit_is_the_time_the_mean_path_takes_to_reach_the_quorum():
    # dn/dt = lam (m - n) - nu n gives n(t) = m rho (1 - exp(-(lam + nu) t)); it reaches theta m
    # at t = ln(rho / (rho - theta)) / (lam + nu).
    for theta, rho, rate in [(0.5, 0.8, 1.0), (0.3, 0.6, 2.0), (0.2, 0.9, 0.5)]:
        t = R.immediate_limit(theta, rho, rate)
        n_over_m = rho * (1 - math.exp(-rate * t))
        assert n_over_m == pytest.approx(theta, rel=1e-12)
    with pytest.raises(ValueError, match="rho > theta"):
        R.immediate_limit(0.5, 0.5, 1.0)


def test_at_the_margin_life_grows_slower_than_any_power_of_the_number_of_seats():
    lam = nu = F(1, 2)
    lives = {m: math.exp(R.log_mean_lifetime(R.Chain(m, m // 2, lam, nu))) for m in (100, 400, 1600, 6400)}
    assert lives[6400] / lives[100] < 2.2  # a 64-fold increase in m buys at most a doubling
    assert all(lives[m] <= math.log(m) / float(lam + nu) for m in lives)
    assert all(a < b for a, b in zip(list(lives.values()), list(lives.values())[1:]))


def test_the_viability_boundary_is_sharp_at_a_large_body():
    theta, m = F(1, 2), 3200
    for ratio, expect_long in [(F(13, 10), True), (F(7, 10), False)]:
        c = R.Chain(m, int(theta * m), 1, ratio)
        life = R.log_mean_lifetime(c)
        if expect_long:
            assert life > 20  # e^20 lifetimes
        else:
            assert life < math.log(3 / float(c.total_rate))


# --- found by the first coverage audit ------------------------------------------------------------------


def test_a_chain_with_no_repair_has_no_stationary_measure():
    with pytest.raises(ValueError, match="no stationary measure"):
        R.stationary_weights(chain_of(5, 3, 1, 0))


def test_the_linear_solver_agrees_on_the_edge_cases_too():
    c = chain_of(5, 3, 1, 1)
    assert R.mean_lifetime_linear(c, 3) == R.mean_lifetime(c, 3) == 0
    assert R.mean_lifetime_linear(chain_of(5, 3, 0, 1), 0) == math.inf
    with pytest.raises(ValueError):
        R.mean_lifetime_linear(c, -1)


def test_the_log_form_covers_a_chain_with_no_repair():
    for m, q in [(5, 3), (9, 9), (40, 7)]:
        c = chain_of(m, q, 2, 0)
        assert R.log_mean_lifetime(c) == pytest.approx(math.log(float(R.mean_lifetime(c, 0))), rel=1e-12)


@pytest.mark.parametrize("theta", [0, 1, -F(1, 2), F(3, 2)])
def test_the_minimum_repair_ratio_needs_a_fraction_strictly_inside_the_unit_interval(theta):
    with pytest.raises(ValueError, match="strictly between"):
        R.min_repair_ratio(theta)


def test_the_immediate_limit_needs_a_positive_total_rate():
    for rate in (0.0, -1.0):
        with pytest.raises(ValueError, match="positive"):
            R.immediate_limit(0.3, 0.6, rate)


def test_at_the_margin_of_a_balanced_body_life_is_half_a_log_plus_a_constant():
    lam = nu = F(1, 2)  # total rate 1, rho = theta = 1/2

    def life(m):
        return math.exp(R.log_mean_lifetime(R.Chain(m, m // 2, lam, nu)))

    for m in (400, 1600, 6400):  # the coefficient: quadrupling the seats adds (1/2) ln 4
        assert life(4 * m) - life(m) == pytest.approx(0.5 * math.log(4), abs=2e-3)
    euler = 0.5772156649015329  # and the constant, observed to four digits for m up to 25600
    for m in (400, 1600, 6400):
        assert life(m) - 0.5 * math.log(m) == pytest.approx((math.log(2) + euler) / 2, abs=1e-3)
