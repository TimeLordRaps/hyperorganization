"""Renewal: how long a body of m seats lasts when its seats are attacked and repaired.

The chain
    n is the number of compromised seats among m. A sound seat is compromised at rate
    `attack` (lambda). A compromised seat is repaired at rate `repair` (nu), but only while
    the body still holds, that is while n < q: the repair is performed by the body itself, so
    once q seats are compromised there is nobody left to perform it and the capture is
    permanent. The lifetime is the time from n = 0 to n = q.

    In n, the chain moves up at rate u_n = (m - n) lambda and down at d_n = n nu while n < q.
    It is a birth-death chain, reversible with weights pi_n = C(m, n) (lambda / nu)^n, so the
    expected time to climb from j to j+1 is a closed form:

        E_j[T_{j+1}] = (1 / (u_j pi_j)) * sum_{i <= j} pi_i,      lifetime = sum_{j < q} E_j[T_{j+1}].

    `mean_lifetime` evaluates that in exact rationals. `mean_lifetime_linear` solves the same
    quantity as a linear system by Gaussian elimination, which shares nothing with the closed
    form. `log_mean_lifetime` is the closed form in log space, for m in the thousands.

Three regimes (theta = q/m; rho = lambda / (lambda + nu), the stationary fraction of seats
that are compromised while the body holds)
    rho < theta   lifetime ~ C exp(m D(theta || rho)) / sqrt(m), D the Bernoulli
                  Kullback-Leibler divergence: redundancy buys exponential life.
    rho > theta   lifetime -> ln(rho / (rho - theta)) / (lambda + nu), whatever m is: the
                  body dies on the time scale of one repair, however many seats it has.
    rho = theta   lifetime ~ (1/2) ln(m) / (lambda + nu), slower than any power of m.
    The first is the standard large-deviation rate of a binomial tail (Cramer's theorem read
    on a reversible chain) and the second is the fluid limit of the mean path. Both are
    checked numerically against the exact lifetime; neither is proved in this package.

What this is a model of
    Seats fail independently and are repaired independently, at rates that do not depend on
    how many seats have failed except through the body holding. Real attacks are correlated
    (a common principal compromises several seats at once): that is `surprisal`'s business,
    and it lowers the effective m. Real repair is slower when more seats are down. Both make
    the true margin narrower than this model's, never wider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction


class Regime(str, Enum):
    """What the number of seats can and cannot buy."""

    EXPONENTIAL = "exponential"  # rho < theta: life grows exponentially in m
    CRITICAL = "critical"  # rho = theta: life grows like ln m
    IMMEDIATE = "immediate"  # rho > theta: life is O(1) whatever m is


def _rate(value: object, label: str) -> Fraction:
    if type(value) is bool or not isinstance(value, (int, float, Fraction)):
        raise TypeError(f"{label} must be an int, float or Fraction, not {value!r}")
    rate = Fraction(value)
    if rate < 0:
        raise ValueError(f"{label} must not be negative")
    return rate


@dataclass(frozen=True, slots=True)
class Chain:
    """m seats; q compromised seats capture the body; attack and repair rates per seat."""

    m: int
    q: int
    attack: Fraction
    repair: Fraction

    def __post_init__(self) -> None:
        if type(self.m) is not int or type(self.q) is not int:
            raise TypeError("m and q must be ints")
        if self.m < 1:
            raise ValueError("a body needs at least one seat")
        if not 1 <= self.q <= self.m:
            raise ValueError("the quorum q must satisfy 1 <= q <= m")
        object.__setattr__(self, "attack", _rate(self.attack, "attack"))
        object.__setattr__(self, "repair", _rate(self.repair, "repair"))
        if self.attack + self.repair == 0:
            raise ValueError("a chain needs a positive total rate")

    @property
    def total_rate(self) -> Fraction:
        return self.attack + self.repair

    @property
    def rho(self) -> Fraction:
        """The stationary fraction of compromised seats while the body holds."""
        return self.attack / self.total_rate

    @property
    def theta(self) -> Fraction:
        return Fraction(self.q, self.m)


def stationary_weights(chain: Chain) -> list[Fraction]:
    """pi_n proportional to C(m, n) (lambda / nu)^n for n = 0..m, with pi_0 = 1.

    This is the stationary measure of the chain with repair always on, which is the measure
    the closed form for the passage times is written in. It needs a positive repair rate.
    """
    if chain.repair == 0:
        raise ValueError("with no repair there is no stationary measure")
    ratio = chain.attack / chain.repair
    return [math.comb(chain.m, n) * ratio**n for n in range(chain.m + 1)]


def _check_start(chain: Chain, start: int) -> None:
    if type(start) is not int or start < 0:
        raise ValueError("start is a natural number of compromised seats")


def mean_lifetime(chain: Chain, start: int = 0) -> Fraction | float:
    """Expected time to reach q compromised seats from `start`, exactly.

    Zero if the body is already at or past its quorum. `math.inf` if there is no attack.
    """
    _check_start(chain, start)
    if start >= chain.q:
        return Fraction(0)
    if chain.attack == 0:
        return math.inf
    m, lam = chain.m, chain.attack
    if chain.repair == 0:
        return sum((Fraction(1) / ((m - j) * lam) for j in range(start, chain.q)), Fraction(0))
    weights = stationary_weights(chain)
    cumulative = sum(weights[:start], Fraction(0))
    total = Fraction(0)
    for j in range(start, chain.q):
        cumulative += weights[j]
        total += cumulative / ((m - j) * lam * weights[j])
    return total


def mean_lifetime_linear(chain: Chain, start: int = 0) -> Fraction | float:
    """The same expectation from the hitting-time equations, solved by exact elimination.

    h_n = 1/(u_n + d_n) + (u_n h_{n+1} + d_n h_{n-1}) / (u_n + d_n), with h_q = 0 and no
    downward move from 0, rearranged to (u_n + d_n) h_n - u_n h_{n+1} - d_n h_{n-1} = 1.
    """
    _check_start(chain, start)
    if start >= chain.q:
        return Fraction(0)
    if chain.attack == 0:
        return math.inf
    size, m = chain.q, chain.m
    matrix = [[Fraction(0)] * size for _ in range(size)]
    rhs = [Fraction(1)] * size
    for n in range(size):
        up, down = (m - n) * chain.attack, n * chain.repair
        matrix[n][n] = up + down
        if n + 1 < size:
            matrix[n][n + 1] -= up
        if n >= 1:
            matrix[n][n - 1] -= down
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
    return rhs[start]


def _log_add(a: float, b: float) -> float:
    """ln(e**a + e**b) for finite b; a may be -inf (an empty running sum)."""
    if a == -math.inf:
        return b
    high, low = max(a, b), min(a, b)
    return high + math.log1p(math.exp(low - high))


def log_mean_lifetime(chain: Chain) -> float:
    """ln of the lifetime from n = 0, in log space, for bodies too big for exact arithmetic."""
    if chain.attack == 0:
        return math.inf
    m, lam, q = chain.m, float(chain.attack), chain.q
    if chain.repair == 0:
        return math.log(sum(1.0 / ((m - j) * lam) for j in range(q)))
    log_ratio = math.log(lam / float(chain.repair))
    running = -math.inf
    terms = []
    for j in range(q):
        log_pi = (math.lgamma(m + 1) - math.lgamma(j + 1) - math.lgamma(m - j + 1)
                  + j * log_ratio)
        running = _log_add(running, log_pi)
        terms.append(running - log_pi - math.log((m - j) * lam))
    top = max(terms)
    return top + math.log(math.fsum(math.exp(t - top) for t in terms))


# ---------------------------------------------------------------------------
# The regimes.
# ---------------------------------------------------------------------------


def regime(theta: object, rho: object) -> Regime:
    """Which regime a quorum fraction theta and a compromised fraction rho are in."""
    t, r = Fraction(theta), Fraction(rho)
    if r < t:
        return Regime.EXPONENTIAL
    if r == t:
        return Regime.CRITICAL
    return Regime.IMMEDIATE


def regime_of(chain: Chain) -> Regime:
    return regime(chain.theta, chain.rho)


def min_repair_ratio(theta: object) -> Fraction:
    """nu / lambda must exceed (1 - theta) / theta for redundancy to buy exponential life."""
    t = Fraction(theta)
    if not 0 < t < 1:
        raise ValueError("theta lies strictly between 0 and 1")
    return (1 - t) / t


def exponent(theta: float, rho: float) -> float:
    """The growth rate of ln(lifetime) per seat: D(theta || rho) if theta > rho, else 0."""
    if not (0.0 < theta < 1.0 and 0.0 < rho < 1.0):
        raise ValueError("theta and rho lie strictly between 0 and 1")
    if theta <= rho:
        return 0.0
    return theta * math.log(theta / rho) + (1 - theta) * math.log((1 - theta) / (1 - rho))


def immediate_limit(theta: float, rho: float, total_rate: float) -> float:
    """The limit of the lifetime as m grows, when rho > theta: ln(rho / (rho - theta)) / rate."""
    if not (0.0 < theta < rho < 1.0):
        raise ValueError("the limit exists only for rho > theta, both strictly between 0 and 1")
    if total_rate <= 0:
        raise ValueError("the total rate is positive")
    return math.log(rho / (rho - theta)) / total_rate
