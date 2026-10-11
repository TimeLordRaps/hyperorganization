"""Claims about governance that this package states, checked against hyperethics' own code.

Not part of the test suite, because it imports hyperethics, which this package never does. Run it
beside a hyperethics checkout:

    PYTHONPATH=<hyperethics>/src:<hyperethics>/tests python scripts/verify_against_hyperethics.py

and it prints what it checked and exits non-zero on the first disagreement. It checks two claims.

BREADTH.  gate_cost <= gate_size * (dearest finite price). A gate that costs S bits to pass needs
          at least S / (dearest price) independent roots, so concentrating a stake forces a wider
          gate. Checked on hyperethics' own random governance systems.

SYNTHESIS. Given purchasable parties with prices, a collusion bound k and a stake S, the smallest
          system that obeys the rule and costs at least S to take is a clique of mutual vetoes
          among the dearest parties, of size max(k + 1, m*) where m* is the fewest of the dearest
          prices that sum to S. Checked against an exhaustive search over every subset that could be
          the clique, building each as a hyperethics Governance and asking hyperethics whether it
          obeys the rule and what the BUYER attacker must spend.
"""

from __future__ import annotations

import itertools
import math
import random
import sys

from hyperethics.capture import Attacker, gate_cost, topple_outlay
from hyperethics.corruption import clauses_failed
from hyperethics.governance import Governance, Party, Power, gate_size
from randomsys import random_governance


def breadth(samples: int = 400) -> int:
    rng = random.Random(20261012)
    checked = 0
    for _ in range(samples):
        system = random_governance(rng)
        finite = [p.price for p in system.parties if p.price != math.inf]
        if not finite or len(finite) != len(system.parties):
            continue
        top = max(finite)
        for power in system.powers:
            cost, size = gate_cost(system, power.name), gate_size(system, power.name)
            assert cost <= size * top, (system.name, power.name, cost, size, top)
            checked += 1
    return checked


def clique(prices: list[int], k: int) -> Governance:
    names = [f"P{i}" for i in range(len(prices))]
    everyone = tuple(names)
    return Governance(
        "clique",
        tuple(Party(n, price=p) for n, p in zip(names, prices)),
        tuple(
            Power(f"w{i}", names[i], reach=everyone, inhabited=everyone,
                  vetoes=tuple(f"w{j}" for j in range(len(names)) if j != i), ex_ante=True,
                  stake=0, term=2)
            for i in range(len(names))
        ),
        collusion_bound=k, amendment="w0",
    )


def formula(prices: list[int], k: int, stake: int) -> int | None:
    ordered = sorted(prices, reverse=True)
    for n in range(1, len(ordered) + 1):
        if sum(ordered[:n]) >= stake:
            return max(k + 1, n) if max(k + 1, n) <= len(ordered) else None
    return None


def exhaustive(prices: list[int], k: int, stake: int) -> int | None:
    best = None
    for size in range(1, len(prices) + 1):
        for chosen in itertools.combinations(sorted(prices, reverse=True), size):
            system = clique(list(chosen), k)
            if clauses_failed(system):
                continue
            if topple_outlay(system, Attacker.BUYER).amount >= stake:
                best = size
                break
        if best is not None:
            break
    return best


def synthesis(samples: int = 120) -> int:
    rng = random.Random(20261013)
    checked = 0
    for _ in range(samples):
        n = rng.randint(2, 7)
        prices = [rng.randint(1, 9) for _ in range(n)]
        k = rng.randint(1, 3)
        stake = rng.randint(1, sum(prices) + 3)
        want = exhaustive(prices, k, stake)
        got = formula(prices, k, stake)
        assert want == got, (prices, k, stake, want, got)
        checked += 1
    return checked


if __name__ == "__main__":
    print(f"breadth:   gate_cost <= gate_size * dearest price on {breadth()} gates")
    print(f"synthesis: smallest rule-obeying gate of cost >= stake is max(k+1, m*) on {synthesis()} instances")
    sys.exit(0)
