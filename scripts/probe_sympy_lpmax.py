"""Reproduce two hazards of sympy's linear-programming solver that made it unfit as a reference oracle (HO-025).

A diagnostic, not a test of this package: what it prints is evidence about the installed sympy (1.14.0 on the
machine of record, 2026-10-11) and may come out differently on another version, which would be good news.

  1. Non-termination that depends on the hash seed. The same linear program, the one for "is option 0 chosen
     by some strictly positive price" on a set of six options with a tie, is solved in a child process under
     several PYTHONHASHSEED values with a deadline. On the machine of record seeds 0 and 1 answered at once
     and seeds 6, 7, 12, 25 and 29 did not finish.
  2. Wrong points. Random small programs with one equality constraint are solved with a deadline each, and
     the returned point is checked against every constraint. On the machine of record, of 292 solved, 6
     violated the equality and 8 an inequality, and 6 more did not finish in 5 seconds. (Needs SIGALRM, so
     POSIX only; skipped elsewhere.)

    python scripts/probe_sympy_lpmax.py

Exit status 0 whatever it observes; 2 if sympy is not installed.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys

TIE = [(0, 2, 1, 0), (1, 1, 1, 0), (1, 1, 1, 0), (0, 0, 1, 2), (2, 0, 1, 0), (1, 0, 2, 0)]

CHILD = """
import sys
from sympy import Add, Symbol
from sympy.solvers.simplex import InfeasibleLPError, lpmax

points = {points!r}
p = points[0]
w = [Symbol("w%d" % k) for k in range(len(p))]
t = Symbol("t")
total = Add(*w)
constraints = [w[k] - t >= 0 for k in range(len(p))] + [total - 1 >= 0, 1 - total >= 0]
for q in points:
    if q != p:
        constraints.append(Add(*[w[k] * (p[k] - q[k]) for k in range(len(p))]) >= 0)
try:
    value, _ = lpmax(t, constraints)
    print("supported" if value > 0 else "not supported")
except InfeasibleLPError:
    print("infeasible")
"""


def hash_seed_hang(seeds: tuple[str, ...] = ("0", "1", "6", "7", "12", "25", "29"), deadline: int = 10) -> None:
    print(f"1. one program, {len(seeds)} hash seeds, {deadline} s each:")
    for seed in seeds:
        try:
            done = subprocess.run([sys.executable, "-c", CHILD.format(points=TIE)], capture_output=True, text=True,
                                  timeout=deadline, env={**os.environ, "PYTHONHASHSEED": seed})
            print(f"   PYTHONHASHSEED={seed:>2}: {done.stdout.strip() or done.stderr.strip()[-80:]}")
        except subprocess.TimeoutExpired:
            print(f"   PYTHONHASHSEED={seed:>2}: DID NOT FINISH in {deadline} s")


def wrong_points(trials: int = 400, deadline: int = 5) -> None:
    try:
        import signal
        signal.SIGALRM  # noqa: B018
    except AttributeError:
        print("2. skipped: this platform has no SIGALRM")
        return
    from sympy import Eq, Symbol
    from sympy.solvers.simplex import InfeasibleLPError, UnboundedLPError, lpmax

    class Deadline(Exception):
        pass

    def alarm(_signum, _frame):
        raise Deadline

    signal.signal(signal.SIGALRM, alarm)
    x = [Symbol(f"x{i}") for i in range(3)]
    rng = random.Random(5)
    stats = {"solved": 0, "violates the equality": 0, "violates an inequality": 0, "did not finish": 0,
             "infeasible": 0, "unbounded": 0}
    for _ in range(trials):
        rows = [[rng.randint(-3, 3) for _ in range(3)] + [rng.randint(0, 6)] for _ in range(rng.randint(2, 5))]
        eq = [rng.randint(0, 3) for _ in range(3)] + [rng.randint(1, 5)]
        objective = sum(rng.randint(-2, 3) * v for v in x)
        constraints = [sum(c * v for c, v in zip(r[:3], x)) <= r[3] for r in rows] + [v >= 0 for v in x]
        constraints.append(Eq(sum(c * v for c, v in zip(eq[:3], x)), eq[3]))
        signal.alarm(deadline)
        try:
            _value, solution = lpmax(objective, constraints)
        except InfeasibleLPError:
            stats["infeasible"] += 1
            continue
        except UnboundedLPError:
            stats["unbounded"] += 1
            continue
        except Deadline:
            stats["did not finish"] += 1
            continue
        finally:
            signal.alarm(0)
        stats["solved"] += 1
        point = {v: solution.get(v, 0) for v in x}
        if sum(c * point[v] for c, v in zip(eq[:3], x)) != eq[3]:
            stats["violates the equality"] += 1
        if not (all(sum(c * point[v] for c, v in zip(r[:3], x)) <= r[3] for r in rows) and all(point[v] >= 0 for v in x)):
            stats["violates an inequality"] += 1
    print(f"2. {trials} random programs with one equality, {deadline} s each: {stats}")


def main() -> int:
    try:
        import sympy
    except ImportError:
        print("UNKNOWN: sympy is not installed")
        return 2
    print(f"sympy {sympy.__version__}, Python {sys.version.split()[0]}")
    hash_seed_hang()
    wrong_points()
    return 0


if __name__ == "__main__":
    sys.exit(main())
