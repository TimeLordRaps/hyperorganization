"""Check top trading cycles against its definitions on EVERY strict preference profile of up to four agents.

For n agents there are (n!)^n strict profiles (4, 216 and 331,776 for n = 2, 3, 4). The endowment can be taken to
be the identity (agent i owns good i): any other endowment is the identity after renaming the goods, which maps the
set of all profiles onto itself, so this covers every market of n agents up to the names of its goods.

For each profile it checks that the outcome is individually rational, Pareto efficient and in the strict core, that
it is the ONLY allocation in the strict core, and that no agent has a profitable misreport. It takes about
ten minutes of processor time for n = 4 (2 minutes 40 seconds on four cores, split over the cores available; on
2026-10-11 all 331,776 profiles passed, with trade in 255,120) and is not part of the test suite, whose version of
the same check covers n = 2 and 3 exhaustively and n = 4 to 6 by seeded sample.

    python scripts/check_rival_exhaustive.py [N]          # N = 2, 3 or 4 (default 4)

Exit status 0: every profile met every definition. 1: a profile did not (printed). It prints one line per chunk.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import os
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hyperoptimization import rival  # noqa: E402  (after the path is set)


def chunk(n: int, first: int) -> tuple[int, int, list]:
    """All profiles whose first agent ranks the goods as the `first`-th ranking; returns (checked, trades, failures)."""
    agents = list(range(n))
    goods = [f"g{i}" for i in range(n)]
    endowment = dict(zip(agents, goods))
    rankings = list(itertools.permutations(goods))
    checked = trades = 0
    failures: list = []
    for rest in itertools.product(rankings, repeat=n - 1):
        prefs = dict(zip(agents, (rankings[first], *rest)))
        outcome = rival.top_trading_cycles(endowment, prefs)
        problems = []
        if not rival.individually_rational(endowment, prefs, outcome):
            problems.append("not individually rational")
        if not rival.pareto_efficient(endowment, prefs, outcome):
            problems.append("not Pareto efficient")
        if rival.core_allocations(endowment, prefs) != [outcome]:
            problems.append("the strict core is not exactly the outcome")
        if rival.profitable_misreports(endowment, prefs):
            problems.append("a profitable misreport exists")
        if problems:
            failures.append((prefs, outcome, problems))
        checked += 1
        trades += bool(rival.movers(endowment, outcome))
    return checked, trades, failures


def main(argv: list[str]) -> int:
    n = int(argv[0]) if argv else 4
    if n not in (2, 3, 4):
        print(__doc__)
        return 2
    total = checked = trades = 0
    failures: list = []
    start = time.perf_counter()
    firsts = range(len(list(itertools.permutations(range(n)))))
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        for done, (c, t, f) in enumerate(pool.map(chunk, itertools.repeat(n), firsts), 1):
            checked, trades, failures = checked + c, trades + t, failures + f
            print(f"  chunk {done} of {len(firsts)}: {checked} profiles so far, {len(failures)} failures, "
                  f"{time.perf_counter() - start:.0f}s", flush=True)
    total = len(list(itertools.permutations(range(n)))) ** n
    print(f"n = {n}: {checked} of {total} strict profiles checked; trade happened in {trades}, not in {checked - trades}; "
          f"{len(failures)} failures")
    for prefs, outcome, problems in failures[:5]:
        print("FAILURE", prefs, outcome, problems)
    return 1 if failures or checked != total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
