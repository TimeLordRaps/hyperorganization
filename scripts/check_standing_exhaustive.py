"""Check the facts of `standing` against their definitions on EVERY strict preference profile of up to four agents.

For n agents there are (n!)^n strict profiles (4, 216 and 331,776 for n = 2, 3, 4). The start can be taken to be the
identity (agent i holds good i): any other start is the identity after renaming the goods, which maps the set of all
profiles onto itself, so this covers every market of n agents up to the names of its goods.

For each profile it checks that envy has a loop exactly when top trading cycles moves someone and exactly when the
start is not Pareto efficient (by trying every rearrangement); that a pecking order exists exactly when the start is
Pareto efficient, by the package and by trying every ordering of the agents; that after the exchange envy has no
loop; and, for EVERY order of the agents from the wealthiest down, that the start made by letting the wealthiest
pick first is a rearrangement of the goods, is left where it is by top trading cycles, has the wealth order as a
pecking order, and has envy pointing only toward the wealthier. It takes about as long as the same check of
`rival` (see the figures in VALIDATION.md) and is not part of the test suite, whose version of it covers n = 2 and 3
exhaustively and n = 4 to 30 by seeded sample.

    python scripts/check_standing_exhaustive.py [N]       # N = 2, 3 or 4 (default 4)

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

from hyperoptimization import rival, standing  # noqa: E402  (after the path is set)


def holds(endowment: dict, prefs: dict, order: tuple) -> bool:
    """A pecking order, restated from its definition: each agent's good is its favourite among the goods held by
    it and by every agent after it."""
    return all(min((endowment[b] for b in order[k:]), key=prefs[a].index) == endowment[a]
               for k, a in enumerate(order))


def chunk(n: int, first: int) -> tuple[int, int, int, int, list]:
    """All profiles whose first agent ranks the goods as the `first`-th ranking; returns (profiles, with a loop of
    envy, ladders, wealth starts, failures)."""
    agents = list(range(n))
    goods = [f"g{i}" for i in range(n)]
    start = dict(zip(agents, goods))
    rankings = list(itertools.permutations(goods))
    orders = list(itertools.permutations(agents))
    checked = loops = wealth_starts = 0
    failures: list = []
    for rest in itertools.product(rankings, repeat=n - 1):
        prefs = dict(zip(agents, (rankings[first], *rest)))
        outcome = rival.top_trading_cycles(start, prefs)
        loop = standing.envy_loop(start, prefs) is not None
        efficient = rival.pareto_efficient(start, prefs, start)
        problems = []
        if not (loop == bool(rival.movers(start, outcome)) == (not efficient)):
            problems.append("envy has a loop, someone trades and the start is inefficient together or not at all")
        found = standing.pecking_order(start, prefs)
        tried = [o for o in orders if holds(start, prefs, o)]
        if (found is not None) != efficient or bool(tried) != efficient or (found is not None and found not in tried):
            problems.append("a pecking order exists exactly when the start is Pareto efficient")
        if any(standing.is_pecking_order(start, prefs, o) != (o in tried) for o in orders):
            problems.append("is_pecking_order disagrees with the definition")
        if not standing.frozen(outcome, prefs):
            problems.append("envy still has a loop after the exchange")
        for ranked in orders:  # ranked[0] is the wealthiest
            wealth = {a: n - place for place, a in enumerate(ranked)}
            held = standing.wealth_start(wealth, prefs)
            wealth_starts += 1
            if sorted(held.values()) != goods:
                problems.append(f"wealth order {ranked}: not a rearrangement of the goods")
            elif (rival.top_trading_cycles(held, prefs) != held or not holds(held, prefs, ranked)
                  or any(wealth[t] <= wealth[a] for a, targets in standing.envy_graph(held, prefs).items()
                         for t in targets)):
                problems.append(f"wealth order {ranked}: moved, or not a pecking order, or envy points down")
        if problems:
            failures.append((prefs, problems))
        checked += 1
        loops += loop
    return checked, loops, checked - loops, wealth_starts, failures


def main(argv: list[str]) -> int:
    n = int(argv[0]) if argv else 4
    if n not in (2, 3, 4):
        print(__doc__)
        return 2
    checked = loops = ladders = wealth_starts = 0
    failures: list = []
    start = time.perf_counter()
    firsts = range(len(list(itertools.permutations(range(n)))))
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        for done, (c, lo, la, w, f) in enumerate(pool.map(chunk, itertools.repeat(n), firsts), 1):
            checked, loops, ladders = checked + c, loops + lo, ladders + la
            wealth_starts, failures = wealth_starts + w, failures + f
            print(f"  chunk {done} of {len(firsts)}: {checked} profiles so far, {len(failures)} failures, "
                  f"{time.perf_counter() - start:.0f}s", flush=True)
    total = len(list(itertools.permutations(range(n)))) ** n
    print(f"n = {n}: {checked} of {total} strict profiles checked; envy has a loop in {loops} (someone trades) and "
          f"not in {ladders} (ladders); {wealth_starts} wealth starts checked; {len(failures)} failures")
    for prefs, problems in failures[:5]:
        print("FAILURE", prefs, problems)
    return 1 if failures or checked != total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
