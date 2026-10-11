"""Compare the package with three independent reference oracles, instance by instance.

The oracles (`oracle_comprehension.py`, `oracle_quantum.py`, `oracle_price.py`) were written by a
subagent from a written specification alone, without sight of the package. Each computes the same quantities
by a different route: comprehension by enumerating every group of people; the quantum values from explicit
projectors on state vectors in exact `sympy` arithmetic; the price test by Fourier-Motzkin elimination on
fractions where the package enumerates the vertices of the polytope itself. A disagreement is a slip in one of
the two, found by running both on the same instances. Agreement shows that implementation slips of that kind
are caught. It does not show that the definitions are right: both were written from the same specification.
Only the quantum stage needs `sympy`; the other two use the standard library.

    python scripts/differential_oracles.py                  # all three stages, the quick sample
    python scripts/differential_oracles.py --full           # the sample sizes quoted in VALIDATION.md
    python scripts/differential_oracles.py --stage quantum  # one stage: comprehension, quantum or price

Exit status 0: every instance agrees. 1: at least one disagreement (printed). 2: a stage could not be run
(UNKNOWN: `sympy` missing, or the stage passed its deadline); no disagreement was found in what did run.
Each stage runs in its own process with a deadline, one after another, and prints progress as it goes; a stage
that passes its deadline is killed and reported as UNKNOWN, never as agreeing.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hyperoptimization import comprehension, deals, entangled  # noqa: E402  (after the path is set)

SIZES = {  # stage -> sample sizes (quick, full)
    "comprehension": {"instances": (1500, 1500), "graphs": (800, 800)},
    "quantum": {"ghz": (12, 12), "chsh": (8, 25)},
    "price": {"small": (120, 1000), "large": (12, 200)},
}
DEADLINES = {"comprehension": (60, 120), "quantum": (120, 300), "price": (300, 900)}  # seconds (quick, full)
Report = Callable[[str], None]

# Option sets whose answers were derived by hand (the specification the oracle was written from).
# (points, which options a strictly positive price can choose, efficient options, options no price can choose)
WORKED: list[tuple[list[tuple[int, ...]], list[bool], list[int], list[int]]] = [
    ([(10, 0), (0, 10), (4, 4)], [True, True, False], [0, 1, 2], [2]),
    ([(0, 1), (1, 0), (1, 1)], [False, False, True], [2], []),
    ([(2, 0), (0, 2), (1, 1)], [True, True, True], [0, 1, 2], []),
    ([(5,)], [True], [0], []),
    ([(3, 3), (3, 3), (1, 5)], [True, True, True], [0, 1, 2], []),
    ([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)], [False, False, True, True], [2, 3], []),
]


@dataclass
class Result:
    stage: str
    checked: int = 0
    mismatches: list[tuple[Any, ...]] = field(default_factory=list)
    seconds: float = 0.0
    note: str = ""
    facts: dict[str, int] = field(default_factory=dict)  # how much of the interesting ground the sample covered

    @property
    def agrees(self) -> bool:
        return self.checked > 0 and not self.mismatches


def reference(name: str) -> ModuleType:
    """Load scripts/oracle_<name>.py as a module, without making `scripts` a package."""
    path = HERE / f"oracle_{name}.py"
    spec = importlib.util.spec_from_file_location(f"oracle_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- comprehension ---------------------------------------------------------------------------------------------------


def compare_comprehension(instances: int = 1500, graphs: int = 800, report: Report = print) -> Result:
    """Cover number, robust quorum and necessary voices against enumeration of every group of people, and
    footprints (material and impact) against a second graph search, on random instances."""
    ref = reference("comprehension")
    result = Result("comprehension")
    start = time.perf_counter()
    makeable = unmakeable = 0
    for seed in range(instances):
        rng = random.Random(900000 + seed)
        names = [f"e{i}" for i in range(rng.randint(1, 7))]
        know = {f"p{i}": [n for n in names if rng.random() < rng.choice([0.3, 0.5, 0.7])]
                for i in range(rng.randint(1, 6))}
        nodes = rng.sample(names, rng.randint(0, len(names)))
        mine = (comprehension.cover_number(know, nodes), comprehension.robust_quorum(know, nodes))
        theirs = (ref.cover_number(know, nodes), ref.robust_quorum(know, nodes))
        result.checked += 1
        if mine != theirs:
            result.mismatches.append(("cover/robust", seed, mine, theirs))
        if theirs[0] is not None:
            makeable += 1
            voices, expected = set(comprehension.necessary_voices(know, nodes)), ref.necessary(know, nodes)
            result.checked += 1
            if voices != expected:
                result.mismatches.append(("necessary", seed, voices, expected))
        else:
            unmakeable += 1
    for seed in range(graphs):
        rng = random.Random(910000 + seed)
        names = [f"n{i}" for i in range(rng.randint(1, 8))]
        graph = {n: [m for m in names if rng.random() < 0.25] for n in names}  # self-loops allowed
        targets = rng.sample(names, rng.randint(1, min(3, len(names))))
        found = comprehension.footprint(graph, targets)
        material, impact = ref.footprint(graph, targets)
        result.checked += 1
        if (set(found.material), set(found.impact)) != (material, impact):
            result.mismatches.append(("footprint", seed, (set(found.material), set(found.impact)),
                                      (material, impact)))
    result.seconds = time.perf_counter() - start
    result.facts = {"makeable": makeable, "unmakeable": unmakeable, "graphs": graphs}
    result.note = f"{makeable} makeable and {unmakeable} unmakeable instances, {graphs} graphs"
    report(f"  comprehension: {result.checked} comparisons ({result.note}) in {result.seconds:.1f}s")
    return result


# --- quantum ---------------------------------------------------------------------------------------------------------


GHZ_STATES = [(1, 1), (3, 4), (5, 12), (1, 0), (0, 1), (7, 24), (1, 2), (2, 3), (4, 3), (-3, 4), (8, 15),
              (Fraction(1, 2), Fraction(3, 4))]


def as_fraction(value: Any) -> Fraction:
    return Fraction(int(value.p), int(value.q))


def exact(value: int | Fraction) -> Any:
    """A sympy Rational from an int or a Fraction."""
    import sympy
    return sympy.Rational(value.numerator, value.denominator) if isinstance(value, Fraction) else sympy.Integer(value)


def compare_quantum(ghz: int = 12, chsh: int = 25, report: Report = print) -> Result:
    """GHZ-type states, random rational CHSH strategies and the magic square: the package's exact win
    probabilities against the oracle's, computed from explicit projectors on the state vector."""
    ref = reference("quantum")
    result = Result("quantum")
    start = time.perf_counter()
    game = entangled.ghz_game()
    for a, b in GHZ_STATES[:ghz]:
        mine = entangled.quantum_value(game, entangled.ghz_strategy(a, b))
        theirs = as_fraction(ref.ghz_value(exact(a), exact(b)))
        result.checked += 1
        if mine != theirs:
            result.mismatches.append(("ghz", (a, b), mine, theirs))
    report(f"  quantum: {result.checked} GHZ-type states compared in {time.perf_counter() - start:.1f}s")
    game = entangled.chsh_game()
    rng = random.Random(77)
    for _ in range(chsh):
        angles = []
        for _k in range(4):
            u = Fraction(rng.randint(-9, 9), rng.randint(1, 9))
            angles.append(((1 - u * u) / (1 + u * u), 2 * u / (1 + u * u)))  # (cos, sin) of a rational angle
        mine = entangled.quantum_value(game, entangled.chsh_strategy(*angles))
        theirs = as_fraction(ref.chsh_value(*[exact(v) for pair in angles for v in pair]))
        result.checked += 1
        if mine != theirs:
            result.mismatches.append(("chsh", angles, mine, theirs))
    mine = entangled.quantum_value(entangled.magic_square_game(), entangled.magic_square_strategy())
    theirs = as_fraction(ref.magic_value())
    result.checked += 1
    if mine != theirs:
        result.mismatches.append(("magic square", None, mine, theirs))
    result.seconds = time.perf_counter() - start
    result.facts = {"ghz": ghz, "chsh": chsh, "magic": 1}
    result.note = f"{ghz} GHZ-type states, {chsh} CHSH strategies, the magic square"
    report(f"  quantum: {result.checked} comparisons ({result.note}) in {result.seconds:.1f}s")
    return result


# --- price -----------------------------------------------------------------------------------------------------------


def random_points(rng: random.Random, big: bool) -> list[tuple[int, ...]]:
    """A random option set of small integer vectors, with repeats and dominated points allowed. One set in
    three is built to contain an option below the blend of the others (price-invisible options are rare in
    uniformly random sets): the axis points m*e_i, a few interior points with positive coordinates summing to
    less than m, and a few random ones."""
    dims = rng.randint(2, 4) if big else rng.randint(2, 3)
    top = 12 if big else 6
    count = rng.randint(4, 8) if big else rng.randint(2, 6)
    if rng.random() < 1 / 3:
        m = rng.randint(dims + 2, top + dims)
        axes = [tuple(m if j == i else 0 for j in range(dims)) for i in range(dims)]
        inside = []
        for _ in range(rng.randint(1, 2)):
            while True:
                q = tuple(rng.randint(1, m) for _ in range(dims))
                if sum(q) < m:
                    inside.append(q)
                    break
        extra = [tuple(rng.randint(0, top) for _ in range(dims)) for _ in range(rng.randint(0, 2))]
        points = axes + inside + extra
        rng.shuffle(points)
        return points
    return [tuple(rng.randint(0, top) for _ in range(dims)) for _ in range(count)]


def compare_price(small: int = 120, large: int = 12, report: Report = print) -> Result:
    """Efficient options, supporting prices and price-invisible options against the oracle's exact simplex:
    the hand-derived worked cases, then random option sets (small: up to 6 options in 2 or 3 dimensions;
    large: up to 8 in 2 to 4, or the planted sets of `random_points`)."""
    ref = reference("price")
    result = Result("price")
    start = time.perf_counter()
    invisible = efficient_points = 0
    for index, (points, want_supported, want_efficient, want_hidden) in enumerate(WORKED):
        found = ([deals.supported(points, i) for i in range(len(points))], deals.efficient(points),
                 deals.price_invisible(points))
        other = ([ref.supported(points, i) for i in range(len(points))], ref.efficient(points),
                 ref.price_invisible(points))
        result.checked += 1
        if not found == other == (want_supported, want_efficient, want_hidden):
            result.mismatches.append(("worked", index, points, found, other, (want_supported, want_efficient, want_hidden)))
    for label, count, big, base in (("small", small, False, 20000), ("large", large, True, 30000)):
        for trial in range(count):
            points = random_points(random.Random(base + trial), big)
            found = ([deals.supported(points, i) for i in range(len(points))], deals.efficient(points),
                     deals.price_invisible(points))
            other = ([ref.supported(points, i) for i in range(len(points))], ref.efficient(points),
                     ref.price_invisible(points))
            result.checked += 1
            invisible += len(other[2])
            efficient_points += len(other[1])
            if found != other:
                result.mismatches.append((label, trial, points, found, other))
            if (trial + 1) % 100 == 0:
                report(f"  price ({label}): {trial + 1} of {count}, {time.perf_counter() - start:.1f}s")
    result.seconds = time.perf_counter() - start
    result.facts = {"worked": len(WORKED), "efficient": efficient_points, "invisible": invisible}
    result.note = (f"{len(WORKED)} worked cases, {small} small and {large} large option sets; "
                   f"{efficient_points} efficient options, {invisible} of them price-invisible")
    report(f"  price: {result.checked} comparisons ({result.note}) in {result.seconds:.1f}s")
    return result


STAGES = {"comprehension": compare_comprehension, "quantum": compare_quantum, "price": compare_price}


def run_stage(name: str, full: bool, report: Report = print) -> Result:
    """Run one stage in this process. `sympy` missing is UNKNOWN (nothing compared), never agreement."""
    if name == "quantum":
        try:
            import sympy  # noqa: F401
        except ImportError:
            report(f"  {name}: UNKNOWN - `sympy` is not installed (pip install sympy); nothing was compared")
            return Result(name, note="sympy missing")
    sizes = {key: pair[1 if full else 0] for key, pair in SIZES[name].items()}
    return STAGES[name](**sizes, report=report)


def stage_status(result: Result) -> int:
    """0 agree, 1 disagree, 2 unknown (nothing was compared)."""
    return 1 if result.mismatches else 0 if result.checked else 2


def supervised(name: str, full: bool) -> int:
    """Run one stage in a child process with its deadline. A stage that is still running at the deadline is
    killed (`subprocess.run` kills the child and reaps it before raising) and reported as UNKNOWN."""
    deadline = DEADLINES[name][1 if full else 0]
    command = [sys.executable, str(Path(__file__).resolve()), "--stage", name, "--unsupervised"]
    command += ["--full"] if full else []
    try:
        return subprocess.run(command, timeout=deadline, check=False).returncode
    except subprocess.TimeoutExpired:
        print(f"  {name}: UNKNOWN - still running after {deadline}s; the stage was killed and nothing is claimed",
              flush=True)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", choices=sorted(STAGES) + ["all"], default="all")
    parser.add_argument("--full", action="store_true", help="the larger sample sizes quoted in VALIDATION.md")
    parser.add_argument("--unsupervised", action="store_true", help="run in this process, with no deadline")
    args = parser.parse_args(argv)
    names = sorted(STAGES) if args.stage == "all" else [args.stage]
    if args.unsupervised:
        status = 0
        for name in names:
            result = run_stage(name, args.full)
            for mismatch in result.mismatches[:5]:
                print("  MISMATCH", mismatch, flush=True)
            status = max(status, stage_status(result))
        return status
    statuses = {}
    for name in names:
        print(f"{name} ({'full' if args.full else 'quick'} sample, deadline {DEADLINES[name][args.full]}s):", flush=True)
        statuses[name] = supervised(name, args.full)
        print(f"  => {('AGREE', 'DISAGREE', 'UNKNOWN')[statuses[name]] if statuses[name] in (0, 1, 2) else 'FAILED'}",
              flush=True)
    worst = max(statuses.values())
    return worst if worst in (0, 1, 2) else 1


if __name__ == "__main__":
    sys.exit(main())
