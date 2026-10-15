"""Compare the package with five independent reference oracles, instance by instance.

The oracles (`oracle_comprehension.py`, `oracle_quantum.py`, `oracle_price.py`, `oracle_ttc.py`,
`oracle_standing.py`) were written by a subagent from a written specification alone, without sight of the package.
Each computes the same quantities by a different route: comprehension by enumerating every group of people; the
quantum values from explicit projectors on state vectors in exact `sympy` arithmetic; the price test by
Fourier-Motzkin elimination on fractions where the package enumerates the vertices of the polytope itself; top
trading cycles by executing one cycle at a time where the package executes a round together, with the core,
efficiency and misreport checks by their own enumerations; envy loops by trying every ordered selection of
agents and pecking orders by trying every ordering, where the package searches depth-first and peels sinks, and
the dial by evaluating rankings and outcomes at settings of its own choosing (the specification is kept in
`validation/oracle-standing-specification.md`). A disagreement is a slip in one of the two, found by running both
on the same instances.
Agreement shows that implementation slips of that kind are caught. It does not show that the definitions are
right: both were written from the same specification. Only the quantum stage needs `sympy`; the others use the
standard library.

    python scripts/differential_oracles.py                  # all five stages, the quick sample
    python scripts/differential_oracles.py --full           # the sample sizes quoted in VALIDATION.md
    python scripts/differential_oracles.py --stage quantum  # one stage: comprehension, quantum, price, rival or standing

Exit status 0: every instance agrees. 1: at least one disagreement (printed). 2: a stage could not be run
(UNKNOWN: `sympy` missing, or the stage passed its deadline); no disagreement was found in what did run.
Each stage runs in its own process with a deadline, one after another, and prints progress as it goes; a stage
that passes its deadline is killed and reported as UNKNOWN, never as agreeing.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
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

from hyperoptimization import comprehension, deals, entangled, rival, standing  # noqa: E402

SIZES = {  # stage -> sample sizes (quick, full)
    "comprehension": {"instances": (1500, 1500), "graphs": (800, 800)},
    "quantum": {"ghz": (12, 12), "chsh": (8, 25)},
    "price": {"small": (120, 1000), "large": (12, 200)},
    "rival": {"outcomes": (1000, 3000), "allocations": (200, 700), "cores": (100, 300), "misreports": (40, 120)},
    "standing": {"markets": (500, 1500), "dials": (200, 600), "weight_cases": (150, 500)},
}
DEADLINES = {"comprehension": (60, 120), "quantum": (120, 300), "price": (300, 900),
             "rival": (120, 300), "standing": (120, 300)}  # seconds (quick, full)
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


# --- rival freedoms --------------------------------------------------------------------------------------------------


def housing_market(rng: random.Random, n: int) -> tuple[dict, dict]:
    """n agents, n goods, a random endowment and a random strict ranking for each agent."""
    goods = [f"g{i}" for i in range(n)]
    held = goods[:]
    rng.shuffle(held)
    return dict(zip(range(n), held)), {a: rng.sample(goods, n) for a in range(n)}


def compare_rival(outcomes: int = 3000, allocations: int = 700, cores: int = 300, misreports: int = 120,
                  report: Report = print) -> Result:
    """Top trading cycles against the reference, and the checkers against theirs. The outcomes: random markets of up to
    eight agents. The checkers (individual rationality, Pareto efficiency, the strict core): the verdicts on arbitrary
    allocations, the outcome and the endowment included, so that both verdicts occur (on outcomes alone every checker
    says yes). The strict core as a set, and the profitable misreports (none, by theorem, for the reference)."""
    ref = reference("ttc")
    result = Result("rival")
    start = time.perf_counter()
    for seed in range(outcomes):
        rng = random.Random(seed)
        endowment, prefs = housing_market(rng, rng.randint(1, 8))
        result.checked += 1
        if rival.top_trading_cycles(endowment, prefs) != ref.ttc(endowment, prefs):
            result.mismatches.append(("outcome", seed, endowment, prefs))
    seen = {name: [0, 0] for name in ("rational", "efficient", "core")}
    for seed in range(allocations):
        rng = random.Random(10_000 + seed)
        endowment, prefs = housing_market(rng, rng.randint(2, 5))
        goods, agents = list(endowment.values()), list(endowment)
        candidates = [rival.top_trading_cycles(endowment, prefs), dict(endowment)]
        for _ in range(6):
            shuffled = goods[:]
            rng.shuffle(shuffled)
            candidates.append(dict(zip(agents, shuffled)))
        for allocation in candidates:
            mine = (rival.individually_rational(endowment, prefs, allocation),
                    rival.pareto_efficient(endowment, prefs, allocation), rival.in_core(endowment, prefs, allocation))
            theirs = (ref.individually_rational(endowment, prefs, allocation), ref.pareto_efficient(prefs, allocation),
                      ref.in_core(endowment, prefs, allocation))
            result.checked += 1
            for name, verdict in zip(seen, theirs):
                seen[name][verdict] += 1
            if mine != theirs:
                result.mismatches.append(("checkers", seed, endowment, prefs, allocation, mine, theirs))
    for seed in range(cores):
        rng = random.Random(20_000 + seed)
        endowment, prefs = housing_market(rng, rng.randint(2, 5))
        mine = {frozenset(a.items()) for a in rival.core_allocations(endowment, prefs)}
        theirs = {frozenset(a.items()) for a in ref.core_allocations(endowment, prefs)}
        result.checked += 1
        if mine != theirs or len(theirs) != 1:
            result.mismatches.append(("core", seed, endowment, prefs, mine, theirs))
    for seed in range(misreports):
        rng = random.Random(30_000 + seed)
        endowment, prefs = housing_market(rng, rng.randint(2, 4))
        mine = {(a, tuple(lie)) for a, lie, _got in rival.profitable_misreports(endowment, prefs)}
        theirs = {(a, tuple(lie)) for a, lie in ref.strategy_proof(endowment, prefs)}
        result.checked += 1
        if mine != theirs:
            result.mismatches.append(("misreports", seed, endowment, prefs, mine, theirs))
    result.seconds = time.perf_counter() - start
    result.facts = {"outcomes": outcomes, "allocations": allocations * 8, "cores": cores, "misreports": misreports,
                    **{f"{name}_{'yes' if v else 'no'}": count[v] for name, count in seen.items() for v in (0, 1)}}
    result.note = (f"{outcomes} outcomes, {allocations * 8} allocations through three checkers (rational "
                   f"{seen['rational'][0]} no / {seen['rational'][1]} yes, efficient {seen['efficient'][0]} / "
                   f"{seen['efficient'][1]}, core {seen['core'][0]} / {seen['core'][1]}), {cores} cores, {misreports} "
                   "misreport sets")
    report(f"  rival: {result.checked} comparisons ({result.note}) in {result.seconds:.1f}s")
    return result


# --- standing --------------------------------------------------------------------------------------------------------


def random_dial(rng: random.Random, n: int, d: int) -> tuple[dict, dict, tuple, dict]:
    """A start, goods with d coordinates (0 to 6), a price and each agent's weights (1 to 6)."""
    names = [f"g{k}" for k in range(n)]
    goods = {g: tuple(rng.randint(0, 6) for _ in range(d)) for g in names}
    price = tuple(rng.randint(1, 6) for _ in range(d))
    weights = {a: tuple(rng.randint(1, 6) for _ in range(d)) for a in range(n)}
    held = names[:]
    rng.shuffle(held)
    return dict(zip(range(n), held)), goods, price, weights


def stretch(pieces: list, lam: Fraction):
    """The piece of the dial that holds the setting `lam`."""
    return next(p for p in pieces if p.low < lam < p.high or p.low == p.high == lam)


def worked_dials() -> list[tuple[str, tuple, list[bool]]]:
    """The dials derived by hand (tests/test_standing.py): which pieces of each trade, from setting 0 up."""
    vectors = {"G1": (3, 1), "G2": (1, 2)}
    held = {"A": "G2", "B": "G1"}
    return [
        ("the swap at equal worth", ({"A": "y", "B": "x"}, {"y": (1, 5), "x": (5, 1)}, (1, 1),
                                     {"A": (3, 1), "B": (1, 3)}), [False, True, True]),
        ("a gap of 1 in worth", (held, vectors, (1, 1), {"A": (3, 1), "B": (1, 4)}),
         [False, False, False, True, True]),
        ("the same gap, the other good listed first", (held, {"G2": (1, 2), "G1": (3, 1)}, (1, 1),
                                                       {"A": (3, 1), "B": (1, 4)}), [False, False, True, True, True]),
        ("trade that opens and closes", (held, vectors, (1, 1), {"A": (1, 3), "B": (1, 4)}),
         [False, False, False, True, True, False, False]),
    ]


def compare_standing(markets: int = 1500, dials: int = 150, weight_cases: int = 500, report: Report = print) -> Result:
    """Envy, loops, pecking orders, serial dictatorship and wealth starts against the reference on random markets of up
    to six agents, and the dial against the reference's own rankings and top trading cycles: at the representative
    of every piece, on a grid of 63 settings (so a breakpoint the package missed shows as a piece that is not
    constant), at nine settings through the rankings, and on four dials whose pieces were derived by hand. The
    characterisation of the weights that can ever trade (`can_thaw`, `thaw_witness`): each witness is run through the
    reference and must swap its two agents; for weights the package says cannot trade, the reference runs random
    markets of those agents at nine settings and must move nobody."""
    ref = reference("standing")
    result = Result("standing")
    start = time.perf_counter()
    seen = {"loop": [0, 0], "ladder_orders": 0, "orders_tried": 0}
    for seed in range(markets):
        rng = random.Random(40_000 + seed)
        n = rng.randint(1, 6)
        endowment, prefs = housing_market(rng, n)
        agents = list(endowment)
        edges = {(i, j) for i, targets in standing.envy_graph(endowment, prefs).items() for j in targets}
        theirs_orders = ref.pecking_orders(endowment, prefs)
        loop = ref.has_envy_loop(endowment, prefs)
        traded = bool(ref.movers(endowment, ref.top_trading_cycles(endowment, prefs)))
        seen["loop"][loop] += 1
        seen["ladder_orders"] += bool(theirs_orders)
        mine = (edges, standing.envy_loop(endowment, prefs) is not None, standing.frozen(endowment, prefs),
                standing.pecking_order(endowment, prefs))
        theirs = (ref.envy_edges(endowment, prefs), loop, not traded, theirs_orders[0] if theirs_orders else None)
        result.checked += 1
        if mine != theirs:
            result.mismatches.append(("market", seed, endowment, prefs, mine, theirs))
        chosen = list(itertools.permutations(agents)) if n <= 4 else [tuple(rng.sample(agents, n)) for _ in range(6)]
        valid = set(theirs_orders)
        for order in chosen:
            seen["orders_tried"] += 1
            result.checked += 1
            if standing.is_pecking_order(endowment, prefs, order) != (order in valid):
                result.mismatches.append(("is_pecking_order", seed, endowment, prefs, order))
        order = tuple(rng.sample(agents, n))
        result.checked += 1
        mine_picks = list(standing.serial_dictatorship(order, prefs).items())  # keys listed in the order of picking
        if mine_picks != list(ref.serial_dictatorship(order, prefs).items()):
            result.mismatches.append(("serial_dictatorship", seed, order, prefs))
        wealth = dict(zip(agents, rng.sample(range(1, 10 * n + 1), n)))
        if seed % 2:
            wealth = {a: Fraction(w, 3) for a, w in wealth.items()}
        result.checked += 1
        if list(standing.wealth_start(wealth, prefs).items()) != list(ref.wealth_start(wealth, prefs).items()):
            result.mismatches.append(("wealth_start", seed, wealth, prefs))
    traded_dials = reversed_dials = pieces_compared = 0
    for seed in range(dials):
        rng = random.Random(50_000 + seed)
        endowment, goods, price, weights = random_dial(rng, rng.randint(2, 5), rng.randint(1, 3))
        pieces = standing.thaw_pieces(endowment, goods, price, weights)
        pattern = []
        for piece in pieces:
            lam = piece.low if piece.low == piece.high else (piece.low + piece.high) / 2
            theirs_allocation = ref.thaw_allocation(endowment, goods, price, weights, lam)
            pattern.append(bool(ref.movers(endowment, theirs_allocation)))
            pieces_compared += 1
            result.checked += 1
            if theirs_allocation != piece.allocation:
                result.mismatches.append(("piece", seed, lam, piece.allocation, theirs_allocation))
        for step in range(1, 64):
            lam = Fraction(step, 64)
            result.checked += 1
            theirs_allocation = ref.thaw_allocation(endowment, goods, price, weights, lam)
            if theirs_allocation != stretch(pieces, lam).allocation:
                result.mismatches.append(("stretch", seed, lam, stretch(pieces, lam).allocation, theirs_allocation))
        for step in range(9):
            lam = Fraction(step, 8)
            result.checked += 1
            mine_rankings = standing.thaw_rankings(goods, price, weights, lam)
            theirs_rankings = {a: ref.thaw_ranking(goods, price, w, lam) for a, w in weights.items()}
            mine_outcome = standing.thaw_outcome(endowment, goods, price, weights, lam)
            if mine_rankings != theirs_rankings or mine_outcome != ref.thaw_allocation(
                    endowment, goods, price, weights, lam):
                result.mismatches.append(("rankings", seed, lam, mine_rankings, theirs_rankings))
        first = pattern.index(True) if True in pattern else None
        expected = None if first is None else standing.Threshold(pieces[first].low,
                                                                 pieces[first].low == pieces[first].high)
        reverses = first is not None and not all(pattern[first:])
        result.checked += 1
        if (standing.thaw_threshold(endowment, goods, price, weights), standing.thaw_reverses(
                endowment, goods, price, weights)) != (expected, reverses):
            result.mismatches.append(("threshold", seed, expected, reverses))
        traded_dials += first is not None
        reversed_dials += reverses
    witnessed = barren = delicate = stretched = 0
    for seed in range(weight_cases):
        rng = random.Random(60_000 + seed)
        d, n = rng.randint(1, 3), rng.randint(2, 4)
        price = tuple(rng.randint(1, 5) for _ in range(d))
        base, mode = tuple(rng.randint(1, 5) for _ in range(d)), rng.choice(("random", "equal", "along", "parallel"))
        strength = [rng.randint(1, 4) for _ in range(n)]
        if mode == "random":
            weights = {a: tuple(rng.randint(1, 5) for _ in range(d)) for a in range(n)}
        elif mode == "equal":
            weights = {a: base for a in range(n)}
        else:
            weights = {a: tuple(strength[a] * c for c in (price if mode == "along" else base)) for a in range(n)}
        witness = standing.thaw_witness(price, weights)
        result.checked += 1
        if (witness is not None) != standing.can_thaw(price, weights):
            result.mismatches.append(("characterisation", seed, price, weights, witness))
        elif witness is not None:
            witnessed += 1
            delicate += witness.lam == Fraction(1, 2)  # weights parallel but unequal, none along the price
            first, second = witness.agents
            pair = {first: weights[first], second: weights[second]}
            if ref.thaw_allocation(witness.endowment, witness.goods, price, pair, witness.lam) != {
                    first: "y", second: "x"}:
                result.mismatches.append(("witness", seed, price, weights, witness))
        else:
            barren += 1
            stretched += len(set(weights.values())) > 1  # unequal multiples of the price
            for _ in range(3):
                goods = {f"g{k}": tuple(rng.randint(0, 6) for _ in range(d)) for k in range(n)}
                held = list(goods)
                rng.shuffle(held)
                endowment = dict(zip(range(n), held))
                for step in range(9):
                    lam = Fraction(step, 8)
                    if ref.movers(endowment, ref.thaw_allocation(endowment, goods, price, weights, lam)):
                        result.mismatches.append(("barren", seed, price, weights, goods, endowment, lam))
    for name, instance, moves in worked_dials():
        result.checked += 1
        mine_pieces = standing.thaw_pieces(*instance)
        theirs_moves = [bool(ref.movers(instance[0], ref.thaw_allocation(
            *instance, p.low if p.low == p.high else (p.low + p.high) / 2))) for p in mine_pieces]
        if [bool(p.movers) for p in mine_pieces] != moves or theirs_moves != moves:
            result.mismatches.append(("worked", name, [bool(p.movers) for p in mine_pieces], theirs_moves, moves))
    result.seconds = time.perf_counter() - start
    result.facts = {"markets": markets, "loop_no": seen["loop"][0], "loop_yes": seen["loop"][1],
                    "with_a_pecking_order": seen["ladder_orders"], "orders_tried": seen["orders_tried"],
                    "dials": dials, "dials_that_trade": traded_dials, "dials_that_reverse": reversed_dials,
                    "pieces": pieces_compared, "worked": len(worked_dials()), "weight_cases": weight_cases,
                    "witnessed": witnessed, "barren": barren, "delicate": delicate, "stretched": stretched}
    result.note = (f"{markets} markets (loops {seen['loop'][1]} yes / {seen['loop'][0]} no; {seen['orders_tried']} "
                   f"orders tried against the reference's list), {dials} dials with {pieces_compared} pieces, 63 grid "
                   f"settings and 9 ranking settings each ({traded_dials} trade, {reversed_dials} reverse), "
                   f"{len(worked_dials())} dials worked by hand, {weight_cases} weight cases ({witnessed} with a "
                   f"witness that swaps in the reference, {delicate} of them with weights parallel but unequal; "
                   f"{barren} that cannot trade and trade nowhere in it, {stretched} of them unequal multiples of "
                   "the price)")
    report(f"  standing: {result.checked} comparisons ({result.note}) in {result.seconds:.1f}s")
    return result


STAGES = {"comprehension": compare_comprehension, "quantum": compare_quantum, "price": compare_price,
          "rival": compare_rival, "standing": compare_standing}


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
