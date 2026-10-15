"""The decision ledger: every modelling choice that carries an expectation a witness could check.

You asked to know which decisions made while building this hold key evidential expectations from
witnesses inside the framework. A witness inside the framework is what the family's checkers
already produce: a `Verdict` that holds in a model, or names a counterexample. So each decision
here records

    CHOICE        what was chosen, and ALTERNATIVES what was not and why;
    EXPECTATION   what a witness should observe if the choice is right;
    REFUTER       the observation that would show it wrong;
    WORLD WITNESS what someone outside the model would have to observe, which is often "none yet";
    CHECK         for every FORM and FRAME decision, code that runs the expectation and returns a
                  Verdict whose evidence says what was read.

A check is not trusted because it passes. `tests/test_decisions.py` breaks, for each decision, the
code its check rests on, and requires the check to fail; and it requires `DECISIONS.md` to be exactly
what `render_markdown` produces, so the numbers in the document were read off a run.

Tags are the family's: FORM (checked exactly), FRAME (holds inside stated assumptions), HYPER (a
proposal nothing here witnesses), OPEN (unresolved). A HYPER or OPEN decision has no check on
purpose: a check that always passed would be a claim of evidence the package does not have.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from importlib import resources

from . import (
    comprehension,
    deals,
    entangled,
    entrenchment,
    family,
    knowledge,
    loop,
    ordinal,
    quorum,
    renewal,
    rival,
    standing,
    surprisal,
    viability,
)
from .ground import OMEGA, Status, Tag, Verdict, rank_key

Check = Callable[[], Verdict]


@dataclass(frozen=True, slots=True)
class Decision:
    id: str
    title: str
    tag: Tag
    choice: str
    alternatives: tuple[str, ...]
    expectation: str
    refuter: str
    world_witness: str
    where: str
    check: Check | None = None


def _verdict(claim: str, problems: list[str], evidence: list[str]) -> Verdict:
    if problems:
        return Verdict(claim, Status.FAILS_IN_MODEL, tuple(problems))
    return Verdict(claim, Status.HOLDS_IN_MODEL, evidence=tuple(evidence))


def _golden() -> dict:
    text = resources.files("hyperoptimization").joinpath("data/golden_l2_gate_bits.json")
    return json.loads(text.read_text(encoding="utf-8"))


def _golden_gl() -> dict:
    text = resources.files("hyperoptimization").joinpath("data/golden_gl_loops.json")
    return json.loads(text.read_text(encoding="utf-8"))


def _decode(value: object) -> float:
    return math.inf if value == "inf" else value  # type: ignore[return-value]


def _system(name: str) -> tuple[dict, dict, set, dict]:
    data = next(s for s in _golden()["systems"] if s["name"] == name)
    prices = {n: _decode(p) for n, p in data["parties"].items()}
    controls = {tuple(edge) for edge in data["controls"]}
    return data, prices, controls, data["gates"]


# ---------------------------------------------------------------------------
# The checks.
# ---------------------------------------------------------------------------


def _d01() -> Verdict:
    golden = _golden()
    problems: list[str] = []
    gates = bracketed = 0
    for system in golden["systems"]:
        prices = {n: _decode(p) for n, p in system["parties"].items()}
        controls = {tuple(edge) for edge in system["controls"]}
        finite = all(p != math.inf for p in prices.values())
        for power, gate in system["gates"].items():
            gates += 1
            need = frozenset(gate["need"])
            want = _decode(gate["gate_cost"])
            got = surprisal.gate_bits(prices, controls, need)
            if got != want:
                problems.append(f"{system['name']}:{power}: gate_bits {got}, hyperethics {want}")
            if finite and want != math.inf:
                probs = {n: Fraction(1, 2 ** int(p)) for n, p in prices.items()}
                lower, exact, upper = surprisal.sandwich_gate(probs, controls, need)
                if not (lower == Fraction(1, 2 ** int(want)) and lower <= exact <= upper):
                    problems.append(f"{system['name']}:{power}: probability outside 2^-cost..bound")
                bracketed += 1
    return _verdict("a price in bits reproduces hyperethics' gate costs and brackets the probability",
                    problems, [
                        f"{gates} gates in {len(golden['systems'])} systems equal hyperethics gate_cost "
                        f"(commit {golden['hyperethics_commit'][:12]})",
                        f"{bracketed} gates: 2^-cost <= exact probability <= union bound, in exact "
                        "rationals"])


def _d02() -> Verdict:
    gap = surprisal.additivity_gap(Fraction(1, 8), [Fraction(0), Fraction(0)])
    _, prices, controls, gates = _system("common-principal")
    need = frozenset(gates["legislate"]["need"])
    together = surprisal.gate_bits(prices, controls, need)
    apart = surprisal.gate_bits(prices, set(), need)
    problems = []
    if abs(gap - 3.0) > 1e-12 or apart - together != 3 or abs(gap - (apart - together)) > 1e-12:
        problems.append(f"additivity gap {gap:.6f} bits, cover discount {apart - together} bits")
    return _verdict("a common principal is a measurable loss of surprisal", problems, [
        f"a principal at 3 bits commanding two seats: additivity gap {gap:.6f} bits",
        f"hyperethics' cheaper cover through the same command: {apart} bits apart, {together} "
        f"together, discount {apart - together}"])


def _d03() -> Verdict:
    x = Fraction(1, 4)
    problems: list[str] = []
    worst = 0.0
    cases = [(6, 4, 2), (6, 4, 3), (8, 5, 2), (8, 5, 4), (9, 6, 3)]  # quorum above the mean m x
    for m, q, groups in cases:
        blocks = [surprisal.Block(f"g{i}", x, m // groups) for i in range(groups)]
        grouped = surprisal.capture_probability(blocks, q)
        independent = quorum.tail(m, q, x)
        if not grouped >= independent:
            problems.append(f"m={m} q={q} groups={groups}: grouped {float(grouped):.4f} < "
                            f"independent {float(independent):.4f}")
        worst = max(worst, surprisal.bits(independent) - surprisal.bits(grouped))
    worked = surprisal.capture_probability(
        [surprisal.Block("a", x, 3), surprisal.Block("b", x, 3)], 4)
    if worked != Fraction(1, 16):
        problems.append(f"two groups of three, quorum four, x=1/4: {worked}, expected 1/16")
    y = Fraction(2, 3)  # now the quorum is BELOW the mean count, and the direction reverses
    reversed_case = surprisal.capture_probability(
        [surprisal.Block(f"g{i}", y, 2) for i in range(3)], 2)
    if not reversed_case < quorum.tail(6, 2, y):
        problems.append("below the mean, positive association should lower the capture probability")
    return _verdict("independence understates capture risk when the quorum is above the mean", problems, [
        f"{len(cases)} groupings with quorum above the mean: grouped capture >= independent, "
        f"understated by up to {worst:.3f} bits",
        "worked case: two groups of three seats, quorum 4, x = 1/4: grouped 1/16, independent "
        f"{float(quorum.tail(6, 4, x)):.4f}",
        f"below the mean (x = 2/3, quorum 2): grouped {float(reversed_case):.4f} < independent "
        f"{float(quorum.tail(6, 2, y)):.4f}, the direction reverses"])


def _tyranny_game() -> viability.Game:
    return viability.Game.from_table(
        nature=("n",),
        table={
            "fair": {"stay": ("fair",), "drift": ("tyranny",), "collapse": ("ruin",)},
            "tyranny": {"hold": ("tyranny",)},
            "ruin": {"hold": ("ruin",)},
        },
        safe={"fair", "tyranny"},
    )


def _d04() -> Verdict:
    game = _tyranny_game()
    inside = viability.kernel(game, corrigible=False)
    law = viability.greatest_law(game, corrigible=False)
    problems = []
    if "tyranny" not in inside:
        problems.append("the stable bad state is not in the kernel")
    if "drift" not in law.get("fair", frozenset()):
        problems.append("the move into it is not permitted")
    if "collapse" in law.get("fair", frozenset()):
        problems.append("the move into ruin is permitted")
    return _verdict("existence alone does not exclude a stable bad state", problems, [
        f"kernel = {sorted(inside)}",
        f"the greatest law at 'fair' permits {sorted(law['fair'])} and forbids "
        f"{sorted(game.allowed('fair') - law['fair'])}: persistence is blind to who is better off"])


def _brute_reach(game: viability.Game, law: dict, start: str) -> set[str]:
    seen, frontier = {start}, [start]
    while frontier:
        s = frontier.pop()
        if s not in game.safe:
            continue
        for a in law.get(s, ()):
            for b in game.nature:
                t = game.target(s, a, b)
                if t not in seen:
                    seen.add(t)
                    frontier.append(t)
    return seen


def _d05() -> Verdict:
    problems: list[str] = []
    pairs = sustained = 0
    for seed in range(20):
        game = viability.random_game(random.Random(5000 + seed), n=3, na=2)
        star = viability.greatest_law(game, corrigible=False)
        # (i) the greatest law is itself sustaining from every state of the kernel, by brute force:
        # without this a law that permitted too much would still be "greater" than every other
        for s0 in sorted(viability.kernel(game, corrigible=False)):
            seen = _brute_reach(game, star, s0)
            if not (all(s in game.safe for s in seen) and all(star.get(s) for s in seen if s in game.safe)):
                problems.append(f"seed {seed}: the greatest law does not sustain existence from {s0}")
            sustained += 1
        # (ii) the module's verdict is brute force, and what any sustaining law permits where it goes
        # lies inside the greatest law: without this a law that permitted too little would pass (i)
        for law in viability.all_laws(game):
            for start in game.states:
                seen = _brute_reach(game, law, start)
                independent = all(s in game.safe for s in seen) and all(
                    law[s] for s in seen if s in game.safe)
                said = viability.sustaining(game, law, start, corrigible=False).status \
                    is Status.HOLDS_IN_MODEL
                pairs += 1
                if independent != said:
                    problems.append(f"seed {seed}, start {start}: brute force {independent}, "
                                    f"sustaining {said}")
                if independent and not all(law[s] <= star.get(s, frozenset()) for s in seen):
                    problems.append(f"seed {seed}, start {start}: a sustaining law permits what the "
                                    "greatest law forbids")
    renamed = 0
    for seed in range(10):
        game = viability.random_game(random.Random(6000 + seed))
        mapping = {s: f"S{i}" for i, s in enumerate(reversed(game.states))}
        twin = game.relabel(states=mapping)
        want = {mapping[s]: p for s, p in viability.greatest_law(game, corrigible=False).items()}
        if viability.greatest_law(twin, corrigible=False) != want:
            problems.append(f"seed {seed}: renaming changed the law")
        renamed += 1
    return _verdict("the optimal law is the unique greatest sustaining law and names no participant",
                    problems, [
                        f"{sustained} kernel states over 20 random games: the greatest law sustains "
                        "existence from each, by brute force",
                        f"{pairs} (law, start) pairs: the module's verdict equals brute force, and every "
                        "sustaining law lies inside the greatest",
                        f"{renamed} games renamed: the law renames with them"])


def _d06() -> Verdict:
    problems: list[str] = []
    exempted = whole = 0
    for seed in range(60):
        game = viability.random_game(random.Random(7000 + seed), n=4)
        star = viability.greatest_law(game, corrigible=False)
        for start in sorted(star):
            if viability.survival_probability(game, star, start, corrigible=False) != 1:
                problems.append(f"seed {seed}: the greatest law does not survive from {start}")
            whole += 1
            for s in sorted(viability.reachable(game, star, start)):
                for a in sorted(game.allowed(s) - star[s]):
                    law = {t: set(star.get(t, game.allowed(t))) for t in game.states}
                    law[s].add(a)
                    law = {t: frozenset(v) for t, v in law.items()}
                    if viability.survival_probability(game, law, start, corrigible=False) >= 1:
                        problems.append(f"seed {seed}: exempting {a!r} at {s} left survival at 1")
                    exempted += 1
    if exempted < 30:
        problems.append(f"only {exempted} exemptions were tried; the sample is too thin to say")
    return _verdict("under full support, survival is certain iff the law is sustaining, and an "
                    "exemption is fatal", problems, [
                        f"{whole} starts: the greatest law survives with probability exactly 1",
                        f"{exempted} single permissions added outside it: survival below 1 each time"])


def _d07() -> Verdict:
    problems: list[str] = []
    game = viability.Game.from_table(
        nature=("n",),
        table={"p": {"stay": ("p",), "wander": ("q",)}, "q": {"linger": ("q",)},
               "r": {"hold": ("r",), "dive": ("p",)}},
        safe={"p", "q", "r"}, home={"p"}, correct={("r", "p")},
    )
    plain = viability.kernel(game, corrigible=False)
    corrigible = viability.kernel(game, corrigible=True)
    law = viability.greatest_law(game, corrigible=True)
    if not corrigible < plain:
        problems.append("the cul-de-sac game: corrigible kernel is not strictly inside the kernel")
    if "wander" in law.get("p", frozenset()):
        problems.append("the irreversible move is permitted")
    strict = 0
    for seed in range(150):
        g = viability.random_game(random.Random(8000 + seed), home=True)
        a, b = viability.kernel(g, corrigible=True), viability.kernel(g, corrigible=False)
        if not a <= b:
            problems.append(f"seed {seed}: corrigible kernel not inside the kernel")
        strict += a < b
    if strict < 5:
        problems.append("corrigibility never cost anything in the sample")
    return _verdict("corrigibility is staying where an overseer can bring the system home", problems, [
        f"cul-de-sac game: kernel {sorted(plain)}, corrigible kernel {sorted(corrigible)}; the "
        f"corrigible law at p permits {sorted(law['p'])} and forbids the move to the dead end",
        f"150 random games with a home: corrigible kernel inside the kernel always, strictly "
        f"inside in {strict}"])


def _d08() -> Verdict:
    problems: list[str] = []
    tampered = 0
    for seed in range(30):
        game = viability.random_game(random.Random(9000 + seed))
        truth = viability.ranks(game, corrigible=False)
        if viability.check_ranks(game, truth, corrigible=False).status is not Status.HOLDS_IN_MODEL:
            problems.append(f"seed {seed}: the true ranks were rejected")
        for state in game.states:
            for value in [*range(len(game.states) + 2), OMEGA]:
                if value == truth[state]:
                    continue
                claim = {**truth, state: value}
                tampered += 1
                if viability.check_ranks(game, claim, corrigible=False).status \
                        is Status.HOLDS_IN_MODEL:
                    problems.append(f"seed {seed}: {state} -> {value} was accepted")
    return _verdict("a rank is a certificate checkable in one pass", problems, [
        "30 random games: the true ranks satisfy the local equation",
        f"{tampered} single-entry tamperings: every one rejected"])


def _d09() -> Verdict:
    problems: list[str] = []
    lam, nu = Fraction(1, 5), Fraction(4, 5)  # rho = 1/5 < theta = 1/2

    def residual(m: int) -> float:
        d = renewal.exponent(0.5, 0.2)
        return renewal.log_mean_lifetime(renewal.Chain(m, m // 2, lam, nu)) - m * d + 0.5 * math.log(m)

    r1, r2 = residual(1600), residual(3200)
    if abs(r2 - r1) >= 0.003:
        problems.append(f"inside the margin the residual moved by {r2 - r1:.4f}")
    limit = renewal.immediate_limit(0.5, 0.8, 1.0)
    life = math.exp(renewal.log_mean_lifetime(
        renewal.Chain(3200, 1600, Fraction(4, 5), Fraction(1, 5))))
    if abs(life - limit) / limit >= 6e-3:
        problems.append(f"past the margin the life {life:.4f} is not {limit:.4f}")
    termless = math.exp(renewal.log_mean_lifetime(renewal.Chain(2000, 1000, 1, 0)))
    if abs(termless - math.log(2)) >= 1e-3:
        problems.append(f"with no repair the life is {termless:.4f}, not ln 2")
    half = Fraction(1, 2)
    grow = (math.exp(renewal.log_mean_lifetime(renewal.Chain(6400, 3200, half, half)))
            - math.exp(renewal.log_mean_lifetime(renewal.Chain(1600, 800, half, half))))
    if abs(grow - 0.5 * math.log(4)) >= 2e-3:
        problems.append(f"at the margin quadrupling the seats added {grow:.4f}, not (1/2) ln 4")
    return _verdict("revitalization is self-repair, and redundancy cannot replace it", problems, [
        f"inside the margin (rho .2, theta .5): ln life - m D + (1/2) ln m is {r1:.4f} at m=1600 and "
        f"{r2:.4f} at m=3200: life grows as exp(m D)",
        f"past the margin (rho .8, theta .5): life {life:.4f} at m=3200 against the limit "
        f"ln(rho/(rho-theta))/rate = {limit:.4f}",
        f"no repair, 2000 seats, half needed: life {termless:.4f} (ln 2 = {math.log(2):.4f}) however "
        "many seats",
        f"at the margin: quadrupling the seats adds {grow:.4f} (predicted (1/2) ln 4 = "
        f"{0.5 * math.log(4):.4f})"])


def _d10() -> Verdict:
    problems: list[str] = []
    cmd, veto = loop.Link.COMMAND, loop.Link.VETO
    cells = 0
    for m in range(1, 7):
        for q in range(1, m + 1):
            for pi in (Fraction(1, 10), Fraction(1, 3)):
                for s in (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1)):
                    lhs = 1 - loop.step(cmd, m, q, pi, 1 - s)
                    rhs = loop.step(veto, m, quorum.dual(m, q), 1 - pi, s)
                    cells += 1
                    if lhs != rhs:
                        problems.append(f"duality fails at m={m} q={q} pi={pi} s={s}")
    sizes = (3, 5, 7, 11, 21, 51, 101)
    for m in sizes:
        q = (m + 1) // 2
        c, v = loop.threshold(cmd, m, q), loop.threshold(veto, m, q)
        if abs(c + v - 1.0) > 1e-9:
            problems.append(f"m={m}: thresholds {c:.6f} + {v:.6f} do not sum to 1")
        if not v > c:
            problems.append(f"m={m}: veto {v:.4f} does not outlast command {c:.4f}")
    c5, v5 = loop.threshold(cmd, 5, 3), loop.threshold(veto, 5, 3)
    return _verdict("command and veto are one equation in dual variables, and vetoing is worth more",
                    problems, [
                        f"duality 1 - G_command(1 - s) = G_veto*(s) exact on {cells} grid points",
                        f"self-dual quorums at m in {sizes}: critical pressures sum to 1 and veto "
                        "exceeds command",
                        f"m = 5, q = 3: command {c5:.4f}, veto {v5:.4f}, a factor of {v5 / c5:.2f}"])


def _d11() -> Verdict:
    problems: list[str] = []
    half = Fraction(1, 2)
    for m in range(3, 22, 2):
        q = (m + 1) // 2
        if quorum.tail(m, q, half) != half:
            problems.append(f"m={m}: T(1/2) is not 1/2")
        if surprisal.council_bits(m, q, half) != 1.0:
            problems.append(f"m={m}: the price at x = 1/2 is not one bit")
    brackets = {q: quorum.critical_point(9, q) for q in range(2, 9)}
    ordered = [brackets[q] for q in range(2, 9)]
    if not all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])):
        problems.append("the critical point does not rise with the quorum")
    if brackets[5] != (half, half) or not brackets[4][1] < half < brackets[6][0]:
        problems.append("the balanced quorum is not critical at exactly 1/2")
    return _verdict("the balanced quorum is the odd majority, critical at exactly one half", problems, [
        "odd majorities m = 3..21: T(1/2) = 1/2 exactly and the council's price at x = 1/2 is "
        "exactly 1 bit",
        "m = 9: the critical point rises with the quorum, " + ", ".join(
            f"q={q}: {float(sum(brackets[q]) / 2):.4f}" for q in (3, 4, 5, 6, 7))])


def _d12() -> Verdict:
    problems: list[str] = []
    price = loop.closing_price(loop.Link.COMMAND, 5, 3, Fraction(1, 10), 1)
    if not price < 0.02:
        problems.append(f"one level of oversight is {price:.4f} from the loop's limit")
    m, q = 51, 26
    t = loop.threshold(loop.Link.COMMAND, m, q)
    near = loop.levels_to_cross(loop.Link.COMMAND, m, q, t * (1 + 1e-4), 0.5)
    nearer = loop.levels_to_cross(loop.Link.COMMAND, m, q, t * (1 + 1e-6), 0.5)
    ratio = nearer / near if near and nearer else math.nan
    if not 9.0 <= ratio <= 11.5:
        problems.append(f"the delay grew by {ratio:.2f} for a 100-fold approach, not about 10")
    return _verdict("closing the loop costs little while healthy and everything near the threshold",
                    problems, [
                        f"5 seats, quorum 3, pi = 0.1: one level of oversight is within {price:.5f} "
                        "of the loop's limit",
                        f"51 seats, quorum 26: {near} levels before collapse at 1e-4 past the "
                        f"threshold, {nearer} at 1e-6: x{ratio:.2f} for 100x closer, i.e. (pi-pi*)^(-1/2)"])


def _d13() -> Verdict:
    problems: list[str] = []
    ranks = [0, 1, 2, 7, OMEGA]
    ords = [ordinal.rank_ordinal(r) for r in ranks]
    if not all(a < b for a, b in zip(ords, ords[1:])):
        problems.append("ranks and their ordinals do not compare alike")
    seq = list(ordinal.goodstein_sequence(4, limit=300))
    if [v for v, _b, _o in seq[:6]] != [4, 26, 41, 60, 83, 109]:
        problems.append("the first terms of the sequence from 4 are not 4, 26, 41, 60, 83, 109")
    if str(seq[0][2]) != "ω^ω":
        problems.append(f"the sequence from 4 starts at {seq[0][2]}, not ω^ω")
    descent = ordinal.check_descent(o for _v, _b, o in seq)
    if descent.status is not Status.HOLDS_IN_MODEL:
        problems.extend(descent.witness)
    if any(ordinal.hereditary_ordinal(v, b).evaluate(b) != v for v, b, _o in seq):
        problems.append("an ordinal does not evaluate back to its term")
    finite = omega_seen = 0
    for seed in range(20):
        game = viability.random_game(random.Random(9500 + seed))
        for rank in viability.ranks(game, corrigible=False).values():
            if rank is OMEGA:
                omega_seen += 1
            elif rank <= len(game.states):
                finite += 1
            else:
                problems.append(f"seed {seed}: a finite rank {rank} exceeds the number of states")
    return _verdict("a phase is an ordinal rank; transfinite ranks appear when the structure nests",
                    problems, [
                        "viability ranks compare as the ordinals 0 < 1 < 2 < 7 < ω",
                        f"20 random games: {finite} finite ranks (each at most the number of states) "
                        f"and {omega_seen} of rank ω; nothing between",
                        f"Goodstein from 4: starts at {seq[0][2]}, 300 steps, the ordinals strictly "
                        f"descend to {seq[-1][2]}; the first terms are 4, 26, 41, 60, 83, 109"])


def _d14() -> Verdict:
    problems: list[str] = []
    graphs = 0
    for n in range(1, 5):
        for governors in itertools.product([None, *range(n)], repeat=n):
            rules = [entrenchment.Rule(f"r{i}", 1, None if g is None else f"r{g}")
                     for i, g in enumerate(governors)]
            heights = entrenchment.authority_height(rules)
            graphs += 1
            if None not in governors and not all(h is OMEGA for h in heights.values()):
                problems.append(f"{governors}: every rule is governed yet a height is finite")
            if None in governors and not any(h == 0 for h in heights.values()):
                problems.append(f"{governors}: an ungoverned rule has no height 0")
            on_cycle = {name for cycle in entrenchment.cycles(rules) for name in cycle}
            if on_cycle and any(heights[name] is not OMEGA for name in on_cycle):
                problems.append(f"{governors}: a rule on a cycle has a finite height")
    ross = entrenchment.protection([
        entrenchment.Rule("entrenched clause", 100, "amendment clause"),
        entrenchment.Rule("amendment clause", 5, "amendment clause"),
    ])
    if ross["entrenched clause"] != 5:
        problems.append(f"the entrenched clause is protected by {ross['entrenched clause']}, not 5")
    return _verdict("protection is the weakest link, and with no sovereign there is no finite rank",
                    problems, [
                        f"all {graphs} governance graphs on up to 4 rules: a finite height anywhere "
                        "means something is ungoverned; with every rule governed, every height is ω",
                        f"Ross: a clause worth {100} bits amendable by a {5}-bit clause is protected "
                        f"by {ross['entrenched clause']:g} bits"])


def _d15() -> Verdict:
    problems: list[str] = []
    termless = float(renewal.mean_lifetime(renewal.Chain(2000, 1000, 1, 0)))
    if abs(termless - math.log(2)) >= 1e-3:
        problems.append(f"unbounded-term: life {termless:.4f}, not ln 2")
    _, prices, controls, gates = _system("suzerain")
    held = surprisal.gate_bits(prices, controls, frozenset(gates["legislate"]["need"]))
    _, prices2, controls2, gates2 = _system("two-key")
    sound = surprisal.gate_bits(prices2, controls2, frozenset(gates2["legislate"]["need"]))
    if sound - held != 3:
        problems.append(f"externally-held: the outside seat saves {sound - held} bits, not 3")
    _, prices3, controls3, gates3 = _system("shadow-court")
    need = frozenset(gates3["govern"]["need"])
    actual = surprisal.gate_bits(prices3, controls3, need)
    declared = surprisal.gate_bits(prices3, set(), need)
    if declared - actual != 3:
        problems.append(f"undeclared-control: the hidden command saves {declared - actual} bits, not 3")
    grounded = [entrenchment.Rule("law", 3, "constitution"), entrenchment.Rule("constitution", 9, None)]
    looped = [entrenchment.Rule("law", 3, "council"), entrenchment.Rule("council", 9, "assembly"),
              entrenchment.Rule("assembly", 9, "council")]
    if entrenchment.apex(grounded) != ("constitution",) or \
            entrenchment.authority_height(grounded)["law"] != 1:
        problems.append("unowned-amendment: a grounded hierarchy should end in an ungoverned rule")
    if entrenchment.apex(looped) or not all(
            h is OMEGA for h in entrenchment.authority_height(looped).values()):
        problems.append("unowned-amendment: a loop of governors should have no apex and no finite height")
    breadth = 0
    for system in _golden()["systems"]:
        p = {n: _decode(v) for n, v in system["parties"].items()}
        e = {tuple(edge) for edge in system["controls"]}
        top = max((v for v in p.values() if v != math.inf), default=0)
        for gate in system["gates"].values():
            need = frozenset(gate["need"])
            covers = surprisal.minimal_covers(p, e, need)
            cost = surprisal.gate_bits(p, e, need)
            if cost != math.inf and cost > min(len(c) for c in covers) * top:
                problems.append(f"{system['name']}: cost {cost} exceeds size x dearest seat")
            breadth += 1
    return _verdict("the seven L2 clauses map onto parameters of these models", problems, [
        "exempt-reach: derived inside the viability model (see D06)",
        f"collusion-capturable: gate cost <= smallest cover x dearest seat, so breadth must grow "
        f"with the stake (checked on {breadth} gates)",
        f"unbounded-term: no repair, half of 2000 seats needed, life {termless:.4f} = ln 2 "
        "whatever the size",
        f"externally-held: an outside seat at price 0 saves {sound - held} bits (suzerain 3, "
        f"two-key {sound})",
        f"undeclared-control: a hidden command saves {declared - actual} bits (shadow-court "
        f"{actual} actual against {declared} declared)",
        "unowned-amendment: an ungoverned rule is an unamendable apex, and authority height is "
        "finite exactly when one exists (D14)",
        "self-funding-capture: NOT MODELLED here (reinvestment is positive feedback on the "
        "compromise probability); open",
        "and the diagnostic cause irreversible-without-ex-ante-check is the corrigible kernel (D07)"])


def _d19() -> Verdict:
    golden = _golden_gl()
    problems: list[str] = []
    if not golden["lob"]["proved"]:
        problems.append("Löb's axiom is not proved")
    guarded = unguarded = 0
    for ring in golden["rings"]:
        want = any(ring["guards"])
        if ring["proved"] != want:
            problems.append(f"ring of {ring['n']} with guards {ring['guards']}: proved "
                            f"{ring['proved']}, expected {want}")
        if not ring["proved"] and not ring["countermodel_nodes"]:
            problems.append(f"ring of {ring['n']} with guards {ring['guards']}: unproved with no "
                            "countermodel recorded")
        guarded += want
        unguarded += not want
    if not guarded or not unguarded:
        problems.append("the golden file lacks guarded or unguarded rings")
    return _verdict("a loop of checks is determined exactly when some link passes through proof",
                    problems, [
                        f"{len(golden['rings'])} rings of 1..5 checkers (hyperlogic "
                        f"{golden['hyperlogic_commit'][:12]}): {guarded} with a proof-check link are "
                        f"proved to approve together, {unguarded} loops of copying links are not, "
                        "each with a countermodel",
                        "Löb's axiom, □(□p → p) → □p, is proved by the same prover"])


def _twin_rooms() -> viability.Game:
    return viability.Game.from_table(
        nature=("n",),
        table={
            "hub": {"go-left": ("left",), "go-right": ("right",), "fall": ("x",)},
            "left": {"stay": ("left",)}, "right": {"stay": ("right",)}, "x": {"stay": ("x",)},
        },
        safe={"hub", "left", "right"},
    )


def _d20() -> Verdict:
    problems: list[str] = []
    settled = topdown = converged = games = 0
    firsts: list[int] = []
    for seed in range(40):
        game = viability.random_game(random.Random(11000 + seed), n=4, na=2, nb=2)
        star = viability.greatest_law(game, corrigible=False)
        for k in range(5):
            law = viability.random_law(game, random.Random(100 * seed + k), density=0.5)
            out = viability.settle(game, law, corrigible=False)
            settled += 1
            if not (set(out) <= set(star) and all(out[t] <= star[t] for t in out)):
                problems.append(f"seed {seed}: a settled law lies above the greatest law")
        permissive = {t: game.allowed(t) for t in game.states}
        if viability.settle(game, permissive, corrigible=False) != star:
            problems.append(f"seed {seed}: settling the permissive law is not the greatest law")
        topdown += 1
        if star:
            games += 1
            pooled, first = viability.converge(game, random.Random(seed), 300, corrigible=False)
            if not all(pooled[t] <= star[t] for t in pooled):
                problems.append(f"seed {seed}: the pooled law lies above the greatest law")
            if first is None:
                problems.append(f"seed {seed}: 300 starts did not pool to the greatest law")
            else:
                converged += 1
                firsts.append(first)
    rooms = _twin_rooms()
    star = viability.greatest_law(rooms, corrigible=False)
    left = viability.settle(rooms, {"hub": frozenset({"go-left"}), "left": frozenset({"stay"})},
                            corrigible=False)
    right = viability.settle(rooms, {"hub": frozenset({"go-right"}), "right": frozenset({"stay"})},
                             corrigible=False)
    pooled = {t: left.get(t, frozenset()) | right.get(t, frozenset()) for t in star}
    if left == star or right == star:
        problems.append("twin rooms: a start that mentions one room reached the whole optimum")
    if pooled != star:
        problems.append("twin rooms: two starts together did not pool to the greatest law")
    median = sorted(firsts)[len(firsts) // 2] if firsts else -1
    return _verdict("the optimal law is what many simulated law structures converge to", problems, [
        f"{settled} settled random laws on 40 games: none lies above the greatest law",
        f"{topdown} games: settling the fully permissive law gives the greatest law in one step",
        f"twin rooms: a start that mentions only the left room settles on {sorted(left)}; two starts "
        "together pool to the whole law",
        f"{converged} of {games} games with a kernel: 300 random starts pool to the greatest law "
        f"(median {median} starts, worst {max(firsts) if firsts else 0})"])


def _subsets(items: Iterable[str]) -> Iterator[frozenset[str]]:
    ordered = sorted(items)
    for size in range(len(ordered) + 1):
        for chosen in itertools.combinations(ordered, size):
            yield frozenset(chosen)


def _d21() -> Verdict:
    problems: list[str] = []
    games = informative = supporting_sets = 0
    for corrigible in (False, True):
        for seed in range(60):
            game = viability.random_game(random.Random(12500 + seed), n=4, home=True)
            kernel = viability.kernel(game, corrigible=corrigible)
            supporting = [y for y in _subsets(game.states)
                          if viability.maintains(game, y, corrigible=corrigible)]
            union = frozenset().union(*supporting)
            if union != kernel:
                problems.append(f"seed {seed}, corrigible {corrigible}: the sets that support themselves "
                                f"unite to {sorted(union)}, the kernel is {sorted(kernel)}")
            if not viability.maintains(game, kernel, corrigible=corrigible):
                problems.append(f"seed {seed}, corrigible {corrigible}: the kernel does not support itself")
            inductive = viability.inductive_kernel(game, corrigible=corrigible)
            if inductive:
                problems.append(f"seed {seed}, corrigible {corrigible}: the least fixed point is "
                                f"{sorted(inductive)}, not empty")
            games += 1
            informative += 0 < len(kernel) < len(game.states)
            supporting_sets += len(supporting)
    lone = viability.Game.from_table(nature=("n",), table={"a": {"hold": ("a",)}}, safe={"a"})
    if viability.kernel(lone, corrigible=False) != frozenset({"a"}) or \
            viability.inductive_kernel(lone, corrigible=False):
        problems.append("a single state that holds itself: its kernel is not {a}, or its inductive "
                        "reading is not empty")
    return _verdict("self-maintenance is a greatest fixed point: no inductive reading, and a set that "
                    "supports itself is its certificate", problems, [
                        f"{games} random games on 4 states, with and without corrigibility: the kernel equals "
                        f"the union of the {supporting_sets} sets that support themselves, found by "
                        "enumerating every subset, and the kernel is one of them",
                        f"{informative} of those games have a kernel that is neither empty nor everything",
                        f"the least fixed point of the same operator is empty in all {games}; one state "
                        "that holds itself has kernel {a} and an empty inductive reading"])


def _d22() -> Verdict:
    problems: list[str] = []
    core = ("hyperethics", "hyperphysics", "metamathethicology")
    data = family.snapshot()
    views = {name: (family.phases_of(name, data=data), family.cores_of(name, data=data))
             for name in family.VIEWS}
    if views["hard"][1]:
        problems.append(f"imports and declared dependencies form a cycle: {views['hard'][1]}")
    for name in ("formal", "substantive", "weighty", "all"):
        if not any(set(core) <= set(found) for found in views[name][1]):
            problems.append(f"the {name} view has no core holding {', '.join(core)}")
    if views["formal"][1] != [core]:
        problems.append(f"the formal view's cores are {views['formal'][1]}, not just {core}")
    widening = ("hard", "formal", "substantive", "all")
    for narrow, wide in zip(widening, widening[1:]):
        for repository, phase in views[narrow][0].items():
            if rank_key(views[wide][0][repository]) < rank_key(phase):
                problems.append(f"{repository}: widening the view from {narrow} to {wide} lowered its "
                                f"phase from {phase} to {views[wide][0][repository]}")
    reliances = {name: sum(len(d) for d in family.depends_on(name, data=data).values())
                 for name in family.VIEWS}
    live = family.live()
    live_cores = {name: family.live_cores_of(name, data=live) for name in family.LIVE_VIEWS}
    if not any(set(core) <= set(found) for found in live_cores["frequent"]):
        problems.append("the live scan's frequent mentions have no core holding "
                        f"{', '.join(core)}")

    def shown_cores(cores: list[tuple[str, ...]]) -> str:
        return "; ".join("+".join(found) if len(found) <= 3 else f"{len(found)} repositories"
                         for found in cores) or "none"

    def tally(name: str) -> str:
        phases, cores = views[name]
        zero = sum(p == 0 for p in phases.values())
        omega = sum(p is OMEGA for p in phases.values())
        return (f"{name} ({reliances[name]} reliances): {zero} repositories at phase 0, {omega} at ω; "
                f"cores: {shown_cores(cores)}")

    return _verdict("the family's reliance has a core with no foundation inside it, though what it imports "
                    "is well founded", problems, [
                        f"snapshot: hyperstratum {data['hyperstratum_commit'][:12]}, graph digest "
                        f"{data['source_digest'][:12]}: {len(data['nodes'])} repositories, "
                        f"{len(data['edges'])} recorded edges",
                        *(tally(name) for name in ("hard", "formal", "substantive", "weighty", "all")),
                        "widening the view from hard to formal to substantive to all never lowers a phase",
                        f"a second record, hyperstratum's live scan (commit {live['hyperstratum_commit'][:12]}, "
                        f"{len(live['fields'])} fields, {len(live['edges'])} edges): "
                        f"ten or more mentions or links: {shown_cores(live_cores['frequent'])}; "
                        f"any mention: {shown_cores(live_cores['mentions'])}; "
                        f"links only: {shown_cores(live_cores['links'])}"])


def _d23() -> Verdict:
    problems: list[str] = []
    controls = {"optimization": "the optimal law, optimising for survival", "viability": "the viability kernel",
                "persistence": "self-maintenance of a realm", "stability": "a stable fixed point",
                "ordinal": "the ordinal rank of a stage", "closure": "the closure of the nesting"}
    for topic, sentence in controls.items():
        if topic not in family.topics_of(sentence):
            problems.append(f"the {topic} pattern does not find {sentence!r}, so a count of zero would mean nothing")
    if family.topics_of("claude/optimistic-einstein branch"):
        problems.append("the table finds a topic in a word that only looks like one")
    items = family.open_items()
    forms = {item["form"] for item in items}
    if forms != {"bracket", "declaration"}:
        problems.append(f"the open items come in the forms {sorted(forms)}, not both conventions")
    tally = family.open_tally()
    if tally["topics"]["optimization"] or tally["topics"]["viability"]:
        problems.append(f"an open item now mentions optimization ({tally['topics']['optimization']}) or "
                        f"viability ({tally['topics']['viability']}): D18's evidence has changed")
    share = tally["untouched"] / tally["items"]
    if share < 0.8:
        problems.append(f"only {share:.0%} of the declared open items mention none of the six topics")
    fields = sorted({item["field"] for item in items})
    persistence = [f"{i['field']}/{i['file']}" for i in items if "persistence" in family.topics_of(i["text"])]
    return _verdict("by their wording almost none of the family's declared open items is about "
                    "optimization, viability or maintenance", problems, [
                        f"{tally['items']} declared open items in {len(fields)} of the "
                        f"{len(family.live()['fields'])} scanned fields: "
                        f"{sum(i['form'] == 'bracket' for i in items)} lines tagged [OPEN] and "
                        f"{sum(i['form'] == 'declaration' for i in items)} open declarations in .hm files",
                        "items mentioning each topic: " + ", ".join(f"{t} {n}" for t, n in tally["topics"].items()),
                        f"{tally['untouched']} of {tally['items']} ({share:.0%}) mention none of them",
                        f"persistence or maintenance is mentioned in {', '.join(persistence)}",
                        "every topic pattern finds a sentence written to contain it, and the word "
                        "'optimistic' is not read as optimization"])


def _territorial(rng: random.Random, leak: float) -> tuple[viability.Game, dict[str, str]]:
    """Six states, three to each of two parties; every move stays in its own territory except that, with
    probability `leak`, its outcome is any state at all."""
    states = [f"s{i}" for i in range(6)]
    owner = {s: "AB"[i // 3] for i, s in enumerate(states)}
    own = {p: [s for s in states if owner[s] == p] for p in "AB"}
    step = {(s, a, b): rng.choice(states) if rng.random() < leak else rng.choice(own[owner[s]])
            for s in states for a in ("x", "y") for b in ("n0", "n1")}
    safe = [s for s in states if rng.random() < 0.8] or states[:1]
    return viability.Game(states, ("x", "y"), ("n0", "n1"), step, safe=safe), owner


def _hamlets() -> viability.Game:
    return viability.Game.from_table(nature=("calm", "storm"),
                                     table={"a": {"hold": ("a", "b")}, "b": {"hold": ("b", "a")}})


def _d24() -> Verdict:
    problems: list[str] = []
    games = admissible = varied = 0
    for seed in range(40):
        rng = random.Random(13000 + seed)
        game = viability.random_game(rng, n=3, na=2, home=True)
        greatest = viability.greatest_law(game, corrigible=True)
        if not greatest:
            continue
        start = [sorted(greatest)[0]]
        before = {s: frozenset(a for a in sorted(game.allowed(s)) if rng.random() < 0.3) & greatest.get(s, frozenset())
                  for s in game.states}
        best = deals.opening_value(game, before, greatest, start)
        if not deals.admissible(best):
            problems.append(f"seed {seed}: the greatest corrigible law is not a coherent opening")
        values = [v for v in (deals.opening_value(game, before, after, start) for after in viability.all_laws(game))
                  if deals.admissible(v)]
        beaten = [v for v in values if not deals.at_least(best, v)]
        if beaten:
            problems.append(f"seed {seed}: a coherent opening beats the greatest corrigible law somewhere: {beaten[0]}")
        games += 1
        admissible += len(values)
        varied += len(set(values)) > 1

    game, base, menu, grouping = deals.three_doors()
    values = [deals.opening_value(game, base, after, ["hub"], dimension=grouping) for after in menu]
    dims = deals.dimensions(*values)
    vectors = [deals.vector(v, dims) for v in values]
    expected = [(11, 0, 10, 10, 1, 1, 1, 1), (0, 11, 10, 10, 1, 1, 1, 1), (5, 4, 4, 8, 1, 1, 1, 1)]
    if vectors != expected or deals.efficient(vectors) != [0, 1, 2]:
        problems.append(f"the three doors are not three coherent and mutually incomparable options: {vectors}")
    hidden = deals.price_invisible(vectors)
    if hidden != [2]:
        problems.append(f"the balanced door is not the one option no price will choose: {hidden}")
    chosen = [(wx, wy) for wx in range(1, 20) for wy in range(1, 20)
              if 2 in deals.select_by_price(vectors, (wx, wy) + (1,) * 6)]
    if chosen:
        problems.append(f"a price chose the balanced door: {chosen[0]}")

    cases = beaten_efficient = 0
    for seed in range(20):
        rng = random.Random(17000 + seed)
        for width in (2, 3):
            points = list(dict.fromkeys(tuple(rng.randint(0, 6) for _ in range(width))
                                        for _ in range(rng.randint(2, 6))))
            front = deals.efficient(points)
            for i, point in enumerate(points):
                weights = deals.supporting_weights(points, i)
                blend = deals.dominating_combination(points, i)
                cases += 1
                if (weights is None) == (blend is None):
                    problems.append(f"{points} option {i}: a supporting price and a dominating blend both exist or neither does")
                elif weights is not None:
                    top = sum(w * x for w, x in zip(weights, point))
                    if any(w <= 0 for w in weights) or any(sum(w * x for w, x in zip(weights, q)) > top for q in points):
                        problems.append(f"{points} option {i}: the supporting price does not rate it highest")
                else:
                    mixed = [sum(w * q[k] for w, q in zip(blend, points)) for k in range(width)]
                    if sum(blend) != 1 or any(w < 0 for w in blend) or any(m < x for m, x in zip(mixed, point)) \
                            or mixed == list(point):
                        problems.append(f"{points} option {i}: the blend does not strictly dominate it")
                    beaten_efficient += i in front

    sizes = []
    for price in (1, 5, 40):
        smallest = None
        for rooms in range(1, 30):
            trap_game = deals.trap_game(rooms)
            trap = deals.opening_value(trap_game, {"hub": {"stay"}}, deals.trap_opening(rooms), ["hub"])
            safe = deals.opening_value(trap_game, {"hub": {"stay"}}, {"hub": {"stay", "pause"}}, ["hub"])
            if deals.admissible(trap):
                problems.append(f"the trap of {rooms} rooms is a coherent opening")
            dimensions = deals.dimensions(trap, safe)
            weights = (1,) * len(dimensions) + (1, 1) + (price,) * 4
            if smallest is None and deals.priced(deals.vector(trap, dimensions), weights) \
                    > deals.priced(deals.vector(safe, dimensions), weights):
                smallest = rooms
        if smallest != (3 * price - 1) // 7 + 1:
            problems.append(f"with {price} on each coherence dimension the trap first outscores the safe opening at "
                            f"{smallest} rooms, not {(3 * price - 1) // 7 + 1}")
        sizes.append(smallest)

    menus = multiple = unseen = 0
    for seed in range(80):
        rng = random.Random(18000 + seed)
        random_world = viability.random_game(rng, n=5, na=3, home=False)
        people = {f"p{i}": frozenset(s for s in random_world.states if rng.random() < 0.5) for i in range(5)}
        options = []
        for size in (1, 2):
            for group in itertools.combinations(sorted(people), size):
                law = viability.greatest_law_within(random_world, frozenset().union(*(people[p] for p in group)),
                                                    corrigible=False)
                options.append(tuple(sum(1 for permitted in law.values() if a in permitted)
                                     for a in random_world.actions))
        points = list(dict.fromkeys(options))
        if len(points) < 3:
            continue
        menus += 1
        front = deals.efficient(points)
        lost = deals.price_invisible(points)
        multiple += len(front) > 1
        unseen += bool(lost)
        if lost and (len(front) < 3 or not set(lost) <= set(front)):
            problems.append(f"seed {seed}: an option hidden from every price is not one of at least three efficient options")

    hamlets, owner = _hamlets(), {"a": "A", "b": "B"}
    if not deals.win_win(hamlets, {}, {"a": {"hold"}, "b": {"hold"}}, ["a", "b"], owner):
        problems.append("the joint opening of the two hamlets is not a win for both")
    for single in ({"a": {"hold"}}, {"b": {"hold"}}):
        if deals.admissible(deals.opening_value(hamlets, {}, single, ["a", "b"])):
            problems.append(f"one hamlet opening alone ({sorted(single)}) is coherent")
    return _verdict("a deal is a vector of freedoms; within lawful existence nothing competes, and where options "
                    "do compete a price is blind to the balanced one", problems, [
                        f"{games} random games with a corrigible kernel, every law of each: {admissible} coherent "
                        f"openings, none better than the greatest corrigible law in any coordinate, and in {varied} "
                        "games they differ from one another, so there is a lattice to be best in",
                        f"three doors (x: {vectors[0][0]}, y: {vectors[1][1]}, balanced: {vectors[2][0]} and "
                        f"{vectors[2][1]}) are all coherent and incomparable; the balanced one is below the half-and-half "
                        "blend of the other two, so no strictly positive price chooses it",
                        f"{cases} options in random point sets: each has a supporting price or a dominating blend, "
                        f"never both (exact linear programs, certificates checked); {beaten_efficient} of them were "
                        "efficient and still strictly beaten by a blend",
                        f"a trap whose size scales freely: at {', '.join(map(str, (1, 5, 40)))} on each coherence "
                        f"dimension it first outscores a safe opening at {', '.join(map(str, sizes))} rooms (3p-1)//7+1",
                        f"base rate: among {menus} random menus of what coalitions of one or two people can certify, "
                        f"{multiple} offer more than one efficient option and {unseen} hide one from every price",
                        "the two hamlets: neither can open its own move alone, together both gain"])


def _d25() -> Verdict:
    problems: list[str] = []
    views = pairs = closed = strict = parts = 0
    for corrigible in (False, True):
        for seed in range(12):
            game = viability.random_game(random.Random(31000 + seed), n=4, home=True)
            kernel = viability.kernel(game, corrigible=corrigible)
            everything = list(_subsets(game.states))
            inside = {known: viability.kernel_within(game, known, corrigible=corrigible) for known in everything}
            for known in everything:
                views += 1
                if not inside[known] <= kernel:
                    problems.append(f"seed {seed}: a view of {sorted(known)} certifies {sorted(inside[known] - kernel)}, "
                                    "outside the kernel")
                supporting = frozenset().union(*(y for y in _subsets(known)
                                                  if viability.maintains(game, y, corrigible=corrigible)))
                if inside[known] != supporting:
                    problems.append(f"seed {seed}: a view of {sorted(known)} certifies {sorted(inside[known])}, not the "
                                    f"union of the sets it holds that support themselves, {sorted(supporting)}")
                if all(game.target(s, a, b) in known for s in known for a in game.allowed(s) for b in game.nature):
                    closed += 1
                    if inside[known] != kernel & known:
                        problems.append(f"seed {seed}: a view of {sorted(known)} closed under its own moves does not "
                                        "certify exactly what is in the kernel")
            for small, large in itertools.product(everything, repeat=2):
                if small <= large:
                    pairs += 1
                    if not inside[small] <= inside[large]:
                        problems.append(f"seed {seed}: knowing {sorted(large)} certified less than knowing {sorted(small)}")
            for left in everything:
                for right in everything:
                    if left & right:
                        continue
                    both = knowledge.pooled(game, {"x": left, "y": right}, corrigible=corrigible)
                    one = knowledge.certified(game, left, corrigible=corrigible)
                    other = knowledge.certified(game, right, corrigible=corrigible)
                    parts += 1
                    if not one | other <= both:
                        problems.append(f"seed {seed}: pooling {sorted(left)} with {sorted(right)} certified less than the parts")
                    strict += one | other != both
    sealed = leaky = gained = 0
    for seed in range(20):
        for leak in (0.0, 1 / 12):
            world, owner = _territorial(random.Random(seed), leak)
            territory = {p: {s for s in world.states if owner[s] == p} for p in "AB"}
            result = knowledge.surplus(world, territory, owner)
            crossing = bool(knowledge.crossings(world, owner))
            sealed += not crossing
            leaky += crossing
            if any(result.gains.values()):
                gained += 1
                if not crossing:
                    problems.append(f"seed {seed}: territories with no crossing gained from pooling")
    hamlets, owner = _hamlets(), {"a": "A", "b": "B"}
    result = knowledge.surplus(hamlets, {"A": {"a"}, "B": {"b"}}, owner)
    if knowledge.certified(hamlets, {"a"}) or knowledge.certified(hamlets, {"b"}) or not result.mutual \
            or knowledge.minimal_knowledge(hamlets, ("a", "hold")) != frozenset({"a", "b"}):
        problems.append("the two hamlets do not need each other: neither should certify its own move alone, "
                        "together both should")
    refuge = viability.Game.from_table(nature=("calm", "storm"),
                                       table={"a": {"hold": ("a", "b")}, "b": {"hold": ("b", "b")}})
    gift = knowledge.surplus(refuge, {"A": {"a"}, "B": {"b"}}, owner)
    if gift.mutual or gift.gains["A"] != frozenset({("a", "hold")}) or gift.gains["B"]:
        problems.append("the refuge is not a one-sided gift: only the hamlet that shelters should gain")
    return _verdict("what a coalition can certify is what its pooled knowledge supports itself on", problems, [
        f"{views} views (every subset of {2 * 12} random games on 4 states, with and without corrigibility): "
        "each certifies only what is in the kernel, exactly the union of the sets it holds that support themselves",
        f"{closed} views closed under their own moves certify exactly; knowing more never certified less in "
        f"{pairs} nested pairs",
        f"{parts} pairs of disjoint views: pooling never certified less than the parts, and certified strictly "
        f"more in {strict} of them",
        f"territories: {sealed} sealed (no allowed move crosses) and {leaky} with a crossing; {gained} gained from "
        "pooling, all with a crossing",
        "two hamlets that shelter in each other in a storm: alone neither certifies its move, together both do, and "
        "each needs the other's knowledge; if only one shelters the gain is one-sided"])


def _d26() -> Verdict:
    problems: list[str] = []
    makeable = instances = 0
    for seed in range(120):
        rng = random.Random(3000 + seed)
        names = [f"e{i}" for i in range(rng.randint(1, 6))]
        density = rng.choice([0.3, 0.5, 0.7])
        know = {f"p{i}": [n for n in names if rng.random() < density] for i in range(rng.randint(1, 6))}
        nodes = rng.sample(names, rng.randint(1, len(names)))
        people = sorted(know)
        held = {p: set(know[p]) for p in people}
        groups = [g for r in range(len(people) + 1) for g in itertools.combinations(people, r)
                  if set(nodes) <= set().union(set(), *(held[p] for p in g))]
        instances += 1
        if not groups:
            if comprehension.cover_number(know, nodes) is not None or comprehension.robust_quorum(know, nodes) is not None:
                problems.append(f"seed {seed}: a footprint nobody understands is said to have a cover or a quorum")
            continue
        makeable += 1
        smallest = min(len(g) for g in groups)
        if comprehension.cover_number(know, nodes) != smallest:
            problems.append(f"seed {seed}: the cover number is not the size of the smallest group that understands "
                            f"the footprint ({smallest})")
        everyone_at = {q: all(set(nodes) <= set().union(set(), *(held[p] for p in g))
                              for g in itertools.combinations(people, q)) for q in range(len(people) + 1)}
        threshold = min(q for q, ok in everyone_at.items() if ok)
        if comprehension.robust_quorum(know, nodes) != threshold or not all(everyone_at[q] for q in range(threshold, len(people) + 1)):
            problems.append(f"seed {seed}: the robust quorum is not the attendance at which whoever comes understands "
                            f"the footprint ({threshold})")
        if comprehension.necessary_voices(know, nodes) != frozenset(set.intersection(*(set(g) for g in groups))):
            problems.append(f"seed {seed}: the people no decision can be made without are not the sole witnesses")
    steps = [Fraction(k, 6) for k in range(7)]
    flips = 0
    for seed in range(60):
        rng = random.Random(7000 + seed)
        names = [f"e{i}" for i in range(rng.randint(2, 6))]
        know = {f"p{i}": [n for n in names if rng.random() < 0.5] for i in range(rng.randint(2, 6))}
        nodes = rng.sample(names, rng.randint(2, len(names)))
        rows = comprehension.sweep(know, nodes, steps)
        for lower, higher in zip(rows, rows[1:]):
            flips += lower.makeable and not higher.makeable
            if not set(higher.pool) <= set(lower.pool) or (not lower.makeable and higher.makeable) \
                    or (lower.cover is not None and higher.cover is not None and higher.cover < lower.cover):
                problems.append(f"seed {seed}: raising the barrier from {lower.barrier} to {higher.barrier} grew the "
                                "pool, lowered the cover or made an unmakeable decision makeable")
    touching = 0
    for seed in range(80):
        rng = random.Random(2000 + seed)
        names = [f"n{i}" for i in range(rng.randint(3, 7))]
        graph = {n: [m for m in names if m != n and rng.random() < 0.25] for n in names}
        for target in graph:
            nodes = comprehension.footprint(graph, [target]).nodes
            for group in entrenchment.cores(graph):
                if set(group) & nodes:
                    touching += 1
                    if not set(group) <= nodes:
                        problems.append(f"seed {seed}: the footprint of {target} holds part of the core {group} and not all")
    for x in (Fraction(1, 10), Fraction(1, 4), Fraction(2, 5)):
        prices = [surprisal.council_bits(m, (m + 1) // 2, x) for m in range(1, 22, 2)]
        if not all(a < b for a, b in zip(prices, prices[1:])):
            problems.append(f"at x = {x} a body with more seats is not dearer to capture")
    loads = comprehension.loads(family.depends_on("formal"))
    core = {"hyperethics", "hyperphysics", "metamathethicology"}
    sizes = sorted(len(fp.nodes) for fp in loads.values())
    if sizes[0] != 1 or sizes[-1] != 8 or any(not core <= loads[n].nodes or len(loads[n].nodes) != 8 for n in core):
        problems.append(f"the family's footprints in the formal view are not as recorded: sizes {sizes}")
    return _verdict("a decision is makeable only if its footprint is understood, which sets the quorum and the "
                    "barrier, and the barrier is paid for in capture resistance", problems, [
                        f"{instances} random instances ({makeable} makeable): the cover number, the robust quorum "
                        "(m - w + 1 for bus factor w) and the people no decision can be made without (the sole "
                        "witnesses) agree with brute force over every group of people",
                        f"60 random instances over 7 rising barriers: the pool never grew, the cover never fell, "
                        f"an unmakeable decision never became makeable; {flips} decisions became unmakeable as the barrier rose",
                        f"{touching} footprints that touch a core hold it whole, on 80 random graphs",
                        "Condorcet in bits, exact: for seat probability 1/10, 1/4 and 2/5 each pair of seats added to a "
                        "majority body (1 to 21 seats) makes it strictly dearer to capture, so every seat the barrier "
                        "removes costs capture resistance",
                        f"the family's repositories under the formal view: footprints of {sizes[0]} to {sizes[-1]} "
                        f"repositories, hyperethics, hyperphysics and metamathethicology 8 each (their core, the four "
                        "repositories it rests on and hyperlogic, which rests on it), i.e. changing any of them asks 8 "
                        "repositories' worth of understanding"])


def _d27() -> Verdict:
    problems: list[str] = []
    ghz, magic, chsh = entangled.ghz_game(), entangled.magic_square_game(), entangled.chsh_game()
    rational = entangled.chsh_strategy((1, 0), (0, 1), (Fraction(119, 169), Fraction(120, 169)),
                                       (Fraction(119, 169), Fraction(-120, 169)))
    witnesses = [(ghz, entangled.ghz_strategy(1, 1), Fraction(3, 4), Fraction(1)),
                 (magic, entangled.magic_square_strategy(), Fraction(8, 9), Fraction(1)),
                 (chsh, rational, Fraction(3, 4), Fraction(577, 676))]
    tables = {}
    for game, strategy, classical, quantum in witnesses:
        found = entangled.classical_value(game)
        if found != classical:
            problems.append(f"the classical value of {game.name} is {found}, not {classical}")
        tables[game.name] = entangled.table(game, strategy)
        got = entangled.table_value(game, tables[game.name])
        if got != quantum or got <= found:
            problems.append(f"the entangled strategy wins {game.name} with {got}, not {quantum}, or does not beat {found}")
    for a, b in [(1, 1), (3, 4), (5, 12), (1, 0), (1, 2), (4, 3)]:
        degree = Fraction(2 * a * b, a * a + b * b)
        value = entangled.quantum_value(ghz, entangled.ghz_strategy(a, b))
        if value != Fraction(1, 2) + degree / 2 or (value > Fraction(3, 4)) != (degree > Fraction(1, 2)):
            problems.append(f"the GHZ-type state ({a}, {b}) is worth {value}, not 1/2 + (degree {degree})/2, or its "
                            "advantage does not begin at degree 1/2")
    rng = random.Random(4)
    product_best = Fraction(0)
    for _ in range(8):
        parts = [(entangled.Gauss(rng.randint(-3, 3), rng.randint(-3, 3)), entangled.Gauss(rng.randint(-3, 3), 1))
                 for _ in range(3)]
        base = entangled.ghz_strategy(1, 1)
        value = entangled.quantum_value(ghz, entangled.Strategy(entangled.product_state(parts), base.measure, base.decode))
        product_best = max(product_best, value)
        if value > Fraction(3, 4):
            problems.append(f"a product state beat the classical value of GHZ with {value}")
    tsirelson = 0
    for _ in range(10):
        angles = []
        for _k in range(4):
            t = Fraction(rng.randint(-9, 9), rng.randint(1, 9))
            angles.append(((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)))
        value = entangled.quantum_value(chsh, entangled.chsh_strategy(*angles))
        tsirelson += (8 * value - 4) ** 2 <= 8
        if (8 * value - 4) ** 2 > 8:
            problems.append(f"a sampled CHSH strategy exceeded Tsirelson's bound: S = {8 * value - 4}")
    for game, _strategy, _classical, _quantum in witnesses:
        if not entangled.is_no_signalling(game, tables[game.name]):
            problems.append(f"in {game.name} one party's output distribution depends on another's input")
    box = entangled.pr_box()
    if not entangled.is_no_signalling(chsh, box) or entangled.table_value(chsh, box) != 1 \
            or entangled.table_value(chsh, box) <= Fraction(577, 676):
        problems.append("the PR box is not a no-signalling table that wins CHSH beyond every quantum strategy found")
    ghz_strategy = entangled.ghz_strategy(1, 1)
    after = entangled.collapse(ghz_strategy.state, [entangled.pauli("XII"), entangled.pauli("IXI"), entangled.pauli("IIX")],
                               (1, 1, 1))
    spent = entangled.quantum_value(ghz, entangled.Strategy(after, ghz_strategy.measure, ghz_strategy.decode))
    if spent > Fraction(3, 4):
        problems.append(f"after the measurements the state still wins GHZ with {spent}, above the classical value")
    ledgers = entangled.magic_square_ledgers()
    if ledgers:
        problems.append(f"{ledgers} classical ledgers of nine signs satisfy every row and column")
    return _verdict("entangled strategies beat classical ones in common-interest games, move no information, and "
                    "are consumed by use", problems, [
                        "GHZ (three parties): classical 3/4 by enumerating every deterministic strategy, entangled 1",
                        "magic square (two parties): classical 8/9, entangled 1; none of the 512 classical ledgers of "
                        "nine signs satisfies every row and column",
                        "CHSH (two parties): classical 3/4, rational angles on a Bell pair 577/676 (the best quantum "
                        "value (2+sqrt 2)/4 is irrational and recalled, not proved)",
                        "a GHZ-type state cos t|000> + sin t|111> is worth exactly 1/2 + (sin 2t)/2: it beats 3/4 only "
                        "beyond sin 2t = 1/2",
                        f"a product state never beat the classical value (best of 8: {product_best}); {tsirelson} of 10 "
                        "random CHSH strategies kept S squared within 8 (Tsirelson's bound, exact)",
                        "no party's output distribution depends on another's input in any of the three strategies; a "
                        f"PR box is no-signalling and wins CHSH with 1, beyond every quantum strategy; after one use the "
                        f"GHZ state is worth {spent} in the GHZ game, below the classical 3/4"])


def _d31() -> Verdict:
    problems: list[str] = []
    markets: list[tuple[str, dict, dict]] = []  # (label, endowment, strict rankings)
    for n in (2, 3):  # every profile of strict rankings; the identity endowment stands for every endowment up to names
        agents, goods = list(range(n)), [f"g{i}" for i in range(n)]
        for choice in itertools.product(itertools.permutations(goods), repeat=n):
            markets.append((f"{n} agents, rankings {choice}", dict(zip(agents, goods)), dict(zip(agents, choice))))
    exhaustive = len(markets)
    rng = random.Random(31)
    for n, count in ((4, 30), (5, 5)):
        goods = [f"g{i}" for i in range(n)]
        for k in range(count):
            held = goods[:]
            rng.shuffle(held)
            markets.append((f"random market {k} of {n} agents", dict(zip(range(n), held)),
                            {a: tuple(rng.sample(goods, n)) for a in range(n)}))
    three = [m for m in markets[:exhaustive] if m[0].startswith("3 agents")]
    for label, endowment, prefs in markets:
        outcome = rival.top_trading_cycles(endowment, prefs)
        if not rival.individually_rational(endowment, prefs, outcome):
            problems.append(f"{label}: an agent ends below where it began")
        if not rival.pareto_efficient(endowment, prefs, outcome):
            problems.append(f"{label}: the outcome is not Pareto efficient")
        if rival.core_allocations(endowment, prefs) != [outcome]:
            problems.append(f"{label}: the strict core is not exactly the outcome")
        if rival.profitable_misreports(endowment, prefs):
            problems.append(f"{label}: an agent gains by misreporting its ranking")
        if bool(rival.movers(endowment, outcome)) == rival.pareto_efficient(endowment, prefs, endowment):
            problems.append(f"{label}: trade did not happen exactly when the starting allocation is inefficient")
        if problems:
            break
    counts = rival.survey(3)
    trades, gameable = counts["trade"], counts["gameable_boston"]
    if gameable == 0:
        problems.append("the checker finds no profitable misreport in the Boston mechanism, so it could not have "
                        "found one in this one either")
    audits = accepted = records = 0
    for label, endowment, prefs in markets[exhaustive:]:
        honest = rival.rounds(endowment, prefs)
        for agent in endowment:
            audits += 1
            if not rival.audit(agent, prefs[agent], endowment, honest):
                problems.append(f"{label}: agent {agent}'s audit of the honest record fails")
    for label, endowment, prefs in three[::6]:
        expected = rival.top_trading_cycles(endowment, prefs)
        for record in rival.all_records(list(endowment)):
            records += 1
            if all(rival.audit(a, prefs[a], endowment, record) for a in endowment):
                accepted += 1
                described = rival.allocation_of(endowment, record)
                if described != expected:
                    problems.append(f"{label}: a record every agent accepts describes {described}, not the top "
                                    "trading cycles outcome")
    starts = 0
    for n in (3, 4, 5):
        goods = [f"g{i}" for i in range(n)]
        same = {a: tuple(goods) for a in range(n)}
        for held in itertools.permutations(goods):
            starts += 1
            if rival.top_trading_cycles(dict(zip(range(n), held)), same) != dict(zip(range(n), held)):
                problems.append(f"with identical rankings the starting allocation {held} moved")
    spread = rival.endowment_spread({a: tuple(f"g{i}" for i in range(4)) for a in range(4)})
    if spread != {a: (0, 3) for a in range(4)}:
        problems.append(f"with identical rankings an agent's best and worst place over all starting allocations is "
                        f"{spread}, not the whole range 0 to 3")
    vectors = {"x-rich": (5, 1), "y-rich": (1, 5)}
    weights = {"A": (3, 1), "B": (1, 3)}
    held = {"A": "y-rich", "B": "x-rich"}
    swapped = rival.top_trading_cycles(held, {a: rival.rank_by_weights(vectors, w) for a, w in weights.items()})
    if swapped != {"A": "x-rich", "B": "y-rich"}:
        problems.append(f"agents who weigh the coordinates differently did not swap two goods of equal worth at the "
                        f"price (1, 1): {swapped}")
    return _verdict("with rival freedoms an exchange without money is individually rational, efficient, in the core and "
                    "truthful, each party can audit it, and the starting allocation decides what is left", problems, [
                        f"{exhaustive} strict ranking profiles of two and three agents, every one, and "
                        f"{len(markets) - exhaustive} seeded markets of four and five agents: the outcome is "
                        "individually rational, Pareto efficient and in the strict core, it is the only allocation in the "
                        "core, and no agent gains by misreporting",
                        f"trade happens in {trades} of the {counts['profiles']} profiles of three agents and in exactly "
                        f"those where the starting allocation is not Pareto efficient; the other {counts['still']} keep "
                        "what they have; nobody ends below where they began",
                        f"the Boston mechanism, a common rule, is gameable in {gameable} of the {counts['profiles']} "
                        "profiles of three agents, and the same checker finds the lie",
                        f"{audits} audits of honest records pass, each from the agent's own ranking and the public record "
                        f"of cycles alone; of {records} records of three agents on {len(three[::6])} markets, the "
                        f"{accepted} that every agent accepts all describe the top trading cycles outcome",
                        f"with identical rankings nothing moves under any of {starts} starting allocations of three to "
                        "five agents, so what an agent gets over all starting allocations runs from its first choice to "
                        "its last (places 0 to 3 for four agents): the start decides",
                        "two goods of equal worth at the price (1, 1), (5, 1) and (1, 5), are swapped by two agents "
                        "who weigh the coordinates differently, with no money; at any common price the goods held are "
                        "worth the same after as before, so the gain is in the private weights only"])



def _d32() -> Verdict:
    problems: list[str] = []

    def holds(endowment: dict, prefs: dict, order: tuple) -> bool:
        """A pecking order, restated from its definition so that this check does not lean on the module it checks:
        each agent's good is its favourite among the goods held by it and by every agent after it."""
        return all(min((endowment[b] for b in order[k:]), key=prefs[a].index) == endowment[a]
                   for k, a in enumerate(order))

    markets: list[tuple[str, dict, dict]] = []
    for n in (2, 3):  # every start and every strict profile
        agents, goods = list(range(n)), [f"g{i}" for i in range(n)]
        for held in itertools.permutations(goods):
            for choice in itertools.product(itertools.permutations(goods), repeat=n):
                markets.append((f"{n} agents, start {held}, rankings {choice}",
                                dict(zip(agents, held)), dict(zip(agents, choice))))
    exhaustive = len(markets)
    rng = random.Random(32)
    for n, count in ((4, 40), (5, 8)):
        goods = [f"g{i}" for i in range(n)]
        for k in range(count):
            held = goods[:]
            rng.shuffle(held)
            markets.append((f"random market {k} of {n} agents", dict(zip(range(n), held)),
                            {a: tuple(rng.sample(goods, n)) for a in range(n)}))
    loops = identity_markets = identity_efficient = 0
    for label, endowment, prefs in markets:
        before = len(problems)
        outcome = rival.top_trading_cycles(endowment, prefs)
        loop = standing.envy_loop(endowment, prefs)
        efficient = rival.pareto_efficient(endowment, prefs, endowment)
        loops += loop is not None
        if endowment == {0: "g0", 1: "g1", 2: "g2"}:  # the identity start of three agents, as in the survey
            identity_markets += 1
            identity_efficient += efficient
        if not ((loop is not None) == bool(rival.movers(endowment, outcome)) == (not efficient)):
            problems.append(f"{label}: envy has a loop, someone trades and the start is not Pareto efficient "
                            "together or not at all, and here they disagree")
        order = standing.pecking_order(endowment, prefs)
        tried = any(holds(endowment, prefs, o) for o in itertools.permutations(endowment))
        if (order is not None) != efficient or tried != efficient:
            problems.append(f"{label}: a pecking order exists (found, and by trying every order) exactly when the "
                            "start is Pareto efficient, and here it does not")
        if order is not None and not holds(endowment, prefs, order):
            problems.append(f"{label}: the pecking order found is not one")
        if not standing.frozen(outcome, prefs):
            problems.append(f"{label}: envy still has a loop after the exchange")
        if len(problems) > before:
            break

    cases: list[tuple[dict, tuple]] = []  # (rankings, the agents from the wealthiest down)
    for choice in itertools.product(itertools.permutations(["g0", "g1", "g2"]), repeat=3):
        for ranked in itertools.permutations(range(3)):
            cases.append((dict(zip(range(3), choice)), ranked))
    exhaustive_wealth = len(cases)
    left_alone = 0
    for n, profiles, orders in ((4, 12, 24), (6, 6, 20), (9, 3, 10)):
        goods = [f"g{i}" for i in range(n)]
        for _ in range(profiles):
            prefs = {a: tuple(rng.sample(goods, n)) for a in range(n)}
            cases.extend((prefs, tuple(rng.sample(range(n), n))) for _ in range(orders))
    for index, (prefs, ranked) in enumerate(cases):
        before = len(problems)
        n = len(ranked)
        wealth = {a: n - place for place, a in enumerate(ranked)}
        label = f"wealth start of {n} agents in the order {ranked}"
        start = standing.wealth_start(wealth, prefs)
        if sorted(start.values()) != sorted(prefs[0]):
            problems.append(f"{label}: not a rearrangement of the goods")
            break
        moved = rival.top_trading_cycles(start, prefs) != start
        left_alone += index < exhaustive_wealth and not moved
        if (n <= 6 and not rival.pareto_efficient(start, prefs, start)) or moved:
            problems.append(f"{label}: not Pareto efficient, or top trading cycles moves it")
        if not holds(start, prefs, ranked):
            problems.append(f"{label}: the wealth order is not a pecking order of it")
        if any(wealth[t] <= wealth[a] for a, targets in standing.envy_graph(start, prefs).items() for t in targets):
            problems.append(f"{label}: an agent envies one who is no wealthier")
        if len(problems) > before:
            break
    try:
        standing.wealth_start({0: 1, 1: 1}, {0: ("g0", "g1"), 1: ("g0", "g1")})
    except ValueError:
        pass
    else:
        problems.append("equal wealth was accepted, though it has no pick order")

    def instance(n: int, d: int) -> tuple[dict, dict, tuple, dict]:
        names = [f"g{k}" for k in range(n)]
        goods = {g: tuple(rng.randint(0, 6) for _ in range(d)) for g in names}
        price = tuple(rng.randint(1, 6) for _ in range(d))
        weights = {a: tuple(rng.randint(1, 6) for _ in range(d)) for a in range(n)}
        held = names[:]
        rng.shuffle(held)
        return dict(zip(range(n), held)), goods, price, weights

    at_price = one = along = grid = 0
    for _ in range(40):
        endowment, goods, price, weights = instance(rng.choice((2, 3, 4, 5)), rng.choice((1, 2, 3)))
        at_price += 1
        if standing.thaw_outcome(endowment, goods, price, weights, 0) != endowment:
            problems.append("at the price alone someone traded, though every agent then ranks alike")
            break
    for _ in range(30):
        endowment, goods, price, weights = instance(rng.choice((3, 4, 5)), 1)
        one += 1
        if any(piece.movers for piece in standing.thaw_pieces(endowment, goods, price, weights)):
            problems.append("with one coordinate someone traded at some setting, though one number ranks every "
                            "agent alike")
            break
    for _ in range(30):
        endowment, goods, price, _unused = instance(rng.choice((3, 4, 5)), rng.choice((2, 3)))
        along += 1
        strength = {a: rng.randint(1, 9) for a in endowment}  # one strength per agent, along the price
        weights = {a: tuple(strength[a] * c for c in price) for a in endowment}
        if any(piece.movers for piece in standing.thaw_pieces(endowment, goods, price, weights)):
            problems.append("with weights that are positive multiples of the price someone traded at some setting")
            break
    for _ in range(25):
        before = len(problems)
        endowment, goods, price, weights = instance(rng.choice((3, 4, 5)), rng.choice((2, 3)))
        pieces = standing.thaw_pieces(endowment, goods, price, weights)
        found = standing.thaw_threshold(endowment, goods, price, weights)
        for step in range(13):
            lam = Fraction(step, 12)
            direct = standing.thaw_outcome(endowment, goods, price, weights, lam)
            stretch = next(p for p in pieces if p.low < lam < p.high or p.low == p.high == lam)
            if direct != stretch.allocation:
                problems.append(f"the outcome at {lam} differs from the outcome the dial gives for the stretch "
                                "that holds it, so a ranking changed inside a stretch")
                break
            if (found is None or lam < found.lam) and rival.movers(endowment, direct):
                problems.append(f"someone moved at {lam}, below the dial's threshold {found}")
                break
            grid += 1
        if len(problems) > before:
            break

    third, half = Fraction(1, 3), Fraction(1, 2)
    vectors = {"G1": (3, 1), "G2": (1, 2)}
    swap = ({"A": "y", "B": "x"}, {"y": (1, 5), "x": (5, 1)}, (1, 1), {"A": (3, 1), "B": (1, 3)})
    gap = ({"A": "G2", "B": "G1"}, vectors, (1, 1), {"A": (3, 1), "B": (1, 4)})
    flipped = ({"A": "G2", "B": "G1"}, {"G2": (1, 2), "G1": (3, 1)}, (1, 1), {"A": (3, 1), "B": (1, 4)})
    reversal = ({"A": "G2", "B": "G1"}, vectors, (1, 1), {"A": (1, 3), "B": (1, 4)})

    def traded(lam: Fraction) -> bool:
        return bool(rival.movers(reversal[0], standing.thaw_outcome(*reversal, lam)))

    for name, got, expected in (
            ("the swap at equal worth, threshold", standing.thaw_threshold(*swap), standing.Threshold(0, False)),
            ("the swap does not reverse", standing.thaw_reverses(*swap), False),
            ("a gap of 1 in worth, breakpoints", standing.thaw_breakpoints(*gap[1:]), (third,)),
            ("a gap of 1 in worth, threshold", standing.thaw_threshold(*gap), standing.Threshold(third, False)),
            ("the same gap with the other good listed first, threshold", standing.thaw_threshold(*flipped),
             standing.Threshold(third, True)),
            ("the reversal, breakpoints", standing.thaw_breakpoints(*reversal[1:]), (third, half)),
            ("the reversal, threshold", standing.thaw_threshold(*reversal), standing.Threshold(third, False)),
            ("the reversal reverses", standing.thaw_reverses(*reversal), True),
            ("the reversal trades at 1/3, 1/2 and 3/5", (traded(third), traded(half), traded(Fraction(3, 5))),
             (False, True, False))):
        if got != expected:
            problems.append(f"worked dial example ({name}): expected {expected}, got {got}")

    priced = witnessed = barren = delicate = stretched = 0
    for _ in range(120):
        before = len(problems)
        d, n = rng.choice((1, 2, 3)), rng.choice((2, 3, 4))
        price = tuple(rng.randint(1, 5) for _ in range(d))
        mode, base = rng.choice(("random", "equal", "along", "parallel")), tuple(rng.randint(1, 5) for _ in range(d))
        strength = [rng.randint(1, 4) for _ in range(n)]  # one strength for each agent, not for each coordinate
        if mode == "random":
            weights = {a: tuple(rng.randint(1, 5) for _ in range(d)) for a in range(n)}
        elif mode == "equal":
            weights = {a: base for a in range(n)}
        else:
            weights = {a: tuple(strength[a] * c for c in (price if mode == "along" else base)) for a in range(n)}
        priced += 1
        witness = standing.thaw_witness(price, weights)
        if (witness is not None) != standing.can_thaw(price, weights):
            problems.append("can_thaw and thaw_witness disagree about whether any goods can trade")
        elif witness is not None:
            witnessed += 1
            delicate += witness.lam == Fraction(1, 2)  # weights parallel but unequal, none along the price
            first, second = witness.agents
            pair = {first: weights[first], second: weights[second]}
            if standing.thaw_outcome(witness.endowment, witness.goods, price, pair, witness.lam) != {
                    first: "y", second: "x"}:
                problems.append("the witness for the weights does not trade at its setting")
        else:
            barren += 1
            stretched += len(set(weights.values())) > 1  # unequal multiples of the price
            goods = {f"g{k}": tuple(rng.randint(0, 6) for _ in range(d)) for k in range(n)}
            held = list(goods)
            rng.shuffle(held)
            if any(piece.movers for piece in standing.thaw_pieces(dict(zip(range(n), held)), goods, price, weights)):
                problems.append("goods traded though the weights are all equal or all along the price")
        if len(problems) > before:
            break

    expected = {"profiles": identity_markets, "ladders": identity_efficient,
                "loops": identity_markets - identity_efficient, "wealth_starts": exhaustive_wealth,
                "wealth_frozen": left_alone}
    try:
        counts = standing.survey(3)
    except ValueError as error:
        counts = expected
        problems.append(f"the survey of three agents failed: {error}")
    if counts != expected:
        problems.append(f"the survey of three agents differs from the counts made here: {counts}, not {expected}")
    sample = standing.dial_survey()
    return _verdict("from a given start an exchange without money moves exactly the loops of envy, a ladder made by "
                    "wealth is frozen under it, and what thaws a ladder is a difference in the direction of what "
                    "agents value", problems, [
                        f"{exhaustive} markets of two and three agents (every start and every strict profile) and "
                        f"{len(markets) - exhaustive} seeded markets of four and five: envy has a loop in {loops} of "
                        "them, and in exactly those someone trades, the start is not Pareto efficient and no order "
                        "of the agents is a pecking order; in the other "
                        f"{len(markets) - loops} the start is a ladder; after one exchange envy has no loop in any",
                        f"{counts['profiles']} profiles of three agents with the identity start: {counts['ladders']} "
                        f"ladders and {counts['loops']} loops, the split into nobody and somebody trading; "
                        f"{len(cases)} wealth starts (every wealth order of every profile of three agents, seeded "
                        "ones of four to nine): each is a rearrangement of the goods, is Pareto efficient where "
                        "that can be tried (up to six), is left where it is by top trading cycles, has the wealth "
                        "order as a pecking order, and has envy pointing only toward the wealthier",
                        f"the dial: at the price alone nobody trades in {at_price} instances; with one coordinate "
                        f"nobody ever trades, at any setting, in {one}; with weights along the price none in "
                        f"{along}; the outcome on every stretch between breakpoints agrees with direct evaluation at "
                        f"{grid} settings, and nobody moves below the threshold",
                        f"of {sample['instances']} seeded instances with two or three coordinates {sample['trade']} "
                        f"trade at some setting, and in {sample['reverse']} of those trade opens and closes again as "
                        "the dial rises, so the dial is not a one-way door",
                        f"{priced} prices with weights of four kinds (random, all equal, all along the price, parallel with "
                        f"different strengths): {witnessed} have a witness, two goods and a setting at which two agents "
                        f"swap ({delicate} of them parallel but unequal, with the setting at the middle of the dial), "
                        f"and in the other {barren}, exactly those with weights all equal or all along the price "
                        f"({stretched} of them unequal multiples of the price), no goods and no start trade at any "
                        "setting",
                        "two goods of equal worth at the price (1, 1) are swapped at any divergence at all "
                        "(threshold 0, never attained); a gap of 1 in worth is closed exactly at 1/3, and the tie "
                        "there is broken by the order the goods are listed in; one instance trades on the stretch "
                        "above 1/3 up to and including 1/2 and nowhere else"])


# ---------------------------------------------------------------------------
# The ledger.
# ---------------------------------------------------------------------------

LEDGER: tuple[Decision, ...] = (
    Decision(
        "D01", "A price is the surprisal of a compromise", Tag.FRAME,
        "Replace money by bits. The price of compromising a party is -log2 of the probability that "
        "it is compromised; a gate costs the cheapest total price of a set of parties that passes "
        "it; a stake is the log2 of how many independent attempts a prize attracts.",
        ("Keep currency units: rejected, because you asked for no monetary construction and the "
         "units are arbitrary.",
         "Use probability itself as the price: rejected, because probabilities multiply, so gate "
         "costs would not add and hyperethics' cover search would not apply.",
         "Effort or time as the price: rejected, because it brings a wage back in."),
        "Someone who recomputes hyperethics' gate costs with prices read as bits gets the same "
        "numbers, and the exact probability of taking each gate lies between 2^-cost and "
        "(number of minimal covers) x 2^-cost.",
        "One gate whose cost differs from hyperethics', or one gate whose exact probability falls "
        "outside that bracket.",
        "For a real body: the frequency with which a coalition of seats is compromised together "
        "should equal the product of its members' frequencies when there is no common cause. Needs "
        "incident data; none is held here.",
        "surprisal.gate_bits, sandwich_gate; tests/test_surprisal.py; data/golden_l2_gate_bits.json",
        _d01),
    Decision(
        "D02", "A common principal is a measurable loss of surprisal", Tag.FRAME,
        "Treat shared command as positive association between compromises, measured by "
        "additivity_gap = bits(product of marginals) - bits(joint), and count roots, not names.",
        ("Treat every seat as independent: rejected, because it overstates the price of any set of "
         "seats that answer to one principal, which is hyperethics' common-principal cause.",
         "Give correlation a free parameter rho: rejected, because rho has no observable meaning "
         "and a shared principal does."),
        "For a principal at 3 bits commanding two seats, the gap is exactly 3 bits: the same "
        "discount hyperethics finds when one Boss at price 3 replaces two seats at price 3 each "
        "(gate cost 6 falls to 3).",
        "A shared-principal system whose gap differs from the cheaper-cover discount.",
        "Compromises across the seats of one body that share an appointer or a funder: their joint "
        "frequency exceeds the product of the marginals by the predicted amount. Needs data.",
        "surprisal.additivity_gap, common_principal_joint; tests/test_surprisal.py",
        _d02),
    Decision(
        "D03", "Independence understates capture risk when compromise is positively associated",
        Tag.FRAME,
        "Keep the independent-seat (mean-field) model for thresholds and report the direction of its "
        "error instead of hiding it.",
        ("Solve a specific ring or graph exactly: deferred. Exact enumeration reaches about a dozen "
         "bodies, below any size where a threshold is sharp.",
         "Choose a correlation parameter: rejected, as in D02."),
        "Group the seats of a body under shared principals, keeping every seat's marginal fixed. "
        "With the quorum above the mean number compromised, capture becomes MORE likely than the "
        "independent formula says; with the quorum below it, LESS likely. A witness running the "
        "grouped computation sees both directions.",
        "A grouping, with the quorum above the mean, whose capture probability is lower than the "
        "independent one (positive association cannot do that).",
        "Positive association across seats of real bodies, measured. The thresholds in this package "
        "are then upper bounds on the pressure a body tolerates, not estimates. Needs data.",
        "surprisal.capture_probability, Block; quorum.tail; tests/test_surprisal.py",
        _d03),
    Decision(
        "D04", "Existence is membership in the viability kernel, and persistence is not goodness",
        Tag.FRAME,
        "Existence means: the system can keep itself inside a safe set for ever against any nature. "
        "The optimal structure is the least restriction that preserves this, and nothing more.",
        ("Existence as expected lifetime: rejected, because it needs a probability on nature and "
         "loses the worst-case guarantee.",
         "Existence as a utility to maximise: rejected, because it builds in a value, which is the "
         "family's open GC-5."),
        "The greatest law permits some stable arrangements a participant would call tyranny. A "
        "witness should find, in a game with a stable bad state, that the state is in the kernel and "
        "the move into it is permitted. Anything that excludes it is an extra axiom, not a "
        "consequence of existence.",
        "A derivation, inside the game, that removes the stable bad state from the kernel without "
        "adding a welfare term.",
        "None available: that stable bad regimes exist is historical, not a witness for or against "
        "this model. Why persistence should count as optimality is D16.",
        "viability.kernel, greatest_law; tests/test_viability.py",
        _d04),
    Decision(
        "D05", "The law is the greatest sustaining law: unique, canonical, naming no one", Tag.FORM,
        "Define the optimal law as the pointwise-greatest assignment of permitted actions that "
        "keeps the system in its kernel for ever. It forbids exactly what ends existence, permits "
        "everything else, and mentions no participant.",
        ("A voted or weighted law: rejected, because the game contains no weights to vote with.",
         "The smallest law that works (permit only what is needed): rejected, because it is the "
         "maximum restriction, not the least."),
        "On every finite game and every start state, a law is sustaining exactly when everything it "
        "permits at the states it reaches lies in the greatest law; renaming states, actions or "
        "nature choices renames the law the same way, so it cannot favour a participant the game "
        "does not already single out.",
        "A game, a law and a start where the two disagree, or a renaming that changes the law.",
        "None needed: a theorem of the model, checked by enumerating every law on the sample.",
        "viability.greatest_law, sustaining, all_laws, Game.relabel; tests/test_viability.py",
        _d05),
    Decision(
        "D06", "Free will is the freedom to take any permitted action: a law limits what is "
        "permitted, not what is chosen", Tag.FRAME,
        "Model the choice among permitted actions as taking each with positive probability (full "
        "support), and nature likewise.",
        ("Best-response agents: rejected, because an agent that sees a fatal move and avoids it "
         "needs no law, which would make the question 'which laws are needed' empty.",
         "An adversarial agent: rejected, because that is nature."),
        "Survival with probability one exactly when the law is sustaining. One permission outside "
        "the greatest law at a state the system reaches puts survival strictly below one: exempting "
        "an author from a prohibition that protects existence is fatal. This derives, inside the "
        "model, the L2 clause exempt-reach.",
        "A sustaining law with survival below 1, or a non-sustaining law with survival 1.",
        "Exposure: an actor with unconstrained access to a catastrophic action should use it with "
        "probability growing toward one with the time it holds the access. Testable on institutional "
        "data; not tested here.",
        "viability.survival_probability; tests/test_viability.py",
        _d06),
    Decision(
        "D07", "Corrigibility is staying where an overseer can still bring the system home",
        Tag.FRAME,
        "Give the game a home set and overseer correction moves, and require the system to stay "
        "inside the set from which home is reachable. A move that leaves it is irreversible.",
        ("Utility indifference or an off-switch game: deferred, because it needs payoffs, which "
         "this package avoids.",
         "Require every action to be reversible: rejected, because it forbids learning; only "
         "reachability of home matters."),
        "The corrigible kernel lies inside the kernel, strictly inside for a system that can "
        "survive by passing through a point of no return, and the corrigible law forbids exactly "
        "the move through it.",
        "A game whose corrigible kernel is not inside its kernel, or where the corrigible law "
        "permits a move out of the home-reachable set.",
        "L2's irreversible-without-ex-ante-check clause: bodies holding irreversible powers with no "
        "prior check should show worse recoverability. Needs data.",
        "viability.reach_home, kernel(corrigible=True); tests/test_viability.py",
        _d07),
    Decision(
        "D08", "A rank is a certificate a verifier can check without trusting the prover", Tag.FORM,
        "Represent the phase of a state as a rank in omega+1 and verify it by the local equation "
        "rank(s) = 1 + max over actions of min over nature of rank(next).",
        ("Stage counts as real-valued time: rejected, because it has no well-foundedness and no "
         "certificate.",
         "Report only the kernel: rejected, because it loses how long a doomed state has."),
        "The true ranks satisfy the equation, and changing any single entry to any other value "
        "breaks it somewhere.",
        "A game where a wrong ranking satisfies the equation.",
        "A third party holding only the game and the claimed ranks can accept or reject in one "
        "pass. That is the witness: the claim is checkable without trusting the claimer.",
        "viability.check_ranks; tests/test_viability.py",
        _d08),
    Decision(
        "D09", "Revitalization is self-repair, and redundancy cannot replace it", Tag.FRAME,
        "Seats are attacked at rate lambda and repaired at rate nu, repair happens only while the "
        "body still holds (the body repairs itself), and the body is lost when q seats are down.",
        ("Repair by an outside agent: rejected, because an outside repairer is an exempt exterior "
         "(L2's externally-held).",
         "Repair always available: rejected, because it removes the point of no return."),
        "Three regimes by rho = lambda/(lambda+nu) against theta = q/m: life grows like "
        "exp(m D)/sqrt(m), D the Bernoulli KL divergence, if rho < theta; tends to "
        "ln(rho/(rho-theta))/(lambda+nu) whatever m is if rho > theta; grows like "
        "(1/2)(ln 2m + gamma)/(lambda+nu) at rho = theta = 1/2. A body that is never repaired dies "
        "in ln(1/(1-theta))/lambda however many seats it has.",
        "Life that grows with m past the margin, or does not grow exponentially inside it.",
        "Institutions without member renewal (unbounded terms) should fail on a time scale that "
        "does not depend on their size. Needs data.",
        "renewal.mean_lifetime, log_mean_lifetime, exponent; tests/test_renewal.py",
        _d09),
    Decision(
        "D10", "A checker that commands and a checker that only vetoes are one equation in dual "
        "variables, and vetoing is worth more", Tag.FRAME,
        "Model an oversight link as COMMAND (compromise propagates: p = pi + (1-pi) T(p)) or VETO "
        "(compromise needs both: p = pi T(p)).",
        ("Only COMMAND, hyperethics' control relation: rejected, because it makes every check as "
         "weak as its commander.",
         "A blended link with a mixing parameter: deferred; the two extremes are the clean cases."),
        "1 - G_command(1-s) = G_veto*(s) exactly, with the dual quorum; for a self-dual quorum the "
        "two critical pressures sum to exactly 1; and the veto threshold exceeds the command "
        "threshold at every odd majority.",
        "A grid point where the identity fails, or an odd majority where veto does not exceed "
        "command.",
        "Comparative: oversight bodies that appoint and remove (command) should fail at lower attack "
        "pressure than ones that can only block. Needs data.",
        "loop.step, threshold; tests/test_loop.py",
        _d10),
    Decision(
        "D11", "The balanced quorum is the odd majority, critical at exactly one half", Tag.FRAME,
        "Call a quorum balanced when capturing and blocking are equally hard (q = m-q+1). That makes "
        "the quorum map self-dual.",
        ("Weight safety above liveness (q above the majority): kept as the general case; the "
         "balanced point is the equal weighting.",
         "Minimise expected loss under a stated prior: deferred, because it needs a prior."),
        "T(1/2) = 1/2 exactly for every odd majority; the price of such a council at x = 1/2 is "
        "exactly one bit; the critical point moves down when the quorum is lowered and up when it "
        "is raised.",
        "A non-majority quorum critical at 1/2, or a majority that is not.",
        "None: a theorem of the model. Whether real bodies should be balanced is a value choice, "
        "and open.",
        "quorum.critical_point, tail; surprisal.council_bits; tests/test_quorum.py",
        _d11),
    Decision(
        "D12", "Closing the loop costs little while healthy and everything near the threshold",
        Tag.FRAME,
        "Report the gap between a finite hierarchy with an unchecked apex and the folded loop (the "
        "closing price), and the delay before collapse past the threshold.",
        ("Report only fixed points: rejected, because it hides the finite-time behaviour real "
         "institutions live in.",),
        "At pi = 0.1 for a 5-seat, 3-quorum command loop, one level of oversight is already within "
        "2% of the loop's limit. Just past the threshold the number of levels before collapse grows "
        "like (pi - pi*)^(-1/2): 100x closer, 10x longer.",
        "A closing price that is not small in the healthy phase, or a delay exponent other than "
        "-1/2.",
        "Critical slowing down: near a saddle-node, recovery from perturbation slows and variance "
        "rises (the generic early-warning signature of a critical transition, recalled from the "
        "literature, not checked here). Seeing it in an institution is the witness.",
        "loop.closing_price, levels_to_cross; tests/test_loop.py",
        _d12),
    Decision(
        "D13", "A phase is an ordinal rank; transfinite ranks appear only when the structure nests",
        Tag.FRAME,
        "Represent the progress of a self-closing process as a rank in a well-ordered set: natural "
        "numbers and omega for a finite game, ordinals below epsilon_0 for processes with "
        "hereditary structure (Goodstein).",
        ("Real-valued Lyapunov functions: kept for smooth dynamics, but they carry no "
         "proof-theoretic information.",
         "Surreal or hyperreal phases: rejected, because nothing here needs them."),
        "Every rank is an ordinal comparing as integers do; ranks of finite games stay below "
        "omega+1; Goodstein from 4 starts at omega^omega and its ordinals strictly descend for "
        "hundreds of terms while the first terms are 4, 26, 41, 60, 83, 109.",
        "A descent that fails to strictly decrease, or a finite game with a rank that is neither a "
        "number nor omega.",
        "None: this is a representation. Whether it is the right shared one across the hyperfields is "
        "D18.",
        "ordinal.Ordinal, goodstein_sequence, check_descent; tests/test_ordinal.py",
        _d13),
    Decision(
        "D14", "Protection is the weakest link, and with no sovereign there is no finite rank",
        Tag.FORM,
        "Model a constitution as rules each governed by another rule, with a price for the procedure "
        "each defines. Protection of a rule is the cheapest procedure above it; authority height is "
        "the number of governors above it.",
        ("Protection as the strength of the rule's own procedure: rejected, because it ignores the "
         "route through the amendment clause (Ross).",
         "A rank assigned by hand: rejected, because on a loop none exists."),
        "A finite height anywhere means some rule nothing governs; with every rule governed, every "
        "height is omega; and an entrenched clause amendable by a cheap clause is protected by the "
        "cheap clause.",
        "A governance graph where every rule is governed and some height is finite, or a clause "
        "protected by more than its cheapest ancestor.",
        "None needed: a theorem of the model, checked over every graph on up to four rules and "
        "against an independent shortest-path search.",
        "entrenchment.protection, authority_height, cycles; tests/test_entrenchment.py",
        _d14),
    Decision(
        "D15", "The seven L2 clauses map onto parameters of these models; one is derived, five are "
        "modelled and one is not", Tag.FRAME,
        "Read each clause of hyperethics' rule as a parameter or a derivation here: exempt-reach as "
        "D06, collusion-capturable as gate breadth, unbounded-term as zero repair, externally-held "
        "as a seat at price 0, undeclared-control as the gap between declared and actual command, "
        "unowned-amendment as an ungoverned rule (D14). Self-funding-capture is not modelled.",
        ("Leave the two packages unrelated: rejected, because the rule is supposed to follow from "
         "the optimality structure, and unrelated packages cannot show that.",),
        "The mapped clauses reproduce hyperethics' numbers: an outside seat saves 3 bits, a hidden "
        "command saves 3 bits, a body with no repair dies in ln 2 / lambda whatever its size, and "
        "gate cost never exceeds the smallest cover times the dearest seat.",
        "A clause whose model-side quantity disagrees with the hyperethics number it should equal.",
        "None needed for the arithmetic. Whether the model describes any real institution is "
        "hyperethics' open nd-adequacy-of-the-capture-model.",
        "tests/test_surprisal.py, tests/test_renewal.py, the golden file",
        _d15),
    Decision(
        "D16", "Why persistence should count as optimality is unresolved", Tag.OPEN,
        "Treat the step from 'the law that preserves existence' to 'the optimal law' as adopted, not "
        "derived, and say so wherever the word optimal is used.",
        ("Treat persistence as obviously good: rejected, because D04 exhibits a stable regime it "
         "does not exclude.",
         "Add a welfare term now: deferred, because any weights are a value the family has not "
         "declared."),
        "If the step were derivable, some principle internal to the framework would exclude every "
        "stable bad state D04 builds, without a new axiom. A witness would find that principle.",
        "The principle; or a proof that none exists inside the game.",
        "No witness is available inside this package. This is the family's GC-5, restated for the "
        "viability game.",
        "viability.py module docstring, L0 GC-5 in hyperethics",
        None),
    Decision(
        "D17", "That physical reality adheres to an optimal hyperethical structure cannot be "
        "witnessed here", Tag.HYPER,
        "Record the proposal that physical stability conditions are kernel conditions and physical "
        "tipping points are saddle-node thresholds of closure maps, as a proposal, and attach no "
        "check to it.",
        ("State it as a result of the models: rejected, because nothing here touches a physical "
         "measurement.",
         "Drop it: rejected, because it is the part of your question about reality, and it has a "
         "stated form that a measurement could refute."),
        "A bistable physical system whose microscopic parameters are measured independently should "
        "have a measured tipping point equal to the threshold computed from them, with no free fit.",
        "A system with independently measured parameters whose tipping point disagrees with the "
        "computed threshold.",
        "No witness is available at the physics level. Bistability, hysteresis and critical slowing "
        "down are observed in real systems; matching a threshold to measured microscopic parameters "
        "is the missing step.",
        "loop.threshold; L2 and L0 in hyperethics",
        None),
    Decision(
        "D18", "Hyperoptimization as the closing catalyst across the family is a claim to test on "
        "the family's own data", Tag.OPEN,
        "Record the claim as open, with the test it implies, instead of asserting it; and run the "
        "first parts of that test: the phase of each repository in the family's recorded reliance "
        "(D22), the wording of the family's declared open items (D23), and a reading by hand of "
        "the six open declarations in its specifications.",
        ("Assert that it closes the open items: rejected, because the map from open items to the "
         "repositories that would close them is not built.",),
        "If hyperoptimization closes cycles across the hyperfields, the dependency graph of the "
        "family's open items has an authority-height structure (D14) in which it is the node that "
        "turns a cycle of unclosed obligations into a grounded chain, and closing items lowers an "
        "ordinal rank (D13). Weaker forms can be read now: the cross-reliances inside a core of "
        "the family's recorded reliance (D22) should be ones that what this package supplies "
        "(viability, certificates, phase) mediates, and some of the declared open items should be "
        "about what it computes (D23).",
        "The map built from hyperstratum's registry shows no such node, or closing items does not "
        "lower the rank; or the open items of a core concern something else. Read as of this "
        "snapshot, the open items of the ethics, physics and metamathethicology core are mostly "
        "about time, reality and the electrical transport of the will, with a few that this "
        "package touches, and by wording none is about optimization or viability (D23): that is "
        "evidence against the claim for that core.",
        "First passes, 2026-10-11, over 122 declared open items in 11 of the 24 pinned fields (116 "
        "lines tagged [OPEN] found by hyperstratum's scan, and 6 open declarations in .hm "
        "specifications): none is closed. By wording (D23) none mentions optimization or "
        "viability. Read by hand, the items this package touches are: hyperethics' consistency "
        "self-maintenance paradox (D21); metamathethicology's stage semantics, whose availability "
        "order is the phase of a formation graph (E4) while the starting stage stays a "
        "convention; hyperethics' adequacy of the capture model (D03: the direction of the error "
        "is computed, its size is not); and hyperphysics' law ranking, which asks whether "
        "priority in resolution is the dependence order of its own table, and E4 shows the "
        "table's dependence order leaves two pairs of laws unordered, so it cannot be that order "
        "alone. Not touched: hyperlogic's gate, implicit control, the charge-constant law, and the "
        "time-bubble, quantum-gravity and naming items that make up most of the rest. 13 of the "
        "24 fields declare no open items in either form. There is no witness for the claim "
        "itself: the map from items to the repositories that would close them is not done.",
        "entrenchment.phases, cores, incomparable_pairs; family.py; the reading by hand is not "
        "reproducible",
        None),
    Decision(
        "D19", "A loop of checks is determined exactly when some link passes through proof", Tag.FRAME,
        "Read a checker that approves only what it can prove safe as the provability modality of GL, "
        "and an approval that copies another's as a plain link. A ring of such links has a "
        "self-consistent outcome, unique and the same for every member, if and only if at least one "
        "link is a proof-check.",
        ("Trust as a number that propagates (D10): kept; it answers how much pressure a loop "
         "survives, not whether its outcome is determined.",
         "A monotone-operator fixed point: gives existence but not uniqueness for loops of copying "
         "links, which is exactly where the outcome is indeterminate.",
         "A game-theoretic equilibrium of mutual oversight: rejected, because it needs payoffs."),
        "A theorem prover for GL, run on every ring of up to five checkers, proves that the members "
        "all approve whenever the defining equations are theorems and one link goes through the box, "
        "and fails with a countermodel on every ring of copying links. Löb's axiom is proved by the "
        "same prover. Determinacy is not soundness: the unique outcome is that everyone approves, "
        "whether or not approval was warranted.",
        "A ring with a proof-check link that the prover cannot settle, or a ring of copying links that "
        "it settles.",
        "Bodies whose reviewers each approve because another reviewer approved, and none checks the "
        "substance, can reach opposite outcomes on the same proposal with nothing in their structure "
        "to choose between them. Observable as flipped outcomes under noise. Needs data.",
        "scripts/regenerate_golden_gl.py; data/golden_gl_loops.json; hyperlogic.provability",
        _d19),
    Decision(
        "D20", "The optimal law is what many simulated law structures converge to, and no single "
        "start need reach it", Tag.FRAME,
        "Define convergence by a settling map: prune what a law cannot survive on, then permit "
        "every action that stays inside what does. Pool the settled laws of many random starts.",
        ("Evolutionary dynamics with mutation and selection: deferred, because it adds a population "
         "model whose parameters mean nothing here.",
         "Gradient or best-response learning of the law: rejected, because it needs payoffs.",
         "Iterate from the fully permissive law: kept, as the one-step special case; it is the "
         "greatest law itself."),
        "A settled law never lies above the greatest law; settling is monotone and idempotent; "
        "settling the fully permissive law gives the greatest law in one step; a start that "
        "mentions only one of two separate safe regions is stuck below the optimum; and the pool of "
        "settled random starts reaches the greatest law. Greatest fixed points are found from above, "
        "so what converges is the diversity of starts, not any one of them.",
        "A settled law with a permission outside the greatest law, or a population that pools above "
        "it, or a single small start that finds a region its law never mentions.",
        "A population of institutions with varied rules in one environment: the rules of those that "
        "persist should lie inside one maximal rule set, and pooling the persisting rules of enough "
        "varied institutions should approach it, while a single institution can be stuck below. "
        "Needs data.",
        "viability.settle, converge, random_law, restricted_kernel; tests/test_viability.py",
        _d20),
    Decision(
        "D21", "Self-maintenance is a greatest fixed point: it has no inductive reading, and a set "
        "that supports itself is its certificate", Tag.FORM,
        "Read a system's maintenance of its own safe condition as membership in the greatest fixed "
        "point of F(Y) = the states with an action that sends every nature choice into Y. A set "
        "that supports itself (Y inside F(Y)) is a finite certificate for each state in it, and "
        "the kernel is the union of all such sets. Read hyperethics' consistency "
        "self-maintenance paradox (a realm's maintenance of its consistency appears to presuppose "
        "the consistency maintained, and the maintainer is itself inside the realm) as having "
        "this shape. Do not claim it is resolved.",
        ("An inductive reading, the least fixed point built up from a base: rejected, because it "
         "is empty for every game with a nature choice (nothing is supported by nothing), so the "
         "paradox would make every realm unmaintainable.",
         "A proof-theoretic reading, the realm proves its own consistency: kept for loops of "
         "checks (D19), rejected here, because a theory that proves its own consistency is "
         "inconsistent (Gödel's second theorem, recalled) and a loop of provers is determined "
         "without being sound.",
         "Exempt the maintainer from the realm's condition: rejected, because an exemption is "
         "fatal (D06) and an ungoverned maintainer is the corrupting apex (D14)."),
        "In every sampled game, with and without corrigibility, the kernel equals the union of "
        "every subset that supports itself, found by enumerating all subsets, and the kernel "
        "supports itself; the least fixed point of the same operator is empty. So a witness needs "
        "only a self-supporting set to certify existence, and no base case to start from.",
        "A game in which the union of the self-supporting sets differs from the kernel, or in "
        "which the kernel does not support itself, or in which building up from the empty set "
        "reaches a non-empty set.",
        "For a real realm a self-supporting set is a list of states with, for each, a move that "
        "keeps every outcome inside the list; it can be audited by exhibiting the list. That the "
        "realm's kernel is non-empty, and that its consistency is a safe condition at all and not "
        "something a theory proves, are claims about the realm that this model does not make.",
        "viability.maintains, inductive_kernel; tests/test_maintenance.py; hyperethics "
        "L0_creation.hm nd-consistency-self-maintenance",
        _d21),
    Decision(
        "D22", "The family's reliance has a core with no foundation inside it, though what it "
        "imports is well founded", Tag.FRAME,
        "Read the family's recorded citations, dependencies and imports (hyperstratum's family "
        "graph) as reliance, source relying on target, and give each repository the ordinal phase "
        "of its reliance, which is authority height for many governors (D14). Read it under five "
        "nested views that differ in which edges count: hard (imports and declared dependencies), "
        "formal (hard, or a citation in a specification or source file), substantive (any citation "
        "beyond the navigation files every repository carries), weighty (three or more citing "
        "files) and all.",
        ("Imports and declared dependencies only: kept as the hard view; a build cannot loop, so "
         "it cannot show the question.",
         "Every citation as a dependency: kept as the all view, rejected as the headline, because "
         "navigation lists put every sibling in every repository and a large cycle there is "
         "expected and says little.",
         "A centrality score from edge weights: rejected, because it ranks where the question is "
         "whether a bottom exists.",
         "A dependency list kept by the owners: better data, and it does not exist yet."),
        "Imports and declared dependencies form no cycle. Under every view that keeps the "
        "specifications, hyperethics, hyperphysics and metamathethicology lie in a common core: "
        "their phases are ω, and so are the phases of every repository that relies on them. "
        "Widening the view never lowers a phase. The same three repositories lie in a common core "
        "of a second, independent record, hyperstratum's live scan of the 24 pinned fields, when "
        "only reliances of ten or more links or mentions are kept. A witness who regenerates the "
        "family graph and the live scan (scripts/regenerate_golden_family.py and "
        "regenerate_golden_live.py) gets the same structure if the family has not changed and a "
        "different one if it has.",
        "A regenerated graph in which the formal view has no core, or in which imports form a "
        "cycle, or in which widening the edge set lowers a phase; or a regenerated live scan in "
        "which the three repositories are not in a common core of the frequent mentions.",
        "The owners of the three repositories can say whether the circle is vicious: whether each "
        "one's results depend on another's for their truth or only mention it. A citation is "
        "evidence of mention, not of need. The static graph lists 21 repositories while "
        "hyperstratum pins 24 (its HS-002): verifier, taxonomy-of-deception, "
        "speedrunning-social-development, throwaway-papers and meta-morphoscripted-origination are "
        "absent from it; the live scan covers all 24 but cannot tell a citation from a mention. "
        "The map from the family's open items to the repositories that would close them is not "
        "built (D18).",
        "family.py; entrenchment.phases, cores; data/golden_family_graph.json, "
        "data/golden_family_live.json; scripts/regenerate_golden_family.py, "
        "regenerate_golden_live.py; tests/test_family.py, tests/test_phases.py",
        _d22),
    Decision(
        "D23", "By their wording almost none of the family's declared open items is about "
        "optimization, viability or maintenance", Tag.FRAME,
        "Test D18's weakest form first: if hyperoptimization closes the family's declared open "
        "items, some of them should be about what this package computes. Read the declared open "
        "items (every line tagged [OPEN] that hyperstratum's scan finds, and every open "
        "declaration in a .hm specification) by wording, with a fixed table of keywords for six "
        "topics (optimization, viability, persistence, stability, ordinal, closure), chosen "
        "before the items were read against it.",
        ("Classify each item by hand: better, and not done; the table is its reproducible proxy, "
         "and every hit was read by eye.",
         "Skip the test because the claim is structural and not lexical: rejected as the only "
         "test; D22 is the structural one and this is the cheap complement.",
         "Read the items as a graph of what would close each: needs the owners (D18)."),
        "Few items are about what this package computes: none mentions optimization or viability, "
        "two mention persistence or maintenance (hyperethics' consistency self-maintenance "
        "paradox, which D21 treats, and a time-machine line that uses the word in another sense), "
        "and about 87% mention none of the six topics. The ones that touch it: the consistency "
        "self-maintenance paradox (D21), metamathethicology's stage semantics, hyperphysics' law "
        "ranking (E4), and ordinatics' limit rules.",
        "An owner's reading that finds optimization-shaped content in many items phrased in other "
        "words; or a regenerated scan whose tally differs; or a topic pattern that fails its own "
        "control sentence.",
        "The owners can classify the items by hand and say which a result of this kind would "
        "close. The proxy ignores synonyms and counts a hit for a word used in any sense: the "
        "maintenance of a time machine is not self-maintenance.",
        "family.open_items, open_tally, TOPICS; data/golden_family_live.json; "
        "scripts/regenerate_golden_live.py; tests/test_family.py",
        _d23),
    Decision(
        "D24", "A deal is a vector of freedoms, not a price: within lawful existence nothing competes, and "
        "where options do compete a price is blind to the balanced one", Tag.FRAME,
        "Value an opening of the law (a law that permits more than the one before) by three kinds of "
        "coordinate and no scalar: the freedom it allows, counted along each dimension of freedom (by "
        "default the action); the freedom that freedom enables, which is the states it makes reachable "
        "and the permissions the law grants there; and four yes-or-no dimensions of coherence with "
        "lawful existence: the system survives for ever (exists), stays where an overseer can bring it "
        "home (corrigible), permits nothing anywhere that the greatest corrigible law forbids (lawful), "
        "and takes no permission from anyone (monotone). Compare by Pareto dominance, which is partial. "
        "Treat coherence as a constraint, admitted only if every dimension holds, never as a weight.",
        ("A scalar price: rejected twice over. A linear price never chooses an efficient option that "
         "lies strictly inside the blend of others, and any finite price on lawfulness is outbid by a "
         "large enough lawless opening.",
         "A welfare for each participant: rejected, as in D16. Nothing here ranks outcomes for people, "
         "and Arrow-type results would apply the moment it did.",
         "A lexicographic order with coherence first: kept, as the admission rule. It is not a scalar "
         "and it loses nothing the vector knew.",
         "Count permissions only, in one dimension: rejected, because it cannot say that one opening "
         "leads on to others."),
        "Among coherent openings the greatest corrigible law is best in every coordinate at once, in "
        "every sampled game with every law enumerated, so within lawful existence there is nothing to "
        "price (the deal-language form of D05). A door opened onto a room that permits nothing is not "
        "coherent, so an opening is closed under what it enables. Where options do compete, as in a menu of three "
        "doors (all x, all y, a balanced mix), the balanced one is efficient and no strictly positive "
        "price chooses it; and for every option in random point sets exactly one of a supporting price "
        "and a dominating blend exists, each checkable without trusting the solver. For a price p on "
        "each coherence dimension a trap of (3p-1)//7+1 rooms outscores the safe opening, for every p "
        "tried. Competing options are rare among random menus (the check records the count).",
        "A game in which a coherent opening is not dominated by the greatest corrigible law; or a menu "
        "in which an option the exact test calls price-invisible is chosen by some strictly positive "
        "price; or an option for which a supporting price and a dominating blend both exist, or neither "
        "does; or a trap in which the lawless opening does not eventually outscore.",
        "For a real body: deals priced in one currency should systematically omit the balanced options, "
        "and parties who can add lawlessness to a deal should be able to buy it. The base rate here is "
        "that competing options are rare, so the claim is about where scarcity enters and not about how "
        "often. Needs data; none is held here.",
        "deals.py; tests/test_deals.py",
        _d24),
    Decision(
        "D25", "What a coalition can certify is what its pooled knowledge supports itself on, and "
        "across a border asymmetry makes a deal necessary", Tag.FORM,
        "Model a participant's view as the game with everything it cannot see treated as unsafe: "
        "ignorance is pessimism. What it can certify as permitted is the greatest law of that view. "
        "Exchange knowledge, not prices: pooled views certify at least what each did alone, and "
        "strictly more where a freedom's safety depends on a state one party cannot see. Parties hold "
        "territories (the states they act at); the surplus from pooling exists only where an allowed "
        "move crosses from one territory into another.",
        ("Probabilistic beliefs and expected value: rejected, because the question is whether a freedom "
         "is lawful and not what it is worth, and beliefs need a prior this model has no basis for.",
         "Trust in what the other party says: rejected, because a claim of safety nobody can check is "
         "D19's determinacy without soundness.",
         "A zero-knowledge or minimal-disclosure protocol: deferred. The smallest knowledge a permission "
         "needs is a self-supporting set (D21), so disclosure can be limited to that set; hiding even "
         "that needs cryptography this package does not carry."),
        "In every sampled game, with and without corrigibility, every subset of the states as a view: it "
        "certifies only what is in the kernel, exactly the union of the sets it holds that support "
        "themselves (the certificate of D21); a view closed under its own moves certifies exactly; "
        "knowing more never certifies less; pooling two disjoint views certifies at least the union of "
        "the parts and often strictly more; territories with no crossing gain nothing from pooling, and "
        "every gain sampled had a crossing. Two hamlets that each shelter in the other in a storm are "
        "the smallest case: alone neither can certify its own move, together both can, and each needs "
        "the other's knowledge. If only one shelters, the other gives knowledge and gains nothing.",
        "A view that certifies something outside the kernel, or fails to certify a state it holds a "
        "self-supporting set around; more knowledge certifying less; pooled views certifying less than "
        "their parts; a gain from pooling between territories no allowed move crosses.",
        "For real parties: a freedom's safety can be established by exhibiting a finite list of "
        "situations with, for each, a move that stays inside the list, and the list can be checked by "
        "someone who did not make it. Whether real parties hold such lists, or would disclose them, is "
        "not shown. Knowledge is non-rival here, so a deal is weakly win-win by construction; the "
        "strategic value of private information is outside the model (D30).",
        "knowledge.py; viability.kernel_within, greatest_law_within; tests/test_knowledge.py",
        _d25),
    Decision(
        "D26", "A decision is makeable only if its footprint is understood: the cover sets the quorum, "
        "coverage sets the barrier, and the barrier is paid for in capture resistance", Tag.FRAME,
        "The footprint of a decision is what its targets rest on, to any depth (the material), together "
        "with what rests on its targets, to any depth (the impact), in a graph of reliance. People "
        "understand parts of it. The decision is makeable only if the people in the pool between them "
        "understand all of it. The quorum required is the larger of a strict majority of the pool (the "
        "balanced quorum of D11) and the robust quorum m - w + 1, where w is the fewest people who "
        "understand the least-understood part (the bus factor): the attendance at which whoever comes "
        "understands everything. The barrier is the share of the footprint a person must understand to "
        "take part; the highest barrier a decision can bear is the last at which the pool still "
        "understands it.",
        ("A quorum fixed apart from the decision: rejected, because it ignores the premise.",
         "Decision rights for experts only: rejected as a default, because the barrier removes seats and "
         "a smaller body is cheaper to capture (Condorcet), computed in bits, and because whoever "
         "certifies understanding then holds the gate (D29).",
         "Everyone affected decides, with no gate: kept as the barrier-0 limit.",
         "The minimum cover alone as the quorum: rejected, because a cover exists but the people who "
         "attend may not be it; the robust quorum is what makes any attendance enough."),
        "On random instances, checked against every group of people: the cover number is the size of "
        "the smallest group that understands the footprint; the robust quorum is exactly the attendance "
        "at which any group understands it; the people no decision can be made without are exactly "
        "those who alone understand some part, and the affected who alone know their own stake are "
        "among them until someone else learns it; raising the barrier shrinks the pool, never lowers "
        "the cover, and can make a decision unmakeable, never the reverse; a footprint holds whole "
        "cores; each pair of seats added to a majority body makes it strictly dearer to capture. In the "
        "family's recorded reliance (formal view) changing hyperethics, hyperphysics or "
        "metamathethicology asks 8 repositories' worth of understanding: their core, the four "
        "repositories it rests on and hyperlogic, which rests on it.",
        "An instance in which the robust quorum is not the exact attendance threshold; a cover below "
        "what brute force finds; a necessary person who is not a sole witness; a barrier raise that "
        "lowers the cover or makes an unmakeable decision makeable; a footprint that holds part of a "
        "core; a body that becomes cheaper to capture when seats are added.",
        "Understanding is a set handed in here. A test of understanding that cannot be gamed does not "
        "exist in this package (D29), and the family's reliance is a record of citations and not of "
        "need (D22). For a real body: decisions should stall exactly where nobody understands part of "
        "their footprint, and the quorum they ask for should track the bus factor. Needs data; none is "
        "held here.",
        "comprehension.py; entrenchment.reliance_closure; tests/test_comprehension.py",
        _d26),
    Decision(
        "D27", "Entangled strategies beat classical ones in common-interest games, and they move no "
        "information", Tag.FRAME,
        "Read an entangled decision as parties who each hold part of a shared entangled state, each "
        "choose a measurement by an input only they know, and give an output; read an exclusively "
        "win-win deal as a game in which everyone wins or loses together. The value of the shared state "
        "is how far the win probability exceeds the best classical strategy, shared randomness "
        "included. All of it is computed in exact rationals over the Gaussian rationals.",
        ("Games whose parties have opposed interests (the quantum prisoner's dilemma): rejected, because "
         "the conclusion that entanglement dissolves the dilemma depends on the strategy set allowed "
         "(recalled from the literature, not modelled), and a deal that is not common-interest is not "
         "exclusively win-win.",
         "A floating-point simulation: rejected, because exactness is this package's standard and these "
         "values are exactly rational.",
         "Entanglement as a channel: rejected, and checked, because in every strategy here a party's "
         "output distribution depends only on its own input."),
        "Three witnesses, exact: GHZ (three parties) is 3/4 classically and 1 with the GHZ state; the "
        "Mermin-Peres magic square (two parties) is 8/9 classically and 1 with two entangled pairs, and "
        "no classical ledger of nine signs satisfies every row and column; CHSH is 3/4 classically and "
        "577/676 with rational angles on a Bell pair. For a GHZ-type state cos t |000> + sin t |111> the "
        "value is exactly 1/2 + (sin 2t)/2, so it beats the classical 3/4 only beyond a threshold in its "
        "degree of entanglement. A product state never beats the classical value. No output depends on "
        "another party's input. A PR box is no-signalling and wins CHSH every time, which no quantum "
        "strategy does. After one use the state is worth no more than a classical one.",
        "A product state, or a classical strategy, that beats the classical value; a strategy in which "
        "one party's output distribution depends on another's input; a sampled quantum strategy whose "
        "CHSH sum exceeds Tsirelson's bound; a GHZ-type state whose value is not the formula.",
        "A Bell-type experiment on the devices a protocol would use, with the locality loophole closed, "
        "and a cost per use for making and distributing the entangled pairs. Nothing here has been run "
        "on a physical system (D28).",
        "entangled.py; tests/test_entangled.py",
        _d27),
    Decision(
        "D28", "Institutional decisions can be entangled in the physical sense", Tag.HYPER,
        "Propose that a governance protocol could give the devices that make its decisions (not the "
        "people) a shared entangled resource, so that parties who cannot or will not communicate "
        "coordinate the opening of a freedom with a probability no classical protocol reaches, in a "
        "game where all of them win or lose together. The value would be the advantage of D27, spent "
        "once per use.",
        ("Classical correlation devices (shared random keys): their value is already inside the "
         "classical value of D27.",
         "Communication with a secure-computation protocol: reaches certainty, and needs the parties to "
         "communicate; entanglement is for where they cannot or will not.",
         "No such resource: the default, and the right one until a witness exists."),
        "A witness would see a protocol in which separated devices, each choosing a measurement by its "
        "party's private input, win a common-interest game above the classical value, certified by a "
        "Bell-type test with the locality loophole closed, and a stated cost per use in energy, time "
        "and hardware, since entanglement is consumed by measuring it and decoheres.",
        "A Bell test on the deployed devices that does not violate the classical bound; or a cost per "
        "use above what communicating would cost; or a protocol in which the advantage vanishes once "
        "the devices are as noisy as the hardware makes them.",
        "No witness. Nothing here has been run on a physical system, the cost of making and "
        "distributing entangled pairs is not modelled, and the mapping from an institution's decisions "
        "to measurements on a quantum state is a proposal.",
        "entangled.py (the exact games only)",
        None),
    Decision(
        "D29", "Understanding has no instrument here: the barrier is only as honest as the test", Tag.OPEN,
        "Treat each person's understanding as a set handed in. Keep the barrier a parameter, with a "
        "derived upper limit (the highest barrier the decision can bear), and claim nothing about how "
        "understanding is measured.",
        ("Credentials or reputation as a stand-in: rejected, because whoever grants them then holds the "
         "gate, which is the capture the barrier was meant to prevent.",
         "A prediction test, in which the examinee predicts the effects of a change on a model held back "
         "from them and is checked against it: proposed, not built.",
         "Sortition plus education: not modelled."),
        "A test of understanding would be one that a person passes exactly when they understand the part "
        "of the footprint it covers, that cannot be passed by memorizing its questions, and that is run "
        "by someone with no stake in who passes.",
        "An examinee who passes without understanding, or one who understands and fails, at a rate that "
        "matters; or a certifier whose pass rate follows who it would like to admit.",
        "Not done. No instrument exists in this package, so a witness cannot be produced here.",
        "comprehension.py (the sets are inputs)",
        None),
    Decision(
        "D30", "Win-win holds by construction until freedoms can be rival", Tag.OPEN,
        "Model only permissions that are not rival: granting one leaves every other holder's permissions "
        "as they were. Then a deal is weakly win-win by definition, 'win' means a permission nobody "
        "could certify before, and information is non-rival too. Say so, and do not call it a discovery.",
        ("Top trading cycles for rival freedoms: the standard answer to exchanging indivisible goods "
         "without money (strategy-proof, Pareto-efficient and individually rational for strict "
         "preferences; Shapley and Scarf, Roth). Built, and checked from the definitions, in rival.py "
         "(D31).",
         "A welfare for each participant: rejected, as in D16.",
         "Money for the rival part: rejected by the owner's premise."),
        "A witness would see, in a model with a rival freedom (a permission to use a unit of something "
        "that runs out), that granting it to one party takes it from another, that win-win then fails for "
        "some deals, and that an exchange without money (cycles of swaps) restores it for the deals that "
        "are in some party's interest.",
        "A model with rival freedoms in which the vector value of D24 still makes every coherent opening "
        "a gain for everyone; or an exchange mechanism without money that is not individually rational.",
        "Partly done. D31 models rival freedoms as indivisible goods and checks an exchange of them. There is "
        "still no witness that a real freedom is of one kind or the other, and no market has been exchanged.",
        "deals.py (non-rival permissions); rival.py (indivisible goods, D31)",
        None),
    Decision(
        "D31", "With rival freedoms an exchange without money is individually rational, efficient and truthful, "
        "and the starting allocation decides the rest", Tag.FRAME,
        "Model a rival freedom as an indivisible good that one party holds at a time (a slot, a seat, a unit "
        "that runs out). Give each party one and a strict private ranking of all of them (which can come from "
        "private weights on the coordinates of a good's vector, so no common price is needed). Exchange by top "
        "trading cycles: every party points at the owner of the good it likes best among those left, the cycles "
        "trade, repeat. No money, and no number that anyone else sees. Check the guarantees from their "
        "definitions on every profile of two and three parties and on sampled larger markets, not from the "
        "theorems.",
        ("The Boston mechanism, a common rule for assigning goods: rejected, and checked: a party can do better by "
         "misreporting, so the checker finds a lie in a fixed share of small markets.",
         "Serial dictatorship or random priority: strategy-proof and efficient, but they ignore what each party "
         "holds, so a party can end below where it began; the guarantee wanted is that nobody loses by joining. "
         "Recalled, not built.",
         "A numeraire for the difference between goods: rejected by the owner's premise.",
         "Ties, several units per party, and tastes that depend on who else gets what: outside the model, and "
         "where the guarantees fail or change."),
        "A witness would see parties each holding one rival freedom and ranking all of them in private; a published "
        "record of the cycles; every party's audit of that record, from its own ranking alone, passing; nobody "
        "ending below where it began; no party able to do better by misreporting; and, where the starting allocation "
        "was not efficient, someone strictly better off. It would also see that what each party ends with depends on "
        "what it held at the start.",
        "A market where the outcome leaves a party below where it began, or leaves a group able to trade among "
        "themselves so that none loses and one gains, or where an audit passes on a record that is not the top "
        "trading cycles outcome, or where a party gains by misreporting; or a model with rival freedoms in which "
        "the guarantee fails without leaving the stated assumptions.",
        "None: no real market has been exchanged. The model has strict rankings, one good each and a starting "
        "allocation that nobody here derives. Real parties have ties, several units and tastes that depend on others, "
        "and the starting allocation is the contested question that money used to settle (B16).",
        "rival.py; tests/test_rival.py",
        _d31),
    Decision(
        "D32", "From a given start an exchange without money moves exactly the loops of envy, a ladder made by "
        "wealth is frozen under it, and what thaws a ladder is a difference in the direction of what agents value",
        Tag.FRAME,
        "Read the present as a start (who holds what) with the agents' rankings, and read standing as envy: an "
        "agent envies another when it ranks the other's good above its own. Then top trading cycles moves someone "
        "exactly when envy has a loop, which is exactly when the start is not Pareto efficient; a start with no "
        "loop is a pecking order (each holds its favourite of what it and those after it hold), a ladder; and a "
        "start made by letting the wealthiest pick first is such a ladder, so abolishing money while keeping "
        "holdings freezes it wherever rankings agree. A dial from the rankings a common price induces to private "
        "rankings, computed exactly in rational arithmetic, shows what unfreezes it: some goods and some start trade "
        "exactly when the agents' weights are neither all equal nor all along the price (nothing at the price alone, "
        "nothing with one coordinate), at exact rational settings that can open and close again. The start is an "
        "input, whoever made it.",
        ("A uniformly random start, from which top trading cycles gives the lottery of random serial dictatorship: "
         "WITHDRAWN by the owner on 2026-10-11 (\"There is no possible uniform start\"): no neutral party exists to "
         "draw it, and the start is what money built.",
         "A transfer from the top of a ladder to its foot: not an exchange in which nobody loses, so it is outside "
         "this model; whether it is lawful turns on what optimality is over participants (D16), which the owner "
         "has not answered.",
         "Status as tastes that depend on who else holds what: outside the model, and where the guarantees of top "
         "trading cycles fail or change (HO-028). The model keeps one thing about status, that agents rank by a "
         "common scale, and that is the dial's setting 0.",
         "Making goods less rival, or rotating who picks first: the other ways to thaw a ladder without any change "
         "in tastes. Not built; they are `[HYPER]` here (HO-030)."),
        "A witness would see a real start with the agents' rankings in which envy has no loop and nobody can trade "
        "without a side payment, and the same agents, once what they value diverges in direction, trading with no "
        "money at the settings the dial predicts; or a start with a loop of envy that top trading cycles turns "
        "into a ladder in one exchange.",
        "A market where trade is possible and envy has no loop, or the reverse; an efficient start with no pecking "
        "order; a start made by letting the wealthiest pick first that the exchange moves, or in which someone "
        "envies a poorer agent; or a dial in which agents rank apart at the price alone, with one coordinate or "
        "with weights all equal or all along the price, or goods that trade though they are, or weights that are "
        "not and no goods that trade, or in which a ranking changes inside a stretch between breakpoints.",
        "None: nothing real has been exchanged. The wealthiest-first start is a stylisation (a market is not a "
        "serial dictatorship); the dial's linear mix is one path among many from a common ranking to private ones; "
        "and real status is a taste that depends on what others hold, which this model does not have. It shows "
        "what is stuck, not how to unstick a ladder by anything but a change in tastes (HO-030).",
        "standing.py; tests/test_standing.py",
        _d32),
)

_BY_ID = {d.id: d for d in LEDGER}


def decision(decision_id: str) -> Decision:
    try:
        return _BY_ID[decision_id]
    except KeyError:
        raise KeyError(f"{decision_id!r} is not in the ledger") from None


def run(item: Decision) -> Verdict | None:
    """Run a decision's check. None if it has none. An error inside a check is a failure."""
    if item.check is None:
        return None
    try:
        return item.check()
    except Exception as error:  # a broken check must read as a failed one, not a crash
        return Verdict(f"{item.id} check", Status.FAILS_IN_MODEL,
                       (f"raised {type(error).__name__}: {error}",))


def run_all() -> list[tuple[Decision, Verdict | None]]:
    return [(item, run(item)) for item in LEDGER]


# ---------------------------------------------------------------------------
# The document.
# ---------------------------------------------------------------------------


def render_markdown() -> str:
    results = run_all()
    lines = [
        "# Decisions that carry evidential expectations",
        "",
        "Generated by `python -m hyperoptimization decisions`. Do not edit by hand: "
        "`tests/test_decisions.py` fails if this file differs from what the ledger renders.",
        "",
        "A **witness inside the framework** is what the family's checkers produce: a verdict that holds "
        "in a model, or a counterexample. Each decision below says what a witness should observe if "
        "the choice is right (**Expectation**), what would show it wrong (**Refuter**), and what "
        "someone outside the model would have to observe (**World witness**, often: none yet). Every "
        "FORM and FRAME decision has a **Model check** that was run to produce this file; every "
        "check is also broken on purpose in the test suite to show it can fail. Tags: FORM checked "
        "exactly, FRAME holds in stated assumptions, HYPER a proposal nothing here witnesses, OPEN "
        "unresolved.",
        "",
        "| id | decision | tag | model check |",
        "|---|---|---|---|",
    ]
    for item, verdict in results:
        shown = "none (by design)" if verdict is None else verdict.status.value
        lines.append(f"| {item.id} | {item.title} | {item.tag.value} | {shown} |")
    for item, verdict in results:
        lines += ["", f"## {item.id}. {item.title}", "", f"**Tag:** {item.tag.value}", "",
                  f"**Choice.** {item.choice}", "", "**Not chosen.**"]
        lines += [f"- {alternative}" for alternative in item.alternatives]
        lines += ["", f"**Expectation.** {item.expectation}", "", f"**Refuter.** {item.refuter}", "",
                  f"**World witness.** {item.world_witness}", "", f"**Where.** {item.where}"]
        if verdict is not None:
            lines += ["", f"**Model check:** {verdict.status.value} — {verdict.claim}"]
            lines += [f"- {entry}" for entry in (verdict.evidence or verdict.witness)]
    return "\n".join(lines) + "\n"


__all__ = ["Decision", "LEDGER", "decision", "render_markdown", "run", "run_all"]
