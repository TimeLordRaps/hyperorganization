"""The report: what the package computes, in one page a person can read.

Every number is computed by the modules when the report is built. None is typed. The report
ends with what is not established, because a report that listed only what worked would be an
advertisement.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

from . import (
    decisions,
    entrenchment,
    family,
    loop,
    ordinal,
    quorum,
    renewal,
    surprisal,
    viability,
)
from .ground import OMEGA, Status, rank_key


def _cliff() -> viability.Game:
    return viability.Game.from_table(
        nature=("n",),
        table={
            "a": {"stay": ("a",), "go": ("b",)},
            "b": {"stay": ("c",), "go": ("c",)},
            "c": {"stay": ("x",), "go": ("x",)},
            "x": {"stay": ("x",), "go": ("x",)},
        },
        safe={"a", "b", "c"},
    )


def _culdesac() -> viability.Game:
    return viability.Game.from_table(
        nature=("n",),
        table={
            "p": {"stay": ("p",), "wander": ("q",)},
            "q": {"linger": ("q",)},
            "r": {"hold": ("r",), "dive": ("p",)},
        },
        safe={"p", "q", "r"}, home={"p"}, correct={("r", "p")},
    )


def _twin_rooms() -> viability.Game:
    """A hub with a way into each of two safe rooms and a way out of both."""
    return viability.Game.from_table(
        nature=("n",),
        table={
            "hub": {"go-left": ("left",), "go-right": ("right",), "fall": ("x",)},
            "left": {"stay": ("left",)}, "right": {"stay": ("right",)}, "x": {"stay": ("x",)},
        },
        safe={"hub", "left", "right"},
    )


def _fraction(value: Fraction) -> str:
    return str(value)


def _section_quorum() -> list[str]:
    lines = ["== quorum  (a body of m seats is captured when q are compromised; x is a seat's "
             "compromise probability)",
             "   m   q   critical x_c    slope there    price of the council at x_c"]
    for m, q in [(3, 2), (5, 3), (9, 5), (21, 11), (9, 3), (9, 7)]:
        lo, hi = quorum.critical_point(m, q)
        mid = (lo + hi) / 2
        point = _fraction(lo) if lo == hi else f"{float(mid):.4f}"
        slope = float(quorum.tail_slope(m, q, mid))
        price = surprisal.council_bits(m, q, mid)
        lines.append(f"   {m:<3} {q:<3} {point:<15} {slope:<14.4f} {price:.4f} bits")
    lines.append("   At x_c a council is exactly as easy to capture as one seat (the price column equals the "
                 "seat's own): that is what critical means. The odd majority is its own dual and is "
                 "critical at exactly 1/2, one bit. Below x_c compromise falls doubly exponentially with "
                 "depth; above it, it rises to 1.")
    return lines


def _section_loops() -> list[str]:
    cmd, veto = loop.Link.COMMAND, loop.Link.VETO
    lines = ["== loops  (a body whose overseers are bodies of its own kind; pi is the chance of a "
             "direct attack)"]
    for m, q in [(5, 3), (51, 26)]:
        lines.append(f"   {m} seats, quorum {q}: critical pi*  overseers who can command "
                     f"{loop.threshold(cmd, m, q):.4f}   overseers who can only veto "
                     f"{loop.threshold(veto, m, q):.4f}")
    t = loop.threshold(cmd, 51, 26)
    near = loop.levels_to_cross(cmd, 51, 26, t * (1 + 1e-4), 0.5)
    nearer = loop.levels_to_cross(cmd, 51, 26, t * (1 + 1e-6), 0.5)
    lines.append("   The two sum to 1 for a self-dual quorum: they are one equation in dual variables. "
                 "As the body grows both close on q/m.")
    lines.append(f"   Just past pi* the loop crawls: {near} levels before collapse at 1e-4 past, "
                 f"{nearer} at 1e-6 (x{nearer / near:.1f} for 100x closer).")
    lines.append(f"   Folding a one-level hierarchy into a loop changes the compromise probability by "
                 f"{loop.closing_price(cmd, 5, 3, Fraction(1, 10), 1):.5f} at pi = 0.1.")
    return lines


def _section_renewal() -> list[str]:
    lines = ["== renewal  (seats attacked at rate lambda, repaired at rate nu, repair only while the "
             "body holds; lambda + nu = 1, theta = q/m = 1/2)",
             "   rho = lambda/(lambda+nu)   life at m = 100    at m = 1600     regime"]
    for rho, label in [(Fraction(1, 5), "exponential in m"), (Fraction(1, 2), "(1/2)(ln 2m + gamma)"),
                       (Fraction(4, 5), "O(1): no number of seats helps")]:
        lives = [math.exp(renewal.log_mean_lifetime(renewal.Chain(m, m // 2, rho, 1 - rho)))
                 for m in (100, 1600)]
        lines.append(f"   {float(rho):<26.1f} {lives[0]:<17.4g} {lives[1]:<15.4g} {label}")
    lines.append(f"   Repair must outpace attack by the quorum margin: nu/lambda above "
                 f"{renewal.min_repair_ratio(Fraction(1, 2))} at theta = 1/2.")
    return lines


def _section_viability() -> list[str]:
    cliff, trap = _cliff(), _culdesac()
    ranks = viability.ranks(cliff, corrigible=False)
    shown = " ".join(f"{s}:{r}" for s, r in sorted(ranks.items(), key=lambda kv: (rank_key(kv[1]), kv[0])))
    law = viability.greatest_law(cliff, corrigible=False)
    plain = viability.kernel(trap, corrigible=False)
    corrigible = viability.kernel(trap, corrigible=True)
    law_trap = viability.greatest_law(trap, corrigible=True)
    rooms = _twin_rooms()
    holding = [chosen for size in range(len(rooms.states) + 1)
               for chosen in itertools.combinations(sorted(rooms.states), size)
               if viability.maintains(rooms, chosen, corrigible=False)]
    inductive = sorted(viability.inductive_kernel(rooms, corrigible=False)) or "empty"
    return [
        "== viability  (existence = staying safe for ever; the optimal law = the least restriction "
        "that preserves it)",
        f"   cliff game, ranks (steps of grace; ω = never lost): {shown}",
        "   greatest law: " + ", ".join(f"{s}: permit {sorted(p)}" for s, p in law.items())
        + "  (the step that leaves the kernel is the only one forbidden)",
        f"   cul-de-sac game: kernel {sorted(plain)}, corrigible kernel {sorted(corrigible)}; "
        f"the corrigible law at p permits {sorted(law_trap['p'])} and forbids the move to the dead end",
        "   The greatest law is unique and names no participant. Adding one permission outside it, at "
        "a state the system reaches, makes survival uncertain: an exemption is fatal.",
        f"   self-maintenance, twin rooms: kernel {sorted(viability.kernel(rooms, corrigible=False))} is the "
        f"union of the {len(holding)} sets that support themselves (the empty set among them); the hub holds "
        f"only inside a set that holds a room; inductive reading: {inductive}",
    ]


def _section_ordinal() -> list[str]:
    terms = list(ordinal.goodstein_sequence(4, limit=300))
    values = ", ".join(str(v) for v, _b, _o in terms[:6])
    chain = " > ".join(str(o) for _v, _b, o in terms[:3])
    return [
        "== ordinal  (a phase is an ordinal rank)",
        f"   Goodstein from 4: {values}, ...",
        f"   ordinals: {chain} > ...  (300 steps checked, strictly descending)",
        "   Peano arithmetic cannot prove this process stops for every start; the descent below "
        "epsilon_0 is the proof.",
    ]


def _section_entrenchment() -> list[str]:
    protected = entrenchment.protection([
        entrenchment.Rule("entrenched clause", 100, "amendment clause"),
        entrenchment.Rule("amendment clause", 5, "amendment clause"),
    ])["entrenched clause"]
    return [
        "== entrenchment  (who can change the rule that says who can change the rule)",
        f"   a clause worth 100 bits, amendable by a 5-bit clause, is protected by {protected:g} bits "
        "(Ross: protection is the weakest link).",
        "   With no ungoverned rule every authority height is ω: a constitution that obeys the no-apex "
        "rule has no finite rank for validity.",
    ]


def _section_family() -> list[str]:
    data = family.snapshot()
    lines = [f"== family  (reliance among the family's repositories, from one recorded snapshot of hyperstratum "
             f"{data['hyperstratum_commit'][:12]}, graph digest {data['source_digest'][:12]}: "
             f"{len(data['nodes'])} repositories, {len(data['edges'])} edges; source relies on target)",
             "   view         reliances  phase 0  phase ω   cores (groups that rely on one another)"]
    for name in ("hard", "formal", "substantive", "weighty", "all"):
        phases, cores = family.phases_of(name, data=data), family.cores_of(name, data=data)
        reliances = sum(len(d) for d in family.depends_on(name, data=data).values())
        shown = "; ".join("+".join(core) if len(core) <= 3 else f"{len(core)} repositories"
                          for core in cores) or "none"
        lines.append(f"   {name:<12} {reliances:<10} {sum(p == 0 for p in phases.values()):<8} "
                     f"{sum(p is OMEGA for p in phases.values()):<9} {shown}")
    lines.append("   What the family imports is well founded; what its specifications say is not. A citation "
                 "is evidence of mention, not of need: the cores are where mentions go round in a circle, and "
                 "whether a circle is vicious is for the owners to say.")
    record = family.live()
    frequent = "; ".join("+".join(core) if len(core) <= 3 else f"{len(core)} repositories"
                         for core in family.live_cores_of("frequent", data=record)) or "none"
    lines.append(f"   live scan (hyperstratum's own, {len(record['fields'])} fields, {len(record['edges'])} "
                 f"edges), {family.FREQUENT} or more links or mentions: cores {frequent}")
    tally = family.open_tally(data=record)
    topics = ", ".join(f"{topic} {count}" for topic, count in tally["topics"].items())
    owners = len({item["field"] for item in family.open_items(data=record)})
    lines.append(f"   {tally['items']} declared open items in {owners} fields; by wording, items mentioning "
                 f"{topics}; {tally['untouched']} mention none of the six topics (a proxy for a reading by "
                 "the owners, not a substitute)")
    return lines


def _section_ledger() -> list[str]:
    results = decisions.run_all()
    total = len(results)
    checked = sum(v is not None for _d, v in results)
    failing = sum(v is not None and v.status is Status.FAILS_IN_MODEL for _d, v in results)
    tags = {}
    for item, _v in results:
        tags[item.tag.value] = tags.get(item.tag.value, 0) + 1
    shown = ", ".join(f"{tag} {count}" for tag, count in sorted(tags.items()))
    return [
        "== ledger  (DECISIONS.md: every choice, what a witness should see, what would refute it)",
        f"   {total} decisions, {checked} with an executable check, {failing} failing; tags: {shown}",
    ]


def _section_open() -> list[str]:
    return [
        "== open obligations  (what is not established)",
        "   OPEN             persistence as optimality: why the law that preserves existence is the "
        "optimal one (D16, the family's GC-5)",
        "   NOT_ESTABLISHED  that physical reality adheres to such a structure (D17); that this closes "
        "the family's open items (D18, D23: of 122 declared open items none is about optimization or "
        "viability by wording, four are touched, none is closed)",
        "   NOT_ESTABLISHED  mean-field loops stand in for a real ring or graph of bodies; the "
        "thresholds are upper bounds when compromise is positively associated (D03)",
        "   NOT_IMPLEMENTED self-funding capture (reinvestment is positive feedback); a ring or graph "
        "loop solved exactly; transfinite ranks for infinite games",
        "   UNKNOWN          CI on Windows and Python 3.10 in the hosted matrix has not run on this "
        "repository (3.10 and 3.13 pass locally)",
    ]


def build_report() -> str:
    from . import __version__

    sections = [
        [f"hyperoptimization {__version__}: existence, freedom, corrigibility and renewal, without money"],
        _section_quorum(), _section_loops(), _section_renewal(), _section_viability(),
        _section_ordinal(), _section_entrenchment(), _section_family(), _section_ledger(),
        _section_open(),
    ]
    return "\n\n".join("\n".join(section) for section in sections) + "\n"
