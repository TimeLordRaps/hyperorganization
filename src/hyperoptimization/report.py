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
    comprehension,
    deals,
    decisions,
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


def _trap_outscores(rooms: int, *, price: int) -> bool:
    """Whether a lawless trap of this many rooms outscores a lawful opening when each coherence dimension
    is priced at `price` and everything else at 1."""
    game = deals.trap_game(rooms)
    trap = deals.opening_value(game, {"hub": {"stay"}}, deals.trap_opening(rooms), ["hub"])
    safe = deals.opening_value(game, {"hub": {"stay"}}, {"hub": {"stay", "pause"}}, ["hub"])
    dims = deals.dimensions(trap, safe)
    weights = (1,) * len(dims) + (1, 1) + (price,) * 4
    return deals.priced(deals.vector(trap, dims), weights) > deals.priced(deals.vector(safe, dims), weights)


def _section_exchange() -> list[str]:
    hamlets = viability.Game.from_table(nature=("calm", "storm"),
                                        table={"a": {"hold": ("a", "b")}, "b": {"hold": ("b", "a")}})
    held = {"A": {"a"}, "B": {"b"}}
    result = knowledge.surplus(hamlets, held, {"a": "A", "b": "B"})
    loads = comprehension.loads(family.depends_on("formal"))
    sizes = sorted(len(fp.nodes) for fp in loads.values())
    heavy = sorted(n for n, fp in loads.items() if len(fp.nodes) == sizes[-1])
    know = {f"p{i}": ["a", "b", "c"] for i in range(3)} | {f"q{i}": ["a"] for i in range(4)}
    open_door = comprehension.requirement(know, ["a", "b", "c"], 0)
    raised = comprehension.requirement(know, ["a", "b", "c"], 1)
    x = Fraction(1, 4)
    game, base, menu, grouping = deals.three_doors()
    values = [deals.opening_value(game, base, after, ["hub"], dimension=grouping) for after in menu]
    dims = deals.dimensions(*values)
    vectors = [deals.vector(v, dims) for v in values]
    trap_rooms = next(rooms for rooms in range(1, 100) if _trap_outscores(rooms, price=40))
    ghz, magic, chsh = entangled.ghz_game(), entangled.magic_square_game(), entangled.chsh_game()
    rational = entangled.chsh_strategy((1, 0), (0, 1), (Fraction(119, 169), Fraction(120, 169)),
                                       (Fraction(119, 169), Fraction(-120, 169)))
    counts, ladders, dial = rival.survey(3), standing.survey(3), standing.dial_survey()
    return [
        "== exchange  (no money: what is exchanged, who may decide, how a deal is valued, what a shared "
        "entangled state buys)",
        f"   two hamlets that shelter in each other in a storm: alone they certify "
        f"{len(result.alone['A'])} and {len(result.alone['B'])} permissions; together {len(result.together)}; "
        "each needs the other's knowledge, and what is exchanged is a set of situations that supports itself, "
        "which anyone can check",
        f"   the footprint of a decision (what it rests on and what rests on it), in the family's formal view: "
        f"{sizes.count(1)} of {len(sizes)} repositories have a footprint of 1; {len(heavy)} "
        f"({', '.join(heavy)}) have {sizes[-1]}",
        f"   seven people, three of whom understand all three parts: required quorum {open_door.required} of "
        f"{len(open_door.pool)} (any four attendees could all be among the four who understand one part), "
        f"capture costs {comprehension.capture_bits(open_door, x):.1f} bits at x = 1/4; raise the barrier to 1: "
        f"{raised.required} of {len(raised.pool)}, {comprehension.capture_bits(raised, x):.1f} bits "
        "(Condorcet: a smaller body is cheaper to capture)",
        f"   three doors, all coherent: (x {vectors[0][0]}), (y {vectors[1][1]}), (x {vectors[2][0]}, y "
        f"{vectors[2][1]}); none dominates another, and the balanced one (option "
        f"{deals.price_invisible(vectors)[0]}) is chosen by no strictly positive price",
        "   within lawful existence nothing competes: the greatest corrigible law is best in every coordinate; "
        "and coherence is a constraint, since a price of 40 on each coherence dimension is outbid by a lawless "
        f"trap of {trap_rooms} rooms",
        "   entangled decisions, exact (win probability: best classical strategy to entangled strategy): "
        f"GHZ (Greenberger-Horne-Zeilinger, three parties) {entangled.classical_value(ghz)} to "
        f"{entangled.quantum_value(ghz, entangled.ghz_strategy(1, 1))}; Mermin-Peres magic square (two parties) "
        f"{entangled.classical_value(magic)} to {entangled.quantum_value(magic, entangled.magic_square_strategy())}; "
        f"CHSH (Clauser-Horne-Shimony-Holt, two parties) {entangled.classical_value(chsh)} to "
        f"{entangled.quantum_value(chsh, rational)} (the limit (2 + sqrt 2)/4 is irrational); a GHZ-type state is "
        "worth 1/2 + (sin 2t)/2; no output depends on another's input; one use consumes the state",
        f"   rival freedoms (indivisible goods, one each, strict private rankings), top trading cycles: of the "
        f"{counts['profiles']} rankings of three agents trade happens in {counts['trade']}, exactly where the start is "
        f"not Pareto efficient; nobody ends below where they began, no agent gains by misreporting "
        f"({counts['gameable_ttc']} do) and the Boston mechanism, a common rule, can be gamed in "
        f"{counts['gameable_boston']}; with identical rankings nothing moves, so the starting allocation decides",
        f"   standing (a start, whoever made it): of the same {ladders['profiles']} rankings envy has a loop in "
        f"{ladders['loops']}, exactly where trade happens, and the other {ladders['ladders']} are ladders (each holds "
        f"its favourite of what those before it left); a start made by letting the wealthiest pick first is a ladder "
        f"the exchange leaves where it is ({ladders['wealth_frozen']} of {ladders['wealth_starts']} such starts), so "
        "abolishing money while keeping holdings freezes it; on a dial from the rankings a common price induces to "
        "private ones some goods and some start trade exactly when the agents' weights are neither all equal nor all "
        f"along the price (nothing at the price alone or with one coordinate), and of {dial['instances']} seeded "
        f"instances {dial['trade']} trade at some setting and in {dial['reverse']} of those trade opens and closes again",
    ]


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
        "loop solved exactly; transfinite ranks for infinite games; rival freedoms beyond one indivisible good each "
        "and strict rankings (ties, several units, tastes that depend on others, a starting allocation that is "
        "derived and not given; D31)",
        "   OPEN             an instrument for understanding (D29); NOT_ESTABLISHED that institutional decisions "
        "can be entangled (D28)",
        "   UNKNOWN          macOS and Python 3.11 are not in the hosted matrix (the hosted run of 2026-10-11 "
        "passed on Linux and Windows under 3.10 and 3.13; 3.11 passes locally)",
        "   OPEN             how to unfreeze a ladder by anything but a change in what agents value, and what "
        "status is when tastes depend on what others hold (D32, HO-030)",
    ]


def build_report() -> str:
    from . import __version__

    sections = [
        [f"hyperoptimization {__version__}: existence, freedom, corrigibility and renewal, without money"],
        _section_quorum(), _section_loops(), _section_renewal(), _section_viability(),
        _section_ordinal(), _section_entrenchment(), _section_family(), _section_exchange(),
        _section_ledger(), _section_open(),
    ]
    return "\n\n".join("\n".join(section) for section in sections) + "\n"
