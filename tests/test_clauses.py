"""Every failure clause of every check in the decision ledger can fire.

`test_decisions.py` breaks, for each decision, the code its check rests on and requires the check to
fail. That shows a check can fail somewhere. It does not show that each of its failure clauses can: a
clause that names a counterexample is a statement like any other, and one that no breakage reaches may
be written so that it never fires, which would leave the check blind to exactly that mistake.

Here each clause is reached on purpose, by one breakage of one function, and the witness must say what
the clause says. The last test then reads the clauses out of the source of `decisions.py` and requires
every `problems.append(...)` to be matched by a witness some breakage produced, so a failure clause
added later without a breakage fails this file.
"""

from __future__ import annotations

import ast
import re
from fractions import Fraction
from pathlib import Path

import pytest
from test_decisions import SABOTAGE

from hyperoptimization import decisions as D
from hyperoptimization import entrenchment, family, loop, ordinal, quorum, surprisal, viability
from hyperoptimization.ground import OMEGA, Status, Verdict

_greatest = viability.greatest_law
_kernel = viability.kernel
_real_gl = D._golden_gl()
_items = family.open_items()
_tally = family.open_tally()


def _one_state_game() -> viability.Game:
    """Everything the greatest law could forbid is empty: a single state with a single safe move."""
    return viability.Game.from_table(nature=("n",), table={"a": {"x": ("a",)}}, safe={"a"})


def _without_twins(game, corrigible=True):
    """The real greatest law, except that it forgets any state whose name starts with a capital S."""
    return {s: p for s, p in _greatest(game, corrigible=corrigible).items() if not s.startswith("S")}


# (decision, object to break, attribute, what to put there, text the witness must contain)
CLAUSES = [
    ("D01", surprisal, "sandwich_gate",
     lambda probs, controls, need: (Fraction(1), Fraction(0), Fraction(0)),
     "probability outside 2^-cost..bound"),
    ("D03", surprisal, "capture_probability",  # seats independent after all: no direction to the error
     lambda blocks, q: quorum.tail(sum(b.weight for b in blocks), q, blocks[0].probability),
     "below the mean, positive association should lower"),
    ("D04", viability, "greatest_law",
     lambda game, corrigible=True: {s: game.allowed(s) for s in game.states},
     "the move into ruin is permitted"),
    ("D05", viability, "sustaining",
     lambda game, law, start, corrigible=True: Verdict("x", Status.HOLDS_IN_MODEL, evidence=("always",)),
     "brute force"),
    ("D05", viability, "greatest_law",  # too small: every sustaining law now lies outside it
     lambda game, corrigible=True: {s: frozenset() for s in game.states},
     "a sustaining law permits what the greatest law forbids"),
    ("D05", viability, "greatest_law", _without_twins, "renaming changed the law"),
    ("D06", viability, "survival_probability", lambda *a, **k: Fraction(1, 2), "does not survive from"),
    ("D06", viability, "random_game", lambda rng, **k: _one_state_game(), "exemptions were tried"),
    ("D07", viability, "kernel",  # corrigibility that adds states instead of removing them
     lambda game, corrigible=True: frozenset(game.states) if corrigible else _kernel(game, corrigible=False),
     "corrigible kernel not inside the kernel"),
    ("D08", viability, "check_ranks",
     lambda *a, **k: Verdict("x", Status.FAILS_IN_MODEL, witness=("always",)),
     "the true ranks were rejected"),
    ("D10", loop, "step", lambda link, m, q, pi, p: Fraction(1, 3), "duality fails at"),
    ("D10", loop, "threshold",
     lambda link, m, q: 0.6 if link is loop.Link.VETO else 0.5,
     "do not sum to 1"),
    ("D11", quorum, "critical_point", lambda m, q, width=None: (Fraction(1, q), Fraction(1, q)),
     "does not rise with the quorum"),
    ("D12", loop, "levels_to_cross", lambda *a, **k: 100, "not about 10"),
    ("D13", ordinal, "rank_ordinal", lambda r: ordinal.Ordinal.of(0), "do not compare alike"),
    ("D13", ordinal, "hereditary_ordinal", lambda n, base: ordinal.Ordinal.of(0), "not ω^ω"),
    ("D13", ordinal.Ordinal, "evaluate", lambda self, base: 0, "does not evaluate back to its term"),
    ("D13", viability, "ranks", lambda game, corrigible=True: {s: 99 for s in game.states},
     "exceeds the number of states"),
    ("D14", entrenchment, "authority_height", lambda rules: {r.name: 0 for r in rules},
     "every rule is governed yet a height is finite"),
    ("D14", entrenchment, "authority_height", lambda rules: {r.name: OMEGA for r in rules},
     "an ungoverned rule has no height 0"),
    ("D15", surprisal, "gate_bits", lambda prices, controls, need: 1e6, "the outside seat saves"),
    ("D15", entrenchment, "apex", lambda rules: tuple(r.name for r in rules),
     "a grounded hierarchy should end in an ungoverned rule"),
    ("D19", D, "_golden_gl", lambda: {**_real_gl, "lob": {**_real_gl["lob"], "proved": False}},
     "Löb's axiom is not proved"),
    ("D19", D, "_golden_gl",
     lambda: {**_real_gl, "rings": [r for r in _real_gl["rings"] if any(r["guards"])]},
     "lacks guarded or unguarded rings"),
    ("D20", viability, "converge",  # a pool that permits more than the optimum
     lambda game, rng, population, **k:
         ({t: game.allowed(t) for t in _greatest(game, corrigible=k.get("corrigible", True))}, 1),
     "the pooled law lies above the greatest law"),
    ("D20", viability, "converge",  # a pool that never fills
     lambda game, rng, population, **k:
         ({t: frozenset() for t in _greatest(game, corrigible=k.get("corrigible", True))}, None),
     "300 starts did not pool to the greatest law"),
    ("D20", viability, "settle",  # a settle that ignores its start and returns the optimum
     lambda game, law, corrigible=True: _greatest(game, corrigible=corrigible),
     "reached the whole optimum"),
    ("D20", viability, "settle",  # a settle that permits everything the optimum has a state for
     lambda game, law, corrigible=True:
         {t: game.allowed(t) for t in _greatest(game, corrigible=corrigible)},
     "a settled law lies above the greatest law"),
    ("D20", viability, "settle",  # a settle that permits nothing, so no number of starts pools
     lambda game, law, corrigible=True: {t: frozenset() for t in _greatest(game, corrigible=corrigible)},
     "two starts together did not pool"),
    ("D21", viability, "maintains", lambda game, held, corrigible=True: False, "unite to"),
    ("D21", viability, "maintains",  # every proper part of the kernel supports itself, the whole does not
     lambda game, held, corrigible=True: frozenset(held) < _kernel(game, corrigible=corrigible),
     "the kernel does not support itself"),
    ("D21", viability, "inductive_kernel", lambda game, corrigible=True: frozenset(game.states),
     "the least fixed point is"),
    ("D21", viability, "inductive_kernel",  # wrong only for the one-state game
     lambda game, corrigible=True: frozenset(game.states) if len(game.states) == 1 else frozenset(),
     "a single state that holds itself"),
    ("D22", entrenchment, "cores", lambda depends_on: [("x", "y")],
     "imports and declared dependencies form a cycle"),
    ("D22", entrenchment, "phases",  # a phase that falls as reliance is added
     lambda depends_on: {n: OMEGA if sum(len(d) for d in depends_on.values()) < 10 else 0
                         for n in depends_on},
     "lowered its phase"),
    ("D22", family, "live_cores_of", lambda view="mentions", data=None: [],
     "the live scan's frequent mentions have no core"),
    ("D23", family, "topics_of", lambda text: set(), "does not find"),
    ("D23", family, "topics_of", lambda text: {"optimization"}, "only looks like one"),  # sees topics everywhere
    ("D23", family, "open_items",  # only one of the two conventions
     lambda data=None: [i for i in _items if i["form"] == "bracket"], "not both conventions"),
    ("D23", family, "open_tally",  # an item that now mentions optimization
     lambda data=None: {**_tally, "topics": {**_tally["topics"], "optimization": 3}},
     "an open item now mentions optimization"),
    ("D23", family, "open_tally",  # most items now touch a topic
     lambda data=None: {**_tally, "untouched": 10}, "mention none of the six topics"),
]

_results: dict[object, tuple[Status, tuple[str, ...]]] = {}


def broken(key: object, decision_id: str, target: object, name: str, fake: object):
    """Run one check with one thing broken; remembered, so no breakage is run twice in a session."""
    if key not in _results:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(target, name, fake)
            verdict = D.run(D.decision(decision_id))
        _results[key] = (verdict.status, verdict.witness)
    return _results[key]


@pytest.mark.parametrize("index", range(len(CLAUSES)), ids=lambda i: f"{CLAUSES[i][0]}:{CLAUSES[i][4][:44]}")
def test_each_failure_clause_fires_when_its_own_breakage_is_made_and_says_what_it_says(index):
    decision_id, target, name, fake, text = CLAUSES[index]
    status, witness = broken(("clause", index), decision_id, target, name, fake)
    assert status is Status.FAILS_IN_MODEL, f"{decision_id} passed with {name} broken"
    assert any(text in w for w in witness), (text, witness)


# --- no clause is left unreached ----------------------------------------------------------------------------------


def _pattern(message: ast.expr) -> re.Pattern[str]:
    """What a witness produced by this `problems.append(message)` looks like: its literal text, with
    each interpolated value matching anything."""
    if isinstance(message, ast.Constant):
        return re.compile(re.escape(message.value), re.S)
    assert isinstance(message, ast.JoinedStr), ast.dump(message)
    return re.compile("".join(re.escape(part.value) if isinstance(part, ast.Constant) else ".*?"
                              for part in message.values), re.S)


def failure_clauses() -> list[tuple[str, int, re.Pattern[str]]]:
    """(decision id, line, pattern) for each `problems.append(...)` in each `_dNN` check."""
    tree = ast.parse(Path(D.__file__).read_text(encoding="utf-8"))
    found = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and re.fullmatch(r"_d\d\d", node.name):
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "append" and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "problems"):
                    found.append((node.name[1:].upper(), call.lineno, _pattern(call.args[0])))
    return found


def test_the_clause_reader_finds_the_clauses_it_should():
    clauses = failure_clauses()
    assert {i for i, _line, _p in clauses} == {d.id for d in D.LEDGER if d.check is not None}
    assert len(clauses) >= 60  # 33 were unreached before this file; the rest were reached by SABOTAGE
    # and it reads a message the way a witness will print it
    [pattern] = [p for i, line, p in clauses if i == "D01" and "gate_bits" in p.pattern]
    assert pattern.search("common-principal:legislate: gate_bits 5.0, hyperethics 8.0")
    assert not pattern.search("common-principal:legislate: price 5.0")


def test_every_failure_clause_of_every_check_is_reached_by_some_breakage():
    witnesses: dict[str, list[str]] = {}
    for decision_id, (target, name, fake) in SABOTAGE.items():
        _status, witness = broken(("whole", decision_id), decision_id, target, name, fake)
        witnesses.setdefault(decision_id, []).extend(witness)
    for index, (decision_id, target, name, fake, _text) in enumerate(CLAUSES):
        _status, witness = broken(("clause", index), decision_id, target, name, fake)
        witnesses.setdefault(decision_id, []).extend(witness)
    unreached = [(i, line) for i, line, pattern in failure_clauses()
                 if not any(pattern.search(w) for w in witnesses.get(i, ()))]
    assert not unreached, f"failure clauses no breakage reaches (decision, line in decisions.py): {unreached}"


# --- a check must also hold on the cases it was written to skip -------------------------------------------------


def test_an_unbuyable_party_makes_a_gate_infinite_and_the_checks_that_read_gates_still_hold(monkeypatch):
    golden = D._golden()
    unbuyable = {"name": "unbuyable", "parties": {"A": 3, "B": "inf"}, "controls": [],
                 "gates": {"x": {"gate_cost": "inf", "need": ["B"]}}}
    monkeypatch.setattr(D, "_golden", lambda: {**golden, "systems": [*golden["systems"], unbuyable]})
    for decision_id in ("D01", "D15"):
        verdict = D.run(D.decision(decision_id))
        assert verdict.status is Status.HOLDS_IN_MODEL, (decision_id, verdict.witness)
    assert f"{len(golden['systems']) + 1} systems" in D.run(D.decision("D01")).evidence[0]
