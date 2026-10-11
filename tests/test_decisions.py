"""The decision ledger: every modelling choice, what a witness should see if it is right, and what
would show it wrong. And the ledger holds itself to the standard it states.

A decision here is a choice made while building the package that carries an expectation a
witness inside the framework could check. A witness inside the framework is what the family's
checkers already produce: a `Verdict` that holds in a model, or names a counterexample. So each
decision records

    the choice and what was not chosen,
    the EXPECTATION  (what a witness should observe if the choice is right),
    the REFUTER      (the observation that would show it wrong),
    the WORLD WITNESS (what someone outside the model would need to observe; often: none yet),
    and, for every FORM and FRAME decision, an executable CHECK.

These tests hold the ledger to four things. It is complete (no empty field). Its checks hold
today. Each check CAN FAIL: for every decision with a check there is a deliberate sabotage of the
code it rests on, and the check must then report a failure, so a check that would pass whatever
the code did is caught. And the rendered document is exactly what the ledger says, so the
numbers in DECISIONS.md are read off a run and not typed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from hyperoptimization import decisions as D
from hyperoptimization import (
    entrenchment,
    loop,
    ordinal,
    quorum,
    renewal,
    surprisal,
    viability,
)
from hyperoptimization.ground import Status, Tag, Verdict

ROOT = Path(__file__).resolve().parent.parent


# --- the ledger is complete --------------------------------------------------------------------------------


def test_ids_are_unique_and_sequential():
    ids = [d.id for d in D.LEDGER]
    assert ids == [f"D{i:02d}" for i in range(1, len(ids) + 1)]
    assert len(ids) >= 17


@pytest.mark.parametrize("decision", D.LEDGER, ids=lambda d: d.id)
def test_every_decision_says_what_it_chose_what_it_did_not_and_what_would_change_its_mind(decision):
    for field in ("title", "choice", "expectation", "refuter", "world_witness", "where"):
        value = getattr(decision, field)
        assert isinstance(value, str) and len(value.strip()) > 20, (decision.id, field)
    assert isinstance(decision.tag, Tag)
    assert decision.alternatives and all(len(a.strip()) > 20 for a in decision.alternatives)
    # a decision must be refutable: the refuter names something observable, not a feeling
    assert not decision.refuter.lower().startswith("none")


def test_every_form_and_frame_decision_has_an_executable_check_and_only_those_may_lack_one():
    for decision in D.LEDGER:
        if decision.tag in (Tag.FORM, Tag.FRAME):
            assert decision.check is not None, decision.id
        else:
            assert decision.check is None, f"{decision.id} is {decision.tag.value}; it states a claim "\
                "nothing here witnesses, so it must not carry a check that pretends to"


def test_the_hyper_and_open_decisions_say_plainly_that_no_witness_is_available_here():
    for decision in D.LEDGER:
        if decision.tag in (Tag.HYPER, Tag.OPEN) and decision.check is None:
            assert "no witness" in decision.world_witness.lower() or \
                "not done" in decision.world_witness.lower(), decision.id


def test_lookup_by_id():
    assert D.decision("D01").id == "D01"
    with pytest.raises(KeyError):
        D.decision("D99")


# --- the checks hold, and say what they measured -------------------------------------------------------------------


@pytest.mark.parametrize("decision", [d for d in D.LEDGER if d.check is not None], ids=lambda d: d.id)
def test_each_check_holds_today_and_records_what_it_read(decision):
    verdict = D.run(decision)
    assert isinstance(verdict, Verdict)
    assert verdict.status is Status.HOLDS_IN_MODEL, (decision.id, verdict.witness)
    assert verdict.evidence and all(isinstance(e, str) and e for e in verdict.evidence)


def test_a_decision_with_no_check_has_no_verdict():
    for decision in D.LEDGER:
        if decision.check is None:
            assert D.run(decision) is None


def test_run_all_pairs_every_decision_with_its_verdict():
    pairs = D.run_all()
    assert [d.id for d, _ in pairs] == [d.id for d in D.LEDGER]
    assert sum(v is not None for _, v in pairs) == sum(d.check is not None for d in D.LEDGER)


# --- every check can fail ---------------------------------------------------------------------------------------------------


def _wrong_gate_bits(prices, controls, need):
    return surprisal.cheapest_cover(prices, set(), need)[0]  # ignores command


_tail = quorum.tail
_capture = surprisal.capture_probability
_real_gl = D._golden_gl()
_closing = loop.closing_price
_protection = entrenchment.protection

SABOTAGE = {
    "D01": (surprisal, "gate_bits", _wrong_gate_bits),
    "D02": (surprisal, "additivity_gap", lambda *a, **k: 0.0),
    "D03": (surprisal, "capture_probability",  # forgets that a group carries several seats
            lambda blocks, q: _tail(len(blocks), q, blocks[0].probability)),
    "D04": (viability, "kernel", lambda game, corrigible=True: frozenset({"fair"})),
    "D05": (viability, "greatest_law",
            lambda game, corrigible=True: {s: game.allowed(s) for s in game.states}),
    "D06": (viability, "survival_probability", lambda *a, **k: Fraction(1)),
    "D07": (viability, "reach_home", lambda game: frozenset(game.states)),
    "D08": (viability, "check_ranks",
            lambda *a, **k: Verdict("x", Status.HOLDS_IN_MODEL, evidence=("always",))),
    "D09": (renewal, "log_mean_lifetime", lambda chain: 0.0),
    "D10": (loop, "threshold", lambda link, m, q: 0.5),
    "D11": (quorum, "tail", lambda m, q, x: _tail(m, q + 1, x)),
    "D12": (loop, "closing_price", lambda *a, **k: 0.5),
    "D13": (ordinal, "bump", lambda n, base: n),
    "D14": (entrenchment, "protection",
            lambda rules: {r.name: float("inf") for r in rules}),
    "D15": (renewal, "mean_lifetime", lambda *a, **k: Fraction(100)),
    "D20": (viability, "settle",  # never prunes: it permits everything, fatal moves included
            lambda game, law, corrigible=True: {s: game.allowed(s) for s in game.states}),
    "D19": (D, "_golden_gl",  # the recorded prover verdicts, each flipped
            lambda: {**_real_gl, "rings": [{**r, "proved": not r["proved"]} for r in _real_gl["rings"]]}),
}


def test_every_check_has_a_sabotage_and_the_table_names_no_decision_that_has_none():
    checked = {d.id for d in D.LEDGER if d.check is not None}
    assert set(SABOTAGE) == checked


@pytest.mark.parametrize("decision_id", sorted(SABOTAGE))
def test_each_check_reports_a_failure_when_the_code_it_rests_on_is_broken(decision_id, monkeypatch):
    module, name, fake = SABOTAGE[decision_id]
    assert D.run(D.decision(decision_id)).status is Status.HOLDS_IN_MODEL  # sound before the sabotage
    monkeypatch.setattr(module, name, fake)
    verdict = D.run(D.decision(decision_id))
    assert verdict.status is Status.FAILS_IN_MODEL, f"{decision_id} passed with {module.__name__}.{name} broken"
    assert verdict.witness  # and it says what failed


# --- the document -----------------------------------------------------------------------------------------------------------


def test_the_rendered_document_is_exactly_what_the_ledger_says():
    assert (ROOT / "DECISIONS.md").read_text(encoding="utf-8") == D.render_markdown(), (
        "DECISIONS.md has drifted from the ledger; regenerate it with "
        "`python -m hyperoptimization decisions > DECISIONS.md`")


def test_the_document_lists_every_decision_and_its_evidence():
    text = D.render_markdown()
    for decision in D.LEDGER:
        assert f"## {decision.id}. {decision.title}" in text
    assert text.count("**Model check:**") == sum(d.check is not None for d in D.LEDGER)
    assert re.search(r"^\| D01 \|", text, re.M)


def test_the_rendering_does_not_depend_on_the_hash_seed():
    src = str(ROOT / "src")
    outputs = set()
    for seed in ("0", "7"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": src}
        done = subprocess.run([sys.executable, "-m", "hyperoptimization", "decisions"], env=env,
                              capture_output=True, text=True, timeout=120, check=True)
        outputs.add(done.stdout)
    assert len(outputs) == 1 and outputs.pop() == D.render_markdown()


def test_the_whole_ledger_runs_in_a_few_seconds():
    import time

    start = time.perf_counter()
    D.run_all()
    assert time.perf_counter() - start < 20


def test_the_recorded_prover_runs_name_the_checkout_they_came_from():
    golden = D._golden_gl()
    assert len(golden["hyperlogic_commit"]) == 40 and "regenerate_golden_gl.py" in golden["command"]
    assert golden["lob"]["proved"] is True
    shapes = {(r["n"], tuple(r["guards"])) for r in golden["rings"]}
    assert len(shapes) == len(golden["rings"])  # no ring recorded twice
    assert {n for n, _ in shapes} == {1, 2, 3, 4, 5}
