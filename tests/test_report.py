"""The report and the command line: what `python -m hyperoptimization` prints.

The report is read by a person deciding whether to trust the package, so its numbers are pinned:
each headline figure below was computed by the modules and is asserted here by value, so a change
in any model that moves a headline figure fails a test instead of quietly changing the report.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hyperoptimization import decisions, report
from hyperoptimization.__main__ import _utf8_streams, main

ROOT = Path(__file__).resolve().parent.parent


def test_the_report_names_each_part_of_the_package():
    text = report.build_report()
    for heading in ("quorum", "loops", "renewal", "viability", "ordinal", "entrenchment", "family",
                    "ledger", "open"):
        assert f"== {heading}" in text.lower(), heading


def test_the_headline_figures_are_the_ones_the_models_compute():
    text = report.build_report()
    # a 5-seat body needing 3: command oversight tolerates 0.1651, veto-only 0.8349
    assert "0.1651" in text and "0.8349" in text
    # the odd majority is critical at exactly one half, and a council there costs exactly one bit
    assert "1/2" in text
    # Goodstein from 4 starts at omega^omega
    assert "ω^ω" in text and "4, 26, 41, 60, 83, 109" in text
    # the cliff game's ranks, worked by hand
    assert "x:0 c:1 b:2 a:ω" in text
    # Ross: a clause worth 100 bits amendable by a 5-bit clause is protected by 5
    assert "protected by 5 bits" in text


def test_self_maintenance_is_shown_as_the_union_of_the_sets_that_support_themselves():
    text = report.build_report()
    # the twin-rooms game, worked by hand: the empty set, each room, both rooms, and the hub with
    # either room or with both; the hub cannot hold itself up alone
    assert "the 7 sets that support themselves" in text and "inductive reading: empty" in text
    assert "kernel ['hub', 'left', 'right']" in text


def test_the_family_section_shows_the_phases_the_recorded_snapshot_gives():
    text = report.build_report()
    assert "87aadc390fea" in text and "b56f1a3c61d9" in text  # the snapshot is named
    assert re.search(r"hard\s+4\s+20\s+0\s+none", text)  # what is imported is well founded
    assert re.search(r"formal\s+15\s+16\s+4\s+hyperethics\+hyperphysics\+metamathethicology", text)  # what is meant is not
    assert "A citation is evidence of mention, not of need" in text
    # a second record, and what the family's declared open items are about
    assert "live scan" in text and "24 fields" in text and "hypergrammar+hypermath+verifier" in text
    assert "122 declared open items in 11 fields" in text
    assert "optimization 0, viability 0, persistence 2" in text and "106 mention none" in text


def test_the_ledger_summary_counts_match_the_ledger():
    text = report.build_report()
    total = len(decisions.LEDGER)
    checked = sum(d.check is not None for d in decisions.LEDGER)
    assert f"{total} decisions" in text and f"{checked} with an executable check" in text
    assert "0 failing" in text


def test_the_open_obligations_are_listed_with_a_status():
    text = report.build_report()
    for needle in ("OPEN", "NOT_IMPLEMENTED", "NOT_ESTABLISHED"):
        assert needle in text
    assert "self-funding" in text and "mean-field" in text and "persistence" in text.lower()


def test_the_report_is_deterministic_and_independent_of_the_hash_seed():
    assert report.build_report() == report.build_report()
    outputs = set()
    for seed in ("0", "5"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT / "src")}
        done = subprocess.run([sys.executable, "-m", "hyperoptimization"], env=env,
                              capture_output=True, text=True, encoding="utf-8", timeout=120, check=True)
        outputs.add(done.stdout)
    assert len(outputs) == 1


@pytest.mark.parametrize("args", [[], ["decisions"]])
def test_the_command_line_writes_utf8_whatever_the_console_code_page(args):
    # A Windows pipe defaults to cp1252, which cannot hold the omega the report and the ledger use;
    # PYTHONIOENCODING reproduces that here. The text must still arrive, and as UTF-8.
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONPATH": str(ROOT / "src")}
    done = subprocess.run([sys.executable, "-m", "hyperoptimization", *args], env=env,
                          capture_output=True, timeout=120)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")[-400:]
    text = done.stdout.decode("utf-8").replace("\r\n", "\n")
    assert "ω" in text
    assert text == (decisions.render_markdown() if args else report.build_report())


def test_both_output_streams_are_reconfigured_and_one_that_cannot_be_is_left_alone(monkeypatch):
    seen = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def reconfigure(self, **kwargs):
            seen.append((self.name, kwargs))

    monkeypatch.setattr(sys, "stdout", Stream("out"))
    monkeypatch.setattr(sys, "stderr", Stream("err"))
    _utf8_streams()
    assert seen == [("out", {"encoding": "utf-8"}), ("err", {"encoding": "utf-8"})]
    seen.clear()
    monkeypatch.setattr(sys, "stderr", object())  # no `reconfigure`: left alone, and no error
    _utf8_streams()
    assert seen == [("out", {"encoding": "utf-8"})]


def test_the_command_line(capsys):
    assert main([]) == 0
    assert "hyperoptimization" in capsys.readouterr().out
    assert main(["decisions"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Decisions that carry evidential expectations") and out.endswith("\n")
    assert main(["--help"]) == 2
    assert "usage" in capsys.readouterr().out
    assert main(["nonsense"]) == 2
    assert "usage" in capsys.readouterr().out
    assert main(["decisions", "extra"]) == 2
    capsys.readouterr()
