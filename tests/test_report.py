"""The report and the command line: what `python -m hyperoptimization` prints.

The report is read by a person deciding whether to trust the package, so its numbers are pinned:
each headline figure below was computed by the modules and is asserted here by value, so a change
in any model that moves a headline figure fails a test instead of quietly changing the report.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from hyperoptimization import report
from hyperoptimization.__main__ import main

ROOT = Path(__file__).resolve().parent.parent


def test_the_report_names_each_part_of_the_package():
    text = report.build_report()
    for heading in ("quorum", "loops", "renewal", "viability", "ordinal", "entrenchment", "ledger",
                    "open"):
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


def test_the_ledger_summary_counts_match_the_ledger():
    from hyperoptimization import decisions

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
                              capture_output=True, text=True, timeout=120, check=True)
        outputs.add(done.stdout)
    assert len(outputs) == 1


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
