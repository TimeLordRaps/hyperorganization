"""The package agrees with independent reference oracles, and the comparison is able to fail.

`scripts/oracle_*.py` were written by a subagent from a written specification alone, without sight
of the package; `scripts/differential_oracles.py` runs both on the same instances. Three things are checked
here, because agreement alone proves little:

  1. each oracle reproduces the worked examples its author was given (an oracle that fails its own examples
     says nothing about the package);
  2. the package and the oracle agree on a sample large enough to reach the awkward cases (unmakeable
     decisions, price-invisible options), which the `facts` of each stage count;
  3. a slip planted in the package, or in the oracle, makes the comparison fail, so a green result is able
     to be red.

Agreement shows that implementation slips of the kinds sampled are caught. The oracles and the package were
written from the same specification, so it does not show that the specification is right.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import runpy
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from hyperoptimization import comprehension, deals, entangled

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("differential_oracles")


def quiet(_line: str) -> None:
    return None


def need_sympy() -> None:
    """The quantum oracle's exact linear algebra is sympy's. It is a declared test dependency, so its absence
    fails the test (a skip would let CI, whose install step falls back to fewer extras, go on without it)."""
    try:
        import sympy  # noqa: F401
    except ImportError:
        pytest.fail("the quantum oracle needs sympy, a declared test dependency: pip install -e '.[test]'",
                    pytrace=False)


# --- the oracles reproduce their own worked examples ------------------------------------------------------------------


@pytest.mark.parametrize("name", ["comprehension", "quantum", "price"])
def test_each_oracle_passes_the_worked_examples_it_was_written_against(name, capsys):
    if name == "quantum":
        need_sympy()
    runpy.run_path(str(SCRIPTS / f"oracle_{name}.py"), run_name="__main__")
    assert capsys.readouterr().out.strip().endswith("OK")


# --- the package agrees with them on a sample that reaches the awkward cases ------------------------------------------


def test_the_package_agrees_with_the_comprehension_oracle_on_every_sampled_instance():
    result = runner.compare_comprehension(report=quiet)
    assert result.agrees, result.mismatches[:3]
    assert result.facts["makeable"] > 1000 and result.facts["unmakeable"] > 200  # both sides of the gate are reached
    assert result.checked == 1500 + result.facts["makeable"] + 800


def test_the_package_agrees_with_the_price_oracle_on_every_sampled_option_set():
    result = runner.compare_price(small=120, large=12, report=quiet)
    assert result.agrees, result.mismatches[:3]
    assert result.facts["worked"] == 6 and result.facts["efficient"] > 300
    assert result.facts["invisible"] > 40  # options no price can choose are reached, not left to chance
    assert result.checked == 6 + 120 + 12


def test_the_package_agrees_with_the_quantum_oracle_on_every_sampled_strategy():
    need_sympy()
    result = runner.compare_quantum(ghz=12, chsh=8, report=quiet)
    assert result.agrees, result.mismatches[:3]
    assert result.checked == 12 + 8 + 1 and result.facts == {"ghz": 12, "chsh": 8, "magic": 1}


# --- a slip on either side makes the comparison fail ------------------------------------------------------------------


def test_a_slip_in_the_package_comprehension_is_caught(monkeypatch):
    real = comprehension.robust_quorum
    monkeypatch.setattr(comprehension, "robust_quorum", comprehension.cover_number)  # the quorum is the cover
    assert any(m[0] == "cover/robust" for m in runner.compare_comprehension(300, 0, quiet).mismatches)
    monkeypatch.setattr(comprehension, "robust_quorum", real)
    monkeypatch.setattr(comprehension, "necessary_voices", lambda know, nodes: [])  # nobody is ever necessary
    assert any(m[0] == "necessary" for m in runner.compare_comprehension(300, 0, quiet).mismatches)
    monkeypatch.undo()
    real_footprint = comprehension.footprint

    class Half:  # the footprint forgets what rests on the targets
        def __init__(self, found):
            self.material, self.impact = found.material, frozenset()

    monkeypatch.setattr(comprehension, "footprint", lambda graph, targets: Half(real_footprint(graph, targets)))
    assert any(m[0] == "footprint" for m in runner.compare_comprehension(0, 100, quiet).mismatches)


def test_a_slip_in_the_oracle_is_caught_just_as_well(monkeypatch):
    real = runner.reference

    def broken(name):
        module = real(name)
        original = module.cover_number
        module.cover_number = lambda know, nodes: (lambda found: found if found in (None, 0) else found + 1)(
            original(know, nodes))  # an oracle that needs one person too many
        return module

    monkeypatch.setattr(runner, "reference", broken)
    assert runner.compare_comprehension(300, 0, quiet).mismatches


def test_a_slip_in_the_package_price_test_is_caught_on_either_branch(monkeypatch):
    real = deals.supported
    monkeypatch.setattr(deals, "supported", lambda points, i: True)  # every efficient option has a price
    assert any(m[0] == "worked" for m in runner.compare_price(small=5, large=0, report=quiet).mismatches)
    monkeypatch.setattr(deals, "supported", lambda points, i: False)  # and none has
    assert runner.compare_price(small=40, large=0, report=quiet).mismatches
    monkeypatch.setattr(deals, "supported", real)
    monkeypatch.setattr(deals, "efficient", lambda points: list(range(len(points))))  # nothing is dominated
    assert runner.compare_price(small=40, large=0, report=quiet).mismatches


def test_a_slip_in_the_price_oracle_is_caught_just_as_well(monkeypatch):
    real = runner.reference

    def broken(name):
        module = real(name)
        original = module.supported
        module.supported = lambda points, i: not original(points, i) if i == 0 else original(points, i)
        return module

    monkeypatch.setattr(runner, "reference", broken)
    assert runner.compare_price(small=20, large=0, report=quiet).mismatches


def test_the_price_oracle_answers_the_same_under_every_hash_seed_and_does_not_hang(tmp_path):
    """The oracle this one replaced hung on a six-option set under 5 of 40 hash seeds (HO-025). Fourier-Motzkin
    on fractions iterates no set or dict, so the answers on the same sets must not depend on the seed."""
    script = tmp_path / "answers.py"
    script.write_text(
        "import hashlib, importlib.util, random, sys\n"
        f"spec = importlib.util.spec_from_file_location('o', {str(SCRIPTS / 'oracle_price.py')!r})\n"
        "o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)\n"
        "digest = hashlib.sha256()\n"
        "tie = [(0, 2, 1, 0), (1, 1, 1, 0), (1, 1, 1, 0), (0, 0, 1, 2), (2, 0, 1, 0), (1, 0, 2, 0)]\n"
        "digest.update(repr([o.supported(tie, i) for i in range(6)]).encode())\n"
        "for seed in range(150):\n"
        "    rng = random.Random(seed)\n"
        "    dims = rng.randint(2, 4)\n"
        "    pts = [tuple(rng.randint(0, 6) for _ in range(dims)) for _ in range(rng.randint(2, 7))]\n"
        "    digest.update(repr((o.efficient(pts), [o.supported(pts, i) for i in range(len(pts))])).encode())\n"
        "print(digest.hexdigest())\n", encoding="utf-8")
    seen = set()
    for hash_seed in ("0", "1", "6", "7", "12", "25", "29", "random"):  # the seeds that hung the old oracle among them
        done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60,
                              env={**os.environ, "PYTHONHASHSEED": hash_seed})
        assert done.returncode == 0, done.stderr
        seen.add(done.stdout.strip())
    assert len(seen) == 1 and len(next(iter(seen))) == hashlib.sha256().digest_size * 2


def test_a_slip_in_the_package_quantum_value_is_caught(monkeypatch):
    need_sympy()
    real = entangled.quantum_value
    monkeypatch.setattr(entangled, "quantum_value", lambda game, strategy: real(game, strategy) - Fraction(1, 10**9))
    result = runner.compare_quantum(ghz=2, chsh=1, report=quiet)
    assert len(result.mismatches) == result.checked  # a billionth is enough: the arithmetic is exact


def test_an_empty_comparison_is_unknown_and_never_agreement(capsys, monkeypatch):
    assert not runner.Result("anything").agrees and runner.stage_status(runner.Result("anything")) == 2
    monkeypatch.setitem(sys.modules, "sympy", None)  # `import sympy` now raises ImportError
    result = runner.run_stage("quantum", False)
    assert result.checked == 0 and runner.stage_status(result) == 2
    assert "UNKNOWN" in capsys.readouterr().out and "sympy" in result.note
    mismatched = runner.Result("anything", checked=3, mismatches=[("x",)])
    assert not mismatched.agrees and runner.stage_status(mismatched) == 1


# --- the runner's process handling ------------------------------------------------------------------------------------


def test_the_runner_exits_0_on_agreement_and_1_on_a_planted_slip(monkeypatch, capsys):
    assert runner.main(["--stage", "comprehension", "--unsupervised"]) == 0
    monkeypatch.setattr(comprehension, "robust_quorum", comprehension.cover_number)
    assert runner.main(["--stage", "comprehension", "--unsupervised"]) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_a_stage_that_is_supervised_agrees_in_its_own_process():
    assert runner.main(["--stage", "comprehension"]) == 0


def test_a_stage_past_its_deadline_is_killed_and_reported_unknown(monkeypatch, capsys):
    monkeypatch.setattr(runner, "DEADLINES", {"comprehension": (0.001, 0.001)})  # python cannot even start in this
    assert runner.supervised("comprehension", False) == 2
    assert "UNKNOWN" in capsys.readouterr().out


def test_the_worked_price_cases_are_the_ones_the_oracle_was_given():
    assert len(runner.WORKED) == 6
    for points, supported, efficient, hidden in runner.WORKED:
        assert [deals.supported(points, i) for i in range(len(points))] == supported
        assert deals.efficient(points) == efficient and deals.price_invisible(points) == hidden
