# Validation

The receipt for `hyperoptimization` 0.1.0.dev0, 2026-10-11. Everything below was observed in one session by one author. Nothing here has been run by anyone else, and no hosted workflow has run on this repository.

## Coordinates

* Repository `hyperoptimization`, local only (it has no remote; see [PROVENANCE.md](PROVENANCE.md) and `TECHNICAL_DEBT.md` HO-009), branch `main`, parent commit `8845d59`. The commit carrying this receipt adds the documents, the specification, the `family` module and the checks listed below. The manifest binds the bytes.
* The implementation, tests, scripts, data, specification and the three generated or load-bearing documents are byte-bound in [`validation/source-manifest.json`](validation/source-manifest.json): 50 files, digest of its canonical `files` mapping `639773b9c83cdac1b801d61f68696fe1abd5cb97c6a3133f6abf126dbfd69b50`. The manifest identifies tested bytes. It does not sign or certify them.
* CPython 3.11.15 for the main run, and the same suite under CPython 3.10.20 and 3.13.14, on Linux 6.18.44. pytest 9.1.1, pytest-timeout 2.4.0, ruff 0.16.10, coverage 7.16.2, uv 0.8.17 (used to build). `pyproject.toml` supplies `-vv -s --durations=10 --timeout=60`.
* **No runtime dependencies.** The package imports only the standard library, and none of the packages whose records it holds.
* Records of other packages, each a snapshot with the commit it was read at: hyperethics `a13770716f57` (gate costs), hyperlogic `09231650d419` (Gödel–Löb prover verdicts), hyperstratum `87aadc390fea` (the static family graph, digest `b56f1a3c61d9…`, and the live scan). Checked against, in this session: hyperethics `a515d89` (cross-verification) and hypermath `a340179` (spec lint).

## Observed checks

```console
python -m pytest tests
python -m ruff check src tests scripts
python -m coverage run --branch --source=src/hyperoptimization -m pytest tests
python -m hyperoptimization
python -m hyperoptimization decisions
PYTHONPATH=<hyperethics>/src:<hyperethics>/tests python scripts/verify_against_hyperethics.py
PYTHONPATH=<hypermath>/src python -m hypermath_foundations translate L0_viability.hm --lint
python scripts/check_sdist.py
```

* **614 passed, zero skips**, under CPython 3.11 with `PYTHONHASHSEED` 0, 1, 3 and 4242 (about 41 s each), under 3.10 (about 50 s) and under 3.13 (about 40 s). By file: clauses 44, decisions 75, documents 3, entrenchment 13, family 23, ground 11, loop 70, maintenance 10, manifest 4, ordinal 28, phases 19, poly 18, quorum 57, renewal 43, report 11, spec 53, surprisal 77, viability 55. The longest single test is about 6 s. Ruff passes with no findings.
* **Coverage, line and branch: 99%** (2233 statements, 2 missed; 978 branches, 1 partial). Every module is at 100% except `__main__.py`, whose two missed lines are the body of `if __name__ == "__main__":`. They run only in a subprocess, which this measurement does not follow; the subprocess tests exercise that path. An earlier version of the register claimed 100% at an audit; the figure measured at checkpoint 44 was 97% and the earlier one could not be reproduced (HO-016).
* **Mutation testing: 131 hand-written mutants, 130 killed, 1 survivor**, run on the tree whose manifest digest is given above (the harness copies the tree once per mutant, changes one piece of text, and runs the suite with the three document-drift tests deselected, since any change that moves a printed number would fail those). Thirteen modules and the documents are covered (quorum 10, polynomials 5, renewal 9, loop 11, prices 10, viability 19, entrenchment 12, ordinal 9, the ledger 14, the contract module 9, the command line 4, the family module 12, the report 3, the spec and documents 4). The survivor, `minimal_covers` with `c < candidate` for `c <= candidate`, is equivalent: `_check_net` removes duplicate names and `combinations` visits each subset once, and the two versions agree on all 66,067 networks of up to four parties (every edge set, every need set, duplicated name lists too). Of the 130 kills, 127 are an assertion failing within one suite run (under 45 s) and 3 are a run that does not terminate and is cut off at 120 s (`v-reach-home`, `o-bump`, `n-phases-subset`). The run found two real test gaps, both closed: a double end root in the exact fixed-point counter (HO-013) and the `.py` branch of the formal view. The mutants are a hand-picked sample of small changes. Killing them shows the tests notice those changes; it does not show the code correct.
* **Every failure clause of every check can fire.** `tests/test_clauses.py` reads the failure clauses out of the source of `decisions.py` (every `problems.append(...)`), breaks one function for each, and requires the witness to say what the clause says; a clause no breakage reaches fails the suite (negative control: an unreachable clause is reported by decision and line).
* **Cross-check against hyperethics' own code** at `a515d89`: gate cost is at most gate size times the dearest price on 1,178 of its random gates; the smallest system that obeys its rule and costs at least a stake has an amendment gate of `max(k+1, m*)` on 120 instances, by exhaustive search that builds each candidate as a hyperethics `Governance`.
* **Specification lint** with the family's translator (hypermath `a340179`): exit 0, 0 statements outside the translatable fragment. The five axioms are prose in comments; the translator does not translate derives or non-derives, so nothing mechanical ties the spec to the code except `tests/test_spec.py` (53 tests: every declared name has a row, every row names code, a test and a decision that exist, and the owner's three messages are in the spec and in PROVENANCE exactly as sent, including the doubled space in the third).
* **Packaging.** An sdist and a wheel were built with `uv build` from a clean copy of the tree. `scripts/check_sdist.py` finds all 50 listed files in the sdist, byte for byte (63 files in it). The wheel carries the four data files and `py.typed`. Installed into a clean virtual environment, `python -I -m hyperoptimization decisions` is byte-identical to `DECISIONS.md` and the report is byte-identical to the source tree's, with 23 decisions and 20 checks holding. The unpacked sdist's own suite passes (614) under 3.10, 3.11 and 3.13.
* **Defects found and repaired while closing this out**, each reproduced first (register rows HO-010 to HO-018): non-deterministic random games under different hash seeds; a ledger check that passed with its own code broken; a wrong Gödel–Löb statement in a golden-file script; a double end root the counter never met; `python -m hyperoptimization` raising `UnicodeEncodeError` on `ω` whenever stdout was a cp1252 pipe, which is the default on a Windows runner (reproduced with `PYTHONIOENCODING=cp1252`); a source distribution missing a test helper and every script; 33 failure clauses in the ledger that no test reached; generated text in `DECISIONS.md` (Löb's axiom, with the box modality typed as two square brackets followed by a parenthesis) that the shared workflow's link check reads as a link to a missing file, found by running that job's pattern by hand and now run by `tests/test_documents.py`; and two claims of mine, withdrawn in the register: a coverage figure of 100% and the statement that only six of the family's open items were machine-readable (hyperstratum's scan finds 116 more).

## What the results are evidence of

* `[FORM]` results (the greatest sustaining law and its uniqueness, rank certificates, exemption is fatal, quorum duality and the single critical point, the dual-variable identity of the loops, closed forms for the renewal chain, prices as surprisal brackets, protection as the weakest link, the phase of a dependency graph, self-maintenance as a greatest fixed point, Goodstein descent) were checked exactly in rationals or exhaustively on every finite structure of a stated size, against definitions that share no code with the implementation.
* `[FRAME]` results hold inside stated assumptions (independent seats, mean-field loops, a birth–death chain with repair only while the body holds, surprisal as price, full support as free will, the Löb ring as the logic of proof-checked approval, a recorded snapshot of the family). Each assumption is a decision in `DECISIONS.md` with what would refute it.
* `[HYPER]` and `[OPEN]` items carry no check on purpose: that physical reality adheres to such a structure (D17), why persistence is the right thing to optimize (D16), and that hyperoptimization closes the family's open items (D18). The first passes at D18 do not support it.

## Not observed

* **No hosted workflow has run on this repository.** The conformance workflow was copied unchanged. The one Windows failure that could be reproduced here (a cp1252 pipe) is fixed and tested; Windows, macOS and the hosted 3.10 and 3.13 legs are unrun (HO-005).
* No independent reviewer and no external reproduction. Hash-seed independence was checked by subprocess runs, not by a second machine.
* The goldens are snapshots (HO-004). If hyperethics, hyperlogic or hyperstratum change what they record, the tests keep passing against the old numbers until the goldens are regenerated deliberately.
* The loop thresholds are mean-field and the repair model is a chain; no real institution was measured (D03, D09, HO-001, HO-002, HO-003).
* The family findings (D22, D23) are about one snapshot of two records. A citation is evidence of mention, not of need; the keyword table misses synonyms and counts a word in any sense; the reading of which four open items this package touches was by hand and cannot be rerun (HO-017, HO-018).
* The repository is not on the hosting platform. Creating it is the owner's step (HO-009).

## What would invalidate it

Any change to a listed file (`tests/test_manifest.py` fails until the manifest is regenerated, which is the moment to rerun the checks above); a regenerated golden; a change to hyperethics, hyperlogic, hyperstratum or hypermath that the records or cross-checks read; a new CPython minor version.

## Reproduce

```console
pip install -e ".[dev]"
python -m pytest tests                      # 614 tests, verbose, per-test timeout, slowest ten listed
python -m ruff check src tests scripts
python -m hyperoptimization                 # the report: computed numbers, then what is not established
python -m hyperoptimization decisions | diff - DECISIONS.md
python scripts/regenerate_manifest.py --check
python scripts/check_sdist.py               # needs `build` or `uv`
```
