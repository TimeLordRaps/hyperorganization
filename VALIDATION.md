# Validation

The receipt for `hyperoptimization` 0.1.0.dev0, for the bytes of the commit that closes out decision D32 (`standing`). Its mutation rounds and long run were made on 2026-10-11; its closeout checks, packaging and hosted run on 2026-10-15. Everything below was observed by one author in one working session series, on one container. Nothing here has been run by anyone else. Part of the checking was done by reference implementations that subagents wrote from a specification; they are untrusted model output and are described as such where they appear. The hosted workflow's result for this commit is recorded under "Hosted run" below, and the history of earlier hosted runs under "History".

## Coordinates

* Repository `hyperoptimization`, branch `main`. The module commit `f27dfc5` (D32, `standing`, its tests and its first checks) sits on `0db8b99`, the owner's correction. The commit that carries this receipt closes D32: the mutation rounds and their fixes, the refusal boundary, the reference runs at their final sizes, the audit row and the packaging checks. Its hash is the head of `main` when this file is read; the manifest binds the bytes the tests ran on.
* The implementation, tests, scripts, data, specifications and the load-bearing documents are byte-bound in [`validation/source-manifest.json`](validation/source-manifest.json): 75 files, digest `98e4d8904ecdbe2ac37301c80d6987a4915d58ee76519ec6bad869c2149fc25c`. `scripts/regenerate_manifest.py --check` reports it current on these bytes. The manifest identifies tested bytes. It does not sign or certify them.
* Interpreters: CPython 3.11.15 for the main runs, under hash seeds 0, 1, 3 and 4242; CPython 3.10.20 and 3.13.14 under hash seed 0. Linux 6.18.44. pytest 9.1.1, pytest-timeout 2.4.0, sympy 1.14.0 (needed only by the quantum reference and its tests), ruff 0.16.10, coverage 7.16.2, uv 0.8.17 (to build). `pyproject.toml` supplies `-vv -s --durations=10 --timeout=60`; the runs below used `--timeout=180` (the cross-version runs), `240` (coverage and the unpacked source distribution).
* **No runtime dependencies.** The package imports only the standard library, and none of the packages whose records it holds.
* Records of other packages, each a snapshot with the commit it was read at: hyperethics `a13770716f57` (gate costs, the committed golden), hyperlogic `09231650d419` (the Gödel–Löb loop verdicts, the committed golden), hyperstratum `87aadc390fea` (the family graph and the live scan, digest `b56f1a3c61d9`, the committed goldens). Checked against at closeout, in the sibling checkouts: hyperethics `a515d89`, hyperlogic `0923165`, hyperstratum `0f054e7`, hypermath `a340179`. Regenerated at closeout, the goldens differ from the committed ones only in the recorded commit fields (see the audit 6 row of `TECHNICAL_DEBT.md`); the committed goldens are unchanged.

## Observed checks at closeout

```console
PYTHONHASHSEED=0|1|3|4242 PYTHONPATH=src python -m pytest tests --timeout=180   # CPython 3.11, and 3.10 and 3.13 with seed 0
python -m coverage run --branch --source=src/hyperoptimization -m pytest tests --timeout=240
python -m coverage report -m
python scripts/differential_oracles.py --full
PYTHONPATH=<hyperethics>/src:<hyperethics>/tests python scripts/verify_against_hyperethics.py
PYTHONPATH=<hypermath>/src python -m hypermath_foundations translate L0_viability.hm --lint
PYTHONPATH=<hypermath>/src python -m hypermath_foundations translate L1_exchange.hm --lint
python scripts/regenerate_manifest.py --check
python scripts/check_sdist.py
python -m ruff check .
```

### Tests

* **1,038 passed and zero skipped in every run**: CPython 3.11 under the four hash seeds (287 to 292 s each, four at a time on four cores), CPython 3.10 (337 s) and CPython 3.13 (224 s). The same 1,038 tests are collected each time. By file: clauses 113, comprehension 25, deals 23, decisions 96, design numbers 22, documents 5, entangled 25, entrenchment 13, family 23, ground 11, knowledge 22, loop 70, maintenance 10, manifest 4, oracles 41, ordinal 28, phases 19, poly 18, quorum 57, renewal 43, report 13, rival 51, spec 109, standing 65, surprisal 77, viability 55. The longest single tests take about 5 to 6 seconds.
* Ruff passes on the whole tree, run last (below).

### Coverage

Line and branch coverage with `--branch`: 4,043 statements and 1,762 branches. Two statements and one partial branch are missed, all in the `if __name__ == "__main__":` block of `__main__.py` (lines 37 and 38), which runs only in a subprocess that this measurement does not follow. Every other module has no missed statement and no partial branch, `standing.py` included. The coverage run took about fourteen minutes on this container.

### Mutation testing

Two rounds of hand-written mutants, each read one survivor at a time.

* **Round 1, the module (`standing.py`): 100 mutants, 88 killed by the first run.** Of the 12 survivors, 8 were test gaps and are closed by tests (among them duplicate agents in an order, a non-market refusal, a ranking with a repeated good, an empty price, the messages of the refusals, and fractional breakpoints). Four are equivalent by argument: a `<=` where `<` is written for distinct goods; a `j != i` removed where it cannot matter; skipped finished roots; and the orientation of the wealth order in a count.
* **Round 2, the characterisation, its clauses and the oracle stage: 35 mutants, 31 killed by the first run.** Of the four survivors, two are equivalent: `al-first`, which restricts `_along` to the pairs that involve the first coordinate, and `al-range`, which adds the pairs with equal indices. On strictly positive vectors the first restriction forces proportionality, and the second adds only trivially equal pairs. Both were checked over 149,882 pairs of positive vectors; the equivalence is argued, not mechanically proved. The other two were real gaps, each now closed and each mutant now failing the new test on a copy of the tree: `d32-strength` (the ledger check did not count the delicate kinds, weights parallel but unequal and unequal multiples of the price; it now asserts reach floors for both) and `rn-barren` (the oracle runner's check of the weight cases that cannot trade had no test; a new test requires a reference that passes goods on whenever every agent scores alike to be reported).
* **Total: 135 mutants, 119 killed by the first run, 10 real gaps closed with tests, 6 equivalent by argument.**

### Differential oracles (`--full`, about 55 s, exit 0)

Every stage agrees with the package.

* Footprint and quorum: 3,473 comparisons (1,500 instances, 1,173 of them makeable; 800 graphs).
* The price test: 1,206 (six hand-derived cases; 1,000 small and 200 larger option sets; 3,389 efficient options, 492 of them price-invisible).
* Quantum values: 38 (12 GHZ-type states, 25 CHSH strategies, the magic square).
* Top trading cycles: 9,020 (3,000 outcomes; three checkers on 5,600 allocations; 300 cores; 120 sets of misreports).
* `standing`: 63,611 (1,500 markets of up to six agents, 878 with a loop of envy and 622 without, with 11,503 orderings tried against the reference's list; 600 dials with 3,304 pieces, 63 grid settings and 9 ranking settings each, of which 126 trade and 44 of those reverse; 4 dials worked by hand; and 500 weight cases, 159 with a witness that swaps in the reference, 74 of them with weights parallel but unequal, and 341 that cannot trade, 197 of them unequal multiples of the price).
* Total: 77,348 comparisons. The debt file had quoted 63,111 for the `standing` stage; that was the stage before the weight cases were added, and it is corrected there.

Each stage can fail. `tests/test_oracles.py` plants a slip in the package and in each oracle and requires disagreement, and a stage killed at its deadline reports UNKNOWN, never agreement. Agreement shows that slips of the kinds sampled are caught. Both sides were written from the same text, so it does not show the definitions right (HO-024).

### The long run: every four-agent profile of `standing`

`scripts/check_standing_exhaustive.py`, 2026-10-11, 1 min 58 s wall and 7 min 39 s of CPU on four cores (the log is kept in the working scratch area, not in the repository): 331,776 of 331,776 strict profiles, with the identity start (every start is that one up to renaming the goods). Envy has a loop in 255,120 of them, so someone trades, and none in 76,656, a ladder; these are the two counts `rival`'s long run found for trade and for none. Every one of 7,962,624 wealth starts (every order of every profile) is a rearrangement of the goods, is left where it is by top trading cycles, has the wealth order as a pecking order, and has envy pointing only toward the wealthier. 0 failures.

### Other reference checks

* **Hyperethics cross-check** at `a515d89`: gate cost is at most gate size times the dearest price on 1,178 of its random gates; the smallest rule-obeying gate of cost at least a stake has amendment gate `max(k+1, m*)` on 120 instances. Unchanged from the earlier receipt.
* **Specification lint**, with the family's translator at hypermath `a340179`: exit 0 for `L0_viability.hm` and for `L1_exchange.hm`, with 0 statements outside the translatable fragment. The axioms are prose in comments, and the translator does not translate derives or non-derives; `tests/test_spec.py` (109 tests) ties the names in the specifications to the rows, the code, the tests and the decisions.

### Packaging

* `scripts/check_sdist.py` on a fresh copy of the final tree: the source distribution (88 files) carries each of the 75 manifest-listed files byte for byte.
* `uv build` on the same copy builds the source distribution and the wheel. The wheel's contents and its install are checked below.
* The wheel has 30 members, listed with `zipfile` from a clean CPython 3.13 environment (it was not installed and imported there). The unpacked source distribution's own suite passes under CPython 3.13: 1,038 passed in 214 s, exit 0. That sdist was built before this file's last edit; it differs from the commit only in `VALIDATION.md` and `TECHNICAL_DEBT.md`, which `tests/test_documents.py` reads, so the suite was run again on the committed bytes as well (below).

### Lint

`python -m ruff check .` on the final tree, run last: all checks passed. The repository's gate is the linter; the formatter was not run, and 59 files would change under it, which this closeout does not do.

### Hosted run

The workflow `conformance`, run 38713844989 on `d60aa31` (a push to `main`, 2026-10-15), concluded `success`: `discover`, `documents`, the four `python` legs (ubuntu-latest and windows-latest, each under CPython 3.10 and 3.13) and `conformance-gate`. Each leg's "Run the test suite" step reported **1,038 passed**: ubuntu 3.13 in 223 s, windows 3.13 in 252 s, windows 3.10 in 259 s, ubuntu 3.10 in 375 s. The job logs were readable once the jobs had finished, so these per-leg counts are read, not inferred. The hosted matrix has no CPython 3.11 leg (3.11 ran locally under four hash seeds) and no macOS leg.

## Scratch checks, not kept in the repository

These were run in the working scratch area. They are reported for what they show, and where a check was worth keeping it is a native test.

* **An independent refusal-boundary fuzz** of the dial functions: 54,000 calls over 6,000 random corruptions of valid instances, each compared with a validity rule written from the requirements. No disagreement. The seeded native version in `tests/test_standing.py` runs 700 corruptions through eight functions.
* **A non-mapping in every mapping slot** of 15 functions, one slot at a time: 682 calls in the scratch check and 527 in the native test; no call answered. The exceptions raised are AttributeError (259), TypeError (112) and ValueError (156) in the scratch run, which is why the docstring now names all three.
* **Scale.** A single cycle of envy through 3,000 agents: `envy_loop` returns the whole cycle in about 6 s with no recursion limit reached; a ladder of 3,000 agents is checked in about 3 s.
* **Two first-run "findings" that are not defects.** The non-mapping containers raise an exception, as the module now documents. And `dial_survey` with a count of 10^100 runs as long as it is asked: the count is a loop bound with no cap, and the caller chooses it.
* **A subagent fuzz** of `standing`, written by a subagent at the smallest model tier as the standing rule requires, stopped twice at its output limit without writing a file, and was stopped. Its report was never received, and no finding came from it.

## Defects found and repaired in this window

Each was reproduced first, then repaired.

1. A random factor drawn once per coordinate instead of once per agent, in three places (`tests/test_standing.py`, `_d32` in `decisions.py`, and the weight-case generator). The slip made "parallel" weights unequal in some coordinates. Each place now draws one strength per agent or vector.
2. Round 1 of the mutation run: eight test gaps (listed above).
3. Round 2 of the mutation run: two real gaps (the ledger's reach of the delicate kinds; the oracle runner's check of the weight cases that cannot trade).
4. The `standing` docstring said that a non-mapping raises AttributeError. The exceptions observed are AttributeError, TypeError and ValueError. Corrected, and the boundary is now a test.
5. Two debt rows quoted 63,111 for the `standing` oracle stage, which predates the weight cases. Corrected to 63,611.
6. The sniff audit's draw printed two numbers and the wrong one was read first, so the hosted workflow was examined before the drawn area. The drawn area was then run, and the record says so (`TECHNICAL_DEBT.md`, audit 6).
7. The hosted workflow runs `python -m pytest -q`. Observed to still print one line per test, because `pyproject.toml` adds `-vv`. Recorded as HO-031 and not changed, because the workflow is shared with the other repositories.

## What the results are evidence of

* `[FORM]` results, checked exactly or exhaustively against definitions that share no code with the implementation, or against a reference written by someone who had not seen it. For `standing`: from a start, envy has a loop exactly when top trading cycles moves someone and exactly when the start is not Pareto efficient; a pecking order exists exactly when the start is Pareto efficient; a wealth start is a ladder that top trading cycles leaves where it is; and, for the dial, some goods and some start trade at some setting exactly when the weights are neither all equal nor all positive multiples of the price, with a constructive witness from strict Cauchy–Schwarz (`thaw_witness`), agreed with `can_thaw` on every case. The first two are checked on every profile of two, three and four agents (the four by the long run); the dial on sampled instances against an independent reference.
* `[FRAME]` results, which hold inside stated assumptions: strict private rankings, one indivisible good to each agent, and no prices in the exchange itself. The dial's straight mix (1 - t) price + t own is one path among many from a common ranking to private ones, and the wealthiest-first start is a stylisation, not a model of any market.
* `[HYPER]` and `[OPEN]` items carry no check on purpose: that the present world's start is a wealth start or any particular start; that tastes which depend on what others hold thaw a ladder by the same mechanism; that a transfer from the top of a ladder is lawful (D16, the owner's; HO-030).

## Not observed

* **macOS, and CPython 3.11 on the hosted runner,** are not in the hosted matrix. CPython 3.11 ran locally under four hash seeds.
* **No independent reviewer, and no second machine.** Hash-seed independence was checked by running the suite under four seeds on one machine, not on a second.
* **Shared definitions.** The reference implementations and the package were written from the same text (HO-024).
* **No real market, start or status measure has been analysed.** Nothing physical was run. The wealth start is a stylisation (HO-030).
* **Snapshots.** The goldens record what their sources said at their commits (HO-004). If hyperethics, hyperlogic or hyperstratum change what they record, the tests keep passing against the old numbers until the goldens are regenerated deliberately.
* The loop thresholds are mean-field and the repair model is a chain; no real institution was measured (D03, D09, HO-001, HO-002, HO-003).
* Understanding has no instrument (D29, HO-020); institutional entanglement has no witness (D28); win-win holds for the permissions modelled, which are not rival (D30), and for rival freedoms only in the weak form that nobody ends below where they began, for indivisible goods held one to a party under strict rankings (D31). Ties, several units, tastes that depend on others, groups who misreport together and a publisher who lies are outside the model (HO-028), and the starting allocation is given, not derived (HO-027, B16).
* The family findings (D22, D23) are about one snapshot of two records. A citation is evidence of mention, not of need; the keyword table misses synonyms; the reading of which four open items this package touches was by hand and cannot be rerun (HO-017, HO-018).
* The repository's name on the hosting platform (`hyperorganization`) is not the field's name (`hyperoptimization`). The owner's answer is awaited, and nothing is registered in hyperstratum until then (HO-029).

## What would invalidate it

Any change to a file listed in the manifest (`tests/test_manifest.py` fails until the manifest is regenerated, which is the moment to rerun the checks above, the linter last); a regenerated golden; a change to hyperethics, hyperlogic, hyperstratum or hypermath that the records or the cross-checks read; a new CPython minor version; a new sympy release (the quantum reference uses it for exact linear algebra).

## History

* **2026-10-11, after the push.** The repository was pushed to `TimeLordRaps/hyperorganization` as `8190793`, a merge of the owner's Initial commit (`a58c56a`, an Apache-2.0 `LICENSE`) and `dbd2bf2`. The hosted workflow `conformance`, run 38174725004 on `8190793`, concluded `success` on `discover`, `documents`, `python` (ubuntu and windows, CPython 3.10 and 3.13) and `conformance-gate`. The step "Run the test suite" took 69 to 199 seconds on its four legs, so it ran the suite. The per-leg pytest summary lines were not read: the log host is refused by the session's network policy, so the per-leg test count is UNKNOWN.
* **Earlier.** `dbd2bf2` (rival freedoms, D31) and `a1a154b` (the exchange layer) were checked as described in their own receipts in the history of this file.

## Reproduce

```console
pip install -e ".[dev]"
python -m pytest                              # 1,038 tests, verbose, per-test timeout, slowest ten listed
python -m ruff check .
python -m hyperoptimization                   # the report: computed numbers, then what is not established
python -m hyperoptimization decisions | diff - DECISIONS.md
python scripts/differential_oracles.py        # add --full for the sizes quoted above
python scripts/check_standing_exhaustive.py   # all 331,776 four-agent profiles, about two minutes on four cores
python scripts/check_rival_exhaustive.py      # all 331,776 four-party ranking profiles, minutes
python scripts/regenerate_manifest.py --check
python scripts/check_sdist.py                 # needs `build` or `uv`
```
