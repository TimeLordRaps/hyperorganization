# hyperoptimization

Optimization as a self-closing process: existence, freedom, corrigibility and renewal, without money.

**Status, 2026-10-11.** `0.1.0.dev0`. Standard library only, Python 3.10 or later. Every result carries one of the family's tags: `[FORM]` checked exactly, `[FRAME]` holds in stated assumptions, `[HYPER]` a proposal nothing here witnesses, `[OPEN]` unresolved. Nothing here shows that reality adheres to any of it.

## What it is

A toolkit of small finite models for one question: when does a system that wants to persist, is free to act, can be corrected and can renew itself end up under a *single optimal law*, and what does that law look like from the inside?

The starting point is Tyler Roost's, quoted in full in [PROVENANCE.md](PROVENANCE.md). Short form: the optimality is "the self-limiting lawful structure that itself is law bound by the optimality"; it "may converge from many simulated law structures"; and there should be "no monetary construction". The answer this package gives, in one paragraph, is in [DESIGN.md](DESIGN.md) §2; the numbered results with their proofs or oracles are §3 to §9.

## Run it

    pip install -e .
    python -m hyperoptimization               # the report: computed numbers, then what is not established
    python -m hyperoptimization decisions     # the decision ledger, as DECISIONS.md
    python -m pytest                          # verbose, per-test timeout, slowest ten listed

The tests are run verbose with a 60 second per-test timeout (`pyproject.toml`), because a quiet suite hides which test is slow or stuck.

## What is in it

| module | what it computes | oracle the tests hold it to |
|---|---|---|
| `quorum` | `T(x)` = P(at least `q` of `m` seats compromised); its single unstable fixed point; duality; flow | enumeration of every outcome; Sturm's theorem in rationals; a float grid |
| `loop` | a body overseen by bodies of its own kind, with overseers who command or only veto; critical pressure; the ghost delay | an exact identity in dual variables; a direct scan of the defining ratio; exact fixed-point counts either side |
| `renewal` | life of a body whose seats are attacked and repaired, repair only while it holds | three independent solvers; numerical limits |
| `viability` | the viability kernel, ranks, the greatest sustaining law, exemption, corrigibility, convergence from many laws; self-maintenance as a greatest fixed point (the kernel is the union of the sets that support themselves; the inductive reading is empty) | every law and every start on random games; independent minimax and reachability; every subset of every sampled game |
| `knowledge` | what a participant who sees part of the game can certify (what it cannot see it treats as unsafe); what pooling the knowledge of parties unlocks, which exists only across a border between territories; the smallest knowledge a permission needs | an independent value iteration; every subset of every sampled game; the sets that support themselves |
| `comprehension` | a decision's footprint (what it rests on and what rests on it) and what it asks of the people deciding: the fewest who between them understand it, the attendance at which whoever comes does, the people who cannot be left out, the entry barrier and its price in capture resistance | brute force over every group of people on small instances; an independent closure; Condorcet's jury theorem in exact rationals |
| `surprisal` | prices as bits; gate cost; the probability a gate is taken, bracketed; the gap a common principal opens | inclusion–exclusion; hyperethics' own numbers through a golden file |
| `entrenchment` | protection as the weakest link; authority height; cycles; the phase of a dependency graph with many governors, and its cores | independent shortest-path search; every governance graph on up to five rules; the longest walk on every digraph on up to four nodes |
| `family` | the family's recorded citations, dependencies and imports read as reliance, with the phase of each repository under five views; a second record from hyperstratum's live scan; the declared open items and what they are about, by wording | two recorded snapshots of hyperstratum; definitions that share no code with `entrenchment`; a keyword table fixed before the items were read |
| `ordinal` | Cantor normal form below ε₀; Goodstein sequences with an ordinal for every term | an independent valuation; hand-worked terms |
| `decisions` | 23 decisions: choice, what was not chosen, what a witness should see, what would refute it, and an executable check for 20 of them | each check, and each of its failure clauses, is broken on purpose in the tests and must then fail |
| `report` | one page of computed numbers and the open obligations | numbers pinned by value |

## How to read a result

`Verdict` has no truth value; `if verdict:` raises. `HOLDS_IN_MODEL` means checked against one finite structure, one parameter set or one grid, and it records the numbers it read as `evidence`. `FAILS_IN_MODEL` must name its counterexample, and one counterexample settles a universal claim. A passing suite is evidence about the cases it ran. Every property test also asserts that its sample was not empty.

## Why there is no money

A price is the surprisal of a compromise, `bits(p) = -log2 p`. Probabilities of independent events multiply and bits add, so hyperethics' cover search is unchanged: the cheapest cover of a gate is its most probable way in, and the probability of taking the gate lies between `2^-cost` and `N * 2^-cost` for `N` minimal covers. A stake is how many independent attempts a prize attracts, in `log2`. Where compromises are not independent, `additivity_gap` says by how many bits the price is overstated.

## What it does not show

Written down in [DESIGN.md](DESIGN.md) §11 and, one by one with what would resolve each, in [DECISIONS.md](DECISIONS.md). The ones that matter most:

* A viability kernel is persistence, not goodness. The greatest law permits some stable arrangements a participant would call tyranny. Why the law that preserves existence should be called optimal is the family's open GC-5.
* The loop thresholds are mean-field. Positive association between compromises makes them upper bounds when the quorum is above the mean; how large the correction is on a real graph is not computed.
* That physical reality adheres to such a structure is a proposal, with a stated form of refuter, and no witness.
* That this closes the family's open items is untested, and the first passes do not support it. Of 122 declared open items (116 lines tagged `[OPEN]` and 6 `.hm` declarations, in 11 of 24 pinned fields) none is about optimization or viability by wording, four are touched by hand and none is closed; the ordinal phase of each repository is computed in two independent records of the family (a core of mutual reliance, hyperethics with hyperphysics and metamathethicology, survives both). The map from open items to the repositories that would close them is not built.

## Family

* [hyperethics](https://github.com/TimeLordRaps/hyperethics), layer 2 (the rule that no power may be corrupting): `data/golden_l2_gate_bits.json` records its `gate_cost`; `scripts/verify_against_hyperethics.py` checks the breadth and synthesis theorems against its own code. Its principal open problem, the consistency self-maintenance paradox, is read in `DESIGN.md` R6 as a self-supporting set (a reading, not a resolution).
* [hyperlogic](https://github.com/TimeLordRaps/hyperlogic): its Gödel–Löb prover supplies `data/golden_gl_loops.json`.
* [hyperstratum](https://github.com/TimeLordRaps/hyperstratum): its recorded family graph supplies `data/golden_family_graph.json`, and its scan of the pinned fields (links, mentions, `[OPEN]` lines) supplies `data/golden_family_live.json`.
* This package imports none of them. Golden files are regenerated, never edited, by `scripts/regenerate_golden.py`, `scripts/regenerate_golden_gl.py`, `scripts/regenerate_golden_family.py` and `scripts/regenerate_golden_live.py`.

## Files

[DESIGN.md](DESIGN.md) the theory and the spec map · [DECISIONS.md](DECISIONS.md) the ledger (generated) · [VALIDATION.md](VALIDATION.md) the receipt · [PROVENANCE.md](PROVENANCE.md) what was said and by whom · [L0_viability.hm](L0_viability.hm) the spec in the family's language · [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) the register · [LICENSE](LICENSE) Apache-2.0.
