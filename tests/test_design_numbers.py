"""The numbers the design quotes are the numbers the models compute.

DESIGN.md section 14 and README.md quote counts and values that were measured on the models: how often the
gain from pooling is mutual, how many of the family's repositories carry a footprint of each size, what the
barrier costs in bits, how a quantum strategy compares with the classical value. A document asserts; only a
computation establishes. A sentence that quotes a number nobody recomputes drifts the first time a seed, a
sample size or a definition changes, and goes on claiming what the code no longer says.

Each test here recomputes quoted numbers from the models and requires the document to contain the sentence
built from the recomputed values, so a change on either side fails. The last tests damage the document on
purpose and require this file to notice, so the comparison is known to be able to fail.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest
from test_deals import menu_counts
from test_entangled import GHZ_TYPE_STATES
from test_knowledge import split_counts
from test_rival import LARGER_MARKETS, RECORD_MARKETS

from hyperoptimization import comprehension, deals, decisions, entangled, family, rival, surprisal

ROOT = Path(__file__).resolve().parent.parent
DESIGN = (ROOT / "DESIGN.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def quoted(sentence: str, document: str = DESIGN, name: str = "DESIGN.md") -> None:
    """The document contains exactly this sentence (words as written, whitespace aside). On a miss the error
    shows where the document and the sentence part ways, so the prose can be corrected without a search. (It
    raises rather than asserts, and hides its own frame, so that pytest does not print the whole document into
    the failure.)"""
    __tracebackhide__ = True
    flat = " ".join(document.split())
    if sentence in flat:
        return
    for cut in range(len(sentence) - 1, 11, -1):  # the longest opening of the sentence the document does say
        at = flat.find(sentence[:cut])
        if at >= 0:
            raise AssertionError(f"{name} does not say: {sentence!r}; it says {flat[at:at + len(sentence) + 8]!r}")
    raise AssertionError(f"{name} does not say: {sentence!r}")


def ledger_line(decision_id: str, pattern: str) -> re.Match[str]:
    """The match of `pattern` in the evidence the ledger prints for a decision, computed afresh."""
    verdict = decisions.run(decisions.decision(decision_id))
    assert verdict is not None
    for line in verdict.evidence:
        found = re.search(pattern, line)
        if found:
            return found
    raise AssertionError(f"{decision_id} prints no evidence line matching {pattern!r}: {verdict.evidence}")


# --- knowledge (14.1) -------------------------------------------------------------------------------------------------


def test_the_quoted_split_counts_are_the_computed_ones():
    counts = split_counts()
    total = sum(counts.values())
    quoted(f"({total:,} sampled splits with a non-empty greatest law)")
    quoted(f"the gain was mutual in {counts['mutual']}, one-sided in {counts['one-sided']} "
           f"and absent in {counts['none']}")


def test_the_quoted_territorial_counts_are_the_ledgers():
    found = ledger_line("D25", r"territories: (\d+) sealed .* and (\d+) with a crossing; (\d+) gained")
    sealed, leaky, gained = map(int, found.groups())
    assert sealed + leaky == 40 and gained <= leaky  # 20 seeds, with and without a leak
    quoted(f"in the ledger's territorial games {gained} of the {leaky} games with a crossing gained")


# --- understanding (14.2) ---------------------------------------------------------------------------------------------


def test_the_quoted_cost_of_the_barrier_in_bits_is_the_computed_one():
    know = {f"p{i}": ["a", "b", "c"] for i in range(3)} | {f"q{i}": ["a"] for i in range(4)}
    seat = Fraction(1, 4)
    open_door = comprehension.requirement(know, ["a", "b", "c"], 0)
    raised = comprehension.requirement(know, ["a", "b", "c"], 1)
    bits_open = comprehension.capture_bits(open_door, seat)
    bits_raised = comprehension.capture_bits(raised, seat)
    assert bits_open == pytest.approx(surprisal.council_bits(7, 5, seat))  # the number is the council price of 5 of 7
    quoted(f"with no barrier the required quorum is {open_door.required} of {len(open_door.pool)}, "
           f"at about {bits_open:.1f} bits; with the barrier at 1 it is {raised.required} of {len(raised.pool)}, "
           f"at about {bits_raised:.1f} bits, for a seat probability of {seat}")


def test_the_quoted_family_footprints_are_the_computed_ones():
    loads = comprehension.loads(family.depends_on("formal"))
    sizes = Counter(len(fp.nodes) for fp in loads.values())
    assert set(sizes) == {1, 5, 6, 8}, sizes  # no size goes unquoted
    quoted(f"{sizes[1]} of {len(loads)} repositories have a footprint of one")
    quoted(f"{sizes[5]} have 5, {sizes[6]} have 6 and {sizes[8]} have 8")
    biggest = {name for name, fp in loads.items() if len(fp.nodes) == 8}
    core = {"hyperethics", "hyperphysics", "metamathethicology"}
    rests_on = {"hypermath", "ordinatics", "grounded-hypercalculi", "grounded-hyperset-theory"}
    assert biggest == core | {"hyperlogic"}
    for name in core:  # each: the core, what it rests on, and hyperlogic, which rests on it
        assert loads[name].nodes == frozenset(core | rests_on | {"hyperlogic"})
    for name in sorted(biggest | rests_on):
        quoted(name)
    substantive = comprehension.loads(family.depends_on("substantive"))
    largest = max(len(fp.nodes) for fp in substantive.values())
    holders = [name for name, fp in substantive.items() if len(fp.nodes) == largest]
    assert len(holders) == 1  # "the largest footprint is N (name)" names one repository
    quoted(f"Under the substantive view the largest footprint is {largest} ({holders[0]})")


# --- deals (14.3) -----------------------------------------------------------------------------------------------------


def test_the_quoted_three_doors_are_the_computed_ones():
    game, base, openings, grouping = deals.three_doors()
    values = [deals.opening_value(game, base, after, ["hub"], dimension=grouping) for after in openings]
    x, y, both = (dict(v.allowed) for v in values)
    new = [v.new_states for v in values]
    quoted(f"Their values are (x {x['x']}), (y {y['y']}) and (x {both['x']}, y {both['y']}), "
           f"with {new[0]}, {new[1]} and {new[2]} new states")
    points = [(1, 1), (2, 0), (0, 2)]  # the tie quoted in the same item
    assert deals.select_by_price(points, (1, 1)) == [0, 1, 2] and deals.supporting_weights(points, 0) is not None
    quoted("with options (1,1), (2,0) and (0,2) the first ranks level with the others at the price (1,1)")


def test_the_quoted_menu_counts_are_the_computed_ones():
    menus, multiple, hidden = menu_counts()
    assert hidden == 0 and 0 < multiple < menus  # the prose says "none hid one": a hidden option would change it
    quoted(f"Among {menus} random menus of what coalitions of one or two people can certify "
           f"(500 random games in the test), {multiple} offered more than one efficient option and none hid "
           "one from every price")
    found = ledger_line("D24", r"among (\d+) random menus .*, (\d+) offer more than one efficient option "
                               r"and (\d+) hide one")
    ledger_menus, ledger_multiple, ledger_hidden = map(int, found.groups())
    assert ledger_multiple == ledger_hidden == 0  # "found none either"
    quoted(f"the ledger's smaller sample ({ledger_menus} menus) found none either")


def test_the_quoted_size_of_the_trap_is_the_ledgers_and_the_formula():
    found = ledger_line("D24", r"at 1, 5, 40 on each coherence dimension it first outscores a safe opening "
                               r"at (\d+), (\d+), (\d+) rooms")
    sizes = list(map(int, found.groups()))
    assert sizes == [(3 * p - 1) // 7 + 1 for p in (1, 5, 40)]
    quoted("a trap of (3p − 1)//7 + 1 rooms (integer division)")


# --- entangled decisions (14.4) ---------------------------------------------------------------------------------------


def test_the_quoted_quantum_values_are_the_computed_ones():
    ghz, magic, chsh = entangled.ghz_game(), entangled.magic_square_game(), entangled.chsh_game()
    ghz_classical, ghz_quantum = entangled.classical_value(ghz), entangled.quantum_value(ghz, entangled.ghz_strategy(1, 1))
    magic_classical = entangled.classical_value(magic)
    magic_quantum = entangled.quantum_value(magic, entangled.magic_square_strategy())
    strategy = entangled.chsh_strategy((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)),
                                       (Fraction(119, 169), Fraction(120, 169)),
                                       (Fraction(119, 169), Fraction(-120, 169)))
    chsh_classical, chsh_quantum = entangled.classical_value(chsh), entangled.quantum_value(chsh, strategy)
    quoted(f"| {ghz.players} | {ghz_classical} | {ghz_quantum}, with the GHZ state")
    quoted(f"| {magic.players} | {magic_classical} | {magic_quantum}, with two entangled pairs")
    tsirelson = (2 + 2**0.5) / 4
    quoted(f"| CHSH | {chsh.players} | {chsh_classical} | {chsh_quantum} ≈ {float(chsh_quantum):.5f} from rational angles")
    quoted(f"the Tsirelson value (2 + √2)/4 ≈ {tsirelson:.5f} is irrational")
    quoted(f"reached to within about {round((tsirelson - float(chsh_quantum)) * 1e6)} × 10⁻⁶")
    quoted(f"(2 + √2)/4, about {tsirelson:.4f}")
    quoted(f"{ghz_classical} to {ghz_quantum}; two parties, the Mermin–Peres magic square: {magic_classical} to "
           f"{magic_quantum}", README, "README.md")
    assert ghz_quantum == magic_quantum == 1 and chsh_quantum < 1


def test_the_quoted_number_of_states_the_formula_was_checked_on_is_the_tests():
    words = {10: "ten"}
    assert len(GHZ_TYPE_STATES) in words and len(set(GHZ_TYPE_STATES)) == len(GHZ_TYPE_STATES)
    quoted(f"(checked on {words[len(GHZ_TYPE_STATES)]} states)")


# --- the guard itself -------------------------------------------------------------------------------------------------


def test_a_changed_number_in_the_document_is_noticed():
    counts = split_counts()
    sentence = f"the gain was mutual in {counts['mutual']}, one-sided in"
    quoted(sentence)  # the document as written passes
    for damage in (str(counts["mutual"] + 1), str(counts["mutual"] - 1)):
        broken = DESIGN.replace(f"mutual in {counts['mutual']},", f"mutual in {damage},")
        assert broken != DESIGN
        with pytest.raises(AssertionError, match="does not say"):
            quoted(sentence, broken)


def test_a_missing_sentence_and_a_changed_unit_are_noticed():
    with pytest.raises(AssertionError, match="does not say"):
        quoted("a sentence the design does not contain, 717 of 1,092")
    with pytest.raises(AssertionError, match="does not say"):
        quoted("at about 6.3 nats", DESIGN)
    with pytest.raises(AssertionError, match="prints no evidence line"):
        ledger_line("D25", r"this line is never printed")


def test_whitespace_in_the_source_lines_does_not_matter_only_the_words():
    quoted("the required quorum is 5 of 7", "x\n  the   required\nquorum is 5   of 7\n", "inline")


# --- rival freedoms (14.5) ----------------------------------------------------------------------------------------------


def test_the_quoted_rival_freedom_numbers_are_the_computed_ones():
    two, three = rival.survey(2), rival.survey(3)
    quoted(f"all {two['profiles']} profiles of two parties and all {three['profiles']} of three")
    quoted(f"on all {math.factorial(4) ** 4:,} profiles of four by a long run outside the suite")
    quoted(f"In {three['trade']} of the {three['profiles']} profiles of three parties someone moves")
    quoted(f"in the other {three['still']} everybody keeps what they hold")
    quoted(f"finds a profitable lie in {three['gameable_boston']} of the {three['profiles']} profiles of three parties")
    assert three["gameable_ttc"] == 0 and two["gameable_boston"] == 0
    sampled = sum(count for _n, count in LARGER_MARKETS)
    found = ledger_line("D31", r"and (\d+) seeded markets of four and five agents")
    quoted(f"on {sampled} seeded markets of four to six parties in the tests and {found.group(1)} in the ledger")
    found = ledger_line("D31", r"of (\d+) records of three agents on (\d+) markets, the (\d+) that every agent accepts")
    records, markets, accepted = found.groups()
    quoted(f"Checked on every record of three parties on {markets} markets ({records} records, {accepted} of them accepted "
           f"by every party, all describing the TTC outcome) and of up to four parties on {RECORD_MARKETS} random markets")


def test_the_quoted_swap_a_price_cannot_see_is_the_computed_one():
    vectors = {"x-rich": (5, 1), "y-rich": (1, 5)}
    weights = {"A": (3, 1), "B": (1, 3)}
    endowment = {"A": "y-rich", "B": "x-rich"}
    outcome = rival.top_trading_cycles(endowment, {a: rival.rank_by_weights(vectors, w) for a, w in weights.items()})
    worth = {(a, g): sum(w * x for w, x in zip(weights[a], vectors[g])) for a in weights for g in vectors}
    before, after = worth["A", endowment["A"]], worth["A", outcome["A"]]
    assert (before, after) == (8, 16) == (worth["B", endowment["B"]], worth["B", outcome["B"]])
    quoted(f"Each doubles its worth by swapping ({before} to {after})")
    quoted("Take two slots, (5, 1) and (1, 5) in freedom of x and of y")


def test_the_quoted_agreement_with_the_exchange_reference_is_the_full_runs_count():
    sizes = {key: pair[1] for key, pair in __import__("importlib").import_module("test_oracles").runner.SIZES["rival"].items()}
    total = sizes["outcomes"] + sizes["allocations"] * 8 + sizes["cores"] + sizes["misreports"]
    quoted(f"agrees on {total:,} comparisons in the full run ({sizes['outcomes']:,} outcomes, three checkers on "
           f"{sizes['allocations'] * 8:,} allocations, {sizes['cores']} cores and {sizes['misreports']} sets of misreports)")
