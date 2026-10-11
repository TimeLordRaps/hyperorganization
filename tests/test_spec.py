"""The spec, the design document, the code and the tests agree about what exists and what was said.

`L0_viability.hm` is prose in the family's dialect; the translator leaves its axioms and derives
untranslated, so nothing mechanical ties it to the code. These tests are that tie. DESIGN.md carries
a table (the spec map) with one row per name the spec declares; this file checks the rows against
the spec, against the code, against the tests, and against the decision ledger, and it checks that
Tyler Roost's three messages are in the spec and in PROVENANCE.md exactly as he sent them.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from hyperoptimization import decisions

ROOT = Path(__file__).resolve().parent.parent
SPEC = (ROOT / "L0_viability.hm").read_text(encoding="utf-8")
DESIGN = (ROOT / "DESIGN.md").read_text(encoding="utf-8")
PROVENANCE = (ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")

MESSAGES = (
    "How does this come into play in an ordinal system of checks and balances? Where a system inside "
    "of the structure is represented as all powerful, like imagine there is a central will space that "
    "can be determined that itself is optimal, and we can thus derive an optimal power structure "
    "around that will space, not dictatorially, but rather a centrally bound system that itself is "
    "mechanically governed, and represents the governance over it as an optimality over all "
    "participants/shareholders inside of the governed system, ie a codified lawset that is "
    "demonstrably stable and optimal would yield an necessity of centralization of power around it, "
    "how do ordinal laws naturally occur in such a system what theorems or proofs or can be written "
    "about such a system, and secondarily what axioms hold in such a system by nature of the systems "
    "optimality structure",
    "Like the optimality is the self-limiting lawful structure that itself is law bound by the "
    "optimality, it may converge from many simulated law structures, also considering no monetary "
    "construction which set of laws naturally fall out from free will, existence, maintainence of "
    "existence ie corrigibility, revitalization expectations like if reality/our physical universe "
    "obeys some underlying optimal hyperethical structure it can be found that the hyperphysical "
    "nature adheres within hypertethical optimalities, honestly optimalities and optimization yields "
    "the hyperfield hyperoptimization I think you can make a new public repo for that, "
    "hyperoptimization also seems like the key missing field we need to close a bunch of hyperfields, "
    "because hyperoptimization acts as its own self-closing feedback convergence catalyst for "
    "ordinal phase representations that themselves yield the necessary base fields from the "
    "hyperfields.",
    "Ive given you a lot to work with, work long and hard and let me know if any conceptual blocks "
    "get in your way, but we are inventing this together so I would like to know what decisions you "
    "make that hold key evidential expectations  from witnesses within the framework.",
)


def flat(text: str) -> str:
    """Collapse comment markers, quote markers and whitespace so a wrapped quotation can be matched."""
    return " ".join(re.sub(r"^\s*(--|>)", " ", text, flags=re.M).split())


def declared(kind: str) -> list[str]:
    return re.findall(rf"^{kind} ([a-z0-9-]+):", SPEC, flags=re.M)


def spec_map() -> list[dict[str, str]]:
    rows, inside = [], False
    for line in DESIGN.splitlines():
        if line.startswith("| spec |"):
            inside = True
            continue
        if not inside:
            continue
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0].startswith("-"):
            continue
        rows.append(dict(zip(("spec", "kind", "code", "test", "decision"), cells)))
    return rows


def all_test_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted((ROOT / "tests").glob("test_*.py")))


# --- what the spec declares -----------------------------------------------------------------------------


def test_the_spec_declares_the_axioms_derives_and_non_derives_the_design_maps():
    assert declared("axiom") == [
        "ax-existence-as-viability", "ax-free-will-as-full-support",
        "ax-corrigibility-as-reachability", "ax-self-repair", "ax-independence",
    ]
    assert declared("derive") == [
        "d-greatest-law", "d-exemption-is-fatal", "d-corrigible-kernel-inside", "d-rank-certificate",
        "d-quorum-duality", "d-one-critical-point", "d-command-veto-duality", "d-renewal-regimes",
        "d-price-brackets-probability", "d-independence-understates", "d-weakest-link",
        "d-no-rank-without-sovereign", "d-guarded-loops", "d-goodstein-descent",
        "d-breadth-and-synthesis", "d-self-maintenance-is-coinductive", "d-dependency-phase",
    ]
    assert declared("non-derive") == [
        "nd-persistence-is-optimal", "nd-physical-adherence", "nd-determinacy-is-soundness",
        "nd-welfare-from-viability", "nd-consistency-self-maintenance-resolved",
        "nd-citation-is-need",
    ]
    assert declared("open") == [
        "nd-persistence-is-optimal", "nd-closing-catalyst", "nd-mean-field-adequacy",
        "nd-self-funding",
    ]


def test_the_design_map_has_a_row_for_every_declared_name_and_no_others():
    names = set(declared("axiom") + declared("derive") + declared("non-derive") + declared("open"))
    rows = spec_map()
    assert {r["spec"] for r in rows} == names
    assert len(rows) == len(names)  # no name twice


def test_each_row_says_the_right_kind():
    for row in spec_map():
        kind = {"axiom": "axiom", "derive": "derive", "non-derive": "non-derive", "open": "open"}[row["kind"]]
        assert row["spec"] in declared(kind), row


# --- what each row points at exists --------------------------------------------------------------------------


@pytest.mark.parametrize("row", spec_map(), ids=lambda r: r["spec"])
def test_every_row_names_code_a_test_and_a_decision_that_exist(row):
    code, test, decision = row["code"], row["test"], row["decision"]
    if code != "—":
        if code.startswith("scripts/"):
            assert (ROOT / code).is_file(), code
        elif code.startswith("data/"):
            assert (ROOT / "src" / "hyperoptimization" / code).is_file(), code
        else:
            module, attr = code.split(".", 1)
            target = importlib.import_module(f"hyperoptimization.{module}")
            assert hasattr(target, attr), code
    if test.startswith("script:"):
        assert (ROOT / "scripts" / test.split(":", 1)[1]).is_file(), test
    elif test != "—":
        assert re.search(rf"^def {re.escape(test)}\(", all_test_text(), flags=re.M), test
    decisions.decision(decision)  # raises KeyError if the ledger has no such id


def test_an_item_with_a_test_has_code_and_an_item_with_neither_is_a_non_derive_or_open():
    for row in spec_map():
        if row["test"] == "—":
            assert row["code"] == "—" and row["kind"] in ("non-derive", "open"), row
        else:
            assert row["code"] != "—", row


# --- what was said ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("index", range(3))
def test_each_message_is_in_the_spec_and_in_provenance_exactly(index):
    message = " ".join(MESSAGES[index].split())
    assert message in flat(SPEC), f"message {index + 1} is missing or altered in L0_viability.hm"
    assert message in flat(PROVENANCE), f"message {index + 1} is missing or altered in PROVENANCE.md"


def test_the_doubled_space_in_the_third_message_survives_in_provenance():
    assert "evidential expectations  from witnesses" in PROVENANCE


def test_nothing_is_attributed_to_the_owner_beyond_what_was_said():
    # every blockquote in PROVENANCE is one of the three messages or the L2 rule it cites
    quoted = [" ".join(line[1:].split()) for line in PROVENANCE.splitlines() if line.startswith(">")]
    allowed = [" ".join(m.split()) for m in MESSAGES]
    assert quoted and all(q in allowed for q in quoted)


# --- the spec's honesty ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("gc", [f"GC-{n}" for n in range(9)])
def test_the_spec_grades_every_graduation_criterion(gc):
    assert re.search(rf"{gc}: ", SPEC)


def test_the_spec_is_honest_about_its_status():
    text = flat(SPEC)
    for admission in (
        "adopted and not derived",
        "A passing check CONFIRMS nothing universally",
        "This layer does not graduate as a theory of optimization",
        "A kernel is persistence, not goodness",
        "Four of nine discharged",
    ):
        assert admission in text, admission


def test_the_spec_declares_no_evaluative_primitive():
    primitives = re.findall(r"^(?:primitive|opaque) ([a-zA-Z0-9-]+) ::", SPEC, flags=re.M)
    assert primitives
    for banned in ("good", "ought", "prefer", "value", "harm", "choose", "welfare", "utility"):
        assert banned not in [p.lower() for p in primitives]


def test_the_only_unit_of_price_is_bits():
    assert re.search(r"^primitive Bits ::", SPEC, flags=re.M)
    assert not re.search(r"^primitive (Money|Currency|Cost|Dollar)", SPEC, flags=re.M | re.I)


def test_the_readme_mentions_every_module_of_the_package():
    modules = [p.stem for p in (ROOT / "src" / "hyperoptimization").glob("*.py")
               if p.stem not in ("__init__", "__main__", "ground", "poly")]
    missing = [m for m in modules if f"`{m}`" not in README]
    assert not missing, f"modules not named in the README table: {missing}"
