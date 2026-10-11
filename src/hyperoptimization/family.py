"""The family read as a graph of reliance, and the ordinal phase of each repository in it.

hyperstratum records which pinned repository cites, depends on or imports which. The golden file
`data/golden_family_graph.json` is one snapshot of that record (the digest of the graph it was read
from, and the hyperstratum commit, are stored in it). An edge (source, target) is read as SOURCE RELIES
ON TARGET, so `entrenchment.phases` gives each repository the phase of its reliance: 0 if it relies on
nothing, one more than the largest phase among what it relies on, and ω if its reliance never bottoms
out. A repository has a finite phase exactly when nothing it reaches lies on a cycle.

Five views of the same record are read, from the most selective to the least. Which edges count as
reliance is a choice, and the ledger (D22) records it with what a witness should see:

    hard         an import or a declared dependency: what the build needs. Acyclic by necessity.
    formal       hard, or a citation that some specification (.hm) or source (.py) file makes.
    substantive  a citation made anywhere beyond the navigation files every repository carries
                 (README, AGENTS, GOVERNANCE, CONTRIBUTING, CITATION, CHANGELOG), or hard.
    weighty      hard, or a citation made in three or more files.
    all          every recorded edge.

A repository citing itself is not a reliance on another and is dropped. Several edges between the
same two repositories (a citation and an import) count once.

What this does not show: a citation is evidence that one repository mentions another, not that it
needs it. The cores found are where mentions go round in a circle; whether the circle is vicious is
for the owners to say. And one snapshot is not the family now.

A second record, independent of the first, is hyperstratum's live scan of the pinned fields
(`data/golden_family_live.json`). It gives the edges the wiki is built from, over all 24 pinned fields:
a link to a repository, or its name in prose, with a count. Three views of it are read: `links` (a link),
`mentions` (its name in prose, once or more) and `frequent` (ten or more of either). It also gives the
family's declared open items: every line carrying an `[OPEN]` tag, and every `open` declaration in a
`.hm` specification (which does not use the tag). `open_tally` says what they are about, by wording
only: a fixed table of keywords, which finds nothing a synonym would carry. It is a proxy for a reading
by the owners, not a replacement.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from importlib import resources
from typing import Any

from . import entrenchment
from .ground import Omega

DATA = resources.files("hyperoptimization").joinpath("data")

NAVIGATION = frozenset({"README.md", "AGENTS.md", "GOVERNANCE.md", "CONTRIBUTING.md", "CITATION.cff",
                        "CHANGELOG.md"})
HARD_KINDS = frozenset({"import", "dependency"})

Edge = Mapping[str, Any]


def _hard(edge: Edge) -> bool:
    return edge["kind"] in HARD_KINDS


def _formal(edge: Edge) -> bool:
    return _hard(edge) or any(name.endswith((".hm", ".py")) for name in edge["evidence"])


def _substantive(edge: Edge) -> bool:
    return _hard(edge) or bool(set(edge["evidence"]) - NAVIGATION)


def _weighty(edge: Edge) -> bool:
    return _hard(edge) or edge["weight"] >= 3


VIEWS: dict[str, Callable[[Edge], bool]] = {
    "hard": _hard,
    "formal": _formal,
    "substantive": _substantive,
    "weighty": _weighty,
    "all": lambda edge: True,
}


def snapshot() -> dict[str, Any]:
    """The recorded graph: repositories with their tiers, edges with kind, weight and evidence."""
    return json.loads(DATA.joinpath("golden_family_graph.json").read_text(encoding="utf-8"))


def depends_on(view: str = "all", *, data: Mapping[str, Any] | None = None) -> dict[str, frozenset[str]]:
    """Each repository and the repositories it relies on, under one view of the record."""
    if view not in VIEWS:
        raise ValueError(f"the view is one of {sorted(VIEWS)}, not {view!r}")
    record = snapshot() if data is None else data
    chosen = VIEWS[view]
    reliance: dict[str, set[str]] = {node["id"]: set() for node in record["nodes"]}
    for edge in record["edges"]:
        if edge["source"] != edge["target"] and chosen(edge):
            reliance[edge["source"]].add(edge["target"])
    return {name: frozenset(deps) for name, deps in reliance.items()}


def phases_of(view: str = "all", *, data: Mapping[str, Any] | None = None) -> dict[str, int | Omega]:
    """The ordinal phase of each repository under one view."""
    return entrenchment.phases(depends_on(view, data=data))


def cores_of(view: str = "all", *, data: Mapping[str, Any] | None = None) -> list[tuple[str, ...]]:
    """The groups of repositories that rely on one another, directly or through others."""
    return entrenchment.cores(depends_on(view, data=data))


# ---------------------------------------------------------------------------
# The live scan: a second record of reliance, and the declared open items.
# ---------------------------------------------------------------------------

FREQUENT = 10  # mentions or links that make a reliance "frequent"

LIVE_VIEWS: dict[str, Callable[[Edge], bool]] = {
    "links": lambda edge: edge["kind"] == "url",
    "mentions": lambda edge: edge["kind"] == "name",
    "frequent": lambda edge: edge["count"] >= FREQUENT,
}

# What an open item is about, by wording only. Chosen from what this package computes, before any item
# was read against it. A synonym is missed on purpose: the table is a reproducible proxy for a reading.
TOPICS: dict[str, re.Pattern[str]] = {
    "optimization": re.compile(r"\boptimi[sz](?:e[sd]?|ing|ations?)\b|\boptim(?:um|a|al|ality)\b", re.I),
    "viability": re.compile(r"\bviab(?:le|ility)\b|\bkernel\b", re.I),
    "persistence": re.compile(r"\bpersist\w*|\bsurviv\w*|\bsustain\w*|\bmaintain\w*|\bmaintenance\b|\bcorrigib\w*",
                              re.I),
    "stability": re.compile(r"\bfixed[- ]point|\battractor|\bstab(?:le|ility)\b|\bequilibri(?:um|a)\b", re.I),
    "ordinal": re.compile(r"\bordinal\w*|\brank(?:ing|s|ed)?\b|\bwell-found\w*", re.I),
    "closure": re.compile(r"\bclosure\b|\bself-referen\w*|\bcircular\b", re.I),
}


def live() -> dict[str, Any]:
    """The recorded live scan: per scanned field its pin, tag counts, `[OPEN]` lines and `.hm` open
    declarations; and the edges (links and mentions, with counts) among the scanned fields."""
    return json.loads(DATA.joinpath("golden_family_live.json").read_text(encoding="utf-8"))


def live_depends_on(view: str = "mentions", *, data: Mapping[str, Any] | None = None) -> dict[str, frozenset[str]]:
    """Each scanned field and the scanned fields it links to or names, under one view of the live scan.
    Edges to a name that was not scanned (an unpinned repository, hyperstratum itself) are dropped,
    and so are self-references."""
    if view not in LIVE_VIEWS:
        raise ValueError(f"the view is one of {sorted(LIVE_VIEWS)}, not {view!r}")
    record = live() if data is None else data
    chosen = LIVE_VIEWS[view]
    reliance: dict[str, set[str]] = {name: set() for name in record["fields"]}
    for edge in record["edges"]:
        if edge["source"] != edge["target"] and edge["target"] in reliance and chosen(edge):
            reliance[edge["source"]].add(edge["target"])
    return {name: frozenset(deps) for name, deps in reliance.items()}


def live_phases_of(view: str = "mentions", *, data: Mapping[str, Any] | None = None) -> dict[str, int | Omega]:
    """The ordinal phase of each scanned field under one live view."""
    return entrenchment.phases(live_depends_on(view, data=data))


def live_cores_of(view: str = "mentions", *, data: Mapping[str, Any] | None = None) -> list[tuple[str, ...]]:
    """The groups of scanned fields that rely on one another under one live view."""
    return entrenchment.cores(live_depends_on(view, data=data))


def open_items(*, data: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every declared open item of the pinned fields: each line carrying an `[OPEN]` tag (form
    "bracket") and each `open` declaration of a `.hm` specification (form "declaration", whose text
    is its name and comment block). Each is {field, file, line, form, text}."""
    record = live() if data is None else data
    items: list[dict[str, Any]] = []
    for name, entry in record["fields"].items():
        for file, line, text in entry.get("open_items", []):
            items.append({"field": name, "file": file, "line": line, "form": "bracket", "text": text})
        for file, line, label, text in entry.get("hm_open", []):
            items.append({"field": name, "file": file, "line": line, "form": "declaration",
                          "text": f"{label}: {text}"})
    return items


def topics_of(text: str) -> set[str]:
    """The topics of TOPICS that the wording of `text` mentions."""
    return {topic for topic, pattern in TOPICS.items() if pattern.search(text)}


def open_tally(*, data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """How many declared open items there are, how many mention each topic, and how many mention none."""
    items = open_items(data=data)
    hits = [topics_of(item["text"]) for item in items]
    return {"items": len(items), "topics": {topic: sum(topic in found for found in hits) for topic in TOPICS},
            "untouched": sum(not found for found in hits)}


__all__ = ["DATA", "FREQUENT", "HARD_KINDS", "LIVE_VIEWS", "NAVIGATION", "TOPICS", "VIEWS", "cores_of",
           "depends_on", "live", "live_cores_of", "live_depends_on", "live_phases_of", "open_items",
           "open_tally", "phases_of", "snapshot", "topics_of"]
