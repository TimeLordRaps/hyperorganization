"""The family read as a graph of reliance, and the ordinal phase of each repository in it.

`data/golden_family_graph.json` is one snapshot of hyperstratum's record of which pinned repository
cites, depends on or imports which. These tests hold the reading of it to three things: the snapshot is
what it says it is, each view of it selects the edges it claims to, and the phases and cores computed
from every view agree with definitions that share no code with `entrenchment` (the longest walk and
Floyd and Warshall's closure, re-stated below).

The pinned findings are findings about THIS snapshot (hyperstratum 87aadc3, graph digest b56f1a3c...),
not about the family as it is today.
"""

from __future__ import annotations

import json

import pytest

from hyperoptimization import family
from hyperoptimization.ground import OMEGA

CORE = ("hyperethics", "hyperphysics", "metamathethicology")


def walk_phase(deps):
    names = sorted(deps)
    n = len(names)
    has_walk = [set(names)]
    for _ in range(n):
        prior = has_walk[-1]
        has_walk.append({v for v in names if any(w in prior for w in deps[v])})
    return {v: OMEGA if v in has_walk[n] else max(k for k in range(n + 1) if v in has_walk[k]) for v in names}


def closure_groups(deps):
    names = sorted(deps)
    reach = {(a, b): b in deps[a] for a in names for b in names}
    for k in names:
        for a in names:
            for b in names:
                if reach[(a, k)] and reach[(k, b)]:
                    reach[(a, b)] = True
    groups, placed = [], set()
    for a in names:
        if a in placed or not reach[(a, a)]:
            continue
        group = tuple(b for b in names if reach[(a, b)] and reach[(b, a)])
        placed.update(group)
        groups.append(group)
    return groups


def edge(source, target, kind="citation", weight=1, evidence=("README.md",)):
    return {"source": source, "target": target, "kind": kind, "weight": weight, "evidence": list(evidence)}


def tiny(*edges):
    names = sorted({e["source"] for e in edges} | {e["target"] for e in edges})
    return {"nodes": [{"id": n, "tier": "x"} for n in names], "edges": list(edges)}


# --- the snapshot is what it says it is ----------------------------------------------------------------------------


def test_the_snapshot_records_where_it_came_from_and_is_internally_consistent():
    data = family.snapshot()
    assert data["schema"] == "vstd-family-graph/1" and data["source"].endswith("family-graph.json")
    assert len(data["source_digest"]) == 64 and int(data["source_digest"], 16) >= 0
    assert len(data["hyperstratum_commit"]) == 40 and int(data["hyperstratum_commit"], 16) >= 0
    ids = [n["id"] for n in data["nodes"]]
    assert ids == sorted(ids) and len(ids) == len(set(ids)) == 21
    assert len(data["edges"]) == 74
    assert {e["kind"] for e in data["edges"]} == {"citation", "import", "dependency"}
    assert sum(e["kind"] == "citation" for e in data["edges"]) == 67
    assert all(e["source"] in ids and e["target"] in ids for e in data["edges"])
    assert all(isinstance(e["weight"], int) and e["weight"] >= 1 and e["evidence"] for e in data["edges"])


def test_the_snapshot_is_the_json_the_regeneration_script_describes():
    raw = json.loads((family.DATA / "golden_family_graph.json").read_text(encoding="utf-8"))
    assert raw == family.snapshot()
    assert "regenerate_golden_family.py" in raw["command"]


# --- the views select what they say ----------------------------------------------------------------------------------


def pairs(view):
    return {(a, b) for a, deps in family.depends_on(view).items() for b in deps}


def test_each_view_refines_the_one_before_it():
    assert pairs("hard") <= pairs("formal") <= pairs("substantive") <= pairs("all")
    assert [len(pairs(v)) for v in ("hard", "formal", "substantive", "weighty", "all")] == [4, 15, 52, 27, 67]


def test_the_boilerplate_views_drop_navigation_links_and_keep_everything_else():
    data = tiny(edge("a", "b", evidence=("README.md", "AGENTS.md", "GOVERNANCE.md")),
                edge("a", "c", evidence=("README.md", "DESIGN.md")),
                edge("b", "c", evidence=("L0_x.hm",)), edge("c", "a", weight=3, evidence=("README.md",)),
                edge("c", "b", kind="import", evidence=("src/c/x.py",)),
                edge("d", "a", evidence=("src/d/x.py",)),  # a citation made in source code, not an import
                edge("b", "b", evidence=("DESIGN.md",)))
    assert family.depends_on("all", data=data) == {"a": {"b", "c"}, "b": {"c"}, "c": {"a", "b"}, "d": {"a"}}
    assert family.depends_on("substantive", data=data) == {"a": {"c"}, "b": {"c"}, "c": {"b"}, "d": {"a"}}
    assert family.depends_on("formal", data=data) == {"a": set(), "b": {"c"}, "c": {"b"}, "d": {"a"}}
    assert family.depends_on("weighty", data=data) == {"a": set(), "b": set(), "c": {"a", "b"}, "d": set()}
    assert family.depends_on("hard", data=data) == {"a": set(), "b": set(), "c": {"b"}, "d": set()}


def test_an_unknown_view_is_refused_and_every_repository_is_a_node_even_with_no_edges():
    with pytest.raises(ValueError, match="view"):
        family.depends_on("everything")
    assert set(family.depends_on("hard")) == {n["id"] for n in family.snapshot()["nodes"]}


# --- what the phases say of this snapshot -----------------------------------------------------------------------


@pytest.mark.parametrize("view", sorted(family.VIEWS))
def test_phases_and_cores_agree_with_the_definitions_on_every_view(view):
    deps = family.depends_on(view)
    assert family.phases_of(view) == walk_phase(deps)
    assert family.cores_of(view) == closure_groups(deps)


def test_what_the_family_imports_is_well_founded():
    assert family.cores_of("hard") == []
    phases = family.phases_of("hard")
    assert phases["metamathethicology"] == 1
    assert sum(p == 0 for p in phases.values()) == 20 and OMEGA not in phases.values()


def test_what_the_family_means_is_not_ethics_physics_and_metamathethicology_rely_on_one_another():
    # the smallest core that survives every view that keeps the specifications: in the .hm files and
    # the code, hyperethics' will, hyperphysics' electrical reading and metamathethicology's transport
    # are mutually presupposed, while no import runs round the loop
    assert family.cores_of("formal") == [CORE]
    for view in ("formal", "substantive", "weighty", "all"):
        assert any(set(CORE) <= set(core) for core in family.cores_of(view)), view
    assert family.cores_of("weighty") == [CORE, ("hypergrammar", "hypermath")]
    assert family.cores_of("substantive") == [
        ("hyperchemistry", "hyperethics", "hypergrammar", "hyperlogic", "hypermath", "hyperphysics",
         "metamathethicology"), ("hyperspace", "hypertime")]
    all_cores = family.cores_of("all")
    assert [len(c) for c in all_cores] == [11, 2, 2]
    assert ("hyperobjectivity", "hypersubjectivity") in all_cores and ("hyperspace", "hypertime") in all_cores


def test_the_formal_view_puts_hypermath_one_step_above_a_foundation_and_the_core_beyond_any_finite_step():
    phases = family.phases_of("formal")
    assert phases["hypermath"] == 1
    assert all(phases[n] is OMEGA for n in CORE + ("hyperlogic",))  # hyperlogic relies on the core
    assert sorted(n for n, p in phases.items() if p is OMEGA) == sorted(CORE + ("hyperlogic",))
    assert sum(p == 0 for p in phases.values()) == 16


# --- the same question put to a second, independent record: hyperstratum's live scan --------------------------


def test_the_live_snapshot_records_what_it_scanned_and_what_it_could_not():
    live = family.live()
    assert len(live["hyperstratum_commit"]) == 40 and live["unscanned"] == ["hyperbiology"]
    assert len(live["fields"]) == 24 and len(live["edges"]) == 149
    assert {e["kind"] for e in live["edges"]} == {"name", "url"}
    assert all(e["count"] >= 1 and e["source"] in live["fields"] for e in live["edges"])
    assert sum(len(v["open_items"]) for v in live["fields"].values()) == 116
    assert sum(len(v["hm_open"]) for v in live["fields"].values()) == 6
    raw = json.loads((family.DATA / "golden_family_live.json").read_text(encoding="utf-8"))
    assert raw == live and "regenerate_golden_live.py" in raw["command"]


def test_the_live_views_select_what_they_say_and_drop_what_was_never_scanned():
    data = {"fields": {"a": {}, "b": {}, "c": {}},
            "edges": [{"source": "a", "target": "b", "kind": "url", "count": 1},
                      {"source": "b", "target": "a", "kind": "name", "count": 2},
                      {"source": "c", "target": "a", "kind": "name", "count": 12},
                      {"source": "a", "target": "ghost", "kind": "name", "count": 99},
                      {"source": "b", "target": "b", "kind": "name", "count": 99}]}
    assert family.live_depends_on("links", data=data) == {"a": {"b"}, "b": set(), "c": set()}
    assert family.live_depends_on("mentions", data=data) == {"a": set(), "b": {"a"}, "c": {"a"}}
    assert family.live_depends_on("frequent", data=data) == {"a": set(), "b": set(), "c": {"a"}}
    with pytest.raises(ValueError, match="view"):
        family.live_depends_on("all")


@pytest.mark.parametrize("view", sorted(family.LIVE_VIEWS))
def test_live_phases_and_cores_agree_with_the_definitions(view):
    deps = family.live_depends_on(view)
    assert set(deps) == set(family.live()["fields"])
    assert family.live_phases_of(view) == walk_phase(deps)
    assert family.live_cores_of(view) == closure_groups(deps)


def test_the_three_repository_core_is_found_again_in_the_live_prose_mentions():
    # frequent mentions (ten or more): exactly the core of the formal view, and two more groups
    assert family.live_cores_of("frequent") == [
        CORE, ("hypergrammar", "hypermath", "verifier"), ("hyperspace", "hypertime")]
    # every mention: one sixteen-repository core that holds the three
    [big] = family.live_cores_of("mentions")
    assert len(big) == 16 and set(CORE) <= set(big)
    # explicit links alone do not hold hyperphysics in a core
    assert ("hyperethics", "metamathethicology") in family.live_cores_of("links")
    assert not any("hyperphysics" in core for core in family.live_cores_of("links"))


# --- the family's declared open items, and what they are about ------------------------------------------------------


def test_the_declared_open_items_are_the_bracketed_lines_and_the_hm_declarations_of_eleven_fields():
    items = family.open_items()
    assert len(items) == 122 and len({i["field"] for i in items}) == 11
    assert sum(i["form"] == "bracket" for i in items) == 116 and sum(i["form"] == "declaration" for i in items) == 6
    by_field = {}
    for item in items:
        by_field[item["field"]] = by_field.get(item["field"], 0) + 1
    assert by_field["hyperphysics"] == 56 and by_field["hyperreality"] == 34
    assert by_field["hyperethics"] == 3 and by_field["metamathethicology"] == 2 and by_field["hyperlogic"] == 1
    assert all(i["text"] and i["file"] and i["line"] >= 1 for i in items)


def test_the_topic_table_is_applied_to_wording_and_nothing_else():
    assert family.topics_of("a kernel that persists") == {"viability", "persistence"}
    assert family.topics_of("the optimal law, optimising for survival") == {"optimization", "persistence"}
    assert family.topics_of("claude/optimistic-einstein branch") == set()  # a word that only looks like one
    assert family.topics_of("self-maintenance of consistency") == {"persistence"}
    assert family.topics_of("a stable fixed point") == {"stability"}
    assert family.topics_of("the ordinal rank of a limit stage") == {"ordinal"}
    assert family.topics_of("") == set()


def test_by_wording_almost_none_of_the_declared_open_items_is_about_what_this_package_computes():
    tally = family.open_tally()
    assert tally["items"] == 122
    # fixed before the items were read against it, and every hit below was read by eye
    assert tally["topics"] == {"optimization": 0, "viability": 0, "persistence": 2, "stability": 4,
                               "ordinal": 6, "closure": 6}
    assert tally["untouched"] == 106  # no topic at all: 87% of the declared open items
    hits = [(i["field"], i["file"]) for i in family.open_items() if "persistence" in family.topics_of(i["text"])]
    assert hits == [("hyperethics", "L0_creation.hm"), ("hyperphysics", "docs/research/TIME_BUBBLE_SHEET.md")]
    ordinal = {(i["field"], i["file"]) for i in family.open_items() if "ordinal" in family.topics_of(i["text"])}
    assert ("hyperphysics", "docs/research/LAW_RANKING.md") in ordinal  # the family's own ranking of laws
    assert ("metamathethicology", "will_electrophysics.hm") in ordinal  # what its stages mean


def test_every_repository_that_has_a_finite_phase_relies_on_nothing_in_a_core():
    for view in sorted(family.VIEWS):
        deps = family.depends_on(view)
        in_core = {n for core in family.cores_of(view) for n in core}
        for n, p in family.phases_of(view).items():
            if p is not OMEGA:
                assert not (deps[n] & in_core), (view, n)
