"""Regenerate `src/hyperoptimization/data/golden_family_graph.json` from a hyperstratum checkout.

hyperstratum records, for the repositories it pins, which one cites, depends on or imports which
(`docs/family/family-graph.json`, schema vstd-family-graph/1). The golden file keeps what the
phase computation needs: each repository's id and tier, and each edge with its kind, weight and the
files that are its evidence. It exists so that this package can compute the family's phases WITHOUT
importing or cloning anything: the test suite reads the JSON.

    python scripts/regenerate_golden_family.py /path/to/hyperstratum

The file is a record of one snapshot, not a specification. The family changes; the digest of the
source graph and the commit it was read at are stored so that a stale record is visible, and
regenerating is a deliberate act. Never regenerate to make a failing comparison go away: a difference
is information about the family (or about this package's reading of it).

The source is hyperstratum's static family graph, not the live edges its wiki computes from the pinned
fields; hyperstratum's own HS-002 records that they disagree (the static graph lists 21 repositories
where 24 are pinned). Reading the live edges as well, and comparing, is the obvious next check.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main(checkout: str) -> None:
    root = Path(checkout)
    source = root / "docs" / "family" / "family-graph.json"
    graph = json.loads(source.read_text(encoding="utf-8"))
    commit = subprocess.run(["git", "-C", checkout, "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    data = {
        "source": "hyperstratum docs/family/family-graph.json",
        "schema": graph["schema"],
        "source_digest": graph["digest"],
        "hyperstratum_commit": commit,
        "command": "python scripts/regenerate_golden_family.py <hyperstratum>",
        "nodes": sorted(({"id": n["id"], "tier": n["tier"]} for n in graph["nodes"]),
                        key=lambda n: n["id"]),
        "edges": sorted(({"source": e["source"], "target": e["target"], "kind": e["kind"],
                         "weight": e["weight"], "evidence": sorted(e["evidence"])}
                        for e in graph["edges"]),
                        key=lambda e: (e["source"], e["target"], e["kind"])),
    }
    out = (Path(__file__).resolve().parent.parent / "src" / "hyperoptimization" / "data"
           / "golden_family_graph.json")
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} from hyperstratum {commit[:12]}, graph digest {graph['digest'][:12]}: "
          f"{len(data['nodes'])} repositories, {len(data['edges'])} edges")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
