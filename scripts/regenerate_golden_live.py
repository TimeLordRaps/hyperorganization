"""Regenerate `src/hyperoptimization/data/golden_family_live.json` from a hyperstratum checkout.

hyperstratum scans every pinned field and records, without interpreting it, which fields it links to or
names in prose (the live edges the wiki is built from) and every line that carries an `[OPEN]` tag.
This script runs that scan and keeps the result, together with the `open` declarations of the `.hm`
specifications (which use a different convention, not the bracket tag), so the package can read the
family's declared open items and a second graph of reliance WITHOUT importing hyperstratum: the test
suite reads the JSON.

    PYTHONPATH=/path/to/hyperstratum/src python scripts/regenerate_golden_live.py /path/to/hyperstratum

The scan is a function of the pinned commits alone, so the record names the hyperstratum commit it was
read at. It is a snapshot, not a specification: never regenerate to make a failing comparison go away.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from hyperstratum import registry, scan

HM_OPEN = re.compile(r"^open ([\w-]+):\s*$")
BLOCK_LIMIT = 600


def hm_open_declarations(field_path: Path) -> list[list[object]]:
    """[file, line, name, text] for each `open name:` in a .hm file; text is its comment block."""
    found: list[list[object]] = []
    for path in scan.iter_files(field_path):
        if path.suffix != ".hm":
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for number, line in enumerate(lines, 1):
            match = HM_OPEN.match(line)
            if not match:
                continue
            block: list[str] = []
            for follower in lines[number:]:
                if not follower.strip():
                    break
                block.append(follower.strip().removeprefix("--").strip())
            found.append([path.relative_to(field_path).as_posix(), number, match.group(1),
                          " ".join(block)[:BLOCK_LIMIT]])
    return sorted(found)


def main(checkout: str) -> None:
    root = Path(checkout)
    commit = subprocess.run(["git", "-C", checkout, "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    reg = registry.load_registry(root)
    names = scan.sibling_names(reg)
    scans, fields, unscanned = {}, {}, []
    for field in reg.fields:
        if not field.path.is_dir() or not any(field.path.iterdir()):
            unscanned.append(field.name)
            continue
        result = scan.scan_field(field, [], names)
        scans[field.name] = result
        fields[field.name] = {
            "pin": field.pin,
            "tags": result.tags,
            "open_items": [list(item) for item in result.open_items],
            "hm_open": hm_open_declarations(field.path),
        }
    edges = [{"source": e.source, "target": e.target, "kind": e.kind, "count": e.count}
             for e in scan.build_edges(scans, reg)]
    data = {
        "source": "hyperstratum.scan.scan_field (no terms) and build_edges over the pinned fields, "
                  "plus the open declarations of their .hm files",
        "command": "PYTHONPATH=<hyperstratum>/src python scripts/regenerate_golden_live.py <hyperstratum>",
        "hyperstratum_commit": commit,
        "unscanned": sorted(unscanned),
        "fields": dict(sorted(fields.items())),
        "edges": sorted(edges, key=lambda e: (e["source"], e["target"], e["kind"])),
    }
    out = (Path(__file__).resolve().parent.parent / "src" / "hyperoptimization" / "data"
           / "golden_family_live.json")
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    bracket = sum(len(v["open_items"]) for v in fields.values())
    declared = sum(len(v["hm_open"]) for v in fields.values())
    print(f"wrote {out} from hyperstratum {commit[:12]}: {len(fields)} fields scanned "
          f"({len(unscanned)} not), {len(edges)} live edges, {bracket} [OPEN] lines, "
          f"{declared} .hm open declarations")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
