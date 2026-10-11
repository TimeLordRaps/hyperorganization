"""Rewrite validation/source-manifest.json: the SHA-256 of every file a validation receipt is run against.

Run it only after the checks the receipt reports have been rerun on the tree as it now stands.
`tests/test_manifest.py` fails whenever a listed file changes without it, which is the moment to
rerun those checks. The manifest identifies tested bytes. It does not sign or certify them.

    python scripts/regenerate_manifest.py          # write
    python scripts/regenerate_manifest.py --check  # exit 1 if the file on disk is not what would be written
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP = ("L0_viability.hm", "L1_exchange.hm", "pyproject.toml", "DESIGN.md", "DECISIONS.md", "PROVENANCE.md")
MANIFEST = ROOT / "validation" / "source-manifest.json"


def listed() -> list[str]:
    """Every spec, document, source, data, test and script file the receipt covers."""
    package = ROOT / "src" / "hyperoptimization"
    found = set(TOP)
    found.add("src/hyperoptimization/py.typed")
    for pattern, base in (("*.py", package), ("*.json", package / "data"),
                          ("*.py", ROOT / "tests"), ("*.py", ROOT / "scripts")):
        found.update(p.relative_to(ROOT).as_posix() for p in base.glob(pattern))
    return sorted(found)


def build() -> dict[str, object]:
    files = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in listed()}
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {"files": files, "manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def render(manifest: dict[str, object]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    text = render(build())
    if argv == ["--check"]:
        current = MANIFEST.read_text(encoding="utf-8") if MANIFEST.exists() else ""
        print("manifest is current" if current == text else "manifest is STALE")
        return 0 if current == text else 1
    if argv:
        print(__doc__)
        return 2
    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(text, encoding="utf-8", newline="\n")
    manifest = json.loads(text)
    print(f"wrote {MANIFEST.relative_to(ROOT).as_posix()}: {len(manifest['files'])} files, "
          f"digest {manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
