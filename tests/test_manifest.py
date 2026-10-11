"""The recorded source manifest describes the tree it sits in.

`validation/source-manifest.json` binds the bytes a validation receipt was run against. A receipt
that cites a digest nothing reproduces is worse than no receipt, so change a listed file and this
file fails until the manifest is regenerated (`scripts/regenerate_manifest.py`), which is the
moment to rerun the checks the receipt reports.

The manifest identifies tested bytes. It does not sign or certify them. The rule for which files
are listed is restated here on purpose, not imported from the script: if the script's rule drifts
from this one, the third test fails.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "validation" / "source-manifest.json").read_text(encoding="utf-8"))
PACKAGE = ROOT / "src" / "hyperoptimization"


def digest(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_the_manifest_digest_is_the_digest_of_its_canonical_files_mapping():
    canonical = json.dumps(MANIFEST["files"], sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == MANIFEST["manifest_sha256"]


def test_every_listed_file_hashes_to_the_value_recorded_for_it():
    stale = [name for name, recorded in MANIFEST["files"].items() if digest(name) != recorded]
    assert not stale, f"changed since the manifest was written: {stale}; rerun the checks, then regenerate"


def test_every_spec_document_source_data_test_and_script_file_is_listed():
    expected = {"L0_viability.hm", "pyproject.toml", "DESIGN.md", "DECISIONS.md", "PROVENANCE.md",
                "src/hyperoptimization/py.typed"}
    expected |= {f"src/hyperoptimization/{p.name}" for p in PACKAGE.iterdir() if p.suffix == ".py"}
    expected |= {f"src/hyperoptimization/data/{p.name}" for p in (PACKAGE / "data").iterdir()
                 if p.suffix == ".json"}
    expected |= {f"tests/{p.name}" for p in (ROOT / "tests").iterdir() if p.suffix == ".py"}
    expected |= {f"scripts/{p.name}" for p in (ROOT / "scripts").iterdir() if p.suffix == ".py"}
    assert set(MANIFEST["files"]) == expected


def test_the_regeneration_script_would_write_exactly_the_file_on_disk():
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "regenerate_manifest.py"), "--check"],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stdout + done.stderr
