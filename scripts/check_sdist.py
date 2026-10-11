"""Check that the source distribution carries every file the validation manifest lists, byte for byte.

A source distribution that leaves out a test helper or a script cannot run its own suite, and one
that changes a byte no longer matches the receipt. This builds one from a copy of the tree (with
`python -m build`, or `uv build` when `build` is not installed), reads it without extracting it,
and compares each listed file's SHA-256 with `validation/source-manifest.json`.

    python scripts/check_sdist.py     # exit 0: all carried; 1: some missing or changed; 3: could not build
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IGNORE = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".coverage",
                                "*.egg-info", "build", "dist")


def build(tree: Path, outdir: Path) -> Path | None:
    commands = ([sys.executable, "-m", "build", "--sdist", "--outdir", str(outdir), str(tree)],
                ["uv", "build", "--sdist", "--out-dir", str(outdir), str(tree)])
    for command in commands:
        if command[0] != sys.executable and shutil.which(command[0]) is None:
            continue
        done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        found = sorted(outdir.glob("*.tar.gz"))
        if done.returncode == 0 and found:
            return found[-1]
    return None


def carried(sdist: Path) -> dict[str, str]:
    """The SHA-256 of every file in the archive, keyed by its path below the top directory."""
    out = {}
    with tarfile.open(sdist) as archive:
        for member in archive.getmembers():
            if member.isfile():
                name = member.name.split("/", 1)[1]
                out[name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    return out


def main() -> int:
    listed = json.loads((ROOT / "validation" / "source-manifest.json").read_text(encoding="utf-8"))["files"]
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "tree"
        shutil.copytree(ROOT, tree, ignore=IGNORE)
        sdist = build(tree, Path(tmp) / "out")
        if sdist is None:
            print("UNKNOWN: neither `python -m build` nor `uv build` produced a source distribution")
            return 3
        have = carried(sdist)
    missing = sorted(set(listed) - set(have))
    changed = sorted(name for name in listed if name in have and have[name] != listed[name])
    print(f"{sdist.name}: {len(have)} files; manifest lists {len(listed)}")
    for label, names in (("MISSING", missing), ("CHANGED", changed)):
        for name in names:
            print(f"  {label} {name}")
    if not (missing or changed):
        print("the source distribution carries every listed file byte for byte")
    return 1 if missing or changed else 0


if __name__ == "__main__":
    sys.exit(main())
