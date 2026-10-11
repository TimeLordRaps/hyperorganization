"""The documents pass the shared conformance workflow's document checks, which are run here first.

`.github/workflows/ci.yml` has a `documents` job: every relative link in a tracked Markdown file must
resolve to a tracked file, and no tracked document may be empty. Its link pattern is deliberately
simple, and so it reads any `[...](target` as a link, including the box modality of provability logic
written `[](  []p -> p )`. That was found by running the job's script by hand on the generated ledger,
after the suite had passed. This file runs the same pattern over the repository's own documents, so
the next such text fails here and not on the hosting platform.
"""

from __future__ import annotations

import posixpath
import re
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[[^\]]*\]\(\s*([^)\s]+)")  # the workflow's own pattern, copied


def documents() -> list[Path]:
    return sorted(ROOT.glob("*.md"))


def dangling(text: str) -> list[str]:
    found = []
    for target in LINK.findall(text):
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
            continue
        path = urllib.parse.unquote(target.split("#", 1)[0]).strip()
        if not path or path.endswith("/"):
            continue
        if not (ROOT / posixpath.normpath(path)).exists():
            found.append(target)
    return found


def test_the_pattern_reads_the_box_modality_as_a_link_and_a_real_link_as_one_that_resolves():
    assert dangling("Löb's axiom [](  []p -> p ) -> []p is proved") == ["[]p"]  # the failure mode
    assert dangling("see [the design](DESIGN.md) and [a site](https://example.org/x)") == []
    assert dangling("see [nothing](NO_SUCH_FILE.md)") == ["NO_SUCH_FILE.md"]


def test_the_repository_has_documents_and_none_is_empty():
    names = {p.name for p in documents()}
    assert {"README.md", "DESIGN.md", "DECISIONS.md", "PROVENANCE.md", "TECHNICAL_DEBT.md"} <= names
    assert [p.name for p in documents() if p.stat().st_size == 0] == []


def test_every_relative_link_in_every_document_resolves():
    problems = {p.name: dangling(p.read_text(encoding="utf-8")) for p in documents()}
    assert {name: found for name, found in problems.items() if found} == {}
