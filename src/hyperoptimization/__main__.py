"""python -m hyperoptimization [decisions]"""

from __future__ import annotations

import sys

from . import decisions, report

USAGE = (
    "usage: python -m hyperoptimization [decisions]\n"
    "  (no argument)  print the report\n"
    "  decisions      print DECISIONS.md exactly as the ledger renders it"
)


def _utf8_streams() -> None:
    """The report is Unicode (ω); a Windows pipe defaults to a code page that cannot hold it."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        sys.stdout.write(report.build_report())
        return 0
    if args == ["decisions"]:
        sys.stdout.write(decisions.render_markdown())
        return 0
    print(USAGE)
    return 2


if __name__ == "__main__":
    _utf8_streams()
    sys.exit(main())
