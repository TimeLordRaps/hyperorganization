"""python -m hyperoptimization [decisions]"""

from __future__ import annotations

import sys

from . import decisions, report

USAGE = (
    "usage: python -m hyperoptimization [decisions]\n"
    "  (no argument)  print the report\n"
    "  decisions      print DECISIONS.md exactly as the ledger renders it"
)


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
    sys.exit(main())
