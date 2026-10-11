"""Regenerate `src/hyperoptimization/data/golden_l2_gate_bits.json` from a hyperethics checkout.

The golden file records what `hyperethics.capture.gate_cost` says for a handful of the
governance systems hyperethics ships, with each party's price read as bits of surprisal
(price p means the party is compromised with probability 2**-p). It exists so that this
package can check its own cover search against hyperethics WITHOUT importing it: the test
suite reads the JSON and never needs hyperethics installed.

    PYTHONPATH=/path/to/hyperethics/src python scripts/regenerate_golden.py /path/to/hyperethics

The file is a record of one commit's output, not a specification. If hyperethics changes
what it computes, the golden file goes stale and the test that reads it keeps passing
against the old numbers; that is why the commit hash is stored and why regenerating is a
deliberate act. Never regenerate to make a failing comparison go away: a difference is
information about one of the two packages.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

from hyperethics import constitutions
from hyperethics.capture import gate_cost
from hyperethics.governance import gate_need

SYSTEMS = ("two_key", "ring_of_four", "separated_powers", "common_principal", "client_state",
           "suzerain", "hidden_only", "shadow_court")


def number(value):
    return "inf" if value == math.inf else value


def record(system):
    return {
        "name": system.name,
        "parties": {p.name: number(p.price) for p in system.parties},
        "controls": sorted([list(edge) for edge in system.controls]),
        "gates": {
            w.name: {
                "need": sorted(gate_need(system, w.name)),
                "gate_cost": number(gate_cost(system, w.name)),
            }
            for w in system.powers
        },
    }


def main(checkout: str) -> None:
    commit = subprocess.run(["git", "-C", checkout, "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    data = {
        "source": "hyperethics.capture.gate_cost with prices read as bits",
        "hyperethics_commit": commit,
        "command": "PYTHONPATH=<hyperethics>/src python scripts/regenerate_golden.py <hyperethics>",
        "systems": [record(getattr(constitutions, name)()) for name in SYSTEMS],
    }
    out = (Path(__file__).resolve().parent.parent / "src" / "hyperoptimization" / "data"
           / "golden_l2_gate_bits.json")
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} from hyperethics {commit[:12]}: {len(data['systems'])} systems")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
