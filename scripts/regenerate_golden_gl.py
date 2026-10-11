"""Regenerate `src/hyperoptimization/data/golden_gl_loops.json` from a hyperlogic checkout.

A ring of checkers A_0 .. A_{n-1} where A_i approves if it can prove, or simply copies, A_{i+1}'s
approval. Read "approves iff it can prove the next approves" as the provability modality of GL
(Goedel-Loeb provability logic), and "approves iff the next approves" as a plain link. Let H be the
conjunction of the n defining equivalences. The statement checked is the RULE form of the loop
theorem,

        if GL proves H then GL proves  A_0 & ... & A_{n-1}

which a decision procedure for theoremhood can test as the single formula  (H & []H) -> (A_0 & ...),
because a theorem H also has its necessitation []H. (The bare formula H -> (A_0 & ...) is not a GL
theorem, even for the guarded ring, and the first version of this script tested that and reported
"not proved" for everything; the difference between H holding at a world and H being provable is the
whole subject.)

Expected, and recorded: PROVED for every ring with at least one link through the box (the loop is
guarded), NOT PROVED with a countermodel for every ring with none (a loop of copying links is
satisfied both by everyone approving and by no one approving).

    PYTHONPATH=<hyperlogic>/src python scripts/regenerate_golden_gl.py <hyperlogic>

The file records one commit's output. Never regenerate to make a failing comparison go away.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

from hyperlogic.provability import box, conj, countermodel, iff, imp, prove, var


def show(f) -> str:
    kind = f[0]
    if kind == "var":
        return f[1]
    if kind == "bot":
        return "F"
    if kind == "not":
        return f"~{show(f[1])}"
    if kind == "box":
        return f"[]{show(f[1])}"
    symbol = {"imp": "->", "and": "&", "or": "|"}[kind]
    return f"({show(f[1])} {symbol} {show(f[2])})"


def ring(n: int, guards: tuple[bool, ...]):
    atoms = [var(f"A{i}") for i in range(n)]
    hypothesis = None
    for i in range(n):
        follows = atoms[(i + 1) % n]
        link = iff(atoms[i], box(follows) if guards[i] else follows)
        hypothesis = link if hypothesis is None else conj(hypothesis, link)
    goal = atoms[0]
    for a in atoms[1:]:
        goal = conj(goal, a)
    return imp(conj(hypothesis, box(hypothesis)), goal)


def record(f) -> dict:
    proved = prove(f)
    model = None if proved else countermodel(f)
    return {
        "formula": show(f),
        "proved": proved,
        "countermodel_nodes": None if model is None else model[0],
    }


def main(checkout: str) -> None:
    commit = subprocess.run(["git", "-C", checkout, "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    p = var("p")
    lob = imp(box(imp(box(p), p)), box(p))
    rings = []
    for n in range(1, 6):
        for guards in itertools.product([True, False], repeat=n):
            if n > 3 and sum(guards) not in (0, 1, n):
                continue  # for larger rings keep none / exactly one / all guarded
            rings.append({"n": n, "guards": list(guards), **record(ring(n, guards))})
    data = {
        "source": "hyperlogic.provability.prove / countermodel, the rule form of the loop theorem",
        "hyperlogic_commit": commit,
        "command": "PYTHONPATH=<hyperlogic>/src python scripts/regenerate_golden_gl.py <hyperlogic>",
        "lob": {"formula": show(lob), "proved": prove(lob)},
        "rings": rings,
    }
    out = (Path(__file__).resolve().parent.parent / "src" / "hyperoptimization" / "data"
           / "golden_gl_loops.json")
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} from hyperlogic {commit[:12]}: {len(rings)} rings")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
