"""The epistemic contract of this package, and the one non-natural rank.

Nothing here is a model of anything. It fixes what the other modules are allowed to
say about what they computed:

- A claim that HOLDS_IN_MODEL has been checked against one finite structure, one
  parameter set or one grid. That is not a proof of anything beyond what was checked,
  and it must never be reported as one. A success has no distinguished witness (an
  existential witness is something that was constructed, and a success constructs none),
  but it may record `evidence`: the numbers that were read, so that someone can see what
  was measured and reproduce it.
- A claim that FAILS_IN_MODEL must name its counterexample. One counterexample settles a
  universal claim.
- `Verdict` has no truth value. `if verdict:` raises, because the interesting distinction
  is between "checked here" and "established", and a bool erases it. This is the same
  contract `hyperethics.ground.Verdict` keeps, extended with `evidence`.

The four tags are the family's. `FORM` is checked exactly (a finite structure, rational
arithmetic, or a proof short enough to be read). `FRAME` holds inside a stated model whose
assumptions are listed. `HYPER` is a proposal that nothing here witnesses. `OPEN` is
unresolved, and a contradiction found is an `OPEN` with its exact conflict.

`OMEGA` is the rank of a state that survives every finite stage of a descending
fixed-point iteration, the first infinite ordinal. It is a singleton that is not an
`int`, so a rank can never be mistaken for a number of steps, and it sorts above all of
them through `rank_key`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Status(str, Enum):
    """Whether a claim was observed to hold in the structure that was checked."""

    HOLDS_IN_MODEL = "HOLDS_IN_MODEL"
    FAILS_IN_MODEL = "FAILS_IN_MODEL"


class Tag(str, Enum):
    """How far a claim reaches. The family's four, in the order it ranks them."""

    FORM = "FORM"
    FRAME = "FRAME"
    HYPER = "HYPER"
    OPEN = "OPEN"


def _name(value: object, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _strings(value: object, label: str) -> None:
    if type(value) is not tuple or any(type(item) is not str for item in value):
        raise TypeError(f"{label} must be a tuple of strings")


@dataclass(frozen=True, slots=True)
class Verdict:
    """The outcome of checking one claim against one finite structure or parameter set.

    `witness` is the counterexample of a failure and is empty for a success.
    `evidence` is what was measured: for a success it says what the check read, so that
    a reader can see how narrow it was. It is never a proof.
    """

    claim: str
    status: Status
    witness: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        _name(self.claim, "claim")
        if type(self.status) is not Status:
            raise TypeError("status must be a Status")
        _strings(self.witness, "witness")
        _strings(self.evidence, "evidence")
        if self.status is Status.HOLDS_IN_MODEL and self.witness:
            raise ValueError("a claim holding in a model has no distinguished witness")
        if self.status is Status.FAILS_IN_MODEL and not self.witness:
            raise ValueError("a failure must name its counterexample")

    def __bool__(self) -> bool:
        raise TypeError(
            "Verdict has no truth value: compare .status explicitly. "
            "HOLDS_IN_MODEL is not validity, and treating it as True erases that."
        )


class Omega:
    """The first infinite ordinal as a rank: survives every finite stage. A singleton."""

    __slots__ = ()
    _instance: Omega | None = None

    def __new__(cls) -> Omega:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ω"

    __str__ = __repr__


OMEGA = Omega()


def rank_key(rank: int | Omega) -> tuple[int, int]:
    """A sort key putting every natural number below `OMEGA`.

    A rank is a natural number (the stage at which a state was eliminated) or `OMEGA`
    (never eliminated). Anything else, including a bool, a float or a negative number,
    is refused rather than ordered.
    """
    if rank is OMEGA:
        return (1, 0)
    if type(rank) is not int or rank < 0:
        raise TypeError(f"a rank is a natural number or OMEGA, not {rank!r}")
    return (0, rank)
