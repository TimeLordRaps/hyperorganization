"""Ordinal phases below epsilon_0: Cantor normal form, and a convergence that outruns arithmetic.

`viability` gives each state a rank in omega + 1. That is all a finite game can have. But a
descent can be longer than any bound a theory can name, and then the ordinal that certifies it
is bigger. This module is the machinery for that: ordinals below epsilon_0 in Cantor normal
form, and the standard example of a process that provably converges, by a descent through
those ordinals, in a way Peano arithmetic cannot prove.

Cantor normal form
    Every ordinal below epsilon_0 is a finite sum  omega^e_1 c_1 + ... + omega^e_k c_k  with
    exponents e_1 > ... > e_k (themselves ordinals below epsilon_0, so the form nests) and
    positive integer coefficients. `Ordinal` stores exactly that, and compares by comparing
    exponents and then coefficients from the left, the way integers compare by digits.
    Ordinary addition is not commutative (1 + omega = omega, omega + 1 > omega); the natural
    (Hessenberg) sum is, and it is the one that adds valuations.

Goodstein's process (Goodstein 1944; Kirby and Paris 1982)
    Write n in hereditary base b: every number, including every exponent, is itself written in
    base b. Replace b by b + 1 everywhere, subtract 1, and repeat with the base one higher.
    Replacing the base by omega instead turns each term into an ordinal, and the bump-and-
    subtract step strictly lowers it. So the sequence reaches 0, whatever n is. The proof is a
    strictly descending chain below epsilon_0; the number of steps for n = 4 is about
    3 * 2^402653211, and Peano arithmetic cannot prove termination for all n, because epsilon_0
    is its proof-theoretic ordinal. `goodstein_sequence` yields each term with its base and its
    ordinal, `check_descent` checks the chain, and the ordinal of a term evaluates back to it.

Why this is in a package about optimization
    A feedback process that closes on itself needs a reason to stop. The reason is a rank in
    a well-ordered set that every step lowers. For a finite game the rank is a natural number
    or omega; for a process with nested structure it can be any ordinal, and the size of the
    ordinal measures how much induction the proof of convergence needs, not how long the
    process runs. That is the content of calling it an ORDINAL PHASE.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .ground import OMEGA, Omega, Status, Verdict, rank_key


def _cmp(a: Ordinal, b: Ordinal) -> int:
    for (ea, ca), (eb, cb) in zip(a.terms, b.terms):
        order = _cmp(ea, eb)
        if order:
            return order
        if ca != cb:
            return -1 if ca < cb else 1
    return (len(a.terms) > len(b.terms)) - (len(a.terms) < len(b.terms))


@dataclass(frozen=True, slots=True)
class Ordinal:
    """An ordinal below epsilon_0 in Cantor normal form: ((exponent, coefficient), ...)."""

    terms: tuple[tuple[Ordinal, int], ...]

    def __post_init__(self) -> None:
        if type(self.terms) is not tuple:
            raise TypeError("terms must be a tuple of (exponent, coefficient) pairs")
        previous: Ordinal | None = None
        for item in self.terms:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("each term is an (exponent, coefficient) pair")
            exponent, coefficient = item
            if type(exponent) is not Ordinal:
                raise TypeError("an exponent is an Ordinal")
            if type(coefficient) is not int or coefficient < 1:
                raise ValueError("a coefficient is a positive integer")
            if previous is not None and not previous > exponent:
                raise ValueError("exponents must be strictly descending")
            previous = exponent

    # --- construction ---------------------------------------------------------------------

    @classmethod
    def zero(cls) -> Ordinal:
        return ZERO

    @classmethod
    def of(cls, n: int) -> Ordinal:
        """The natural number n as an ordinal."""
        if type(n) is not int:
            raise TypeError(f"a finite ordinal is built from an int, not {n!r}")
        if n < 0:
            raise ValueError("an ordinal below omega is a natural number")
        return cls(((ZERO, n),)) if n else ZERO

    @classmethod
    def omega(cls) -> Ordinal:
        return cls(((ONE, 1),))

    @classmethod
    def omega_power(cls, exponent: Ordinal) -> Ordinal:
        return cls(((exponent, 1),))

    @classmethod
    def from_terms(cls, terms: Iterable[tuple[Ordinal, int]]) -> Ordinal:
        return cls(tuple((e, c) for e, c in terms))

    # --- inspection ---------------------------------------------------------------------------

    @property
    def is_zero(self) -> bool:
        return not self.terms

    @property
    def is_successor(self) -> bool:
        return bool(self.terms) and self.terms[-1][0].is_zero

    @property
    def is_limit(self) -> bool:
        return bool(self.terms) and not self.terms[-1][0].is_zero

    def to_int(self) -> int:
        if not self.terms:
            return 0
        if len(self.terms) == 1 and self.terms[0][0].is_zero:
            return self.terms[0][1]
        raise ValueError(f"{self} is not finite")

    def _max_coefficient(self) -> int:
        best = 0
        for exponent, coefficient in self.terms:
            best = max(best, coefficient, exponent._max_coefficient())
        return best

    def evaluate(self, base: int) -> int:
        """The integer obtained by replacing omega with `base`.

        The base must exceed every coefficient, so that the map is order-preserving and
        injective: it is the inverse of `hereditary_ordinal` on the digits of a number.
        """
        if type(base) is not int or base < 2 or base <= self._max_coefficient():
            raise ValueError(
                f"base must be an int of at least 2 and larger than every coefficient "
                f"(the largest here is {self._max_coefficient()}), not {base!r}"
            )
        return sum(c * base ** e.evaluate(base) for e, c in self.terms)

    # --- order ----------------------------------------------------------------------------------

    def __lt__(self, other: object) -> bool:
        return _cmp(self, other) < 0 if isinstance(other, Ordinal) else NotImplemented

    def __le__(self, other: object) -> bool:
        return _cmp(self, other) <= 0 if isinstance(other, Ordinal) else NotImplemented

    def __gt__(self, other: object) -> bool:
        return _cmp(self, other) > 0 if isinstance(other, Ordinal) else NotImplemented

    def __ge__(self, other: object) -> bool:
        return _cmp(self, other) >= 0 if isinstance(other, Ordinal) else NotImplemented

    # --- arithmetic -----------------------------------------------------------------------------------

    def __add__(self, other: object) -> Ordinal:
        """Ordinary ordinal addition: associative, not commutative, 1 + omega = omega."""
        if not isinstance(other, Ordinal):
            return NotImplemented
        if not other.terms:
            return self
        lead, lead_coefficient = other.terms[0]
        kept: list[tuple[Ordinal, int]] = []
        for exponent, coefficient in self.terms:
            if exponent > lead:
                kept.append((exponent, coefficient))
            elif exponent == lead:
                lead_coefficient += coefficient
        return Ordinal(tuple(kept) + ((lead, lead_coefficient),) + other.terms[1:])

    def __mul__(self, n: object) -> Ordinal:
        """alpha * n for a natural number n: alpha added to itself n times."""
        if type(n) is not int or n < 0:
            return NotImplemented
        if n == 0 or not self.terms:
            return ZERO
        (exponent, coefficient), rest = self.terms[0], self.terms[1:]
        return Ordinal(((exponent, coefficient * n),) + rest)

    def natural_sum(self, other: Ordinal) -> Ordinal:
        """The Hessenberg sum: add coefficients of equal exponents. Commutative; adds valuations."""
        merged: dict[Ordinal, int] = {}
        for exponent, coefficient in self.terms + other.terms:
            merged[exponent] = merged.get(exponent, 0) + coefficient
        return Ordinal(tuple(sorted(merged.items(), key=lambda item: item[0], reverse=True)))

    def successor(self) -> Ordinal:
        return self + ONE

    def predecessor(self) -> Ordinal:
        if not self.is_successor:
            raise ValueError(f"{self} has no predecessor")
        exponent, coefficient = self.terms[-1]
        head = self.terms[:-1]
        return Ordinal(head + (((exponent, coefficient - 1),) if coefficient > 1 else ()))

    def fundamental(self, n: int) -> Ordinal:
        """The n-th term of the standard fundamental sequence of a limit ordinal.

        alpha = delta + omega^e with delta the rest. If e = e' + 1 then omega^e [n] = omega^e' n;
        if e is a limit then omega^e [n] = omega^(e [n]). So omega[n] = n, omega^2[n] = omega n
        and omega^omega[n] = omega^n.
        """
        if type(n) is not int or n < 0:
            raise ValueError("the index of a fundamental sequence is a natural number")
        if not self.is_limit:
            raise ValueError(f"{self} is not a limit ordinal, so it has no fundamental sequence")
        (exponent, coefficient) = self.terms[-1]
        prefix = Ordinal(self.terms[:-1] + (((exponent, coefficient - 1),) if coefficient > 1 else ()))
        if exponent.is_successor:
            tail = Ordinal.omega_power(exponent.predecessor()) * n
        else:
            tail = Ordinal.omega_power(exponent.fundamental(n))
        return prefix + tail

    # --- display ----------------------------------------------------------------------------------------

    def __str__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for exponent, coefficient in self.terms:
            if exponent.is_zero:
                parts.append(str(coefficient))
                continue
            if exponent == ONE:
                shown = "ω"
            else:
                inner = str(exponent)
                shown = "ω^" + (inner if inner.isdigit() or inner == "ω" else f"({inner})")
            parts.append(shown if coefficient == 1 else f"{shown}·{coefficient}")
        return "+".join(parts)

    def __repr__(self) -> str:
        return f"Ordinal({self})"


ZERO = Ordinal(())
ONE = Ordinal(((ZERO, 1),))


def rank_ordinal(rank: int | Omega) -> Ordinal:
    """A viability rank as an ordinal: the natural number k as k, and OMEGA as omega."""
    rank_key(rank)  # refuses anything that is not a rank
    return Ordinal.omega() if rank is OMEGA else Ordinal.of(rank)


# ---------------------------------------------------------------------------
# Goodstein.
# ---------------------------------------------------------------------------


def _digits(n: int, base: int) -> list[tuple[int, int]]:
    """[(exponent, digit)] of n in `base`, highest exponent first, zero digits omitted."""
    out = []
    exponent = 0
    while n:
        n, digit = divmod(n, base)
        if digit:
            out.append((exponent, digit))
        exponent += 1
    out.reverse()
    return out


def _check_number(n: object, base: object) -> None:
    if type(n) is not int or n < 0:
        raise ValueError("n must be a natural number")
    if type(base) is not int or base < 2:
        raise ValueError("the base must be an int of at least 2")


def hereditary_ordinal(n: int, base: int) -> Ordinal:
    """n written in hereditary `base`, with the base replaced by omega."""
    _check_number(n, base)
    return Ordinal(tuple((hereditary_ordinal(e, base), d) for e, d in _digits(n, base)))


def bump(n: int, base: int) -> int:
    """n written in hereditary `base`, with the base replaced by base + 1. No subtraction."""
    _check_number(n, base)
    return sum(d * (base + 1) ** bump(e, base) for e, d in _digits(n, base))


def goodstein_sequence(n: int, *, limit: int | None = None) -> Iterator[tuple[int, int, Ordinal]]:
    """(value, base, ordinal) for each term: n in base 2, then bump-and-subtract with base + 1.

    Ends after the term 0, or after `limit` steps if given. Every value is the ordinal's
    evaluation at its base, and the ordinals strictly descend.
    """
    _check_number(n, 2)
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit is a natural number of steps")
    value, base, steps = n, 2, 0
    while True:
        yield value, base, hereditary_ordinal(value, base)
        if value == 0 or (limit is not None and steps >= limit):
            return
        value = bump(value, base) - 1
        base += 1
        steps += 1


def check_descent(chain: Iterable[Ordinal]) -> Verdict:
    """Whether every term of the chain is strictly smaller than the one before it."""
    items = list(chain)
    for index in range(1, len(items)):
        if not items[index - 1] > items[index]:
            return Verdict(
                "the ordinals strictly descend", Status.FAILS_IN_MODEL,
                (f"step {index}: {items[index - 1]} !> {items[index]}",),
            )
    return Verdict("the ordinals strictly descend", Status.HOLDS_IN_MODEL,
                   evidence=(f"{len(items)} terms",))


__all__ = [
    "ONE", "Ordinal", "ZERO", "bump", "check_descent", "goodstein_sequence",
    "hereditary_ordinal", "rank_ordinal",
]
