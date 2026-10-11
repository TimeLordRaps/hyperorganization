"""Information as the thing that is exchanged: what a coalition can lawfully permit is what its
pooled knowledge supports itself on.

A participant sees part of the game and must treat the rest as unsafe (`viability.kernel_within`:
ignorance is pessimism). What it can then certify is a part of the greatest law, and it grows as the
participant learns. That one rule gives a theory of deals in which nothing is paid:

    soundness     nothing a view certifies lies outside the greatest sustaining law, so a deal made from
                  certificates can only open freedoms the system can bear (coherence is built in);
    monotone      knowing more never certifies less, so disclosing never costs the discloser a freedom
                  it already could certify;
    exactness     a view closed under the moves it can make certifies exactly, on what it holds;
    coinduction   a permission is certified exactly when the view holds a set that supports itself
                  around it, so the knowledge a permission needs is a self-supporting set, a finite
                  object a verifier can check without trusting the prover (`minimal_knowledge`).

Pool the knowledge of parties that each hold a territory and two things follow. What any party could
certify alone is still certified (`pooled`), and sometimes strictly more is: a freedom whose safety
depends on a state the party cannot see. That surplus exists only where the territories touch
(`crossings`): with no allowed move from one territory into another each party's view is already
closed and pooling adds nothing. Where it exists it can be mutual (two hamlets that each shelter in
the other in a storm: neither can certify its own safe move alone, together they certify both) or
one-sided (one hamlet is safe on its own, the other shelters in it: the first gives knowledge,
gains nothing and loses nothing). This is the sense in which information asymmetry makes a deal
necessary where it exists and not merely possible, and in which what is exchanged is a certificate and not a price:
a number cannot be checked for safety, a self-supporting set can.

What this does not show: that disclosure is free. Knowledge is non-rival here (giving it away leaves
the giver's view unchanged), so a deal is weakly win-win by construction; the strategic value of
private information, rent, and rival freedoms (a permission to use something that runs out) are
outside the model. "Win" means a permission nobody had been able to certify, not a payoff.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from . import viability
from .viability import Game

Permission = tuple[str, str]  # (state, action)


def permissions(law: Mapping[str, Iterable[str]]) -> frozenset[Permission]:
    """A law as the set of its permissions."""
    return frozenset((s, a) for s, permitted in law.items() for a in permitted)


def certified(game: Game, known: Iterable[str], *, corrigible: bool = True) -> frozenset[Permission]:
    """The permissions a participant who sees only `known` can certify."""
    return permissions(viability.greatest_law_within(game, known, corrigible=corrigible))


def _knowledge(game: Game, knowledge: Mapping[str, Iterable[str]]) -> dict[str, frozenset[str]]:
    if not isinstance(knowledge, Mapping):
        raise TypeError("knowledge maps each party to the states it can see")
    held: dict[str, frozenset[str]] = {}
    for party, states in knowledge.items():
        if type(party) is not str or not party:
            raise ValueError("every party needs a name")
        if isinstance(states, str):
            raise TypeError(f"what {party!r} knows is a collection of state names, not one string")
        held[party] = frozenset(states)
        for state in held[party]:
            if state not in game.states:
                raise ValueError(f"{party!r} knows {state!r}, which is not a state of this game")
    return held


def pooled(game: Game, knowledge: Mapping[str, Iterable[str]], parties: Iterable[str] | None = None,
           *, corrigible: bool = True) -> frozenset[Permission]:
    """What the named parties (all of them by default) can certify once their knowledge is pooled."""
    held = _knowledge(game, knowledge)
    chosen = list(held) if parties is None else list(parties)
    for party in chosen:
        if party not in held:
            raise ValueError(f"{party!r} is not a party with knowledge")
    return certified(game, frozenset().union(*(held[p] for p in chosen)), corrigible=corrigible)


def _owners(game: Game, owner: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(owner, Mapping):
        raise TypeError("owner maps each state to the party that acts there")
    if set(owner) != set(game.states):
        raise ValueError("owner must name every state of the game, and only those")
    for state, party in owner.items():
        if type(party) is not str or not party:
            raise ValueError(f"the owner of {state!r} must be a nonempty string")
    return dict(owner)


def crossings(game: Game, owner: Mapping[str, str]) -> frozenset[tuple[str, str, str]]:
    """Every allowed move (state, action, successor) that leaves the party's territory for another's."""
    who = _owners(game, owner)
    return frozenset(
        (s, a, game.target(s, a, b))
        for s in game.states for a in game.allowed(s) for b in game.nature
        if who[game.target(s, a, b)] != who[s]
    )


@dataclass(frozen=True, slots=True)
class Surplus:
    """What pooling does for each party.

    alone     the permissions each party can certify from what it knows by itself;
    together  the permissions the parties can certify from all they know;
    gains     for each party, the permissions at states it owns that only the pooled knowledge unlocks.
    """

    alone: dict[str, frozenset[Permission]]
    together: frozenset[Permission]
    gains: dict[str, frozenset[Permission]]
    owners: frozenset[str]

    @property
    def mutual(self) -> bool:
        """Every party that owns a state gains, and there are at least two of them."""
        return len(self.owners) >= 2 and all(self.gains[p] for p in self.owners)


def surplus(game: Game, knowledge: Mapping[str, Iterable[str]], owner: Mapping[str, str],
            *, corrigible: bool = True) -> Surplus:
    """The gains from pooling when each party sees what it knows and acts at the states it owns."""
    held = _knowledge(game, knowledge)
    who = _owners(game, owner)
    for party in set(who.values()):
        if party not in held:
            raise ValueError(f"{party!r} owns a state but has no knowledge entry")
    alone = {p: certified(game, states, corrigible=corrigible) for p, states in held.items()}
    together = certified(game, frozenset().union(*held.values()), corrigible=corrigible)
    gains = {p: frozenset(perm for perm in together if who[perm[0]] == p) - alone[p] for p in held}
    return Surplus(alone, together, gains, frozenset(who.values()))


def minimal_knowledge_all(game: Game, permission: Permission, *, corrigible: bool = True
                          ) -> list[frozenset[str]]:
    """Every smallest set of states whose knowledge certifies `permission`, in a fixed order.

    Empty if the permission is not in the greatest law: a freedom that ends existence is not
    certifiable by anyone, however much they know. Exhaustive over subsets, so for games of a handful
    of states (HO-006).
    """
    state = permission[0]
    if state not in game.states:
        raise ValueError(f"{state!r} is not a state of this game")
    others = sorted(set(game.states) - {state})
    for extra in range(len(others) + 1):
        found = [frozenset({state, *chosen}) for chosen in itertools.combinations(others, extra)
                 if permission in certified(game, {state, *chosen}, corrigible=corrigible)]
        if found:
            return sorted(found, key=sorted)
    return []  # even knowing everything does not certify it: it is outside the greatest law


def minimal_knowledge(game: Game, permission: Permission, *, corrigible: bool = True
                      ) -> frozenset[str] | None:
    """A smallest set of states whose knowledge certifies `permission`; None if nothing can."""
    found = minimal_knowledge_all(game, permission, corrigible=corrigible)
    return found[0] if found else None


__all__ = ["Permission", "Surplus", "certified", "crossings", "minimal_knowledge",
           "minimal_knowledge_all", "permissions", "pooled", "surplus"]
