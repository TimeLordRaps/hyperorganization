# The specification the standing oracle was written from

This is the text, verbatim below the rule, that was given to a subagent (the smallest model tier, by the
owner's standing instruction) on 2026-10-11, together with the instruction to read nothing else on the machine
and to use only the standard library. The text below says `oracle_status.py` and "status" because the module was first going to be called `status`;
it became `standing` (the package already has a class `Status`) before the file was copied into the repository.
The subagent returned `oracle_status.py`, which is `scripts/oracle_standing.py` here (its own
two-line header comment replaced; its worked examples moved into the file's `__main__` block) and reported that
the specification did not contradict its own worked examples. Its decisions where the text was silent, in its
words: serial dictatorship takes the goods from each agent's full ranking; `wealth_start` takes its agents from the
wealth mapping; floats are converted exactly by `Fraction` and not refused; `lam` is not range-checked; the loop
test returns False for a single agent.

Two instructions in the text are choices made to keep the reference independent of the package and are recorded
as such: the loop test must try every ordered selection of agents (no depth-first search, colouring or
topological sort), and pecking orders must be found by trying every ordering. The package does the opposite: a
depth-first search for the loop and a peeling of the agents that envy nobody for the order.

It is kept so that "written from a specification alone" can be read, not taken on trust. Agreement between the
oracle and the package shows that implementation slips are caught; both were written from the same definitions,
so it does not show that the definitions are right (`TECHNICAL_DEBT.md`, HO-024).

---

# Specification for an independent reference implementation: "status"

Write ONE Python 3 file, `oracle_status.py`, in THIS directory only. Standard library only (fractions, itertools
are fine). No network. Do NOT read any other file on this machine: write it from this text alone. Finish with a
self-test file `test_oracle_status.py` (plain asserts, runnable with `python3 test_oracle_status.py`) that checks
the worked examples at the end. Prefer the SIMPLEST, most literal code that follows the definitions; speed does
not matter (inputs are at most 6 agents). Do not use floats anywhere: use `fractions.Fraction`.

## Setting

A *market* has n agents and n goods. `endowment` is a dict agent -> good that gives each agent exactly one good
(all goods distinct). `prefs` is a dict agent -> list of ALL the goods, best first (a strict ranking, no ties).
Agents and goods are hashable labels (ints or strings). The order of keys in `endowment` is the *agent order*.

## Functions to implement (exact names and signatures)

1. `envy_edges(endowment, prefs) -> set` : the set of pairs `(i, j)`, i != j, such that agent i ranks the good
   endowed to j STRICTLY above the good endowed to i (i *envies* j).

2. `has_envy_loop(endowment, prefs) -> bool` : True iff there is a sequence of two or more DISTINCT agents
   a_0, a_1, ..., a_{k-1} such that (a_t, a_{t+1 mod k}) is an envy edge for every t. Decide it by TRYING EVERY
   ordered selection of k distinct agents for k = 2..n (itertools.permutations); do NOT use depth-first search,
   colouring, or topological sorting.

3. `pecking_orders(endowment, prefs) -> list` : EVERY ordering (tuple) of the n agents, found by trying all n!
   permutations, such that for each position k, the good endowed to the agent at position k is that agent's
   favourite among the goods endowed to the agents at positions k, k+1, ..., n-1 (itself and everyone after it).
   Return them in the order itertools.permutations(list(endowment)) produces them.

4. `serial_dictatorship(order, prefs) -> dict` : `order` is a tuple of all the agents. In turn, each agent takes
   its favourite good among the goods not yet taken. Returns dict agent -> good, with keys in the order of `order`.

5. `wealth_start(wealth, prefs) -> dict` : `wealth` is a dict agent -> int or Fraction, all values DISTINCT
   (raise ValueError if two agents have equal wealth). The wealthiest agent picks first, and so on down: the
   result of `serial_dictatorship` on the agents sorted by decreasing wealth. Returns dict agent -> good with
   keys in the order of `prefs`.

6. `top_trading_cycles(endowment, prefs) -> dict` : Gale's top trading cycles, written from this definition.
   Repeat while agents remain: every remaining agent POINTS AT the owner (by `endowment`) of the good it ranks
   highest among the goods endowed to the remaining agents (an agent whose own good is its favourite points at
   itself). The pointers form at least one cycle (a self-pointer is a cycle of length one). For every cycle, each
   agent on it receives the good endowed to the agent it points at; all agents on cycles leave. The result is
   dict agent -> good received, keys in the order of `endowment`. (Cycles in one round are disjoint, so the
   order in which they are processed does not matter.)

7. `thaw_ranking(goods, price, weights, lam) -> tuple` : `goods` is a dict good -> sequence of d numbers (int or
   Fraction), the *vector value* of the good; the dict's key order is the *listing order*. `price` is a sequence of d
   numbers; `weights` is ONE agent's sequence of d numbers; `lam` is a Fraction in [0, 1]. The agent's effective
   weights are `(1 - lam) * price[c] + lam * weights[c]` for each coordinate c (exact arithmetic). The agent's
   score of a good is the exact dot product of its effective weights with the good's vector. Return all goods as a
   tuple, best (highest score) first; goods with EQUAL scores are ordered by listing order (earlier first).

8. `thaw_allocation(endowment, goods, price, all_weights, lam) -> dict` : `all_weights` is a dict agent ->
   weights sequence. Build every agent's ranking with `thaw_ranking`, then return `top_trading_cycles` of the
   endowment under those rankings.

9. `movers(endowment, allocation) -> tuple` : the agents (in the order of `endowment`) whose good in `allocation`
   differs from their endowed good.

## Worked examples (your self-test must check these exactly)

A. endowment {"A": "g1", "B": "g2"}; prefs {"A": ["g2","g1"], "B": ["g1","g2"]}:
   envy_edges == {("A","B"), ("B","A")}; has_envy_loop is True; pecking_orders == [];
   top_trading_cycles == {"A": "g2", "B": "g1"}.
B. endowment {"A": "g2", "B": "g1"}; prefs {"A": ["g1","g2"], "B": ["g1","g2"]}:
   envy_edges == {("A","B")}; has_envy_loop is False; pecking_orders == [("B","A")];
   top_trading_cycles == {"A": "g2", "B": "g1"} (nobody moves).
C. serial_dictatorship(("B","A"), {"A": ["g1","g2"], "B": ["g1","g2"]}) == {"B": "g1", "A": "g2"}.
   wealth_start({"A": 1, "B": 5}, {"A": ["g1","g2"], "B": ["g1","g2"]}) == {"A": "g2", "B": "g1"}.
D. The price-blind swap: goods {"y": (1,5), "x": (5,1)} (listing order y, x); price (1,1);
   all_weights {"A": (3,1), "B": (1,3)}; endowment {"A": "y", "B": "x"}.
   At lam = 0 both rank ("y","x") (equal scores 6 and 6, listing order breaks the tie) and
   thaw_allocation == {"A": "y", "B": "x"}. At lam = Fraction(1,2): A's effective weights are (2,1) and ranks
   ("x","y") (scores 11 and 7); B's are (1,2) and ranks ("y","x"); thaw_allocation == {"A": "x", "B": "y"}.
   At lam = 1 the same swap.
E. Three agents, identical rankings ["g1","g2","g3"], endowment {1:"g3", 2:"g1", 3:"g2"}:
   envy_edges == {(1,2), (1,3), (3,2)}; has_envy_loop is False; pecking_orders == [(2,3,1)];
   top_trading_cycles gives everybody their endowed good.

When finished, reply with ONE short paragraph: which file(s) you wrote, whether the self-test passes, and anything
in this specification that was ambiguous or that you had to decide yourself.
