# Design

**Status, 2026-10-11.** Version 0.1.0.dev0. Tags are the family's: `[FORM]` checked exactly (a finite structure, rational arithmetic, or a proof short enough to read here), `[FRAME]` holds inside stated assumptions, `[HYPER]` a proposal nothing here witnesses, `[OPEN]` unresolved. References to the literature are recalled from memory and were not re-fetched while writing this; treat each as a pointer to look up, not a verified quotation.

## 0. What this is

A toolkit of small finite models for one question: when does a system that wants to persist, and is free to act, and can be corrected, and can renew itself, end up governed by a *single optimal law*, and what does that law look like from the inside? It has no money anywhere. Every price is a number of bits.

It answers your three questions as follows (the starting point is quoted in full in `PROVENANCE.md`):

| you asked | where |
|---|---|
| how ordinal laws occur in a centrally bound, mechanically governed system | §8 (five orders), §9 |
| what theorems can be written about it | §3 to §9: each numbered result, with its proof or its oracle |
| what axioms hold by nature of the optimality structure | §10 |
| which laws fall out of free will, existence, corrigibility and renewal, without money | §3, §6, §7 |
| what decisions I made that carry evidential expectations | `DECISIONS.md` (generated from `src/hyperoptimization/decisions.py`) |
| what got in the way | §11 |
| where this meets the family's own open problems | §3 (R6), §8 (E4), §9 (F1, F2) |

## 1. Notation

Every quantity is dimensionless unless a unit is given.

| symbol | meaning | domain | unit |
|---|---|---|---|
| m | seats in a body | integer ≥ 1 | count |
| q | seats whose compromise captures the body | integer, 1 ≤ q ≤ m | count |
| θ | q / m | (0, 1) | none |
| x, p, π | probabilities (a seat's, a body's, a direct attack's) | [0, 1] | none |
| T(x) | P(Bin(m, x) ≥ q) = Σ_{j ≥ q} C(m, j) x^j (1−x)^(m−j) | [0, 1] | none |
| T\* | the dual quorum map T_{m, m−q+1} | | none |
| x_c | the fixed point of T strictly inside (0, 1), when there is one | (0, 1) | none |
| λ, ν | attack and repair rates per seat | ≥ 0 | 1 / time |
| ρ | λ / (λ + ν) | [0, 1] | none |
| D(θ‖ρ) | θ ln(θ/ρ) + (1−θ) ln((1−θ)/(1−ρ)), Kullback–Leibler divergence of two Bernoulli laws | ≥ 0 | nats per seat |
| bits(p) | −log₂ p | [0, ∞] | bits, a logarithmic unit with no physical dimension |
| S, K, X | a game's states, safe set and viability kernel | | |
| L\* | the greatest sustaining law | | |
| rank | a natural number, ω, or an ordinal below ε₀ | | ordinal |

Abbreviations, expanded where first used: GL is Gödel–Löb provability logic; RG is the renormalization group; CNF is Cantor normal form; PA is Peano arithmetic; L2 is hyperethics layer 2 (the rule that no power may be corrupting); GC is a graduation criterion.

## 2. The idea in one paragraph

A system *exists* if it can keep itself inside a safe set for ever against whatever the environment does. Given that, and given that a law can limit what is *permitted* but not what is *chosen* among the permitted, there is exactly one law that keeps the system existing while taking away no freedom that existence does not require. It forbids precisely the actions that leave the set of states from which the system can survive. It is unique, it mentions no participant, and nothing above it grants it validity: it is derived, not authorized. Bodies that apply it are quorums of seats; a loop of bodies that oversee one another collapses discontinuously above a critical attack pressure; repair must outpace attack by the quorum margin or no number of seats helps; and the question of who may change the rule has an arithmetic of its own, in which a constitution that has no sovereign has no finite rank for validity. Everything above is checked on finite models. What it does not show is that persistence is the right thing to optimize (§11, B1).

## 3. Existence, freedom, corrigibility: the viability game

**Setting.** A finite set of states S, actions allowed at each state, and nature's choices. The system picks an action, nature picks a choice, the state moves. A safe set K ⊆ S is where the system exists. (Viability theory, Aubin.)

**Kernel.** X is the greatest fixed point of "in K, and some allowed action sends every nature choice back into X". Computed from above: V₀ = K, V_{k+1} = K ∩ {s : some action keeps every successor in V_k}.

**R1 (ranks are certificates) `[FORM]`.** Let rank(s) be the first k with s ∉ V_k, and ω if none. Then rank is the unique solution of

    rank(s) = 0                                     if s ∉ K,
    rank(s) = 1 + max_a min_b rank(step(s, a, b))   if s ∈ K        (1 + ω = ω).

*Proof.* Put A_j = {s : rank'(s) > j} for a solution rank'. A_0 = K = V_0. If A_j = V_j, then s ∈ A_{j+1} iff s ∈ K and 1 + max_a min_b rank'(next) > j + 1, iff s ∈ K and some action sends every nature choice into A_j = V_j, iff s ∈ V_{j+1}. So A_j = V_j for every j, and rank' = rank. ∎ A verifier holding only the game and a claimed ranking checks it in one pass and never trusts the prover; `check_ranks` does, and every single-entry tampering is caught (thousands of them, in the tests and again in the ledger, D08).

**R2 (the greatest sustaining law) `[FORM]`.** A *law* assigns each state a set of permitted actions. It is *sustaining from s* if, however the permitted actions are chosen and whatever nature does, the system never leaves K and never reaches a state where nothing is permitted. Let L\*(s) = {a allowed at s : every nature choice leads into X} for s ∈ X. Then a law is sustaining from s iff everywhere it reaches from s it permits only actions of L\*, and what it reaches lies in X.

*Proof.* (⇒) The set R the law reaches from s is safe, closed under every permitted action and every nature choice, with nothing stuck in it. A set with those properties lies inside the greatest fixed point, so R ⊆ X; each permitted action keeps its successors in R ⊆ X, so it belongs to L\*. (⇐) The law keeps successors in X, where L\* is never empty, and X ⊆ K. ∎ So L\* is itself sustaining from every state of X, and it contains every sustaining law state by state: the unique pointwise-greatest sustaining law, the *least restriction that preserves existence*. Checked by enumerating **every** law and **every** start on random games and against brute-force reachability (`test_a_law_sustains_existence_exactly_when_it_permits_nothing_outside_the_greatest_law`). It is invariant under renaming states, actions and nature choices, and under automorphisms of the game, so it cannot favour a participant the game does not already single out.

This is the sense in which your "not dictatorially" is formal: the optimum is canonical, a function of the game and nothing else. Where it permits exactly one action the law is an *obligation* (`forced_moves`): a duty to repair is a prohibition on every alternative.

**R3 (exemption is fatal) `[FORM]`, derives hyperethics' `exempt-reach`.** Take free will to mean *full support*: every permitted action is taken with positive probability at every step, and so is every nature choice. Then (a) the survival probability from s is 1 iff the law is sustaining; (b) otherwise it is strictly below 1; (c) it is exactly 0 if every state the law reaches can reach an unsafe one.

*Proof of (b).* A reachable state t outside X has finite rank r. From a state of rank r ≥ 1, whichever action is taken some nature choice leads to a state of strictly smaller rank, so a positive-probability path of length ≤ r reaches an unsafe state. ∎ (a) and (c) follow. Hence one permission outside L\* at a state the system reaches — an author exempt from a prohibition that protects existence — makes survival uncertain. That is hyperethics' exempt-reach clause, **derived** here instead of adopted. Exact rational arithmetic; bounds checked by an independent step-by-step distribution (`stepwise_bounds`).

**R4 (corrigibility) `[FRAME]`.** Add a *home* set and overseer *correction moves*; `reach_home` is the set from which the overseer can bring the system home. Requiring the system to stay inside it is the same game on a smaller safe set, so R1 to R3 hold again with the *corrigible* kernel. A move that leaves the corrigible region is *irreversible*: afterwards nobody can correct the system, though it may survive for ever. The corrigible kernel is inside the kernel, strictly so when survival can be bought by passing through a point of no return; the price of corrigibility is exactly the difference.

**R5 (convergence from many laws) `[FRAME]`.** `settle(law)` = prune what the law cannot survive on, then permit every action that stays inside what survives. It never lies above L\*, is monotone and idempotent, and settling the fully permissive law gives L\* in one step. But greatest fixed points are found from above: a start that mentions only one of two separate safe regions is stuck below the optimum (`twin_rooms`). What converges is the *diversity of starts*: pooling the settled laws of random starts reached L\* in every sampled game that has a kernel, within a few starts (D20 gives the counts).

**R6 (self-maintenance is coinductive) `[FORM]`.** Let F(Y) be the states of the region from which some allowed action sends every nature choice into Y. F is monotone, so it has a greatest and a least fixed point; the greatest is the kernel (R1). Call Y *self-supporting* when Y ⊆ F(Y). *Coinduction:* the kernel is the union of every self-supporting set (Knaster–Tarski), so to certify that a state exists for ever it is enough to exhibit one self-supporting set that holds it, a finite object a verifier checks in one pass without trusting the prover (the rank of R1 is its quantitative form). *No induction:* the least fixed point, built up from ∅, is ∅ for every game, because F(∅) = ∅ whenever nature has a choice: nothing is supported by nothing. Maintenance has no base case to be built up from. Checked against every subset of every sampled game, with and without corrigibility (`test_the_kernel_is_the_union_of_every_set_that_supports_itself`, D21).

*Reading for hyperethics `[HYPER]`.* The principal open problem of hyperethics (`L0_creation.hm`, `nd-consistency-self-maintenance`, Tyler Roost's phrase, the Universal Consistency Self-Maintenance Paradox) is that a realm's self-maintenance of consistency appears to presuppose the very consistency being maintained, and under immanence the maintainer is itself inside the realm. R6 says what kind of circle that is if consistency is read as a safe condition and the maintainer's own integrity is a state of the game: a self-supporting set, which is benign (coinductive), not vicious (an inductive reading would be empty). The sharpening by immanence is R3, since the maintainer cannot be exempt and so its moves lie inside the greatest law. What is left is whether the realm's kernel is non-empty, which is a fact about the realm, and whether consistency is a safe condition at all and not something a theory proves (a consistent theory cannot prove its own consistency, and a loop of provers is determined without being sound, E3). This is a reading, not a resolution (`nd-consistency-self-maintenance-resolved`).

## 4. Quorum bodies and the council of councils

A body of m seats, each compromised independently with probability x, is captured when at least q are: T(x) above. If the seats of the next body up are bodies of this kind, the compromise probability one level up is T(x), and the levels are iterates of T: a one-dimensional renormalization-group (RG) flow.

**Q1 `[FORM]`.** T has exactly one fixed point in (0, 1) iff 2 ≤ q ≤ m−1; it is unstable; 0 and 1 attract. *Proof.* T′(x) = m C(m−1, q−1) x^(q−1) (1−x)^(m−q) > 0, and T″ has the sign of (q−1)(1−x) − (m−q)x, positive before x\* = (q−1)/(m−1) and negative after. f = T − x has f(0) = f(1) = 0 and f′(0) = f′(1) = −1 (for 2 ≤ q ≤ m−1). On [0, x\*] f is convex and starts downward, so it crosses zero at most once more; on [x\*, 1] it is strictly concave and ends with f > 0 just left of 1, so it has at most one zero. If there is a zero z ≤ x\*, then f > 0 on (z, x\*], and the chord from (x\*, f(x\*)) to (1, 0) keeps f > 0 on [x\*, 1): exactly one zero. Otherwise the single zero lies in (x\*, 1). In both cases f is negative before the zero and positive after. For q = 1 or q = m, f has constant sign inside. ∎ Also counted exactly by Sturm's theorem in rationals for every (m, q) up to m = 22 and checked against an independent float grid.

**Q2 (duality) `[FORM]`.** T_{m,q}(x) + T_{m,m−q+1}(1−x) = 1. *Proof.* P(Bin(m, x) ≥ q) = P(Bin(m, 1−x) ≤ m−q) = 1 − P(Bin(m, 1−x) ≥ m−q+1). ∎ The odd majority q = (m+1)/2 is its own dual, so T(½) = ½ exactly: it is critical at exactly one half, where a council costs exactly **one bit**, and at the critical point a council is exactly as easy to capture as a single seat (T(x_c) = x_c). Lowering the quorum moves x_c down, raising it moves it up. This is the sense in which a balanced body — capture and blocking equally hard — has a unique symmetric critical point; the *choice* of balance as the optimality criterion is adopted, not derived (D11).

**Q3 `[FORM]`.** T(x) ≤ C(m, q) x^q (union bound over q-subsets), so below x_c compromise falls doubly exponentially with depth.

**Q4 `[FRAME]`, numerical.** Near the critical point T′(x_c) ≈ A(θ)√m with A(θ) = φ(Φ⁻¹(θ)) / √(θ(1−θ)), φ and Φ the standard normal density and distribution (the de Moivre–Laplace limit read at the critical point); A(½) = √(2/π). Checked at m = 101, 401, 1601; the error is O(1/√m).

## 5. Loops of checks

L2 shows a finite system with no unchecked apex must close its checks into a loop. Mean-field model: every body is compromised with the same probability p, and is overseen by a quorum body of m seats captured when q are. With π the probability of a direct attack, two kinds of link:

* **COMMAND**: the overseers can command the body, so compromise propagates: p = π + (1−π) T(p). p = 1 is always a fixed point.
* **VETO**: the overseers can only block it, so a direct attack succeeds only if they fail too: p = π T(p). p = 0 is always a fixed point.

**L1 `[FORM]`, one equation in dual variables.** Put s = 1 − p. Then 1 − G_command(1−s) = (1−π) T\*(s) = G_veto\*(s), the VETO loop with π replaced by 1−π and the dual quorum. Exact on a grid in rationals. A captured checker that commands is the mirror image of one that only blocks.

**L2 (thresholds) `[FRAME]`.** The healthy state disappears (COMMAND) or a captured state appears (VETO) at a saddle-node (a transcritical exchange with the end fixed point in the two edge cases below): π\*_veto = 1/Λ(m, q) and π\*_command = 1 − 1/Λ(m, m−q+1), with Λ(m, q) = sup_c T(c)/c, attained where the tangent from the origin touches T. For a self-dual quorum the two sum to exactly 1. Checked against a direct max/min scan of the defining ratio and certified by an exact count of fixed points on either side. Examples: 5 seats, quorum 3: command 0.1651, veto 0.8349; 51 seats, quorum 26: 0.3546 and 0.6454. At three seats, quorum two, T/c = 3c − 2c² peaks at c = 3/4 with value 9/8, so veto 8/9 and command 1/9 *exactly*, with the two fixed points merging into one double root there (`fixed_points` returns it as semistable). Two edge cases swallow the interior root into the end root instead of merging it with a partner: if every seat is required (q = m) the command threshold is exactly 1 − 1/m and the root at p = 1 is double there (slope (1−π)m − 1 = 0, curvature m − 1); if one seat suffices (q = 1) the veto threshold is exactly 1/m and the root at p = 0 is double. The exact counter must divide an end root out to its full multiplicity; a mutation test showed that a version dividing it out once passed every earlier test, which is why these cases are now tested. As m grows both close on q/m with a gap of order √(ln m / m).

**L3 (the ghost) `[FRAME]`.** Just past π\* the iteration crawls through the ghost of the vanished fixed point, and the number of levels before it crosses a fixed level diverges like (π − π\*)^(−1/2): 100× closer, 10× longer (155 levels at 10⁻⁴ past, 1,568 at 10⁻⁶, m = 51). This is the generic signature of critical slowing down before a saddle-node (the early-warning literature, Scheffer and others, recalled).

**L4 (the price of the no-apex rule) `[FRAME]`.** A hierarchy of depth L ending in an unchecked apex has bottom-body compromise p_L, the L-th iterate from p₀ = π; the folded loop is the limit. For COMMAND the iterates increase to the least fixed point, so a finite hierarchy is never worse than the loop, and the gap shrinks at the slope of the map at the limit. At π = 0.1 with 5 seats and quorum 3, one level is already within 0.0024 of the loop. So the no-apex rule is nearly free while the loop is healthy, and near π\* the apex is a loan: it defers collapse by the length of the ghost, at the price of being unchecked.

**Caveat `[OPEN]`.** This is mean-field. A real ring or graph has correlated neighbours. §7 (P2) shows the direction of the error when compromise is positively associated; how large it is on a particular graph is not computed (B2).

## 6. Revitalization

n is the number of compromised seats among m. A sound seat is compromised at rate λ; a compromised seat is repaired at rate ν **only while n < q** (the body repairs itself; after q nothing is left to). It is a birth–death chain, reversible with weights C(m, n)(λ/ν)^n.

**N1 `[FORM]`.** The expected time to reach q from n = 0 is Σ_{j<q} (1/(u_j π_j)) Σ_{i≤j} π_i with u_j = (m−j)λ and π_i = C(m, i)(λ/ν)^i, in exact rationals. Three independent solvers agree: this closed form, Gaussian elimination on the hitting-time equations, and a uniformization power series.

**N2 (three regimes) `[FRAME]`, numerical.** With ρ = λ/(λ+ν), θ = q/m:
* ρ < θ: life ≈ C e^{mD(θ‖ρ)}/√m. Redundancy buys exponential life (Cramér's large-deviation rate, read on a reversible chain).
* ρ > θ: life → ln(ρ/(ρ−θ))/(λ+ν) **whatever m is**: the body dies on the time scale of one repair however many seats it has. You cannot out-vote decay.
* ρ = θ = ½: life ≈ ½(ln 2m + γ)/(λ+ν), γ Euler's constant, to four digits for m up to 25,600.
* No repair (ν = 0): life = (H_m − H_{m−q})/λ → ln(1/(1−θ))/λ, independent of m: unbounded terms are zero repair.

The repair-to-attack ratio must exceed (1−θ)/θ for redundancy to matter at all.

## 7. Prices without money

**P1 `[FORM]` + independence `[FRAME]`.** Read the price of compromising a party as bits(p) = −log₂ p. Under independence, probabilities multiply and bits add, so the cheapest set of parties that passes a gate in hyperethics' sense is the *most probable* way in. Then 2^−C ≤ P(gate taken) ≤ min(1, N·2^−C), C the cost in bits of the cheapest cover, N the number of minimal covers. *Proof.* The cheapest cover alone is one way in; the event "gate taken" is the union over minimal covers of "all its roots compromised", bounded by the sum of their probabilities, none exceeding 2^−C. ∎ Checked in exact rationals against inclusion–exclusion, and against hyperethics' own `gate_cost` through a golden file (`src/hyperoptimization/data/golden_l2_gate_bits.json`, regenerated by `scripts/regenerate_golden.py`, never importing hyperethics). A *stake* is log₂ of how many independent attempts a prize attracts: an attacker facing a cover of c bits succeeds by luck once in 2^c attempts, so a prize drawing 2^s attempts makes covers up to s bits affordable.

**P2 `[FRAME]`, common principal.** A shared principal is positive association between compromises. `additivity_gap` = bits(product of marginals) − bits(joint): for a principal at 3 bits commanding two otherwise unbreakable seats it is exactly 3 bits, the discount hyperethics finds when one Boss replaces two seats (gate cost 6 falls to 3). Group the seats of a body under shared principals holding every marginal fixed: with the quorum **above** the mean count, capture is **more** likely than the independent formula says; **below** the mean, less (worked: two groups of three seats, quorum 4, x = ¼: grouped 1/16, independent 0.0376). So the thresholds above are upper bounds on tolerable pressure when compromise is positively associated.

**P3 (breadth) `[FORM]`.** gate_cost ≤ gate_size × (dearest finite price). So a gate that must cost S bits needs at least S / (dearest price) independent roots: concentrating a stake forces a wider gate. Verified on 1,178 of hyperethics' own random gates (`scripts/verify_against_hyperethics.py`).

**P4 (synthesis) `[FORM]`.** Given purchasable parties with prices, a collusion bound k and a stake S, **any** system that obeys hyperethics' rule and costs at least S to take has at least max(k+1, m\*) parties in its amendment gate, where m\* is the fewest of the dearest prices summing to at least S; and the clique of mutual vetoes among the dearest parties attains it. *Proof.* The amendment gate must be passed by more than k independent roots (collusion-capturable otherwise), and, with no command edges, costs at most the sum of its members' prices, so it needs m\* parties; command edges only lower costs. A clique of the dearest max(k+1, m\*) parties with mutual vetoes, bounded terms and no stake meets every clause. ∎ Checked against an exhaustive search over 120 instances that builds each candidate as a hyperethics `Governance`. Example: a 40-bit stake, collusion bound 2, parties at 12 bits: m\* = 4, so a gate of 4. **Centralize the law, decentralize the gate**: the law is one canonical object (R2); the width of its gate is fixed by the stake, in bits.

## 8. Who can change the rule that says who can change the rule: the ordinal arithmetic of checks and balances

A rule is governed by another rule (the one that says how it may be changed) or by nothing; each rule carries the price in bits of the procedure it defines.

**E1 (weakest link) `[FORM]`.** protection(u) = min over every rule v above u of the price of v's procedure. To change u, satisfy its governor's procedure or first change the governor. Around a cycle every member has the same protection. Ross's paradox: an entrenched clause worth 100 bits amendable by a 5-bit clause is protected by 5 bits. Checked against an independent shortest-path search over sets of already-changed rules.

**E2 (no rank without a sovereign) `[FORM]`.** The authority height of a rule is the number of governors above it. If every rule has a governor then in a finite set every chain returns to itself (pigeonhole), so every height is ω; a finite height anywhere means some rule nothing governs, an unamendable apex. Checked over **every** functional graph on up to five rules.

**E3 (guarded loops) `[FORM]`, by Löb.** A ring of checkers A₀…A_{n−1}, each approving iff it can *prove* the next approves (a link through the provability modality of GL) or simply copying it. If GL proves the defining equations H, it proves that every member approves iff at least one link goes through proof. For rings of copying links GL proves nothing, and the prover returns a countermodel (everyone approves and no one approves both satisfy H). The decision procedure of hyperlogic proves it for all 27 rings of up to five checkers tested; the single formula tested is (H ∧ □H) → ⋀A_i, because H alone does not imply it even for the guarded ring: *H holding* and *H being provable* are different. For general n, de Jongh–Sambin fixed points give the same result. **Determinacy is not soundness**: the unique outcome is that everyone approves, whether or not approval was warranted.

**E4 (phase of a dependency graph) `[FORM]`.** For a finite graph of reliance put phase(x) = 0 if x relies on nothing, 1 + the largest phase among what it relies on otherwise, and ω if there is no finite solution. Then phase(x) is finite iff nothing reachable from x lies on a cycle. *Proof.* If a cycle x₁ → … → x_k → x₁ is reachable, a finite solution would give phase(x₁) ≥ k + phase(x₁), which is impossible. If none is reachable, the reachable part is acyclic and induction in reverse topological order gives the solution. ∎ With one governor per node this is E2 exactly (checked against `authority_height` on every functional graph up to five nodes); in general it is the rank of the well-founded part of a relation. `entrenchment.phases` peels the graph layer by layer; the tests compare it with the longest walk on all 65,536 digraphs on four nodes, self-loops included, and on random larger graphs, and `cores` with a closure computed by Floyd and Warshall's method. `cores` lists the groups that rely on one another: the places where no finite phase exists.

**The five orders.** What "ordinal" means here is not one thing but five, and they pull in different directions:

1. *Authority (validity), Kelsen.* Height above a rule in the chain of governors. Finite iff there is a sovereign (E2); with many governors, the phase of E4.
2. *Entrenchment (protection).* A cost in bits, set by the weakest link (E1). It flattens around a loop.
3. *Accountability.* Checks form cycles, not ranks (L2's no-apex theorem); power is not rankable.
4. *Phase.* The stage at which a process is decided: viability ranks (R1), the stage count of the quorum flow (ω exactly at a rational critical point, diverging like log(1/distance)/log T′(x_c) near it), and, for nested structure, ordinals below ε₀ (§9).
5. *Proof strength.* How much induction a proof that the system is stable needs. A lawset that certifies its own stability inside the theory it enforces runs into Gödel's second theorem and Löb's theorem; guarded loops (E3) are the sound way to close such a loop, with the limitation above.

**Reconciling "a central optimal law" with "no sovereign".** The validity order can be grounded (order 1 is finite) only at the price of an unamendable apex, which L2 calls corrupting. The accountability order must cycle. These do not conflict if the apex of validity is *not an agent*: L\* (R2) has no governor because it is derived, not authorized; changing it is not an act of will but a change of the game. The agents that enforce it sit in loops of checks, whose strength is their cheapest procedure (E1) and whose gate breadth is set by the stake (P4). So centralization of the *law* is forced (it is unique); centralization of the *power* is not, and is in fact forbidden as the stake grows.

## 9. Ordinal phases

`Ordinal` is a Cantor normal form below ε₀ (order, ordinary addition with 1 + ω = ω, natural sum, fundamental sequences with ω[n] = n, ω²[n] = ω·n, ω^ω[n] = ω^n). Checked against an independent valuation that replaces ω by a number larger than every coefficient.

**G1 `[FORM]`.** Goodstein's process: write n in hereditary base b, replace b by b+1 everywhere, subtract 1, repeat with base b+1. Replacing the base by ω turns each term into an ordinal and the step strictly lowers it, so the process reaches 0 whatever n is. Every term's ordinal evaluates back to the term, and the ordinals strictly descend for 1,500 terms each of the sequences from 1 through 7; the sequence from 4 starts at ω^ω and begins 4, 26, 41, 60, 83, 109. PA cannot prove termination for all n (Kirby–Paris), because ε₀ is its proof-theoretic ordinal; the descent below ε₀ is the proof. A finite game's ranks stay below ω + 1. So an *ordinal phase* is the rank of a process in a well-ordered set that every step lowers; its size measures how much induction the proof of convergence needs, not how long the process runs.

**F1 (the family's own phases) `[FRAME]`.** hyperstratum records which pinned repository cites, depends on or imports which. The snapshot used here is hyperstratum `87aadc3`, graph digest `b56f1a3c…`: 21 repositories, 74 recorded edges (67 citations, 4 imports, 3 declared dependencies). Read an edge as *source relies on target*, compute E4, and do it under five nested views of which edges count (the choice is D22's): *hard* (4 reliances: imports and declared dependencies), *formal* (15: plus citations made in a specification or source file), *substantive* (52: plus citations beyond the navigation files every repository carries), *weighty* (27: hard, or three or more citing files) and *all* (67).

* What the family *imports* is well founded: no cycle, 20 repositories at phase 0, metamathethicology at 1.
* What its specifications *say* is not. In the formal view hyperethics, hyperphysics and metamathethicology rely on one another (the will tensor of hyperethics' L1, hyperphysics' electrical reading, metamathethicology's transport), and hyperlogic, which relies on all three, inherits phase ω; hypermath sits at phase 1. That three-repository core survives every view that keeps the specifications. In the *all* view an eleven-repository core appears that holds every repository of the foundation tier, plus two pairs (hyperobjectivity and hypersubjectivity, hyperspace and hypertime). That is the weaker reading, because navigation lists put every sibling in every repository.
* Only five repositories (hyperbiology, hyperchaos, hyperemergence, hyperorder, hyperprobability) have phase 0 in the *all* view: they cite nothing in the family.
* The snapshot is hyperstratum's static family graph, which lists 21 repositories, while hyperstratum pins 24 (its own HS-002 records that the two disagree). Absent from the graph: verifier, taxonomy-of-deception, speedrunning-social-development, throwaway-papers and meta-morphoscripted-origination. It lists hyperbiology, which is not yet pinned, and hyperstratum itself.
* A second, independent record: hyperstratum's live scan over all 24 pinned fields (149 edges, each a link to a repository or its name in prose, with a count; same commit). Keeping reliances of ten or more links or mentions, exactly hyperethics, hyperphysics and metamathethicology form a core, beside hypergrammar with hypermath and verifier, and hyperspace with hypertime. Counting any mention, one sixteen-repository core holds the three. Counting explicit links only, hyperethics and metamathethicology rely on each other and hyperphysics is in no core. So the three-repository core does not depend on the static graph, and its exact shape does depend on how a reliance is counted.

Reading. The build is well founded and the meaning is not, which is the shape E2 predicts for a system with no apex: authority is a loop, here at the level of reference. A citation is evidence of mention, not of need; the cores are where mentions go round in a circle, and whether a circle is vicious is for the owners to say (`nd-citation-is-need`). One snapshot is not the family now.

**F2 (what the family's declared open items are about) `[FRAME]` by wording, `[HYPER]` by hand.** hyperstratum's scan finds 116 lines tagged `[OPEN]` in 8 of the 24 pinned fields, and the `.hm` specifications declare 6 more with `open` (hyperethics 3, hyperlogic 1, metamathethicology 2): 122 declared open items in 11 fields. The other 13 fields declare none in either form. (An earlier version of this section counted only the six declarations and called the rest unreadable. That was wrong.)

* *By wording* (D23). A table of keywords for six topics (optimization, viability, persistence, stability, ordinal, closure), fixed before the items were read against it. None of the 122 mentions optimization or viability. Two mention persistence or maintenance: hyperethics' consistency self-maintenance paradox, and a time-machine line that uses the word in another sense. 106 of the 122 (87%) mention none of the six topics. Most of the rest is about time, reality, naming and transport: hyperphysics' time-bubble sheet (18 lines), quantum-gravity candidates (8), the end of base reality (7), self-consistent correction (7); hyperreality's provenance and field specification (33). It is a proxy for a reading by the owners: it misses synonyms and counts a word in any sense.
* *By hand*, the items this package touches, none closed:
  * hyperethics `nd-consistency-self-maintenance`, its principal open problem: the shape R6 treats.
  * metamathethicology `nd-stage-semantics`: asks what its stages ω, ω+1, ω+2 mean, and says they order *availability* (a term is available only after what it was formed from). That is the phase of E4 on the formation graph; why the starting stage is ω stays a convention, as the file says.
  * hyperethics `nd-adequacy-of-the-capture-model`: meets B2; the direction of the error under correlation is computed, its size is not.
  * hyperphysics `docs/research/LAW_RANKING.md` (five lines tagged `[OPEN]`, three of them questions), from the owner's 2026-10-06 statement that laws are absolute but some take priority in resolution, with whole-forms above them. It ranks laws by dependence in five tiers. E4 on its "depends on" column gives phases 0, 1, 2, 2, 3, not the tier numbers 0 to 4, and leaves two pairs unordered: conservation with dynamics, and conservation with statistical laws (`test_the_ranking_of_laws_in_hyperphysics_by_dependence_has_phases_that_are_not_its_tier_numbers`). So its question, whether "priority in resolution" is the dependence order, has a partial answer: it cannot be that order alone, since dependence does not decide those pairs. What does is open. A candidate, not built: order the laws, keep the highest-ranked absolute for as long as the system can survive with them, and let the rest yield; that is a lexicographic order of the kernels of the laws taken together, and it differs from dependence in general.
* Not touched: hyperlogic's `nd-the-gate` (a decision about when material may move), hyperethics' `nd-implicit-control` (which reading is meant, B6), metamathethicology's `nd-charge-constant-law` (a physical map), and the time-bubble, quantum-gravity and naming items.

The open items of the ethics, physics and metamathethicology core (hyperphysics 56, hyperethics 3, metamathethicology 2) are mostly about time, reality and the electrical transport of the will; by wording none is about optimization or viability. As of this snapshot that is evidence *against* D18 for that core, within the limits of a wording proxy.

## 10. Axioms

**Adopted (five), each an assumption of the model and no measurement of any institution.**
* `ax-existence-as-viability`: existence is the ability to stay safe for ever against any nature.
* `ax-free-will-as-full-support`: a law limits what is permitted, not what is taken.
* `ax-corrigibility-as-reachability`: stay where an overseer can reach home.
* `ax-self-repair`: repair is done by the body itself and only while it holds.
* `ax-independence`: seats are independent, bodies are quorums, all bodies alike.

**Derived, by the optimality structure.** From the five: the least restriction is unique and canonical (R2); no one can be exempt (R3, hyperethics' exempt-reach); irreversible moves are those that leave the home-reachable set (R4); repair must outpace attack by the quorum margin (N2); the gate that protects the law must widen with the stake (P3, P4); a constitution with no sovereign has no finite validity rank (E2, E4); a loop of checks is determined only through proof (E3); and maintenance has only a coinductive reading, certified by a self-supporting set (R6). These are the "axioms that hold by nature of the optimality structure" in the sense that each is a theorem of the model, not an extra assumption.

**Not claimed.** I did not derive anything about aggregating preferences. Arrow-type and Gibbard–Satterthwaite-type impossibility results (recalled) would apply the moment a welfare ordering over participants is added; none is added here (B1).

## 11. Conceptual blocks

These got in the way. Each is also in `DECISIONS.md` as a decision, with what would resolve it.

* **B1: persistence is not goodness (D04, D16, `[OPEN]`).** The greatest law permits some stable arrangements a participant would call tyranny (`_tyranny_game`). Why the law that preserves existence is the optimal one is the family's GC-5. Without it, "optimal" in this package means "least restriction that preserves existence" and nothing more.
* **B2: mean-field versus a real graph (D03, `[OPEN]`).** The loop thresholds assume independent seats and identical bodies. The direction of the error is computed (positive association makes capture likelier above the mean count); its size on a ring or graph is not.
* **B3: the physical claim has no witness here (D17, `[HYPER]`).** That reality adheres to such a structure is not something finite models can test. The stated form of a refuter is in `DECISIONS.md`. Bistability, hysteresis and critical slowing down are observed in real systems; matching a threshold to independently measured microscopic parameters is the missing step.
* **B4: free will is formalizable only as full support.** The alternative (rational best response) makes laws unnecessary for agents that see fatal moves, which empties the question. The link to hyperethics' will tensor (L1) is not made: the greatest law could be read as an *invariant will* whose source is the system's own existence condition (self-sourced in L1's sense), but that is a proposal, `[HYPER]`.
* **B5: "revitalization" has two readings.** Adopted: self-repair while the body holds. Not done: renewal as periodic re-creation of the body.
* **B6: "implicit control" and "allowable" are inherited unresolved from L2.**
* **B7: "central will space" is a function over states here, not a space.** No topology or metric on laws is constructed. "Convergence from many simulated law structures" is made precise as R5.
* **B8: the closing-catalyst claim (D18, `[OPEN]`).** First passes are done (F1, F2): the family's recorded reliance has a core with phase ω that its imports do not, in two independent records; and of 122 declared open items none is about optimization or viability by wording (D23), four are touched by this package by hand, and none is closed. Read as of this snapshot that is evidence against the claim for the ethics, physics and metamathethicology core. The map from open items to the repositories that would close them is not built; it needs the owners, since only the owners can say what would close an item.
* **B9: self-funding capture is not modelled (D15).** Reinvestment is positive feedback on the compromise probability.

## 12. Spec map

`tests/test_spec.py` parses this table: every name declared in `L0_viability.hm` appears in it, every named test exists, every named code object exists, every decision id is in the ledger.

| spec | kind | code | test | decision |
|---|---|---|---|---|
| ax-existence-as-viability | axiom | viability.kernel | test_a_rank_is_zero_off_the_safe_set_and_omega_exactly_on_the_kernel | D04 |
| ax-free-will-as-full-support | axiom | viability.survival_probability | test_survival_probability_is_one_for_the_greatest_law_and_matches_stepwise_bounds_otherwise | D06 |
| ax-corrigibility-as-reachability | axiom | viability.reach_home | test_no_corrigible_law_ever_leaves_the_set_from_which_home_is_reachable | D07 |
| ax-self-repair | axiom | renewal.Chain | test_closed_form_and_linear_system_agree_exactly | D09 |
| ax-independence | axiom | quorum.tail | test_tail_is_the_probability_of_at_least_q_successes_by_enumeration | D03 |
| d-greatest-law | derive | viability.greatest_law | test_a_law_sustains_existence_exactly_when_it_permits_nothing_outside_the_greatest_law | D05 |
| d-exemption-is-fatal | derive | viability.survival_probability | test_one_permission_outside_the_greatest_law_at_a_reachable_state_ends_existence | D06 |
| d-corrigible-kernel-inside | derive | viability.reach_home | test_corrigibility_costs_freedom_and_never_gains_it | D07 |
| d-rank-certificate | derive | viability.check_ranks | test_changing_any_single_rank_to_any_other_value_is_caught | D08 |
| d-quorum-duality | derive | quorum.dual | test_dual_quorum_is_the_blocking_threshold_and_duality_is_exact | D11 |
| d-one-critical-point | derive | quorum.interior_fixed_points | test_exactly_one_interior_fixed_point_iff_the_quorum_is_neither_any_nor_all | D11 |
| d-command-veto-duality | derive | loop.step | test_the_duality_identity_is_exact | D10 |
| d-renewal-regimes | derive | renewal.mean_lifetime | test_redundancy_buys_exponential_life_with_the_kullback_leibler_rate | D09 |
| d-price-brackets-probability | derive | surprisal.sandwich_gate | test_the_cost_brackets_the_probability_within_the_number_of_minimal_covers | D01 |
| d-independence-understates | derive | surprisal.capture_probability | test_grouping_seats_under_principals_raises_capture_risk_above_the_mean_and_lowers_it_below | D03 |
| d-weakest-link | derive | entrenchment.protection | test_protection_is_the_weakest_link_by_independent_shortest_path | D14 |
| d-no-rank-without-sovereign | derive | entrenchment.authority_height | test_no_finite_rank_exists_unless_something_is_unamendable | D14 |
| d-guarded-loops | derive | data/golden_gl_loops.json | test_the_recorded_prover_runs_name_the_checkout_they_came_from | D19 |
| d-goodstein-descent | derive | ordinal.goodstein_sequence | test_every_term_carries_an_ordinal_that_evaluates_back_to_it_and_the_ordinals_descend | D13 |
| d-breadth-and-synthesis | derive | scripts/verify_against_hyperethics.py | script:verify_against_hyperethics.py | D15 |
| d-self-maintenance-is-coinductive | derive | viability.maintains | test_the_kernel_is_the_union_of_every_set_that_supports_itself | D21 |
| d-dependency-phase | derive | entrenchment.phases | test_every_digraph_on_four_nodes_agrees_with_the_walk_definition | D22 |
| nd-persistence-is-optimal | non-derive | — | — | D16 |
| nd-physical-adherence | non-derive | — | — | D17 |
| nd-determinacy-is-soundness | non-derive | — | — | D19 |
| nd-welfare-from-viability | non-derive | — | — | D04 |
| nd-consistency-self-maintenance-resolved | non-derive | — | — | D21 |
| nd-citation-is-need | non-derive | — | — | D22 |
| nd-closing-catalyst | open | — | — | D18 |
| nd-mean-field-adequacy | open | — | — | D03 |
| nd-self-funding | open | — | — | D15 |

## 13. Roadmap

Ordered by what would most change the conclusions.
1. A ring or graph of bodies solved exactly for N ≤ 12, to size B2.
2. The item-level closure map (B8): F1 and F2 are the repository-level and wording-level first passes. Make the family's 122 declared open items nodes, what would close each a governor, and read the ranks; this needs the owners, since only they can say what would close an item. Build the lexicographic-kernel candidate for "priority in resolution" (F2) and compare it with dependence on the law ranking.
3. Reinvestment (B9) as positive feedback in `loop`.
4. Infinite games: transfinite ranks, with the attractor construction.
5. The hydra game and the Hardy hierarchy beside Goodstein, with `ordinatics`.
6. A proof of the N2 limits in place of the numerical check.
