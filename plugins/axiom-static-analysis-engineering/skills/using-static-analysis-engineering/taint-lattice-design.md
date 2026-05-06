---
name: taint-lattice-design
description: Use when designing the abstract domain of a dataflow analyzer — the set of values the analyzer tracks for each variable and how they combine at control-flow joins. Covers lattice formalism (partial order, join, meet, top, bottom, monotonicity, finite ascending chains); the tier model and when to extend it; the dual-lattice problem (confidentiality vs integrity); and the most common anti-pattern: a "lattice" that is secretly Boolean wrapped in tier names. Produces `02-abstract-domain-spec.md`.
---

# Taint Lattice Design

## What the Lattice Is For

A static analyzer that propagates *anything* — taint, ownership, capability, type, range, nullability — is computing a value per program point in some **abstract domain**. The abstract domain is a **lattice**: a partially ordered set with well-defined joins (and usually meets), where two pieces of information can always be combined into the most precise piece consistent with both.

Every analyzer has an abstract domain. Most analyzers fail to write it down. The result is the recurring failure mode in this pack:

> **A "lattice" that is secretly Boolean.** The team names tiers — `Trusted`, `Sanitised`, `Untrusted` — but every transfer function is `tainted = a or b`. The names exist; the algebra doesn't. The analyzer ships, accumulates contradictions, and rules grow exceptions until nobody can describe what the engine actually computes.

`02-abstract-domain-spec.md` exists so that the algebra is in writing before the inference is.

## Lattice Vocabulary (in 90 seconds)

A lattice $(L, \sqsubseteq)$ is a set $L$ with a partial order $\sqsubseteq$ ("at most as precise as") such that any two elements have:

- **A join** $a \sqcup b$ — the *least upper bound*: the smallest element that is $\sqsupseteq$ both $a$ and $b$.
- **A meet** $a \sqcap b$ — the *greatest lower bound* (often unused in dataflow but defined for completeness).

A lattice usually has:

- **Top** $\top$ — the worst-case / "could be anything" element. (For taint: "definitely tainted, no recovery.")
- **Bottom** $\bot$ — the empty / "no information yet" element. (For taint: "haven't analysed this yet.")

**Direction matters and is a frequent source of confusion.** Two conventions:

- **"Information-as-precision":** $\bot$ = no information, $\top$ = full information. Dataflow analyses for *may-properties* (may-be-tainted) often go this way: starting at $\bot$, joining adds information.
- **"Information-as-conservatism":** $\bot$ = best-case (clean), $\top$ = worst-case (tainted). For *over-approximation* analyses, $\top$ is the safe answer when in doubt.

Pick one. Document it. The wrong direction makes joins do the opposite of what rules expect. Most security taint analyses use the second convention: $\top = \text{tainted}$, joins climb toward worst-case.

## The Three Properties That Make Inference Terminate

A static analyzer that walks a fixed-point algorithm over a lattice terminates if and only if:

1. **Monotonic transfer functions** — every transfer function $f$ satisfies $a \sqsubseteq b \implies f(a) \sqsubseteq f(b)$. New information never *un-taints* (or, more generally, never moves a value backward in the order). Without monotonicity, the worklist algorithm can oscillate.
2. **Finite height (or ascending chain condition)** — every ascending chain $a_0 \sqsubseteq a_1 \sqsubseteq \dots$ is eventually constant. A lattice with infinite ascending chains can be valid mathematically but lethal to a fixed-point algorithm; you need a **widening operator** (an explicit accelerator that jumps to $\top$ after $k$ refinements) to bound iteration.
3. **Decidable order** — the analyzer can compare two elements in finite time. (Almost always trivial; mentioned here because property-bag domains can violate it.)

`02-` must explicitly state which of these properties hold, why, and — if (2) requires widening — what the widening operator is.

## Designing the Tier Set

For taint analysis specifically, the lattice usually starts as a totally-ordered chain of "trust tiers." The minimum useful design names at least three:

```
                    Untrusted   ← user input, network, file contents, env vars
                        ⊔
                    Sanitised   ← passed through a recognised cleansing function
                        ⊔
                    Constant    ← compile-time literal, derived only from constants
                        ⊔
                     Trusted    ← installation-fixed config, hard-coded credentials
```

Three problems people hit immediately:

**Problem 1: Two-tier lattices.** "Tainted / clean" is technically a lattice but is almost never expressive enough. A real codebase has at least: input that hasn't been touched, input that's been *partially* sanitised (escaped for HTML but not for SQL), and input that's been fully sanitised for a specific sink. Two tiers force you to either lie or proliferate special-case rules.

**Problem 2: One lattice for orthogonal properties.** A value that is "untrusted as input" and "untrusted as output" needs *two* lattices joined into a product domain, not a single "tainted" tier. Confidentiality (Bell-LaPadula: information flows from low to high) and integrity (Biba: information flows from high to low) are dual lattices; collapsing them into one yields contradictions ("the value is both Public and Untrusted, but Sensitive things must not flow to Public sinks, which means…").

**Problem 3: Sanitisers as boolean.** A sanitiser doesn't return "clean"; it returns "clean *for this sink class*." `html_escape()` produces a value safe for HTML output but unsafe for SQL. Encoding this requires the sanitised tier to be **parameterised by the sink class**, or you accept that the lattice cannot distinguish them and rely on rule-side annotations.

## Worked Example: A Three-Tier Web Taint Lattice

```
order: ⊥ ⊑ Trusted ⊑ Constant ⊑ Sanitised(sink_class) ⊑ Untrusted ⊑ ⊤

join (a ⊔ b):
  - if a == b, return a
  - if either is ⊤, return ⊤
  - if either is Untrusted, return Untrusted
  - if both are Sanitised(s) with same s, return Sanitised(s)
  - if both are Sanitised(s1), Sanitised(s2) with s1 ≠ s2, return Untrusted
    (CRITICAL: sanitisation for different sinks does NOT compose into a stronger
     guarantee. Joining HTML-safe and SQL-safe gives "safe for nothing
     specific," which conservatively means "untrusted at the sink.")
  - if a == Constant and b == Trusted (or vice versa), return Constant
  - otherwise: take the higher of the two

transfer functions:
  - literal:                Constant
  - request.GET[...]:        Untrusted
  - os.environ[...]:         Trusted (operator-controlled) or Untrusted (user-influenceable)
                             — declare this in 02-, do not let inference decide
  - html_escape(x):          x ⊔ ⊥ → Sanitised(html)        if x is not already Sanitised(other)
                             (preserves stronger sanitisations only if they imply HTML-safety;
                              usually they don't, so re-sanitise)
  - sql_param_bind(x):       Sanitised(sql)
  - x + y:                   x ⊔ y
  - format("...", x, y):     x ⊔ y ⊔ ⊥                         (string formatting joins arguments)
  - cast(x, T):              x                                 (type coercion does not sanitise)
```

Note three things in the worked example:

- **Joining two different sanitisations downgrades to Untrusted, not to a "doubly-sanitised" super-tier.** This is the lattice forcing honesty: there is no super-tier, because there is no sink that consumes both.
- **`os.environ` is declared, not inferred.** Some env vars are operator-controlled (Trusted); some are user-influenceable (Untrusted). The lattice cannot tell. Declaring this is `02-`'s job.
- **Type casts do not sanitise.** A frequent rule-side bug is treating `str(x)` as a sanitiser; it isn't. The lattice makes this a one-liner: transfer is identity.

## Extension: Adding a Tier

Adding a new tier is **lattice-breaking**. From the router's `02-`-change rule:

> If you change `02-` lattice tiers added/removed/renamed, you also re-emit `03-` (transfer functions), `04-` (rules consuming the new tier), `05-` (FP rate baseline resets). **Yes — version-bump + re-baseline.**

The extension procedure:

1. **Justify the new tier** — what real-world value cannot be expressed by the existing tiers? "Sanitised for HTML and SQL" is not a justification (use a product lattice); "Operator-asserted-safe-for-this-call-site only" might be.
2. **Place the new tier in the order** — strictly between two existing tiers, or at top/bottom. Drawing the Hasse diagram is mandatory: if the new tier doesn't fit a poset, it isn't a tier.
3. **Define joins** with every other tier. Tabulate. Asymmetric or non-commutative joins are bugs.
4. **Define transfer functions** for every operation that produces or consumes the new tier.
5. **Update rules** that previously fired on a tier subset that the new tier should be in (or out of).
6. **Re-baseline FP rate** — a new tier changes the population of findings; the old baseline is now meaningless.
7. **Bump the analyzer version** and emit a migration note in `99-`.

Skipping any of these and the lattice silently denies its own properties. The most common skip: adding a tier without re-tabulating joins. The result: $a \sqcup b \neq b \sqcup a$, which is not a join.

## The Product Lattice (When You Have Two Properties)

A value that is both "untrusted as input" and "sensitive as output" needs:

$$L_{\text{combined}} = L_{\text{taint}} \times L_{\text{sensitivity}}$$

Order: $(a_1, a_2) \sqsubseteq (b_1, b_2) \iff a_1 \sqsubseteq b_1 \land a_2 \sqsubseteq b_2$.
Join: $(a_1, a_2) \sqcup (b_1, b_2) = (a_1 \sqcup b_1, a_2 \sqcup b_2)$.

Properties (monotonicity, finite height) lift componentwise. Transfer functions decompose: a sanitiser updates the taint component; a redaction updates the sensitivity component; most operations update one or neither, not both.

**Anti-pattern: cross-tier joins.** A flat enum `{Trusted, Sanitised, Untrusted, Sensitive, Public}` is *not* a lattice over both properties — what does $\text{Untrusted} \sqcup \text{Sensitive}$ even mean? Make it a product. The flat enum is the "boolean lattice masquerading as types" anti-pattern with extra steps.

## Soundness and Completeness Statement

`02-` must declare which side the analyzer errs on. The choice is forced by the lattice direction:

- **Sound (over-approximating)** — analyzer reports every real positive plus some false positives. The right-error: false positives. The lattice $\top$ is "possibly bad"; in doubt, climb. This is the right choice for security taint analyzers — a missed injection is worse than a refused-to-run.
- **Complete (under-approximating)** — analyzer reports only definite positives, missing some real ones. The right-error: false negatives. The lattice $\top$ is "definitely bad"; in doubt, retreat. This is occasionally right for advisory linters where noise costs more than misses.
- **Neither** — engineering hybrid (some checks sound, some complete). Acceptable but each rule must declare its side; cross-rule reasoning requires extra care.

Per Rice's theorem, no non-trivial semantic property of programs is decidable in general. **An analyzer claiming to be both sound and complete on a non-trivial property is wrong.** The honest version: "sound for the abstract domain, complete relative to the soundness loss." Write it that way.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Boolean lattice with tier-shaped names | Joins are `or`; rules accumulate exceptions | Promote to a real lattice; tabulate joins; re-derive transfer functions |
| Direction confusion (⊥ vs ⊤) | "Sanitisation makes things more tainted" | Pick one convention in `02-` and propagate consistently |
| Treating sanitised-for-A as sanitised-for-B | False negatives at the sink | Parameterise sanitisation by sink class; downgrade on cross-sink joins |
| One lattice for orthogonal properties | Contradictions; ambiguous "tainted-and-sensitive" cells | Build a product lattice; lift transfer functions componentwise |
| Adding a tier without re-tabulating joins | Non-commutative joins; oscillating worklist | Treat tier addition as lattice-breaking; emit migration in `99-` |
| Casts and type coercions counted as sanitisers | False negatives where `str(x)` "cleans" things | Transfer for cast = identity; sanitisation requires explicit transformation |
| No widening on infinite-height domains (e.g., interval analysis) | Inference doesn't terminate | Add widening at chosen depth $k$ with explicit jump-to-$\top$ |
| "Sound and complete" claim | Either lying or trivial property | Pick a side; cite Rice; document the loss |

## The Decision Output (`02-abstract-domain-spec.md`)

A complete `02-` answers:

1. **The lattice elements** — explicit set, including $\top$ and $\bot$.
2. **The partial order** — Hasse diagram, even ASCII; ambiguity here is fatal downstream.
3. **The join (and meet, if used)** — full table for finite lattices; algorithm for parameterised ones.
4. **Direction convention** — $\top$ = worst-case-tainted or $\top$ = full-information; document which.
5. **Monotonicity statement** — every transfer function is monotonic; proof or argument.
6. **Finite-height / widening** — chain length bound, or explicit widening operator.
7. **Transfer functions** — for every operator the analyzer cares about (assignment, call, format, concat, index, cast, sanitiser, sink).
8. **Sources** — how raw values enter the lattice (request inputs, env vars, file reads, network).
9. **Sinks** — what consumes lattice values (`os.system`, SQL execute, template render, file write).
10. **Sanitisers** — what transforms a higher tier into a lower one, parameterised by sink class where applicable.
11. **Soundness/completeness statement** — which side errs; cite Rice for the impossibility.
12. **Out-of-scope properties** — what the lattice intentionally does NOT track. (E.g., "this lattice does not track confidentiality; for that, build a product lattice with sensitivity tiers.")

## Cross-References

- `ast-visitation-patterns.md` — what produces the nodes the lattice values attach to
- `three-phase-inference.md` — how the lattice values propagate; depends on monotonicity and finite-height from this sheet
- `plugin-architecture-for-analyzer-rules.md` — how rules consume lattice values (rule fires when value at sink is $\sqsupseteq$ a threshold)
- `false-positive-economics.md` — when a "false positive" is the lattice over-approximating correctly vs the lattice being wrong
- `static-vs-runtime-tradeoffs.md` — invariants the lattice cannot express and that fall back to runtime
- `cross-module-flow-analysis.md` — how the lattice value crosses function summaries and module boundaries
- Cross-pack: `ordis-security-architect:design-controls` — the threat model that forces the tier set
