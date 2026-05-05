---
name: using-static-analysis-engineering
description: Use when designing or extending a static analyzer — a linter, a taint tracker, a contract checker, an entity extractor, a governance rule engine, or any tool that reads source code and produces verdicts about it without running the program. Routes through AST visitation patterns, taint-lattice (abstract-domain) design, three-phase inference (variable → function → callgraph), plugin architecture for rules, false-positive economics, and the static-vs-runtime boundary. Engineering pack: how to build the analyzer. For consuming an existing analyzer's output to map a codebase, use `/system-archaeologist` instead.
---

# Using Static Analysis Engineering

## Overview

**A static analyzer is an engine: AST → abstract domain → fixed-point inference → verdict. Treat it as one, or your "linter" calcifies into a pile of regexes that nobody trusts.**

This pack treats *building* an analyzer as a discipline distinct from running one. A real analyzer has a chosen visitation strategy (visitor, walker, transformer), an abstract domain it computes over (a lattice with defined join semantics, monotonicity, and finite height), a phased inference pipeline that terminates because the lattice does, an extension surface that allows new rules without forking the engine, and an honest economics for false positives — because the rate at which suppressions accumulate determines whether the analyzer is load-bearing or ceremonial five years from now.

This is the *producer-side* counterpart to architecture analysis:

- **`axiom-system-archaeologist` consumes analyzers** — runs existing tools, ingests their findings, and synthesises a system map. The analyzer is an oracle; the archaeologist is its reader.
- **`axiom-static-analysis-engineering` (this pack) builds analyzers** — designs the AST visitor, the abstract domain, the inference order, the rule plugin model, and the suppression discipline. The analyzer is the artifact; the engineer is its author.
- **The two pair**: an archaeologist that finds a gap in coverage hands the gap to this pack; this pack ships the new rule; the archaeologist re-runs and the gap closes. Cross-link, don't duplicate.

## When to Use

Use this pack when:

- You are building a new analyzer from scratch (a linter, a taint tracker, a contract checker, an entity extractor, a graph-extraction tool, a governance rule engine).
- You inherited an analyzer that works but cannot be extended — rules live as ad-hoc functions, no shared IR, every new check is a special case.
- You need to add typed dataflow (taint, ownership, capability tracking) to a system that currently does pattern matching, and the cost of getting the lattice wrong is years of false positives.
- A team is about to "just write some checks" and you can already see them inventing a fragile shadow-AST that won't survive contact with three-letter dynamic constructs (decorators, metaclasses, eval, dynamic import).
- You need to choose between *static* enforcement, *runtime* enforcement, or *both* for the same property, and the team is leaning whichever way the last bug landed.
- Suppressions in your existing analyzer are growing faster than rules, and nobody can tell you whether the `# noqa` from 2022 is still load-bearing.

Do **not** use this pack when:

- You want to *run* an existing analyzer (ruff, mypy, pylint, semgrep, eslint, clippy) — that is a Python/Rust/JS engineering tooling problem; use `/python-engineering`, `/rust-engineering`, or framework-specific guidance.
- You want to *consume* an analyzer's output to build a system map → `/system-archaeologist`.
- You want a turnkey lint config — this pack designs the engine; off-the-shelf analyzers come with their own rule sets.
- You are designing the *audit trail of decisions* an analyzer makes (who suppressed what, when, why, with what authority) → suppressions are decisions; cross-link to `/audit-pipelines`. This pack handles the *engine*; that pack handles the *evidence*.
- You are doing rule design at the policy or compliance level (NIST control families, SOC 2 criteria) — that is a `/security-architect` or `/sdlc-engineering` problem; this pack builds the engine that *enforces* whatever policy lands.

## Start Here

If your input is "we want a static analyzer for *X*" and you have not run this pack before:

1. Read `ast-visitation-patterns.md` — choose visitor vs walker vs transformer. The choice constrains everything downstream. Emit `01-visitation-strategy.md`.
2. Read `taint-lattice-design.md` — define the abstract domain. Pick the lattice (the tier set, the join), prove monotonicity and finite height, write the extension rule. Emit `02-abstract-domain-spec.md`.
3. Read `three-phase-inference.md` — order the inference (variable → function → callgraph), specify the worklist, prove termination from the lattice properties of step 2. Emit `03-inference-pipeline-spec.md`.
4. Use the **Routing** section below for plugin architecture, false-positive economics, and the static-vs-runtime boundary.
5. Run the **Consistency Gate** before declaring `99-analyzer-engineering-specification.md` ready.

Steps 1–3 are the spike. The visitation strategy defines what the analyzer *sees*; the abstract domain defines what it *means*; the inference order defines how it *propagates*. If those three artifacts hold together, the rest is fill-in. Most "the analyzer became unmaintainable" stories trace to one of these three: ad-hoc visitation (no shared traversal), a domain that's secretly boolean (just "tainted/clean") even though the code says it's a lattice, or inference that runs in whatever order the developer happened to write functions in.

## How to Access Reference Sheets

All reference sheets are in the same directory as this `SKILL.md`. When you see a link like `[taint-lattice-design.md](taint-lattice-design.md)`, read the file from the same directory.

## Pipeline Position

```
axiom-static-analysis-engineering              axiom-system-archaeologist
  BUILDS analyzers                  ←-cross-ref-→   CONSUMES analyzers
  designs AST visitation,                          runs analyzers, ingests
  abstract domain, inference,                      findings, synthesises a
  rule plugin model, suppression                   system map and dependency
  discipline                                       graph
  ─────────────────────────────────────────────────────────────────────
                            ↓
        A coverage gap the archaeologist finds (an unreachable
        subsystem, an untyped edge, a missing flow) becomes a rule
        request to this pack. This pack ships the rule. The
        archaeologist re-runs and the gap closes.

axiom-audit-pipelines (evidence)               axiom-static-analysis-engineering (engine)
  decisions are evidence;             ←-cross-ref-→   suppressions are decisions;
  canonical bytes, fingerprint                       waiver lifecycle, audit trail
  chains, signed exports                              of who suppressed what
  ─────────────────────────────────────────────────────────────────────
        Suppressions emitted by this pack's analyzer ARE
        audit-grade decisions. Their lifecycle (granted, reviewed,
        expired, re-granted) lives in the audit pack's pipeline.
        Cross-link in 04-rule-plugin-spec.md and 05-fp-economics.md.

ordis-security-architect (policy)              axiom-static-analysis-engineering (enforcement)
  threat models, control families,    ←-cross-ref-→   the engine that enforces
  required invariants                                  whatever invariants land
  ─────────────────────────────────────────────────────────────────────
        Security architect says "untrusted input must not reach
        os.system without sanitisation." This pack builds the
        analyzer that enforces it as a taint rule with a defined
        lattice and a tractable false-positive rate.
```

## Expected Artifact Set

The pack produces a numbered artifact set in an `analyzer-engineering/` workspace:

| # | Artifact | Producer skill |
|---|----------|----------------|
| 00 | `scope-and-targets.md` | router (this SKILL.md) |
| 01 | `visitation-strategy.md` | `ast-visitation-patterns` |
| 02 | `abstract-domain-spec.md` | `taint-lattice-design` |
| 03 | `inference-pipeline-spec.md` | `three-phase-inference` |
| 04 | `rule-plugin-spec.md` | `plugin-architecture-for-analyzer-rules` |
| 05 | `false-positive-economics.md` | `false-positive-economics` |
| 06 | `static-runtime-boundary.md` | `static-vs-runtime-tradeoffs` |
| 99 | `analyzer-engineering-specification.md` | router-owned consolidation |

**Planned for v0.2.0** (numbered slots reserved; do not collide):

| # | Artifact | Producer skill (planned) |
|---|----------|--------------------------|
| 07 | `callgraph-construction.md` | `callgraph-construction` (resolution depth, virtual dispatch, dynamic imports) |
| 08 | `cross-module-flow.md` | `cross-module-flow-analysis` (boundary semantics, summary functions) |
| 09 | `decorator-as-assertion-spec.md` | `decorator-as-assertion` (runtime + static dual enforcement, descriptor pattern) |
| 10 | `manifest-and-coherence.md` | `manifest-driven-configuration-with-coherence-validation` |
| 11 | `sarif-and-ci.md` | `sarif-emission-and-ci-integration` (GitHub Code Scanning, exit-code semantics) |
| 12 | `scaling-and-incrementality.md` | `scaling-to-large-codebases` (caching, parallelism, watch mode) |
| 13 | `llm-assisted-explanation.md` | `llm-assisted-rule-explanation` (the pattern, not the LLM) |

**Planned commands (v0.2.0):** `/scaffold-analyzer`, `/design-tier-model`, `/design-rule-set`.
**Planned agents (v0.2.0):** `rule-designer`, `false-positive-analyst`.

## Spec Dependency Graph

The numbered artifacts are not independent — changes propagate. Read this before editing any spec.

```
01-visitation-strategy.md           (the substrate — what gets walked, in what order)
        │
        ▼
02-abstract-domain-spec.md          (the IR — the lattice the analyzer computes over)
        │
        ▼
03-inference-pipeline-spec.md       (the algorithm — phased fixed-point over the lattice)
        │
        ▼
04-rule-plugin-spec.md              (the extension surface — how rules consume the IR)
        │
        ▼
05-false-positive-economics.md      (the operational reality — suppression lifecycle)
        │
        ▼
06-static-runtime-boundary.md       (the scope statement — what NOT to enforce statically)
```

**Coordinated re-emission rules:**

| If you change | You also re-emit | Lattice-breaking? |
|---------------|------------------|-------------------|
| `01-` visitation strategy (visitor → walker, or visit order) | `03-` (worklist seeding may change), `04-` (rule entry points) | No (semantic-equivalent rewrites permitted; document) |
| `02-` lattice tiers added/removed/renamed | `03-` (transfer functions), `04-` (rules consuming the new tier), `05-` (FP rate baseline resets) | **Yes — version-bump + re-baseline** |
| `02-` join semantics changed (e.g., greatest-lower-bound flipped to least-upper-bound) | `03-` entirely, `04-` rules consuming join, `05-` baseline reset | **Yes — this is a different analyzer** |
| `03-` inference ordering changed (variable → function → callgraph reordered) | `04-` rule fire-order assumptions, regression suite re-run | Yes if termination proof is affected |
| `04-` plugin loading model changed (decorator → registry, or vice versa) | All existing rules re-registered; `05-` FP attribution may shift | No (rules are the same; loader is different) |
| `05-` suppression lifecycle changed (grant period, review cadence) | `04-` if rule metadata schema changes; cross-link to `audit-pipelines:retention-expiry-and-rtbf` | No |
| `06-` static/runtime boundary moved (a check moves from runtime to static or vice versa) | Both sides — runtime check removed/added; `04-` rule added/removed; `05-` FP economics updated | Maybe (depends on which direction) |

A change not listed above is *not exempt*; it is evaluated against the consistency gate's affected checks. The default for ambiguity: treat as lattice-breaking unless `02-` explicitly tolerates it.

## Analyzer Tier

Every analyzer is classified during `taint-lattice-design` and recorded in `00-scope-and-targets.md`. The tier determines which artifacts are required by the consistency gate.

| Tier | Trigger | Required artifacts |
|------|---------|--------------------|
| XS | Single-rule pattern matcher (one regex, one AST shape, one verdict) | `00, 01`; `02–06` may be one-page memos |
| S | Small ruleset, single project, no taint propagation, suppressions tracked manually | XS set + `02, 04`; `05` is an inline checklist |
| M | Multi-rule analyzer with at least one dataflow rule, used across multiple projects | S set + full `02, 03, 04, 05` |
| L | Taint analyzer with multi-tier lattice, plugin extension surface, CI-blocking, ≥1000 LoC analyzer | M set + full `06`, planned `07-callgraph-construction.md`, planned `11-sarif-and-ci.md` (interim memos until v0.2.0 ships) |
| XL | Cross-module / cross-language analyzer with formal soundness/completeness claims, regulator visibility, security-bearing | L set + planned `08-cross-module-flow.md`, formal proof obligations recorded in `02-` and discharged in `03-` |

Tier is authoritative. If any sheet's guidance forces an artifact above your declared tier, that artifact becomes required — this is a tier promotion, not a waiver.

**v0.1.0 scope honesty:** L and XL tiers reference v0.2.0-planned artifacts. For now, L/XL projects should record interim positions (e.g., "callgraph: function-name resolution only; virtual-dispatch and dynamic-import sheets pending v0.2.0") in `99-` and re-gate when v0.2.0 ships.

## Routing

### Scenario: "We want a new analyzer for *X*"

1. `ast-visitation-patterns` → `01-` (pick visitor / walker / transformer based on whether you analyse, query, or rewrite)
2. `taint-lattice-design` → `02-` (define the lattice, prove monotonicity and finite height)
3. `three-phase-inference` → `03-` (order the phases, specify the worklist, prove termination)
4. `plugin-architecture-for-analyzer-rules` → `04-` (rule registry, lifecycle, conflict resolution)
5. `false-positive-economics` → `05-` (suppression lifecycle, waiver discipline, FP-rate budget)
6. `static-vs-runtime-tradeoffs` → `06-` (the boundary statement — what stays runtime, what moves static)
7. Consolidate into `99-analyzer-engineering-specification.md` and run the consistency gate.

### Scenario: "Our existing analyzer is unmaintainable; rules are ad-hoc, suppressions are out of control"

1. Reverse-engineer the implicit lattice (`taint-lattice-design`). The analyzer has one whether it acknowledges it or not. Write it down. Most pre-existing analyzers turn out to have a boolean lattice masquerading as a typed system.
2. Reverse-engineer the implicit inference order (`three-phase-inference`). Where does propagation actually happen? Across functions? Across files? Document the truth, not the marketing.
3. Triage the suppression set (`false-positive-economics`). Group by rule, by age, by waiver justification. Most "false positives" are rules with the wrong lattice; some are real bugs being silenced.
4. Decide whether to re-engineer in place (steps 4–6 of the previous scenario) or wrap the existing analyzer with a stricter post-filter that's easier to govern.

### Scenario: "We need static enforcement of an invariant that is currently runtime-only (or vice versa)"

1. Read `static-vs-runtime-tradeoffs` first (`06-`). Most "we should make this static" requests fail under cost analysis: the invariant depends on values, not types, and statics can only see types.
2. If static is genuinely tractable: `taint-lattice-design` (`02-`) for the abstract domain that captures the invariant; `three-phase-inference` (`03-`) for propagation; `plugin-architecture-for-analyzer-rules` (`04-`) for the rule.
3. If dual enforcement is the answer (runtime catches the residual, static catches the bulk): plan the v0.2.0 sheet `decorator-as-assertion-spec` interim — record the runtime guard and the static rule together, with a stated agreement contract.

### Scenario: "Suppressions are growing faster than rules"

1. `false-positive-economics` (`05-`) — the lifecycle is broken. Read the suppression lifecycle and audit trail subsections.
2. Cross-link to `axiom-audit-pipelines` for waiver-as-decision: every `# noqa: RULE` is a procedural decision and lives in the same evidence regime as governor verdicts.
3. If the rule itself is wrong, `taint-lattice-design` (`02-`) — refine the lattice rather than suppress the symptoms.

### Specialist Agents (planned for v0.2.0)

- **`agent: rule-designer`** *(planned)* — Given a desired invariant in plain English, drafts a static rule against the existing lattice and inference pipeline. Will run the rule against a test corpus and report the FP/FN profile before adoption.
- **`agent: false-positive-analyst`** *(planned)* — Reviews the suppression set for systemic issues (a single rule with disproportionate suppressions, a waiver pattern that signals lattice mis-design, expiring waivers that have no review).

For v0.1.0, these workflows run manually using the protocols in `plugin-architecture-for-analyzer-rules.md` and `false-positive-economics.md`.

### Slash Commands (planned for v0.2.0)

- `/scaffold-analyzer` — drop in a base AST visitor + rule registry + emission scaffolding (SARIF or native), aligned to the declared analyzer tier.
- `/design-tier-model` — interactive elicitation: what tiers does your trust hierarchy actually need? Output feeds `taint-lattice-design`.
- `/design-rule-set` — bootstrap a manifest + initial rule set against an existing analyzer or a fresh scaffold.

## Consistency Gate

Run before emitting `99-analyzer-engineering-specification.md`. Each check produces a pass/fail line in the gate report. Failures must be addressed or recorded as explicit waivers (with reactivation conditions); silent drops are the failure mode this pack exists to prevent.

| # | Check | Question |
|---|-------|----------|
| 1 | Tier coverage | Every artifact required by the declared tier exists. (For L/XL pre-v0.2.0, interim memos are recorded with re-gate triggers.) |
| 2 | Visitation honesty | `01-` names the visitation strategy and lists what is *not* visited (comments, types-only nodes, synthetic nodes from desugaring). "We walk the AST" without scope statement fails. |
| 3 | Lattice well-formedness | `02-` proves the abstract domain is a lattice: partial order specified, join (and meet, if used) specified, monotonicity of transfer functions stated, finite height stated (or chain-condition argued). A "lattice" with neither monotonicity nor finite height is a soup. |
| 4 | Termination proof | `03-` shows the inference terminates: lattice from `02-` has finite ascending chains, transfer functions are monotonic, worklist algorithm is specified. Hand-waving "it converges" fails. |
| 5 | Soundness/completeness statement | `02-` and `03-` together state which side the analyzer errs on (sound = no false negatives, accept some false positives; complete = no false positives, accept some false negatives; neither = engineering choice with stated rationale). "Both sound and complete" without a Rice-theorem-aware caveat fails. |
| 6 | Plugin contract | `04-` defines the rule lifecycle (load → validate → enable → fire → emit → unload), the metadata schema (id, severity, category, taxonomy alignment), and the conflict-resolution policy when two rules fire on the same node. |
| 7 | Suppression lifecycle | `05-` defines the waiver lifecycle: who can grant, for how long, with what justification, and the review/expiry mechanism. "Just add `# noqa`" without lifecycle fails. |
| 8 | FP-rate budget | `05-` states a target false-positive rate and the action triggered when it is exceeded (refine the rule, refine the lattice, retire the rule). "We minimise false positives" without a number fails. |
| 9 | Static-runtime boundary | `06-` states which invariants this analyzer enforces, which are runtime-only, and which require both. The boundary is testable — a developer reading `06-` can correctly classify a new invariant. |
| 10 | Cross-pack handoff | If `axiom-system-archaeologist` consumes this analyzer, `04-` declares the output schema. If `axiom-audit-pipelines` is in play, `05-` cross-references the waiver-as-decision lifecycle. If `ordis-security-architect` is in play, `02-` cites the threat model that motivates the lattice tiers. |
| 11 | Test corpus | At least one test corpus exists with seeded true positives and seeded true negatives for every shipped rule. Without a corpus, "the analyzer works" is an assertion, not a property. |

A `99-analyzer-engineering-specification.md` whose gate report is older than its latest numbered artifact is stale and must be re-gated before downstream citation.

## Update Workflows

| Change shape | Re-run | Re-gate |
|--------------|--------|---------|
| New rule (within existing lattice) | `04-` (registration), test corpus extended | Checks 6, 11 |
| New lattice tier | `02-`, `03-` (transfer functions), `04-` (rules consuming), `05-` (FP rebaseline), test corpus extended | Checks 3, 4, 5, 8, 11 |
| Visitation strategy migration (visitor → walker) | `01-`, `03-` (worklist seeding), `04-` (rule entry points) | Checks 2, 4 |
| Inference order change | `03-` (full re-derive), termination proof re-stated | Checks 4, 5 |
| Plugin model migration | `04-`, all rule registrations re-done | Check 6 |
| Suppression policy change | `05-`, cross-link to audit pack | Check 7 |
| Move check from runtime to static (or vice versa) | `06-`, both sides updated, test corpus extended | Check 9 |
| New downstream consumer (system-archaeologist, IDE plug-in, CI gate) | `04-` output schema versioned, planned `11-sarif-and-ci.md` interim memo | Check 10 |

Bump the `99-` semver on every re-emission. Re-gate before downstream citation.

## Stop Conditions

| Condition | Response |
|-----------|----------|
| The desired invariant depends on runtime values, not types or structure (e.g., "this string is a valid SQL identifier") | Stop. Statics can't answer this. Move to runtime; record the determination in `06-`. Do not invent a half-static rule that lies. |
| The team disagrees on what "false positive" means and the disagreement is values, not vocabulary (one party considers "rule fires on a sanitiser" a TP because the sanitiser shouldn't exist; another considers it FP) | Stop at `05-`. Resolve before tuning the rule, otherwise every refinement makes one party angrier. |
| A required cross-module / cross-language analysis is impossible at v0.1.0 scope | Record the limitation in `99-` with the v0.2.0 sheet that will address it (planned `08-cross-module-flow.md`); proceed at the lower tier; re-gate when v0.2.0 ships. |
| The proposed lattice is not actually a lattice (joins are non-commutative, or there's no top, or the order is partial-but-not-bounded) | Return to `02-`. Either fix the lattice or pick a simpler abstract domain. Do not paper over with engineering hacks; soundness depends on the algebra. |
| Suppressions are required to ship and the suppression-lifecycle sheet has not been written | Stop and write `05-` first, even minimally. Suppressions without lifecycle calcify; once they're in, the cost of imposing lifecycle later is *every PR*. |

## Decision Tree

```
Is the property statically decidable from source structure / types?
├─ No (depends on runtime values) → wrong pack; move to runtime; document in 06-
└─ Yes / partially → Continue

Are you BUILDING the analyzer or RUNNING/CONSUMING one?
├─ Running an existing analyzer (ruff, mypy, semgrep) → /python-engineering, /rust-engineering
├─ Consuming an analyzer's output for a system map → /system-archaeologist
└─ Building / extending → Continue

Pure pattern match (regex over AST, no propagation), or dataflow (taint, ownership, capability)?
├─ Pure pattern → tier XS / S; lattice may be trivial; focus on visitation + rule plugin
└─ Dataflow → tier M+; full spike (visitation + lattice + inference)

Is the analyzer security-bearing (CI-blocking, regulator-visible, control-enforcing)?
├─ No → standard tier (S/M)
└─ Yes → tier L/XL; soundness statement required; suppression lifecycle is non-negotiable

Are suppressions already accumulating in the existing tool?
├─ Yes → start at false-positive-economics (05-); the symptom is downstream of a
        lattice problem (02-) or an inference problem (03-) most of the time
└─ No → standard routing
```

## Integration with Other Skillpacks

### System archaeology (axiom-system-archaeologist)

```
axiom-system-archaeologist consumes analyzers (output → entity catalog,
  dependency graph, security surface map)
→ this pack designs analyzers (input → AST, lattice, inference, rules)
→ a coverage gap the archaeologist finds becomes a rule request to this pack
→ the analyzer extension ships; the archaeologist re-runs; the gap closes
```

The boundary: archaeologist *reads* analyzer outputs to synthesise; this pack *produces* analyzers. They are sibling, not nested. Cross-link in `04-rule-plugin-spec.md` (output schema) and in the archaeologist's intake skills (which analyzers are in scope and what they emit).

### Audit pipelines (axiom-audit-pipelines)

```
axiom-audit-pipelines: decisions are evidence; canonical bytes, fingerprint
  chains, signed exports, retention, threat model OF the log
axiom-static-analysis-engineering (this pack): suppressions are decisions

→ a `# noqa: RULE` is a procedural decision (someone decided the rule does
  not apply here, with stated rationale, at a stated time, by a stated
  actor)
→ the suppression lifecycle in 05- is the audit-pipeline lifecycle
  applied to the suppression set
→ cross-link rather than duplicate: 05- cites
  audit-pipelines:retention-expiry-and-rtbf for the waiver-expiry mechanism
```

### Security architecture (ordis-security-architect)

```
ordis-security-architect produces threat models and required invariants
→ this pack ships the analyzer that enforces them
→ the lattice tiers in 02- correspond to trust boundaries in the
  security architecture (untrusted source → sanitiser → sink)
→ the rule taxonomy in 04- aligns with the control taxonomy
  (CWE alignment, control-family mapping)
```

The boundary: security architect designs *what* must be enforced; this pack designs *how* to enforce it statically (when statically tractable). When it isn't, fall back to runtime — see `06-static-runtime-boundary.md`.

### SDLC governance (axiom-sdlc-engineering)

```
this pack produces 99-analyzer-engineering-specification.md
→ sdlc-engineering manages spec lifecycle (rule-set versioning, ADR for
  material lattice changes, retention policy of the analyzer specification
  separate from the suppressions and from the analyzer outputs)
```

### Solution architecture (axiom-solution-architect)

```
solution-architect's 04-solution-overview.md cites this pack's 99- when
  static analysis is a load-bearing control
solution-architect's adrs/ cite specific choices (lattice shape, inference
  order, plugin loading model)
solution-architect's 17-risk-register.md cites this pack's 99- for
  rule-coverage risk and false-positive-rate risk
```

### Determinism and replay (axiom-determinism-and-replay)

If your analyzer is itself part of a CI pipeline whose results must reproduce across machines (the same code at the same commit must yield the same findings on dev and CI), the analyzer is a deterministic system in the sense of `axiom-determinism-and-replay`. Most analyzers are; non-determinism in static analysis is usually iteration order over hash maps or wall-clock-keyed caches. Cross-link rather than duplicate.

## Quick Reference

| Need | Use This |
|------|----------|
| Choose visitation strategy (visitor / walker / transformer) | `ast-visitation-patterns` |
| Design the abstract domain (lattice, tiers, join) | `taint-lattice-design` |
| Order the inference and prove termination | `three-phase-inference` |
| Design the rule extension surface | `plugin-architecture-for-analyzer-rules` |
| Govern suppressions and FP-rate | `false-positive-economics` |
| Decide static vs runtime for an invariant | `static-vs-runtime-tradeoffs` |
| Run an existing analyzer | wrong pack — `/python-engineering`, `/rust-engineering` |
| Consume an analyzer's output for a system map | wrong pack — `/system-archaeologist` |
| Callgraph, cross-module flow, decorator-as-assertion, manifests, SARIF, scaling, LLM-explanation | *(planned for v0.2.0)* |
| Scaffold an analyzer, design a tier model, design a rule set (commands) | *(planned for v0.2.0)* |
| Rule-designer agent, false-positive-analyst agent | *(planned for v0.2.0)* |

## The Bottom Line

**An analyzer is an engine over a lattice. Pick the visitation, define the abstract domain with monotonicity and finite height, order the inference so it terminates, expose rules through a versioned plugin contract, govern suppressions as auditable decisions, and state the static/runtime boundary in writing. Design the spec before writing the engine; gate the spec for consistency before downstream citation. Without these, you don't have an analyzer — you have a script that occasionally agrees with you.**

---

## Static-Analysis-Engineering Specialist Skills Catalog

After routing, load the appropriate specialist sheet for detailed guidance.

**Shipped in v0.1.0:**

1. [ast-visitation-patterns.md](ast-visitation-patterns.md) — Visitor, walker, transformer; lossless vs structural ASTs; parent tracking, source-position preservation, comment handling; choice criteria
2. [taint-lattice-design.md](taint-lattice-design.md) — Lattice formalism (partial order, join, monotonicity, finite height); tier model; extension rules; the "boolean lattice masquerading as types" anti-pattern
3. [three-phase-inference.md](three-phase-inference.md) — Variable → function → callgraph; worklist algorithm; termination proof; whole-program vs incremental; cycle handling
4. [plugin-architecture-for-analyzer-rules.md](plugin-architecture-for-analyzer-rules.md) — Rule discovery, lifecycle, metadata schema, conflict resolution, deprecation, output schema versioning
5. [false-positive-economics.md](false-positive-economics.md) — Suppression vs refinement; waiver lifecycle; FP-rate budget; cross-link to audit-pipelines for waiver-as-decision
6. [static-vs-runtime-tradeoffs.md](static-vs-runtime-tradeoffs.md) — What statics can decide; the Rice-theorem ceiling; dual enforcement; cost model

**Planned for v0.2.0:**

7. `callgraph-construction.md` — Resolution strategies, virtual dispatch, dynamic imports, monomorphisation
8. `cross-module-flow-analysis.md` — Boundary semantics, summary functions, library stubs
9. `decorator-as-assertion.md` — Runtime + static dual enforcement, descriptor pattern, agreement contracts
10. `manifest-driven-configuration-with-coherence-validation.md` — YAML overlays, schema enforcement, tier consistency
11. `sarif-emission-and-ci-integration.md` — GitHub Code Scanning, exit-code semantics, suppression tracking in SARIF
12. `scaling-to-large-codebases.md` — Incremental analysis, caching, parallelism, watch mode
13. `llm-assisted-rule-explanation.md` — The pattern (rule output → LLM explanation), not the LLM
