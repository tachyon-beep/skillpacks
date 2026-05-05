---
name: false-positive-economics
description: Use when an analyzer is shipping or has shipped and you face the operational reality of false positives — suppressions accumulating, developer trust eroding, or the question "should we refine the rule or just suppress?" Covers suppression-vs-refinement decision making, the waiver lifecycle (granted, reviewed, expired, re-granted), the FP-rate budget, and the cross-link to audit-pipelines for waiver-as-decision. Produces `05-false-positive-economics.md`.
---

# False-Positive Economics

## Why "Just Suppress It" Loses

Every false positive is a tax. The tax is paid in three currencies:

- **Developer attention** — every `# noqa` requires reading, deciding, justifying. The cost compounds because suppressions live forever by default.
- **Signal-to-noise** — once developers learn to ignore the analyzer ("oh, it always cries about that"), they ignore the *true* positives too. The analyzer's value collapses asymptotically.
- **Suppression rot** — a `# noqa: STA001` from 2022 is a signed assertion that the rule was wrong on this line. The code, the rule, and the lattice have all changed since. The assertion is almost certainly stale; nobody knows which way.

The economic question is not "how do we eliminate false positives" — Rice's theorem ensures you can't, for any non-trivial property — but "what is the false-positive *budget*, and what is the discipline that keeps the budget intact as the analyzer evolves?"

`05-false-positive-economics.md` is where that budget and discipline live.

## The Suppression-vs-Refinement Decision

When a rule fires on code the team believes is correct, there are exactly two responses:

### Suppress

Add a `# noqa: STA001` (or analyzer-equivalent), record the justification, move on. Cheap per-incident. Doesn't change the rule.

**Right when:**

- The code really is an exception — a one-off escape hatch in a well-defined location.
- The rule is correct in general; this site is anomalous for a documented reason.
- Refining the rule to recognise this case would require encoding a fact that's not statically tractable (a runtime invariant the developer knows holds).

**Wrong when:**

- The same suppression is needed in many places — the rule is over-firing systemically.
- The justification is "this is fine because [reason that should be in the lattice]" — the lattice is mis-modelling the problem.
- The justification is "we'll come back to this" — it will never get fixed.

### Refine

Improve the rule (or the lattice it sits on) so the false positive disappears for everyone. Expensive per incident. Permanent fix.

**Right when:**

- The same suppression appears in many places.
- The "correct" pattern the rule misses is recognisable from the AST or the lattice with modest extension.
- The lattice has the wrong tier set or wrong sanitisation model — the rule is downstream of a `02-` problem.

**Wrong when:**

- The fix would require more lattice expressiveness than the analyzer can support (Rice).
- The "correct pattern" is project-specific (then it lives in project config or a custom rule, not the shared engine).
- The refinement would harm precision elsewhere (false positive count drops in this codebase but rises in others).

### The Decision Heuristic

A useful rule of thumb:

| Suppression count for this rule | Action |
|----------------------------------|--------|
| 1–2 across the codebase | Suppress; record |
| 3–10 | Investigate; usually refine |
| 11+ | Refine, or retire the rule. The signal is that the rule does not match the codebase's reality |

Numbers are rules of thumb, not law; the underlying signal is that suppressions follow a power law per rule. A rule with 1 suppression and a rule with 100 suppressions are categorically different problems.

## The Waiver Lifecycle

A suppression without a lifecycle is technical debt that accrues compound interest. Treat every suppression as a **waiver**: a time-bound permission, granted by a named actor, with a stated justification, with a review trigger.

```
                granted
                   │
                   ▼
              ┌────────┐
              │ active │ ─────── reviewed (no change) ─────┐
              └────────┘                                    │
                   │                                        │
          expired (time limit hit)                          │
                   │                                        │
                   ▼                                        │
             ┌──────────┐                                   │
             │ expired  │ ◄─────────────────────────────────┘
             └──────────┘
                   │
            (forced review)
                   │
              ┌────┴────┐
              │         │
              ▼         ▼
         re-granted   removed
           (active)   (rule no longer suppressed; either fix code or refine rule)
```

**The discipline:**

- **Every waiver has a granter** — a person, a team, or an automated process whose authority covers the rule and the location. Anonymous waivers are worse than no waiver.
- **Every waiver has a justification** — free text is fine; the requirement is that *some* text exists. "FP" is not a justification; "html_escape applied at line 12; rule cannot see the call due to dynamic dispatch" is.
- **Every waiver has an expiry** — by policy. 90 days, 180 days, end-of-quarter — your choice. The CI fails on expired waivers; the dev re-justifies (extends, re-grants), refines (removes the waiver because the rule changed), or retires the code.
- **Every waiver is auditable** — see the audit-pipelines cross-link below.

**Suppressions without these properties are just `# noqa` comments**, and they will outlive the rule, the developer, and (often) the codebase.

## Cross-Link: Suppressions ARE Decisions

A `# noqa: STA001 — granted by alice@team, 2025-09-12, expires 2026-03-12, justification: "html_escape applied at line 12; rule cannot see call due to dynamic dispatch"` is a procedural decision. It belongs in the same evidence regime as a governor verdict or a rule firing.

This means:

- The suppression set is an **append-only log** with the same canonical-encoding and fingerprint discipline as any other audit log. See `axiom-audit-pipelines:decision-log-architecture.md` for the architecture; this sheet provides the suppression-specific lifecycle.
- The **expiry mechanism** is the audit pack's `retention-expiry-and-rtbf.md`: the lifecycle says when a waiver can be erased (after expiry + review), what the cryptographic erasure or redaction-with-witness mechanism is, and who has the authority.
- The **provenance** of a waiver is the audit pack's `decision-provenance.md`: the granter, the justification, the rule version, the lattice version, and the analyzer version that were in scope when the waiver was granted.

For a small team with a small ruleset, you can run the waiver lifecycle in version control (the `# noqa` comment + a CI check that parses justifications and checks expiries). For a regulated environment with ATO/RMF obligations, the suppression set should be in the audit pipeline proper.

`05-` declares which level applies and cross-references the audit pack accordingly.

## The False-Positive-Rate Budget

Every analyzer needs a target FP rate and an action triggered when the rate is exceeded.

**A workable budget statement** (for the consistency gate, check 8):

```
Target false-positive rate per rule: ≤ 5% (95th-percentile precision)
Measurement: rolling 30-day window, computed against the test corpus
   plus a sample of in-the-wild findings reviewed by the rule owner

Action when budget is exceeded for a rule:
   1. Investigate: is it the rule, the lattice, or the test corpus?
   2. Refine: rule, lattice, or both. New version bumps `99-` semver.
   3. Retire: if refinement is impossible, deprecate the rule (see
      `04-rule-plugin-spec.md` deprecation flow).
   4. Suppress is not an action at the budget level; suppressions are
      per-site, not per-rule.

Action when budget is exceeded analyzer-wide:
   - Stop adding new rules. The analyzer is paying interest on existing
     debt. Spend a release on refinement before any new rule lands.
```

The numbers are illustrative; pick yours. The **structure** is non-negotiable: target, measurement, action.

**Measurement is the hard part.** "False positive rate" requires a ground truth, which requires a test corpus, which is part of the consistency gate. Without a corpus, FP rate is rhetoric. The corpus is built and maintained by rule authors as part of the metadata contract (see `04-` `examples_violation` / `examples_clean`).

## Two Anti-Patterns That Look Like Discipline

### "Aggressive suppression" (the broken-window mode)

The team learns that the analyzer is noisy and develops a habit: ship the change, suppress whatever fires, move on. Justifications devolve to "FP" or omitted entirely. CI passes. Suppressions accumulate exponentially.

**Symptom:** suppression count grows faster than rule count.
**Cause:** no lifecycle, no budget, no visibility on suppression rate.
**Fix:** institute lifecycle (this sheet); set a budget; report suppression rate as a first-class CI metric alongside finding rate.

### "Aggressive refinement" (the chasing-zero mode)

The team treats every false positive as an emergency: refine the rule, refine the lattice, add special cases. Rules accumulate complexity; the lattice grows tiers without justification; precision improves marginally; soundness erodes.

**Symptom:** rule code grows linearly with suppression incidents; lattice tier count grows without lattice version bumps.
**Cause:** no acknowledgement of the Rice ceiling; no cost model for refinement.
**Fix:** budget includes a refinement cost model (developer hours per FP-rate-percentage-point); recognise that some FPs are economic to suppress.

The right discipline is **a budget with both directions**: suppress some, refine some, retire some, all governed by lifecycle.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Suppressions without expiry | Rules dying under accumulated `# noqa`s | Mandatory expiry; CI fails on expired |
| "FP" as a justification | No information for future review | Mandatory free-text justification; CI fails on empty |
| No suppression-rate metric | The discipline only fires on-the-day | Report suppression rate per release alongside finding rate |
| Refinement without test corpus | Refinement is anecdotal; regressions slip in | Mandatory test corpus from rule metadata; CI runs it |
| Refining in response to single FPs | Rule bloat; lattice bloat | Apply the 1–2 / 3–10 / 11+ heuristic; lone FPs suppress |
| Suppressions tracked in version control only, then promoted to "real audit" later | Lifecycle reset; old waivers don't carry over | Decide tier in `05-` upfront; if growing into audit pipeline, design for the migration |
| Aggressive-zero culture | Lattice and rules accumulate special cases until incomprehensible | Acknowledge Rice; budget the refinement cost; some FPs survive |
| Aggressive-suppress culture | Analyzer is ceremonial within a year | Treat suppression rate as a first-class metric; investigate growth |
| No conflict between this sheet and `04-` deprecation flow | A rule deprecates while waivers point at it | The deprecation flow's orphan-suppression diagnostic is the cross-link; check that it's wired |

## The Decision Output (`05-false-positive-economics.md`)

A complete `05-` answers:

1. **Suppression-vs-refinement policy** — the heuristic; who decides; what triggers a refinement instead of a suppression.
2. **Waiver lifecycle** — granted-by-whom, justification format, expiry policy, review trigger, re-grant flow, removal flow.
3. **Suppression storage** — version control comments, manifest file, audit pipeline; what's authoritative, what's derived.
4. **FP-rate budget** — target rate, measurement methodology, action on breach (per rule and analyzer-wide).
5. **Suppression-rate budget** — separate from FP rate; tracks growth in the suppression set.
6. **Test corpus discipline** — who maintains it; what triggers an entry; CI integration.
7. **Cross-link to audit-pipelines** — explicit pointer to which sheets in that pack apply, at what tier.
8. **Deprecation interaction** — what happens to waivers when a rule deprecates (see `04-`).
9. **Reporting** — suppression rate, FP rate, expiry queue, all visible in a dashboard or release report.
10. **Escalation** — when a rule's FP rate or suppression rate exceeds threshold without resolution, who is notified, what authority can retire the rule.

## Cross-References

- `taint-lattice-design.md` — many "false positives" are actually lattice imprecision; refine `02-` before the rule
- `three-phase-inference.md` — over-approximation in any phase shows up as FP; check for missing callgraph edges or stale summaries before blaming the rule
- `plugin-architecture-for-analyzer-rules.md` — deprecation flow consumes this sheet's waiver lifecycle (orphan-suppression diagnostic)
- `static-vs-runtime-tradeoffs.md` — some FPs are "this can't be statically tractable" — fall back to runtime, document the limit
- Cross-pack: `axiom-audit-pipelines:decision-log-architecture` — the suppression set as a decision log
- Cross-pack: `axiom-audit-pipelines:retention-expiry-and-rtbf` — the cryptographic-erasure / redaction-with-witness mechanism for expired waivers
- Cross-pack: `axiom-audit-pipelines:decision-provenance` — what binds a waiver to the rule version, lattice version, analyzer version it was granted under
- Cross-pack: `axiom-sdlc-engineering:governance-and-risk` — escalation authority for unresolved high-FP rules
