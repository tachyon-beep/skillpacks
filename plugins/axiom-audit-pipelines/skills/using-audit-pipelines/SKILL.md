---
name: using-audit-pipelines
description: Use when a system makes procedural decisions that must be defensible after the fact — rule firings, governor decisions, state transitions, gate verdicts, eligibility determinations, automated approvals — and you need a verifiable provenance chain rather than ordinary observability. Routes through canonical encoding, decision-log architecture, fingerprint chains, signed exports, immutable storage, retention with right-to-be-forgotten, partial replay, and performance budgets.
---

# Using Audit Pipelines

## Overview

**Procedural decisions are first-class artifacts. Every decision has a verifiable provenance chain.**

This pack treats *the decision* — not the log line that mentions it — as the unit of evidence. A decision-log entry binds together inputs, the rule or policy version, the code version, the output, and a timestamp; entries are canonically encoded, fingerprinted, and chained so that tampering, replay, partial deletion, or time-shift becomes detectable. Exports carry their own integrity proofs so a downstream consumer can verify a subset without re-trusting the producer.

This is the *evidence* counterpart to security architecture's *controls*:

- `ordis-security-architect` answers **what must be protected and how** — threat models of the system, controls at trust boundaries, defense-in-depth design.
- **`axiom-audit-pipelines` (this pack) answers what proves a decision happened, with what inputs, under which ruleset version and code version** — and how that proof survives review, export, partial deletion, and adversarial scrutiny.
- The **threat model OF the audit log itself** lives in this pack (because the log is the evidence, and adversaries who reach the log are attacking the evidence). System-level threat modelling stays in `ordis-security-architect`. Cross-link the two; do not duplicate STRIDE tables.

## When to Use

Use this pack when:

- A system component decides something a regulator, auditor, customer, or court might later ask "prove it" about.
- Procedural decisions are produced by a rule engine, policy engine, governor, scheduler, or workflow engine, and the question "why did it do that?" must be answerable months later.
- You need to export a subset of decisions to an auditor, downstream system, or compliance pipeline *without breaking integrity proofs over the rest*.
- A regulator imposes retention, redaction, or right-to-be-forgotten obligations that conflict naively with append-only storage.
- You need to replay a state machine from a partial trail to investigate an incident, reconstruct a customer's experience, or test a fix against historical inputs.
- The team writes "we'll just log it" and you can already see the audit going wrong.

Do **not** use this pack when:

- You are designing system controls or threat-modelling the system → `/security-architect`
- You are debugging non-determinism in a simulator or replay-for-debugging → simulation-foundations / determinism guidance (replay-for-debugging is a different problem from replay-from-audit-trail; this pack handles the latter)
- You are designing application logging for ordinary observability (latency, errors, throughput) — read `audit-aware-logging-vs-observability.md` to understand why those are different problems, then use ordinary observability tooling
- You only need a tamper-evident commit log for engineers (use git, signed commits, and CI provenance)

## Start Here

If your input is a system that already makes decisions and you have not run this pack before:

1. Read `decision-log-architecture.md` — define what counts as a decision in *this* system, choose entry shape, emit `00-scope-and-decisions.md` and `01-decision-log-schema.md`.
2. Read `canonical-encoding-for-fingerprinting.md` — pick the canonicalisation form (RFC8785 JCS unless you can defend an alternative), enumerate float / map / timezone gotchas, emit `02-canonical-encoding-spec.md`.
3. Read `fingerprint-chains-and-integrity.md` — choose chain construction (linked hash vs Merkle), recovery-from-gaps strategy, partial-trust verification model, emit `03-chain-and-integrity-spec.md`.
4. Use the **Routing** section below to walk the rest at the right depth for your tier.
5. Run the **Consistency Gate** below before declaring `99-audit-pipeline-specification.md` ready.

Steps 1–3 are the spike: if those three artifacts hold together, the rest is fill-in. If they don't, no later sheet can save you.

## How to Access Reference Sheets

All reference sheets are in the same directory as this `SKILL.md`. When you see a link like `[canonical-encoding-for-fingerprinting.md](canonical-encoding-for-fingerprinting.md)`, read the file from the same directory.

## Pipeline Position

```
ordis-security-architect (controls)         axiom-audit-pipelines (evidence)
  threat-models the SYSTEM         ←-cross-ref-→   threat-models the LOG
  designs controls at boundaries                   designs canonical encoding,
                                                   chain, signing, retention
                                                   replay, perf budget
  ─────────────────────────────────────────────────────────────────────
                            ↓
                axiom-solution-architect picks up:
                  17-risk-register.md cites threats from BOTH packs
                  04-solution-overview.md cites the audit-pipeline spec
                  ADRs cite chain construction, canonical-encoding choice
```

## Expected Artifact Set

The pack produces a numbered artifact set in an `audit-pipeline/` workspace:

| # | Artifact | Producer skill |
|---|----------|----------------|
| 00 | `scope-and-decisions.md` | `decision-log-architecture` |
| 01 | `decision-log-schema.md` | `decision-log-architecture` |
| 02 | `canonical-encoding-spec.md` | `canonical-encoding-for-fingerprinting` |
| 03 | `chain-and-integrity-spec.md` | `fingerprint-chains-and-integrity` |
| 04 | `signing-and-export-spec.md` | `signing-and-export-integrity` |
| 05 | `provenance-bindings.md` | `decision-provenance` |
| 06 | `storage-and-retention.md` | `immutable-storage-patterns` + `retention-expiry-and-rtbf` |
| 07 | `threat-model.md` | `threat-model-for-audit-logs` |
| 08 | `replay-capability.md` | `partial-replay-from-trail` |
| 09 | `audit-vs-observability-boundary.md` | `audit-aware-logging-vs-observability` |
| 10 | `performance-budget.md` | `performance-budget-for-audit-grade-pipelines` |
| 99 | `audit-pipeline-specification.md` | router-owned consolidation (this SKILL.md) |

## Spec Dependency Graph

The numbered artifacts are not independent — changes propagate. Read this before editing any spec.

```
02-canonical-encoding-spec.md       (the bytes)
        │
        ▼
01-decision-log-schema.md           (what fields, in what canonical bytes)
        │
        ▼
03-chain-and-integrity-spec.md      (hash over those bytes, chain construction)
        │
        ▼
04-signing-and-export-spec.md       (signature over hash, export format)
        │
        ▼
05-provenance-bindings.md           (input/ruleset/code closure over the entry)
        │
        ▼
08-replay-capability.md             (replay reads canonical bytes by version)
```

**Coordinated re-emission rules:**

| If you change | You also re-emit | Chain-breaking? |
|---------------|------------------|------------------|
| `02-` canonicalisation rule | `01-` (`entry_version` bump) and `03-` (chain-breaking event note) | Yes |
| `01-` mandatory field added/removed/renamed | `02-` if encoding rule affected, `03-` (chain-breaking event note) | Yes |
| `03-` hash function changed | `01-` if `inputs_commitment.ref.hash` was that function, `04-` signature-over-hash recomputation | Yes |
| `03-` chain construction changed (linked-hash → Merkle) | `04-` export format, `08-` replay traversal | Yes (with documented transition) |
| `04-` signature scheme changed | `06-` if signature material is held in storage, `07-` adversary update | No (chain-side unaffected) |
| `06-` retention class redrawn | `07-` partial-deletion threat, `08-` replay coverage shrinks | No |

A change not listed above is *not exempt*; it gets evaluated against the chain-breaking-event rule in `03-` and against the consistency gate's affected checks. The default for ambiguity is "treat as chain-breaking" — it is cheap to over-document a transition and expensive to under-document one.

## Scope Tier

Every workflow is classified during `decision-log-architecture` and recorded in `00-scope-and-decisions.md`. The tier determines which artifacts are required by the consistency gate.

| Tier | Trigger | Required artifacts |
|------|---------|--------------------|
| XS | Single decision type, single producer, no export, internal-only | `00, 01, 02, 03, 06`; `99` is a one-page memo |
| S | Multiple decision types under one ruleset, no cross-system export | XS set + `04, 07, 09` |
| M | Decisions consumed by a downstream system or persisted across deploys | S set + `05, 08, 10` |
| L | Decisions are subject to regulatory audit, external export, or legal hold | M set + full `07-threat-model.md` (not summary), explicit cryptographic agility plan in `04-`, documented incident-response runbook in `06-` |
| XL | Decisions feed an external regulator's pipeline, ATO/RMF process, or court evidence | L set + signed-export key-rotation runbook, third-party verifier compatibility statement, retention-vs-RTBF reconciliation memo with named legal authority |

Tier is authoritative. If any sheet's guidance forces an artifact above your declared tier, that artifact becomes required — this is a tier promotion, not a waiver.

## Routing

### Scenario: "We have a system that makes decisions; design its audit pipeline"

1. `decision-log-architecture` → `00-`, `01-` (also classifies the tier)
2. `canonical-encoding-for-fingerprinting` → `02-`
3. `fingerprint-chains-and-integrity` → `03-`
4. `signing-and-export-integrity` → `04-` (S tier and above)
5. `decision-provenance` → `05-` (M tier and above)
6. `immutable-storage-patterns` + `retention-expiry-and-rtbf` → `06-`
7. `threat-model-for-audit-logs` → `07-` (always at least summary; full document at L+)
8. `partial-replay-from-trail` → `08-` (M tier and above)
9. `audit-aware-logging-vs-observability` → `09-` (the delineation sheet — write last)
10. `performance-budget-for-audit-grade-pipelines` → `10-` (M tier and above)
11. Consolidate into `99-audit-pipeline-specification.md` and run the consistency gate below.

### Scenario: "Verify the integrity of an existing trail"

Use the `/verify-integrity` slash command, which dispatches the `integrity-auditor` agent against an existing trail directory.

### Scenario: "Review a system for missing audit instrumentation"

Use the `audit-architecture-reviewer` agent. Provide the system's design artifacts (a SAD, an HLD, or a code map) and it reports decision points lacking provenance, with severity.

### Scenario: "Bootstrap audit-trail scaffolding for a new component"

Use the `/scaffold-audit-trail` slash command. It drops in canonical-encoding, chain, and storage scaffolding aligned to your declared tier; it does not replace the design artifacts above — it implements them.

### Scenario: "I'm not sure what counts as a decision in our system"

Use the `/design-decision-log` slash command. It runs an interactive elicitation: which producers, which decision types, which fields are mandatory, where the boundary sits. Output feeds `decision-log-architecture` step 1.

### Specialist Agents

- **`agent: audit-architecture-reviewer`** — Reviews a system for decision points lacking provenance. Reads design artifacts, reports gaps with severity. Invoked via `/scaffold-audit-trail`'s gap-analysis option or directly via the `Task` tool.
- **`agent: integrity-auditor`** — Runs verification on a real trail and reports gaps, signature mismatches, chain breaks, and recovery scope. Invoked via `/verify-integrity`.

**Agents vs. skills:** Skills *produce* design artifacts (the spec). Agents *audit or critique* either a design or a running trail. Load a skill when designing; dispatch an agent when reviewing.

## Consistency Gate

Run before emitting `99-audit-pipeline-specification.md`. Each check produces a pass/fail line in the gate report. Failures must be addressed or recorded as explicit waivers (with reactivation conditions); silent drops are the failure mode this pack exists to prevent.

| # | Check | Question |
|---|-------|----------|
| 1 | Tier coverage | Every artifact required by the declared tier exists. |
| 2 | Schema completeness | Every field promised in `01-decision-log-schema.md` is producible by the system as described in `00-scope-and-decisions.md`. No "TBD" in mandatory fields. |
| 3 | Encoding determinism | `02-` names the canonicalisation form (RFC8785 JCS or alternative with rationale), and addresses every gotcha class enumerated in `canonical-encoding-for-fingerprinting.md`: floats, map ordering, timezones, Unicode normalization, integer width. |
| 4 | Chain integrity | `03-` specifies chain construction, gap-recovery procedure, partial-trust verification, and clock dependence — and `04-` (if present) is consistent with `03-` on signature granularity. |
| 5 | Provenance closure | `05-` (M+) closes the loop: every output in the schema can be traced to its inputs, ruleset version, and code version. No dangling outputs. |
| 6 | Retention / RTBF reconciliation | `06-` resolves the apparent contradiction between append-only and right-to-be-forgotten with a named mechanism (cryptographic erasure, redaction-with-witness, segregated PII, or legal-authority-recorded waiver). The mechanism is named, not hand-waved. |
| 7 | Threat coverage | `07-` covers the standard adversaries (tamper, replay, time-shift, partial deletion, key compromise, insider). Each has a control or an explicit accepted-risk entry. |
| 8 | Replay scope honesty | `08-` (M+) states what *can* and *cannot* be replayed from N entries, and where state outside the trail is required. No "fully replayable" claims without a state-coverage proof. |
| 9 | Boundary clarity | `09-` distinguishes audit-grade events from ordinary observability events with a stable rule, not a slogan. The rule is testable: a developer reading it can correctly classify a new event class. |
| 10 | Performance honesty | `10-` (M+) states amortisation strategy, write/read budgets, and the pipeline's behaviour under burst load. "Stream everything" or "batch everything" without numbers fails the check. |
| 11 | Cross-pack handoff | If `ordis-security-architect` artifacts exist for the same system, `07-` cross-references the system threat model rather than duplicating it. If `axiom-solution-architect` artifacts exist, `99-` is cited from `04-solution-overview.md` and from any ADR that touches the audit subsystem. |

A `99-audit-pipeline-specification.md` whose gate report is older than its latest numbered artifact is stale and must be re-gated before downstream citation.

## Update Workflows

| Change shape | Re-run | Re-gate |
|--------------|--------|---------|
| New decision type | `00-`, `01-`, `05-` cross-ref | Checks 1, 2, 5 |
| New field on existing decision | `01-`, `02-` if encoding rule affected, `08-` if replay-relevant | Checks 2, 3, 8 |
| Canonicalisation form change | `02-`, `03-` (chain re-verification on legacy entries), `04-` | Checks 3, 4 |
| Chain construction change | `03-`, `04-`, `08-`, migration plan in `99-` | Checks 4, 8 |
| Key rotation event | `04-`, `06-` if rotation drives storage layout, `07-` adversary update | Checks 4, 6, 7 |
| Retention policy change | `06-`, `08-` (replay coverage shrinks), `07-` (partial-deletion threat) | Checks 6, 7, 8 |
| New regulator obligation | `00-` tier may promote, `06-`, `07-` | Re-gate at promoted tier |
| New downstream consumer of exports | `04-`, `05-` (exposure surface), `09-` (boundary may shift) | Checks 4, 5, 9 |

Bump the `99-` semver on every re-emission. Re-gate before downstream citation.

## Stop Conditions

| Condition | Response |
|-----------|----------|
| The system makes no decisions a regulator/auditor/customer would ask "prove it" about | Stop. The system needs ordinary observability, not an audit pipeline. Record the determination in a one-page memo. |
| Decisions are too low-level to log atomically (per-tick simulation, per-microsecond control loop) | Aggregate to the meaningful decision boundary (the trade, the verdict, the dispatch) and log there. Log the aggregation rule. Do not log every tick. |
| Append-only conflicts with regulator's RTBF requirement and legal authority is unclear | Stop at `06-`. Escalate to legal/compliance owner. Do not improvise a redaction scheme. |
| Required cryptographic agility (post-quantum readiness) is unsettled | Record current choice in `04-` with explicit `crypto-agility-plan: deferred-to-vN+1`, raise risk in `07-`, proceed. Do not block on speculative cryptography. |
| Performance budget is impossible at the proposed entry shape | Return to `01-` and consider summary entries with on-demand detail expansion, or per-class entry shapes. Do not silently drop entries to fit. |

## Decision Tree

```
Does the system make decisions someone might later need to verify?
├─ No → wrong pack; use ordinary observability
└─ Yes → Continue

Are decisions exposed to a regulator, auditor, or external party?
├─ No → tier XS or S likely
└─ Yes → tier M, L, or XL — full pipeline required

Is there an existing trail you need to verify, not design?
├─ Yes → /verify-integrity (integrity-auditor agent)
└─ No → Continue

Is there a system design you suspect is under-instrumented?
├─ Yes → audit-architecture-reviewer agent (gap report)
└─ No → /design-decision-log (interactive) → decision-log-architecture skill
```

## Integration with Other Skillpacks

### Security architecture (ordis-security-architect)

```
ordis produces system threat model + control set
→ this pack reads them; threat-model-for-audit-logs cross-references rather than duplicates
→ this pack produces 07-threat-model.md (the log itself)
→ ordis can then add controls protecting log integrity (HSM, write-once media, signing service)
```

The boundary: ordis owns *system* threat modelling and control design; this pack owns *audit-log* threat modelling and the design of the evidence itself. They cross-link.

### Solution architecture (axiom-solution-architect)

```
solution-architect's 04-solution-overview.md cites this pack's 99-
solution-architect's adrs/ cite specific choices (RFC8785, chain shape, signature scheme)
solution-architect's 17-risk-register.md cites threats from both this pack's 07- and ordis' threat model
```

If solution-architect is in play and the system has procedural decisions, this pack's `99-` is a normal input to `04-` and to ADRs.

### Determinism and replay (axiom-determinism-and-replay)

```
axiom-determinism-and-replay designs replayable systems
  (seeds, RNG isolation, snapshots, divergence detection)
→ this pack designs the audit trail of decisions made by such systems
→ replay-from-audit (this pack's 08-) reconstructs decisions from the trail
→ replay-for-debugging (determinism pack) reconstructs system state from inputs
→ different questions, different scopes, cross-link in 99-
```

The boundary: this pack records what was decided; the determinism pack arranges that, given recorded inputs, the system would behave the same way again. A system that uses both packs has *evidence* of decisions and *reproducibility* of behaviour, and the consistency between them is itself an audit-grade property.

### SDLC governance (axiom-sdlc-engineering)

```
this pack produces 99-audit-pipeline-specification.md
→ sdlc-engineering manages spec lifecycle (versioning, ADR for material changes,
  retention policy of the spec itself separate from retention of the trail)
```

### Compliance frameworks

When a framework drives the requirement (SOC 2, HIPAA, PCI-DSS, GDPR, EU AI Act, NIS2), record the framework as the originating constraint in `00-scope-and-decisions.md`. Trace it through `01-` (which fields the framework requires), `06-` (retention obligations), and `07-` (the compliance-exposure threat).

### Observability stack (separate skill)

Audit-grade pipelines and ordinary observability share infrastructure (storage, dashboards, alerts) but are different problems. Read `09-audit-vs-observability-boundary.md` for the rule. Common failure mode: cramming audit obligations into the observability stack, or routing observability noise through the audit pipeline. Either kills both.

## Quick Reference

| Need | Use This |
|------|----------|
| Define what counts as a decision | `decision-log-architecture` |
| Pick canonical encoding (RFC8785 etc.) | `canonical-encoding-for-fingerprinting` |
| Construct the chain (linked vs Merkle) | `fingerprint-chains-and-integrity` |
| Sign exports / rotate keys / partial export | `signing-and-export-integrity` |
| Bind decision to inputs + ruleset + code | `decision-provenance` |
| Append-only storage at rest | `immutable-storage-patterns` |
| Distinguish audit from observability | `audit-aware-logging-vs-observability` |
| Threat-model the audit log itself | `threat-model-for-audit-logs` |
| Reconcile retention with right-to-be-forgotten | `retention-expiry-and-rtbf` |
| Replay state from N entries | `partial-replay-from-trail` |
| Decide stream vs amortise | `performance-budget-for-audit-grade-pipelines` |
| Verify an existing trail | agent: `integrity-auditor` (`/verify-integrity`) |
| Find decision points lacking provenance | agent: `audit-architecture-reviewer` |
| Drop in scaffolding for a new component | command: `/scaffold-audit-trail` |
| Interactive: what counts as a decision | command: `/design-decision-log` |

## The Bottom Line

**Decisions are evidence. Evidence is canonical bytes, chained fingerprints, signed exports, named retention, modelled adversaries, replayable scope, and budgeted performance — produced as a specification before code is written, gated for consistency before citation downstream.**

---

## Audit-Pipelines Specialist Skills Catalog

After routing, load the appropriate specialist sheet for detailed guidance:

1. [decision-log-architecture.md](decision-log-architecture.md) — What is a decision, mandatory fields, entry boundary, scope classification
2. [canonical-encoding-for-fingerprinting.md](canonical-encoding-for-fingerprinting.md) — RFC8785 JCS, deterministic JSON, gotchas (floats, maps, timezones, Unicode, integer width)
3. [fingerprint-chains-and-integrity.md](fingerprint-chains-and-integrity.md) — Linked-hash and Merkle constructions, gap recovery, partial-trust verification
4. [signing-and-export-integrity.md](signing-and-export-integrity.md) — HMAC and asymmetric signing, key rotation, signature granularity, exporting subsets without breaking proofs
5. [decision-provenance.md](decision-provenance.md) — Linking outputs to inputs + ruleset version + code version
6. [immutable-storage-patterns.md](immutable-storage-patterns.md) — Append-only, encryption-at-rest (SQLCipher and alternatives), write-once-verify-many
7. [audit-aware-logging-vs-observability.md](audit-aware-logging-vs-observability.md) — The boundary rule, what crosses, what doesn't
8. [threat-model-for-audit-logs.md](threat-model-for-audit-logs.md) — Tamper, replay, time-shift, partial deletion, key compromise, insider, regulator subpoena
9. [retention-expiry-and-rtbf.md](retention-expiry-and-rtbf.md) — Cryptographic erasure, redaction-with-witness, segregated PII, legal-authority waivers
10. [partial-replay-from-trail.md](partial-replay-from-trail.md) — Reconstructing state from N entries, replay scope, state-coverage proofs
11. [performance-budget-for-audit-grade-pipelines.md](performance-budget-for-audit-grade-pipelines.md) — Stream-vs-amortise tradeoff, burst behaviour, write/read/verify budgets
