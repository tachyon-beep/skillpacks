# Decision-Log Architecture

## Overview

**A decision-log entry is the *decision* — not the line that mentions it. It binds inputs, ruleset version, code version, output, and time into one canonically-encoded artifact that the rest of the pipeline (chain, signing, replay) operates on.**

Most audit failures begin here, before any cryptography is wrong: the log records *that something happened* but not *why*, so months later no one can answer the auditor. This sheet defines what counts as a decision in the system, what fields are mandatory, where the boundary sits, and how the entry shape is versioned without invalidating older entries.

## When to Use

Use this sheet when:

- A system has rule firings, governor verdicts, scheduler dispatches, eligibility determinations, gate checks, automated approvals, policy evaluations, or workflow transitions that someone might later be asked to justify.
- The team is reaching for "we'll just add a log line" and you need a richer artifact.
- The system already has logging but cannot answer "why did it decide X for input Y on date Z?"
- You are starting `axiom-audit-pipelines` and need to emit `00-scope-and-decisions.md` and `01-decision-log-schema.md`.

## Core Principle

> The unit of evidence is the decision, not the event. An event is "the system did something." A decision is "the system chose between alternatives, given inputs, against a ruleset, in a code version, at a time." The latter is what an auditor asks about.

## What Counts as a Decision

A decision is anything where, given the same inputs and ruleset, you should produce the same output, and a third party might later ask you to demonstrate that.

Concrete examples (generic):

- A policy engine evaluates a request and emits *allow / deny / step-up*.
- A risk model evaluates a transaction and emits *approve / decline / review*.
- A workflow engine evaluates conditions and transitions a case from state A to state B.
- A scheduler chooses a worker for a task.
- A pricing engine emits a quote.
- A content-moderation classifier emits a verdict.
- A governor or guard accepts or rejects a proposed action from another component.
- A rule engine fires a rule and produces a derived fact.

Counter-examples — these are *events*, not decisions:

- A request arrived. (No alternative was chosen; record the decision the request triggered.)
- A row was written. (Which decision required the write?)
- A button was clicked. (Belongs in product analytics, not the audit log.)
- A heartbeat fired. (Observability.)
- A timer expired. (Unless the expiry itself is the decision — "the cooling-off period elapsed therefore the application auto-approves" — log the resulting decision, not the timer.)

The test: *can someone defensibly disagree with the output and demand to see the reasoning?* If yes, it is a decision and belongs in the audit pipeline. If no, it belongs in observability.

## Decision Boundary — Where to Cut

The hardest design choice in this pack. Cut too fine and you log every internal step (per-rule firing inside an evaluation); cut too coarse and you lose the why (only the final verdict, not which rule produced it). Three useful patterns:

### Pattern A — One entry per externally-observable decision

The boundary is "what an outside observer sees." Internal rule firings collapse into one entry that names the rule(s) that contributed. Best for: customer-facing decisions, regulatory filings, where the unit of accountability is the externally-visible verdict.

**Example shape:** one entry per request/response cycle through a policy engine; rule firings appear as a structured `evidence` array within the entry.

### Pattern B — One entry per atomic rule firing, plus one summary entry

Each individual rule firing becomes an entry; an additional summary entry binds them together. Best for: complex rule engines where individual firings have independent legal or analytical interest, and where the chain of reasoning itself is auditable.

**Cost:** roughly N+1 entries per decision; chain growth and verifier cost scale with N. Address in `performance-budget-for-audit-grade-pipelines.md`.

### Pattern C — Hybrid with reference IDs

External-decision entries carry a `reasoning_ref` to a separate, append-only reasoning log of rule firings. The reasoning log uses its own (cheaper) integrity scheme. Best for: high-throughput systems where most decisions are non-controversial and only a fraction get audited deeply.

**Cost:** the reasoning log is now a second pipeline with its own threat model and retention policy. Document explicitly.

A pipeline can use different patterns for different decision types — a content-moderation verdict on user-visible content might be Pattern A while internal scheduler decisions are Pattern C. Record the choice per decision type in `00-scope-and-decisions.md`.

## Mandatory Fields

Every entry MUST carry these fields. They are non-negotiable; missing fields fail Check 2 of the consistency gate.

| Field | Type | Purpose |
|-------|------|---------|
| `entry_id` | UUIDv7 or ULID | Unique, time-ordered identifier. Not a database PK; an audit-domain identifier. |
| `entry_version` | integer | Schema version. Used during long-term verification when entry shape evolves. |
| `decision_type` | string | Stable identifier for the decision class — `policy.evaluate`, `risk.score`, `case.transition`. Used by replay and threat-model partitioning. |
| `producer_id` | string | Stable identifier of the producing component / service / model. Stable means: same identifier across deploys, hostnames, instances. |
| `decided_at` | RFC 3339 UTC timestamp | Wall-clock time of the decision. Sub-second precision per `02-canonical-encoding-spec.md`. |
| `inputs_commitment` | one-of `{ inline, ref }` | Binding to the canonical bytes of the inputs the decision saw. `inline: <canonical bytes>` permitted when canonical size ≤ 4 KB; `ref: { hash, uri }` required above that threshold. The hash is what verifies; the uri is informational. See `decision-provenance.md`. |
| `ruleset_id` | string | Stable identifier of the rule/policy/model bundle that produced the decision. |
| `ruleset_version` | string | Version (semver, hash, or build-id) of the ruleset. |
| `code_version` | string | Build hash or release tag of the producing code. |
| `output` | structured value | The decision itself — the alternative chosen, the verdict, the next state, the score. Schema typed per `decision_type`. |
| `output_hash` | hex string | Hash of `output` after canonicalisation. Allows redaction of `output` while preserving chain integrity. See `retention-expiry-and-rtbf.md`. |
| `prev_hash` | hex string or null | Hash of the previous entry in the chain. Null only for the first entry. See `fingerprint-chains-and-integrity.md`. |
| `entry_hash` | hex string | Hash of the canonical encoding of all preceding fields, computed as the last step of entry construction. |

`output_hash` and `entry_hash` are not redundant: `output_hash` is the hash of just the output substructure (cheap to compute, stable through redaction); `entry_hash` is the hash of the full entry and is what the chain operates on.

## Recommended Optional Fields

| Field | When required | Purpose |
|-------|---------------|---------|
| `request_id` / `correlation_id` | When the decision is part of a larger request flow | Cross-pipeline traceability into observability |
| `caller_id` | When relevant for accountability | Who or what asked for the decision |
| `evidence` | Pattern A and Pattern B | Structured rule firings or model features that contributed to the output |
| `reasoning_ref` | Pattern C | Pointer to a more detailed reasoning record |
| `confidence` | Probabilistic decisions | Score, calibration band, model uncertainty |
| `explanation` | Decisions where the regulator demands explanation (EU AI Act, ECOA) | Human-readable explanation; canonicalised same as other fields |
| `signed_by` | When per-entry signing is in use | See `signing-and-export-integrity.md` |
| `tags` | Always permitted | Routing, retention class, sensitivity class — but tags MUST NOT change the meaning of the decision |

Forbidden fields:

- **PII not necessary for the decision.** PII expands the threat surface and complicates retention. Keep the audit log narrow; route PII through a separate, named pipeline if needed (see `retention-expiry-and-rtbf.md`).
- **Free-text human notes.** Notes belong in a case-management system, not an integrity-protected entry. They mutate. The audit log does not.
- **Mutable foreign keys.** A reference into a mutable table loses meaning when the table changes. Inline the relevant facts at decision time.

## Field Discipline

### `producer_id` must be stable across deploys

`producer_id = "policy-engine-v1.2.3"` ties the producer to a specific build. When you redeploy, every entry says the producer changed — you lose the ability to compare across deploys. Instead: `producer_id = "policy-engine"` and put the build in `code_version`.

### `ruleset_id` is independent of `code_version`

A ruleset can change without code changing (new rule added) and vice versa (code refactor with no rule changes). Keep them separate. This is what lets `decision-provenance.md` answer "did this decision change because the rules changed or the code changed?"

### `decided_at` is wall clock, not monotonic

Wall clock is what an auditor cares about. Use a separate field (e.g., `decided_at_monotonic_ns`) if you need ordering precision beyond what wall-clock timestamps give. Never substitute monotonic time for wall-clock time.

### `entry_id` is time-ordered

UUIDv7 or ULID, both of which embed millisecond timestamps. This makes range queries cheap and lets storage layers partition naturally. Random UUIDv4 forces full scans for time-window queries.

### `inputs_commitment` — inline vs ref

`inputs_commitment` is a one-of, not two fields. Exactly one form is present per entry:

- `inputs_commitment: { inline: <canonical bytes> }` — permitted when the canonical size is ≤ 4 KB. The inputs are stored within the entry and covered by `entry_hash` directly.
- `inputs_commitment: { ref: { hash: "<hex>", uri: "<optional informational uri>" } }` — required when canonical size > 4 KB. The hash binds the entry to those exact bytes; the URI is informational. The hash is what verifies, not the URI.

Schema validators MUST reject entries that present both forms or neither. The 4 KB threshold is a default; pipelines may pick a different threshold and pin it in `01-`.

If the same inputs appear across many decisions (e.g., a customer profile), `ref`-form means *one* stored object — content-addressed dedup is a property of this design, not an extra optimisation.

**Cross-spec coupling note:** The hash function used in `ref.hash` is the same hash function specified in `03-chain-and-integrity-spec.md`. A hash-function migration in `03-` therefore coordinates with `inputs_commitment` re-issuance for any new inputs after the migration boundary.

## Schema Versioning Without Chain Breakage

Entry shape evolves. Audit logs are append-only. These are reconcilable but only with discipline.

The rule:

> Old entries verify against the schema version they were written with. New entries are written against the current schema. The chain is heterogeneous by design.

Mechanics:

- `entry_version` is set at write time, never rewritten.
- The verifier loads schema version V and applies version-V canonicalisation rules to entries with `entry_version: V`. It applies version-W rules to entries with `entry_version: W`.
- The schema-version registry is itself an audit artifact, content-addressed, append-only. Removing a schema version is a chain-breaking event.

Permitted changes within the same `entry_version`:

- Adding optional fields (verifier ignores unknown fields if explicitly tolerated by the schema; safer is to add a *minor* version bump).

Changes that require a new `entry_version`:

- Adding a mandatory field.
- Removing or renaming a field.
- Changing the canonicalisation rule for any field.
- Changing the hash algorithm.
- Changing the chain construction.

When you bump `entry_version`, write a migration note in `01-decision-log-schema.md` describing the change, the date of the cutover, and the verifier's expected behaviour for the boundary.

## Scope Tier Classification

Run this classification at the end of `decision-log-architecture` and record in `00-scope-and-decisions.md`. The result drives the consistency-gate's required-artifact set.

| Tier | Characteristics |
|------|-----------------|
| XS | One decision type, one producer, internal-only consumption, no external export, no regulatory driver |
| S | Multiple decision types under one ruleset, internal consumers across services, no external export |
| M | Decisions persisted across deploys, consumed by a downstream system in a different trust zone, ad-hoc audit possible |
| L | Decisions are subject to scheduled regulatory audit, external export to compliance partners, or legal-hold capability is required |
| XL | Decisions feed an external regulator's pipeline, ATO/RMF process, court evidence, or are themselves the deliverable to an external party |

Heuristic checklist for tier promotion:

- Does any consumer of the trail sit outside the producer's trust zone? → S minimum.
- Does any decision feed a downstream system that itself signs or exports? → M minimum.
- Is there a regulator, auditor, or contract that names the trail? → L minimum.
- Does the trail need to be admissible as third-party evidence (court, regulator, ATO)? → XL.

When in doubt, promote one tier. The cost of an over-engineered audit pipeline at S is hours; the cost of an under-engineered one at L is the audit failing.

## Anti-Patterns

| Anti-pattern | Symptom | Fix |
|--------------|---------|-----|
| "Log line as audit entry" | Entries are unstructured strings; mandatory fields missing | Move to structured entry with the 13 mandatory fields above |
| Coarse decision boundary swallows the why | Entries say "decision made" with no rule reference | Split into Pattern A (with `evidence` array) or Pattern C |
| Fine boundary explodes volume and cost | Per-rule entries, no summary, every report runs N+1 queries | Pattern B with summary entry, or Pattern C with reasoning_ref |
| `producer_id = "host-7-prod"` | Entries change identity on redeploy | `producer_id` is the logical component name; build/host go elsewhere |
| `ruleset_version` and `code_version` collapsed | Cannot tell why a decision changed | Separate fields; `decision-provenance.md` explains why |
| Schema evolution by silent edit | Old entries fail to verify; verifier or schema is "fixed" to ignore | `entry_version` bump; old entries verify against old schema |
| PII inlined "because we'll need it later" | RTBF requires destroying the entry; chain breaks | Hold PII separately; reference by id; see `retention-expiry-and-rtbf.md` |
| Mutable references to operational data | Decisions reference rows that get updated; meaning drifts | Inline the facts that drove the decision; reference is informational only |
| `decided_at` from monotonic source | Auditor cannot align with their clock | Wall clock; monotonic in a separate field if needed |

## Spec Output (`00-scope-and-decisions.md` and `01-decision-log-schema.md`)

### `00-scope-and-decisions.md`

1. **System under audit** — name, owner, deployment context.
2. **Decision inventory** — every decision type, producer, consumer, with classification:

   | decision_type | producer | regulatory driver (if any) | consumer trust zone | pattern (A / B / C) |

3. **Boundary rule per decision_type** — what is one entry vs many.
4. **Tier** — declared per the table above, with the trigger that selected it.
5. **Out-of-scope** — explicitly named: events, observability noise, decisions made outside this pipeline.

### `01-decision-log-schema.md`

1. **Per `decision_type`**: full entry shape, mandatory and optional fields, types, validation rules.
2. **Cross-cutting fields**: the 13 mandatory fields and their constraints.
3. **`entry_version`** — current value, history of past versions, migration notes.
4. **Forbidden fields** — explicit list, with rationale.
5. **Validation strategy** — schema language (JSON Schema, protobuf, Avro), where validators run, what happens to entries that fail validation (must NEVER be silently dropped — see `audit-aware-logging-vs-observability.md`).

## Common Mistakes

| Mistake | Why it fails | Fix |
|---------|--------------|-----|
| Treating an HTTP access log as an audit log | Access logs lack ruleset_version and code_version; they record events, not decisions | Build the audit log as a separate artifact with the 13 mandatory fields |
| Allowing free-text fields | They mutate; they break canonicalisation discipline | Forbidden in entries; route to case management |
| Conflating the request and the decision | The request is the input; the decision is the output | One entry per decision; reference the request via `request_id` |
| Logging only the rule that fired (not which alternative was chosen) | "Why allow?" needs the alternative space too | `output` names the alternative; `evidence` names the rule(s) |
| Optional `decision_type` | Without a stable type, replay and threat partitioning don't work | Mandatory; closed enum |
| `inputs` always inline | Large inputs blow entry size; chain grows linearly with input size | Content-addressed reference past a size threshold |

## The Bottom Line

**The decision is the unit of evidence. Define it, bound it, name its mandatory fields, separate ruleset from code, version the schema without breaking old entries, and classify the tier honestly. Get this right and the rest of the pack mostly fills itself in.**

---

**Retrieval test (run at end of build):** "A team is auditing a workflow engine that emits state transitions for cases. Walk through the boundary choice (Pattern A / B / C), the mandatory fields, and how `producer_id` should be encoded if the engine runs in three regions with different deploy cadences."
