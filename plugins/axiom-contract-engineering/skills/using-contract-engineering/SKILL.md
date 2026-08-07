---
name: using-contract-engineering
description: Use when designing, evolving, or reviewing **typed contracts between subsystems** — the records that cross authority boundaries in a pipeline, service mesh, or multi-stage system. Use when a consumer crashed on a missing field and the proposed fix is a default or a lenient reader; when a schema is changing meaning, unit, or shape under a slowly-upgrading fleet; when one record is derived from others and must be reproducible; when a consumer must be provably unable to see a field; when decisions must stay interpretable after thresholds are retuned. Pairs with `/determinism-and-replay` (whole-system replay) and `/solution-architect` (which boundaries exist at all). Do not load for REST/GraphQL endpoint design (`/web-backend`) or wire-protocol performance tuning.
---

# Using Contract Engineering

## Overview

**A contract is not a struct definition. It is the record of what one subsystem may say to another — and the discipline of making everything else unsayable.**

Most systems treat cross-boundary records as data plumbing: define a class, serialize it, add fields as needed, default the missing ones, and keep the readers tolerant so deploys don't break. Every one of those reflexes is individually reasonable and collectively fatal. The tolerant reader converts contract drift into silent reinterpretation. The default converts "unmeasured" into a confident number. The unversioned meaning change makes one name denote two things forever. The passthrough field becomes the channel through which upstream preferences steer downstream decisions while every record stays schema-legal.

This pack teaches contract engineering as a discipline: schema design where illegal states are unrepresentable, versioning where meaning changes fail closed, resolution that is reproducible from recorded inputs, blinding that is structural rather than promissory, identity that is content-addressed, and testing that targets the failure modes of contracts rather than the happy path of serializers.

The center of gravity is a defect class that has killed real systems: **silent defaults** — plumbing that quietly teaches downstream consumers that *unmeasured means zero*. Every sheet in this pack exists to make that class, and its relatives, unrepresentable by construction rather than policed by review.

## When to Use

Use this pack when:

- You are designing the record layer for a multi-subsystem pipeline — the schemas, their versioning rules, their resolver, their views, their tests — before the subsystems exist (contract-first) or while they harden.
- A field can be unmeasured, unavailable, or intermittently sampled, and consumers must never mistake absence for a value.
- A schema must evolve under a mixed fleet where old producers emit the previous format for months.
- One record is deterministically derived from others (intent + policy → request; evidence + thresholds → decision) and audits or replays must reconstruct it.
- A consumer must be structurally unable to see a field (regulatory redaction, evaluation blinding, provider anonymity).
- Records need durable semantic identity — deduplication, equivalence classes, cross-stage binding, audit joins.
- You are reviewing an existing contract suite or a contract-touching diff for drift, shims, covert channels, or silent defaults.

Do **not** use this pack when:

- The question is REST/GraphQL API surface design, pagination, auth — load `/web-backend`.
- The question is whole-system determinism (RNG streams, snapshots, replay harnesses) — load `/determinism-and-replay`; this pack covers the contract-layer slice.
- The question is which subsystem boundaries should exist at all — load `/solution-architect` or `/system-architect`; this pack starts once the boundaries are chosen.
- The question is serialization framework performance (protobuf vs flatbuffers throughput) without a semantics problem.

## Start Here

If your input is "we have (or need) typed records between subsystems and they must not lie," read in this order:

1. [`contract-first-boundaries.md`](contract-first-boundaries.md) — the base discipline: typed immutable records, one authority per record class, plain-language role statements, illegal states unrepresentable, distinct concerns in distinct records.
2. [`silent-default-elimination.md`](silent-default-elimination.md) — the killer defect class: explicit absence encoding, validity masks, fail-loud parsing, version-gated absence semantics, the prohibition on tolerant readers.
3. [`schema-versioning-and-evolution.md`](schema-versioning-and-evolution.md) — any change in meaning, unit, basis, normalisation, or provenance is a new version; readers fail closed on versions they don't declare; expand/contract migration without shims.
4. [`deterministic-resolution.md`](deterministic-resolution.md) — derived records come from pure resolvers over recorded inputs: canonical outputs, no covert channels, no hidden state, narrow-only authority.
5. [`blinding-by-construction.md`](blinding-by-construction.md) — excluded fields are absent from the view's schema, not redacted at serialization; forbidden fields are schema-invalid and rejected on arrival.
6. [`canonical-identity.md`](canonical-identity.md) — content-addressed semantic hashing; every downstream record binds the same canonical hash; raw-to-canonical traceability.
7. [`dependency-direction.md`](dependency-direction.md) — the contract layer imports nothing from subsystems, enforced by an import-lint gate, not convention.
8. [`versioned-policy-parameters.md`](versioned-policy-parameters.md) — thresholds, weights, and operating points live in versioned policy records; decisions bind the policy version in force.
9. [`definition-lifecycle.md`](definition-lifecycle.md) — named definitions move draft → approved → locked through recorded events, never silent edits.
10. [`contract-testing.md`](contract-testing.md) — golden fixtures, real round-trips, canonicalisation properties, schema-invalid rejection, authority tests, replay tests.

## Sheet Index

| Sheet | Role |
|-------|------|
| [`contract-first-boundaries.md`](contract-first-boundaries.md) | Typed immutable records, one authority per class, unrepresentable illegal states |
| [`silent-default-elimination.md`](silent-default-elimination.md) | Absence encoding, validity masks, fail-loud parsing, no tolerant readers |
| [`schema-versioning-and-evolution.md`](schema-versioning-and-evolution.md) | Version = meaning; fail-closed gates; expand/contract migration; no shims |
| [`deterministic-resolution.md`](deterministic-resolution.md) | Pure resolvers, canonical outputs, covert-channel closure, narrow-only authority |
| [`blinding-by-construction.md`](blinding-by-construction.md) | Structural blinding: view types, allowlist projection, policy closure, canaries |
| [`canonical-identity.md`](canonical-identity.md) | Content-addressed semantic hashing; cross-stage hash binding; traceability |
| [`dependency-direction.md`](dependency-direction.md) | Contracts as leaf package; import-lint as a CI gate |
| [`versioned-policy-parameters.md`](versioned-policy-parameters.md) | Policy records; decisions bind policy versions; hysteresis and veto points |
| [`definition-lifecycle.md`](definition-lifecycle.md) | Draft → approved → locked via recorded events; content-hash enforcement |
| [`contract-testing.md`](contract-testing.md) | The test taxonomy for contracts; anti-vacuity rules |

## The Failure-Mode Catalogue

The recurring defects this pack exists to close. The `contract-reviewer` agent audits directly against this list.

1. **Silent defaults** — absence converted to a plausible value (`.get(field, 0)`, schema defaults on measurements, proto3 scalars); downstream acts on data that was never measured. *(silent-default-elimination)*
2. **Tolerant readers** — parsers that paper over drift: dual-key fallbacks, versionless leniency, quiet type coercion. Drift becomes invisible instead of loud. *(silent-default-elimination, schema-versioning-and-evolution)*
3. **Fail-open version gates** — `if version >= 2:` parses future majors under today's semantics; a missing version is presumed rather than rejected. The reader promises compatibility on behalf of producers that don't exist yet. *(schema-versioning-and-evolution)*
4. **Version-in-name-only** — the schema's meaning changed (unit, basis, normalisation) and the version didn't; one identifier now names two incompatible meanings, unrecoverably. *(schema-versioning-and-evolution)*
5. **Compat shims** — reader code that lets two meanings coexist "temporarily," with no retirement metric; the migration never completes and the dialects multiply. *(schema-versioning-and-evolution)*
6. **Covert channels through permitted fields** — free-text notes, batch position, aliases, field ordering, serialization variance: schema-legal paths by which upstream variation steers downstream output. *(deterministic-resolution, blinding-by-construction)*
7. **Resolver logic that grew preferences** — tie-breaks, heuristics, and vendor hints living as unversioned code instead of versioned policy records; every deploy silently re-decides history. *(deterministic-resolution, versioned-policy-parameters)*
8. **Hidden resolver state** — carry-forward caches and process memory consumed by resolution but recorded nowhere; outputs unreproducible from stored inputs. *(deterministic-resolution)*
9. **Blinding by ignoring** — the sensitive field is present-and-redacted (or present-and-promised-unread) instead of absent from the view's schema; deny-list projections that leak every future field by default. *(blinding-by-construction)*
10. **Dual sources of truth** — one schema defined in two places (dataclass + JSON schema doc; two copies of one policy weight) that drift independently; a header citing a schema file that doesn't exist. *(contract-first-boundaries, versioned-policy-parameters)*
11. **Unversioned policy** — thresholds as code constants; decision records that carry a wire version but no policy version; incident hotfixes that change decision semantics with no version event. *(versioned-policy-parameters)*
12. **Silent definition edits** — an approved schema or rubric edited in place; lifecycle state as a mutable flag with no event trail. *(definition-lifecycle)*
13. **Contract tests that test the constructor** — round-tripping the schema's own output; asserting unknown future versions parse; fixtures generated by the serializer under test; canaries without positive controls. *(contract-testing)*

## Boundary

This pack does **not** cover:

- **API endpoint design** (REST resources, GraphQL schemas, pagination, auth) — `/web-backend`. This pack governs the records behind the endpoint, not the HTTP surface.
- **Whole-system determinism and replay** (RNG streams, snapshot completeness, divergence localisation) — `/determinism-and-replay`. This pack's `deterministic-resolution.md` covers the contract-layer resolver; the replay harness around it is that pack's job.
- **Choosing subsystem boundaries** — `/solution-architect`, `/system-architect`, `/procedural-architecture`. This pack begins once you know which boundaries exist.
- **Serialization framework internals** — protobuf/Avro/Arrow encoding performance, zero-copy tricks. The discipline here is framework-agnostic; examples use plain Python for clarity.
- **Data-pipeline orchestration** (scheduling, backfills, DAGs) — this pack governs the records that flow, not the machinery that flows them.

## Routing by Symptom

### "A consumer is crashing on a missing field — what's the right fix?"

**Route to**: [`silent-default-elimination.md`](silent-default-elimination.md), then [`schema-versioning-and-evolution.md`](schema-versioning-and-evolution.md).

**Why**: the crash is the contract telling you producer and consumer disagree. The fix is a version-gated absence declaration, not a default and not an unconditional `.get()` — both convert the disagreement into silent reinterpretation. The versioning sheet gives the gate; the silent-default sheet gives the absence encoding behind it.

### "We need to change a field's unit / meaning / normalisation"

**Route to**: [`schema-versioning-and-evolution.md`](schema-versioning-and-evolution.md).

**Why**: same name + new meaning has no safe deployment order under a mixed fleet. The sheet gives the rename + expand/contract path, the fail-closed gate, and the parse-version counter that tells you when the old branch can die.

### "Two runs produced different outputs from the same inputs"

**Route to**: [`deterministic-resolution.md`](deterministic-resolution.md), then [`canonical-identity.md`](canonical-identity.md).

**Why**: either the resolver consumed something it didn't record (hidden state, clock, environment) or "the same inputs" weren't canonically equal (serialization variance, aliases, ordering). The resolution sheet closes the first; the identity sheet gives the equivalence discipline for the second.

### "The reviewer/evaluator must not see who produced this"

**Route to**: [`blinding-by-construction.md`](blinding-by-construction.md).

**Why**: redaction at serialization fails open the day someone adds a field. Blinding is a separate view type with no such field, an allowlisted projection with a policy closure walk, and a canary test — three layers, each catching what the previous one misses.

### "We retuned the thresholds and now old decisions look wrong"

**Route to**: [`versioned-policy-parameters.md`](versioned-policy-parameters.md), then [`definition-lifecycle.md`](definition-lifecycle.md).

**Why**: decisions that don't bind a policy version are reinterpreted by every retune. Policy lives in versioned records that resolvers consume as input; the lifecycle sheet governs how a new policy version becomes the one in force.

### "How do we know two candidates are actually the same thing?"

**Route to**: [`canonical-identity.md`](canonical-identity.md).

**Why**: equality by storage ID splits equivalents; equality by raw bytes splits on formatting. Content-addressed hashing over an explicitly-declared semantic subset, with invariance and separation property tests, is the only equality that survives serialization and stage boundaries.

### "The contract package keeps growing helpers and imports"

**Route to**: [`dependency-direction.md`](dependency-direction.md).

**Why**: one subsystem import in the contract layer couples every subsystem to it and opens the path for policy to leak into schema. The sheet gives the import-linter gate that makes the direction mechanical.

### "Are our contract tests actually testing anything?"

**Route to**: [`contract-testing.md`](contract-testing.md).

**Why**: if the fixtures come from the schema's own constructor, no. The sheet gives the taxonomy — golden wire fixtures per version, absence paths, fail-closed gates, canonicalisation properties, authority tests, replay tests — and the coverage rule that says when you're done.

## Pipeline Position

```
axiom-contract-engineering (this pack)       axiom-determinism-and-replay
  what may be said across boundaries:  ←-→   whole-system reproducibility:
  schemas, versions, resolvers, views,       RNG streams, snapshots, replay
  identity, contract tests                   harnesses, divergence localisation
  ────────────────────────────────────────────────────────────────
       The contract layer is the vocabulary of a replayable system.
       Records designed here are what that pack's harness replays.

axiom-contract-engineering (this pack)       axiom-solution-architect / system-architect
  the records at each boundary and     ←-→   which boundaries exist, which
  the discipline they enforce                subsystems own which concerns
  ────────────────────────────────────────────────────────────────
       Architecture chooses the boundaries; this pack makes the
       boundaries typed, versioned, and unlieable-across.

axiom-contract-engineering (this pack)       axiom-audit-pipelines
  canonical identity, recorded         ←-→   evidence chains: hash chains,
  resolution, definition lifecycle           signatures, anchoring, export
  ────────────────────────────────────────────────────────────────
       This pack makes records trustworthy at creation;
       audit-pipelines makes their history tamper-evident after.
```

## How to Access Reference Sheets

All reference sheets are in the same directory as this `SKILL.md`. When you see a link like [`silent-default-elimination.md`](silent-default-elimination.md), read the file from the same directory as this file.

## Quick Reference

| Symptom / Need | Sheet |
|----------------|-------|
| Missing field crash; tempted by a default | `silent-default-elimination.md` |
| Metric sometimes unmeasurable | `silent-default-elimination.md` |
| Unit / meaning / basis change | `schema-versioning-and-evolution.md` |
| Mixed fleet, slow upgrade | `schema-versioning-and-evolution.md` |
| `version >= N` in a reader | `schema-versioning-and-evolution.md` |
| Derived record must be reproducible | `deterministic-resolution.md` |
| Output depends on field order / batch position | `deterministic-resolution.md` |
| Consumer must not see a field | `blinding-by-construction.md` |
| Free-text field crossing a boundary | `blinding-by-construction.md` |
| Dedup / equivalence / cross-stage identity | `canonical-identity.md` |
| Contract package importing subsystems | `dependency-direction.md` |
| Thresholds in code constants | `versioned-policy-parameters.md` |
| Decision records lack policy version | `versioned-policy-parameters.md` |
| Approved schema edited in place | `definition-lifecycle.md` |
| Round-trip tests from constructors | `contract-testing.md` |
| "Prove field X cannot arrive" | `contract-testing.md`, `blinding-by-construction.md` |

## Commands and Agents

The pack ships three slash commands and two agents.

**Commands:**

- `/design-contract-suite` — forward-design a contract suite from a system's boundaries and record classes: schemas with absence encoding, versioning rules, resolver shape, blinded views, canonical-hash plan, and the contract-test plan. Dispatches `contract-suite-architect`.
- `/review-contracts` — audit a contract suite (or a contract-touching diff) against the 13-entry failure-mode catalogue; severity-rated findings with evidence and the sheet that closes each gap. Dispatches `contract-reviewer`.
- `/audit-contract-drift` — mechanical sweep for drift signals: readers whose accepted fields diverge from the schema, version constants untouched by semantic diffs, dual schema definitions, defaults on measurement fields, deny-list projections.

**Agents:**

- **`contract-suite-architect`** — producer SME. Given a system's boundaries and record classes, designs the complete contract suite: record schemas with authority table, versioning and evolution rules, resolver contract, blinded views, canonical identity plan, policy records, and test plan. Follows the SME Agent Protocol.
- **`contract-reviewer`** — critic SME. Audits a contract suite or diff against the failure-mode catalogue; severity by blast radius, evidence per finding, refuses to rubber-stamp (a zero-finding audit is reported as a defect of the audit). Follows the SME Agent Protocol.

## Cross-References

- `axiom-determinism-and-replay` — whole-system record/replay; this pack's resolver discipline is the contract-layer slice of that pack's determinism tiers.
- `axiom-audit-pipelines` — tamper-evident history over the records this pack designs; canonical encoding discipline is shared.
- `axiom-solution-architect` / `axiom-system-architect` — boundary selection and architecture decisions upstream of contract design.
- `axiom-static-analysis-engineering` — building the import-lint and schema-conformance analyzers that enforce this pack's gates mechanically.
- `axiom-web-backend` — the HTTP/API surface in front of contract records.
- `ordis-quality-engineering` — test-suite health for the contract-test suites this pack specifies.
