---
name: using-solution-architect
description: Routes a design brief, HLD, epic, or brownfield change through the full solution-architecture workflow — triage, NFR quantification, tech selection, ADRs, traceability, integration/migration, optional TOGAF/ArchiMate, and consolidated SAD with consistency gate
---

# Using Solution Architect

## Overview

**Solution Architect produces forward design artifacts from a brief / HLD / epic / brownfield change.**

This pack is the forward-design counterpart to the backward-looking Axiom pair:

- `axiom-system-archaeologist` → documents existing code (neutral)
- `axiom-system-architect` → assesses existing architecture (critical)
- **`axiom-solution-architect` (this pack)** → designs new/changed solutions (forward)

## When to Use

Use solution-architect skills when:

- You have a business brief, HLD, epic, or brownfield change request
- You need a traceable artifact set (not a one-off diagram or ad-hoc ADR)
- The design will be reviewed, handed off, or implemented by another team
- The context is enterprise (TOGAF phases, ArchiMate tooling, ARB submission)
- User asks: "Design me a solution for…" / "Take this brief and architect it" / "What artifacts do we need for this?"

Do **not** use this pack when:

- You are assessing an existing system → use `/system-architect`
- You are documenting an existing system → use `/system-archaeologist`
- You need process governance (branching, CI/CD, ADR lifecycle) → use `/sdlc-engineering`

## How to Access Reference Sheets

All reference sheets are in the same directory as this `SKILL.md`. When you see a link like `[quantifying-nfrs.md](quantifying-nfrs.md)`, read the file from the same directory.

## The Pipeline

```
archaeologist (docs) → architect (assesses) → (future) project-manager
                                 ↑
solution-architect (designs) ────┘  (solution-architect output can later
                                    be critiqued by system-architect)
```

## Expected Artifact Set

The pack produces numbered artifacts in a `solution-architecture/` workspace:

| # | Artifact | Producer |
|---|----------|----------|
| 00 | `scope-and-context.md` | `triaging-input-maturity` |
| 01 | `requirements.md` | `triaging-input-maturity` |
| 02 | `nfr-specification.md` | `quantifying-nfrs` |
| 03 | `nfr-mapping.md` | `quantifying-nfrs` |
| 04 | `solution-overview.md` | `resisting-tech-and-scope-creep` |
| 05 | `tech-selection-rationale.md` | `resisting-tech-and-scope-creep` |
| 06 | `descoped-and-deferred.md` | `resisting-tech-and-scope-creep` |
| 07 | `c4-context.md` | router catalog guidance |
| 08 | `c4-containers.md` | router catalog guidance |
| 09 | `component-specifications.md` | router catalog guidance |
| 10 | `data-model.md` | router catalog guidance |
| 11 | `interface-contracts.md` | router catalog guidance |
| 12 | `sequence-diagrams.md` | router catalog guidance |
| 13 | `deployment-view.md` | router catalog guidance |
| 14 | `requirements-traceability-matrix.md` | `maintaining-requirements-traceability` |
| 15 | `integration-plan.md` | `designing-for-integration-and-migration` |
| 16 | `migration-plan.md` (brownfield only) | `designing-for-integration-and-migration` |
| 17 | `risk-register.md` | `designing-for-integration-and-migration` |
| — | `adrs/NNNN-<title>.md` | `writing-rigorous-adrs` |
| — | `archimate-model/` (enterprise only) | `mapping-to-togaf-archimate` |
| — | `togaf-deliverable-map.md` (enterprise only) | `mapping-to-togaf-archimate` |
| 99 | `solution-architecture-document.md` | `assembling-solution-architecture-document` |

## Scope Tier

Every workflow is classified at the end of `triaging-input-maturity` into one of five tiers. The tier is recorded in `00-scope-and-context.md` and determines which artifacts are required by the consistency gate.

| Tier | Trigger | Required structural artifacts |
|------|---------|-------------------------------|
| XS | Single-component change, ≤1 integration, no new data, no new NFR targets | `00, 01, 02, 04, 09, 14, 15, 17`; ADRs only if a decision is made |
| S | ≤3 components, ≤2 integrations, existing NFR envelope | XS set + `05, 11` |
| M | New subsystem, new integrations, or new NFR targets | S set + `07, 08, 10, 13` |
| L | Cross-system, multiple new services, new data stores | M set + `12` (sequence diagrams) + C4 component view (produced as a subsection of `09-component-specifications.md` — not a new numbered artifact) |
| XL / enterprise | Governed by ARB / TOGAF / regulator | L set + `archimate-model/`, `togaf-deliverable-map.md` |

The tier is authoritative. If `04-solution-overview.md` or any ADR references an artifact from a higher tier, that artifact becomes required regardless of the declared tier — this is a tier promotion, not a waiver. Brownfield adds `16-migration-plan.md` at every tier.

## Catalog-Level Guidance for Router-Owned Artifacts

The artifacts in this section (`07, 08, 09, 10, 11, 12, 13`) are produced under router guidance with no dedicated specialist skill. The consistency gate's Check 1b enforces a quality floor for each — follow the criteria below when producing them, and expect the gate to reject artifacts that fall below the floor.

**Required quality floor per artifact:**

- `07-c4-context.md`: exactly one system box; named external actors and systems; no internal detail.
- `08-c4-containers.md`: technology labels on each container; one page; no component-level elements.
- `09-component-specifications.md`: every component has name, single-sentence responsibility, public interface, dependencies, consumed NFR IDs (cross-ref `03-`), satisfied requirement IDs.
- `10-data-model.md`: every entity names its owning service / bounded context; cardinalities stated; logical (not ORM-specific).
- `11-interface-contracts.md`: machine-readable where the protocol supports it (OpenAPI, AsyncAPI, Protobuf, GraphQL SDL) or prose contract including inputs, outputs, errors, idempotency, versioning.
- `12-sequence-diagrams.md`: 3–5 scenarios; each scenario has at least one failure-path variant; source is PlantUML or Mermaid and checked in.
- `13-deployment-view.md`: environments, runtime topology, scaling posture, zones/regions, network boundaries.

The longer-form guidance below expands on each. **An artifact that is present but fails its floor fails Check 1b — independent of file presence.**

### C4 views (07, 08)

- **Context diagram (07):** users, external systems, and the system-under-design. One box for the system. No internal detail.
- **Container diagram (08):** the system broken into deployable/runnable units (services, databases, queues, front-ends). Show technology labels (e.g., "PostgreSQL 16", "FastAPI"). Keep to one page.
- Skip component and code-level diagrams unless specifically requested — they duplicate `09-component-specifications.md` and rot fast.
- One diagram per purpose. Do not produce five overlapping diagrams.

### Component specifications (09)

Per component: name, responsibility (one sentence), public interface, dependencies (components it calls), consumed NFRs (which NFRs it is load-bearing for — cross-reference `03-nfr-mapping.md`), and requirement IDs it satisfies. The `maintaining-requirements-traceability` skill checks this cross-referencing.

### Data model (10)

Entities + relationships + ownership (which service/bounded-context owns which entity). Include cardinality and lifecycle notes where non-obvious. Avoid ORM-specific schema; stay at the logical level.

### Interface contracts (11)

Machine-readable where possible (OpenAPI, AsyncAPI, Protobuf, GraphQL SDL). If prose, include: method/operation, inputs, outputs, errors, idempotency, versioning stance.

### Sequence diagrams (12)

One per critical scenario (happy path + notable failure paths). PlantUML or Mermaid source checked in. Do not produce one per endpoint — pick the 3-5 scenarios that actually reveal design tension.

### Deployment view (13)

Target environments, runtime topology, scaling posture, regions/zones, network boundaries. This is the operations/SRE handoff surface.

## Routing

### Scenario: "Design me a solution for X"

1. Read the input. Use: `triaging-input-maturity` → `00-scope-and-context.md`, `01-requirements.md` (triage also classifies scope tier and records it in `00-`; the tier is internal routing state, not a durable artifact)
2. Use: `quantifying-nfrs` → `02-`, `03-`
3. Use: `resisting-tech-and-scope-creep` → `04-`, `05-`, `06-`
4. Produce router-owned artifacts required by the scope tier (`07-13` subset per the Scope Tier table) following the quality floor above
5. Use: `writing-rigorous-adrs` as decisions arise throughout → `adrs/`
6. Use: `maintaining-requirements-traceability` → `14-`
7. Use: `designing-for-integration-and-migration` — always for `15-` and `17-` (integration contracts and architectural risk apply to greenfield too); additionally produces `16-` for brownfield
8. If enterprise context: use `mapping-to-togaf-archimate` → `archimate-model/`, `togaf-deliverable-map.md`
9. Use: `assembling-solution-architecture-document` → `99-`

### Scenario: "Critique this design package"

Use the `solution-design-reviewer` agent via `/review-solution-design`.

### Scenario: "Is Kafka the right choice for this?"

Use the `tech-selection-critic` agent. Red-teams a tech choice against requirements and constraints.

## Integration with Other Skillpacks

### Security (ordis-security-architect)

```
Solution architect produces 02 (NFRs), 04 (overview), 09 (components)
→ ordis-security-architect reads these, produces threat model + controls
→ Threats feed back into 17 (risk-register) and adrs/
```

When the threat model exists, the solution architect's `17-risk-register.md` does not duplicate threat entries — it carries a pointer to the threat ID and records only the architectural consequence and mitigation. See the "Security surface" category in `designing-for-integration-and-migration`.

### Compliance drivers

When compliance frameworks are in scope (SOC 2, HIPAA, PCI-DSS, GDPR, data residency), the framework itself is a driver for NFRs (auditability, retention, encryption-at-rest, residency) and for risks (compliance exposure). Record the framework as a `CON-*` in `01-requirements.md` and trace it through `02-` → `14-` → `17-`. If `ordis-security-architect` is in play, the threat model carries the canonical control mapping; the solution architect carries the architectural consequences.

### Documentation (muna-technical-writer)

```
Solution architect produces 99-solution-architecture-document.md
→ muna-technical-writer polishes for target register (exec, technical, public, policy)
```

### SDLC governance (axiom-sdlc-engineering)

```
Solution architect produces adrs/
→ axiom-sdlc-engineering/design-and-build manages ADR lifecycle + governance
```

### Brownfield input (axiom-system-archaeologist)

```
Archaeologist produces 01-discovery-findings.md, 02-subsystem-catalog.md
→ Solution architect's triage consumes these as input context
→ No duplicate analysis
```

### Domain packs

Consult during tech selection:

- `yzmir-ml-production` → ML-serving tech choices
- `axiom-web-backend` → API framework choices
- `axiom-rust-engineering` / `axiom-python-engineering` → language-level patterns

## Decision Tree

```
Do you have a design brief / HLD / epic / change request?
├─ No → Clarify input with user first
└─ Yes → Continue

Is this brownfield (modifying existing system)?
├─ Yes → Is there archaeologist output for the system?
│         ├─ No → Run /system-archaeologist first, OR record [ASSUMED]
│         │       brownfield context in 00- and raise RSK
│         │       (see designing-for-integration-and-migration)
│         └─ Yes → triaging-input-maturity consumes it
│         → designing-for-integration-and-migration produces 15, 16, 17
└─ No (greenfield) → triaging-input-maturity
                  → designing-for-integration-and-migration produces 15 and 17 (skip 16)

A small change request against a well-documented brownfield system is usually tier XS or S —
run the XS/S artifact subset from the Scope Tier table, not the full pipeline.
```

### Enterprise activation — decide early, record the decision

Keyword presence alone (a stakeholder saying "ArchiMate" in passing) does not activate enterprise mode. Activate only when one of the following is true:

- ARB submission is a release gate for this project
- The customer organization publishes a TOGAF-aligned deliverable set (even if not named as such)
- A governing enterprise-architecture function must countersign the SAD
- ArchiMate models are a required artifact for downstream tooling

```
├─ Activated → Activate mapping-to-togaf-archimate; record
│              "Enterprise: activated — [driver]" in 00-scope-and-context.md
└─ Not activated → Record "Enterprise: not activated — [reason]" in 00-;
                   do not load mapping-to-togaf-archimate
```

The gate report carries the activation state explicitly so a reader can tell whether enterprise mode was considered-and-declined or forgotten.

## Stop Conditions

The pipeline is designed to be run end-to-end for M-tier and above. For lighter tiers, and when inputs are adverse, declare a stop condition explicitly rather than silently dropping steps. Every stop condition results in a recorded artifact (`00-` note, waiver, or RSK entry) — silent drops are the pattern this pack is built to prevent.

| Condition | Response |
|-----------|----------|
| Tier is XS and no cross-system impact | Run the XS artifact subset (see Scope Tier table); emit the SAD with tier noted; skip enterprise binding unless explicitly required |
| Requirements cannot be quantified because the business context is absent | Stop at `quantifying-nfrs`. Do not proceed to tech selection with adjective-only NFRs. Escalate to the business owner. |
| Brownfield with no archaeologist output and no budget to run archaeology | Record `[ASSUMED]` context in `00-` with explicit unknowns, raise `RSK-NN: brownfield context unverified` (High/High, mitigation = run archaeologist), proceed but mark the SAD as "provisional — unverified brownfield context" |
| Enterprise binding required but no TOGAF/ArchiMate skill capacity on the team | Produce the non-enterprise artifact set; record as gate waiver with rationale; schedule the TOGAF mapping as a follow-up task |
| Stakeholder insists on big-bang cutover with no business-time-constraint reason | Stop `16-` production, return to stakeholder with the "reshape into stages" guidance from `designing-for-integration-and-migration` |

## Update Workflows

A SAD is a living document. After v1.0.0, changes come as scope extensions, requirement drift, new decisions, or learning from production. The table below names what re-runs and what re-gates for each change shape.

| Change shape | Re-run | Re-gate |
|--------------|--------|---------|
| New or changed requirement | `01-`, `02-` if NFRs affected, `14-`, `09-` cross-ref, affected ADRs | Checks 2, 3, 4 |
| New ADR (no requirement change) | `adrs/`, `05-` if tech choice | Checks 4, 5 |
| Component split or merge | `09-`, `14-`, affected interface contracts (`11-`), affected sequences (`12-`) | Checks 1b, 2 |
| New integration | `15-`, `17-` | Checks 6, 7 |
| Migration stage added or redesigned | `16-`, `17-` | Checks 6, 7, 1 |
| NFR re-target (e.g., scale target raised) | `02-`, `03-`, affected components in `09-`, affected ADRs | Checks 3, 4, 7 |
| Scope descope | `06-`, `01-`, `14-` (remove), affected `09-` | Checks 1, 2 |

Bump the SAD version (semver) on every re-emission. The gate report is versioned alongside the SAD. A SAD whose gate report is older than its latest numbered artifact is stale and must be re-gated before the SAD is cited downstream.

## Quick Reference

| Need | Use This |
|------|----------|
| Classify input + emit scope/requirements | `triaging-input-maturity` |
| Quantified NFRs + per-component mapping | `quantifying-nfrs` |
| Tech selection + YAGNI discipline | `resisting-tech-and-scope-creep` |
| Rigorous ADRs | `writing-rigorous-adrs` |
| Requirements traceability matrix | `maintaining-requirements-traceability` |
| Integration + migration + risks | `designing-for-integration-and-migration` |
| TOGAF phase mapping + ArchiMate views | `mapping-to-togaf-archimate` |
| Consolidate into SAD with consistency gate | `assembling-solution-architecture-document` |
| Critique a draft design package | agent: `solution-design-reviewer` |
| Red-team a tech choice | agent: `tech-selection-critic` |

## Related Documentation

- **Design spec:** `docs/superpowers/specs/2026-04-17-axiom-solution-architect-pack-design.md`
- **Archaeologist plugin:** `axiom-system-archaeologist`
- **System architect plugin:** `axiom-system-architect`

## The Bottom Line

**Forward design. Constraints before technology. Traceability by default. Numbered artifacts, one consolidated SAD, one consistency gate before emission.**

---

## Solution Architect Specialist Skills Catalog

After routing, load the appropriate specialist skill for detailed guidance:

1. [triaging-input-maturity.md](triaging-input-maturity.md) — Classify input shape, decide workflow, emit scope & requirements
2. [resisting-tech-and-scope-creep.md](resisting-tech-and-scope-creep.md) — Constraints-first tech selection, YAGNI audit, stakeholder-pressure resistance
3. [quantifying-nfrs.md](quantifying-nfrs.md) — Quantified NFRs with measurement methods + per-component mapping
4. [writing-rigorous-adrs.md](writing-rigorous-adrs.md) — Full ADR template with alternatives, tradeoffs, rollback, expiry
5. [maintaining-requirements-traceability.md](maintaining-requirements-traceability.md) — RTM, orphan detection
6. [designing-for-integration-and-migration.md](designing-for-integration-and-migration.md) — Integration reality, migration plan, architectural risk discipline
7. [mapping-to-togaf-archimate.md](mapping-to-togaf-archimate.md) — TOGAF ADM phase mapping, ArchiMate layer/element discipline, viewpoint subsets
8. [assembling-solution-architecture-document.md](assembling-solution-architecture-document.md) — Consolidation + cross-artifact consistency gate
