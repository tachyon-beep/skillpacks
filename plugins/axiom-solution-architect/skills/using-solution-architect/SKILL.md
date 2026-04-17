---
name: using-solution-architect
description: Use when you have a high-level design, brief, epic, or brownfield change and need to produce a complete traceable solution architecture artifact set (ADRs, C4, NFRs, RTM, integration/migration plans, risks, TOGAF/ArchiMate views, consolidated SAD) — routes to the appropriate specialist skill based on input maturity and enterprise context
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

## Catalog-Level Guidance for Router-Owned Artifacts

For artifacts produced under router guidance (no dedicated skill), follow these rules:

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

1. Read the input. Use: `triaging-input-maturity` → `00-scope-and-context.md`, `01-requirements.md`, workflow plan
2. Use: `quantifying-nfrs` → `02-`, `03-`
3. Use: `resisting-tech-and-scope-creep` → `04-`, `05-`, `06-`
4. Produce router-owned artifacts (`07-13`) following catalog guidance above
5. Use: `writing-rigorous-adrs` as decisions arise throughout → `adrs/`
6. Use: `maintaining-requirements-traceability` → `14-`
7. Use: `designing-for-integration-and-migration` → `15-`, (`16-` if brownfield), `17-`
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
│         ├─ No → Consider running /system-archaeologist first
│         └─ Yes → triaging-input-maturity consumes it
└─ No (greenfield) → triaging-input-maturity

Does the context mention TOGAF, ArchiMate, ARB, phase gates?
├─ Yes → Activate mapping-to-togaf-archimate in the workflow
└─ No → Keep TOGAF/ArchiMate binding out of the workflow
```

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

## Status

**Current Status:** v1.0.0 — 9 skills, 2 agents, 2 plugin commands, 1 router slash command.

**Production-ready skills:**

- ✅ using-solution-architect (router)
- ✅ triaging-input-maturity
- ✅ resisting-tech-and-scope-creep
- ✅ quantifying-nfrs
- ✅ writing-rigorous-adrs
- ✅ maintaining-requirements-traceability
- ✅ designing-for-integration-and-migration
- ✅ mapping-to-togaf-archimate
- ✅ assembling-solution-architecture-document

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
