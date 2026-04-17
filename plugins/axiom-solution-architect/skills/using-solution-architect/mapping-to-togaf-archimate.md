# Mapping to TOGAF / ArchiMate

## Overview

**Enterprise architecture frameworks exist so that designs survive handoff.**

When a solution design will enter an organisation's EA function (ARB, enterprise repository, multi-project programme), ad-hoc diagrams fail. Artifacts must map to the framework the EA team uses — most commonly TOGAF (process + deliverable vocabulary) and ArchiMate (modelling language).

**Core principle:** Right layer, right element, right relationship, right viewpoint. Nothing "close enough."

## When to Use

This skill activates when `triaging-input-maturity` flags enterprise context:

- TOGAF, ADM, or specific phase references (Phase A/B/C/D/E/F/G/H)
- ArchiMate references or EA tool names (Sparx EA, Archi, BiZZdesign, Visual Paradigm)
- Architecture Review Board (ARB) submission
- "Enterprise architecture", "target state", "current state", explicit phase gates

Skip this skill entirely for product-engineering work without enterprise context. Do not produce TOGAF/ArchiMate bindings "for completeness" — they're overhead without an EA consumer.

## What This Skill Produces

1. `archimate-model/` directory — structural model by layer, plus viewpoint subsets
2. `togaf-deliverable-map.md` — cross-reference mapping each numbered artifact to TOGAF ADM phase + Architecture Content Framework deliverable

## ArchiMate Layer Taxonomy (short reference)

| Layer | Purpose | Typical elements |
|-------|---------|------------------|
| Strategy | Capabilities, courses of action, resources | Capability, CourseOfAction, Resource |
| Business | Business actors, processes, services, functions | BusinessActor, BusinessRole, BusinessProcess, BusinessService, BusinessFunction, BusinessObject, Contract |
| Application | Application components, services, interfaces, data | ApplicationComponent, ApplicationService, ApplicationInterface, DataObject |
| Technology | Nodes, devices, system software, networks | Node, Device, SystemSoftware, Network, CommunicationNetwork, TechnologyService, Artifact |
| Physical | Equipment, facilities, distribution networks | Equipment, Facility, DistributionNetwork |
| Implementation & Migration | Work packages, deliverables, plateaus, gaps | WorkPackage, Deliverable, Plateau, Gap |

**Common layer mistakes (reject on sight):**

- Placing an application component on the Business layer because "it represents a business capability" — capabilities are Strategy layer elements; the app realising them is Application layer, related by `realization`.
- Placing a database on the Business layer because "it holds business data" — DataObjects (Application) or Artifacts (Technology) hold data; Business layer holds BusinessObjects, which are conceptual.
- Placing infrastructure on the Application layer because "it's part of our stack" — nodes, devices, and networks are Technology layer.

## ArchiMate Relationships

The common structural relationships and their strict meanings:

| Relationship | Meaning | Example |
|--------------|---------|---------|
| Composition | Strong whole-part, part belongs exclusively to whole | Order composed of OrderLineItems |
| Aggregation | Whole-part, part can exist independently / be shared | Portfolio aggregates Products |
| Assignment | Active-element allocated to / deployed on another | ApplicationComponent *assigned-to* Node |
| Realization | More abstract element implemented by more concrete element | BusinessService realized by ApplicationService |
| Serving | One element provides its functionality to another | ApplicationService serving BusinessProcess |
| Used-by (inverse of Serving) | Consumer side | BusinessProcess used-by ApplicationService (rarely drawn this way) |
| Access | Behavioural element acts on a passive element | BusinessProcess *reads/writes* BusinessObject |
| Flow | Transfer between active elements | ApplicationComponent *flows to* ApplicationComponent |
| Triggering | Temporal or causal sequence | BusinessProcess triggers BusinessProcess |

**Rules:**

- Composition and aggregation are **not** interchangeable. Composition = "if the whole disappears, the part disappears too"; aggregation = "the part can outlive the whole."
- Assignment is for active-to-active deployment ("runs on"); realization is for abstract-to-concrete ("implements"). Not the same.
- If you can't tell which relationship applies, you don't understand the elements well enough — clarify the elements before drawing the line.

## Viewpoints (not filtered copies)

A **viewpoint** selects elements and relationships relevant to a specific stakeholder concern. It is not "the master model coloured differently."

Standard viewpoints used in v1.0.0:

### CIO view

- Concerns: strategic alignment, investment, risk, roadmap
- Elements: Strategy layer (Capabilities), Business layer (major BusinessServices), Implementation & Migration (Plateaus)
- Suppressed: Application layer detail, Technology layer
- Output: `archimate-model/viewpoints/cio-view.md`

### ARB view

- Concerns: standards adherence, architectural integrity, cross-programme impact
- Elements: Application layer (Components, Interfaces), Technology layer (Nodes, Services), relationships (realization, assignment)
- Suppressed: process-level Business detail unless directly relevant
- Output: `archimate-model/viewpoints/arb-view.md`

### Engineering view

- Concerns: implementation detail, deployment, integration contracts
- Elements: Application (Components, Services, Interfaces, DataObjects), Technology (Nodes, Artifacts, Networks)
- Suppressed: Strategy layer, high-level Business elements
- Output: `archimate-model/viewpoints/engineering-view.md`

Each viewpoint file names the **concerns it addresses** and **which elements/relationships it shows** — a viewpoint without a concern statement is a filter.

## Model Files

Break the ArchiMate model by layer for readability:

```
archimate-model/
├── strategy-layer.md           (rare — only for enterprise-strategic designs)
├── business-layer.md
├── application-layer.md
├── technology-layer.md
├── physical-layer.md           (rare — only if physical assets relevant)
├── implementation-migration-layer.md
└── viewpoints/
    ├── cio-view.md
    ├── arb-view.md
    └── engineering-view.md
```

Each layer file declares elements and the relationships that originate from them. Cross-layer relationships (realization, assignment) live with the more concrete end to avoid duplication.

**Format:** textual / Mermaid / PlantUML is acceptable for v1.0.0. If the target EA tool requires Open Group Exchange File Format (.xml), the structural content here translates straightforwardly — call it out in the `togaf-deliverable-map.md` as an export step rather than duplicating the model.

### Example element declaration (textual)

```markdown
## ApplicationComponent: order-service

- Layer: Application
- Realizes: BusinessService:order-capture (Business layer)
- Assigned-to: Node:order-service-container (Technology layer)
- Serving: BusinessProcess:place-order (Business layer)
- Composed-of: ApplicationService:validate-order, ApplicationService:persist-order
- Depends-on (Flow → ): ApplicationComponent:payment-service
- NFR load (from 03-nfr-mapping.md): NFR-01, NFR-02, NFR-07
```

## `togaf-deliverable-map.md`

Map every numbered artifact to its TOGAF phase and Architecture Content Framework deliverable.

```markdown
# TOGAF Deliverable Map

| Artifact | TOGAF ADM phase | Architecture Content Framework deliverable |
|----------|-----------------|--------------------------------------------|
| 00-scope-and-context.md | Preliminary + Phase A (Vision) | Statement of Architecture Work; Architecture Vision |
| 01-requirements.md | Phase A / Requirements Management | Architecture Requirements Specification |
| 02-nfr-specification.md | Phase A / Requirements Management | Architecture Requirements Specification (NFR section) |
| 03-nfr-mapping.md | Phases B–D | Architecture Definition Document (Quality/NFR cross-cuts) |
| 04-solution-overview.md | Phases B–D | Architecture Definition Document (Solution overview) |
| 05-tech-selection-rationale.md | Phase D + E | Architecture Definition Document; Opportunities & Solutions (tech tradeoffs) |
| 06-descoped-and-deferred.md | Phase E / Opportunities & Solutions | Architecture Roadmap (explicit exclusions) |
| 07-c4-context.md | Phase B (Business Architecture) | Business Architecture catalogs and diagrams (context subset) |
| 08-c4-containers.md | Phases C & D | Application / Technology Architecture diagrams |
| 09-component-specifications.md | Phase C (Application Architecture) | Application Components catalog |
| 10-data-model.md | Phase C (Data Architecture) | Data Entity / Business Function matrix; Logical Data Model |
| 11-interface-contracts.md | Phase C | Application Interface catalog |
| 12-sequence-diagrams.md | Phases B & C | Process flow diagrams; Application sequence diagrams |
| 13-deployment-view.md | Phase D (Technology Architecture) | Technology Architecture diagrams |
| 14-requirements-traceability-matrix.md | Requirements Management (continuous) | Architecture Requirements Specification traceability |
| 15-integration-plan.md | Phase D + E | Technology Architecture; Interoperability Requirements |
| 16-migration-plan.md | Phase F (Migration Planning) | Implementation & Migration Plan |
| 17-risk-register.md | Phase E + F | Risks and Issues register |
| adrs/ | All phases | Architecture Building Block decisions |
| 99-solution-architecture-document.md | Phase G (Implementation Governance) | Architecture Contract |

## Export notes
- Target EA tool: [e.g., Sparx Enterprise Architect 16]
- Export format: Open Group ArchiMate Exchange File Format (XML) — translate the textual model at handover
- Viewpoint export: generate CIO/ARB/Engineering viewpoint images from the tool after import
```

## Pressure Responses

### "Let's just pick the closest ArchiMate relationship — tooling won't notice"

**Response:** "Tooling won't, reviewers will. A mislabelled relationship ripples: composition vs aggregation changes generated documentation, impact analysis, and cost rollups in mature EA suites. Five minutes on the right relationship saves a week of cleanup downstream."

### "We don't need a separate CIO view; just colour-code the master"

**Response:** "A CIO view is defined by the concerns it addresses (strategy, investment, risk) and the elements it *selects* for those concerns. A colour-coded master shows everything in different paints; a viewpoint shows what matters and hides what doesn't. If the CIO's question is 'are we aligned with the strategy,' they need to see capabilities and business services, not deployment nodes."

### "TOGAF phase mapping is bureaucratic"

**Response:** "It's bureaucratic when it's make-work. It's valuable when an ARB needs to see that each artifact has a phase home — it tells them which gates the artifact has passed. Fifteen minutes of mapping here saves an ARB meeting spent arguing about whether this is Phase B or Phase C."

## Anti-Patterns to Reject

### ❌ Element on wrong layer

Database on Business layer; service on Strategy layer; process on Technology layer. Layer is a commitment, not a preference.

### ❌ Composition and aggregation used interchangeably

Part of how ArchiMate tooling rolls up cost, risk, and coverage depends on these being distinct.

### ❌ Viewpoint that shows everything

A viewpoint is a *subset*. If it doesn't hide anything, it's the master model in a different hat.

### ❌ Partial phase mapping

"Most artifacts have a phase; the rest are 'various'." Every artifact has a phase (or is explicitly out-of-scope for ADM, recorded as such).

### ❌ "ArchiMate said" without an exchange path

If the design mentions ArchiMate but produces only prose, the EA tool ingestion path is unclear. Name the target tool and the export format — even if export is a manual later step.

## Scope Boundaries

**This skill covers:**

- ArchiMate layer, element, and relationship discipline
- Viewpoint subsets (CIO, ARB, Engineering as defaults)
- TOGAF ADM phase mapping
- Architecture Content Framework deliverable mapping

**Not covered (v1.0.0):**

- Full ADM phase automation (driving each phase as its own workflow)
- ARB submission template packaging
- ArchiMate Exchange File Format XML generation (produce textual model; tool-team translates)
- TOGAF Preliminary / Phase H (architecture change management) — these are governance cycles beyond a single solution design
