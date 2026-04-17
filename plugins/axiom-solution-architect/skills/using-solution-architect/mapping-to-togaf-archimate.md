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

## ArchiMate Layer and Aspect Taxonomy (short reference)

ArchiMate 3.2 organises the language into **layers** (Strategy, Business, Application, Technology, Physical) and **aspects** (Motivation, Implementation & Migration) that cut across layers. We use "layer" loosely below to include aspects for convenience.

| Layer / Aspect | Purpose | Typical elements |
|----------------|---------|------------------|
| Motivation (aspect) | Why — goals, requirements, constraints, drivers, outcomes | Stakeholder, Driver, Assessment, Goal, Outcome, Principle, Requirement, Constraint, Value, Meaning |
| Strategy (layer) | Long-term positioning — capabilities, courses of action, resources | Capability, CourseOfAction, Resource, ValueStream |
| Business (layer) | Organisational behaviour and structure | BusinessActor, BusinessRole, BusinessProcess, BusinessService, BusinessFunction, BusinessObject (Contract is a specialisation of BusinessObject) |
| Application (layer) | Software components and their interaction | ApplicationComponent, ApplicationService, ApplicationInterface, ApplicationFunction, ApplicationProcess, DataObject |
| Technology (layer) | Runtime platform — deployment, infrastructure | Node, Device, SystemSoftware, Network, CommunicationNetwork, TechnologyService, Artifact |
| Physical (layer) | Physical world — manufacturing, IoT, logistics (rare) | Equipment, Facility, DistributionNetwork, Material |
| Implementation & Migration (aspect) | Transformation work — packages, deliverables, plateaus, gaps | WorkPackage, Deliverable, ImplementationEvent, Plateau, Gap |

**How axiom-solution-architect artifacts bind to Motivation:**

- `01-requirements.md` → Motivation `Requirement` elements (functional requirements FR-NN)
- `02-nfr-specification.md` → Motivation `Requirement` elements typed as quality-attribute requirements (NFR-NN), plus `Goal` elements for the intent behind the NFRs
- Constraints (CON-NN) from `01-requirements.md` → Motivation `Constraint` elements
- `17-risk-register.md` → Motivation `Assessment` elements (a risk is a negatively-valued assessment of a driver)

Without a Motivation binding, requirements and risks have no home in the ArchiMate model; the ARB's first question ("where do the NFRs sit?") has no answer.

**TOGAF content metamodel bridge:** TOGAF's Architecture Content Framework names its own data element, `DataEntity`, in matrices such as the Data Entity / Business Function matrix. `DataEntity` maps onto ArchiMate `DataObject` at the Application layer. When the deliverable map references a TOGAF matrix that names `DataEntity`, read it as the `DataObject` bound to the same concept.

**Common layer mistakes (reject on sight):**

- Placing an application component on the Business layer because "it represents a business capability" — capabilities are Strategy layer elements; the app realising them is Application layer, related by `realization`.
- Placing a database on the Business layer because "it holds business data" — **business data** at the conceptual level is a `BusinessObject` (Business layer); the **logical data structure** is a `DataObject` (Application layer); the **physical deployable binary** of the database engine is an `Artifact` (Technology layer), assigned to a `Node`. Three distinct elements, three layers. If you can't say which one you mean, you are still at the conceptual level — which means `BusinessObject`.
- Placing infrastructure on the Application layer because "it's part of our stack" — nodes, devices, and networks are Technology layer.

## ArchiMate Relationships

The common structural relationships and their strict meanings:

| Relationship | Meaning | Example |
|--------------|---------|---------|
| Composition | Strong whole-part, part belongs exclusively to whole | Order composed of OrderLineItems |
| Aggregation | Whole-part, part can exist independently / be shared | Portfolio aggregates Products |
| Assignment | Active-element allocated to / deployed on another | ApplicationComponent *assigned-to* Node |
| Realization | More abstract element implemented by more concrete element | BusinessService realized by ApplicationService |
| Serving | One element provides its functionality to another (drawn provider → consumer) | ApplicationService *serving* BusinessProcess |
| Access | Behavioural element acts on a passive element | BusinessProcess *reads/writes* BusinessObject |
| Flow | Transfer between active elements | ApplicationComponent *flows to* ApplicationComponent |
| Triggering | Temporal or causal sequence | BusinessProcess triggers BusinessProcess |

**Note on "used-by":** ArchiMate 2.x had a distinct `UsedBy` relationship. ArchiMate 3.x folded it into `Serving`, drawn provider → consumer. There is no standalone `used-by` edge in 3.x — if you encounter "used-by" in older documentation, read it as `Serving` with the arrow reversed. Tooling on 3.1 / 3.2 (Archi, Sparx, BiZZdesign) will reject a `used-by` relationship.

**Rules:**

- Composition and aggregation are **not** interchangeable. Composition = "if the whole disappears, the part disappears too"; aggregation = "the part can outlive the whole."
- Assignment is for active-to-active deployment ("runs on"); realization is for abstract-to-concrete ("implements"). Not the same.
- If you can't tell which relationship applies, you don't understand the elements well enough — clarify the elements before drawing the line.

## Views and Viewpoints

ISO 42010 / ArchiMate distinction:

- A **viewpoint** is the template — stakeholders, concerns, model-kind, selection rules.
- A **view** is the concrete instance for a specific system.

We define three custom viewpoints below (concerns + selection rules) and produce one view per viewpoint for each solution. The view files are `archimate-model/viewpoints/<name>-view.md` — the filename aligns with the viewpoint, the contents are the view for this solution.

**Note on the standard catalogue:** ArchiMate 3.2 ships with ~23 standard viewpoints (Strategy, Capability Map, Goal Realization, Outcome Realization, Business Process Cooperation, Product, Application Cooperation, Information Structure, Technology, Implementation & Deployment, Migration, Layered, etc.). Prefer a standard viewpoint if the EA tool or the consuming ARB specifies one. The three below are pragmatic stakeholder defaults mapping approximately to CIO ≈ Strategy + Capability Map, ARB ≈ Layered, Engineering ≈ Application Cooperation + Technology.

### CIO viewpoint (instance: `cio-view.md`)

- Stakeholders: CIO, CTO, VP Engineering
- Concerns: strategic alignment, investment, risk, roadmap
- Model-kind: layered model emphasising Strategy and top-level Business
- Selects: Strategy layer (Capabilities, CourseOfAction), Business layer (major BusinessServices), Motivation (Goals, Assessments), Implementation & Migration (Plateaus)
- Suppresses: Application layer detail, Technology layer, element-level interfaces
- Output: `archimate-model/viewpoints/cio-view.md`

### ARB viewpoint (instance: `arb-view.md`)

- Stakeholders: Architecture Review Board
- Concerns: standards adherence, architectural integrity, cross-programme impact
- Model-kind: layered model with Application/Technology emphasis
- Selects: Application layer (Components, Interfaces), Technology layer (Nodes, Services), cross-layer relationships (realization, assignment), Motivation (Principles, Constraints)
- Suppresses: process-level Business detail unless directly relevant
- Output: `archimate-model/viewpoints/arb-view.md`

### Engineering viewpoint (instance: `engineering-view.md`)

- Stakeholders: engineering leads, senior developers
- Concerns: implementation detail, deployment, integration contracts, operational ownership
- Model-kind: application cooperation + technology infrastructure
- Selects: Application (Components, Services, Interfaces, DataObjects), Technology (Nodes, Artifacts, Networks), Implementation & Migration (WorkPackages when relevant)
- Suppresses: Strategy layer, high-level Motivation
- Output: `archimate-model/viewpoints/engineering-view.md`

Each view file names the **concerns it addresses**, the **elements and relationships it shows**, and **what is suppressed** — a view without these is a filter, not a view.

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

Each layer file declares elements and the relationships that originate from them. Cross-layer relationships (realization, assignment) live with the more concrete end — this keeps the realizing element's file self-contained, so the abstract element's file does not need to be edited every time a new realizer appears.

**Format:** textual / Mermaid / PlantUML is acceptable. If the target EA tool requires the **Open Group ArchiMate Model Exchange File Format** (MEF, file extension `.xml`), the structural content here translates straightforwardly — call the translation out in `togaf-deliverable-map.md` as an export step rather than duplicating the model.

### Example element declaration (textual)

Naming convention: we use `Element:instance-name` as the in-prose identifier (e.g., `BusinessService:order-capture`). The colon is a local convention, not part of the ArchiMate standard — tools may prefer separate type and name fields.

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
| 00-scope-and-context.md | Phase A (Architecture Vision) | Statement of Architecture Work; Architecture Vision |
| 01-requirements.md | Phase A / Requirements Management | Architecture Requirements Specification |
| 02-nfr-specification.md | Phase A / Requirements Management | Architecture Requirements Specification (NFR section) |
| 03-nfr-mapping.md | Phases B–D | Architecture Definition Document (Quality/NFR cross-cuts) |
| 04-solution-overview.md | Phases B–D | Architecture Definition Document (Solution overview) |
| 05-tech-selection-rationale.md | Phase C (application tier) + Phase D (technology tier) | Architecture Definition Document (tech rationale sections) |
| 06-descoped-and-deferred.md | Phase E (Opportunities & Solutions) | Architecture Roadmap (explicit exclusions); Transition Architecture inputs |
| 07-c4-context.md | Phase B (Business Architecture) | Business Architecture catalogs and diagrams (context subset) |
| 08-c4-containers.md | Phases C & D | Application / Technology Architecture diagrams |
| 09-component-specifications.md | Phase C (Application Architecture) | Application Components catalog |
| 10-data-model.md | Phase C (Data Architecture) | Data Entity / Business Function matrix; Logical Data Model |
| 11-interface-contracts.md | Phase C | Application Interface catalog |
| 12-sequence-diagrams.md | Phase B (business flows) + Phase C (application flows) | Process flow diagrams (B); Application sequence diagrams (C) |
| 13-deployment-view.md | Phase D (Technology Architecture) | Technology Architecture diagrams |
| 14-requirements-traceability-matrix.md | Requirements Management (continuous) | Architecture Requirements Specification traceability |
| 15-integration-plan.md | Phase D (primary) + Phase E (sequencing) | Technology Architecture; Interoperability Requirements |
| 16-migration-plan.md | Phase F (Migration Planning) | Implementation & Migration Plan |
| 17-risk-register.md | Phases E + F; Risk Management (continuous) | Risks and Issues register; Transformation risks |
| adrs/ | All phases | Architecture Decisions (Solution Building Block selection; cross-cutting architectural decisions) |
| 99-solution-architecture-document.md | Phase F (produced) + Phase G (governed under Architecture Contract) | Architecture Definition Document consolidated; Architecture Contract |

**Legend:**

- Phases listed indicate where the artifact is *primarily produced* in the ADM. Most artifacts are revisited during Requirements Management (continuous) and Phase H (Architecture Change Management).
- "Architecture Content Framework deliverable" names the formal TOGAF artifact that the numbered artifact maps onto — use these names when packaging for an ARB submission.
- ADRs record decisions about Solution Building Blocks (SBBs — the specific products, services, and patterns chosen for this solution). They are not Architecture Building Blocks (ABBs — reusable enterprise reference models), which are decided at enterprise level, not solution level.

## Export notes
- Target EA tool: `[target EA tool from 00-scope-and-context.md]` (e.g., Sparx Enterprise Architect, Archi, BiZZdesign, Visual Paradigm)
- Export format: **Open Group ArchiMate Model Exchange File Format** (MEF, `.xml`) — translate the textual model at handover
- Viewpoint export: generate CIO/ARB/Engineering view images from the tool after import
```

## Pressure Responses

### "Let's just pick the closest ArchiMate relationship — tooling won't notice"

**Response:** "Tooling won't, reviewers will. A mislabelled relationship ripples: composition vs aggregation changes generated documentation, impact analysis, and cost rollups in mature EA suites. Five minutes on the right relationship saves a week of cleanup downstream."

### "We don't need a separate CIO view; just colour-code the master"

**Response:** "A CIO view is defined by the concerns it addresses (strategy, investment, risk) and the elements it *selects* for those concerns. A colour-coded master shows everything in different paints; a viewpoint shows what matters and hides what doesn't. If the CIO's question is 'are we aligned with the strategy,' they need to see capabilities and business services, not deployment nodes."

### "TOGAF phase mapping is bureaucratic"

**Response:** "Phase mapping is the evidence that each artifact passed the right gate. When an ARB asks which phase a deliverable belongs to, the map is the answer — fifteen minutes of mapping here saves an ARB meeting spent arguing about Phase B versus Phase C."

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

**Not covered:**

- Full ADM phase automation (driving each phase as its own workflow)
- ARB submission template packaging
- ArchiMate Model Exchange File Format (MEF) XML generation (produce textual model; tool-team translates)
- TOGAF Preliminary (establishing the enterprise's architecture capability) and Phase H (architecture change management) — these are governance cycles that run once per organisation / continuously, not within a single solution design
