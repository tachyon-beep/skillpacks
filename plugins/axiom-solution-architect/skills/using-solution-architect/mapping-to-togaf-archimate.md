# Mapping to TOGAF / ArchiMate

## Overview

**Enterprise architecture frameworks exist so that designs survive handoff.**

When a solution design will enter an organisation's EA function (ARB, enterprise repository, multi-project programme), ad-hoc diagrams fail. Artifacts must map to the framework the EA team uses — most commonly TOGAF (process + deliverable vocabulary) and ArchiMate (modelling language).

**Core principle:** Right layer, right element, right relationship, right viewpoint. Nothing "close enough."

- [ArchiMate Layer and Aspect Taxonomy](#archimate-layer-and-aspect-taxonomy-short-reference)
- [ArchiMate Relationships](#archimate-relationships)
- [Views and Viewpoints](#views-and-viewpoints)
- [Model Files](#model-files) — `archimate-model/` directory
- [`togaf-deliverable-map.md`](#togaf-deliverable-mapmd)
- [Pressure Responses](#pressure-responses)
- [Anti-Patterns to Reject](#anti-patterns-to-reject)
- [Stop Conditions](#stop-conditions)

## When to Use

See the router's Start Here (SKILL.md) if this is your first pass through the pack.

Activate only when `triaging-input-maturity` confirms enterprise context. The four activation criteria (`SKILL.md`) are:

- ARB submission is a release gate for this project
- The customer organization publishes a TOGAF-aligned deliverable set (even if not named as such)
- A governing enterprise-architecture function must countersign the SAD
- ArchiMate models are a required artifact for downstream tooling

**Keywords are probes, not triggers.** Phrases such as TOGAF, ADM, phase A–H, ArchiMate, Sparx EA / Archi / BiZZdesign, "enterprise architecture", "target state", "current state", "architecture governance", or explicit phase gates appearing in a brief mean *ask which of the four criteria applies*. They do not by themselves activate this skill. Activation is recorded in `00-scope-and-context.md` as `Enterprise: activated — [criterion]` or `Enterprise: not activated — [reason]`; silence is a gate failure.

Do not produce TOGAF/ArchiMate artifacts for product-engineering work without an EA consumer. Overhead without a reader is waste.

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
- `02-nfr-specification.md` → Motivation `Requirement` elements (NFR-NN), each related to a `Goal` element representing the quality-attribute intent via `Realization` or `Influence`. ArchiMate 3.2 has no standard stereotype for "NFR"; if the target EA tool supports local stereotypes, declare `«NFR»` as a tool convention in `togaf-deliverable-map.md` (this is a tool convention, not a standard element).
- Constraints (CON-NN) from `01-requirements.md` → Motivation `Constraint` elements
- `17-risk-register.md` → Motivation `Assessment` elements (a risk is a negatively-valued assessment of a driver)

Without a Motivation binding, requirements and risks have no home in the ArchiMate model; the ARB's first question ("where do the NFRs sit?") has no answer.

> **Note — TOGAF content metamodel bridge:** TOGAF's Architecture Content Framework calls its data element `DataEntity` (Data Entity / Business Function matrix). `DataEntity` maps to ArchiMate `DataObject` at the Application layer. Where the deliverable map references `DataEntity`, read `DataObject`.

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
| Influence | Motivation-aspect: an element positively or negatively affects another (drivers → goals, assessments → goals, NFRs → goals) | Driver *influences* Goal; Assessment *influences* Requirement |
| Association | Generic fallback when no stronger relationship applies; commonly used to link NFRs to the elements they constrain | Requirement *associated-with* ApplicationComponent |

`Specialisation` (one element is a subtype of another) is also a structural relationship in the standard but is rarely used in solution-level modelling; reserve it for type hierarchies (e.g., `Contract` specialises `BusinessObject`).

**Note on "used-by":** ArchiMate 2.x had a distinct `UsedBy` relationship. ArchiMate 3.x folded it into `Serving`, drawn provider → consumer. There is no standalone `used-by` edge in 3.x — if you encounter "used-by" in older documentation, read it as `Serving` with the arrow reversed. Tooling on 3.1 / 3.2 (Archi, Sparx, BiZZdesign) will reject a `used-by` relationship.

**Rules:**

- Composition and aggregation are **not** interchangeable. Composition = "if the whole disappears, the part disappears too"; aggregation = "the part can outlive the whole."
- Assignment is for active-to-active deployment ("runs on"); realization is for abstract-to-concrete ("implements"). Not the same.
- If you can't tell which relationship applies, you don't understand the elements well enough — clarify the elements before drawing the line.

## Views and Viewpoints

ISO 42010 / ArchiMate distinction:

- A **viewpoint** is the template — stakeholders, concerns, model-kind, selection rules.
- A **view** is the concrete instance for a specific system.

We produce views from the **standard ArchiMate 3.2 viewpoint catalogue** by default. ArchiMate 3.2 ships ~23 standard viewpoints (Strategy, Capability Map, Goal Realization, Outcome Realization, Business Process Cooperation, Product, Application Cooperation, Information Structure, Technology, Implementation & Deployment, Migration, Layered, etc.). Prefer a standard viewpoint if the EA tool or the consuming ARB specifies one.

The table below names the stakeholder alias, the standard viewpoint(s) it maps to, and the file used in `archimate-model/viewpoints/`. The file name aligns with the stakeholder alias; the contents are the view for this solution, selected using the standard viewpoint's rules.

| Stakeholder alias | Standard viewpoint(s) | Filename | Selection rule (summary) |
|---|---|---|---|
| CIO alias | Strategy; Capability Map; Goal Realization | `cio-view.md` | Strategy + top-level Business; suppress Application/Technology detail |
| ARB alias | Layered; Application Cooperation | `arb-view.md` | Application + Technology; cross-layer realization/assignment; Motivation principles/constraints |
| Engineering alias | Application Cooperation; Technology; Implementation & Deployment | `engineering-view.md` | Components, interfaces, nodes, artifacts; Work Packages when migration-relevant |

Each view file names the **concerns it addresses**, the **elements and relationships it shows**, and **what is suppressed** — a view without these is a filter, not a view.

### CIO alias view (instance: `cio-view.md`)

- Derived from: Strategy, Capability Map, Goal Realization viewpoints
- Stakeholders: CIO, CTO, VP Engineering
- Concerns: strategic alignment, investment, risk, roadmap
- Selects: Strategy layer (Capabilities, CourseOfAction), Business layer (major BusinessServices), Motivation (Goals, Assessments), Implementation & Migration (Plateaus)
- Suppresses: Application layer detail, Technology layer, element-level interfaces
- Output: `archimate-model/viewpoints/cio-view.md`

### ARB alias view (instance: `arb-view.md`)

- Derived from: Layered, Application Cooperation viewpoints
- Stakeholders: Architecture Review Board
- Concerns: standards adherence, architectural integrity, cross-programme impact
- Selects: Application layer (Components, Interfaces), Technology layer (Nodes, Services), cross-layer relationships (realization, assignment), Motivation (Principles, Constraints)
- Suppresses: process-level Business detail unless directly relevant
- Output: `archimate-model/viewpoints/arb-view.md`

### Engineering alias view (instance: `engineering-view.md`)

- Derived from: Application Cooperation, Technology, Implementation & Deployment viewpoints
- Stakeholders: engineering leads, senior developers
- Concerns: implementation detail, deployment, integration contracts, operational ownership
- Selects: Application (Components, Services, Interfaces, DataObjects), Technology (Nodes, Artifacts, Networks), Implementation & Migration (WorkPackages when relevant)
- Suppresses: Strategy layer, high-level Motivation
- Output: `archimate-model/viewpoints/engineering-view.md`

If the ARB or EA tool specifies a different standard viewpoint (e.g., Information Structure, Migration), produce an additional view under the standard viewpoint's name and cross-reference from the stakeholder alias file.

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

Each layer file declares elements and the relationships that originate from them. Place cross-layer relationships (realization, assignment) in the more-concrete element's file — this keeps the realizing element self-contained and prevents cascading edits to the abstract element's file every time a new realizer appears.

**Format:** Use textual prose, Mermaid, or PlantUML. If the target EA tool requires the **Open Group ArchiMate Model Exchange File Format** (MEF, file extension `.xml`), the structural content here translates straightforwardly — call the translation out in `togaf-deliverable-map.md` as an export step rather than duplicating the model.

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

### Worked layer binding — order-fulfilment capability (example)

This shows how a two-component slice (`order-service` + `payment-service`, plus their data and business context) binds across three ArchiMate layers and the Motivation aspect. Use it as a pattern check when building `application-layer.md`, `business-layer.md`, `technology-layer.md`, and the Motivation extract.

---

**`business-layer.md` (extract):**

```markdown
## BusinessService: order-capture

- Layer: Business
- Realized-by: ApplicationService:submit-order (Application layer)
- Serving: BusinessProcess:place-order
- Owner: order-fulfilment team

## BusinessObject: Order

- Layer: Business
- Accessed-by: BusinessProcess:place-order (read/write)
- Implemented-as: DataObject:order-record (Application layer)
```

**`application-layer.md` (extract):**

```markdown
## ApplicationComponent: order-service

- Layer: Application
- Realizes: BusinessService:order-capture (Business layer)
- Assigned-to: Node:order-service-container (Technology layer)
- Serving: BusinessProcess:place-order (Business layer)
- Composed-of: ApplicationService:submit-order, ApplicationService:query-order
- Depends-on (Flow →): ApplicationComponent:payment-service
- NFR load (from 03-nfr-mapping.md): NFR-01, NFR-07

## DataObject: order-record

- Layer: Application
- Implements: BusinessObject:Order (Business layer)
- Accessed-by: ApplicationComponent:order-service (read/write), ApplicationComponent:payment-service (read)
- Stored-on: Artifact:order-db-schema (Technology layer → postgresql-node)
```

**`technology-layer.md` (extract):**

```markdown
## Node: order-service-container

- Layer: Technology
- Assigned-from: ApplicationComponent:order-service (Application layer)
- Composed-of: SystemSoftware:jvm-21, Artifact:order-service-jar
- Hosted-on: Node:eks-node-group (eu-west-1)
- Network-boundary: vpc-private-subnet
```

**Motivation extract (inline in `business-layer.md` or a dedicated `motivation-aspect.md`):**

```markdown
## Requirement: NFR-01-latency

- Aspect: Motivation
- Maps-to: 02-nfr-specification.md → NFR-01 (P99 read latency ≤ 120 ms)
- Realizes: Goal:responsive-user-experience (via Realization)
- Associated-with: ApplicationComponent:order-service, ApplicationComponent:api-gateway

## Constraint: CON-REG-01-residency

- Aspect: Motivation
- Maps-to: 01-requirements.md → CON-REG-01 (EU data residency, GDPR Art. 44–49)
- Associated-with: Node:order-service-container, Artifact:order-db-schema
```

---

**Anti-pattern checks (apply after binding):**

- `order-service` is Application layer, not Business. The *capability* it provides (order-capture) is a `BusinessService` (Business layer) realized-by the component. Two distinct elements — do not collapse.
- `order-record` as a `DataObject` is Application layer. The conceptual `Order` as a `BusinessObject` is Business layer. They are connected via `Implements` — do not model one and claim it covers both.
- `order-service-container` is a `Node` at Technology layer, not Application. The component is `Assigned-to` the node; the node does not "contain" the component in the ArchiMate sense (that would require a different relationship).
- `NFR-01-latency` is a `Requirement` on the Motivation aspect, not a label on `order-service`. It `Realizes` the `Goal` for the quality attribute and is `Associated-with` the components that carry the NFR load. It is not an attribute of the component.
- CIO alias view (`cio-view.md`) suppresses `DataObject`, `Artifact`, `Node` — the CIO sees `BusinessService:order-capture` and the capability it realises, not the PostgreSQL node.

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
| 10-data-model.md | Phase C — Data Architecture (primary) + Application Architecture (entity ownership cross-cut) | Data Entity / Business Function matrix; Logical Data Model; Application Components → Data Entities ownership matrix |
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
- Target EA tool: *[from 00-scope-and-context.md]* (e.g., Sparx Enterprise Architect, Archi, BiZZdesign, Visual Paradigm)
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

### Element on wrong layer

Database on Business layer; service on Strategy layer; process on Technology layer. Layer is a commitment, not a preference.

### Composition and aggregation used interchangeably

Part of how ArchiMate tooling rolls up cost, risk, and coverage depends on these being distinct.

### Viewpoint that shows everything

A viewpoint is a *subset*. If it doesn't hide anything, it's the master model in a different hat.

### Partial phase mapping

"Most artifacts have a phase; the rest are 'various'." Every artifact has a phase (or is explicitly out-of-scope for ADM, recorded as such).

### "ArchiMate said" without an exchange path

If the design mentions ArchiMate but produces only prose, the EA tool ingestion path is unclear. Name the target tool and the export format — even if export is a manual later step.

### NFR typed as a tool-specific construct without a Motivation binding

"The tool has a `QualityRequirement` stereotype" is not a substitute for a `Requirement` element related to a `Goal`. Model the standard element first; the stereotype is a tool convention on top, not a replacement.

## Stop Conditions

| Condition | Response |
|-----------|----------|
| Target EA tool not identified | Do not produce the `archimate-model/` export step. Record `target-ea-tool: TBD` in `togaf-deliverable-map.md` export notes and raise an open question in `00-scope-and-context.md`. The model can still be produced in textual/Mermaid form; tool export is deferred. |
| ARB viewpoint requirements unknown | Produce the three stakeholder-alias views (CIO, ARB, Engineering) derived from the standard catalogue. Flag in the gate report (Check 8) that the ARB's preferred viewpoint set has not been confirmed — the ARB may require a different subset from the ArchiMate 3.2 catalogue. |
| TOGAF phase mapping conflicts with internal governance | Record the conflict in `togaf-deliverable-map.md` under a `## Conflicts` section; do not silently remap. Escalate to the EA function before the ARB submission. |
| Motivation aspect not modelled by consuming EA function | Produce the Motivation extract inline in the relevant layer file (most commonly `business-layer.md`) and flag in the export notes that the Motivation subset may need to be re-homed into a `motivation-aspect.md` file at import. Do not omit the Motivation binding — requirements and risks must map to Motivation elements. |

## Scope Boundaries

**This skill covers:**

- ArchiMate layer, element, and relationship discipline
- View production from the standard ArchiMate 3.2 viewpoint catalogue (stakeholder aliases: CIO, ARB, Engineering)
- TOGAF ADM phase mapping
- Architecture Content Framework deliverable mapping

**Not covered:**

- Full ADM phase automation (driving each phase as its own workflow)
- ARB submission template packaging
- ArchiMate Model Exchange File Format (MEF) XML generation (produce textual model; tool-team translates)
- TOGAF Preliminary (establishing the enterprise's architecture capability) and Phase H (architecture change management) — these are governance cycles that run once per organisation / continuously, not within a single solution design
