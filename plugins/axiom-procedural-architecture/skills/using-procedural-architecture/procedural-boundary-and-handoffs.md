---
name: procedural-boundary-and-handoffs
description: Where this pack stops. Declares what it owns, what it does not own, the eight adjacent territories with named handoff targets, the handoff pattern to use mid-task, and a boundary-violation smell catalog for detecting when the pack is colonising adjacent disciplines.
---

# Procedural Boundary and Handoffs

## Opening Warning

This pack is general enough to colonise its neighbours. Staged procedures are
everywhere — wizards, troubleshooting trees, curricula, approval pipelines,
runbooks — and the same structural discipline applies to all of them. That
generality is the pack's value. The risk is overapplication: continuing to
reason under this pack's epistemics past the point where another pack's apply.
The discipline of declaring boundaries is what keeps the pack useful rather
than imperial.

If you find yourself arguing about tech choices, rendering decisions, runtime
dynamics, or system topology while nominally inside this pack, you have crossed
a boundary. This sheet names those boundaries, the packs that own the other
side, and the pattern for getting back.

---

## What This Pack Owns

**The structural shape of staged procedures.** This is a precise claim.

This pack owns: **stage decomposition** (right grain, right order, declared entry
conditions, exit artifacts); **dependencies and ordering** (explicit, acyclic,
no silent coupling, preconditions met before use); **decision points** (what
information is required, whether options are MECE, whether branches are real);
**exit artifacts** (what each stage produces for the audit trail and downstream
inputs); **soundness and correctness invariants** (no deadlock, no orphan
stages, no unreachable exits); **capacity and bottleneck characterisation**
(Little's Law, utilisation-knee, discrete-event simulation when closed-form
queueing is insufficient); and **audience parameters** (prerequisites,
working-memory, error cost, reversibility appetite, latency tolerance, recovery
options) as explicit inputs to decomposition decisions.

All of this is structural. The pack reasons about the *shape* of a procedure —
not the content of its stages, not how it is rendered, not what happens to it
under continuous runtime dynamics.

---

## What This Pack Does Not Own

### 1. Code-implementation-plan critique → `axiom-planning`

`axiom-planning` is the code-implementation-plan instance of this pack's general
discipline. When the procedure is specifically an implementation plan for a code
change — task ordering, branch hygiene, plan-review heuristics — that is
`axiom-planning`'s job; it carries code-specific heuristics this pack does not
have. Cross-link both ways: `plan-review` may cite this pack for structural
smells; this pack defers when the decomposition is a code-change plan.

### 2. System shape → `axiom-system-architect`

Procedure shape and system shape are different objects of analysis. A procedure
is a directed flow of stages with state and decision points. A system is a
collection of components with responsibilities and interfaces. When the question
is "what services should the backend split into?" or "where should this module
boundary be?", that is `axiom-system-architect` — `/system-architect`.

### 3. Continuous-time execution dynamics → `yzmir-simulation-foundations`

This pack stops where continuous-time models begin: ODEs, control loops,
real-time feedback, stability analysis, vector fields. The boundary is testable:
if you can draw the procedure as a directed graph of stages, you are here; if
you need a vector field or a differential equation, you are not. Slash command:
`/simulation-foundations`.

### 4. Game-style emergent flow → `bravos-simulation-tactics`

Player-driven systems and emergent rule interactions involve stakeholder dynamics
this pack does not model. `bravos-simulation-tactics` handles systems where
behaviour emerges from actor interactions rather than from the decomposition
itself. Slash command: `/simulation-tactics`.

### 5. Rendering a procedure as documentation prose → `muna-technical-writer`

This pack produces the structural artifact. `muna-technical-writer` renders it
as prose: document design, writing register, structure, clarity for human
readers. Slash command: `/technical-writer`.

### 6. Rendering a procedure as a wizard UI → `lyra-ux-designer`

This pack determines what the wizard asks and in what order. `lyra-ux-designer`
determines how it asks: screen layout, interaction patterns, progressive
disclosure in the UI sense, accessibility, platform conventions. Slash command:
`/ux-designer`.

### 7. Rendering a procedure as site information architecture → `lyra-site-designer`

The structural decomposition is upstream input. The site's navigation,
hierarchy, and reading paths for readers navigating static pages are
`lyra-site-designer`'s job. Slash command: `/site-designer`.

### 8. Content judgement inside a stage → domain packs

This pack audits structure. "Is Kafka the right technology here?", "what does a
correct health check cover?", "which authentication method is technically
superior?" — these are domain-content questions. Common redirects:

- Web services and APIs → `axiom-web-backend`
- Rust systems work → `axiom-rust-engineering`
- Python engineering → `axiom-python-engineering`
- LLM / AI engineering → `yzmir-llm-specialist`
- Security and threat modelling → `ordis-security-architect`
- ML production → `yzmir-ml-production`

When the critic finds a structural defect whose remediation requires
domain-content judgment: document the structural finding with severity and
evidence, note that domain-content judgment is required, name the domain pack.
Do not adjudicate content from inside this pack.

---

## The Handoff Pattern

When you find yourself across a boundary mid-task:

1. **Stop.** Do not continue reasoning under this pack's epistemics past the
   boundary. The most common failure mode is to keep going because the adjacent
   territory *looks* structural from inside this pack's frame.
2. **Name the boundary explicitly.** State which boundary has been crossed.
   "This is a content judgment about which queueing technology to use — outside
   this pack."
3. **Point to the right pack.** Name the pack and slash command. The receiving
   pack has the right heuristics; this pack does not.
4. **Return after the adjacent work is complete.** The adjacent pack's output —
   the tech choice, the rendered prose, the UX design — is the input to resume
   structural reasoning. Resume with that input, not before it.

Handoffs are not failures. A handoff is the pack working correctly. A pack that
never hands off is colonising.

---

## Boundary-Violation Smell Catalog

These are the specific ways this pack colonises adjacent territory. Each entry
gives the symptom, a test sentence that distinguishes procedural from the
adjacent concern, and the redirect.

---

### Smell 1: Tech-Choice Colonisation

**Symptom.** The conversation moves from "at this stage the user selects a
secrets backend" (structural: a decision point exists, its options need to be
MECE) to "the right secrets backend is Vault because X, Y, Z" (content: a
technology recommendation).

**Test.** Is this a claim about the *shape* of the decision point, or about
which option is *correct*? If correct, that is domain content.

**Redirect.** Note the structural finding. Hand the content judgment to
`axiom-web-backend`, `axiom-rust-engineering`, `yzmir-llm-specialist`, or the
appropriate domain pack.

---

### Smell 2: Rendering Colonisation

**Symptom.** The conversation moves from "this decision point must appear after
the user provides their region preference" (structural: ordering and
information-readiness) to "this option should be a radio button, not a dropdown"
(rendering: UI component choice).

**Test.** Is this about what information must be present before a decision
fires, or about how that decision is presented? If presentation, that is
rendering.

**Redirect.** Hand the UI question to `lyra-ux-designer` or the prose rendering
to `muna-technical-writer`.

---

### Smell 3: Dynamics Colonisation

**Symptom.** The conversation moves from "at 92% utilisation the legal-review
queue exhibits nonlinear wait-time growth" (structural-capacity: queueing
observation) to "what happens under a continuous feedback controller with
correlated bursts?" (execution dynamics: continuous-time model).

**Test.** Can this be answered by reasoning about stages and queues as discrete
objects, or does it require a continuous-time model? ODEs and vector fields are
not here.

**Redirect.** Hand the continuous-time dynamics question to
`yzmir-simulation-foundations`.

---

### Smell 4: Implementation-Plan Colonisation

**Symptom.** The procedure is a code-change plan and the conversation drifts
into code-specific concerns: task ordering around schema migrations, rollback
steps tied to database irreversibility. That is `axiom-planning` territory.

**Test.** Is this procedure specifically a code-implementation plan for a code
change? If yes, `axiom-planning` has code-specific heuristics this pack does
not.

**Redirect.** This pack's structural principles may be cited by `axiom-planning`
review, but code-specific plan work belongs to `axiom-planning`.

---

### Smell 5: Architecture Colonisation

**Symptom.** The conversation moves from "this approval stage requires input
from both engineering and legal reviewers before it fires" (structural: a join
dependency) to "engineering and legal review should be separate services with an
async queue" (system shape: a component boundary and integration decision).

**Test.** Is this about the *flow of stages in a procedure*, or about *how
system components are bounded and integrated*? Components and boundaries are
system shape.

**Redirect.** The structural join is this pack's output. Hand the
component-boundary question to `axiom-system-architect`.

---

### Smell 6: Site-IA Colonisation

**Symptom.** The conversation moves from "the onboarding procedure has six
stages with these dependencies" (structural: stages and ordering) to "should
each stage be its own page, or grouped under a parent route in the navigation?"
or "what should the breadcrumb hierarchy look like?" (site information
architecture: how the procedural shape is rendered into a navigable structure of
pages).

**Test.** Is the question about *what* the procedure does at each stage and how
stages depend on each other, or about *how* the procedure is rendered into a
hierarchy of pages, sections, and reading paths? Hierarchy of pages is site IA.

**Redirect.** Hand the navigation hierarchy, page grouping, and reading-path
questions to `lyra-site-designer`. The structural decomposition is upstream
input; the site shape is downstream output.

---

### Smell 7: Emergent-Flow Colonisation

**Symptom.** The conversation moves from "the procedure has these stages and
this audience" (structural: predetermined shape, declared audience parameters)
to "what happens when multiple actors interact under the procedure
simultaneously and the procedural shape itself emerges from those interactions
rather than being declared up front?" (emergent flow: the procedure is being
authored by the interaction, not designed in advance).

**Test.** Are stages predetermined and audience parameters known, or is the
procedure being authored by actor interaction in real time? If the latter, the
shape is emergent rather than declared.

**Redirect.** Hand the emergent-flow question to `bravos-simulation-tactics`.
This pack reasons about declared procedural shape; emergent shape under
multi-actor interaction is a different object.

---

## Inbound Relationships

This pack does not only hand off — it also receives. The most common upstream
source is `axiom-system-archaeologist`.

### From `axiom-system-archaeologist` — recovered procedures

When system archaeology surfaces a procedure-shaped object inside an existing
codebase (extracted control flow, a multi-stage workflow inferred from call
graphs, a wizard reconstructed from UI traversal), the archaeology output is the
input to this pack's critic role. The archaeologist's job ends at "here is the
procedure that exists"; this pack's job begins at "is the procedure that exists
structurally sound?"

**Handoff artifact shape.** What this pack expects to receive:

- A stage list (or prose that can be decomposed into stages without ambiguity).
- Dependencies and ordering as observed (even if implicit — the archaeologist
  may report "stages B and C both follow A but with no declared ordering
  between them").
- Decision points and branches as observed in the recovered procedure.
- Audience parameters if known; if not, the critic infers from context or flags
  the gap.

**What this pack produces in response.** A standard critic-role output — a
severity-rated findings list with evidence and a machine-readable summary —
covering the recovered procedure's structural defects (orphan stages,
re-entrancy blindness, escape-hatch overuse, MECE violations, dependency
inversions, etc.). The findings feed back to the archaeology consumer as
structural debt the codebase carries.

**Slash command.** `/system-archaeologist` for the recovery; `/review-decomposition`
for the structural critique of what was recovered.

---

## Cross-References

Outbound — packs named as handoff targets in this sheet:

- `axiom-planning` (`/axiom-planning`) — code-implementation-plan instances of procedural discipline
- `axiom-system-architect` (`/system-architect`) — system shape: components, services, modules
- `yzmir-simulation-foundations` (`/simulation-foundations`) — continuous-time dynamics, ODEs, control theory
- `bravos-simulation-tactics` (`/simulation-tactics`) — game-style emergent flow, player-driven systems
- `muna-technical-writer` (`/technical-writer`) — rendering procedures as documentation prose
- `lyra-ux-designer` (`/ux-designer`) — rendering procedures as wizard UI
- `lyra-site-designer` (`/site-designer`) — rendering procedures as site information architecture
- `axiom-web-backend`, `axiom-rust-engineering`, `axiom-python-engineering`,
  `yzmir-llm-specialist`, `ordis-security-architect`,
  `yzmir-ml-production` — domain packs for content judgement inside stages

Inbound — packs that feed this pack:

- `axiom-system-archaeologist` (`/system-archaeologist`) — recovered procedures extracted from existing codebases, handed in for structural critique

The routing table in [SKILL.md](SKILL.md) routes the symptom "is this question
really for this pack?" to this sheet. The other 12 reference sheets route to
this sheet when their findings require domain-content judgment that is outside
the structural audit's scope.
