# Triaging Input Maturity

## Overview

**Your job is classification, not design.** Solution architecture fails fast when the wrong workflow is chosen for the input shape. Triage decides the workflow before a single artifact is produced.

**Core principle:** Produce `00-scope-and-context.md` and `01-requirements.md` before anything else. Even under pressure to "just start designing."

## When to Use

Use this skill when:

- A new design request arrives (brief, HLD, epic, change request)
- You are tempted to skip straight to C4 diagrams or tech selection
- The user says "just design it, I don't have time for questions"
- Input maturity is mixed (some sections detailed, others missing)

## Input Classification

Classify the input on four axes. Record the classification in `00-scope-and-context.md`.

### 1. Shape

| Shape | Signals | Implication |
|-------|---------|-------------|
| Business/problem brief | Prose, stakeholder asks, no structure | High elicitation cost; expect 5-10 clarifying questions |
| High-level design | Structured sections, major components named, some decisions made | Elicit gaps; respect existing decisions unless they conflict with requirements |
| Epic / feature spec | Scoped capability, existing system context | Brownfield workflow; scope the change, not the host system |
| Brownfield change request | Modification to running system | Requires archaeologist context; integration & migration mandatory |

### 2. Greenfield vs brownfield

- **Greenfield:** no existing system to integrate with. Skip `16-migration-plan.md`.
- **Brownfield:** integrates with or modifies an existing system. `16-migration-plan.md` is required. Check for archaeologist output:

```bash
# Look for archaeologist workspace
ls docs/arch-analysis-*/ 2>/dev/null
ls **/02-subsystem-catalog.md 2>/dev/null
```

If no archaeologist output exists and the brownfield change is non-trivial, recommend running `/system-archaeologist` first rather than proceeding blind.

### 3. Enterprise context

Activate `mapping-to-togaf-archimate` in the workflow plan if any of these appear in the input or the user's framing:

- TOGAF, ADM, phase A/B/C/D/E/F/G/H
- ArchiMate, viewpoint, Sparx EA, Archi, BiZZdesign
- ARB, Architecture Review Board, architecture governance
- "enterprise architecture", "target state", "current state"
- Explicit phase gates or stage-gate process

Otherwise, keep the TOGAF/ArchiMate workflow out — unnecessary overhead for product-engineering work.

### 4. Scope size

| Size | Signal | Workflow adjustment |
|------|--------|---------------------|
| XS | Single component change | Minimal artifact set — can skip C4 containers, deployment view |
| S | Single bounded context | Full artifact set, single-page diagrams |
| M | Multi-service feature | Full artifact set, expect 3-8 ADRs |
| L | New subsystem or major rework | Full artifact set, expect 8-15 ADRs, detailed migration |
| XL | Greenfield system or org-wide | Decompose into sub-projects and triage each — do not try to fit in one SAD |

**If you classify XL, stop and decompose.** A single solution architecture package cannot cover multiple independent subsystems usefully. Run triage per sub-project.

## The Required Deliverables

Triage must produce all three before handing off:

### `00-scope-and-context.md`

```markdown
# Scope and Context

## Problem statement
[One paragraph — what is being solved and why, in the user's words]

## Input classification
- Shape: [brief | HLD | epic | change request]
- Mode: [greenfield | brownfield]
- Enterprise context: [yes | no] — [if yes, which signals]
- Scope size: [XS | S | M | L | XL]
- Archaeologist context available: [yes → path | no | not applicable]

## In scope
- [Bullet list of capabilities / components / concerns that this design covers]

## Out of scope
- [Explicitly excluded capabilities / concerns]

## Stakeholders
- [Role / team / individual] — [interest]

## Assumptions
- [Numbered list — each must be confirmable with the user]
  1. …

## Open questions
- [Numbered list — these must be resolved before assembly]
  1. …
```

### `01-requirements.md`

```markdown
# Requirements

## Functional requirements
FR-01  [capability in one sentence, user-facing where possible]
FR-02  …

## Non-functional requirements
(See `02-nfr-specification.md` for quantified detail.)
NFR-01 [category — performance | availability | security | …]
NFR-02 …

## Constraints
CON-01 [regulatory, contractual, technical, or organisational constraint]
CON-02 …
```

Requirement IDs (FR-NN, NFR-NN, CON-NN) are load-bearing — the RTM and component specs reference them.

### Workflow plan

Emit a short plan of which specialist skills run next:

```markdown
# Workflow plan

1. quantifying-nfrs               → 02-nfr-specification.md, 03-nfr-mapping.md
2. resisting-tech-and-scope-creep → 04-solution-overview.md, 05-tech-selection-rationale.md, 06-descoped-and-deferred.md
3. (router-owned)                 → 07-c4-context.md, 08-c4-containers.md, 09-component-specifications.md, 10-data-model.md, 11-interface-contracts.md, 12-sequence-diagrams.md, 13-deployment-view.md
4. writing-rigorous-adrs          → adrs/   (runs throughout as decisions arise)
5. maintaining-requirements-traceability → 14-requirements-traceability-matrix.md
6. designing-for-integration-and-migration → 15-integration-plan.md, [16-migration-plan.md if brownfield], 17-risk-register.md
7. [if enterprise] mapping-to-togaf-archimate → archimate-model/, togaf-deliverable-map.md
8. assembling-solution-architecture-document → 99-solution-architecture-document.md
```

Omit steps that don't apply (e.g., migration for greenfield, TOGAF binding for non-enterprise).

## Asking Clarifying Questions

**One question per message. Prefer multiple choice. Stop after the highest-value questions are answered.**

Do not ask questions whose answers will not change the design. "What colour should the UI be?" is not triage; it's a taste check.

### High-value question template

- **Scale:** "What request rate and data volume are we designing for — roughly (a) <100 req/s, (b) 100-10k, (c) 10k-100k, (d) >100k?"
- **Availability:** "What uptime target — (a) best effort, (b) 99%, (c) 99.9%, (d) 99.99%+?"
- **Latency:** "P99 latency ceiling for the primary interaction — (a) <50ms, (b) <200ms, (c) <1s, (d) >1s acceptable?"
- **Data residency:** "Any region/residency constraints — (a) none, (b) EU only, (c) specific sovereignty rules?"
- **Brownfield integration:** "Which existing systems must this integrate with — list them, and which ones we own vs vendor?"

Record answers inline as they come and update `00-scope-and-context.md` / `01-requirements.md` directly.

## Pressure Responses

### "Just design it, I don't have time for questions"

**Response:** "I'll produce `00-scope-and-context.md` with your brief as-is and my assumptions explicit. Every assumption becomes an open question for you to confirm later. Without triage, we ship a design that answers the wrong question."

Proceed to emit `00-scope-and-context.md` and `01-requirements.md` with best-effort content. Every gap becomes an explicit assumption in the assumptions list and an item in the open-questions list. Never fabricate requirements to fill the list — mark them `[ASSUMED]`.

### "It's obvious what's needed, skip the requirements list"

**Response:** "The RTM in `14-` and the consistency gate in the assembly step both depend on requirement IDs. Skipping this breaks traceability end to end. I'll keep it minimal but complete."

### "We already picked [tech]; no need to document constraints"

**Response:** Constraints and tech selection are different artifacts. Constraints go in `01-requirements.md` (`CON-NN`). The pre-picked tech is recorded in `05-tech-selection-rationale.md` by `resisting-tech-and-scope-creep`, which will check whether the pre-pick actually satisfies the constraints.

## Anti-Patterns to Reject

### ❌ Skipping `00-scope-and-context.md`

Going straight to C4 diagrams or tech selection. The assembly step's consistency gate will fail, and downstream packs (security, technical-writer) won't have the scope statement they consume.

### ❌ Requirement IDs that don't trace

`FR-01: "good performance"` is not a requirement; it's a wish. Performance goes in `02-nfr-specification.md` with a measurement method.

### ❌ Inflating the assumptions list to avoid asking questions

Every `[ASSUMED]` is a question you should have asked. Three or four is normal; twenty means you're avoiding the user.

### ❌ Triaging "XL" and pressing on

An XL-scope design compressed into one SAD will be incoherent. Decompose or fail honestly.

## Scope Boundaries

**This skill covers:**

- Input classification (shape, mode, enterprise, size)
- Scope statement (`00-`) and requirements (`01-`) production
- Workflow plan emission
- Clarifying-question elicitation

**Not covered:**

- NFR quantification (use `quantifying-nfrs`)
- Design choices (use `resisting-tech-and-scope-creep`)
- Decomposition of XL scope into sub-projects (triage flags it; user decides how to decompose)
