# Triaging Input Maturity

## Overview

**Your job is classification, not design.** Solution architecture fails fast when the wrong workflow is chosen for the input shape. Triage decides the workflow before a single artifact is produced.

**Input maturity** is the ratio of specified-to-unspecified across the four
axes below: shape (how structured), mode (greenfield/brownfield context known),
enterprise (governance context known), and scope (sized honestly). A mature
input is complete enough across these four that triage can emit `00-` and
`01-` with fewer than five `[ASSUMED]` markers. An immature input needs
elicitation; a contradictory input needs a stakeholder round-trip (see
*Handling contradictory input* below).

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
| Verbal / meeting-notes | No written brief; hallway / meeting origin | Transcribe input first, get confirmation in writing; then re-classify |
| Implementation-ready ticket | Brief names components, APIs, tables before a single requirement | Back out the *why*; re-derive requirements; treat named tech as `[ASSUMED]` in `01-` so `resisting-tech-and-scope-creep` can check it against actual drivers |

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

**Keyword presence alone does not activate enterprise mode.** A stakeholder saying
"ArchiMate" in passing, or the brief using the phrase "target state", is a
*signal to investigate* — not a trigger. Activate `mapping-to-togaf-archimate`
only when one of the router's four activation criteria is actually true:

- ARB submission is a release gate for this project
- The customer organization publishes a TOGAF-aligned deliverable set (even if not named as such)
- A governing enterprise-architecture function must countersign the SAD
- ArchiMate models are a required artifact for downstream tooling

Keywords worth probing for (but never sufficient on their own): TOGAF, ADM,
phase A/B/C/D/E/F/G/H, ArchiMate, viewpoint, Sparx EA, Archi, BiZZdesign, ARB,
Architecture Review Board, architecture governance, "enterprise architecture",
"target state", "current state", explicit phase gates. When any of these
appear, ask which of the four activation criteria the project actually meets.

The activation decision is recorded explicitly in `00-scope-and-context.md`
as either `Enterprise: activated — [driver]` (naming which of the four
criteria applied) or `Enterprise: not activated — [reason]` (see *Output
contract* below). Silence on this line is a gate failure. Otherwise, keep the
TOGAF/ArchiMate workflow out — unnecessary overhead for product-engineering
work.

### 4. Scope tier

The scope tier determines which artifacts the consistency gate requires. Use
the router's triggers verbatim — downstream gates read the tier from `00-` and
enforce the artifact subset for it.

| Tier | Trigger | Workflow adjustment |
|------|---------|---------------------|
| XS | Single-component change, ≤1 integration, no new data, no new NFR targets | Minimal artifact set; ADRs only if a decision is made |
| S | ≤3 components, ≤2 integrations, existing NFR envelope | XS set + tech-selection rationale and interface contracts |
| M | New subsystem, new integrations, or new NFR targets | Full artifact set, expect 3-8 ADRs |
| L | Cross-system, multiple new services, new data stores | Full artifact set + sequence diagrams + C4 component view, expect 8-15 ADRs, detailed migration |
| XL | Governed by ARB / TOGAF / regulator | L set + ArchiMate model + TOGAF deliverable map. Decompose into sub-projects if multiple independent subsystems are in play — do not try to fit in one SAD. |

**Tier promotion.** If an ADR or `04-solution-overview.md` references an
artifact from a higher tier (e.g., an XS-tier workflow ends up citing a
deployment-view decision), the tier is promoted — the higher-tier artifacts
become required. Record the promotion in `00-` with a one-line rationale.

**If you classify XL with multiple independent subsystems, stop and decompose.**
A single solution architecture package cannot cover multiple independent
subsystems usefully. Run triage per sub-project.

### Output contract — tier and enterprise activation in `00-`

`00-scope-and-context.md` MUST carry two explicitly-labelled lines for the
consistency gate to read mechanically:

- `scope_tier:` — one of `XS | S | M | L | XL`, matching the trigger table above.
- `Enterprise: activated — [driver]` *or* `Enterprise: not activated — [reason]`
  — where `[driver]` is one of the four router-declared activation criteria
  (ARB release gate, TOGAF-aligned deliverable set, EA countersign required,
  ArchiMate required for tooling) and `[reason]` is a short justification
  (typically "no ARB, no TOGAF tooling, no EA countersign required").

These lines live in the `## Input classification` block (see the `00-`
template below). A gate report that cannot find them either fails Check 1 or
treats the tier as unknown — which blocks file-presence validation for the
entire workflow.

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
- scope_tier: [XS | S | M | L | XL]  — per the router's trigger table
- Enterprise: [activated — driver | not activated — reason]  — using the router's four activation criteria
- Archaeologist context available: [yes (fresh ≤ 30 d) → path | yes (stale > 30 d) → path + re-run recommendation | no | not applicable]

## In scope
- [Bullet list of capabilities / components / concerns that this design covers]

## Out of scope
- [Explicitly excluded capabilities / concerns]

## Stakeholders

Split into three groups so validation sign-off is unambiguous:

- **Accountable (validates outcome)** — one named role who signs off "this solves the stated problem" at end of delivery. Typical: product owner, sponsoring exec, customer principal.
- **Responsible (builds / runs)** — the teams that implement and operate.
- **Consulted / informed** — everyone else with a legitimate stake.

Accountable stakeholders must be identifiable by name or role, not "the
business" — that's the validation-theatre anti-pattern.

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
CON-NN are binding limits on the solution. Sub-tag each constraint so the
tech-selection step can tell which are negotiable.

- `CON-REG-NN`  Regulatory (GDPR, HIPAA, SOX, export control). Non-negotiable.
  **Compliance frameworks (SOC 2, HIPAA, PCI-DSS, GDPR, data residency, retention
  regimes) are recorded here as `CON-REG-*` and traced through `02-` → `14-` → `17-`.**
  The router's "Compliance drivers" section refers to these as `CON-*` generically;
  in this pack they always carry the `-REG` sub-tag so the RTM and the gate can
  filter them distinctly from contractual or technical constraints.
- `CON-CTR-NN`  Contractual (customer SLA, vendor lock-in, license). Binding, usually time-boxed.
- `CON-TEC-NN`  Technical (existing platform, corporate SSO, vendor stack). Sometimes negotiable — `resisting-tech-and-scope-creep` re-tests these against alternatives.
- `CON-ORG-NN`  Organisational (team skillset, corporate standards, political). Frequently the weakest constraint; cite the specific policy or stakeholder if it is load-bearing.

Example:

- CON-REG-01  EU personal data must remain in eu-west-*. (GDPR Art. 44–49.)
- CON-CTR-01  Top-3 customer SLA requires 99.9% monthly availability. (See NFR-07.)
- CON-TEC-01  Must deploy on corporate EKS fleet v1.27+. (No greenfield infra.)
- CON-ORG-01  Owning team is a 4-person Go shop. (No polyglot unless justified.)
```

Requirement IDs (FR-NN, NFR-NN, CON-*-NN) are load-bearing — the RTM and component specs reference them.

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

**One question per message. Prefer multiple choice. Stop after the highest-value
questions are answered.** Cap at **five rounds** of clarification; beyond that,
proceed with explicit `[ASSUMED]` markers rather than blocking indefinitely.
Five unanswered questions is itself a finding about input maturity and should
be recorded in `00-scope-and-context.md` open questions.

Do not ask questions whose answers will not change the design. "What colour should the UI be?" is not triage; it's a taste check.

### High-value question template

- **Scale:** "What request rate and data volume are we designing for — roughly (a) <100 req/s, (b) 100-10k, (c) 10k-100k, (d) >100k?"
- **Availability:** "What uptime target — (a) best effort, (b) 99%, (c) 99.9%, (d) 99.99%+?"
- **Latency:** "P99 latency ceiling for the primary interaction — (a) <50ms, (b) <200ms, (c) <1s, (d) >1s acceptable?"
- **Data residency:** "Any region/residency constraints — (a) none, (b) EU only, (c) specific sovereignty rules?"
- **Brownfield integration:** "Which existing systems must this integrate with — list them, and which ones we own vs vendor?"

Record answers inline as they come and update `00-scope-and-context.md` / `01-requirements.md` directly.

## Handling contradictory input

Contradictions are not missing-info and cannot be covered by `[ASSUMED]`. They
require a stop-and-ask, recorded as a blocking open question.

**Detect:**

- Two sections of the input disagree on a numeric target (e.g., scale vs. latency vs. budget).
- Pre-picked tech contradicts a stated constraint (e.g., "use DynamoDB" with `CON-REG-NN: EU data residency, no AWS global services`).
- A stakeholder ask contradicts a contractual constraint (e.g., "reduce cost" vs. an existing customer SLA).
- Greenfield framing with brownfield specifics ("new system" but "must use existing auth service").

**Response:**

1. Record the contradiction in `00-scope-and-context.md` under a new `## Contradictions` heading, citing both sources verbatim.
2. Do **not** guess. Pick-one-and-proceed produces a design that answers the wrong half of the brief.
3. Emit the clarifying question as a blocker — the workflow plan cannot proceed past `triaging-input-maturity` until each contradiction has a resolution.
4. If the stakeholder is unavailable and the work must proceed, fork the design: emit two `00-scope-and-context.md` variants (one per interpretation), explicitly flagged, and require the stakeholder to pick before assembly.

A contradiction silently resolved in favour of one side is a scope decision
made by the architect on behalf of the business — out of role.

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

### ❌ Refusing to start without a complete brief

The symmetric opposite of assumption-spam. If the input is 70% clear, produce
`00-` and `01-` covering the 70%, mark the 30% as open questions, and proceed.
Waiting for perfect input is its own avoidance pattern — the stakeholder often
doesn't know the answer until they see a draft.

### ❌ Triaging "XL" and pressing on

An XL-scope design compressed into one SAD will be incoherent. Decompose or fail honestly.

## Re-triaging mid-design

Input changes after triage are normal, not failures. When the brief, scope, or
constraints change after `00-` / `01-` exist:

1. **Annotate** `00-scope-and-context.md` under a new `## Change log` heading —
   append the change, date, source, and the axes affected (shape / mode /
   enterprise / scope).
2. **Reclassify** on the four axes. A change from S → L scope means workflow
   adjustment (more ADRs, full migration plan); a change from greenfield →
   brownfield means re-running the archaeologist check.
3. **Re-run dependent steps.** If constraints changed, `resisting-tech-and-scope-creep`
   re-runs. If NFRs changed, `quantifying-nfrs` re-runs. If requirements
   changed, the RTM re-runs orphan detection (see
   `maintaining-requirements-traceability` change-propagation section).
4. **Update the workflow plan** — steps that became newly-applicable (e.g.,
   TOGAF binding after a governance change) are added; steps that became
   inapplicable are struck out, not silently removed.

A mid-design input change that does **not** trigger at least step 1 is a silent
scope change — the single most common way solution architecture packages become
inconsistent with the business.

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
