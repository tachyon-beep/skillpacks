---
description: Interactive elicitation for "what counts as a decision" in a system. Walks through producers, decision types, mandatory fields, decision boundary, and tier classification. Produces draft `00-scope-and-decisions.md` and `01-decision-log-schema.md` ready for refinement via the `decision-log-architecture` skill.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[system_or_component_name]"
---

# Design Decision Log Command

You are running an interactive elicitation to identify the decision points in a system that should be in the audit pipeline. The output is a draft of `00-scope-and-decisions.md` and `01-decision-log-schema.md` — the *first two artifacts* of the `audit-pipeline/` workspace, refined through the `decision-log-architecture` skill afterward.

This command is for the question "what should we audit?" If you already know what you're auditing and need to design the rest of the pipeline, use the `using-audit-pipelines` skill directly. For verification of an existing trail, use `/verify-integrity`. For scaffolding code, use `/scaffold-audit-trail`.

## Invocation Path

`/design-decision-log` is interactive — it works through `AskUserQuestion` to elicit information one piece at a time, then synthesises the result into the two starting artifacts. The user can short-circuit by pasting an existing system description; the command will then ask only the gaps.

## Preconditions

The command takes a single argument: a system or component name.

```bash
INPUT="${ARGUMENTS}"

if [ -z "${INPUT}" ]; then
  # AskUserQuestion: "Which system are we identifying decisions in?"
  :
fi
```

### Resume vs fresh

If `audit-pipeline/<INPUT>/00-scope-and-decisions.md` already exists, ask:

1. **Refine** — load the existing draft, ask only about gaps and ambiguities.
2. **Replace** — archive existing, start fresh.
3. **Discuss** — read the existing draft, identify weak points, suggest improvements without writing new files.

## Workflow

### Step 1 — System overview

```
AskUserQuestion: "What does this system do, in one or two sentences? Who uses it,
who relies on its outputs?"
```

The answer scopes the elicitation. A trading-execution engine has different decision shapes from a content-moderation service from a vehicle-routing optimiser.

### Step 2 — Producer enumeration

```
AskUserQuestion: "What components in the system make decisions? Examples: a
policy engine, a risk model, a workflow engine, a scheduler, a guard or
governor that approves/rejects proposed actions, a rule firing module."
```

For each named producer, capture:

- Component name (stable identifier; will become `producer_id`).
- Component owner (team / individual).
- Component's role.

### Step 3 — Decision type enumeration

For each producer, ask:

```
AskUserQuestion (per producer): "What decisions does <producer> make? For each:
  - A short stable name (will become decision_type, e.g., 'policy.evaluate').
  - What gets decided? (a verdict, a transition, a numeric output, a routing choice, ...)
  - Who or what consumes the decision?
  - Is the consumer in a different trust zone (different team, customer, downstream service, regulator)?"
```

For each decision type, capture:

- `decision_type` name.
- Output shape (allow/deny, transition, score, routing).
- Consumer.
- Trust-zone crossing.

### Step 4 — Regulatory or contractual drivers

```
AskUserQuestion: "Are any of these decisions subject to:
  - Regulatory audit (regulator name, framework)?
  - Contractual audit (customer's audit rights, service-level agreement)?
  - Legal hold or e-discovery exposure?
  - Compliance frameworks: SOC 2, HIPAA, PCI-DSS, GDPR, EU AI Act, NIS2, sectoral?"
```

Drivers determine tier; capture per decision_type if drivers vary.

### Step 5 — Decision boundary

For each decision_type, apply the boundary discipline from `decision-log-architecture.md`:

```
AskUserQuestion: "When <producer> makes a <decision_type>, what's the unit
of accountability?
  - Pattern A: one entry per externally-observable decision (rule firings inside
    collapse into evidence array)
  - Pattern B: one entry per atomic rule firing, plus a summary entry
  - Pattern C: external decision + reference to a separate reasoning log

Which fits, and why?"
```

Capture the choice + reasoning per decision_type.

### Step 6 — Mandatory and optional fields

For each decision_type, walk through the 13 mandatory fields from `decision-log-architecture.md`:

- `entry_id`, `entry_version`, `decision_type`, `producer_id`, `decided_at`,
  `inputs_commitment`, `ruleset_id`, `ruleset_version`, `code_version`, `output`,
  `output_hash`, `prev_hash`, `entry_hash`.

Most are uniform across decision_types and need no per-type input. Confirm:

- `inputs_commitment` — typical input size; inline-or-ref threshold; is the input set complete?
- `ruleset_id` / `ruleset_version` — does the producer have a ruleset registry? How does it version?
- `code_version` — git-sha + build provenance? Container image digest?

For optional fields (`request_id`, `caller_id`, `evidence`, `reasoning_ref`, `confidence`, `explanation`, `signed_by`, `tags`):

```
AskUserQuestion (per decision_type): "Which of these are needed?
  - request_id: linkage to broader request flow
  - caller_id: identity of who/what asked
  - evidence: structured rule firings or model features (Patterns A/B)
  - reasoning_ref: pointer to detailed reasoning (Pattern C)
  - confidence: probabilistic decisions
  - explanation: regulator-required explanation (EU AI Act, ECOA)
  - signed_by: per-entry signing in use
  - tags: routing, retention, sensitivity
"
```

### Step 7 — Forbidden fields and PII

```
AskUserQuestion: "Does the decision logically involve personally-identifiable
information (PII), commercial-sensitive data, or legally privileged content?
  - If yes, can we keep the audit pipeline PII-free by referencing subjects via
    opaque ids (segregated-PII architecture, recommended for new systems)?
  - Or do we need plaintext PII in the entry (legacy, regulator-required, etc.)?"
```

The answer drives `06-storage-and-retention.md` (cryptographic erasure vs segregated-PII vs redaction-with-witness).

### Step 8 — Tier classification

Apply the tier table from `using-audit-pipelines/SKILL.md`:

```
Suggest a tier based on:
  - Number of decision types and producers
  - Cross-trust-zone consumers
  - Regulatory drivers
  - Legal-hold / external-export requirements

AskUserQuestion: "Suggested tier: <tier>. Accept, promote, or override?"
```

Capture the tier and the trigger that selected it.

### Step 9 — Synthesise drafts

Write `audit-pipeline/<INPUT>/00-scope-and-decisions.md`:

- System overview
- Decision inventory (per `decision-log-architecture.md` table)
- Per-decision_type pattern (A / B / C) with reasoning
- Tier with trigger
- Out-of-scope items (events that are NOT decisions)

Write `audit-pipeline/<INPUT>/01-decision-log-schema.md`:

- Per-decision_type entry shape (mandatory + optional fields)
- Cross-cutting field constraints
- `entry_version` value (initially 1)
- Forbidden fields (explicit, with rationale)
- Validation strategy

Both are *drafts* — refinement comes from reading the `decision-log-architecture` skill in full and tightening the per-field validation rules, the `entry_version` migration policy, and the field-by-field constraints. State this in the document headers ("DRAFT — refine with `decision-log-architecture` skill before downstream work").

### Step 10 — Suggested next steps

```
The drafts of 00-scope-and-decisions.md and 01-decision-log-schema.md are ready.

Suggested next steps:
1. Refine via the `decision-log-architecture` skill (full pass through the sheet)
2. Then: `canonical-encoding-for-fingerprinting` -> 02-
3. Then: `fingerprint-chains-and-integrity` -> 03-
4. Continue through the spec set per `using-audit-pipelines/SKILL.md`
5. When 00-10 are complete, run the consistency gate to produce 99-
6. To scaffold code from the spec: /scaffold-audit-trail <component>
```

## Output Location

Drafts in `audit-pipeline/<INPUT>/`. The directory may be created if it does not exist.

## Common Mistakes (in elicitation)

| Mistake | Fix |
|---------|-----|
| Eliciting from one developer in isolation | The boundary between decision and event is contested; involve consumers (compliance, downstream teams) |
| Treating a feature flag toggle as a decision | A feature flag is a code-version influence; the decision affected by it is what gets audited |
| Capturing every internal step as a decision | Boundary too fine; choose Pattern A or C |
| Capturing only the final verdict | Boundary too coarse; use Pattern A with `evidence` |
| Producing the schema with `request_id` mandatory | Optional unless every decision is part of a request flow |
| Tier classification skipped | Tier drives required artifacts; absent tier means consistency gate cannot run |
| Drafts treated as final | The drafts feed into the `decision-log-architecture` skill for refinement; they are not the spec |
