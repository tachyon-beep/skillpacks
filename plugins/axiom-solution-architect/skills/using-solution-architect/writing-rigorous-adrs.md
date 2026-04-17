# Writing Rigorous ADRs

## Overview

**An ADR documents a choice — not a conclusion.**

The difference: a conclusion records what won. A choice records what was considered, what won, and *why*. The consequences of the loser get remembered; the consequences of the winner are planned for.

**Core principle:** Every ADR has at least two alternatives, trace-able decision drivers, explicit consequences (good and bad), a rollback plan, and an expiry date.

## When to Use

- Recording any significant architectural decision: language, datastore, messaging, deployment target, auth, major framework, style (monolith/microservices/event-driven/serverless)
- A decision in `05-tech-selection-rationale.md` affects multiple components, downstream teams, or the rollback path
- You are tempted to write a one-paragraph "we chose X" without alternatives

**When *not* to use:** trivial decisions (JSON library, linter rules). A one-line note in `05-` is enough.

## ADR File Convention

- Path: `adrs/NNNN-<kebab-title>.md`
- NNNN is a zero-padded 4-digit sequence (0001, 0002, …)
- Title: the *decision* in imperative form ("Use PostgreSQL 16 as the primary datastore"), not the problem ("Datastore selection")

## ADR Template

Every ADR MUST include every section. Empty sections are the most common drift mode.

```markdown
# ADR-NNNN: [Title — the decision, in imperative form]

- **Status:** [Proposed | Accepted | Superseded by ADR-NNNN | Deprecated]
- **Date decided:** YYYY-MM-DD
- **Decision makers:** [Roles, not names where possible]
- **Expiry / review date:** YYYY-MM-DD  [default: 18 months from decision]

## Context

[2-5 sentences: what forced this decision, what the state of the world was when it was made. Reference the requirements and NFRs driving it by ID (FR-NN, NFR-NN, CON-NN). A reader who has never seen the project should understand why we were deciding this.]

## Decision drivers

- **DRIVER-1:** [In the form: "From NFR-02, we need 3000 writes/s sustained" or "From CON-01, data must stay in EU"]
- **DRIVER-2:** …

(Drivers are what the alternatives are scored against. If a driver doesn't appear in `01-requirements.md` or `02-nfr-specification.md`, either add it there or reject it as a driver — drivers must trace.)

## Alternatives considered

### Option A — [Name]

- Fit vs DRIVER-1: …
- Fit vs DRIVER-2: …
- Ops cost: …
- Rejected because: … (or "Chosen")

### Option B — [Name]

…

### Option C — [Name]

…

(At least two alternatives. A single option is not a decision; it's a reflex.)

## Decision

**We will [action] because [primary reason tied to drivers].**

[1-3 sentences elaborating the decision. Explicit scope: what this decision does and does not cover.]

## Consequences

### Positive

- [Consequence tied to a driver — e.g., "NFR-02 satisfied with room for 3× burst"]
- …

### Negative (accepted trade-offs)

- [Consequence we are knowingly accepting — e.g., "Operational cost grows with cross-AZ replication"]
- …

### Neutral / to monitor

- [Consequence we want to track but don't yet know the impact of — e.g., "Team will need to learn X"]
- …

## Rollback / exit criteria

**When we would revisit:**
- [Trigger 1 — e.g., "Sustained write rate > 9000/s for 1 hour"]
- [Trigger 2 — e.g., "Vendor pricing change > 50% on core tier"]

**How we would unwind:**
- [Concrete steps — e.g., "Access layer is repository-pattern; hot tables migrate to columnar store behind the same interface. Estimated effort: 4-6 engineer-weeks."]

(If you cannot describe the exit, you do not understand the decision well enough.)

## Links

- Requirements / NFRs / Constraints: FR-…, NFR-…, CON-…
- Related ADRs: ADR-…
- Supersedes: ADR-NNNN (if applicable)
- External references: [standards, docs, vendor pages]

## Review log

- YYYY-MM-DD — [Reviewer] — [Decision: still valid | revisit | superseded]
```

## Pressure Responses

### "We already decided, just document it"

**Response:** "Fine — the template still applies. Post-hoc ADRs are legitimate when we record them honestly. If you can't produce at least one alternative that was considered, the decision was a reflex, and the ADR should say so in the 'Alternatives considered' section: 'Option B was not evaluated at the time; this ADR is being written post-hoc to expose the gap.' That's a truthful record."

### "No need for a rollback plan — this one's not going to fail"

**Response:** "Every architectural decision has an exit, even if we never take it. Without one, a future team inheriting this decision has no information about when to question it. If the exit truly is 'rebuild from scratch', that's a legitimate rollback plan — say so, and call out the cost."

### "Alternatives waste time — we know what we want"

**Response:** "The alternatives are the evidence. An ADR without them is a claim without proof. One paragraph per alternative is enough; we don't need a full evaluation, we need proof that the chosen option beat specific competitors, not just any option."

### "ADRs don't expire"

**Response:** "Systems evolve; the context that made a decision right can change. An 18-month review flag doesn't mean the decision expires — it means the ADR returns for confirmation. Decisions with no review become invisible forever."

## Anti-Patterns to Reject

### ❌ Single-option ADR

One "alternative" section containing the chosen option only. This is a decision log entry, not an ADR. Either add real alternatives or demote to a one-liner in `05-`.

### ❌ Drivers that don't trace

"Driver: we want flexibility" is not a driver — it doesn't appear in `01-` or `02-`. Either bring it into the requirements (naming specifically what the flexibility enables) or strike it.

### ❌ Consequences that are all positive

If every consequence is positive, the decision is insufficiently interrogated. Every significant architectural decision has costs — find them and name them.

### ❌ ADRs that silently relax NFRs

"ADR-0006: adopt eventual consistency" while `02-NFR-05` still says "strong consistency required." Either the NFR changes (and the NFR file is updated with a note of the tradeoff) or the ADR is wrong. The assembly-skill consistency gate will catch this; catching it earlier here saves rework.

### ❌ ADRs older than their expiry

An ADR whose review date has passed, still driving decisions. At the expiry, either confirm (new review-log entry), supersede (status: Superseded by ADR-NNNN), or deprecate.

### ❌ "Status: Proposed" that never gets moved

A proposed ADR that shipped anyway. The status is a lie at that point. Accept or reject; don't leave undecided ADRs running the design.

## Interaction with `axiom-sdlc-engineering/design-and-build`

The SDLC pack's `design-and-build` skill owns ADR lifecycle governance (who approves, when status changes, archive rules). This skill owns ADR *content quality*. A rigorous ADR produced here satisfies the content criteria that lifecycle governance expects.

If the project follows SDLC governance, the ADR needs an approval signature block at the bottom; add it in the project's house style. The template above does not prescribe one because styles vary (GitHub PR approval, Markdown sign-off, JIRA link, etc.).

## Scope Boundaries

**This skill covers:**

- ADR file structure, naming, and lifecycle status values
- Alternatives, drivers, consequences, rollback discipline
- Expiry and review log
- NFR-contradiction check (surface, or block)

**Not covered:**

- ADR lifecycle governance (approval flow, archive) — `axiom-sdlc-engineering/design-and-build`
- Minor decisions that belong in `05-tech-selection-rationale.md` rather than an ADR
- Tradeoff matrices (those live in `05-`; the ADR summarises the outcome)
