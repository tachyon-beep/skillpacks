# Writing Rigorous ADRs

## Overview

**An ADR documents a choice — not a conclusion.**

The difference: a conclusion records what won. A choice records what was considered, what won, and *why*. The consequences of the loser get remembered; the consequences of the winner are planned for.

**Core principle:** Every ADR has traceable decision drivers, at least two genuine alternatives (or an explicit *constrained — no alternatives existed* statement), explicit consequences (good and bad), a rollback plan, and an expiry / review date.

The pattern is Michael Nygard's (2011) "Architecture Decision Records"; the specifics below tighten Nygard's original with MADR-style drivers, explicit negatives, rollback, and expiry.

## When to Use

- Recording any significant architectural decision: language, datastore, messaging, deployment target, auth, major framework, style (monolith/microservices/event-driven/serverless)
- A decision in `05-tech-selection-rationale.md` affects multiple components, downstream teams, or the rollback path
- You are tempted to write a one-paragraph "we chose X" without alternatives
- **Post-hoc:** a significant decision was already made without an ADR and is worth capturing now — see `Pressure Responses → "We already decided, just document it"`

### When *not* to write an ADR

False positives are the most common ADR failure. Do not write an ADR for:

- **Implementation details that change often.** Choice of JSON library, utility library (lodash), test matchers, CSS framework — note in `05-tech-selection-rationale.md`.
- **Decisions inside an already-decided framework.** Once "use Django" is an ADR, "use Django REST Framework" is a direct consequence, not a new decision. Mention it in `05-` or in the original ADR's consequences.
- **Naming conventions, code style, linter rules.** Belongs in a style guide or CONTRIBUTING.md.
- **Reversible-within-a-sprint decisions affecting a single module.** Code comment or `05-` suffices.
- **Policy pass-throughs.** "We do code reviews" — policy, not architecture.
- **Forced choices with no genuine alternative.** "The client mandates Oracle 19c" is a constraint (record as `CON-NN` in `01-requirements.md`), not a decision. A post-hoc ADR is legitimate only if the constraint itself is architecturally significant and the forgone alternatives merit the historical record.

> If in doubt: can you name two genuine options that a competent architect would have evaluated? If not, don't write an ADR.

## ADR File Convention

- Path: `adrs/NNNN-<kebab-title>.md`
- NNNN is a zero-padded 4-digit sequence (0001, 0002, …)
- Title: the *decision* in imperative form ("Use PostgreSQL 16 as the primary datastore"), not the problem ("Datastore selection")

## ADR Template

Every ADR MUST include every section. Empty sections are the most common drift mode.

```markdown
# ADR-NNNN: [Imperative title — e.g., "Use PostgreSQL 16 as the primary datastore"]

- **Status:** [Proposed | Accepted | Superseded by ADR-NNNN | Superseded in part by ADR-NNNN | Deprecated]
- **Date decided:** YYYY-MM-DD
- **Decision makers:** [Roles, not names where possible]
- **Expiry / review date:** YYYY-MM-DD  [appropriate to the decision horizon — typically 12–24 months for application stacks, 6–12 for fast-moving tech (LLM providers, frontend frameworks), 3–5 years for infrastructure or regulatory decisions]

## Context

[2-5 sentences: what forced this decision, what the state of the world was when it was made. Reference the requirements and NFRs driving it by ID (FR-NN, NFR-NN, CON-NN). A reader who has never seen the project should understand why we were deciding this.]

## Decision drivers

- **DRIVER-1:** [In the form: "From NFR-02, we need 3000 writes/s sustained" or "From CON-01, data must stay in EU"]
- **DRIVER-2:** …

(Drivers are what the alternatives are scored against. Every driver must trace to an entry in `01-requirements.md` (including `CON-NN` constraints) or `02-nfr-specification.md`. If a driver has no such entry, either add it there or reject it as a driver — untraced drivers are opinions, not drivers.)

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

(At least two *genuine* alternatives. A single option is usually a reflex — but a decision forced by contract, regulation, or mandated standard has no alternatives. In that case, replace the "Alternatives considered" section with a single subsection titled **"Constrained decision — no alternatives"** naming the constraint source (contract clause, regulation, mandated standard, `CON-NN`) and what would have been considered absent the constraint. Inventing strawman alternatives to satisfy the rule is worse than admitting the constraint.)

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
- Supersedes: ADR-NNNN (if applicable — see *Lifecycle transitions*)
- External references: [standards, docs, vendor pages]

## Traceability

- **Requirements satisfied / impacted:** FR-…, NFR-…, CON-… (each must also be updated in `14-requirements-traceability-matrix.md` to reference this ADR — forward link here, back link there)
- **RTM updated:** [ ] yes / [ ] pending at time of ADR merge
- **Requirements conflict check:** Does this decision relax, tighten, or contradict any listed FR/NFR/CON? If yes, the requirement file must be updated with the trade-off note — or the ADR is wrong. The assembly consistency gate will catch silent relaxations (e.g., an ADR that adopts eventual consistency while `02-NFR-05` still requires strong consistency); catching it here is cheaper.

## Review log

- YYYY-MM-DD — [Reviewer] — [Decision: still valid | revisit | superseded]
```

## Lifecycle transitions

Four operations change an ADR's state after it ships: supersession, deprecation, partial supersession, and amendment. They are distinct — do not blur them.

### Supersession

A newer ADR replaces an older one when the decision has genuinely changed.

- Create the new ADR (next `NNNN`).
- In the new ADR's **Status:** `Accepted`.
- In the new ADR's **Links → Supersedes:** `ADR-NNNN` (the old one).
- **Update the old ADR's Status** to `Superseded by ADR-NNNN`. Change nothing else — the old ADR is a historical record, not a working document.
- Add a review-log entry in the old ADR noting the supersession date and the new ADR number.

Do **not** edit the old ADR's context, decision, or consequences. Future readers need to see the decision as it stood, with its superseded pointer as the only change.

### Deprecation

An ADR is deprecated (not superseded) when the thing it decided is no longer being done, with no replacement. E.g., "Use Kafka for event streaming" deprecated because the event-streaming feature is being removed.

- Update the ADR's **Status** to `Deprecated`.
- Add a review-log entry with the deprecation date and reason.
- No new ADR is created.

**Superseded vs. Deprecated:** Superseded means "the question still exists, the answer has changed — here is the new answer." Deprecated means "the question no longer exists." If you can't name a replacement ADR, it's deprecation.

### Partial supersession

A new ADR replaces only part of an earlier one — e.g., swap the datastore but keep the messaging choice from an original multi-decision ADR.

- **Do not** write a single supersession ADR that replaces the whole original.
- Write one new ADR per sub-decision that is actually changing.
- Mark the original as `Superseded in part by ADR-NNNN, ADR-MMMM` and list which sub-decisions each replaces.
- Better: split the original into separate ADRs retroactively if it grouped decisions that should have been separate. Note the split in the review log.

Mixing supersessions in a single ADR corrupts the traceability graph — the RTM cannot point at "half of an ADR."

### Amendment

An ADR's *content* needs a correction (typo, clarification, link update) without the decision changing.

- Edit in place.
- Add a review-log entry: `YYYY-MM-DD — [Editor] — amendment: [what and why]`.
- Do **not** amend the decision, drivers, or alternatives. If those change, it is a supersession, not an amendment.

### The bidirectional linking rule

Every supersession or deprecation-with-replacement must be reflected in **both** ADRs:

- New ADR → old ADR via `Links → Supersedes: ADR-NNNN`.
- Old ADR → new ADR via `Status: Superseded by ADR-NNNN`.

A dangling forward or back-reference is an integrity failure. The assembly consistency gate should catch it, but catching it here is cheaper.

### Moving from Proposed to Accepted

A `Proposed` ADR that ships is a lie in progress. Before merge, the status must move to `Accepted` (or the ADR must be rejected and removed). If the project uses SDLC governance, the governance step owns the move — but the ADR author is responsible for ensuring the status is correct on ship day. See `axiom-sdlc-engineering/design-and-build` for approval-flow specifics.

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

An ADR whose review date has passed, still driving decisions. At the expiry, either confirm (new review-log entry), supersede (new ADR with bidirectional link), or deprecate.

### ❌ Dangling supersession links

New ADR claims `Supersedes: ADR-0003` but ADR-0003 still shows `Status: Accepted`. Or ADR-0003 shows `Status: Superseded by ADR-0007` but ADR-0007 does not list ADR-0003 in its `Supersedes` field. One-way supersession links are integrity failures — see *Lifecycle transitions → The bidirectional linking rule*.

### ❌ Supersession used where deprecation fits (or vice versa)

Superseded = "the question still exists, the answer changed — here is the new ADR." Deprecated = "the question no longer exists, no replacement." Marking an ADR `Superseded by ADR-NNNN` when ADR-NNNN doesn't actually replace the same decision confuses readers and poisons the traceability graph.

### ❌ Multi-decision ADR replaced by one supersession ADR

Original ADR decided datastore + messaging + cache; new ADR changes only the datastore. Writing a single supersession ADR that replaces all three is wrong — only the datastore changed. Write one ADR per sub-decision that is actually changing and mark the original `Superseded in part`.

### ❌ "Status: Proposed" that never gets moved

A proposed ADR that shipped anyway. The status is a lie at that point. Accept or reject; don't leave undecided ADRs running the design.

## ADR index

Maintain `adrs/README.md` (or `adrs/index.md`) listing every ADR by number, title, and current status. Without an index, 30+ ADRs become unbrowsable — new readers cannot find what is still in force versus what has been superseded. Tools like `adr-tools` generate this automatically; a hand-maintained table is fine if the rules are observed.

## Interaction with `axiom-sdlc-engineering/design-and-build`

**This skill stops at:** ADR *content quality* — structure, drivers, alternatives, consequences, rollback, expiry, lifecycle-state hygiene.

**The SDLC pack picks up at:** ADR lifecycle *governance* — who approves, the state-machine for status transitions, archive and retention rules, ARB signatures.

A rigorous ADR produced here satisfies the content criteria that lifecycle governance expects. If the project follows SDLC governance, the ADR needs an approval signature block at the bottom in the project's house style — GitHub PR approval, Markdown sign-off, JIRA link, etc. The template does not prescribe one because styles vary.

## Scope Boundaries

**This skill covers:**

- ADR file structure, naming, and lifecycle status values
- Alternatives, drivers, consequences, rollback discipline
- Expiry and review log
- NFR-contradiction check (surface, or block)
- Lifecycle transitions: supersession, deprecation, partial supersession, amendment, and bidirectional linking
- ADR-to-RTM traceability (forward link from ADR; back-link enforced in `maintaining-requirements-traceability`)
- ADR index maintenance

**Not covered:**

- ADR lifecycle *governance* (approval flow, archive, ARB sign-off) — `axiom-sdlc-engineering/design-and-build`
- Minor decisions that belong in `05-tech-selection-rationale.md` rather than an ADR
- Tradeoff matrices (those live in `05-`; the ADR summarises the outcome)
