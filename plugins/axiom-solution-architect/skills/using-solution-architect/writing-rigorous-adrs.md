# Writing Rigorous ADRs

## Overview

**An ADR documents a choice — not a conclusion.**

The difference: a conclusion records what won. A choice records what was considered, what won, and *why*. The consequences of the loser get remembered; the consequences of the winner are planned for.

**Core principle:** Every ADR has traceable decision drivers, at least two genuine alternatives (or an explicit *constrained — no alternatives existed* statement), explicit consequences (good and bad), a rollback plan, a reversibility tag, an expiry / review date, and a cost dimension named in at least one driver.

**Contents:** [Template](#adr-template) · [Lifecycle transitions](#lifecycle-transitions) (supersession, deprecation, partial supersession, amendment) · [Pressure Responses](#pressure-responses) · [Anti-Patterns](#anti-patterns-to-reject) · [SDLC handoff](#interaction-with-axiom-sdlc-engineeringdesign-and-build) · [Security handoff](#interaction-with-ordis-security-architect)

The pattern is Michael Nygard's (2011) "Architecture Decision Records"; the specifics below tighten Nygard's original with MADR-style (Markdown Architectural Decision Records) drivers, explicit negatives, rollback, reversibility, cost as a first-class dimension, and expiry.

## When to Use

See the router's Start Here (SKILL.md) if this is your first pass through the pack.

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
- **Forced choices with no genuine alternative.** "The client mandates Oracle 19c" is a constraint (record as `CON-CTR-NN` or `CON-REG-NN` in `01-requirements.md`), not a decision. A post-hoc ADR is legitimate only if the constraint itself is architecturally significant and the forgone alternatives merit the historical record.

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
- **Reversibility:** [Easy — ≤1 engineer-week, single module | Moderate — 1–6 engineer-weeks, multiple modules | Hard — >6 engineer-weeks, cross-cutting | One-way — contractual, regulatory, or data-shape commitment that is effectively non-revertible]
- **Blast radius on rollback:** [which components, teams, and external consumers are affected by reverting this decision]
- **Next review date:** YYYY-MM-DD  [review cadence — typically 12–24 months for application stacks, 6–12 for fast-moving tech (LLM providers, frontend frameworks), 3–5 years for infrastructure or regulatory decisions]
- **Hard expiry:** YYYY-MM-DD or N/A  [use only when the decision is known to be time-bounded — e.g., "Kafka 2.x chosen; Kafka 4.x forces revisit by Q3-2027"]
- **Recorded post-hoc:** No | Yes — decision made on YYYY-MM-DD, ADR written N months later

## Context

[2-5 sentences: what forced this decision, what the state of the world was when it was made. Reference the requirements and NFRs driving it by ID (FR-NN, NFR-NN, CON-*-NN). A reader who has never seen the project should understand why we were deciding this.]

## Decision drivers

- **DRIVER-1 [NFR-02]:** [one-sentence restatement — e.g., "3000 writes/s sustained, measured per `02-nfr-specification.md`"]
- **DRIVER-2 [CON-REG-01]:** [e.g., "EU data residency (GDPR Art. 44–49)"]
- **DRIVER-3 [COST]:** [if the decision has a cost driver — e.g., "first-year run cost ≤ $30k, see `05-tech-selection-rationale.md`"]

(Drivers are what the alternatives are scored against. Every driver MUST carry at least one ID in square brackets: `FR-NN`, `NFR-NN`, `CON-*-NN`, or the reserved driver class `[COST]` (when the driver is a cost/licence/TCO target named in `05-`). A driver without a bracketed ID is an opinion, not a driver, and the consistency gate cannot cross-check it. At least one cost-class driver should appear for decisions that affect infrastructure, licence, or operational spend — a decision whose drivers never name cost is a decision that will be re-opened when the bill arrives.)

## Alternatives considered

### Option A — [Name]

- Fit vs DRIVER-1: …
- Fit vs DRIVER-2: …
- **Cost (build):** [one-off engineering + licence + migration cost, stated in dollars or engineer-weeks with the rate assumed]
- **Cost (run, first year):** [infra + licence + operational burden per year; name the assumed scale]
- **Cost (run, at target scale):** [same at the scale named in NFR-01/02/… — not "best effort"]
- **Reversibility:** [Easy | Moderate | Hard | One-way] — with one-line blast-radius note
- Rejected because: … (or "Chosen")

### Option B — [Name]

… (same fields as Option A — asymmetric cost evaluation is a red flag for `tech-selection-critic`)

### Option C — [Name]

…

(At least two *genuine* alternatives. A single option is usually a reflex — but a decision forced by contract, regulation, or mandated standard has no alternatives. In that case, replace the "Alternatives considered" section with a single subsection titled **"Constrained decision — no alternatives"** containing three mandatory fields:

1. **Constraint source:** the exact `CON-*-NN` ID from `01-requirements.md`, plus the authoritative citation (contract clause reference, regulation article, standard number). "The CTO mandated this" is not a constraint; it is a preference (record it as `CON-ORG-NN` and accept that it is the weakest constraint class and does NOT qualify for this escape).
2. **Foreclosed alternatives:** the two or three options that would have been evaluated absent the constraint, each with a one-line rationale for why the constraint forecloses them. "No alternatives considered" here is unacceptable — the point of the escape is to record what was foreclosed, not to skip evaluation entirely.
3. **Re-test trigger:** the condition under which the constraint would be re-examined (e.g., "contract renewal in 2027"; "regulation under active review by [body]"). A constraint with no re-test trigger is an assertion, not a constraint.

Inventing strawman alternatives to satisfy the two-genuine-alternatives rule is worse than admitting the constraint; but the escape itself must still carry evidence. `tech-selection-critic` treats an under-evidenced constrained-decision ADR as a High finding (asymmetric evaluation class).)

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

- Requirements / NFRs / Constraints: FR-…, NFR-…, CON-*-…
- Related ADRs: ADR-…
- Supersedes: ADR-NNNN (if applicable — see *Lifecycle transitions*)
- Threat model entries (if produced by `ordis-security-architect`): THREAT-NN (STRIDE row / attack-tree node)
- Security controls implemented / relaxed: CTRL-NN (from threat-model control set)
- External references: [standards, docs, vendor pages]

## Traceability

- **Requirements satisfied / impacted:** FR-…, NFR-…, CON-*-… (each must also be updated in `14-requirements-traceability-matrix.md` to reference this ADR — forward link here, back link there)
- **Threats addressed / enabled:** THREAT-NN — explicit statement of which threats this decision mitigates, and which it creates or leaves open. A decision that changes the trust boundary, authN/authZ posture, data-at-rest or data-in-transit stance, or segmentation MUST have at least one entry here (or a `N/A — not a security-affecting decision` line).
- **Controls implemented / impacted:** CTRL-NN — if the threat model names controls, name which ones this decision realises or weakens.
- **RTM updated:** [ ] yes / [ ] pending at time of ADR merge
- **Requirements conflict check:** Does this decision relax, tighten, or contradict any listed FR/NFR/CON? If yes, the requirement file must be updated with the trade-off note — or the ADR is wrong. The consistency gate will catch silent relaxations (e.g., an ADR that adopts eventual consistency while `02-NFR-05` still requires strong consistency); catching it here is cheaper.
- **Security-control conflict check:** Does this decision weaken or bypass a control named in the threat model? If yes, the threat-model control row must be updated with the trade-off note — or the ADR is wrong.

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
- **Propagate:** a deprecation is a silent orphaning risk and MUST trigger:
  1. `14-requirements-traceability-matrix.md` — remove the ADR from every "Satisfied by ADRs" cell that referenced it; re-run orphan detection to catch requirements left without any ADR.
  2. `17-risk-register.md` — if the decision was load-bearing for any risk mitigation, re-score the risk without it.
  3. Consumer notification — list the components, teams, or external consumers whose design referenced this ADR, and confirm they know the decision is deprecated (a one-line note in the review log: "Consumers notified: [list]").
  A deprecation that skips step 1 is the pattern that produces silent orphan requirements.

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

### Bidirectional linking

Every supersession must update **both** ADRs:

- New ADR `Links → Supersedes: ADR-NNNN`.
- Old ADR `Status: Superseded by ADR-NNNN`.

One-way links are integrity failures. The consistency gate catches them; catching them here is cheaper.

### Moving from Proposed to Accepted

A `Proposed` ADR that ships is a lie in progress. Before merge, the status must move to `Accepted` (or the ADR must be rejected and removed). If the project uses SDLC governance, the governance step owns the move — but the ADR author is responsible for ensuring the status is correct on ship day. See `axiom-sdlc-engineering/design-and-build` for approval-flow specifics.

## Pressure Responses

### "We already decided, just document it"

**Response:** "Fine — the template still applies. Post-hoc ADRs are legitimate when we record them honestly. Mark `Recorded post-hoc: Yes` in the header with the real decision date. If you can't produce at least one alternative that was considered, the decision was a reflex, and the ADR should say so in the 'Alternatives considered' section: 'Option B was not evaluated at the time; this ADR is being written post-hoc to expose the gap.' That's a truthful record."

### "No need for a rollback plan — this one's not going to fail"

**Response:** "Every architectural decision has an exit, even if we never take it. Without one, a future team inheriting this decision has no information about when to question it. If the exit truly is 'rebuild from scratch', that's a legitimate rollback plan — say so, mark reversibility as `One-way`, and call out the cost."

### "Alternatives waste time — we know what we want"

**Response:** "The alternatives are the evidence. An ADR without them is a claim without proof. One paragraph per alternative is enough; we don't need a full evaluation, we need proof that the chosen option beat specific competitors, not just any option."

### "ADRs don't expire"

**Response:** "Systems evolve; the context that made a decision right can change. A `Next review date` doesn't mean the decision expires — it means the ADR returns for confirmation. Decisions with no review become invisible forever. `Hard expiry` is reserved for decisions that are *actually* time-bounded (e.g., a vendor contract renewal in 2027)."

### "Cost is a run-time concern, not a design concern"

**Response:** "Cost drives reversibility. A decision whose first-year run cost was unmodelled is the decision a CFO reopens in month seven — usually at the worst possible time. Naming cost at decision time, in the same template as NFR fit, is the cheapest way to prevent that reopening. If the real cost is 'we don't know', say that — `[COST] unmodelled, flagged for re-test at 3-month mark` is a legitimate entry; a silent cost cell is not."

## Anti-Patterns to Reject

### ❌ Single-option ADR

One "alternative" section containing the chosen option only. This is a decision log entry, not an ADR. Either add real alternatives or demote to a one-liner in `05-`.

### ❌ Drivers that don't trace

"Driver: we want flexibility" is not a driver — it carries no bracketed ID and does not appear in `01-` or `02-`. Either bring it into the requirements (naming specifically what the flexibility enables, with an `FR-NN`/`NFR-NN`/`CON-*-NN`/`[COST]` tag) or strike it.

### ❌ Consequences that are all positive

If every consequence is positive, the decision is insufficiently interrogated. Every significant architectural decision has costs — find them and name them.

### ❌ ADRs that silently relax NFRs

"ADR-0006: adopt eventual consistency" while `02-NFR-05` still says "strong consistency required." Either the NFR changes (and the NFR file is updated with a note of the tradeoff) or the ADR is wrong. The consistency gate will catch this; catching it earlier here saves rework.

### ❌ ADRs that silently bypass security controls

An authN, authZ, data-handling, or segmentation decision whose Traceability section does not list `THREAT-NN` or `CTRL-NN`. If the threat model exists and this decision touches its domain, the back-links are mandatory — the pack's handoff with `ordis-security-architect` depends on them. An explicit `N/A — not a security-affecting decision` line is acceptable; silence is not.

### ❌ ADRs missing a cost driver when the decision affects spend

An infrastructure, licence, or operational-spend decision whose drivers list contains no `[COST]` entry and whose alternatives carry no `Cost (build)` / `Cost (run)` rows. The bill is coming regardless of whether the ADR acknowledged it.

### ❌ ADRs without reversibility or blast-radius tags

A header missing `Reversibility:` and `Blast radius on rollback:` collapses "commit cheaply and learn" and "one-way door" into the same artifact. `solution-design-reviewer` flags this as a Weak-ADR finding.

### ❌ ADRs older than their next-review date

An ADR whose `Next review date` has passed, still driving decisions. At the review date, either confirm (new review-log entry), supersede (new ADR with bidirectional link), or deprecate. Past `Hard expiry` with no action is a harder failure — the decision was declared time-bounded and the time has run out.

### ❌ Dangling supersession links

New ADR claims `Supersedes: ADR-0003` but ADR-0003 still shows `Status: Accepted`. Or ADR-0003 shows `Status: Superseded by ADR-0007` but ADR-0007 does not list ADR-0003 in its `Supersedes` field. One-way supersession links are integrity failures — see *Lifecycle transitions → Bidirectional linking*.

### ❌ Supersession used where deprecation fits (or vice versa)

Superseded = "the question still exists, the answer changed — here is the new ADR." Deprecated = "the question no longer exists, no replacement." Marking an ADR `Superseded by ADR-NNNN` when ADR-NNNN doesn't actually replace the same decision confuses readers and poisons the traceability graph.

### ❌ Multi-decision ADR replaced by one supersession ADR

Original ADR decided datastore + messaging + cache; new ADR changes only the datastore. Writing a single supersession ADR that replaces all three is wrong — only the datastore changed. Write one ADR per sub-decision that is actually changing and mark the original `Superseded in part`.

### ❌ "Status: Proposed" that never gets moved

A proposed ADR that shipped anyway. The status is a lie at that point. Accept or reject; don't leave undecided ADRs running the design.

### ❌ "Constrained decision — no alternatives" used as a laundering route

The escape clause in the alternatives section is for genuinely forced choices. Using it to skip evaluation — by citing a weak constraint (`CON-ORG-NN` preference), by omitting the foreclosed-alternatives field, or by leaving the re-test trigger blank — fails the critique that `tech-selection-critic` runs on constrained-decision ADRs.

## ADR index

Maintain `adrs/README.md` (or `adrs/index.md`) listing every ADR by number, title, and current status. Without an index, 30+ ADRs become unbrowsable — new readers cannot find what is still in force versus what has been superseded. Tools like `adr-tools` generate this automatically; a hand-maintained table is fine if the rules are observed.

## Interaction with `axiom-sdlc-engineering/design-and-build`

**This skill stops at:** ADR *content quality* — structure, drivers, alternatives, consequences, rollback, reversibility, cost, expiry, lifecycle-state hygiene.

**The SDLC pack picks up at:** ADR lifecycle *governance* — who approves, the state-machine for status transitions, archive and retention rules, ARB signatures.

### Field mapping: solution-architect ADR → SDLC DAR template

The sibling pack's DAR template (`axiom-sdlc-engineering/skills/design-and-build/architecture-and-design.md` → ADR template at "ADR Template (Level 3)") has named fields. This skill's ADR template maps into those fields as follows, so the governance pack can ingest without transformation:

| This skill's ADR field | SDLC DAR template field | Notes |
|------------------------|--------------------------|-------|
| Title (imperative) | Decision Title | — |
| Status | Status | Both use the same vocabulary (Proposed / Accepted / Superseded / Deprecated) |
| Date decided | Date | — |
| Decision makers | Authors | Role-oriented here; SDLC pack may require names for audit |
| Reversibility + Blast radius | Implementation Notes → Rollback plan | SDLC pack adds governance-level rollback authority |
| Next review date / Hard expiry | Validation → Review schedule | SDLC pack adds 30-/90-day retro cadence |
| Context | Context → Problem statement | — |
| Decision drivers (with IDs) | Decision Drivers | SDLC pack formats as checklist; this skill formats as ID-tagged list — both are valid |
| Alternatives considered | Considered Options + Alternatives Analysis | SDLC pack's scored matrix is a superset; this skill's per-option bullets collapse into the matrix rows |
| Decision | Decision Outcome | — |
| Consequences (Positive / Negative / Neutral) | Positive Consequences / Negative Consequences (Accepted Tradeoffs) | Neutral-to-monitor is an extension — record in SDLC pack as an advisory note |
| Rollback / exit criteria | Implementation Notes → Rollback plan + Validation → Failure criteria | — |
| Traceability (Requirements / Threats / Controls) | References + Metadata (Related) | SDLC pack may require specific cross-link formats per platform (GitHub, Azure DevOps) |
| Review log | (Not present in SDLC pack template) | SDLC pack tracks in governance; this skill keeps it in-document as a fallback |
| Approval signature block | Governance-owned | This skill does not prescribe format; SDLC pack does per platform |

A rigorous ADR produced here satisfies the content criteria that lifecycle governance expects. If the project follows SDLC governance, the ADR needs an approval signature block at the bottom in the project's house style — GitHub PR approval, Markdown sign-off, JIRA link, etc. The template does not prescribe one because styles vary.

## Interaction with `ordis-security-architect`

**This skill stops at:** ADR *content quality*, including back-links to threat-model entries and controls.

**The security-architect pack picks up at:** threat modelling itself — STRIDE enumeration, attack trees, control design, residual-risk scoring.

The handoff is mechanical: `ordis-security-architect` emits `THREAT-NN` and `CTRL-NN` identifiers; this skill requires ADRs whose decisions touch security posture to cite those identifiers in the Links and Traceability sections. A security-affecting ADR without `THREAT-NN` / `CTRL-NN` back-links is flagged by `solution-design-reviewer` as a Weak-ADR finding. If no threat model exists for the project, a security-affecting ADR must still state `Threats addressed: N/A — no threat model produced for this design` and that absence is itself a finding in `17-risk-register.md`.

## Scope Boundaries

**This skill covers:**

- ADR file structure, naming, and lifecycle status values
- Alternatives, drivers (with cost as a first-class dimension), consequences, rollback discipline
- Reversibility tagging and blast-radius discipline
- Next-review-date and hard-expiry discipline, plus review log
- NFR-contradiction check (surface, or block)
- Security-control and threat-model back-link discipline (handoff with `ordis-security-architect`)
- Lifecycle transitions: supersession, deprecation (with downstream propagation), partial supersession, amendment, and bidirectional linking
- ADR-to-RTM traceability (forward link from ADR; back-link enforced in `maintaining-requirements-traceability`)
- ADR-to-SDLC-DAR field mapping (handoff with `axiom-sdlc-engineering/design-and-build`)
- ADR index maintenance

**Not covered:**

- ADR lifecycle *governance* (approval flow, archive, ARB sign-off) — `axiom-sdlc-engineering/design-and-build`
- Threat modelling itself — `ordis-security-architect`
- Minor decisions that belong in `05-tech-selection-rationale.md` rather than an ADR
- Tradeoff matrices (those live in `05-`; the ADR summarises the outcome)
