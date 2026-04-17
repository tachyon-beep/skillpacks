# Assembling the Solution Architecture Document

## Overview

**The SAD is the stakeholder deliverable. The consistency gate is the quality bar.**

This skill runs at the end of the workflow. It does two things:

1. **Consistency gate** — cross-artifact checks across the numbered files; failures block emission unless waived with recorded rationale
2. **Assembly** — consolidates the numbered artifact set into a single readable Solution Architecture Document at `99-solution-architecture-document.md`

**Core principle:** The gate fails loud. The SAD is not emitted until the gate passes or the waivers are explicit.

## When to Use

- The numbered artifacts are all drafted (or the user has declared the design otherwise complete)
- You are tempted to "just paste everything together and call it done"
- A reviewer or stakeholder needs the consolidated deliverable

## The Consistency Gate

Run these checks in order. Record the result of each. Emit the SAD only when all checks pass or have documented waivers.

### Check 1 — File presence

| Required for all workflows | Required for brownfield | Required for enterprise |
|----------------------------|-------------------------|-------------------------|
| `00-scope-and-context.md` | `16-migration-plan.md` | `archimate-model/` with business/application/technology files |
| `01-requirements.md` | | `archimate-model/viewpoints/` with at least cio/arb/engineering |
| `02-nfr-specification.md` | | `togaf-deliverable-map.md` |
| `03-nfr-mapping.md` | | |
| `04-solution-overview.md` | | |
| `05-tech-selection-rationale.md` | | |
| `06-descoped-and-deferred.md` | | |
| `09-component-specifications.md` | | |
| `14-requirements-traceability-matrix.md` | | |
| `15-integration-plan.md` | | |
| `17-risk-register.md` | | |
| `adrs/` (may be empty only if the workflow produced no significant decisions — rare) | | |

Router-owned artifacts (`07`, `08`, `10`, `11`, `12`, `13`) may be absent for XS/S scopes, but if `04-solution-overview.md` references them, they must exist.

### Check 2 — Traceability

- Every `FR-*` in `01-` appears in `14-` with ≥1 satisfying component
- Every `NFR-*` in `02-` appears in `14-` and in `03-` with ≥1 load-bearing component
- Every `CON-*` in `01-` appears in `14-` with an "addressed by" entry
- Every component in `09-` appears in `14-` with ≥1 requirement (or is marked decorative with rationale)
- Orphan report section in `14-` is either empty or non-empty with proposed actions for each orphan

### Check 3 — Quantification

- Every NFR in `02-` has Target (numbers), Measured (method), and Source/driver populated
- No adjective-only NFRs ("fast", "secure", "scalable") in `02-`
- Every NFR conflict noted in `02-` has a resolution statement

### Check 4 — ADR rigor

For every file in `adrs/`:

- Status is one of: Proposed, Accepted, Superseded by ADR-NNNN, Deprecated
- Expiry / review date is populated and in the future (or marked for imminent review)
- Alternatives considered ≥ 2 (or explicit "post-hoc ADR" statement)
- Drivers trace to FR/NFR/CON IDs
- Consequences include at least one Negative / accepted trade-off
- Rollback / exit criteria populated
- No contradictions with `02-` (ADR that silently relaxes an NFR fails the check)

### Check 5 — Tech selection coverage

- Every significant decision in `05-` is also recorded as an ADR
- Every ADR referenced in `05-` exists in `adrs/`
- Tradeoff matrix rows in `05-` have at least two candidates

### Check 6 — Integration and migration

- Every integration in `15-` has: direction, ownership, contract summary, failure modes, observability
- For brownfield: every migration stage in `16-` has: success criteria, rollback triggers (SLI-observable), rollback procedure
- No big-bang cutovers in `16-` without a waiver in `06-descoped-and-deferred.md` citing business reason

### Check 7 — Risk register

- Every entry in `17-` has: category, likelihood, impact, observable triggers, mitigation (design + runtime), owner, review
- Overloaded components flagged by `03-` appear in `17-` with a risk entry
- No ops-generic risks ("server could fail") in `17-` — they belong to operational runbooks

### Check 8 — Enterprise binding (if activated)

- `togaf-deliverable-map.md` covers every numbered artifact with an ADM phase
- `archimate-model/` layer files have no cross-layer element mistakes (application components on business layer, etc.)
- Viewpoint files name their concerns and selected elements — not colour-filtered copies

### Gate outcome

Produce a gate report even if everything passes:

```markdown
# Consistency Gate Report

**Run date:** YYYY-MM-DD

## Check 1 — File presence: PASS
## Check 2 — Traceability: PASS
## Check 3 — Quantification: FAIL
- NFR-09 ("intuitive admin UI") is adjective-only. Either quantify (e.g., "admin onboarding < 30 min for a new user") or move to UX scope and remove from NFR set.
## Check 4 — ADR rigor: PASS
…

## Waivers
- [Check X, item Y]: waived because [explicit rationale]. Waiver recorded in the SAD.
```

**A waiver requires an explicit rationale.** "Waived because time-pressure" is not a rationale; it's an abdication. A rationale names the business / technical reason the check is accepted-as-failed.

## Emitting `99-solution-architecture-document.md`

```markdown
# Solution Architecture Document

**Project:** [name]
**Version:** [semver]
**Date:** YYYY-MM-DD
**Authors:** [roles]
**Consistency gate:** PASS (gate report date: YYYY-MM-DD)

## 1. Scope and context
[Pulled from 00-]

## 2. Requirements
[Summary from 01-; full list linked]

## 3. Non-functional requirements
[Summary from 02-; full quantification linked]

## 4. Solution overview
[From 04-; architecture-at-a-glance, dominant style, key choices]

## 5. Component architecture
[Summary of 09-, with component catalog; C4 context (07) and containers (08) inline or linked]

## 6. Data model
[From 10-]

## 7. Interfaces
[From 11-]

## 8. Key decisions
[ADR summary — title, decision, reason, link to full ADR]

## 9. NFR satisfaction
[From 03-: how each NFR is satisfied, with links to the load-bearing components]

## 10. Traceability
[Link to 14-RTM; summary of coverage stats]

## 11. Integration
[From 15-]

## 12. Migration plan (brownfield only)
[From 16-]

## 13. Risks
[From 17-]

## 14. TOGAF / ArchiMate view (enterprise only)
[Link to archimate-model/ and togaf-deliverable-map.md; include cio-view.md inline for stakeholder reading]

## 15. Descoped / deferred
[From 06-]

## Appendix A — Consistency gate waivers
[If any]
```

The SAD is *assembled from* the numbered artifacts, not a rewrite. Long sections link rather than duplicate. The SAD's job is to give a reader the picture without requiring them to open every numbered file, but the numbered files remain the authoritative source.

## Pressure Responses

### "The gate is too strict, just emit the SAD"

**Response:** "The gate exists because inconsistent SADs have historically shipped and caused downstream rework. Every check exists because an artifact family has failed on it. If a specific check is wrong for your project, record a waiver with rationale — the SAD gets emitted and the waiver is transparent."

### "Can we emit the SAD without all the artifacts?"

**Response:** "The SAD is a view onto the artifacts. Missing artifacts mean the SAD is fiction about parts of the design. Produce the missing artifact at a stub level (explicit 'out-of-scope' rationale) rather than silently omitting."

### "The RTM has orphans — we'll fix it later"

**Response:** "Orphans are either scope reductions (record in `06-`) or missing implementations (fix in `09-`). Either way they surface before emission. Emitting a SAD with hidden orphans ships a lie."

## Anti-Patterns to Reject

### ❌ Silent waiver

Skipping a check without recording it. The gate's value is in being transparent.

### ❌ SAD that duplicates the artifacts

Copy-pasting every numbered file into the SAD makes it unreadable and divergent (the copy rots). Link, summarise, quote sparingly.

### ❌ SAD without a gate report date

No gate report = no gate ran. The SAD must cite the gate report run it passed.

### ❌ Gate report with all checks at PASS but obvious gaps

If the RTM is clearly missing requirements but Check 2 passed, the checker was superficial. Gate checks are substantive, not ceremonial.

## Scope Boundaries

**This skill covers:**

- Consistency gate execution (Checks 1-8)
- SAD assembly (`99-`)
- Waiver recording

**Not covered:**

- Producing the underlying artifacts (that's every other skill's job)
- Stakeholder-polish / editorial register (that's `muna-technical-writer` after assembly)
- Post-emission sign-off / approval flow (governance, owned by `axiom-sdlc-engineering`)
