# Assembling the Solution Architecture Document

## Overview

**The SAD is the stakeholder deliverable. The consistency gate is the quality bar.**

This skill runs at the end of the workflow. It does two things:

1. **Consistency gate** — cross-artifact checks across the numbered files; failures block emission unless waived with recorded rationale
2. **Assembly** — consolidates the numbered artifact set into a single readable Solution Architecture Document at `99-solution-architecture-document.md`

**Core principle:** The gate fails loud. The SAD is not emitted until every check passes or each failure carries a recorded waiver with rationale.

## When to Use

- The numbered artifacts are all drafted (or the user has declared the design otherwise complete)
- You are tempted to "just paste everything together and call it done"
- A reviewer or stakeholder needs the consolidated deliverable

## The Consistency Gate

Run these checks in order. Record the result of each. Emit the SAD only when all checks pass or have documented waivers.

### Check 1 — File presence (tiered)

Identify the tier declared in `00-scope-and-context.md` (XS / S / M / L / XL). Required files by tier:

- **All tiers:** `00-scope-and-context.md`, `01-requirements.md`, `02-nfr-specification.md`, `03-nfr-mapping.md`, `04-solution-overview.md`, `06-descoped-and-deferred.md`, `09-component-specifications.md`, `14-requirements-traceability-matrix.md`, `15-integration-plan.md`, `17-risk-register.md`. The `adrs/` directory exists (may be empty only if the tier is XS and no decisions were made — record the "no decisions" rationale in `04-`).
- **S and above:** add `05-tech-selection-rationale.md`, `11-interface-contracts.md`.
- **M and above:** add `07-c4-context.md`, `08-c4-containers.md`, `10-data-model.md`, `13-deployment-view.md`.
- **L and above:** add `12-sequence-diagrams.md`.
- **Brownfield (any tier):** add `16-migration-plan.md`.
- **XL (enterprise):** add `archimate-model/` (business, application, technology layers, ≥1 viewpoint per concern) and `togaf-deliverable-map.md`.

**Tier promotion rule:** if `04-solution-overview.md` or any ADR in scope references an artifact from a higher tier, the gate promotes the tier and the higher-tier requirements apply. This is not a waiver; it's a consistency correction.

### Check 1b — Router-owned artifact quality floor

For each structural artifact present (`07, 08, 09, 10, 11, 12, 13`), confirm:

- `07-c4-context.md`: exactly one system box; named external actors and systems; no internal detail.
- `08-c4-containers.md`: technology labels on each container; one page; no component-level elements.
- `09-component-specifications.md`: every component has name, single-sentence responsibility, public interface, dependencies, consumed NFR IDs (cross-ref `03-`), satisfied requirement IDs.
- `10-data-model.md`: every entity names its owning service / bounded context; cardinalities stated; logical (not ORM-specific).
- `11-interface-contracts.md`: machine-readable where the protocol supports it (OpenAPI, AsyncAPI, Protobuf, GraphQL SDL) or prose contract including inputs, outputs, errors, idempotency, versioning.
- `12-sequence-diagrams.md`: 3–5 scenarios; each scenario has at least one failure-path variant; source is PlantUML or Mermaid and checked in.
- `13-deployment-view.md`: environments, runtime topology, scaling posture, zones/regions, network boundaries.

An artifact that is present but fails its floor fails Check 1b — independent of file presence.

### Check 2 — Traceability

- Every `FR-*` in `01-` appears in `14-` with ≥1 satisfying component
- Every `NFR-*` in `02-` appears in `14-` and in `03-` with ≥1 load-bearing component
- Every `CON-*` in `01-` appears in `14-` with an "addressed by" entry
- Every component in `09-` appears in `14-` with ≥1 requirement. A component without a requirement is one of:
  - **Cross-cutting infrastructure** (e.g., logging, auth, service-mesh sidecar) — acceptable only if `09-` marks it `cross-cutting: true` with a sentence naming the non-functional concern it serves, and that concern is an NFR in `02-`. The gate follows the link; "cross-cutting" without a followable NFR fails.
  - **Speculative / future scope** — move the component to `06-descoped-and-deferred.md` with a reactivation trigger.
  - **Orphan** — Check 2 fails. Fix the RTM or remove the component. "Decorative with rationale" is not acceptable.
- Every component in `09-` has an NFR-contract table matching its row in `03-nfr-mapping.md` exactly. Any row present in one but not the other — or differing on contribution, measurement span, or owner — fails Check 2. (This closes the NFR-contract diff gate that `quantifying-nfrs.md` declares.)
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

- Every integration in `15-` has: direction, ownership, contract summary, failure modes, observability, and **versioning stance** (how the contract evolves; deprecation policy).
- Integrations and migration stages that operate against a load-bearing NFR from `03-` (e.g., availability, consistency, latency) name the NFR explicitly and state whether the integration/stage strengthens, preserves, or temporarily relaxes the NFR. A relaxation without a scheduled restoration is a Check-3 conflict and must be resolved or waived.
- For brownfield: every migration stage in `16-` has success criteria, rollback triggers (SLI-observable), and a rollback procedure. Stages involving data movement additionally name a pattern from the `designing-for-integration-and-migration` data-migration pattern selection table with its six required fields (pattern, comparison harness, divergence SLO, abort criterion, cutover reversibility window, backpressure stance).
- No big-bang cutovers in `16-` without a waiver in `06-descoped-and-deferred.md` citing a business reason.

### Check 7 — Risk register

- Every entry in `17-` has: category, likelihood, impact, observable triggers, mitigation (design + runtime), owner, review
- Every High/High or High/Medium risk traces to a **recorded origin**: (a) an ADR that caused or accepted it, (b) an `00-scope-and-context.md` assumption or stakeholder decision (e.g., unverified brownfield context), or (c) an external driver recorded in `01-requirements.md` (e.g., regulation, vendor SLO below target). If the origin is an ADR that does not acknowledge the risk, either add the risk to that ADR's Negative consequences or raise a superseding ADR. A High-level risk with no traceable origin fails. Risks without an ADR-bearing origin must not be forced to mint ceremonial ADRs — their origin trace is to `00-` or `01-`.
- Overloaded components flagged by `03-` appear in `17-` with a risk entry
- No ops-generic risks ("server could fail") in `17-` — they belong to operational runbooks

### Check 8 — Enterprise binding (if activated)

- `togaf-deliverable-map.md` covers every numbered artifact with an ADM phase
- `archimate-model/` layer files have no cross-layer element mistakes (application components on business layer, etc.)
- Viewpoint files name their concerns and selected elements — not colour-filtered copies

### Sampling and audit protocol — making checks substantive

An LLM or human running this gate is susceptible to ticking every box without reading the artifacts. For each check, perform the specified procedure and record the sampled items in the gate report. A gate report that says "Check 2: PASS" without naming the sampled IDs is a ceremonial gate report.

| Check | Procedure |
|-------|-----------|
| Check 1 / 1b | List files actually present in the workspace. Compare to tier requirements. Do not rely on prior knowledge of the workspace. |
| Check 2 | Pick 3 random `FR-*` IDs, 3 random `NFR-*` IDs, and 3 random components. For each, trace through `01-`/`02-` → `14-` → `09-`. For each sampled component, diff its `09-` NFR-contract table against its `03-` row. A single broken trace or mismatch fails the check. |
| Check 3 | Pick 3 random NFRs from `02-`. Verify target, measurement method, and source are all present; verify none are adjective-only. |
| Check 4 (audit, not sample) | For every ADR at Status=Proposed or Status=Superseded, read Drivers and Consequences in full. For Status=Accepted ADRs, sample at least 50% (minimum 3). Count Negatives/trade-offs and alternatives. Rationale for full-pass on Proposed/Superseded: those states carry the highest risk of missing rigor. |
| Check 5 | List decisions in `05-`. List ADR titles. Match them. |
| Check 6 | For every stage in `16-`, state the rollback trigger and procedure aloud in the gate report. A vague trigger becomes obvious when stated. |
| Check 7 | For every RSK entry, state the observable trigger. If you cannot describe how you would detect the risk materializing, the entry fails. For every High-level RSK, state the recorded origin (ADR / `00-` / `01-`) and confirm the origin file carries the trace. |
| Check 8 | For every ArchiMate file, confirm at least one element's layer matches the filename. |

### Gate outcome

Produce a gate report even if everything passes:

```markdown
# Consistency Gate Report

**Run date:** YYYY-MM-DD
**Tier:** [XS | S | M | L | XL]
**Brownfield:** [yes | no]
**Enterprise mode:** [activated — driver | not activated — reason]

## Check 1 — File presence: PASS (tier: M; files present: [list])
## Check 1b — Quality floor: PASS (artifacts audited: 07, 08, 09, 10, 11, 13)
## Check 2 — Traceability: PASS (sampled: FR-02, FR-11, FR-17, NFR-03, NFR-07, NFR-12, components: order-api, event-router, reconciler)
## Check 3 — Quantification: FAIL (sampled: NFR-09, NFR-03, NFR-15)
- NFR-09 ("intuitive admin UI") is adjective-only. Either quantify (e.g., "admin onboarding < 30 min for a new user") or move to UX scope and remove from NFR set.
## Check 4 — ADR rigor: PASS
…

## Waivers
- [Check X, item Y]: waived because [explicit rationale]. Waiver recorded in the SAD.
```

#### What a substantive Check 2 PASS entry looks like

The sampling protocol's value is entirely in being substantive. Compare the ceremonial vs substantive forms:

**Ceremonial (not acceptable):**

```markdown
## Check 2 — Traceability: PASS
```

**Substantive (required):**

```markdown
## Check 2 — Traceability: PASS

Sampled: FR-02, FR-11, NFR-03, NFR-07; components: order-service, api-gateway, notification-service.

- FR-02 ("user receives order confirmation within 5 s"): appears in `14-` row 3; satisfied by `order-service` + `notification-service`; verified by E2E test `order/confirmation-latency`; ADR-0005 referenced. Trace complete.
- FR-11 ("admin can export orders as CSV"): appears in `14-` row 22; satisfied by `admin-service`; no ADR (correct — this is an S-tier feature using existing export framework, recorded as minor decision in `05-`). Trace complete.
- NFR-03 ("P99 write latency ≤ 200 ms"): appears in `02-` with target, measurement point (server-side span), and load-test acceptance criteria. Appears in `14-` row NFR-03; load-bearing components `write-service` + `primary-store`; verified by load-test rig `write-p99-rig`. NFR budget in `03-`: write-service ≤ 130 ms + primary-store ≤ 60 ms = 190 ms (within 200 ms). Trace complete.
- NFR-07 ("99.9% availability"): appears in `14-` row NFR-07; satisfied by `api-gateway` + `read-service` + `primary-store`; ADRs 0004 and 0008 referenced; SLO burn-rate alert named. Component `primary-store` NFR contract table in `09-` matches `03-` entry (both show "RDS Multi-AZ, automatic failover ≤ 60 s"). Trace complete.
- `order-service`: appears in `14-` rows FR-01, FR-02, FR-07, NFR-01, NFR-03, CON-REG-01. Each row populated. No orphan.
- `api-gateway`: appears in `14-` rows NFR-01, NFR-07. Marked `cross-cutting: true` for routing concern in `09-`; NFR in `02-` is traceable. Acceptable.
- `notification-service`: appears in `14-` row FR-02. Single requirement — unusual but acceptable for a narrow adapter. No orphan flag.

No broken traces found in sample. Check 2: PASS.
```

The substantive form *names the things it checked and confirms they pass*, rather than asserting the result without evidence. A Check-2 PASS that does not cite the sampled IDs and the trace through them is a ceremonial pass — treat it as FAIL.

**A waiver requires an explicit rationale.** "Waived because time-pressure" is not a rationale; it's an abdication. A rationale names the business / technical reason the check is accepted-as-failed. Waivers live in the gate report and in Appendix A of the SAD; `06-descoped-and-deferred.md` records scope decisions, not gate waivers. The distinction: a gate waiver says "this check failed and we are shipping anyway — here is why"; a `06-` entry says "this capability or artifact is outside this design's scope — here is the trigger for reconsideration." An incomplete artifact is a waiver candidate; an out-of-scope feature is a `06-` candidate.

## Emitting `99-solution-architecture-document.md`

```markdown
# Solution Architecture Document

**Project:** [name]
**Version:** [semver]
**Date:** YYYY-MM-DD
**Authors:** [roles]
**Consistency gate:** PASS | PASS-WITH-WAIVERS | FAIL-WAIVED-FOR-RELEASE (gate report date: YYYY-MM-DD; see Appendix A for waivers)

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

## 8. NFR satisfaction
[From 03-: how each NFR is satisfied, with links to the load-bearing components. This section precedes Key decisions because most ADRs cite NFRs as drivers; reading NFR satisfaction first makes the ADRs scannable.]

## 9. Key decisions
[ADR summary — title, decision, reason driven by which NFR/FR/CON IDs, link to full ADR]

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

**Response:** "Orphans are scope reductions (record in `06-`) or missing implementations (fix in `09-`). Either way, they surface before emission. A SAD with hidden orphans is fiction."

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
