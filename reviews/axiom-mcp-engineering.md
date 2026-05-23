# Review: axiom-mcp-engineering

**Version:** 0.1.0  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent
**Pack path:** `/home/john/skillpacks/plugins/axiom-mcp-engineering/`

---

## 1. Inventory

### Filesystem

```
plugins/axiom-mcp-engineering/
├── .claude-plugin/
│   └── plugin.json
├── agents/                                 (empty)
├── commands/                               (empty)
└── skills/
    └── using-mcp-engineering/
        └── SKILL.md                        (247 lines)
```

### Components

| Type | Count | Notes |
| --- | --- | --- |
| Skills (router) | 1 | `using-mcp-engineering` — pure router; references 13 future sheets |
| Skills (reference sheets) | **0** | All 13 sheets referenced inline are **unwritten** |
| Commands | **0** | Three planned: `/design-mcp-server`, `/review-mcp-server`, `/audit-mcp-tools` |
| Agents | **0** | Two planned: `mcp-server-architect`, `mcp-server-critic` |
| Hooks | 0 | N/A for this pack |
| Slash-command wrapper | **0** | No `/home/john/skillpacks/.claude/commands/mcp-engineering.md` |
| Marketplace registration | **No** | `axiom-mcp-engineering` is absent from `.claude-plugin/marketplace.json` |

### Plugin metadata
`plugin.json:1-26` — name `axiom-mcp-engineering`, version `0.1.0`, license CC-BY-SA-4.0, 13 keywords, accurate author/repo block. Description is long-form (full paragraph) and accurately characterises the pack as a scaffold release (`"Router + 13 sheets (scaffold; v0.1.0 ships router only)."`).

### Self-declared scope
`SKILL.md:209-222` "v0.1.0 Scope" section explicitly states: shipped = router + plugin metadata; not yet shipped = 13 sheets, 3 slash commands, 2 SME agents.

---

## 2. Domain & Coverage

### User-defined scope (inferred from router description and Overview)
- **Intent:** Engineering MCP (Model Context Protocol) servers as production-grade interfaces for LLM-agent clients. Server-side discipline.
- **In scope:** Tool API surface design; idempotency under retry; structured error envelopes; schema versioning under model drift; transport reliability (stdio/HTTP); output-shape and pagination; the four MCP primitives (tools/resources/prompts/sampling); composition across servers; tool-call observability; golden-conversation testing.
- **Out of scope (declared at `SKILL.md:38-44`):** General REST/GraphQL APIs (→ `/web-backend`); client-side prompt engineering / agent loop (→ `/llm-specialist`); general in-process plugin systems (→ `/system-architect` or `/procedural-architecture`); cryptographic provenance (→ `/audit-pipelines`); system-level replay infrastructure (→ `/determinism-and-replay`); operating MCP as a host/client.
- **Audience:** Practitioner-to-expert. Two role epistemics (architect / critic) over one shared corpus.

### Coverage map (per router's planned 13 sheets, SKILL.md:180-207)

| Cluster | Sheet | Status |
| --- | --- | --- |
| Architect — what server IS | `tool-api-design.md` | Missing |
| | `mcp-primitive-selection.md` | Missing |
| | `output-shape-and-pagination.md` | Missing |
| Discipline — what server GUARANTEES | `idempotency-and-atomicity.md` | Missing |
| | `error-envelopes-and-recovery.md` | Missing |
| | `schema-versioning-and-drift.md` | Missing |
| | `authentication-and-trust.md` | Missing |
| Beyond tools | `resources-prompts-sampling.md` | Missing |
| | `composition-and-namespaces.md` | Missing |
| | `transport-reliability.md` | Missing |
| Quality & ops (critic) | `testing-mcp-servers.md` | Missing |
| | `observability-for-tool-calls.md` | Missing |
| | `mcp-server-smells.md` | Missing |
| Boundary sheet (absorbed into routing for v0.1.0) | `mcp-boundary-and-handoffs.md` | Deferred per `SKILL.md:207` |

**Coverage status:** 0 / 13 sheets written. Domain *map* is comprehensive; domain *implementation* is empty.

### Domain currency
MCP is an evolving domain (Anthropic's protocol has shifted since launch; the four primitives have been refined; the wire format and capability negotiation are still in flux). Sheets, when written, should be flagged for periodic currency review against the live spec.

### Gap analysis

**Foundational gaps (blocking — domain unaddressable without them):**
- All 13 sheets. The router cannot do work alone; the routing table points to files that do not exist.

**Tooling gaps (medium):**
- No slash-command wrapper (`/mcp-engineering` or similar). Without it, users cannot invoke the router via `/X`. Per `/home/john/skillpacks/CLAUDE.md` (lines 78-97), router skills *must* be paired with a `.claude/commands/<name>.md` wrapper at repo root.
- No marketplace.json entry. The pack is invisible to `/plugin marketplace add tachyon-beep/skillpacks` users.

**Process/role gaps (per v0.1.0 roadmap):**
- 3 slash commands (`/design-mcp-server`, `/review-mcp-server`, `/audit-mcp-tools`) — not yet created.
- 2 SME agents (`mcp-server-architect`, `mcp-server-critic`) — not yet created. Both, when implemented, must follow the SME Agent Protocol (Confidence/Risk/Information Gaps/Caveats sections).

**Cross-pack cross-reference gap (minor):**
- The router references `/procedural-architecture` at `SKILL.md:40, 76-86`. The plugin `axiom-procedural-architecture` exists at `plugins/axiom-procedural-architecture/`, but **no slash-command wrapper** `.claude/commands/procedural-architecture.md` exists, so the cited slug is non-functional today. (This is a defect of the procedural-architecture pack's slash-command exposure, not of this pack — but the cross-reference is currently broken from the user's perspective.)

---

## 3. Fitness Scorecard

Per `reviewing-pack-structure.md` rubric (Critical / Major / Minor / Pass).

| Dimension | Grade | Evidence |
| --- | --- | --- |
| Router quality | **Pass (with caveat)** | Router is unusually well-structured for a scaffold: explicit Architect/Critic role split (`SKILL.md:114-124`), pipeline-position diagrams (`SKILL.md:48-110`), routing table with 17 symptom→sheet rows (`SKILL.md:154-174`), consistency gate (`SKILL.md:224-239`), anti-overconfidence section (`SKILL.md:241-247`). The Start Here flow (`SKILL.md:128-150`) is well-segmented by user intent. Caveat: every internal link in the routing table is currently dead. |
| Skill descriptions | **Pass** | `SKILL.md:3` is a model-quality description: stacks symptoms ("tools confuse agents, return unstructured errors, deadlock under concurrent calls, double-execute under retry, lose state across reconnects") with explicit "Do not use for..." cross-pack handoffs to four sibling packs by slug. Matches the description discipline of `axiom-determinism-and-replay` and `axiom-rust-workspaces`. |
| Frontmatter conformance | **Pass** | `SKILL.md:1-4` — two-key frontmatter (`name:`, `description:`); no extraneous fields; YAML parses cleanly; description on a single physical line (no embedded literal newlines that would break strict YAML parsers). |
| Component cohesion | **Critical** | 0 sheets / 0 commands / 0 agents for a router that names 13 sheets, 3 commands, 2 agents. The pack as shipped does no work — it routes to nothing. The `v0.1.0 Scope` block (`SKILL.md:209-222`) is honest about this, but honesty does not change the user-facing reality: invoking this pack today returns guidance to read files that do not exist. |
| Slash-command exposure | **Major** | No `.claude/commands/mcp-engineering.md` wrapper exists. Per project convention (`CLAUDE.md:78-97`), every router skill ships with a paired slash command. Users cannot reach this router via `/mcp-engineering` today. |
| Marketplace registration | **Major** | `axiom-mcp-engineering` is absent from `/home/john/skillpacks/.claude-plugin/marketplace.json`. Confirmed via `grep -n "mcp" .claude-plugin/marketplace.json` returning zero matches. Users running `/plugin marketplace add tachyon-beep/skillpacks` cannot discover this pack. |
| SME agent protocol | **N/A (deferred)** | Zero agents exist. Roadmap states two SME agents in v0.2.0+; both planned roles (architect / critic) are SME-style and must follow the protocol when shipped. Router pre-commits the critic to severity + evidence (`SKILL.md:239`), which is consistent with SME protocol. No violation today; obligation on future ships. |
| Anti-pattern coverage | **Pass (router scope)** | `SKILL.md:241-247` Anti-Overconfidence section is substantive: warns architects against database-mirror tool surfaces, REST-style error envelopes, "schemas that obviously will not change"; warns critics against rubber-stamping and reading-the-surface-the-way-the-architect-wrote-it. Smell catalog *referenced* on `SKILL.md:205` (sheet 13) but its content does not exist yet. |
| Cross-skill linkage | **Pass (with one broken slug)** | Pipeline-position section (`SKILL.md:48-110`) cross-links to `/web-backend`, `/llm-specialist`, `/procedural-architecture`, `/audit-pipelines`, `/determinism-and-replay` with prose explaining each relationship. 4/5 target slugs resolve to a real `.claude/commands/*.md` wrapper today. `/procedural-architecture` does not — this is a defect of the *target* pack, but the user-visible breakage is in this pack's text. |

### Overall: **Critical (acknowledged scaffold)**

The pack is structurally **Critical** under the rubric — component cohesion fails because the router routes to nothing — but the pack itself **self-declares as v0.1.0 scaffold** (`SKILL.md:209-222`). The rubric does not have a "scaffold" grade. The honest reading is:

- As a **finished product**, this pack is unusable: Critical.
- As a **scaffold release for sheets to be written against**, the router is high quality and ready to host the 13 sheets when they are produced.

Recommend treating this review as gating the v0.2.0 / v1.0.0 ship: the router is acceptable to merge as-is, but the pack should not be advertised on marketplace as shippable until at least the most-cited sheets (`tool-api-design`, `idempotency-and-atomicity`, `error-envelopes-and-recovery`, `output-shape-and-pagination`, `mcp-server-smells`) exist.

---

## 4. Behavioral Tests

The only behavior available to test is the router itself (`using-mcp-engineering`). I cannot run the standard "discover the right sheet for symptom X" pressure test because no sheets exist — discovery would route to a dead link in every case. Instead, I ran three router-scoped pressure tests on the SKILL.md text.

### Test R1: Boundary discrimination (would the model correctly *decline* to use this pack?)

**Scenario:** "I'm designing a REST endpoint for a customer-facing dashboard. The frontend will hit it. Should I use this pack?"

**Expected behaviour:** Router declines, redirects to `/web-backend`.

**Observation from `SKILL.md:38-39`:**
> "Designing a general REST or GraphQL API for human-written clients → use `/web-backend`. MCP discipline assumes an LLM client; non-MCP API design has different constraints and a different audience."

**Result:** **Pass.** The boundary is named, the redirect target is correct, and the reason is given (audience asymmetry).

### Test R2: Confusable-domain pressure (would the model wrongly pull this pack for a near-miss query?)

**Scenario:** "I'm writing a Python plugin system with pluggy and entry points so users can extend my CLI tool."

**Expected behaviour:** Router declines (this is in-process plugin architecture, not MCP).

**Observation from `SKILL.md:40`:**
> "Building general in-process plugin systems (pluggy, entry points, registry patterns) → use `/system-architect` or `/procedural-architecture`. MCP is one specific protocol with one specific client model; plugin architecture in your own runtime is a different problem."

**Result:** **Pass.** The exact technologies (`pluggy`, `entry points`) are named, which inoculates against the common "plugins ≈ MCP" confusion. Cross-reference to `/procedural-architecture` is currently broken at the slash-command layer (no wrapper exists), but the routing instruction is correct.

### Test R3: Time-pressure rationalisation against the consistency gate

**Scenario:** "I shipped a new MCP tool yesterday under deadline pressure. It works in manual tests. We're cutting a release in two hours. Can I skip the consistency-gate checklist?"

**Expected behaviour:** Router resists the shortcut. The consistency gate (`SKILL.md:224-239`) is the discipline-bearing surface; under pressure, the model often rationalises skipping it.

**Observation from `SKILL.md:226-227`:**
> "Run this checklist before declaring any architect or critic deliverable done. **Failures are blocking, not advisory.** Silent passes are the failure mode this pack exists to prevent."

And `SKILL.md:234`:
> "At least one golden-conversation regression test exists per shipping tool. Not 'we asked Claude to use it and it worked.' [...] If the surface is not regression-protected, every new model version is a release candidate by default."

**Result:** **Pass.** The text pre-empts both the time-pressure shortcut ("blocking, not advisory") and the manual-test shortcut ("'we asked Claude to use it and it worked' is not a test"). Plus the Anti-Overconfidence section (`SKILL.md:241-247`) explicitly names the architect's "first design feels clean" trap.

### Test R4: Architect/critic disagreement enforcement (the load-bearing pipeline claim)

**Scenario:** The router claims at `SKILL.md:122` that "If architect and critic always agree, the pipeline is broken." Test: would a model running this pack treat zero-disagreement as a defect of the critic, as required?

**Observation from `SKILL.md:122` and `SKILL.md:247`:**
> "A run that produces no disagreement is evidence the critic is reading the surface the way the architect wrote it — same blind spots, same defaults — and the audit is theatre. Treat zero-disagreement runs as a defect of the critic, not as a virtue of the architect."

And:
> "If the run produced no architect-critic disagreement, the pipeline is broken. Re-run the critic with a fresh frame, or assume the audit is theatre."

**Result:** **Pass.** The claim is repeated twice with operational consequence (re-run the critic with a fresh frame). This is consistent with the dual-role-with-disagreement pattern used by the more mature sibling pack `axiom-determinism-and-replay`. Whether the *agents* (when they exist) actually enforce it is a v0.2.0 test, not testable today.

### Tests not runnable

- **Specialist-sheet pressure tests** — N/A, zero sheets exist.
- **Command behavioural tests** — N/A, zero commands exist.
- **Agent SME-protocol tests** — N/A, zero agents exist.
- **End-to-end "architect designs a tool, critic finds the smell"** — N/A, neither role has an executable artifact.

---

## 5. Findings

### Critical

**C1. Pack does no work without the 13 sheets.**
- **Evidence:** `SKILL.md:180-207` lists 13 sheets, all referenced by the routing table (`SKILL.md:154-174`). All 13 are missing from disk (`ls plugins/axiom-mcp-engineering/skills/using-mcp-engineering/` shows only `SKILL.md`).
- **Impact:** Every routing decision the router makes today points to a non-existent file. A user invoking the router gets a high-quality menu of files that do not exist.
- **Note:** This is self-acknowledged at `SKILL.md:209-222`. The criticality is structural, not deceptive. The pack should not be advertised as production-ready until at least the foundation-cluster sheets ship.

### Major

**M1. Plugin not registered in marketplace.json.**
- **Evidence:** `grep "axiom-mcp" /home/john/skillpacks/.claude-plugin/marketplace.json` returns zero matches; all sibling axiom-* plugins ARE registered (lines 14, 25, 41, 52, 65, 79, 95, 109, 125, 142...).
- **Impact:** Users running `/plugin marketplace add tachyon-beep/skillpacks` followed by `/plugin` will not see this pack. It is effectively invisible to consumers.
- **Fix:** Add catalog entry to `marketplace.json` with `name: "axiom-mcp-engineering"`, `source: "./plugins/axiom-mcp-engineering"`, appropriate `category` (likely `engineering` or a new `protocols` bucket), and version `0.1.0`. Consider gating this on at least the first sheet shipping, so first-install users get something more than a stub.

**M2. No slash-command wrapper at `.claude/commands/mcp-engineering.md`.**
- **Evidence:** `ls /home/john/skillpacks/.claude/commands/ | grep -i mcp` returns nothing. Project convention (`/home/john/skillpacks/CLAUDE.md:78-97`) requires every router skill be paired with a slash-command wrapper. Sibling packs (`pyo3-interop.md`, `rust-workspaces.md`, `determinism-and-replay.md`) all comply.
- **Impact:** Users cannot reach the router via `/mcp-engineering`. They must trigger it by skill description match alone, which is less reliable.
- **Fix:** Create `/home/john/skillpacks/.claude/commands/mcp-engineering.md` modelled on `pyo3-interop.md` — a short wrapper that names the sibling-pack relationships, lists the sheets-to-be (or a subset of them, marked roadmap), and invokes the `using-mcp-engineering` skill.

**M3. Roadmap obligation: 3 commands + 2 SME agents not yet existing.**
- **Evidence:** `SKILL.md:124` names `/design-mcp-server`, `/review-mcp-server`, `/audit-mcp-tools`, and agents `mcp-server-architect`, `mcp-server-critic`. `commands/` and `agents/` directories are empty.
- **Impact:** The router pre-commits the pack to a producer/critic two-role architecture with disagreement enforcement, but the artifacts that *enforce* that architecture do not exist. The discipline lives entirely in prose, which is brittle.
- **Fix (when sheets ship):** Both agents must follow `meta-sme-protocol:sme-agent-protocol` with the four-section output contract (Confidence / Risk / Information Gaps / Caveats). The pack already promises severity + evidence at `SKILL.md:239`, which is consistent.

### Minor

**m1. Cross-reference to `/procedural-architecture` is currently dead.**
- **Evidence:** `SKILL.md:40, 76-86` link to `/procedural-architecture`. The plugin exists (`plugins/axiom-procedural-architecture/`), but no `.claude/commands/procedural-architecture.md` wrapper exists. Confirmed by `ls .claude/commands/ | grep -i proc` → no matches.
- **Impact:** Users following the cross-link will get a "no such command" failure. Note: this is a defect in the *target* pack's slash-command exposure, but it manifests as broken text in this pack.
- **Fix:** Either (a) fix the target pack's wrapper independently; (b) replace the slug citation in this pack with `using-procedural-architecture` skill name; or (c) leave as-is and accept that the cross-link will work once the target pack ships its wrapper.

**m2. Sheet 7 (`mcp-boundary-and-handoffs.md`) intentionally absorbed into the routing table; ensure this stays explicit when sheets are written.**
- **Evidence:** `SKILL.md:207` — "Boundary sheet (absorbed into the routing table for v0.1.0; promoted to its own sheet if the routing question becomes a recurrent ask)".
- **Impact:** When the 13 numbered sheets ship, future-self may forget the boundary sheet is *deliberately* absent and add a 14th. Risk of inconsistency between the router's "13 sheets" framing and what ships.
- **Fix:** When promoting the boundary sheet (if it happens), update `SKILL.md:180-207` and the router description.

**m3. Description is exceptionally long (one ~115-word paragraph at `plugin.json:4`).**
- **Evidence:** `plugin.json:4` — single field, ~115 words. Most sibling packs (e.g. `axiom-determinism-and-replay`, `axiom-pyo3-interop`) hold descriptions to ~30-50 words.
- **Impact:** Discovery surfaces (e.g. `/plugin marketplace` listing) typically truncate. Important information may not surface for users browsing the catalog.
- **Fix:** Consider trimming to a one-sentence functional summary (the first 20 words are good) plus a `"...applies wherever you ship an MCP server agents will retry, drift against, and operate concurrently."` tagline. Keep the longer form in `SKILL.md`'s Overview.

### Polish

**p1. Sibling-pack `axiom-web-backend` is named at `SKILL.md:49, 112` as a contrast pair.** When that pack's `using-web-backend` router is updated, consider adding a reciprocal cross-link from there to here. (Not blocking; nice-to-have for discoverability symmetry.)

**p2. `SKILL.md:124` parenthetical** — `"(Commands and agents are roadmap for v0.2.0; the v0.1.0 ship is router only.)"` — appears once but the analogous note about sheets being roadmap appears only at lines 209-222. Consider a one-line "scaffold notice" at the top of the Overview so a hurried reader is not surprised when every internal link is dead.

**p3. Routing-table row at `SKILL.md:172`** mentions `authentication-and-trust.md` (sheet 7) but the entry-track for "operator" at `SKILL.md:144-149` does not cite it; for a security-adjacent sheet this is a small completeness gap. Once the sheet ships, consider adding it to the operator track.

---

## 6. Recommended Actions

### Immediate (gates v0.1.0 → v0.2.0)

1. **Add marketplace.json entry** for `axiom-mcp-engineering` so the pack is discoverable. Mark it with `version: 0.1.0` and consider tagging it as `experimental` or `scaffold` in the description if the marketplace schema supports it.
2. **Create `.claude/commands/mcp-engineering.md` slash-command wrapper** following the project convention. Use `pyo3-interop.md` as the template.
3. **Add a one-line scaffold notice** to the top of `SKILL.md` (above Overview) so first-time readers understand the sheet links are roadmap, not bugs.

### Foundation-cluster sheets (gates the pack becoming actually useful — v0.2.0)

In priority order, write the four most-routed-to sheets first:

1. `tool-api-design.md` — most-cited sheet (5 routing-table rows).
2. `mcp-server-smells.md` — load-bearing for the critic role; the consistency gate at `SKILL.md:235` requires the smell catalog be runnable as a checklist.
3. `idempotency-and-atomicity.md` — high-stakes (data corruption under retry), cited at `SKILL.md:24, 162, 163` and the consistency gate.
4. `error-envelopes-and-recovery.md` — required to make the consistency gate's three error classes (`SKILL.md:231`) operationally testable.

### Remaining sheets (v0.3.0 or later)

5-13. The other nine sheets in the order listed at `SKILL.md:180-207`.

### Roles (v0.4.0 / v1.0.0)

- Build the three slash commands and the two SME agents only after enough sheets exist to make their workflows non-vacuous. Both agents must comply with `meta-sme-protocol:sme-agent-protocol` (Confidence / Risk / Information Gaps / Caveats output sections).

### Once-only cleanups

- Fix or rephrase the `/procedural-architecture` cross-link (m1).
- Decide on the boundary-sheet promotion question (m2) before shipping the 13 sheets.
- Tighten the `plugin.json` description (m3).

---

## 7. Reviewer Notes

- **The pack's self-honesty is unusual and good.** `SKILL.md:209-222` ("v0.1.0 Scope") explicitly enumerates what is and is not shipped, which made the inventory step trivial. Most scaffold packs hide their incompleteness; this one names it. Treat that as a model for other scaffold ships.
- **The router quality is genuinely high for a scaffold.** The dual-role-with-mandatory-disagreement pattern at `SKILL.md:114-124` and the consistency gate at `SKILL.md:224-239` are the load-bearing discipline of this pack, and they are already operationally usable as prose. A producer or critic could extract value from the router alone before any sheet ships — which matches the author's claim at `SKILL.md:222`.
- **The Critical grade is mechanical, not editorial.** Under the rubric, "component cohesion" fails when the router routes to nothing. The pack is genuinely Critical-as-shipped; it is also genuinely a high-quality scaffold. Both can be true. Recommend the marketplace.json entry, when added, flags the pack appropriately so users know what they are installing.
- **Stage 5 was skipped per task instructions.** No edits performed; this is a report-only review.
- **No specialist sheets were behaviorally tested** — they do not exist. The four router-scoped pressure tests passed. When sheets ship, the next review should run the full gauntlet (A/B/C from `testing-skill-quality.md`) against each.
- **Confidence assessment.** High confidence on inventory and structural findings (filesystem + grep are authoritative). Medium confidence on the priority order of sheet authorship — it is reasoned from routing-table reference counts and consistency-gate dependencies, but the pack author may have a different ordering in mind.
- **Risk assessment.** Low risk to leave as-is short-term (the pack is invisible without marketplace registration anyway). Medium risk if marketplace registration is added *without* first-foundation sheets — users will install a stub and get frustrated. Recommend bundling M1 (marketplace entry) with at least the first two foundation sheets.
- **Information gaps.** The author's intended cadence between v0.1.0 and v0.2.0 is unknown. If this is "scaffold landed today, sheets next week", the marketplace/slash-command fixes can wait. If this is "scaffold landed today, sheets in three months", users will look for the pack and not find it.
