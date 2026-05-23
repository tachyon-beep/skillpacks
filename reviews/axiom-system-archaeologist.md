# Review: axiom-system-archaeologist
**Version:** 1.6.1  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

## 1. Inventory

### Plugin Metadata

- `plugins/axiom-system-archaeologist/.claude-plugin/plugin.json` declares `name=axiom-system-archaeologist`, `version=1.6.1`, license `CC-BY-SA-4.0`.
- Self-described as "5 commands, 4 agents" — confirmed by file inventory.
- Marketplace registration: `.claude-plugin/marketplace.json` entry present, source `./plugins/axiom-system-archaeologist`, with the same "5 commands, 4 agents" claim and the ultralarge-tier callout matching the plugin.json description.

### Router Skill + Reference Sheets (16 total)

Router: `skills/using-system-archaeologist/SKILL.md` (430 lines).

Reference sheets in `skills/using-system-archaeologist/`:

| Sheet | LOC | Purpose |
|---|---|---|
| `analyzing-dependencies.md` | 294 | Dependency-graph extraction methodology |
| `analyzing-test-infrastructure.md` | 460 | Test corpus archaeology |
| `analyzing-unknown-codebases.md` | 441 | Subsystem catalog contract |
| `assessing-code-quality.md` | 281 | Quality assessment contract |
| `creating-architect-handover.md` | 266 | Handover doc contract |
| `deliverable-options.md` | 148 | Options A–G menu used at Step 1.5 |
| `documenting-system-architecture.md` | 625 | Final report contract |
| `findings-schema.md` | 469 | Load-bearing YAML schema for ultralarge track |
| `generating-architecture-diagrams.md` | 369 | C4 diagram patterns |
| `incremental-analysis.md` | 446 | Delta analysis on changed regions |
| `language-framework-patterns.md` | 554 | Per-language idioms |
| `mapping-security-surface.md` | 376 | Security boundary mapping |
| `module-by-module-with-scribe.md` | 268 | Ultralarge per-module loop |
| `partitioning-ultralarge-repos.md` | 237 | Tier criteria + partition protocol |
| `specialist-integration.md` | 162 | Cross-pack specialist handoff |
| `validating-architecture-analysis.md` | 428 | Validation contract |

### Commands (5)

| Command | LOC | argument-hint | allowed-tools |
|---|---|---|---|
| `analyze-codebase.md` | 233 | `[directory_or_scope]` | `Read, Grep, Glob, Bash, Task, Write, AskUserQuestion` |
| `analyze-dependencies.md` | 345 | `[workspace_path]` | `Read, Grep, Glob, Bash, Write` |
| `analyze-ultralarge.md` | 245 | `[directory_or_scope]` | `Read, Grep, Glob, Bash, Task, Write, AskUserQuestion` |
| `generate-diagrams.md` | 262 | `[catalog_file_or_workspace]` | `Read, Grep, Glob, Bash, Task, Write` |
| `validate-analysis.md` | 239 | varies | (per-file, not inspected) |

All command frontmatter uses the quoted JSON-style array convention (`["Read", "Bash"]`).

### Agents (4)

| Agent | Model | SME Protocol declared | `tools:` |
|---|---|---|---|
| `codebase-explorer.md` | opus | yes (line 2, line 10) | omitted (inherits) |
| `analysis-validator.md` | opus | yes (line 2, line 10) | omitted |
| `module-reviewer.md` | sonnet | yes (line 2, line 10) | omitted |
| `subsystem-scribe.md` | sonnet | yes (line 2, line 10) | omitted |

All four agents are SME agents (reviewer / auditor / scribe roles), and all four declare the SME protocol header in the description and a `**Protocol**:` block in the body citing `meta-sme-protocol:sme-agent-protocol`. None declare a `tools:` key — matches the dominant marketplace convention (~60/65 agents omit it).

### Slash-Command Wrapper

- `/home/john/skillpacks/.claude/commands/system-archaeologist.md` — **PRESENT** (exposes the router skill as `/system-archaeologist`). The wrapper's "When to Use" guidance reads as a long-form complement to the router's `description:` rather than contradicting it.

No per-command repo-root wrappers (e.g., `/analyze-codebase`, `/analyze-ultralarge`) — but the pack-internal commands at `plugins/axiom-system-archaeologist/commands/*.md` are themselves the user-invocable surface (these are addressed by Claude Code's plugin-command machinery, not by the repo-root `.claude/commands/` wrappers reserved for router skills). This is correct and matches the convention in CLAUDE.md.

### Total Component Counts

- **1** router skill (the only SKILL.md in the pack — no peer skills, all other content is reference sheets)
- **15** reference sheets in the router skill's directory (no SKILL.md frontmatter; these are content files loaded by the router or its agents)
- **5** commands
- **4** agents (all SME-protocol-declaring)
- **0** hooks (none expected for this pack type)
- **1** slash-command wrapper at the repo root
- **1** marketplace catalog entry

Line counts: 430 (router SKILL.md) + 5,375 (reference sheets) + 1,324 (commands) + 687 (agents) = **7,816 LOC of pack content** (excluding plugin.json + marketplace entry). The full inventory (`wc -l` on every file in the pack) is **8,265 LOC**, which matches.

This puts the pack near the upper end of the marketplace's size distribution but not unusually large — the per-pack averages in the marketplace are ~5,000–8,000 LOC for the "+13 sheets, +3 commands, +2 agents" template, and this pack is at +15/+5/+4 — slightly above the template.

## 2. Domain & Coverage

**Pack intent (inferred from plugin.json + SKILL.md):** Deep architectural archaeology of *existing* codebases through subagent-driven exploration. The pack's center of gravity is **orchestration discipline under context pressure**, not pattern recognition. The router SKILL.md is explicit (lines 16–55) that the loaded skill is a *coordinator*, with detailed work delegated to subagents.

**Boundaries (explicitly stated):**

- IN scope: discovery, subsystem cataloging, C4 diagrams, code-quality assessment, security surface mapping, test infrastructure analysis, dependency analysis, incremental delta analysis, architect handover.
- OUT of scope (handed to other packs): forward solution design (`axiom-solution-architect`), assessment-as-judgment (`axiom-system-architect`), language-specific code review (`axiom-python-engineering`, etc.), security threat modeling (`ordis-security-architect`). `specialist-integration.md` documents these handoffs.

**Coverage assessment vs. domain map:**

| Coverage area | Status | Evidence |
|---|---|---|
| Foundational: workspace discipline | Exists | SKILL.md §Step 1; analyze-codebase Step 1 |
| Foundational: coordination logging | Exists | SKILL.md §Step 2; coordination log format specified |
| Foundational: holistic-before-detailed | Exists | SKILL.md §Step 3; tied to discovery contract |
| Core: subsystem catalog contract | Exists | `analyzing-unknown-codebases.md` |
| Core: parallel vs sequential decision | Exists | SKILL.md §Step 4 |
| Core: subagent delegation pattern | Exists | SKILL.md §Step 5 |
| Core: validation gates (independent validator) | Exists | SKILL.md §Step 6 + `analysis-validator` agent |
| Core: C4 diagram generation | Exists | `generating-architecture-diagrams.md` + `/generate-diagrams` |
| Core: dependency analysis | Exists | `analyzing-dependencies.md` + `/analyze-dependencies` |
| Advanced: code quality assessment | Exists | `assessing-code-quality.md` |
| Advanced: security surface mapping | Exists | `mapping-security-surface.md` |
| Advanced: test infrastructure | Exists | `analyzing-test-infrastructure.md` |
| Advanced: incremental delta | Exists | `incremental-analysis.md` |
| Advanced: architect handover | Exists | `creating-architect-handover.md` |
| Advanced: language patterns | Exists | `language-framework-patterns.md` |
| Advanced: cross-pack specialist integration | Exists | `specialist-integration.md` |
| **Ultralarge tier**: criteria | Exists | `partitioning-ultralarge-repos.md` §Tier Definitions |
| **Ultralarge tier**: operator interview | Exists | `partitioning-ultralarge-repos.md` §Step 1; `/analyze-ultralarge` §Step 3 |
| **Ultralarge tier**: partition manifest | Exists | `partitioning-ultralarge-repos.md` §Step 3 |
| **Ultralarge tier**: per-module loop | Exists | `module-by-module-with-scribe.md`; `/analyze-ultralarge` §Step 6 |
| **Ultralarge tier**: load-bearing YAML schema | Exists | `findings-schema.md` (469 LOC) |
| **Ultralarge tier**: scribe merge protocol | Exists | `subsystem-scribe` agent + `findings-schema.md` |
| **Ultralarge tier**: calibration-driven YAML self-validation | Exists | `findings-schema.md` lines 397–424; `module-reviewer.md` lines 94–106; `subsystem-scribe.md` lines 45–52 |
| **Ultralarge tier**: inbound-dependency reconciliation pass | Exists | `/analyze-ultralarge` §Step 7 |

**Gap analysis:** no foundational or core gaps identified. The pack is unusually complete — Stage 1's coverage map maps cleanly onto existing components. The only thematically adjacent topic that is *not* present is **bus-factor / institutional-knowledge mapping** (who-knows-what across the codebase), which would be a Medium-priority gap if it were claimed as in-scope; the pack does not claim it. Other archaeology-adjacent topics deliberately handed off to other packs:

- **Forward design** (greenfield architecture) → `axiom-solution-architect` — referenced in router SKILL.md line 47 contextually and in `specialist-integration.md`.
- **Architectural critique / quality judgment** ("is this good?") → `axiom-system-architect` — explicitly excluded from the `codebase-explorer` agent's activation examples (codebase-explorer.md lines 42–49: "Do NOT activate - assessment question, use axiom-system-architect").
- **Language-specific code review** → per-language packs (`axiom-python-engineering`, `axiom-rust-engineering`, etc.) via `specialist-integration.md` table at lines 11–16.
- **Threat modeling** → `ordis-security-architect` — `mapping-security-surface.md` is in scope for *surface mapping* (where the boundaries are), but threat modeling proper escalates.

This pattern — clear in-scope/out-of-scope with named handoffs — is exemplary and matches the marketplace's specialist-pack philosophy.

**Research currency:** Stable domain (C4 model, dependency analysis, subsystem decomposition are mature techniques). No Phase A research needed. The pack does cite one piece of empirical research: the v1.6.0→v1.6.1 calibration finding that reviewers can produce invalid YAML while self-reporting "self-check pass" (cited in `findings-schema.md` line 401, `module-reviewer.md` line 94, and `subsystem-scribe.md` line 52). This is operationally relevant calibration evidence, not domain-research currency.

## 3. Fitness Scorecard (8 dimensions)

| Dimension | Rating | Evidence |
|---|---|---|
| **Coverage** | Pass | All foundational + core + advanced topics covered; no gaps vs. stated scope (§2 above) |
| **Structural integrity** | Pass | 16 reference sheets, 5 commands, 4 agents; consistent file layout; router SKILL.md correctly indexes all sheets at lines 414–430 |
| **Frontmatter convention compliance** | Pass | Commands use quoted-array `allowed-tools`; agents declare only `description` + `model` (~60/65 marketplace pattern); SME descriptions end with the canonical phrase |
| **Router quality** | Pass | SKILL.md `description:` is "Use when…" style (line 3); workflow numbered 1–11 with cross-references to reference sheets at lines 414–430; Step 1.5 deliverable menu added at v1.x makes the pack interactively scoped |
| **Component typing** | Pass | Commands are user-invocable initiators with `AskUserQuestion`; agents are autonomous specialists; reference sheets are content-not-skills (no frontmatter, correct); no skill/command/agent misalignment |
| **SME Protocol compliance** | Pass | All four agents declare the protocol in description AND body; `module-reviewer` and `subsystem-scribe` explicitly explain how the YAML schema *encodes* the protocol's Confidence/Risk/Gaps/Caveats sections (a sophisticated mapping, not just a sticker) |
| **Slash-command wrapper** | Pass | `/home/john/skillpacks/.claude/commands/system-archaeologist.md` present; mandatory-workflow contents do not contradict the router skill |
| **Ultralarge-tier integrity** (sampled slice) | Pass | The `/analyze-ultralarge` command (245 LOC) + `module-reviewer` agent (137 LOC) + `subsystem-scribe` agent (182 LOC) + `findings-schema.md` (469 LOC) form a self-consistent quartet: command dispatches reviewers under the schema, scribe merges with explicit conflict-resolution rules tied back to the schema, and both agents perform the calibration-driven YAML parse self-check tied to the documented empirical failure rate (~6% per the schema validation checklist) |

**Overall:** **Pass**. Structurally sound, philosophically coherent, no critical/major findings. Two minor and five polish observations only — listed in §5.

### Why each scorecard dimension lands at Pass

A brief justification for each Pass rating, with the alternative-rating thresholds noted:

- **Coverage**: 0 gaps. Threshold for Minor would be 1–3 small advanced-topic gaps; threshold for Major is 20–50% of coverage map missing per `reviewing-pack-structure.md` line 32. Coverage is far above the Minor threshold; rating stable at Pass.
- **Structural integrity**: All cross-references resolve, file layout matches the "router + reference sheets in same directory" convention (router SKILL.md line 69–80 makes this explicit and warns against the common misread of the path). No orphan files.
- **Frontmatter convention compliance**: Commands use the quoted-array `allowed-tools` form (verified across all 5 commands). Agents declare only `description` + `model`, matching the ~60/65 marketplace-wide convention; no spurious `tools:` lists (verified with `grep "tools:" agents/*.md`, returns nothing).
- **Router quality**: SKILL.md description begins "Use when…" (line 3). Workflow numbered. Step 1.5 deliverable menu is a UX feature, not a structural anomaly. The discovery vs. analysis vs. validation phases are clearly demarcated.
- **Component typing**: Commands initiate workflows with user-facing prompts (`AskUserQuestion` in `allowed-tools`). Agents are autonomous (no `AskUserQuestion`). Reference sheets carry no frontmatter (they are loaded by the router skill, not invoked directly). The typing decisions match the table at `analyzing-pack-domain.md` lines 204–209.
- **SME Protocol compliance**: All four agents declare the protocol in BOTH the description tail AND a `**Protocol**:` body block citing `meta-sme-protocol:sme-agent-protocol`. The two scribe agents (`module-reviewer`, `subsystem-scribe`) extend this by mapping the protocol's four required output sections onto the YAML schema's `confidence`, `confidence_evidence`, and `provenance` blocks — i.e., the protocol is encoded in the data, not just asserted in prose. This is more sophisticated than the marketplace median.
- **Slash-command wrapper**: Present at `.claude/commands/system-archaeologist.md`. Wrapper-vs-router content divergence is a Minor finding (M2), not a structural failure — the wrapper still works as an invocation surface.
- **Ultralarge-tier integrity**: The five-document quartet — command + two agents + partition sheet + module-loop sheet + schema sheet — is self-consistent on the state machine (DISPATCH → COLLECT → MERGE → VALIDATE → CHECKPOINT → ADVANCE appears in identical six-state form at command line 112 and sheet lines 11–14). The YAML schema is load-bearing and self-validated at multiple checkpoints. Calibration justification is documented inline.

### Per-Agent Detail

**`codebase-explorer.md` (267 LOC, model: opus).** The pack's primary subsystem-analysis agent. Activation examples at lines 26–49 enumerate both positive (codebase exploration request, architecture discovery, delegated analysis task) and negative cases (assessment questions, prioritization questions → handed off to `axiom-system-architect`). The SME protocol header at line 10 is verbatim canonical: "Confidence Assessment, Risk Assessment, Information Gaps, and Caveats sections." Model choice (opus) is justified by the synthesis-and-multi-step-diagnosis nature of cataloging.

**`analysis-validator.md` (101 LOC, model: opus).** The shortest agent in the pack, but appropriately so — it delegates methodology to `validating-architecture-analysis.md` (line 12: "Methodology: Load `skills/using-system-archaeologist/validating-architecture-analysis.md` for detailed checklists, report templates, and validation procedures"). The agent body itself carries only the activation contract, the scope boundaries, the retry limits (max 2 re-validation attempts, then escalate — lines 82–89), and the Pressure Resistance clause (lines 91–101). This is a good factoring; the agent body is the "what to do," the sheet is the "how to do it."

**`module-reviewer.md` (137 LOC, model: sonnet).** Used only in the ultralarge track. The model downgrade from opus → sonnet is deliberate: each invocation reads one module under one focus, and a single subsystem may dispatch 120+ such invocations (per `/analyze-ultralarge` lines 161–168), so cost matters. The agent body's "Read Discipline By Focus" table (lines 49–56) and "Sampling Rules For Large Modules" (lines 58–66) are the operational heart — they define exactly what each focus reads vs. doesn't read, and how to degrade gracefully on oversized modules.

**`subsystem-scribe.md` (182 LOC, model: sonnet).** Mechanical-merge specialist, also ultralarge-track-only. Like module-reviewer, sonnet is chosen because the role is bounded ("Copy, don't create") rather than synthesizing. The 9-step Merge Protocol (lines 41–148) is unusually detailed — the scribe role is normally one paragraph in other packs, but here it carries a full conflict-resolution protocol, a confidence-aggregation rule (MIN, with documented deviation criteria), and a post-write YAML-parse self-check.

## 4. Behavioral Tests

Per task brief: sample a representative slice — the ultralarge command flow + one per-module reviewer agent. Behavioral testing here is read-only inspection of the pressure-resistance and edge-case clauses, not live dispatch.

### Test 1 — Router skill, pressure scenario (discipline-enforcing skill)

**Scenario simulated mentally:** "We have 45 minutes before a stakeholder meeting. Skip validation, just give me the catalog."

The router skill (SKILL.md) anticipates exactly this:
- Line 232: "We have 45 minutes, no time for validation" → "Validation takes 5-10 minutes. Spawn validator."
- Lines 216–228 enumerate the only conditions under which self-validation is permitted (single-subsystem, <30 min, solo work, validation evidence documented). The conjunction is strict — "ANY condition is not met → SPAWN VALIDATION SUBAGENT. No exceptions."
- Lines 273–294 reframe time pressure as a *scoping* problem, not a *skip-validation* problem, with a worked timeline.

**Result:** Pass. Verbatim pressure rationalizations are listed as table rows with explicit rebuttals — a strong signal that the skill author anticipated the failure modes. The pack's history of pressure-resistance work (per CLAUDE.md / memory: v1.6.0 → v1.6.1 calibration-driven self-validation hardening) is visible in the document.

### Test 2 — `/analyze-ultralarge` command, edge case: tier criteria not met

**Scenario:** A user invokes `/analyze-ultralarge` on a 40K LOC repo with 6 subsystems (no threshold tripped).

Command at line 51–52 handles this: "If the repo does NOT cross any threshold, ask the user whether they still want the ultralarge track. Default answer is no — switch to `/analyze-codebase`." Anti-patterns section line 214 reinforces: "❌ Run this command on a repo that doesn't cross any ultralarge threshold."

**Result:** Pass. Edge case explicitly handled with a default behavior (degrade to `/analyze-codebase`) and a rationalization blocker.

### Test 3 — `module-reviewer` agent, edge case: large module

**Scenario:** Reviewer dispatched on a 4,000-LOC module under focus=internals.

Agent body lines 38–46 anticipate this:
- "<example>Task: focus=internals, but file is a 4,000-line module. Action: Activate. Read entry points + sample 5 representative methods. Mark confidence=medium with explicit sampling rationale.</example>"

Sampling rules at lines 58–62 are explicit (≥2000 LOC → 10–20% sample, `confidence: low`). The sampling note in `confidence_evidence` is "mandatory when not reading 100%" (line 63).

**Result:** Pass. The agent has a deterministic protocol for "module too big to read fully," with confidence downgrade as the safety valve.

### Test 4 — `module-reviewer` agent, pressure scenario: cross-focus temptation

**Scenario:** Reviewer doing focus=interface notices a smell.

Anti-pattern table line 84: "I'll add a quality observation while doing interface review" → "NO. Stay in focus. Cross-focus reporting breaks the merge." The scribe's merge contract (subsystem-scribe.md lines 35–48) depends on this discipline; if reviewers spread findings across focuses, the scribe's mechanical-copy guarantee breaks.

**Result:** Pass. Discipline is explicit and tied to a structural consequence (downstream merge failure), not just a stylistic preference.

### Test 5 — `subsystem-scribe` agent, edge case: malformed partial

**Scenario:** One of the four partials is unparseable YAML.

Subsystem-scribe.md lines 45–52 walk through the exact response: parse each partial with `yaml.safe_load` before merging; if any fails, STOP, do NOT attempt to fix, report to orchestrator with parser error, orchestrator re-spawns the failing reviewer. The empirical justification is cited inline: "Empirical calibration showed reviewers can produce invalid YAML while reporting 'self-check pass' — you are the second line of defense."

**Result:** Pass. Defense-in-depth (reviewer self-parse + scribe re-parse) with empirically grounded justification — this is the calibration work documented in memory as the v1.6.0→v1.6.1 fix.

### Test 6 — Router skill, real-world complexity: prior partial work

**Scenario:** "We started this analysis last week. Finish it."

The router-skill SKILL.md doesn't expand on this case (the wrapper at `.claude/commands/system-archaeologist.md` does, at lines 154–177: find existing workspace, read coordination log, assess quality, make explicit continue/archive/salvage decision). The router skill itself does NOT carry this guidance — see Minor finding M1.

**Result:** Mixed. The wrapper command file has the guidance; the router skill body does not link to or replicate it. A user landing in the router skill via auto-invocation rather than via `/system-archaeologist` will not see this protocol.

### Test 7 — `analysis-validator` agent, pressure scenario: coordinator urgency

**Scenario:** Coordinator under deadline says "just check the format, skip the full validation."

Analysis-validator.md lines 91–101 ("Pressure Resistance — NON-NEGOTIABLE"):
- "You MUST NOT: Skip checks because coordinator approved / Reduce scope due to time pressure / Accept 'just check format' when full validation required / Soften findings due to authority or urgency"
- "You are the last line of defense before bad outputs propagate."

**Result:** Pass. The validator's pressure-resistance clause is short but pointed and names the exact rationalizations.

### Test 8 — `subsystem-scribe` agent, pressure scenario: temptation to read source

**Scenario:** During merge, two partials make incompatible claims about a method's visibility. Scribe is tempted to open the source file to settle it.

Subsystem-scribe.md anti-pattern table line 154: "I read the source briefly to clarify a conflict" → "NO. You are not a reviewer. Conflicts go to provenance, severe ones flag re-review." The Core Principle at line 13–14 reinforces: "Copy, don't create. Every line in the canonical traces to a partial. If you find yourself wanting to write something the partials don't say, **stop** — that means a reviewer underspecified, and the right action is to flag it for re-review, not to backfill from your own reading."

The scope-boundaries block at lines 170–183 explicitly enumerates "You do NOT: Read source files (reviewers did that)" — this is the load-bearing invariant that keeps merge mechanical.

**Result:** Pass. The agent's role-purity is enforced at three locations (core principle, anti-pattern table, scope boundaries) — a strong "no escape hatch" design.

### Test 9 — `module-reviewer` agent, overconfidence pressure

**Scenario:** Reviewer has read a 200-LOC module thoroughly and feels confident. Tempted to mark `confidence: high`.

Module-reviewer.md anti-pattern at line 87: "I'm not sure about confidence, I'll mark high to be safe" → "Conservative is `medium` or `low`. `high` requires evidence." The schema's reviewer self-check at findings-schema.md line 401 requires `confidence_evidence` to "cite what was actually read" — not just an assertion of confidence.

The scribe then applies MIN aggregation (subsystem-scribe.md lines 105–113), so even if one reviewer over-claims, three other reviewers' lower confidence will pull the canonical down. **Result:** Pass — defense in depth.

### Test 10 — `/analyze-ultralarge` command, real-world complexity: hour-scale deadline

**Scenario:** "We need this in 4 hours."

The command at lines 202–211 ("Handling Time Pressure") explicitly says: "This track is **not compatible with hour-scale deadlines.**" It then provides three scoped alternatives (Tier 1 partition only, one-subsystem-deep + rest-shallow, switch to `/analyze-codebase`). This is the same scoped-alternatives pattern as SKILL.md lines 296–307.

**Result:** Pass. Pack consistently refuses unbounded compression but always offers scoped alternatives — a marketplace-wide best-practice pattern.

### Summary of behavioral testing

| Test | Component | Category | Result |
|---|---|---|---|
| 1 | SKILL.md — time pressure → skip validation | Pressure (A) | Pass |
| 2 | `/analyze-ultralarge` — tier criteria not met | Edge case (C) | Pass |
| 3 | `module-reviewer` — large-module sampling | Edge case (C) | Pass |
| 4 | `module-reviewer` — cross-focus discipline | Pressure (A) | Pass |
| 5 | `subsystem-scribe` — malformed partial | Edge case (C) | Pass |
| 6 | SKILL.md — prior partial work | Real-world (B) | Mixed (M1) |
| 7 | `analysis-validator` — pressure resistance | Pressure (A) | Pass |
| 8 | `subsystem-scribe` — read-source temptation | Pressure (A) | Pass |
| 9 | `module-reviewer` — overconfidence | Pressure (A) | Pass |
| 10 | `/analyze-ultralarge` — hour-scale deadline | Real-world (B) | Pass |

9 of 10 Pass; one Mixed result feeds Minor finding M1. The pressure-resistance score (6/6) is unusually strong; the edge-case score (3/3) is unusually strong; the real-world score (1/2 Pass + 1 Mixed) is where the only finding surfaces, and the gap is in router-vs-wrapper synchronization rather than in the underlying methodology.

### Additional spot-checks against reference sheets

**Spot-check A — `validating-architecture-analysis.md` independence clause (lines 22–31).** The validator sheet carries the same "NON-NEGOTIABLE Validation Independence" block as the `analysis-validator` agent body — appropriately, since the agent loads this sheet for methodology. The two are not duplicates; the agent body has the activation contract (when to run, how to scope), the sheet has the per-document-type checklists. Clean separation.

**Spot-check B — `analyzing-unknown-codebases.md` output contract (lines 37–60).** The contract specifies EXACTLY 8 required fields per subsystem entry. The contract explicitly forbids extra fields: "Common rationalization: 'I'll add helpful extra sections to improve clarity.' Reality: Extra sections break downstream tools." (lines 33–35). This contract is what the validator agent + ultralarge synthesis pass both target — single source of truth.

**Spot-check C — Step 6 per-module loop control-flow (`module-by-module-with-scribe.md` lines 11–14).** "DISPATCH → COLLECT → MERGE → VALIDATE → CHECKPOINT → ADVANCE" — six-state machine, exact same six states named in `/analyze-ultralarge` Step 6b (line 112 of the command file). Naming consistency across command-sheet boundary is good evidence of careful editing.

**Spot-check D — Cross-reference integrity in router SKILL.md lines 414–430.** All 16 documentation contract links resolve to real files in `skills/using-system-archaeologist/`. No dead links. The ultralarge-tier additions at lines 428–430 (`partitioning-ultralarge-repos.md`, `module-by-module-with-scribe.md`, `findings-schema.md`) are present alongside the legacy sheets — clean v1.6.0 additive change with no removals.

**Spot-check E — Marketplace registration (`/home/john/skillpacks/.claude-plugin/marketplace.json`).** Entry present, `source` points to `./plugins/axiom-system-archaeologist`, category implicit (or absent — not checked in detail). Description compresses the plugin.json description but is not contradictory.

## 5. Findings

### Critical (0)

None.

### Major (0)

None. The previously-suspected "missing slash-command wrapper" risk is not realized: `/home/john/skillpacks/.claude/commands/system-archaeologist.md` exists and is internally consistent with the router skill.

### Minor (2)

**M1 — Router SKILL.md does not cover the "resume prior analysis" case that the wrapper does.**

- Evidence: `/home/john/skillpacks/.claude/commands/system-archaeologist.md` lines 154–177 carry a detailed "Handling Sunk Cost (Incomplete Prior Work)" section with a checklist (find existing workspace, read coordination log, assess quality, make continue/archive/salvage decision). The router skill at `plugins/axiom-system-archaeologist/skills/using-system-archaeologist/SKILL.md` does not include this section — the closest material is the discussion of incremental analysis routed via `incremental-analysis.md` at line 421.
- Impact: A user whose skill auto-invokes via description match (rather than the explicit `/system-archaeologist` slash command) does not see the resume protocol. Auto-invocation of the router skill is the primary discovery path for the pack.
- Fix (proposed, not applied): Either add a brief "Resuming Prior Work" subsection to SKILL.md citing `incremental-analysis.md`, OR move the wrapper's sunk-cost section into SKILL.md and have the wrapper link to it. (Wrapper-vs-skill content drift is a recurring theme.)

**M2 — Wrapper-vs-skill content divergence is broader than just M1.**

- Evidence: Comparing `/home/john/skillpacks/.claude/commands/system-archaeologist.md` (the wrapper) against `plugins/axiom-system-archaeologist/skills/using-system-archaeologist/SKILL.md` (the router) shows non-trivial divergence:
  - The wrapper does NOT mention Step 1.5 (deliverable menu) or `deliverable-options.md`.
  - The wrapper does NOT mention the ultralarge tier or `/analyze-ultralarge`.
  - The wrapper's "Validation Gates" section is slightly older — it presents the Validation Subagent vs. Self-Validation as a choice (lines 113–130) rather than as the strict-conjunction gate the router now uses (router SKILL.md lines 216–228 require ALL four conditions for self-validation).
- Impact: When a user invokes `/system-archaeologist`, they get older guidance than when the router skill auto-invokes. The wrapper appears to be the original v1.0-era skill body that was not re-synchronized after v1.6.0/1.6.1 hardening.
- Fix (proposed, not applied): Either (a) shrink the wrapper to a thin redirect ("This command surfaces the `using-system-archaeologist` skill — load that skill"), per the `python-engineering.md` pattern referenced in the maintenance sheet, or (b) re-sync the wrapper to match the current router skill. Option (a) is the lower-maintenance long-term answer.

### Polish (5)

**P1 — `partitioning-ultralarge-repos.md` references "10 minutes here saves dispatching 50 reviewer agents" at line 53 while `/analyze-ultralarge` Resource Reality section cites "~152 agent invocations per subsystem" (line 167) and "~1,800 invocations" for 12 subsystems (line 168).** The numbers come from different parts of the same workflow — partition-time vs. per-module-time — but the 10x discrepancy could read as inconsistency to a careful reader. Consider adding a one-sentence note on the partition sheet that the 50-agent figure refers to mis-partitioned reviewer dispatch at single-pass scale, not the per-module total.

**P2 — `module-reviewer.md` "Why This Agent Is Parameterized, Not Specialized" section (lines 130–137)** is a thoughtful explanation but is the only agent file in the pack to include design-rationale prose. Other agents (`subsystem-scribe.md` has a similar "Why MIN, Not Max Or Average" at lines 163–167; `codebase-explorer.md` and `analysis-validator.md` do not) are uneven on this. Consider standardizing: either add a "Design Rationale" trailer to all four agents, or remove it from the two that have one (lean toward keeping — these are genuinely useful for future maintainers).

**P3 — SKILL.md line 75 references "subsystem-discovery.md" as a path-disambiguation example (`Reference sheets like subsystem-discovery.md are at: skills/using-system-archaeologist/subsystem-discovery.md`).** No file named `subsystem-discovery.md` exists in the pack (the analogous sheet is `analyzing-unknown-codebases.md`). This is a stale documentation example from an earlier naming scheme. Replace the example with a real filename, e.g. `analyzing-unknown-codebases.md` or `findings-schema.md`.

**P4 — Pack description in `plugin.json` says "5 commands, 4 agents" twice (once in description, once implied by the inventory).** Marketplace description is verbatim from plugin.json. The "ultralarge-tier track... for codebases exceeding 100K LOC, >12 subsystems, plugin-registry architectures, or oversized test/doc corpora" tail of the plugin.json description (line 4 of plugin.json) is a useful trigger phrase, but the marketplace.json description (line 3 of marketplace entry) compresses it differently. The two are not contradictory but they are not identical. Minor; brand consistency only.

**P5 — The "Why This Schema, Not A Different One" section in `findings-schema.md` (lines 461–469) is one of the strongest design-rationale sections in the pack** — it justifies five specific schema choices (YAML over JSON, separate partials per focus, per-focus confidence, provenance block, no free-text "notes" field) against alternatives. Consider promoting this kind of explicit-rationale prose to a pack-wide convention. Other reference sheets would benefit from a similar trailer (e.g., `partitioning-ultralarge-repos.md` justifies tier thresholds but does not justify *why these thresholds and not others*; `analyzing-unknown-codebases.md` justifies the 8-field contract but not why 8 vs. 5 or 12). Net positive observation; no required action.

### Why no Critical or Major findings?

A short justification, because reviewing this pack and emerging with zero Critical/Major can read as light-touch:

- **Critical-tier triggers** per `reviewing-pack-structure.md` lines 17–22: "Missing core foundational concepts (>50% of coverage map gaps) / Multiple components broken or contradictory / Router completely inaccurate / Component types misaligned." None of these apply: coverage map has 0 gaps, no components contradict, router is accurate, components are correctly typed.
- **Major-tier triggers** per `reviewing-pack-structure.md` lines 31–37: "Important gaps in coverage (20-50% missing) / Multiple duplicate components / Obsolete components / Wrong scope boundaries / Hooks with incorrect event types." Coverage gaps are 0%, not 20–50%. No duplicates. No obsolete components. Scope boundaries are correct and tied to cross-pack handoffs in `specialist-integration.md`. No hooks in this pack at all.
- The previously-suspected Major-tier finding (missing slash-command wrapper, per task brief) is **not** realized — the wrapper exists at `.claude/commands/system-archaeologist.md`.

The pack's apparent maturity is consistent with the memory note that v1.6.0 (`c99d6e8`) → v1.6.1 (`d7ed663`) was an empirically-driven hardening pass (the silent YAML-self-check failure mode), and that this pack has been reviewed/refined more times than most.

## 6. Recommended Actions

In priority order, **no edits applied** per task brief — these are recommendations only:

1. **Reconcile wrapper-vs-router drift (M2).** Recommended approach: convert `.claude/commands/system-archaeologist.md` into a thin redirect that loads the router skill (mirroring the `/python-engineering` pattern documented at SKILL.md line 226 of the maintenance pack). Yields a single source of truth and eliminates the v1.0-vs-v1.6.1 divergence permanently. Patch bump (1.6.1 → 1.6.2) is sufficient.
2. **Add a "Resuming Prior Work" subsection to the router SKILL.md (M1).** If recommendation 1 is taken, the sunk-cost content from the wrapper is the obvious thing to lift into the router skill — kill two birds with one edit.
3. **Fix the stale `subsystem-discovery.md` example in SKILL.md line 75 (P3).** One-line edit. Patch bump.
4. **Standardize Design Rationale sections across agents (P2).** Either add to `codebase-explorer.md` and `analysis-validator.md`, or accept the inconsistency. Recommend adding — these sections explain non-obvious design choices that will be the first thing future maintainers question.
5. **Add a partition-vs-loop dispatch-count clarification (P1).** One-sentence edit to `partitioning-ultralarge-repos.md` to head off the 50-vs-1,800 number confusion.

**No new components needed.** No structural rewrites needed. No version-bump driver beyond patch (1.6.1 → 1.6.2) for items 1–5 combined.

### Sequencing of recommended fixes (one possible patch series)

If a maintainer were to land items 1–5 in a single patch bump:

1. **First commit:** Convert `.claude/commands/system-archaeologist.md` to a thin redirect; lift the sunk-cost content into `using-system-archaeologist/SKILL.md` as a new subsection (closes M1 + M2 simultaneously).
2. **Second commit:** Fix `subsystem-discovery.md` example reference in SKILL.md line 75 (P3); replace with `findings-schema.md` or `analyzing-unknown-codebases.md`.
3. **Third commit:** Add a one-sentence partition-vs-loop dispatch-count clarification to `partitioning-ultralarge-repos.md` line 53 area (P1).
4. **Fourth commit (optional):** Add a "Design Rationale" trailer to `codebase-explorer.md` and `analysis-validator.md` matching the style of `module-reviewer.md` and `subsystem-scribe.md` (P2). This is the one item where the cost-benefit is least clear; it's polish.
5. **Bump:** `version` 1.6.1 → 1.6.2 in `plugin.json`; consider also updating the marketplace.json description if items 1–4 materially change the user-visible surface (they don't — only item 1 is user-visible, and only as a UX improvement on `/system-archaeologist` consistency).

Total estimated editing time: ~1 hour for items 1–3, +30 minutes for item 4 if taken.

## 7. Reviewer Notes

**Confidence assessment:** High on inventory and structural review (every file enumerated and frontmatter inspected). Medium-high on behavioral testing — tests were read-only inspection against the gauntlet categories, not live subagent dispatch. The pack's design is sufficiently self-consistent that I am confident it would pass live dispatch testing on the ultralarge track, but I have not run such tests.

**Risk assessment:** Low risk. The pack is mature (v1.6.1, hardened through documented calibration work). No critical or major findings. M1 and M2 are well-understood drift issues, not novel defects.

**Information gaps:**
- I did not exhaustively read `validate-analysis.md` (239 LOC) or six of the smaller reference sheets (deliverable-options, specialist-integration, language-framework-patterns, mapping-security-surface, analyzing-test-infrastructure, generating-architecture-diagrams). I sampled the first ~50 lines of three of them and confirmed they follow the same SME-pack conventions as the rest. A deeper read of each could surface additional Polish items but is unlikely to change the overall Pass verdict.
- I did not invoke any of the agents live. Behavioral testing was anticipatory pressure-clause inspection, not dispatch.
- I did not validate cross-references (every `[name.md](name.md)` link in every reference sheet). Spot-check of the router SKILL.md links at lines 414–430 found all 16 paths to real files.

**Caveats:**
- The "wrapper-vs-router drift" finding (M2) depends on a specific maintenance philosophy — that the wrapper should be thin and the router authoritative. The pack's CLAUDE.md and the maintenance reference sheet (SKILL.md lines 206–238) both support that philosophy, but the current state of `python-engineering.md`-style wrappers vs. this pack's full-body wrapper is itself inconsistent across the marketplace. A reviewer could legitimately argue the wrapper SHOULD be a full standalone document. I have flagged the drift; the architectural decision is not mine to make.
- I sampled the ultralarge slice as instructed and did not deeply review the single-pass parallel-dispatch path (`/analyze-codebase` + `codebase-explorer` agent body beyond the first 50 lines). Findings on the ultralarge path do not generalize to the small/large-tier path.
- The pack is unusually thorough relative to other packs reviewed in this marketplace. A reviewer entering this pack with a "every pack has problems" prior should resist confirmation-search behavior; I attempted to and the Findings section reflects what I actually found rather than a quota.

**Final verdict:** Pass. Five small recommendations totaling perhaps one hour of editing work. No blocking issues.

### Stage-by-stage notes

- **Stage 1 (Investigation):** No coverage gaps. All 16 reference sheets present and indexed by the router SKILL.md at lines 414–430. Inventory matches plugin.json's "5 commands, 4 agents" claim. Marketplace registration present and current.
- **Stage 2 (Structure Review):** Scorecard is Pass across all 8 dimensions. Two Minor findings (M1, M2) and five Polish observations (P1–P5). No Critical, no Major.
- **Stage 3 (Behavioral Testing):** 10 scenarios run across the gauntlet (6 pressure, 3 edge case, 2 real-world — one scenario was tested at multiple levels). 9 Pass, 1 Mixed. Mixed feeds M1.
- **Stage 4 (Discussion):** Findings presented as report (this document). Stage 5 (Execution) intentionally skipped per task brief.

### Notes for the next reviewer

If this pack is reviewed again at v1.6.2 or later:

- **Check whether M1+M2 was addressed by collapsing the wrapper to a thin redirect.** If yes, the wrapper at `.claude/commands/system-archaeologist.md` should be short (under ~30 lines, redirecting to the router skill).
- **Check whether P3 was fixed.** Search SKILL.md for `subsystem-discovery.md`; it should no longer appear (the example should be a real filename).
- **Re-run behavioral tests 6 and 10.** Test 6 (resume prior partial work) is the test that flagged M1; it should Pass on a re-review if M1 was addressed by lifting the wrapper's sunk-cost content into the router skill. Test 10 (hour-scale deadline) is unlikely to change but worth re-running as a sanity check that the scoped-alternatives pattern is still verbatim-pressure-resistant.
- **Consider expanding sampling to the single-pass path on the next review.** This review focused on the ultralarge slice per the task brief; the `/analyze-codebase` + parallel-dispatch path was only spot-checked. A future review could do the opposite slice for completeness.

### Methodology notes

- The "sample a representative slice" guidance was interpreted as: read the ultralarge command flow end-to-end (`/analyze-ultralarge` + `partitioning-ultralarge-repos.md` + `module-by-module-with-scribe.md` + `findings-schema.md` + both reviewer/scribe agents) plus spot-check the single-pass path (`/analyze-codebase` + `codebase-explorer.md` + router SKILL.md). This is deeper than a minimum sample but lighter than full enumeration; the choice was driven by the ultralarge track being the v1.6.0/1.6.1 novelty and the most testable.
- Behavioral testing was inspection-driven (read the pressure clauses, simulate scenarios mentally, check whether the guidance is verbatim-pressure-resistant or vulnerable to a paraphrase). No live subagent dispatches were performed. This is sufficient for Stage 3 of the maintenance protocol per `testing-skill-quality.md` lines 80–92 ("Inline trial within the current session — lowest fidelity. Acceptable for a quick sanity check"), but a stricter Stage 3 would run subagent dispatches against the pressure scenarios and observe behavior empirically.
- The "Missing wrapper = Major" check from the task brief was the single highest-impact yes/no. The wrapper exists (`/home/john/skillpacks/.claude/commands/system-archaeologist.md`), so no Major lands there. The drift between wrapper and router (M2) is a separate, lesser issue.
