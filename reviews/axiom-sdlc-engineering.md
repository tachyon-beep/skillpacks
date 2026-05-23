# Review: axiom-sdlc-engineering

**Version:** 1.1.2 (`/home/john/skillpacks/plugins/axiom-sdlc-engineering/.claude-plugin/plugin.json:3`)
**Reviewed:** 2026-05-22
**Reviewer:** general-purpose subagent (Opus 4.7 1M)
**Rubric:** `meta-skillpack-maintenance:using-skillpack-maintenance` Stages 1–4

---

## 1. Inventory

### Plugin Metadata
- **Path:** `/home/john/skillpacks/plugins/axiom-sdlc-engineering/`
- **Marketplace registration:** Present (`.claude-plugin/marketplace.json:~lookup`, category not visible in snippet but `source: ./plugins/axiom-sdlc-engineering` resolves correctly).
- **Description claim:** "8 skills, 4 agents" — matches counts below (7 specialist + 1 router = 8 skills; 4 agents).

### Skills (8)

| # | Skill | Path | Frontmatter `name`/`description` | Body lines |
|---|-------|------|----------------------------------|------------|
| 1 | `using-sdlc-engineering` (router) | `skills/using-sdlc-engineering/SKILL.md` | OK; "Use when users request SDLC guidance…" | 232 |
| 2 | `requirements-lifecycle` | `skills/requirements-lifecycle/SKILL.md` | OK; "Use when defining requirements…" | 363 |
| 3 | `design-and-build` | `skills/design-and-build/SKILL.md` | OK; "Use when making architecture decisions…" | 248 |
| 4 | `quality-assurance` | `skills/quality-assurance/SKILL.md` | OK; "Use when deciding test strategy…" | 462 |
| 5 | `governance-and-risk` | `skills/governance-and-risk/SKILL.md` | OK; "Use when making architectural decisions without documentation…" | 428 |
| 6 | `quantitative-management` | `skills/quantitative-management/SKILL.md` | OK; "Use when establishing measurement programs…" | 257 |
| 7 | `platform-integration` | `skills/platform-integration/SKILL.md` | OK; "Use when implementing CMMI processes in GitHub or Azure DevOps…" | 340 |
| 8 | `lifecycle-adoption` | `skills/lifecycle-adoption/SKILL.md` | OK; "Use when starting new projects with CMMI…" | 744 |

### Reference Sheets (37 across 7 specialists)

Each specialist directory hosts 4–11 sibling `.md` reference sheets that the parent `SKILL.md` loads on demand. Sampled and present per the router's reference table (e.g. `quality-assurance/testing-practices.md`, `peer-reviews.md`, `validation-with-stakeholders.md`, `defect-management.md`, `qa-metrics.md`, `level-scaling.md`).

**Every specialist except `using-sdlc-engineering` and `platform-integration` ships its own `level-scaling.md` sheet.** Sizes:
- `requirements-lifecycle/level-scaling.md` — 793 lines
- `design-and-build/level-scaling.md` — 500 lines
- `quality-assurance/level-scaling.md` — 349 lines
- `quantitative-management/level-scaling.md` — 350 lines
- `governance-and-risk/level-scaling.md` — 302 lines

These five files cover the same CMMI L2→L3→L4 scaling axis from five different process-area angles. Whether they are *duplicates* or *correctly specialised* is assessed in §2 and §3.

### Commands (1 plugin-local; 1 repo-root wrapper)

| Location | Path | Frontmatter |
|----------|------|-------------|
| Plugin-local | `commands/using-sdlc-engineering/COMMAND.md` | **None — no `---` block, no `description`, no `allowed-tools`, no `argument-hint`.** Body is a how-to-use document. |
| Repo-root wrapper | `/home/john/skillpacks/.claude/commands/sdlc-engineering.md` | **No frontmatter either.** Body is a near-verbatim copy of the router SKILL.md content (lines 1–228) without the `---` block. |

**Sibling-pack convention** (e.g. `plugins/axiom-python-engineering/commands/profile.md`) places each command as a single `.md` directly under `commands/` with full YAML frontmatter: `description`, `allowed-tools: ["Read","Bash",...]`, `argument-hint`. This pack departs from that convention on both files.

### Agents (4)

| Agent | Path | `description` ends with "Follows SME Agent Protocol…"? | `model` | `tools` declared | Body cites `meta-sme-protocol`? | Body requires 4 SME sections? |
|-------|------|-----|---------|-----|-----|-----|
| `sdlc-advisor` | `agents/sdlc-advisor.md:2` | Yes (line 2) | sonnet | No (good — inherits) | Yes (line 10) | Yes (Output Format §, line 292) |
| `architecture-decision-reviewer` | `agents/architecture-decision-reviewer.md:2` | Yes (line 2) | opus | No | Yes (line 10) | Yes (Output Format §, line 285) |
| `quality-assurance-analyst` | `agents/quality-assurance-analyst.md:2` | Yes (line 2) | opus | No | Yes (line 10) | Yes (Output Format §, line 461) |
| `bug-triage-specialist` | `agents/bug-triage-specialist.md:2` | Yes (line 2) | opus | No | Yes (line 10) | Yes (Output Format §, line 443) |

All four agents are SME-style reviewers and **all four** comply with `meta-sme-protocol:sme-agent-protocol`. Model selection is appropriate (router on sonnet, deep-reasoning specialists on opus). No agent declares `tools:` — they inherit the parent context as the marketplace convention recommends.

### Hooks

None. (No `hooks/` directory; not required for this domain.)

### Slash-Command Wrapper

`/home/john/skillpacks/.claude/commands/sdlc-engineering.md` **exists** — pack passes the "missing wrapper = Major" check from the rubric.

### Leaked Build Artifacts (in plugin distribution)

The following are checked into the plugin directory and will ship to users:

- `plugins/axiom-sdlc-engineering/IMPLEMENTATION_PROGRESS.md`
- `plugins/axiom-sdlc-engineering/SECTION3_COMPLETE.md`
- `plugins/axiom-sdlc-engineering/WEEK1_PROGRESS.md`
- `plugins/axiom-sdlc-engineering/.final-readiness-report.md`
- `plugins/axiom-sdlc-engineering/.shakedown-report.md`
- `plugins/axiom-sdlc-engineering/.improvements-applied.md`
- `plugins/axiom-sdlc-engineering/.final-verification-report.md`
- `plugins/axiom-sdlc-engineering/.agents-summary.md`
- `plugins/axiom-sdlc-engineering/.test-scenarios/` (29 RED/GREEN/REFACTOR artefacts)
- `plugins/axiom-sdlc-engineering/skills/lifecycle-adoption/SKILL.md.backup-2026-01-24` (132 KB)
- `plugins/axiom-sdlc-engineering/skills/lifecycle-adoption/SKILL.md.backup-extracted` (120 KB)

The dot-prefixed files and `.test-scenarios/` are hidden from typical `ls`, but they are tracked in git and present in the plugin directory. The two `SKILL.md.backup-*` files are 252 KB combined and live inside a *skills* subdirectory — the runtime skill loader will not parse them (frontmatter scan typically targets `SKILL.md` exactly), but they bloat the install, confuse maintainers, and violate the user's documented "never leak project-specific data into distributable skillpacks" preference (see MEMORY index).

---

## 2. Domain & Coverage

### Intended Domain (inferred from router and plugin.json)

CMMI-based SDLC framework spanning **Levels 2–4** of the seven CMMI Maturity Model process areas the pack actually addresses:

| CMMI Process Area | Covered by Skill |
|-------------------|------------------|
| RD + REQM (Requirements Development + Management) | `requirements-lifecycle` |
| TS + CM + PI (Technical Solution + Configuration Mgmt + Product Integration) | `design-and-build` |
| VER + VAL (Verification + Validation) | `quality-assurance` |
| DAR + RSKM (Decision Analysis + Risk Mgmt) | `governance-and-risk` |
| MA + QPM + OPP (Measurement & Analysis + Quantitative Project Mgmt + Org Process Performance) | `quantitative-management` |
| GitHub / Azure DevOps platform mapping | `platform-integration` |
| Adoption & retrofitting on existing projects | `lifecycle-adoption` |

### Coverage Map

**Foundational** (CMMI Level 2 baseline):
- Requirements elicitation, traceability — **Covered** (`requirements-lifecycle`, plus 6 sibling sheets)
- Configuration management, branching — **Covered** (`design-and-build/configuration-management.md`)
- Peer review, basic testing — **Covered** (`quality-assurance/peer-reviews.md`, `testing-practices.md`)
- Basic risk identification — **Covered** (`governance-and-risk/rskm-methodology.md`)

**Core** (Level 3 organisational standards):
- ADR process, alternatives analysis — **Covered** (`design-and-build/architecture-and-design.md`, `governance-and-risk/dar-methodology.md`)
- UAT, validation with stakeholders — **Covered** (`quality-assurance/validation-with-stakeholders.md`)
- Defect management and RCA — **Covered** (`quality-assurance/defect-management.md` + `bug-triage-specialist` agent)
- Measurement program, GQM, DORA — **Covered** (`quantitative-management/measurement-planning.md`, `dora-metrics.md`)
- Platform-specific implementations (GitHub + Azure DevOps) — **Covered** (10 sibling sheets under `platform-integration/`)

**Advanced** (Level 4 quantitative):
- Statistical process control, control charts — **Covered** (`quantitative-management/statistical-analysis.md`)
- Process baselines, Cp/Cpk — **Covered** (`quantitative-management/process-baselines.md`)
- Predictive models for quality — Surface-level coverage in `bug-triage-specialist.md:298` (defect prediction mentioned in metrics, but no dedicated reference sheet).

**Cross-cutting**:
- Adoption strategy, retrofitting — **Covered** (`lifecycle-adoption` — 744-line SKILL.md plus 8 sibling sheets).
- Change management & team resistance — **Covered** (`lifecycle-adoption/change-management.md`, 16 KB).
- Audit preparation — **Mentioned** across multiple skills (audit-trail sheets under `platform-integration/`), no dedicated cross-cutting "audit-prep" skill.

### Domain Stability

CMMI is a **stable** framework (CMMI v2.0 published 2018, v3.0 in 2023 — minor refinements). The platform-integration content (GitHub Actions, Azure Pipelines) is *moderately evolving* — sampled snippets reference Issues/PRs/Actions/Boards which are stable; CODEOWNERS, branch protection, and OData analytics are current. **No urgent currency concerns** for the snapshot reviewed.

### Gaps

None of the gaps below are foundational; all are minor coverage refinements:

- **Audit-preparation workflow** is fragmented across `lifecycle-adoption`, `platform-integration/*-audit-trail.md`, and `requirements-lifecycle`. Could be consolidated, but not blocking.
- **Software security / supply chain** (SCA, SBOM, dependency vuln management) is out of scope by design (handed off to `ordis-security-architect` per router cross-pack table). Document that explicitly in router's "Do NOT use for" if not already (it is — see `using-sdlc-engineering/SKILL.md:21`).
- **Continuous deployment / progressive delivery patterns** (feature flags appear briefly in `quality-assurance/SKILL.md:218–224` only). The pack defers CI/CD pipeline architecture to `axiom-devops-engineering`. Cross-link not present in router's cross-pack table — minor.

---

## 3. Fitness Scorecard

| Dimension | Rating | Evidence |
|-----------|--------|----------|
| **Router quality** | **Pass** | `using-sdlc-engineering/SKILL.md` has trigger description, decision tree (lines 47–80), routing table (84–98), cross-pack coordination table (104–113), rationalisation counters (140–148), red flags (150–162), multi-skill roadmaps (166–182). Very thorough — among the strongest routers in the marketplace. |
| **Skill descriptions** | **Pass** | All 8 SKILL.md files use the `Use when …` repo convention. Each description names concrete triggers (e.g. `quality-assurance` lists "deciding test strategy, struggling with code reviews, shipping without tests, or conflating verification with validation"). |
| **Frontmatter conformance (skills)** | **Pass** | All 8 SKILL.md headers parse cleanly: `name:` + `description:` only; no spurious `allowed-tools` on skills (matches marketplace norm). |
| **Frontmatter conformance (commands)** | **Critical** | `commands/using-sdlc-engineering/COMMAND.md` has **no YAML frontmatter at all** — no `description`, no `allowed-tools`, no `argument-hint`. The repo-root wrapper `/home/john/skillpacks/.claude/commands/sdlc-engineering.md` is **also frontmatterless**. Compare to `plugins/axiom-python-engineering/commands/profile.md:1-5`. Without `description:`, the slash-command will not appear correctly in `/help` and tool-restriction is wide open. |
| **Component cohesion** | **Major** | Five sibling `level-scaling.md` files (across `requirements-lifecycle/`, `design-and-build/`, `quality-assurance/`, `quantitative-management/`, `governance-and-risk/`) totalling 2,294 lines repeat the L2 / L3 / L4 framing. Each is *contextualised* to its process area, so they aren't blind duplicates — but the framing scaffolding (level definitions, overhead percentages, escalation criteria) is restated five times. Refactor opportunity, not a defect. |
| **Slash-command exposure** | **Major** | Wrapper exists at the right path (`.claude/commands/sdlc-engineering.md`) — that's good — but it is a 228-line *copy* of the router SKILL.md content without a `---` frontmatter block. The marketplace convention (per the rubric's example `python-engineering.md`) is a thin wrapper, not a content duplication. The pack also names its plugin-local command directory `using-sdlc-engineering/COMMAND.md` rather than the conventional `commands/<verb>.md` flat layout used by every other Axiom pack sampled. |
| **SME agent protocol** | **Pass** | All 4 agents declare "Follows SME Agent Protocol with confidence/risk assessment." in their `description`, cite `meta-sme-protocol:sme-agent-protocol` in the body (`agents/*.md:10`), and require the four canonical sections (Confidence Assessment, Risk Assessment, Information Gaps, Caveats) in their Output Format. Best-in-class compliance. |
| **Anti-pattern coverage** | **Pass** | Each skill has an explicit anti-pattern section. Examples: `quality-assurance/SKILL.md:237–317` (Test Last, Rubber Stamp, Ice Cream Cone, Whack-a-Mole, Validation Theater); `design-and-build/SKILL.md` references resume-driven design and BDUF; `governance-and-risk/SKILL.md` covers groupthink, authority deferral, sunk-cost reasoning. Anti-patterns are *behaviourally framed* (detection cues + counter-responses), not just lists. |
| **Cross-skill linkage** | **Pass** | Router has explicit cross-pack table (`using-sdlc-engineering/SKILL.md:104–113`); `quality-assurance/SKILL.md:422–432` ("Integration with Other Skills") cross-links to `axiom-python-engineering`, `ordis-quality-engineering`, etc. Agents cross-route to one another (`sdlc-advisor.md:330–345`). |
| **Distribution hygiene** | **Major** | 3 root-level progress markdowns (`WEEK1_PROGRESS.md`, `SECTION3_COMPLETE.md`, `IMPLEMENTATION_PROGRESS.md`) + 5 dot-prefixed status reports + `.test-scenarios/` (29 files) + two `SKILL.md.backup-*` files (252 KB combined inside a skill directory) all ship with the plugin. Violates user's "no project-specific leaks" preference per MEMORY index. |

**Overall:** **Major** — core component quality is high (router + skills + agents are strong), but distribution hygiene and command-frontmatter conformance are degraded. No Critical defect that makes the plugin unusable; multiple Major issues that reduce polish and conformance.

---

## 4. Behavioral Tests

Three pressure-tests on the router and two specialists; one targeted test on the slash-command wrapper. Because this is a report-only review, tests are *desk-checked* — i.e. read the skill content and ask "what would Claude do under this prompt?" using verbatim skill text.

### Test 1: Router under simplicity-temptation pressure

**Scenario:** "I'm a solo dev on a 3-week prototype. I just want to know what testing to do. Don't over-engineer this."

**Skill consulted:** `using-sdlc-engineering/SKILL.md`

**Expected guidance:** Detect Level 2 from context clue ("solo dev", "prototype"), route to `quality-assurance` with Level 2 framing.

**Observed (from skill text):**
- CMMI level detection lines 28–38 explicitly handle this: `"startup", "small team", "2-3 developers" → Level 2 (lightweight)`. ✓
- Rationalisation counters at line 143 (`"We're too small for CMMI"`) preempt the pushback. ✓
- Routing table line 91 sends "testing strategy" → `quality-assurance`. ✓
- Multi-skill roadmaps section is *not* invoked because this is a single concern. ✓

**Result:** **Pass.** The router holds under simplicity pressure and routes correctly with appropriate level scaling.

### Test 2: `quality-assurance` under "skip-tests" pressure

**Scenario:** "We need to ship Friday. Skip the tests — we'll write them next sprint. We've done this before, it's fine."

**Skill consulted:** `quality-assurance/SKILL.md`

**Expected guidance:** Refuse blanket skip; offer TEST-HOTFIX exception protocol with 48-hour retrospective testing requirement; warn that "later" never comes.

**Observed:**
- "Exception Protocol: Shipping Without Tests" section (lines 186–233) directly addresses this. ✓
- Line 188 calls out the rationalisation by name: `"Tests later" = tests never (documented historical pattern)`. ✓
- Lines 191–198 enumerate **non-negotiable cases** (security, compliance, payment, data migration). ✓
- TEST-HOTFIX requires 48-hour retrospective testing (line 207). ✓
- Frequency limit of >5/month = systemic problem (line 211). ✓
- Anti-pattern "Test Last" section (241–251) reinforces. ✓

**Result:** **Pass.** Strong resistance to the most common pressure case in this domain.

### Test 3: `governance-and-risk` under authority-deferral pressure

**Scenario:** "The CTO already chose Kubernetes. We just need to write up the ADR. Don't make this complicated."

**Skill consulted:** `governance-and-risk/SKILL.md`

**Expected guidance:** Reject authority-without-analysis pattern; require alternatives analysis *before* authority input (not after); document the alternatives even when the choice seems pre-determined.

**Observed:**
- Lines 28–33 list "Deferring to authority without independent analysis (CTO says, tech lead suggests)" as a use-when trigger. ✓
- Quick Reference row at line 47 explicitly handles this: `"Authority wants specific option | DAR with independent analysis | Level 3+ | Analyze alternatives BEFORE authority input"`. ✓
- The `architecture-decision-reviewer` agent (`agents/architecture-decision-reviewer.md:99–116`) lists "Appeal to popularity" and "Buzzword bingo" (Kubernetes-without-context) as resume-driven design red flags requiring REJECTION. ✓

**Result:** **Pass.** Skill + specialist agent together cover the scenario robustly.

### Test 4: Slash-command wrapper activation

**Scenario:** User types `/sdlc-engineering` in a fresh session.

**Expected behaviour:** The wrapper file at `.claude/commands/sdlc-engineering.md` is loaded as a command and instructs Claude on how to act — typically by routing to the router skill.

**Observed:**
- The file exists at the expected path. ✓
- The file has **no YAML frontmatter** — no `description:`, no `allowed-tools:`, no `argument-hint:`. The command will technically still resolve (Claude Code tolerates frontmatterless commands) but:
  - It will not have a `description` for `/help` listings (degrades discoverability).
  - There is no `allowed-tools` restriction, so the command runs with full inherited tool access (this is *probably* fine for a router, but it's not auditable).
- The body is a 228-line content copy of `using-sdlc-engineering/SKILL.md` rather than a thin wrapper instructing "load the router skill via the Skill tool". Compare to the marketplace convention (e.g. `python-engineering.md`, per the rubric example).

**Result:** **Fix needed** — wrapper is structurally present (so not "missing"), but does not conform to the marketplace's frontmatter+thin-wrapper convention.

### Test 5: `bug-triage-specialist` agent under "just close it" pressure

**Scenario:** "This is the third time this bug came back. Just close it and move on, we don't have time."

**Skill consulted:** `agents/bug-triage-specialist.md`

**Expected behaviour:** Block closure; flag systemic issue (>3 similar = systemic per line 258); require RCA per Level 3; demand process-level prevention not just code-level fix.

**Observed:**
- Pattern threshold (`>3 similar bugs = systemic issue`, line 258) ✓
- "SYSTEMIC ISSUE DETECTED" template (lines 262–278) with required actions ✓
- "Defect Whack-a-Mole" anti-pattern section (lines 339–360) — `"RCA mandatory for ALL recurring defects (Level 3 requirement)"`. ✓
- DO NOT CLOSE criteria (lines 243–247): "RCA required but not completed" blocks closure. ✓
- Required output explicitly includes `Information Gaps` and `Caveats` per SME protocol. ✓

**Result:** **Pass.** Agent holds under pressure and applies CMMI L3 rigor consistently.

### Summary of behavioural tests

| Component | Test | Result |
|-----------|------|--------|
| `using-sdlc-engineering` router | Simplicity pressure | Pass |
| `quality-assurance` | Test-skip pressure | Pass |
| `governance-and-risk` | Authority-deferral pressure | Pass |
| Slash-command wrapper | Activation/frontmatter | **Fix needed** (frontmatter absent, body duplicates router) |
| `bug-triage-specialist` agent | "Just close it" pressure | Pass |

---

## 5. Findings

### Critical

**C1. Plugin-local command file has no YAML frontmatter.**
- **Path:** `/home/john/skillpacks/plugins/axiom-sdlc-engineering/commands/using-sdlc-engineering/COMMAND.md`
- **Evidence:** File starts with `# SDLC Engineering Command` on line 1 — no `---` block.
- **Why critical:** Every other Axiom/Yzmir/Lyra/Muna/Ordis pack sampled places commands at `commands/<verb>.md` with frontmatter (`description`, `allowed-tools`, `argument-hint`). The current layout (`commands/using-sdlc-engineering/COMMAND.md`) is non-conformant and the command will not be discoverable via the standard `/help` mechanism or restrictable by tools.
- **Fix:** Move to `commands/sdlc-engineering.md` (or `using-sdlc-engineering.md`) and add frontmatter matching the marketplace pattern. See `plugins/axiom-python-engineering/commands/profile.md:1-5` for the template.

### Major

**M1. Repo-root slash-command wrapper is frontmatterless and duplicates router content.**
- **Path:** `/home/john/skillpacks/.claude/commands/sdlc-engineering.md`
- **Evidence:** Line 1 is `# SDLC Engineering Router` — no `---`. The body (228 lines) is a near-verbatim copy of `using-sdlc-engineering/SKILL.md` rather than a thin wrapper.
- **Fix:** Add YAML frontmatter (`description`, `allowed-tools: ["Skill", "Read"]` at minimum). Reduce body to a thin "When to use → invoke `using-sdlc-engineering` skill" wrapper. Keeps router content in one place (the SKILL.md).

**M2. Distribution leaks — implementation/test artefacts ship with the plugin.**
- **Paths:**
  - Root-level: `WEEK1_PROGRESS.md`, `SECTION3_COMPLETE.md`, `IMPLEMENTATION_PROGRESS.md`
  - Hidden but tracked: `.final-readiness-report.md`, `.shakedown-report.md`, `.improvements-applied.md`, `.final-verification-report.md`, `.agents-summary.md`
  - Test scaffolding: `.test-scenarios/` (29 RED/GREEN/REFACTOR comparison files)
  - Backup files inside skill directory: `skills/lifecycle-adoption/SKILL.md.backup-2026-01-24` (132 KB), `SKILL.md.backup-extracted` (120 KB)
- **Why major:** User's documented preference (MEMORY: `feedback_no_project_leaks.md`) is to never leak project-specific data into distributable skillpacks. The two SKILL.md.backup-* files in particular are inside a `skills/` subdirectory where the skill loader walks — they will not be parsed (the loader expects exactly `SKILL.md`) but they bloat the install and may confuse some tooling.
- **Fix:** Add to `.gitignore` or delete from the plugin directory. Add the `.test-scenarios/`, `*PROGRESS.md`, `SECTION*_COMPLETE.md`, `.shakedown-*`, `.improvements-*`, `.final-*`, `.agents-summary.md`, and `**/*.backup*` patterns to a per-plugin or repo-level `.gitignore`.

**M3. Five sibling `level-scaling.md` reference sheets restate the same CMMI L2/3/4 scaffolding.**
- **Paths:** `requirements-lifecycle/level-scaling.md` (793 lines), `design-and-build/level-scaling.md` (500), `quality-assurance/level-scaling.md` (349), `quantitative-management/level-scaling.md` (350), `governance-and-risk/level-scaling.md` (302).
- **Why major (not critical):** Each *is* contextualised to its process area, so they aren't dead duplicates. But the L2/3/4 definition framing (team-size thresholds, overhead percentages, audit triggers) appears five times — extraction to a shared `using-sdlc-engineering/level-scaling.md` (or per-pack `axiom-sdlc-engineering/level-scaling-shared.md`) would reduce ~1,500 lines of redundancy and make consistency maintenance one-touch.
- **Fix:** Extract shared L2/L3/L4 framing once; each specialist's sheet retains only the process-area-specific scaling guidance and `→ See ../level-scaling-shared.md for level definitions`. Document the shared sheet's path in the router.

### Minor

**Mi1. Plugin command directory name is non-conformant.**
- **Path:** `commands/using-sdlc-engineering/COMMAND.md`
- **Other packs use:** `commands/<verb>.md` (flat). E.g. `axiom-python-engineering/commands/{profile,typecheck,delint,create-project-scaffold}.md`.
- **Fix:** Flatten to `commands/sdlc-engineering.md`.

**Mi2. Plugin description in `plugin.json` and `marketplace.json` says "8 skills, 4 agents".**
- This is **accurate** (1 router + 7 specialists = 8; 4 agents present). No fix needed, but note that the count is router-inclusive — some other packs report specialist-only counts. Consistency across the marketplace is a minor doc concern.

**Mi3. Router cross-pack table omits `axiom-devops-engineering`.**
- `using-sdlc-engineering/SKILL.md:104–113` lists handoffs to `axiom-python-engineering`, `lyra-ux-designer`, `ordis-security-architect`, `muna-technical-writer`, `ordis-quality-engineering` — but the marketplace also has `axiom-devops-engineering` (CI/CD pipeline architecture) which is a natural handoff target for `design-and-build` (build-and-integration topics).
- **Fix:** Add a row: "CI/CD pipeline implementation | `design-and-build` (process) | `axiom-devops-engineering` (pipeline architecture) | …".

**Mi4. Predictive-quality-model coverage is thin.**
- `bug-triage-specialist.md:298` and `quantitative-management` Level 4 sections mention defect prediction, but there's no dedicated reference sheet. Defensible at Level 4 (this is genuinely advanced material) but worth a small sheet or explicit "see external resources" pointer.

### Polish

**P1. `lifecycle-adoption/SKILL.md` at 744 lines is the largest router-adjacent skill in the pack.**
- Reads as a knowledge-dense, well-organised file — not a structural problem — but could benefit from a slightly tighter Quick Reference section near the top so users in a fresh session can answer "where do I start?" in <50 lines without scrolling.

**P2. `quality-assurance/SKILL.md` opens with a doc reference `See docs/sdlc-prescription-cmmi-levels-2-4.md` (line 18).**
- This `docs/` directory is **not present** in the plugin distribution. The reference appears in multiple skills (`requirements-lifecycle:22`, `design-and-build:14`, `governance-and-risk:18`, etc.). Either ship the prescription doc with the plugin or remove the dead reference. (Verified: `find plugins/axiom-sdlc-engineering -name "sdlc-prescription*"` returns nothing.)

---

## 6. Recommended Actions

Sequenced by impact / risk:

1. **Add YAML frontmatter to the slash-command wrapper** (`.claude/commands/sdlc-engineering.md`) and the plugin-local command file (`commands/using-sdlc-engineering/COMMAND.md`). Convert both to thin wrappers (~10–20 lines) that invoke `using-sdlc-engineering` via the Skill tool. *Addresses C1, M1, Mi1.*

2. **Remove distribution leaks.** Delete (or move out of the plugin tree) the eight progress/status `.md` files, the `.test-scenarios/` directory, and the two `SKILL.md.backup-*` files. Add patterns to `.gitignore`. *Addresses M2.*

3. **Resolve the dead `docs/sdlc-prescription-cmmi-levels-2-4.md` references.** Either ship that doc with the plugin (under `plugins/axiom-sdlc-engineering/docs/`) or remove the `See docs/...` lines from each SKILL.md. *Addresses P2.*

4. **Extract shared L2/L3/L4 framing** into a single reference sheet to remove ~1,500 lines of restated scaffolding. *Addresses M3.*

5. **Add `axiom-devops-engineering` handoff row to the router's cross-pack table.** *Addresses Mi3.*

6. **Optional / future:** Add a dedicated Level 4 predictive-quality-models reference sheet under `quantitative-management/`. *Addresses Mi4.*

7. **Optional / future:** Add a Quick Reference banner to `lifecycle-adoption/SKILL.md`. *Addresses P1.*

After actions 1–3, recommend a **patch bump** to v1.1.3. Actions 4–5 warrant a **minor bump** to v1.2.0.

---

## 7. Reviewer Notes

- The pack is **substantively strong**: router design, anti-pattern coverage, SME agent protocol compliance, and behavioural-pressure resistance are among the best of the maintenance reviews I've performed in this marketplace. The CMMI L2/3/4 scaling discipline is internally consistent across all seven specialists and four agents.
- All issues found are **structural and hygienic**, not semantic — the pack's *guidance* is sound; its *packaging* needs cleanup.
- The slash-command wrapper question deserves clarification with the user: the rubric's "missing wrapper = Major" check passes (a wrapper file exists at the right path) but the wrapper is content-bloated and frontmatterless. I treated this as Major (M1) rather than Critical because the wrapper does function — it just doesn't conform.
- Test methodology was desk-check (read the skill text and verify it covers the scenario) rather than live subagent dispatch, per the report-only constraint. A follow-up live subagent test against the three behavioural scenarios above would be a high-value confirmation step.
- I did not load `implementing-fixes.md` per the instruction to skip Stage 5.
- Citations are all absolute paths and line numbers within `/home/john/skillpacks/`.
