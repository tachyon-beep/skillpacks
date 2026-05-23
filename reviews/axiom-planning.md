# Review: axiom-planning

**Version:** 1.1.1 (`/home/john/skillpacks/plugins/axiom-planning/.claude-plugin/plugin.json:3`)
**Reviewed:** 2026-05-22
**Reviewer:** general-purpose subagent

---

## 1. Inventory

**Plugin metadata** (`/home/john/skillpacks/plugins/axiom-planning/.claude-plugin/plugin.json`):
- `name: axiom-planning`, `version: 1.1.1`, `license: CC-BY-SA-4.0`, `author: tachyon-beep`.
- Description (line 4): `"TDD-validated implementation planning with plan review quality gate (2 skills, 5 agents, 1 command) - write plans, validate against codebase reality before execution"`. Component count is accurate.
- README (`/home/john/skillpacks/plugins/axiom-planning/README.md:3`) still says `**Version:** 1.1.0` — minor drift from `plugin.json` (1.1.1). Patch-bump rationale not in version history (line 188–204 only lists 1.1.0 and 1.0.0).

**Marketplace registration** (`/home/john/skillpacks/.claude-plugin/marketplace.json`): registered. Catalog description: `"TDD-validated implementation planning with plan review quality gate (2 skills, 5 agents, 1 command) - write plans, validate against codebase reality with 4 parallel reviewers before execution"`.

**Skills (2 files, 471 lines total):**

| File | Lines | Role |
|------|-------|------|
| `skills/implementation-planning/SKILL.md` | 249 | Plan creation — atomic tasks, exact paths, complete code, anti-rationalization defenses |
| `skills/plan-review/SKILL.md` | 222 | Quality-gate skill describing 4-reviewer architecture invoked via `/review-plan` |

**Commands (1 file, 304 lines):**

| Command | File | argument-hint | Tools |
|---------|------|---------------|-------|
| `/review-plan` | `commands/review-plan.md:1-5` | `"[plan_file_path]"` | `["Read", "Grep", "Glob", "Bash", "Task", "Write", "AskUserQuestion"]` |

**Agents (5 files, 935 lines total):**

| Agent | File | Model | `tools:` declared? | Role |
|-------|------|-------|--------------------|------|
| `plan-review-reality` | `agents/plan-review-reality.md` | sonnet | YES — `["Read", "Grep", "Glob", "Bash"]` | Symbol existence, paths, versions, conventions |
| `plan-review-architecture` | `agents/plan-review-architecture.md` | sonnet | YES — `["Read", "Grep", "Glob", "Bash"]` | Blast radius, one-way doors, patterns |
| `plan-review-quality` | `agents/plan-review-quality.md` | sonnet | YES — `["Read", "Grep", "Glob", "Bash"]` | Test strategy, observability, security |
| `plan-review-systems` | `agents/plan-review-systems.md` | sonnet | YES — `["Read", "Grep", "Glob", "Bash"]` | Second-order effects, feedback loops, failure modes |
| `plan-review-synthesizer` | `agents/plan-review-synthesizer.md` | opus | YES — `["Read", "Write"]` | Consolidate four reports → verdict |

**Hooks:** none. No `hooks/` directory; not applicable to this pack.

**Router skill (`using-X` SKILL.md):** **none exists.** This pack ships only two specialist skills (`implementation-planning`, `plan-review`) with no router. Implication: per `/home/john/skillpacks/CLAUDE.md` and `reviewing-pack-structure.md`, *no* `.claude/commands/<pack-short>.md` slash-command wrapper is strictly required by the "every router needs a wrapper" rule.

**However:** `/home/john/skillpacks/plugins/axiom-procedural-architecture/skills/using-procedural-architecture/SKILL.md` references this pack four times as `/axiom-planning` (lines 3, 44, 63, 155), e.g. *"Do not use for implementation-plan critique of code changes (use `/axiom-planning` instead)"*. That URL form does not resolve — there is no `/home/john/skillpacks/.claude/commands/axiom-planning.md`. The sibling pack is making a cross-pack handoff to a slash command that has no wrapper. See **Major-2** below.

**Cross-skill references (outbound):**
- `skills/implementation-planning/SKILL.md:60` invokes `superpowers:executing-plans` as `REQUIRED SUB-SKILL` in the plan header template.
- `skills/implementation-planning/SKILL.md:228` and lines 213–217 of README list `superpowers:executing-plans`, `superpowers:subagent-driven-development`, `superpowers:test-driven-development`, `superpowers:brainstorming` as the surrounding workflow.
- `commands/review-plan.md:246, 256` reference `superpowers:executing-plans`.

These are reasonable — the pack positions itself as a refined alternative to `superpowers:writing-plans` (README §"What's Different from superpowers:writing-plans", lines 52–66) and hands off to other `superpowers:` skills for adjacent stages.

**Description-trigger conformance:**
- `implementation-planning` description (line 3): `"Use when you have specifications…"` — conforms to repo "Use when…" convention.
- `plan-review` description (line 3): `"Use after implementation-planning to validate plans against codebase reality…"` — does *not* start with "Use when" but uses "Use after" which is a meaningful temporal constraint. Marginal conformance.
- All five agent descriptions are short scope statements; none end with the SME-Agent-Protocol marker phrase. See **Critical-1** below.

---

## 2. Domain & Coverage

**Domain (inferred from artefacts, no user scope provided):** *implementation-plan-as-artifact* — production of TDD-shaped, atomic-task implementation plans plus a multi-perspective quality gate that validates plans against codebase reality before execution. Adjacent to but distinct from `axiom-procedural-architecture` (general staged-procedure design) and `superpowers:writing-plans` (the upstream pack this one was forked from).

**Coverage map (model-derived):**

| Concept area | Status | Evidence |
|--------------|--------|----------|
| Plan-document structure (header, task template) | EXISTS | `implementation-planning/SKILL.md:54-150` |
| Atomic task granularity discipline | EXISTS | lines 34-52 |
| Code-completeness standard (no pseudocode) | EXISTS | lines 152-167 |
| Exact-path / exact-command discipline | EXISTS | lines 154-160 |
| TDD interleaving (RED-GREEN per task) | EXISTS | template lines 88-149 |
| Rationalization defenses (Red Flags table) | EXISTS | lines 196-208 |
| Common Mistakes table | EXISTS | lines 183-194 |
| Execution handoff (subagent vs new session) | EXISTS | lines 210-229 |
| Cross-skill reference syntax (avoid `@`) | EXISTS | lines 169-181 |
| Plan-review verdict logic | EXISTS | `plan-review/SKILL.md:99-111` |
| Four-reviewer architecture (Reality/Architecture/Quality/Systems) | EXISTS | `plan-review/SKILL.md:60-99`; 4 agents |
| Synthesizer with priority scoring | EXISTS | `plan-review-synthesizer.md:60-78` |
| Cost warning + simplified mode | EXISTS | `commands/review-plan.md:11-39, 269-283` |
| JSON output schema | EXISTS | `plan-review-synthesizer.md:102-181` |
| **Plan-revision workflow** (after CHANGES_REQUESTED) | **MISSING** | `/review-plan` says "fix blocking issues, then run /review-plan again" but no skill governs the *fix-the-plan* loop. No `plan-revision` or `responding-to-plan-review` skill. |
| **Brownfield-plan input** (existing partial plan, not authored by `implementation-planning`) | PARTIAL | `commands/review-plan.md:287-292` warns "plans in other formats may produce incomplete reviews" but no guidance on how to *adapt* such plans. |
| **Plan-size / scope-control heuristics** | PARTIAL | `plan-review-architecture.md:28-43` covers blast radius weighting *during* review; `implementation-planning/SKILL.md` does not constrain max-task-count or max-files-per-plan during *authoring*. |
| **Estimation discipline / DoD acceptance criteria** | PARTIAL | Definition-of-Done checklists exist (line 145), but no acceptance-criteria authoring guidance distinct from per-task DoD. |
| **Multi-plan / dependency-between-plans coordination** | MISSING | No guidance on epic-shaped work that produces multiple plans. |
| **Router skill** (using-axiom-planning) | MISSING | See Major-1. |

Domain is **stable** (the discipline is process documentation, not an evolving framework). No Phase-A research needed.

---

## 3. Fitness Scorecard

| Dimension | Rating | Evidence |
|-----------|--------|----------|
| **Router quality** | Major | No `using-X` router skill exists. With only 2 specialist skills the absence is defensible, but the pack's sibling (`axiom-procedural-architecture`) links to `/axiom-planning` as if a router slash command exists; users following that link find nothing. |
| **Skill descriptions** | Pass | Both descriptions are activation-correct. `implementation-planning` uses "Use when"; `plan-review` uses "Use after" (semantically correct, mildly off-pattern). Each description does meaningful work in the trigger sentence. |
| **Frontmatter conformance** | Pass | Skills have `name` + `description` only (per repo norm). Command has `description`, `allowed-tools` (quoted JSON-style array), `argument-hint` (quoted) — matches marketplace style exactly. Agents have `description`, `model`, `allowed-tools` — `allowed-tools` on agents is uncommon in this marketplace (~5/65 agents per the maintenance sheet) but is a *restriction*, so explicit declaration is defensible. |
| **Component cohesion** | Pass | The 4-reviewer + synthesizer + command + 2-skill assembly is internally consistent. Scope boundaries between agents are crisp (each agent's "I check / I do NOT check" footer at `plan-review-reality.md:116-120`, `plan-review-architecture.md:166-170`, `plan-review-quality.md:175-179`, `plan-review-systems.md:200-204`). Reviewers do not overlap; synthesizer correctly limits itself to consolidation (`plan-review-synthesizer.md:252-262`). |
| **Slash-command exposure** | Major | `/review-plan` command exists and is correctly wired. But cross-pack references to `/axiom-planning` (a non-existent slash command) suggest the pack was expected to be a router with a wrapper. Either add a router + wrapper (`/axiom-planning`), or update the sibling pack's references to `/review-plan` or `implementation-planning` directly. |
| **SME Agent Protocol** | **Critical** | All 5 agents are reviewer / synthesizer / critic-style — exactly the SME shape. **Zero** agents declare "Follows SME Agent Protocol with confidence/risk assessment" in the description. **Zero** agent bodies cite `meta-sme-protocol:sme-agent-protocol`. **Zero** require the four output sections (Confidence Assessment, Risk Assessment, Information Gaps, Caveats). This is a structural violation of the marketplace's SME protocol — and notably, comparable review packs (e.g. `embedded-database-reviewer`, `sqlite-schema-architect`) do comply. Callers of these agents will not receive the confidence/risk discipline they expect from an SME agent. |
| **Anti-pattern coverage** | Pass | `implementation-planning/SKILL.md` has a Common Mistakes table (8 rows, lines 183-194) and Red Flags table (5 rows, lines 196-208). README §"What's Different" (lines 52-66) explicitly tracks 8 fixes against the parent pack. Each reviewer agent lists scope-bounded failure patterns (e.g., one-way doors at `plan-review-architecture.md:46-57`, security patterns at `plan-review-quality.md:81-91`). |
| **Cross-skill linkage** | Minor | Strong outbound links to `superpowers:executing-plans` etc. README §"Related Skills" (lines 207-217) lists complementary skills. Internal handoff diagram (`README.md:11-13`) clearly positions the pack in workflow. **Gap:** `plan-review` skill never tells users what to do when verdict = `CHANGES_REQUESTED` other than "fix and re-run" — no link to "plan-revision" guidance (because none exists). The implementation-planning skill does not back-reference plan-review as the next gate. |

**Overall:** **Major** (with one Critical embedded in the SME protocol dimension).

The core artefacts (skills, command, reviewer architecture) are well-formed and behaviorally coherent. What blocks a higher rating:
1. **Critical:** All five agents are SME-shaped but none follow the SME Agent Protocol — a hard violation of the cross-pack contract.
2. **Major:** Sibling-pack references to `/axiom-planning` resolve to nothing.
3. **Major:** No `plan-revision` guidance closes the `CHANGES_REQUESTED → fix plan → re-review` loop.

---

## 4. Behavioral Tests

Tests are designed against the rubric in `testing-skill-quality.md`. Without runtime dispatch in this report-only run, "likely actual" is inferred from textual analysis of each component — i.e., does the document *contain instructions that would resist the failure mode if followed*. I quote the load-bearing line.

### Test 1 — Pressure: time-and-simplicity on `implementation-planning`

**Scenario:** *"I need a quick plan to add a single helper function — just one task, no need for the full TDD ceremony. Give me a one-paragraph plan."*

**Expected behavior:** Skill should refuse to collapse the structure for a single-step task and either (a) redirect to "implement directly" (because the skill's own "Don't use for" list at lines 27-32 covers this case) or (b) produce the full task structure anyway.

**Likely actual behavior:** Pass. The skill has an explicit "Don't use for: Single-file changes (just implement directly)" clause at line 28. The Red Flags table at line 200 includes *"This step is quick, combine with next" → Atomic steps = atomic commits. Keep separate."* Both anti-shortcut defenses are present and visible.

**Verdict:** Pass.

### Test 2 — Pressure: skip-the-quality-gate on `plan-review` + `/review-plan`

**Scenario:** *"I already reviewed the plan myself, it's fine. Skip the review and just approve it so we can start coding."*

**Expected behavior:** Command should hold firm on the cost-warning gate but offer a graceful exit (Cancel) rather than self-approving.

**Likely actual behavior:** Mostly Pass. The command's cost-warning (`commands/review-plan.md:11-39`) offers exactly three options — Yes / Simplified / Cancel — none of which is "approve without review". The skill itself frames itself as a quality gate (`plan-review/SKILL.md:48-58`: *"Extra minutes here save days of rework downstream. A missed hallucination becomes a runtime error."*). **Caveat:** the skill's "Core Philosophy" exhortation is rhetorical, not enforcement — a Claude under pressure could legitimately Cancel and proceed; nothing in the pack tells the *caller* (e.g., an outer planning workflow) that bypassing review is itself a smell. This is a soft fail of the "discipline-enforcing" criterion — the gate is offered, not insisted upon. Marginal.

**Verdict:** Pass-with-Minor (gate is well-structured; no enforcement beyond it).

### Test 3 — Edge case: brownfield plan not produced by `implementation-planning`

**Scenario:** *"Here's a plan I wrote three weeks ago in a freeform style — please review it with /review-plan."*

**Expected behavior:** Either (a) graceful degradation with explicit "this isn't the expected format, results will be incomplete" warning, or (b) instructions for how to adapt it.

**Likely actual behavior:** Partial Pass. The command has a "Compatibility Note" at `commands/review-plan.md:286-292`: *"Plans in other formats may produce incomplete reviews."* The skill's "Limitations" section at lines 175-185 echoes this. **However**, the warning is reactive ("results will be incomplete") rather than constructive ("here's how to normalize it"). No skill, command, or agent says "first run the plan through `implementation-planning` to normalize, then review." The compatibility check is also not run — there is no preflight inspection step in Step 1 of the workflow (`commands/review-plan.md:73-116`).

**Verdict:** Fix needed (Minor) — add a preflight format check and a normalize-or-degrade decision step.

### Test 4 — SME-agent contract: caller expects Confidence/Risk sections

**Scenario:** *I dispatch `plan-review-reality` (via the Task tool inside `/review-plan`) and consume its output programmatically, expecting the standard SME four-section tail.*

**Expected behavior:** The agent's output ends with **Confidence Assessment**, **Risk Assessment**, **Information Gaps**, **Caveats** — verbatim headings, per the marketplace SME contract.

**Likely actual behavior:** **Fail.** The agent's output format (`plan-review-reality.md:69-114`) ends with `## Blocking Issues` and `## Warnings` only. There is no Confidence Assessment, no Risk Assessment, no Information Gaps section, and no Caveats. Same pattern across `plan-review-architecture.md` (ends at `## Recommendations`, line 162), `plan-review-quality.md` (ends at `## Warnings`, line 172), `plan-review-systems.md` (ends at `## Recommendations`, line 195), and `plan-review-synthesizer.md` (ends at `## Quality Checks`, line 241).

The synthesizer's JSON output (`plan-review-synthesizer.md:102-181`) carries `reviewer_summaries.<x>.status / blocking / warnings` but not confidence or risk fields. A downstream agent expecting the SME protocol shape will not find it.

**Verdict:** **Fail (Critical).** See Critical-1 below.

### Test 5 — Real-world complexity: plan touches 14 files

**Scenario:** `/review-plan` is run against a plan with 14 modified files including 2 DB migrations and 1 API contract change.

**Expected behavior:** Architecture reviewer should classify blast radius as "Very High", flag the migrations and API change as one-way doors, recommend splitting.

**Likely actual behavior:** Pass. `plan-review-architecture.md:28-43` defines the bands precisely (`13+ files | Very High | Strongly recommend splitting`); the one-way-door table at lines 47-55 explicitly lists "Database migrations" (High, requires rollback), "API contract changes" (High, requires versioning), and "Schema changes" (High, requires backward compatibility). The agent has both the heuristic and the threshold.

**Verdict:** Pass.

### Test 6 — Cross-pack handoff: user follows `/axiom-planning` link from `axiom-procedural-architecture`

**Scenario:** Reading `using-procedural-architecture/SKILL.md`, a user sees *"use `/axiom-planning` instead"* (line 44) and types `/axiom-planning` in the Claude Code prompt.

**Expected behavior:** Slash command resolves to a router skill or wrapper that orients them to `implementation-planning` and `plan-review`.

**Likely actual behavior:** **Fail.** No `.claude/commands/axiom-planning.md` exists. The slash command does not resolve. The user has to discover `implementation-planning` and `/review-plan` by reading the pack README. Cross-pack onboarding is broken at the entry door.

**Verdict:** Fail (Major). See Major-2 below.

---

## 5. Findings

### Critical

**Critical-1: SME Agent Protocol non-compliance across all 5 agents.**

All five agents in this pack are reviewer/critic/synthesizer-style — the canonical SME shape per `using-skillpack-maintenance:SKILL.md` lines 169-184. The repo standard for SME agents is:

1. Description ends with the phrase **"Follows SME Agent Protocol with confidence/risk assessment."** (or equivalent).
2. Body cites `meta-sme-protocol:sme-agent-protocol`.
3. Body requires the four output sections: **Confidence Assessment**, **Risk Assessment**, **Information Gaps**, **Caveats** — verbatim headings.

Audit of all five agent files shows **none** of these three requirements is met:

| Agent | Description ends with SME phrase? | Body cites meta-sme-protocol? | Requires 4 sections? |
|-------|----------------------------------|-------------------------------|----------------------|
| `plan-review-reality` | No (`agents/plan-review-reality.md:2`) | No | No |
| `plan-review-architecture` | No (`agents/plan-review-architecture.md:2`) | No | No |
| `plan-review-quality` | No (`agents/plan-review-quality.md:2`) | No | No |
| `plan-review-systems` | No (`agents/plan-review-systems.md:2`) | No | No |
| `plan-review-synthesizer` | No (`agents/plan-review-synthesizer.md:2`) | No | No |

Comparable packs comply: e.g., `embedded-database-reviewer` and `sqlite-schema-architect` in `axiom-embedded-database` both declare SME compliance with the required sections (per the existing `reviews/axiom-embedded-database.md` audit). The omission here is a structural drift, not a deliberate exemption — there is no marker in the agent files saying "exempt because executor" (the executor-exemption note in `analyzing-pack-domain.md:78` covers autonomous executors like `delinting-specialist`, not reviewers).

**Impact:** Callers integrating these agents into larger workflows cannot reliably surface confidence/risk because the agents don't produce that shape. Cross-marketplace tools that assume the SME contract (e.g., dashboards summarising agent outputs) will fail silently.

**Fix:** Add the SME-protocol marker line to each agent description; add the `**Protocol**:` body line per the convention at `using-skillpack-maintenance:SKILL.md:174-178`; extend each agent's "Output Format" section to require Confidence Assessment, Risk Assessment, Information Gaps, and Caveats. Extend the synthesizer JSON schema to carry per-reviewer confidence and risk fields so the verdict integrates them.

### Major

**Major-1: No router skill (`using-axiom-planning`).**

The pack has 2 sibling specialist skills with no router. The 2-skill count makes the absence defensible *internally*, but:

- The pack ships an implicit workflow (`brainstorming → implementation-planning → plan-review → executing-plans`, README:11-13) that exactly fits the router pattern.
- The marketplace convention is that any pack with more than one user-facing skill ships a router for discoverability.
- The sibling pack `axiom-procedural-architecture` already references this pack as `/axiom-planning` (see Major-2).

A small `using-axiom-planning` router skill (Start Here, Routing by Symptom, Boundary, Cross-References to `superpowers:` neighbours) would close both the discoverability gap and the broken-link gap from Major-2.

**Major-2: Broken cross-pack reference `/axiom-planning`.**

`/home/john/skillpacks/plugins/axiom-procedural-architecture/skills/using-procedural-architecture/SKILL.md` references `/axiom-planning` four times (lines 3, 44, 63, 155). The slash command does not exist. Either:

- Add a router (per Major-1) and a wrapper at `/home/john/skillpacks/.claude/commands/axiom-planning.md`, OR
- Rewrite the sibling pack's references to `/review-plan` (the command that *does* exist) or `implementation-planning` (the skill name).

The first option is preferred because it scales — other packs may want to handoff here too. If chosen, the wrapper should follow the pattern at `/home/john/skillpacks/.claude/commands/python-engineering.md` and similar wrappers.

**Major-3: No `CHANGES_REQUESTED → revise plan → re-review` workflow.**

`/review-plan` ends a `CHANGES_REQUESTED` verdict with *"Fix the blocking issues listed above, then run `/review-plan` again"* (`commands/review-plan.md:259-267`). But nothing in the pack governs the "fix" step:

- No skill on plan-revision discipline.
- `implementation-planning` does not back-link to plan-review or describe revision passes.
- The synthesizer's `recommendations[].resolution` field is per-issue, but there's no guidance on how to integrate fixes without breaking the existing plan's atomic-task structure (e.g., when fixing a hallucinated symbol means re-ordering tasks).

A `plan-revision` skill (or a section in `implementation-planning`) would close the loop. Without it, the pack is a one-shot pipeline — fine for greenfield, brittle when iterating.

### Minor

**Minor-1: Version drift between `plugin.json` (1.1.1) and README (1.1.0).**

`README.md:3` says `**Version:** 1.1.0`. `plugin.json:3` says `1.1.1`. The README §"Version History" (lines 188-204) does not list a 1.1.1 entry. Bring the README's version banner and version history in sync.

**Minor-2: `plan-review` description uses "Use after…" not "Use when…".**

`skills/plan-review/SKILL.md:3`. Semantically correct (the trigger *is* temporal — only fires after a plan exists), but mildly off the repo's dominant "Use when…" convention. Could be rewritten as: *"Use when validating an implementation plan against codebase reality, risk, complexity, and project conventions before execution."*

**Minor-3: Brownfield-plan preflight missing.**

`commands/review-plan.md:286-292` warns that non-`implementation-planning`-shaped plans may produce incomplete reviews, but Step 1 of the workflow (lines 73-116) does not perform a format preflight. Add a preflight that checks for the plan header signature (e.g., `# .* Implementation Plan`, presence of task numbering) and degrades or routes accordingly.

**Minor-4: Implementation-planning does not back-reference plan-review.**

`implementation-planning/SKILL.md:210-229` ("Execution Handoff") offers two execution options (subagent-driven, parallel session) but does not mention `/review-plan` as the optional gate before either. Adding a third bullet — *"Optionally, run `/review-plan` first to validate against codebase reality (recommended for high-risk plans)"* — would make the workflow position explicit.

**Minor-5: `plan-review/SKILL.md:185` claims "Plans in other formats may produce incomplete reviews"** — same statement as the command. Either de-duplicate (point one at the other) or keep both but cross-link them.

**Minor-6: README §"Testing Status"** (lines 96-103) reports RED-GREEN-REFACTOR figures for the `implementation-planning` skill only. The `plan-review` skill (added in v1.1.0 per the version history) has no testing-status entry. Either run the same TDD validation on `plan-review` and add results, or note the asymmetry explicitly.

### Polish

**Polish-1: Reviewer agents share an identical `allowed-tools` list** (`["Read", "Grep", "Glob", "Bash"]`) but each declares it individually. Acceptable; consider documenting "all reviewer agents share this tool set" in a header comment for future-proofing.

**Polish-2: `plan-review-reality.md:32-38`** uses double-backslash regex syntax in the example patterns (`\\(`). This is a minor cosmetic issue (the patterns are illustrative, not executable), but the double-escaping is unusual and likely an artefact of escaping in some earlier source. Single-backslash is sufficient inside a markdown code fence.

**Polish-3: The Co-Authored-By line** in the example commit at `implementation-planning/SKILL.md:142` cites `Claude Sonnet 4.5` — out of date relative to the current marketplace baseline (Opus/Sonnet 4.7 era per CLAUDE.md memory). Either update or generalise to `Claude <noreply@anthropic.com>`.

**Polish-4: Synthesizer's `priority_score`** is the product `Severity × Likelihood × Reversibility` (`plan-review-synthesizer.md:60-78`). Maximum value is 4×3×3 = 36; minimum 1×1×1 = 1. The example `priority_score: 12` (line 117) is sensible (Critical × Possible × Difficult = 4×1×3 or High × Likely × Difficult = 3×2×2). No bug, but worth documenting the range in the agent body so consumers can normalise.

**Polish-5: `commands/review-plan.md:85-93`** uses a Python-literal `locations = [...]` block inside a `bash` fence, with the comment "illustrative, not executable". Switch the fence to `text` or `python` to avoid lint confusion.

---

## 6. Recommended Actions

In priority order:

1. **[Critical] Bring all 5 agents into SME Agent Protocol compliance.** Update descriptions (add the marker phrase), bodies (cite `meta-sme-protocol:sme-agent-protocol` near the top), and output formats (require Confidence Assessment / Risk Assessment / Information Gaps / Caveats sections). Update the synthesizer JSON schema to carry per-reviewer confidence + risk. Bump to v1.2.0.

2. **[Major] Add a `using-axiom-planning` router skill** and a wrapper at `/home/john/skillpacks/.claude/commands/axiom-planning.md`. Use `superpowers:writing-skills` per the maintenance protocol (do not write the router inline). Then the cross-pack `/axiom-planning` references resolve.

3. **[Major] Add a `plan-revision` skill** (or an explicit "Revising a plan after CHANGES_REQUESTED" section in `implementation-planning`) governing the iteration loop between review verdicts.

4. **[Minor] Sync README version banner with `plugin.json`** (1.1.1) and add a 1.1.1 entry to §"Version History" explaining the patch contents.

5. **[Minor] Add a brownfield-plan preflight** to Step 1 of `/review-plan` that detects format and either degrades gracefully or instructs the user to normalise via `implementation-planning` first.

6. **[Minor] Back-link `/review-plan`** from `implementation-planning`'s execution-handoff section.

7. **[Minor] Add RED-GREEN-REFACTOR testing status for `plan-review`** to the README (or document why it's not validated).

8. **[Polish] Update the example Co-Authored-By model name** to a current value.

9. **[Polish] Document the synthesizer `priority_score` range** (1-36) explicitly.

10. **[Polish] Fix the `bash` fence around the Python `locations = [...]` block** in `commands/review-plan.md:84-93`.

**Recommended version bump:** Minor → **1.2.0**. The SME protocol fix (Critical-1) is a structural change that affects every downstream caller of the reviewer agents. Per the maintenance sheet's "Default for maintenance: Minor bump" guidance, this is the right rung. Reserve Major for cases where components are removed or the philosophy shifts.

---

## 7. Reviewer Notes

**Confidence:** Medium-High. The audit covered every file in the pack (2 skills + 5 agents + 1 command + README + plugin.json = 9 files, 1710 lines). Frontmatter conformance, scope-boundary text, and output formats were read in full. Marketplace registration and the sibling-pack cross-reference were verified directly. Behavioral tests were inferred from textual analysis rather than runtime subagent dispatch — a higher-fidelity follow-up would dispatch general-purpose subagents per `testing-skill-quality.md:81-93`.

**Risk:** Low to the audit conclusions themselves; the SME-protocol gap (Critical-1) is a binary structural check and unambiguous. The router/wrapper gap (Major-1, Major-2) is also binary. Major-3 (plan-revision workflow) is a judgement call — reasonable maintainers could decide the gap is acceptable scope. The Minors and Polish items are all small and uncontested.

**Information gaps:**
- No user-stated scope was provided for this review (per `analyzing-pack-domain.md` Phase D). The audit treated the artefacts themselves as the scope baseline. A user might say "plan-review is intentionally not SME-compliant because the per-reviewer outputs feed into the synthesizer which produces the final SME-shape", but the synthesizer also doesn't ship SME sections, so that defence isn't currently load-bearing.
- I did not test the agents under actual subagent dispatch — the report-only constraint precluded that. Test 4's "Fail" verdict is based on the agents' documented output format, not observed runtime behavior. A runtime test could plausibly show that Sonnet, when running the agent, adds something resembling confidence sections spontaneously — but that's not the same as the agent *requiring* them, which is what the SME contract checks.
- The pack's relationship to the user's broader `superpowers:` workflow could be tighter or looser depending on whether the maintainer intends this pack as a *replacement* for `superpowers:writing-plans` (current README framing) or a *complement*. Both readings produce the same recommendations above, so the ambiguity does not change the verdict.

**Caveats:**
- This is a report-only pass. No edits were made. No version was bumped. No git operations were performed.
- The version-drift Minor (README says 1.1.0, plugin.json says 1.1.1) is unverified as to *what* changed at 1.1.1 — the commit history was not inspected. The maintainer may have an explanation that supersedes the Minor finding.
- The "/axiom-planning" broken reference (Major-2) is *only* observable because of a sibling pack. If the sibling pack is itself in flux (v0.1.x per CLAUDE.md memory), the maintainer may prefer to fix the broken reference on the sibling side rather than build a router here. Either side closes the gap.
