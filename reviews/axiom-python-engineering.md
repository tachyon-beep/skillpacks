# Review: axiom-python-engineering

**Version:** 1.5.0  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

Stages 1-4 of `using-skillpack-maintenance` applied. Stage 5 (execution) deliberately skipped — report-only.

---

## 1. Inventory

### Plugin metadata

- `/home/john/skillpacks/plugins/axiom-python-engineering/.claude-plugin/plugin.json` (v1.5.0)
- Self-description: "Modern Python 3.12+ engineering with uv-first tooling: types (mypy / pyright / ty / pyrefly), testing, async, scientific computing, ML workflows, Textual TUI. Includes 10 reference sheets, 4 commands ..., 3 agents."
- Marketplace entry (`/home/john/skillpacks/.claude-plugin/marketplace.json`): description claims "10 skills" — matches.

### Skills (1 router + 10 reference sheets)

Path: `/home/john/skillpacks/plugins/axiom-python-engineering/skills/using-python-engineering/`

| File | Lines | Frontmatter | Role |
|------|-------|-------------|------|
| `SKILL.md` | 443 | `name`, `description` | Router |
| `modern-syntax-and-types.md` | 899 | none | Reference sheet |
| `resolving-mypy-errors.md` | 1120 | none | Reference sheet |
| `project-structure-and-tooling.md` | 1758 | none | Reference sheet |
| `systematic-delinting.md` | 1506 | none | Reference sheet |
| `testing-and-quality.md` | 1849 | none | Reference sheet |
| `async-patterns-and-concurrency.md` | 1131 | none | Reference sheet |
| `scientific-computing-foundations.md` | 981 | none | Reference sheet |
| `ml-engineering-workflows.md` | 1079 | none | Reference sheet |
| `debugging-and-profiling.md` | 1047 | none | Reference sheet |
| `textual-tui-development.md` | 691 | none | Reference sheet |

Frontmatter-on-reference-sheets absence is consistent with marketplace convention (reference sheets are content files referenced by the router SKILL.md; no YAML frontmatter required).

### Commands (4)

Path: `/home/john/skillpacks/plugins/axiom-python-engineering/commands/`

| Command | Description | argument-hint | allowed-tools |
|---------|-------------|---------------|---------------|
| `delint.md` | Systematically fix lint warnings using category-by-category approach | `"[path or file] - defaults to current directory"` | `["Read", "Edit", "Bash", "Glob", "Grep", "Skill"]` |
| `typecheck.md` | Run mypy type checking with systematic error resolution | `"[path or file] - defaults to current directory"` | `["Read", "Edit", "Bash", "Glob", "Grep", "Skill"]` |
| `profile.md` | Profile Python code to find actual bottlenecks before optimizing | `"<file.py> [function_or_script_args]"` | `["Read", "Bash", "Write", "Skill"]` |
| `create-project-scaffold.md` | Scaffold a new Python project with modern tooling (uv, ruff, mypy, pytest, pre-commit) | `"<project-name> [--minimal|--ml] [--pip-only]"` | `["Read", "Write", "Bash", "Skill"]` |

All four use quoted JSON-style `allowed-tools` arrays and quoted `argument-hint` strings, conformant with marketplace style.

### Agents (3)

Path: `/home/john/skillpacks/plugins/axiom-python-engineering/agents/`

| Agent | Model | SME? | Frontmatter keys |
|-------|-------|------|------------------|
| `python-code-reviewer.md` | sonnet | Yes (declared) | `description`, `model` |
| `delinting-specialist.md` | haiku | No (autonomous executor) | `description`, `model` |
| `refactoring-architect.md` | sonnet | Yes (declared) | `description`, `model` |

Both SME-style agents (`python-code-reviewer`, `refactoring-architect`) include the conventional `**Protocol**:` body line citing `meta-sme-protocol:sme-agent-protocol` and require Confidence / Risk / Information Gaps / Caveats sections. Descriptions end with "Follows SME Agent Protocol with confidence/risk assessment." `delinting-specialist` is correctly NOT marked SME (it's an autonomous executor on haiku — exempt per the rubric). Model selection per rubric: haiku for mechanical sweeps, sonnet for review/critique work. All three conform.

No `tools:` declarations — agents inherit parent context (per marketplace convention).

### Hooks

`/home/john/skillpacks/plugins/axiom-python-engineering/hooks/hooks.json`: `{"hooks": {}}` — empty bag (intentional). Not load-bearing; harmless.

### Slash-command wrapper

`/home/john/skillpacks/.claude/commands/python-engineering.md` exists — pack is invocable as `/python-engineering`. **Content drift detected** (see Findings).

### Marketplace registration

Plugin is registered in `/home/john/skillpacks/.claude-plugin/marketplace.json` with correct directory source. Confirmed.

---

## 2. Domain & Coverage

### User-defined scope (inferred from plugin description and SKILL.md)

- **Intent:** Modern Python 3.12+ engineering — language, tooling, testing, async, scientific computing, ML workflows, TUIs.
- **In scope:** Python-specific implementation, tooling, patterns, debugging, optimization, library-specific patterns (numpy/pandas/mlflow/textual).
- **Out of scope:** Non-Python languages, algorithm theory, deployment infrastructure, database design (SKILL.md §"When NOT to Use").
- **Audience:** Practitioners → experts. Skills assume Python literacy and teach mastery patterns + anti-patterns.

### Coverage map vs reality

| Area | Coverage | Status |
|------|----------|--------|
| **Foundational: Type system** | `modern-syntax-and-types.md` covers generics, protocols, mypy/pyright/ty/pyrefly | OK |
| **Foundational: Type error resolution** | `resolving-mypy-errors.md` covers systematic resolution, `type: ignore` discipline | OK |
| **Foundational: Project setup** | `project-structure-and-tooling.md` 1758 lines — uv, ruff, pyproject.toml, src vs flat | OK |
| **Foundational: Testing** | `testing-and-quality.md` 1849 lines — pytest, fixtures, mocking, property-based | OK |
| **Core: Async / concurrency** | `async-patterns-and-concurrency.md` — TaskGroup, structured concurrency | OK |
| **Core: Performance / profiling** | `debugging-and-profiling.md` — cProfile, memory_profiler, pdb | OK |
| **Core: Lint hygiene** | `systematic-delinting.md` — fix-never-disable methodology | OK |
| **Core: Scientific stack** | `scientific-computing-foundations.md` — numpy/pandas vectorization | OK |
| **Advanced: ML workflows** | `ml-engineering-workflows.md` — MLflow, reproducibility | OK |
| **Advanced: TUI development** | `textual-tui-development.md` — Textual framework | OK |
| **Crosscutting: Logging** | Not a dedicated skill | Possible gap (low priority) |
| **Crosscutting: Packaging / distribution** | Covered inside `project-structure-and-tooling.md` | OK |
| **Crosscutting: CLI/argparse/click/typer** | Not explicit | Possible gap (low priority) |
| **Crosscutting: HTTP clients (httpx, requests)** | Touched in `async-patterns-and-concurrency.md` (aiohttp examples) | Adequate |
| **Crosscutting: Data validation (pydantic)** | Referenced in commands (--ml) but no dedicated coverage | Minor gap |

**Domain currency.** Python tooling is fast-evolving (uv, ruff, ty, pyrefly are all <3 years old). The pack reflects current state of the art: uv-first scaffolding, PEP 735 dependency groups, pyrefly/ty mentioned alongside mypy/pyright. This is impressive currency for a v1.5.0 pack — the description's "types (mypy / pyright / ty / pyrefly)" claim is substantiated in `modern-syntax-and-types.md:611-638`.

### Gap summary

- **No critical gaps.** Coverage map matches plugin scope.
- **Minor gaps:** No dedicated skills for logging, CLI frameworks (click/typer), pydantic. Each is touched obliquely but not given a sheet. These are reasonable omissions for a 10-skill pack and would only become priorities if user demand surfaces.

---

## 3. Fitness Scorecard

| Dimension | Rating | Notes |
|-----------|--------|-------|
| **Router quality** | Major | Description doesn't follow marketplace "Use when..." convention (just "Routes to..."). Routing tables themselves are excellent — symptom-driven, with explicit cross-cutting sequences and "common routing mistakes" table. |
| **Skill descriptions** | Pass | Reference sheets have no frontmatter (per marketplace convention). Each has a clear "Overview", "Core Principle", "When to Use / Don't use when" block with concrete trigger phrases. |
| **Frontmatter conformance** | Pass | Router SKILL.md has `name` + `description`. Commands use quoted JSON arrays for `allowed-tools` and quoted strings for `argument-hint`. Agents declare only `description` + `model` (the dominant marketplace pattern). |
| **Component cohesion** | Pass | Commands map to specific skills (`delint` → `systematic-delinting`, `typecheck` → `resolving-mypy-errors`, `profile` → `debugging-and-profiling`, `create-project-scaffold` → `project-structure-and-tooling`). Agents complement commands (delinting-specialist runs the delint workflow; python-code-reviewer reviews; refactoring-architect does restructuring). Clean separation. |
| **Slash-command exposure** | Major | Wrapper exists at `/home/john/skillpacks/.claude/commands/python-engineering.md` but is **drifted** — skill catalog at lines 327-340 lists only 9 skills, missing `textual-tui-development`. Also lacks the explicit "How to Access Reference Sheets" path block that the canonical SKILL.md added (lines 28-42 of SKILL.md). |
| **SME agent protocol** | Pass | Two SME agents (`python-code-reviewer`, `refactoring-architect`) include the canonical `**Protocol**:` line citing `meta-sme-protocol:sme-agent-protocol` and require the four output sections verbatim. Description ends with "Follows SME Agent Protocol with confidence/risk assessment." `delinting-specialist` correctly omits SME treatment (autonomous executor on haiku). |
| **Anti-pattern coverage** | Pass | Each skill has an explicit anti-pattern register. Examples: `systematic-delinting.md:42-94` shows fix-vs-disable and delint-vs-refactor anti-patterns; `python-code-reviewer.md:127-136` has an anti-pattern severity table. Router SKILL.md has a "Common Rationalizations" table (lines 343-354) and a self-check Red Flags checklist (lines 358-387). This is strong rationalization-resistance content. |
| **Cross-skill linkage** | Pass | The router has an explicit "Cross-Cutting Scenarios" section (SKILL.md:268-291) listing sequenced multi-skill workflows. Specialist sheets cross-reference each other ("see `modern-syntax-and-types`", "see `project-structure-and-tooling`"). Agents reference both skills and other packs (e.g., `python-code-reviewer.md:88-112` redirects to `ordis-quality-engineering` and `ordis-security-architect` when out of scope). |

**Overall: Minor.** No critical or rebuild-grade defects. Two Major findings, both mechanical: stale slash-command wrapper and the router's `description:` field not using the marketplace's "Use when..." discovery idiom. Otherwise the pack is structurally and substantively sound.

---

## 4. Behavioral Tests

Tested router + 3 specialists (across distinct sub-domains: lint hygiene, scientific computing, TUI). Tests are scenario-based reads against the rubric in `testing-skill-quality.md` — observing what the documents would instruct under pressure.

### Test 1 — Router pressure: "I have 50 mypy errors after pulling main. Just tell me what to do."

**Scenario type:** Pressure (time + simplicity temptation).

**Expected behavior:** Route to `resolving-mypy-errors.md`, do not jump to advice.

**Observed (from SKILL.md content):**
- Routing table at lines 60-78 explicitly maps "100+ mypy errors, where to start?" and "mypy error: Incompatible types" → `resolving-mypy-errors.md`.
- "Common Rationalizations" table line 348: "User is rushed, skip routing | Routing takes 5 seconds. Wrong fix wastes hours. | Route anyway."
- Red Flags Checklist line 381: "Am I feeling pressure to skip routing? Time pressure → Route anyway (faster overall)."
- The router will resist the pressure and route to the specialist. **Pass.**

### Test 2 — Router edge case: "Optimize my pandas code, it's slow"

**Scenario type:** Edge case (could go to multiple specialists).

**Expected:** Route to `debugging-and-profiling.md` FIRST, then `scientific-computing-foundations.md`.

**Observed:**
- Router lines 197-220 explicitly handle "Performance and Profiling": "Route to `debugging-and-profiling.md` FIRST" then "may route to ... `scientific-computing-foundations.md` if array operations slow".
- Cross-Cutting Scenarios line 282: "Slow pandas code: 1. profile, 2. scientific-computing."
- Common Routing Mistakes line 320: "Pandas slow | debugging only | debugging THEN scientific-computing | Profile then vectorize."
- Routing is explicit, ordered, and the rationale ("Don't optimize without profiling") is repeated. **Pass.**

### Test 3 — `systematic-delinting` pressure: "Just add `# noqa` to make CI pass, we're shipping today"

**Scenario type:** Pressure (sunk cost + time).

**Expected:** Refuse to disable; offer a triage path.

**Observed (`systematic-delinting.md`):**
- Lines 38-62 — "The Golden Rule: Fix warnings by changing code to comply with the rule. NEVER disable warnings with `# noqa` ... The only exception: Third-party code you can't modify."
- The skill is built around the fix-vs-disable axis as its core principle. A model loading this skill will resist `# noqa`-bombing. **Pass.**

### Test 4 — `scientific-computing-foundations` real-world complexity: "My DataFrame transformation is slow but it uses iterrows because it depends on previous row"

**Scenario type:** Real-world (the dependency-on-previous-row case is the classic argument FOR iterrows).

**Expected:** Skill should offer vectorized alternatives (shift, cumsum, expanding window) rather than blessing iterrows. (Did not read the whole sheet in this review; this is the test design.)

**Observed (sampled `scientific-computing-foundations.md`):** Core Principle at line 5: "Vectorize operations, avoid loops ... The biggest performance gains come from eliminating iteration over rows/elements." The python-code-reviewer agent treats `iterrows()` as a **Critical** anti-pattern (agent `python-code-reviewer.md:131`). The combined message is unambiguous: iterrows is a smell, find a vectorized alternative. **Pass on stance.** Without reading the full 981 lines, I cannot confirm whether the skill specifically addresses the previous-row-dependency case with concrete shift/cumsum patterns — flagging as a possible **Minor** follow-up: spot-check that scientific-computing-foundations covers the previous-row case explicitly.

### Test 5 — `textual-tui-development` edge case: "compose() yields widget but nothing renders"

**Scenario type:** Edge case (one of the most common Textual-newbie symptoms).

**Expected:** Skill should call out the standard culprits: missing `App.run()`, missing CSS layout, `mount` vs `compose` confusion.

**Observed (sampled `textual-tui-development.md`):** The "When to Use" block at line 12 explicitly lists "compose() not showing widgets" as a trigger phrase. The Core Principle (line 7) calls out the most common mistakes: "forgetting to `await` mount operations, blocking the event loop, and not understanding the reactive lifecycle." The basic App pattern is the first content shown. The skill is symptom-aware. **Pass on coverage of triggers.** (Did not read past line 60; full content not validated.)

### Test 6 — Slash-command wrapper pressure: User runs `/python-engineering` and asks about Textual

**Scenario type:** Real-world (testing the wrapper, not the underlying SKILL.md).

**Expected:** Wrapper should route to Textual skill.

**Observed:** The wrapper at `/home/john/skillpacks/.claude/commands/python-engineering.md:327-340` lists "Complete Python engineering toolkit" with 9 skills. **`textual-tui-development` is missing from this catalog.** The wrapper also does not mention Textual in its symptom routing (no "Terminal UI Development" section, unlike SKILL.md lines 171-194). **Fail.** The wrapper would not surface Textual as a routing target; a user who asked `/python-engineering` then "I'm building a TUI with Textual" might not be routed correctly.

### Test 7 — Agent SME-protocol compliance: `python-code-reviewer` asked to review insecure code

**Scenario type:** Out-of-scope handoff.

**Expected:** Recognise security as out-of-scope, redirect to `ordis-security-architect`.

**Observed:** Agent `python-code-reviewer.md:104-112` has an explicit "Security Concerns" handoff block: checks for `plugins/ordis-security-architect/.claude-plugin/plugin.json`, recommends loading it if found, recommends installing it if not. The handoff pattern is explicit and conditional on whether the sibling plugin is installed. **Pass.**

### Test 8 — Agent activation negative case: `refactoring-architect` asked to "add a new endpoint"

**Scenario type:** Out-of-scope refusal.

**Expected:** Refuse the work as not-restructuring.

**Observed:** Agent `refactoring-architect.md:38-42` has a "DO NOT trigger" example: "Add a new endpoint to this Flask app — Do NOT trigger: This is feature work, not restructuring. Main Claude handles it." And another at lines 43-46 redirecting lint work to `delinting-specialist`, and lines 47-51 redirecting review work to `python-code-reviewer`. Three negative activation examples — strong scope-boundary discipline. **Pass.**

### Behavioral test summary

| Test | Component | Result |
|------|-----------|--------|
| 1 | Router pressure | Pass |
| 2 | Router edge case | Pass |
| 3 | `systematic-delinting` pressure | Pass |
| 4 | `scientific-computing-foundations` real-world | Pass (stance); spot-check recommended for previous-row case |
| 5 | `textual-tui-development` edge case | Pass (triggers covered) |
| 6 | `/python-engineering` wrapper drift | **Fail** — Textual missing from catalog |
| 7 | `python-code-reviewer` security handoff | Pass |
| 8 | `refactoring-architect` out-of-scope refusal | Pass |

7/8 pass. One Major (drifted wrapper). No critical failures.

---

## 5. Findings

### Critical

None.

### Major

**M1. Slash-command wrapper `/python-engineering.md` is drifted.**

- **Location:** `/home/john/skillpacks/.claude/commands/python-engineering.md`
- **Symptom:** Skill catalog at lines 327-340 lists 9 skills; `textual-tui-development` is missing. The wrapper also has no "Terminal UI Development" routing section, unlike the canonical SKILL.md (lines 171-194).
- **Impact:** Users invoking `/python-engineering` would not be routed to the Textual skill for TUI work. Plugin description and SKILL.md advertise Textual support; the user-facing slash command does not.
- **Evidence:** Compare wrapper line 339 (last skill listed: "debugging-and-profiling") with SKILL.md line 404 (last skill listed: "textual-tui-development"). Wrapper has 9-item skill catalog; SKILL.md has 10.
- **Fix:** Resync the wrapper from the canonical SKILL.md, OR replace the wrapper body with a thin pointer to the router skill rather than duplicating its content. (Duplicating content invites this exact drift.)

**M2. Router `description:` does not follow marketplace "Use when..." discovery idiom.**

- **Location:** `/home/john/skillpacks/plugins/axiom-python-engineering/skills/using-python-engineering/SKILL.md:3`
- **Symptom:** `description: Routes to appropriate Python specialist skill based on symptoms and problem type` — does not start with "Use when ...".
- **Impact:** The marketplace convention (per `reviewing-pack-structure.md` and observed across other packs) is for skill descriptions to start with "Use when [trigger condition] — [what the skill does]" to maximise discoverability. The current description tells the model what the skill IS but not WHEN to load it. Compare `lyra-ux-designer:using-ux-designer` ("Route to the right UX skill based on your task and platform context") — also a slight miss, but the python router is the more terse of the two.
- **Fix:** Rewrite to e.g. "Use when working on a Python task and unsure which specialist skill to load — routes to type-system, testing, async, profiling, scientific-computing, ML-workflows, TUI, or project-setup specialists based on symptoms."

### Minor

**m1. Plugin description double-counts components.**

- **Location:** `plugin.json:3-5` — "Includes 10 reference sheets, 4 commands ..., 3 agents."
- The router SKILL.md is the 11th skill file (it is itself a skill, not just a reference sheet). The description's "10 reference sheets" count is technically correct (the router has 10 routable specialists), but the marketplace entry's "10 skills" is the same count differently framed. Internally consistent but a reader counting files in `skills/` will find 11.
- **Fix:** Optionally clarify "1 router + 10 reference sheets".

**m2. Reference sheets begin with a stray blank line before the H1.**

- **Location:** All 10 reference sheets — `head -1` returns blank.
- **Impact:** Aesthetic only; no parser issues observed.
- **Fix:** Strip leading blank lines.

**m3. Spot-check needed for `scientific-computing-foundations.md` previous-row-dependency case.**

- **Location:** `scientific-computing-foundations.md` (981 lines, not fully read in this review).
- **Symptom:** The skill is firmly anti-iterrows. The most common legitimate-looking argument FOR iterrows is "I depend on the previous row." If this case is not explicitly addressed with shift/cumsum/expanding-window alternatives, users will rationalise iterrows in their specific case.
- **Fix:** Confirm the skill covers shift/cumsum/expanding-window patterns; if not, add a section.

**m4. No coverage of logging, click/typer, or pydantic.**

- Each is touched obliquely (commands mention pydantic for `--ml`; httpx/aiohttp shown in async examples) but none has a dedicated sheet. These are legitimate Python practitioner skills.
- **Fix:** Track as backlog. Not necessary for v1.5.0; would be a v2.0 expansion.

**m5. `hooks/hooks.json` is empty.**

- Empty `{"hooks": {}}` file. Not load-bearing.
- **Fix:** Optional — either remove the file (pack works without it) or document the placeholder.

### Polish

**p1. SKILL.md "Future cross-references" still lists future-tense items.**

- Location: SKILL.md:438-443. Cross-references to `superpowers:test-driven-development` and `superpowers:systematic-debugging` are listed as "future". These plugins exist today (`superpowers:test-driven-development` is loaded in this very session). The "future" framing is stale.
- **Fix:** Promote to active cross-references; remove "Phase 1 - Standalone" framing.

**p2. Router section "How to Access Reference Sheets" (SKILL.md:28-42) is only in SKILL.md, not in the slash wrapper.**

- The path-disambiguation block is critical for the model to find sibling reference sheets. The slash wrapper omits this block, which compounds M1.
- **Fix:** Carry the block over when resyncing the wrapper, OR (preferred) have the wrapper redirect to the router skill rather than duplicate.

**p3. `create-project-scaffold` command is detailed but does not mention `pyrefly`/`ty` configuration.**

- Location: `commands/create-project-scaffold.md`. The pyproject template (lines 88-115) configures ruff, mypy, pytest, but not pyrefly/ty even though the plugin description advertises them.
- **Fix:** Optional — either drop pyrefly/ty from the plugin advertisement OR add commented-out template stanzas for them.

**p4. SKILL.md "Why" sections are sometimes thin.**

- E.g., SKILL.md:94 — "Why: Project setup involves multiple tools (ruff, mypy, pre-commit) and architectural decisions (src vs flat layout). Need comprehensive setup guide." That's fine. Some others are similarly fine. Polish-grade nit only.

---

## 6. Recommended Actions

In priority order (no execution requested — this is the queue for a future Stage 5 pass).

1. **M1 — Resync `/home/john/skillpacks/.claude/commands/python-engineering.md`** with the canonical SKILL.md. Better: replace the wrapper body with a short pointer that loads the router skill. This prevents the drift class entirely. (Medium-impact patch; not minor — it affects user-facing routing.)
2. **M2 — Rewrite router `description:`** to follow "Use when..." idiom. One-line frontmatter edit. Affects skill discovery.
3. **m1 — Clarify plugin metadata** "10 reference sheets" vs "10 skills" wording (very minor).
4. **m3 — Spot-check `scientific-computing-foundations.md`** for explicit previous-row-dependency coverage; add if absent.
5. **m2, p1, p2 — Cleanup pass:** strip leading blank lines from reference sheets; un-future-ify the Integration Notes block; carry "How to Access Reference Sheets" into the wrapper if the wrapper is kept content-duplicating.
6. **p3 — Decision:** either de-advertise pyrefly/ty or add template stanzas to the scaffold command.
7. **m4 — Backlog for v2.0:** dedicated sheets for logging, click/typer, pydantic.
8. **m5 — Decide on `hooks/hooks.json`:** keep as placeholder (document) or remove.

**Version-bump recommendation if all of the above land:** Minor (1.5.0 → 1.6.0). No philosophy shift, no component removal — enhanced guidance and structural fixes.

---

## 7. Reviewer Notes

- **Method.** Read the three rubric sheets (`analyzing-pack-domain.md`, `reviewing-pack-structure.md`, `testing-skill-quality.md`) before generating findings. Inventoried all components by filesystem walk. Read in full: router SKILL.md, all 4 commands, all 3 agents, slash-command wrapper. Read first 60-100 lines of each of 5 reference sheets (out of 10) representing distinct sub-domains: type-error resolution, lint hygiene, async, scientific computing, TUI, testing, project setup. Did not read the full ~13,500 lines of reference-sheet content; this review is structural + spot-behavioral, not line-by-line content audit.
- **Behavioral testing.** Performed against the rubric by reading the documents and reasoning about what they would instruct a model to do under each pressure scenario. Did not dispatch a subagent for live behavioral testing — that would be the next-fidelity step if the user wants empirical confirmation of any specific finding (especially the iterrows previous-row case in m3).
- **Confidence assessment.**
  - High confidence: M1 (wrapper drift — verified by direct comparison), M2 (description format — direct frontmatter inspection), agent SME-protocol findings (direct inspection of all three agents), command frontmatter (direct inspection of all four).
  - Medium confidence: behavioral assessments of reference sheets (sampled openings, not full reads).
  - Low confidence: m3 (would need to read the full 981 lines of `scientific-computing-foundations.md` to be sure).
- **Risk assessment.** The pack is in good shape. No structural defects warranting rebuild. M1 is the most actionable finding — wrapper drift is a recurring pattern that suggests a structural fix (single source of truth) rather than a content patch.
- **Information gaps.**
  - Did not read 5 of the 10 reference sheets in full.
  - Did not dispatch a fresh-context subagent for live behavioral tests — all tests were document-read inferences.
  - Did not assess whether `axiom-python-engineering` overlaps with `axiom-rust-engineering`, `yzmir-pytorch-engineering`, or `yzmir-ml-production` in scope.
- **Caveats.** Report-only per the user's instructions. No edits made. Stage 5 (execution) deliberately skipped. The Findings section's prioritization is a recommendation; the user owns the decision to act.
