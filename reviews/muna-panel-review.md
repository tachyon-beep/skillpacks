# Review: muna-panel-review
**Version:** 0.3.2  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

**Scope.** Stages 1-4 of `using-skillpack-maintenance` applied read-only: inventory and domain mapping (Stage 1), structural fitness scorecard (Stage 2), reasoning-only behavioral tests (Stage 3), and findings/discussion (Stage 4). Stage 5 (execution / fixes) is explicitly out of scope.

**Sources read.** Plugin metadata, marketplace registration, all 3 agents in full (`persona-reader.md`, `persona-designer.md`, `panel-synthesiser.md`), the orchestration skill `reader-panel-review/SKILL.md` in full, `ARCHITECTURE.md` in full, `process.md` Phases 1-5 plus execution notes (~75% coverage; skimmed Phase 5 outline/sign-off review formats and the calibration/known-limitations sections in full), partial `config-template.md` (first 60 lines) and `config.md` (first 40 lines) — these are user-facing reference rather than runtime methodology so only spot-checked.

## 1. Inventory

**Plugin metadata** (`plugins/muna-panel-review/.claude-plugin/plugin.json:1-19`)
- `name`: `muna-panel-review`
- `version`: `0.3.2`
- License: `CC-BY-SA-4.0`. Author/repo: `tachyon-beep`.
- Description (line 4): "Simulated audience panel review — persona-driven editorial intelligence for document suites. 3 agents (persona-reader, panel-synthesiser, persona-designer), 1 orchestration skill."
- Keywords: `muna`, `panel`, `review`, `editorial`, `audience`, `persona`.

**Marketplace registration** (`.claude-plugin/marketplace.json`)
- Registered. Source: `./plugins/muna-panel-review`. Category not shown in extract but entry is present.
- Marketplace description (line ~equivalent): "Simulated audience panel review — persona-driven editorial intelligence for document suites. 3 agents (persona-reader, panel-synthesiser, persona-designer), 1 orchestration skill, **3 slash commands**" — this claim diverges from the plugin's own `plugin.json` (which says "1 orchestration skill" with no command count). See Findings.
- Cross-referenced by `lyra-creative-writing` ("Composes with muna-panel-review for beta-reader simulation") — verified at marketplace.json description for lyra-creative-writing.

**Skills (1)**

| Skill | Path | Lines | Description |
|---|---|---|---|
| `reader-panel-review` | `plugins/muna-panel-review/skills/reader-panel-review/SKILL.md` | 510 | Orchestrator coordinator guide for running the panel |

This is NOT a router skill (no `using-*` naming). It is a single workflow skill that orchestrates agents.

Skill body anatomy (`reader-panel-review/SKILL.md`):
- `:1-4` — frontmatter (name + 1-line description, no `allowed-tools`).
- `:6-16` — overview, framing the primary output as "editorial intelligence".
- `:16-23` — Cost Warning section (`SKILL.md:17`: "This is an expensive skill"; required user-confirmation handshake before starting).
- `:25-30` — Prerequisites (config file, document files, output directory).
- `:34-50` — Phase 0: optional persona-designer dispatch.
- `:54-92` — Phase 1: config parsing (required fields, validation summary, halt-on-missing).
- `:96-181` — Phase 2: working directory setup (4 steps — output dirs, hashed `.chapters/` store, manifest.json, TOC build).
- `:184-258` — Phase 3: team creation and per-persona spawn-prompt template.
- `:261-335` — Phase 4: the chapter pump — five message types (chapter-request / document-switch / return-to-abandoned / reading-complete / agent-failure), each with explicit coordinator actions.
- `:339-357` — voice re-anchoring payload (duplicated from `process.md:399-419`).
- `:361-394` — context-management carry-forward payload (recent journal paths + arc-summary table; duplicated from `process.md:967-984`).
- `:398-408` — time-budget tracking.
- `:412-428` — coordinator state-tracking fields.
- `:432-460` — error handling (Step A missing, chapter not found, agent failure).
- `:464-501` — Phase 5: completion and post-reading (collision tests, synthesiser dispatch, presentation, cleanup).
- `:506-511` — authority split between this skill and `process.md`.

Phase 4 ("the chapter pump") is the runtime core. `SKILL.md:265-287` defines the canonical request-handling sequence: enforce Step A gate (read journal, verify expectations are concrete not hedge-y), look up hashed filename in manifest, check re-anchoring schedule, check time-budget reminder, check carry-forward eligibility, send message, update tracking state. This is the pack's most operationally specific section and it lines up with the agent's turn-output protocol (`persona-reader.md:52-56`) one-to-one.

**Empty leftover directory:** `plugins/muna-panel-review/skills/using-panel-review/` exists as an empty directory — no SKILL.md, no reference sheets. Either a renamed/abandoned router or scaffolding that was never populated. See Findings.

**Reference / support documents (alongside the skill, in the plugin root)**

| File | Lines | Purpose |
|---|---|---|
| `process.md` | 1133 | The methodology bible — Phases 1–5, journal format, voice guidelines, synthesis structure, execution notes |
| `config.md` | 129 | Worked example config (cloud-migration suite) |
| `config-template.md` | 261 | Config-file format reference |
| `ARCHITECTURE.md` | 167 | Component architecture and workflow diagram |

These files live at the plugin root, NOT inside the SKILL.md's directory. They are referenced by absolute path from the SKILL.md and from agents. This is an unusual layout for this marketplace — most plugins put reference content inside `skills/using-X/*.md`. See Findings.

**Commands (0 inside the plugin)**
- `plugins/muna-panel-review/commands/` does NOT exist.
- However, three slash-command wrappers exist at the **repo root** `.claude/commands/`:
  - `panel-review.md` (29 lines) — invokes the `reader-panel-review` skill
  - `panel-config.md` (33 lines) — guides the user through writing a config file (mostly inline guidance, no agent dispatch)
  - `panel-designer.md` (30 lines) — describes spawning the `persona-designer` agent (but does NOT actually invoke the skill or dispatch the agent — it is informational only)
- None of the three wrappers declares frontmatter (no `description:`, `allowed-tools:`, or `argument-hint:` blocks). They begin with a markdown title.
- Per the rubric (`reviewing-pack-structure.md:128-141`): "Plugin missing from `marketplace.json`" / "Slash-command wrapper exists, no router skill" — this is the inverse of those: wrappers exist at repo root but the plugin itself ships no commands, and the wrappers reference a non-router skill. See Findings.

**Agents (3, in `agents/`)**

| Agent | Path | Lines | Model | Tools | Role |
|---|---|---|---|---|---|
| `persona-reader` | `agents/persona-reader.md` | 56 | `sonnet` | `Read, Write` | Configurable reader; one instance per persona |
| `persona-designer` | `agents/persona-designer.md` | 106 | `sonnet` | `Read, Write, Glob` | Cold-start panel designer; reads documents and writes a config file |
| `panel-synthesiser` | `agents/panel-synthesiser.md` | 156 | `opus` | `Read, Write, Glob` | Cross-panel synthesis with epistemic tier grading |

All three agents declare `tools:` restrictions. The rubric (`SKILL.md:25, 99`) notes most repo agents omit `tools:`; this pack deliberately uses restrictions as the safety mechanism (e.g., `persona-reader` has no `Bash` and no `Glob` so it cannot browse for chapters — the hashed-filename design depends on this).

Per-agent body-level evidence:
- `persona-reader.md:9-12` opens with three identity directives: "Stay in character. Your voice, blind spots, and concerns are defined by your persona spec. You are allowed to be wrong, confused, bored, hostile, or disengaged. These are findings, not failures." This permission-to-be-difficult is the core anti-flattening mechanism — without it, simulated readers default to a balanced analytical voice.
- `persona-reader.md:14-25` ("Operating Rules") declares 8 non-negotiable rules — tool restriction is rule 1, Step A gate is rule 3, the turn-output protocol is rule 4. `:52-56` enumerates the four valid idle-output patterns.
- `persona-reader.md:27-29` ("Shutdown Handling") instructs the agent to approve coordinator-issued shutdown requests as normal end-of-life signal, not an interruption to resist.
- `persona-reader.md:31-35` ("Journal Memory Rules") makes the agent's saved files its external memory; on returning to abandoned documents, the agent re-reads its own earlier journal entries. Retrospective reassessment ("if your feelings about an earlier chapter have changed based on later reading, note the shift") is captured as a finding rather than papered over.
- `persona-reader.md:42-49` ("Startup Sequence") is a 4-step bootstrap: read `process.md` Phases 2-3, read inline persona spec, write `00-suite-orientation.md`, request first chapter. All four steps happen before the chapter pump engages.
- `persona-reader.md:36-40` ("Tool Restriction Reassurance") explicitly addresses the agent's likely confusion about missing Glob/Bash — a defensive design feature for an agent that runs inside a restricted tool set.
- `persona-designer.md:24-34` ("Analysis Approach") enumerates seven things to identify in a document suite (audience addressed / talked-about / decision-maker / affected / decision chain / vocabulary boundaries / institutional perspectives).
- `persona-designer.md:77-91` ("Contamination Constraint") names what the designer may NOT do (predict verdicts, predict chapter-specific reactions, write reading-behaviour routes informed by chapter content). This is the contamination firewall — same pattern as the panel-synthesiser's contamination constraint at `panel-synthesiser.md:96-103`.
- `panel-synthesiser.md:69-79` defines three sequential analytical passes (reading-path / per-chapter / thematic) and maps each pass to specific synthesis sections (2,4,9 / 3,5,6,7 / 1,8,10,11,12,13). The agent cites `process.md` Phase 4 as the runtime authority for the actual synthesis structure.
- `panel-synthesiser.md:81-103` defines Section 13 (panel gaps and suggested personas) with the contamination constraint — identity fields only, no behavioural predictions.

None of the three agent descriptions end with "Follows SME Agent Protocol with confidence/risk assessment." None of the bodies cite `meta-sme-protocol:sme-agent-protocol`. None requires the four output sections (Confidence Assessment, Risk Assessment, Information Gaps, Caveats). See Findings — the three agents are arguably non-SME executors (they produce structured documents, not advice), so SME-protocol non-compliance may be intentional.

Spawn-prompt template fidelity (`SKILL.md:201-256`):
- The coordinator's spawn prompt for each `persona-reader` includes: full inline persona spec, scenario framing (or explicit "none"), absolute path to `process.md` with phase pointers (2 and 3), per-document tables of contents (titles and chapter numbers only — no content, no file paths, no hashed filenames), the output directory path, and a "Begin" instruction.
- The "Documents Available" section (`SKILL.md:232-245`) is deliberately limited to titles and chapter numbers — no file paths. This pairs with the agent's tool restrictions (`persona-reader.md:5`: `Read, Write` only) and the hashed-filename store to make pre-emptive chapter access structurally impossible.
- The synthesiser spawn prompt (`SKILL.md:480-486`) provides: output directory path, absolute path to `process.md`, absolute path to the config file. The synthesiser glob-discovers persona outputs from those starting points.
- The designer spawn prompt (`SKILL.md:36-46`) provides: document paths or directory, optional audience context, optional desired panel size, paths to `process.md` and `config-template.md`, output path for the config file.

**Agent communication primitives.**
- The pack uses `TeamCreate` / `SendMessage` / `TeamDelete` (8 references in `SKILL.md`, 7 in `process.md`, 8 in `ARCHITECTURE.md`). The skill assumes a Claude Code runtime with the team-mode subagent dispatch primitives available. This is a runtime dependency that should be documented at the plugin level — see Polish.

**Hooks:** None.

**Support documents — content survey.**

`process.md` (1133 lines) is the methodology authority. Section breakdown:
- `:1-8` framing — distinct from `expert-panel-review.md` (mentioned but not present in this pack).
- `:10-23` "When to use" + prerequisites — five trigger conditions for running a panel.
- `:25-35` process overview — 5 phases plus optional 3b (collision tests) and variant (delegation chain).
- `:39-143` Phase 1: panel design — design principles, persona spec format, document suite, scenario framing, panel sizing (`:84-89` — 7+ for full review, 3-5 for chapter/section, 2-3 for derivative), persona priority (`:90-101` — four archetypes), control persona (`:104-122`), unreliable narrator (`:125-143`).
- `:149-435` Phase 2: chapter-by-chapter mood journals — cardinal rule (`:149-161` with banner), execution model, suite orientation entry template, realistic reading paths (`:225-241` enumerating skip/jump/switch/abandon/stop), three-step mood journal format (Step A `:247-266`, Step B `:268-328`, Step C `:330-345`), light journal variant (`:350-381`), voice guidelines (`:383-398`), voice re-anchoring (`:399-419`), chapter mapping (`:421-435`).
- `:439-535` Phase 3: final reflections — in-character output formats by persona type (`:443-458` — 10 persona types with appropriate output formats), overall verdict template (`:462-529`).
- `:539-595` Phase 3b: persona collision tests (optional).
- `:599-785` Phase 4: cross-panel synthesis — synthesis structure (`:603-722` — 13 sections), methodology rules (`:748-766`), quality gate (`:768-779`).
- `:789-873` Phase 5: derivative document development (no executable surface — see Findings).
- `:877-1133` Execution Notes — orchestrator specification (`:885-957`), agent workflow (`:959-1011`), manual execution (`:1013-1015`), delegation chain variant (`:1017-1057`), adaptation guide (`:1059-1068`), calibration (`:1070-1082`), process checksum manifest (`:1083-1125`), what this process does NOT do (`:1127-1133`).

`ARCHITECTURE.md` (167 lines) — purpose statement, three-agent architecture (per-agent role/model/tools/prompt/behaviour), the one orchestration skill, full workflow diagram with text art, output directory layout, and file inventory.

`config.md` (129 lines partial reading) — worked cloud-migration example: 5-document suite (suite map + 2 technical + 2 derivative), scenario framing (v0.9 internal review draft, Q3 timeline), and (presumably — only first 40 lines read) 6 persona specs with all required fields.

`config-template.md` (261 lines partial reading) — format reference: document suite table format, chapter-to-file mapping convention, suite map (optional), presentation order, scenario framing structure (framing statement template + framing-effects table requirements).

**Slash-command wrapper alignment**
- The plugin's single skill is `reader-panel-review` (not a `using-X` router) — per the rubric (`SKILL.md:228`), this means no slash-command wrapper is *required* by the router convention.
- However, three repo-root wrappers exist: `panel-review.md` (invokes the skill), `panel-config.md` (inline guidance), `panel-designer.md` (informational).
- Marketplace description claims "3 slash commands" (see above) — this matches the three repo-root wrappers. But these wrappers ship at the repo level, not inside the plugin, so users installing the plugin without the skillpacks repo do NOT receive them. See Findings.

## 2. Domain & Coverage

**Domain.** Simulated audience panel review — persona-driven editorial intelligence for document suites. The pack treats a panel run as a three-phase pipeline: design personas → simulate independent readers chapter-by-chapter → synthesise cross-panel findings.

**Boundaries (declared in `process.md:1129-1133`).** Does NOT do:
- Technical correctness review (defers to a hypothetical `expert-panel-review.md`)
- Copy editing
- Approval / clearance
- Replace real audience testing
- Model network dynamics (cross-persona coordination during reading)

**Audience.** Practitioners producing document suites for institutional consumption — policy documents, RFCs, security frameworks, technical specifications. The worked example (`config.md`) is a cloud-migration suite; the design admits clinical protocols, RFCs, etc.

**Coverage map vs implementation:**

Foundational:
- Panel design principles (span decision chain, vary depth, blind spots, control persona, unreliable narrator) — `process.md:39-143`. **Status: Exists.**
- Persona specification format — `process.md:55-70`. **Status: Exists.**
- Cardinal rule (no read-ahead) — `process.md:149-161`. **Status: Exists, heavily-warned.**
- Step A gate (expectations before reading) — `process.md:247-266`, `SKILL.md:271`, `persona-reader.md:20`, `process.md:949-957`. **Status: Exists across all components.**

Core techniques:
- Three-step journal format (A: expectations / B: impressions / C: navigation) — `process.md:243-345`. **Status: Exists.**
- Light journal variant — `process.md:350-381`. **Status: Exists.**
- Voice re-anchoring (every 3rd chapter or document switch) — `process.md:399-419`, `SKILL.md:340-357`. **Status: Exists.**
- Carry-forward (context management at 15+ chapters) — `process.md:967-984`, `SKILL.md:361-394`. **Status: Exists.**
- Final reflections in-character — `process.md:439-535`. **Status: Exists.**
- Collision tests (optional) — `process.md:539-595`. **Status: Exists.**
- Cross-panel synthesis (13 sections + quality gate + manifest) — `process.md:599-785`, `panel-synthesiser.md:69-117`. **Status: Exists.**
- Epistemic tier grading (Tier 1 text-surface / Tier 2 affective / Tier 3 institutional) — `process.md:756-762`. **Status: Exists.**
- Convergent-reasons test (vs convergent-conclusions) — `process.md:764`. **Status: Exists.**
- Derivative document development (Phase 5) — `process.md:789-873`. **Status: Exists, but no skill/command/agent covers Phase 5 directly.** See Findings.

Advanced / variants:
- Delegation chain simulation — `process.md:1017-1057`. **Status: Documented in `process.md` but has no skill/agent surface area.** See Findings.
- Calibration against real audience feedback — `process.md:1070-1082`. **Status: Documented as guidance.**
- Hashed-filename anti-browse mechanism — `SKILL.md:117-135`, `ARCHITECTURE.md:34`. **Status: Exists, structurally enforced.**
- Process checksum manifest (YAML) — `process.md:1083-1126`, `panel-synthesiser.md:111-117`. **Status: Exists.**

Cross-cutting:
- Anti-contamination discipline (Step A gate + hashed filenames + designer/synthesiser contamination constraints) — multiple locations. **Status: Strong.**
- Output directory structure — `ARCHITECTURE.md:127-148`, `SKILL.md:96-181`. **Status: Exists, well-specified.**
- Error handling (Step A missing, chapter not found, agent failure) — `SKILL.md:432-460`. **Status: Exists.**

Output directory layout (`ARCHITECTURE.md:129-148`):
```
{output-dir}/
├── .chapters/                    # Hashed chapter files (coordinator's working copy)
│   ├── manifest.json             # Mapping: original filename → hashed filename
│   ├── a7f3b2.md, c9d1e4.md, ...
├── {persona-slug}/
│   ├── 00-suite-orientation.md
│   ├── 01-..., 02-..., ...
│   ├── overall-verdict.md
│   └── collision-{other}.md      (if collision tests run)
├── 00-reader-panel-synthesis.md
└── 00-process-manifest.yaml
```

The `.chapters/` prefix (dot-prefixed) signals "coordinator-private" — agents have output paths inside `{persona-slug}/`, never under `.chapters/`. The hashed-filename store is the structural anti-browse mechanism: even if an agent had `Glob`, the filenames are opaque hashes with no semantic content. The `manifest.json` is the only mapping back to original names, and it lives outside the agent's tool access.

Synthesis collation discipline (`panel-synthesiser.md:46-67`, `process.md:730-738`):
- Before writing any synthesis content, the synthesiser builds three working documents — `00-collation-reading-paths.md`, `00-collation-chapter-index.md`, `00-collation-themes.md`. These are intermediate artefacts the synthesiser operates on rather than re-reading 100+ journal files by memory.
- The collation step is enforced ("Without this step, the synthesiser is searching hundreds of journal files by memory — it will miss patterns and produce a weaker synthesis") — `panel-synthesiser.md:48`.
- Each collation document feeds specific synthesis passes: reading-paths → Pass 1 → sections 2/4/9; chapter-index → Pass 2 → sections 3/5/6/7; themes → Pass 3 → sections 1/8/10/11/12/13.

**Domain stability.** Stable — this is a methodological pack, not a tracking-the-frontier pack. No Phase-A research required.

**Known simulation limitations (declared in `process.md:1078-1082`).** The pack is honest about its constraints:
- "LLM personas tend to be more patient, more thorough, and more analytical than real readers. Real executives stop reading sooner. Real junior developers skim more aggressively. The panel will over-read compared to real audiences."
- "LLM personas have difficulty sustaining emotional responses across long reading sessions. Real readers who become frustrated at chapter 5 may carry that frustration to chapter 15. LLM personas tend to reset to neutral."
- "LLM personas cannot simulate network effects — a real vendor who talks to other vendors before responding behaves differently from an isolated simulated vendor."

These limitations are surfaced inside the methodology rather than discovered by users post-hoc. The calibration / validation guidance (`process.md:1070-1076`) further hedges: "Treat panel findings as hypotheses, not conclusions" — strongly aligned with the epistemic-tier grading.

**Gaps.**
- **Phase 5 (Derivative Document Development) has no executable surface.** `process.md:789-873` defines outline-review and sign-off-review formats, but no skill, command, or agent picks up the synthesis output and runs Phase 5. The skill ends at "Phase 5: Completion and Post-Reading" (`SKILL.md:464-501`) which is synthesiser + cleanup — Phase 5-as-process-doc and Phase 5-as-skill-step have the same name but different content. See Findings (Major).
- **Delegation chain variant has no executable surface.** `process.md:1017-1057` describes a multi-stage chain simulation but no skill/agent/command covers it.
- **No README at plugin root.** Other plugins in the marketplace ship a README; this one has `ARCHITECTURE.md` instead. Acceptable but unconventional.

## 3. Fitness Scorecard (8 dimensions, with 2 supplementary)

| # | Dimension | Rating | Evidence |
|---|---|---|---|
| 1 | **Skill discovery & frontmatter** | Minor | The single skill `reader-panel-review` has a valid 2-line frontmatter (`SKILL.md:1-4`). Description does NOT start with "Use when..." per the repo's dominant convention (`SKILL.md:133` of using-skillpack-maintenance). Trigger phrasing is functional but inconsistent with marketplace norm. |
| 2 | **Component-type alignment** | Major | The three agents are appropriately typed (executors, not advisors — non-SME). The one skill is correctly a workflow orchestrator. BUT: the three repo-root wrappers conflate "command", "informational page", and "skill invoker" — `panel-config.md` does not invoke any agent or skill; `panel-designer.md` does not actually dispatch the designer (just describes it). Only `panel-review.md` is a real skill invocation. |
| 3 | **Router / wrapper alignment** | Major | Marketplace description claims "3 slash commands" but the plugin ships no `commands/` directory. The three wrappers at `.claude/commands/` are repo-local and will not travel to end-users who install via the marketplace catalog. Either move the wrappers into `plugins/muna-panel-review/commands/` or revise the marketplace description to drop the claim. Also: empty `skills/using-panel-review/` directory is orphaned scaffolding. |
| 4 | **Agent design (SME-protocol where applicable)** | Pass | The three agents are non-SME executors (config writer, reader simulator, synthesis writer) — they produce structured artefacts, not advisory output. SME-protocol non-compliance is appropriate. Model selection is justified in `ARCHITECTURE.md:17, 41, 57` (sonnet for cost-driven simulators, opus for synthesis quality). Tools restrictions are intentional safety mechanisms (no-browse for the reader; controlled access for the designer/synthesiser). |
| 5 | **Methodological depth & coverage** | Pass | Coverage of design / reading / synthesis is thorough. The 13-section synthesis template (`process.md:603-722`), epistemic tier grading (`process.md:756-762`), and convergent-reasons test (`process.md:764`) are sophisticated and battle-tested in design. Phase 5 (derivative documents) and the delegation-chain variant are documented but not operationalised — see Findings. |
| 6 | **Anti-contamination & rationalisation resistance** | Pass | This is the pack's strongest dimension. The cardinal rule appears with 🚨-banner emphasis in `process.md:151-161` AND `process.md:986-1011`. The Step A gate is enforced in three places (skill, agent, process). Hashed filenames + tool restrictions make "read ahead" structurally impossible, not just discouraged. The designer/synthesiser contamination constraints (`persona-designer.md:77-91`, `panel-synthesiser.md:96-103`) prevent a different leak mode (predicting reactions from content knowledge). |
| 7 | **Repository convention compliance** | Minor | Skill description doesn't open with "Use when..." (Minor). Reference content lives at plugin root rather than inside `skills/<router>/` (unusual but functional — `process.md` is shared by all agents at runtime). Wrappers at `.claude/commands/` lack frontmatter (no `description`, `allowed-tools`, `argument-hint`) — most repo wrappers also omit frontmatter at the root level, so this may be repo-level convention. |
| 8 | **Cost / scale safety** | Pass | The skill prominently flags cost (`SKILL.md:16-23`): "A 13-persona panel reading a 20-chapter document suite can run hundreds of turns. The opus synthesiser at the end adds further cost." Required confirmation before starting. Carry-forward design (`process.md:967-984`) bounds context per agent. Calibration / validation guidance in `process.md:1070-1082` correctly frames panel findings as hypotheses, not conclusions. |
| 9 | **Process auditability** | Pass | The process checksum manifest (`process.md:1083-1126`, `panel-synthesiser.md:111-117`) is a YAML appendix capturing panel design vs execution (panels designed N, executed N, omitted personas and why), per-persona stats (chapters read, light vs full journals, stopped_reason, voice_anchors_applied, carry_forwards_used), and synthesis metadata (passes_completed, quality_gate, findings_by_tier). "This manifest makes completeness auditable at a glance" — `process.md:1123`. A reader of the synthesis can see whether all designed personas were run, whether any were operator-terminated rather than persona-decided, and whether the quality gate was met. Audit transparency is built into the deliverable. |
| 10 | **Documentation completeness within the pack** | Minor | `ARCHITECTURE.md` (167 lines) provides component architecture, workflow diagram, output directory structure, and file inventory. `process.md` is the methodology bible. `config-template.md` (261 lines) and `config.md` (129 lines) provide format reference and a worked example. The pack is heavily self-documented. The Minor concern is that no `README.md` exists at the plugin root; `ARCHITECTURE.md` is the de facto entry point but doesn't open with a clear "what this plugin is for" hook visible to a directory-browsing user. |

(8 + 2 supplementary dimensions = 10 — supplementary ones cover concerns the canonical 8-dimension scorecard does not address: process auditability and pack-internal documentation completeness.)

**Overall: Minor with one Major cluster.** The pack's methodology, anti-contamination design, and agent boundaries are strong. The Major cluster is structural: marketplace metadata vs filesystem reality (wrappers at repo root not in plugin), an empty `using-panel-review/` directory, and `panel-designer.md` / `panel-config.md` wrappers that do not actually invoke anything. Methodology coverage gap (Phase 5, delegation chain) is Minor — they're documented in `process.md` and could be added later without disrupting the v0.3.2 surface.

## 4. Behavioral Tests

Three reasoning-only scenarios. No live subagent dispatch — read-only review.

### Test 1 — Pressure scenario: "just simulate one persona reading the whole thing"

**Scenario.** A user says: "I don't have time for the full panel. Just spin up one CISO persona, give them the document, and write me a summary."

**Trace through the skill.**
- `SKILL.md:16-23` declares the cost-warning protocol but does not refuse single-persona runs. The pack's design (`process.md:84-101`) explicitly says: "Chapter or section review: 3–5 personas. Pick the personas most relevant to the content under review." The persona-priority guidance (`process.md:90-101`) gives an explicit fallback: "If you cannot run the full panel… prioritise personas that maximise tension." So one-persona is implicitly discouraged but not blocked.
- The cardinal rule (`process.md:151-161`) holds independently of panel size — even a 1-persona panel runs chapter-by-chapter, with Step A, no read-ahead.
- The Step A gate (`SKILL.md:271`, `persona-reader.md:20`) is structural: the coordinator does not provide the chapter file until Step A is verified. A "just give them the whole thing" shortcut would have to bypass the chapter pump entirely — which means abandoning the skill, not bending it.
- Hashed filenames (`SKILL.md:117-135`): even if a user wanted to give the agent the whole document, the agent has `Read, Write` only and cannot browse. The coordinator controls access.

**Verdict.** **Pass.** Pressure resistance is structural (tool restrictions + hashed filenames + Step A gate). The pressure scenario can be resisted by following the skill's mechanisms. The pack could be more explicit that "one persona is a degraded panel, not a panel" — `process.md:99-101` says a four-persona panel built from archetypes beats a 13-persona panel where most occupy similar positions, but does not warn against a 1-persona panel directly. Minor polish opportunity.

**Sub-test 1a — coordinator pressure.** A user says "skip Step A verification, it's slowing things down." Trace: `SKILL.md:271` is unambiguous: "Do NOT provide the chapter file until Step A is verified." `process.md:949-957` ("The Step A gate") explains *why* — it makes contamination structurally detectable, not just discouraged. The skill's authority split (`SKILL.md:506-511`) makes Step A enforcement an owned responsibility of the coordinator skill, not a methodological aspiration in `process.md`. **Pass** — pressure resistance is explicit at multiple layers.

**Sub-test 1b — Step A hedge attack.** A persona-reader writes a Step A entry that says "I expect this chapter to contain content relevant to the document's topic" — technically a Step A, but vacuous. Trace: `SKILL.md:271` instructs the coordinator to verify Step A "contains concrete expectations — not vague hedges, but a specific prediction about what the chapter will contain and what the persona needs from it." This is enforcement-by-coordinator-judgment, not enforcement-by-regex. The coordinator's response when verification fails is at `SKILL.md:432-444` — send a remediation message asking the agent to write a concrete prediction before proceeding. **Pass with caveat:** the "concrete vs hedge" judgment depends on the coordinator's interpretation. A defensive coordinator could under-enforce. The skill could provide 2-3 examples of acceptable vs unacceptable Step A entries to anchor the judgment.

### Test 2 — Edge case: the agent rationalises read-ahead

**Scenario.** A `persona-reader` agent, mid-pump, says: "I want to skim the next two chapters quickly to understand context before writing Step A."

**Trace.**
- `persona-reader.md:20` is explicit: "NEVER read a chapter file until you have written and saved your Step A expectations for that chapter." The agent has no Glob, no Bash, only Read — and only knows about hashed filenames it has been given. It can only read what the coordinator has sent.
- The coordinator only sends one chapter at a time (`SKILL.md:267-289`, `process.md:944-947`: "Batch chapters. One chapter per cycle. The Step A → content → Steps B+C sequence is the atomic unit.").
- If the agent "asks" for two chapters in its turn output, the skill's turn-output protocol (`persona-reader.md:52-56`) recognises only single-chapter requests. The coordinator would respond with one chapter or with an error.
- `process.md:986-1011` ("MANDATORY NO-READ-AHEAD DISCIPLINE") explicitly enumerates the "implementation patterns that violate this (DO NOT USE)" — first item: "Giving the agent the full document and asking it to write all journals at once".

**Verdict.** **Pass.** The agent cannot rationalise read-ahead because the structural protections prevent it from happening even if the agent tried. The Step A gate makes contamination detectable (`process.md:949-957`: "An agent that has already read ahead cannot write a genuine prediction"). The discipline is enforced by the system, not by the agent's good behaviour.

**Sub-test 2a — voice drift over long sessions.** A persona-reader at chapter 12 has gradually drifted toward a generic analytical voice — the CISO and the dev-lead are sounding interchangeable. Trace: voice re-anchoring fires every 3rd chapter (`SKILL.md:275, 339-357`; `process.md:399-419`) with the expanded payload (voice sample, consultation voice, blind spots first 2-3 items, key question, emotional-vocabulary palette). The 3-chapter interval is "more aggressive than typical re-anchoring but the marginal token cost is trivial compared to the signal preservation" (`process.md:417`). If drift persists despite re-anchoring, the synthesiser is instructed to flag it: `process.md:418-419` ("If two personas' journal entries become difficult to distinguish — similar vocabulary, similar sentence structure, similar analytical frame — this is a finding about the implementation, not the document.") **Pass** — drift is anticipated and the mitigation is both periodic (re-anchoring) and observable (synthesis flag).

**Sub-test 2b — context overflow at 15+ chapters.** A persona-reader has read 15 chapters and the context window is straining. Trace: `SKILL.md:279, 361-394` define the carry-forward payload. Part 1 (`SKILL.md:368-376`) gives the agent paths to the last 3 journal entries to re-read in full; Part 2 (`SKILL.md:378-393`) is an arc-summary table the coordinator builds from older entries. The arc-summary preserves "the emotional trajectory without the full text" (`process.md:984`). This is bounded-context discipline — the agent's context never accumulates past ~3 full journals plus a compact arc table, regardless of how long the suite is. **Pass.**

**Sub-test 2c — agent failure mid-read.** A persona-reader emits malformed output or stops responding. Trace: `SKILL.md:326-335` (message type 5: agent failure). Coordinator records failure in tracking state, sets `stopped_reason: agent_failure`, removes the agent from the active roster, does NOT retry. "The panel continues with the remaining agents" — failure of one persona is contained, not cascading. The process manifest (`process.md:1095-1104`) captures stopped_reason per persona, so failed personas are visible in the audit trail. **Pass.**

### Test 3 — Real-world complexity: ambiguous user, no config

**Scenario.** A user says: "I have these five markdown files in `docs/proposal/`. Run a panel review on them. I don't know how many personas to use."

**Trace through the skill.**
- `SKILL.md:25-30` ("Prerequisites") requires a config file. The skill handles the no-config case via Phase 0 (`SKILL.md:34-50`): offer to spawn the `persona-designer` agent.
- The designer (`persona-designer.md`) reads the documents freely (contamination is fine for the designer — it is not a reader), identifies audiences and decision chains, produces a config file with persona specs, scenario framing, panel config, and a "Panel gaps and deferred personas" section.
- Cost flag (`SKILL.md:16-23`) is presented before starting. The skill explicitly asks the user to confirm panel size before any agents are spawned (`SKILL.md:20-23`).
- The skill then proceeds through config parsing (`SKILL.md:54-92`), validation summary, and chapter pump.

**Verdict.** **Pass.** The Phase 0 affordance handles the no-config case cleanly. The persona-designer's contamination constraint (`persona-designer.md:77-91`) is correctly framed — the designer can read everything because it is not a simulated reader. The handoff back to the user for config review (`SKILL.md:48`) is the right place for a human-in-the-loop checkpoint.

**Minor concern (Test 3, polish).** The `panel-designer.md` wrapper at `.claude/commands/panel-designer.md:7-15` describes the persona-designer but does NOT instruct Claude to spawn it directly. A user running `/panel-designer` gets prose, not action. To actually run the designer, the user has to read the skill or invoke `/panel-review`. The wrapper is informational rather than active — see Findings.

**Sub-test 3a — config validation halt.** The user provides a config with one persona missing the `Voice sample` field. Trace: `SKILL.md:60-67` enumerates the required fields per persona and `SKILL.md:71-72` instructs: "If any required field is missing from any persona, report it and stop." This is fail-fast, not best-effort — the coordinator does not proceed past Phase 1 with malformed config. The validation summary (`SKILL.md:82-91`) reports persona count, document count, scenario-framing presence, control persona, unreliable narrator, and warnings — and "Wait for user confirmation before proceeding." Two synchronous halt points are enforced: validation halt and user-confirmation halt. **Pass.**

**Sub-test 3b — control-persona divergence.** After the panel runs, the control persona's actual reading diverged from the pre-registered prediction. Trace: `process.md:116-121` defines the three-outcome interpretation framework — (1) actual matches predicted = confidence increase; (2) actual diverges plausibly = "possibly the most valuable single finding in the panel"; (3) actual diverges implausibly = downweight findings from similar personas. The synthesiser must report this in its output, and the skill (`SKILL.md:495`) requires it: "Report the control persona comparison: did actual reading match pre-registered predictions?" Calibration is a first-class output, not a hidden assumption. **Pass.**

**Sub-test 3c — collision tests on a panel with no opposed verdicts.** All personas land in roughly agreement. Trace: `process.md:586-589` ("When to skip") is explicit: "Skip if the panel is small (fewer than 6 personas) or if the overall verdicts show no significant disagreements. Collision tests are only valuable when there is genuine tension to test. Running collisions between personas who agree is a waste of tokens." The skill (`SKILL.md:467-473`) requires the coordinator to read both verdicts before spawning collision agents — checking "they have opposed or divergent positions worth testing." Wasted-collision prevention is built in. **Pass.**

## 5. Findings (Critical / Major / Minor / Polish)

### Critical
- None.

### Major
1. **Marketplace description claims "3 slash commands" but plugin ships no `commands/` directory.**
   - Evidence: `.claude-plugin/marketplace.json` description for `muna-panel-review` says "3 agents (persona-reader, panel-synthesiser, persona-designer), 1 orchestration skill, 3 slash commands". `ls plugins/muna-panel-review/commands/` returns "No such file or directory". Three wrappers exist at `/home/john/skillpacks/.claude/commands/panel-{review,config,designer}.md` but these ship with the skillpacks repo, not with the plugin.
   - Impact: Users installing the plugin via the marketplace receive 1 skill + 3 agents, not 3 slash commands. Discovery is broken.
   - Fix: Either (a) move the three wrappers into `plugins/muna-panel-review/commands/` (with frontmatter — `description`, `allowed-tools`, `argument-hint`) so they install with the plugin, or (b) revise the marketplace description to drop the "3 slash commands" claim and the plugin.json description to match. Option (a) is preferred — the wrappers are useful entry points.

2. **`panel-config.md` and `panel-designer.md` wrappers do not actually invoke anything.**
   - Evidence: `.claude/commands/panel-config.md:1-33` provides inline guidance ("Tell me… I'll help you design personas") but does not invoke a skill or dispatch an agent. `.claude/commands/panel-designer.md:1-30` describes what the persona-designer would do but does not say "spawn the persona-designer agent" or "load the reader-panel-review skill". Only `.claude/commands/panel-review.md:18` actively invokes the skill ("Load the skill: `muna-panel-review:reader-panel-review`").
   - Impact: A user running `/panel-designer` gets prose, not action. Contrast `panel-review.md:18` which directs Claude to load the skill — `panel-designer.md` should either dispatch the agent or load the skill and trigger Phase 0.
   - Fix: Rewrite `panel-designer.md` to instruct Claude to dispatch the `muna-panel-review:persona-designer` agent with the user-provided documents. Rewrite `panel-config.md` to either dispatch the designer or load the skill at Phase 0. Alternatively, drop the wrappers and let `/panel-review` handle Phase 0 itself (the skill already does — `SKILL.md:34-50`).

3. **Empty orphan directory `skills/using-panel-review/`.**
   - Evidence: `ls -la plugins/muna-panel-review/skills/using-panel-review/` returns an empty directory (no SKILL.md, no .md files).
   - Impact: Suggests an abandoned migration to a router pattern, or scaffolding that was never populated. Confuses inventory; may trip plugin loaders that scan for `using-*/SKILL.md`.
   - Fix: Remove the directory, OR populate it with a real router SKILL.md if a router pattern is intended (which would also require introducing reference sheets — see Phase-5 / delegation-chain gap below).

### Minor
4. **Skill description does not open with "Use when…".**
   - Evidence: `plugins/muna-panel-review/skills/reader-panel-review/SKILL.md:3`: "Orchestrate a simulated reader panel review — spawn persona-reader agents as teammates, manage the chapter pump, enforce the Step A gate, and coordinate synthesis".
   - Impact: Discovery convention drift. The rubric (`SKILL.md:133`) flags "Use when..." as the dominant repo convention. This skill works functionally but is slightly less discoverable to Claude under skill-search.
   - Fix: "Use when running a simulated audience panel review on a document suite — orchestrates persona-reader agents, enforces the Step A gate, and coordinates synthesis." (Or similar.)

5. **Phase 5 (Derivative Document Development) and Delegation Chain variant have no executable surface.**
   - Evidence: `process.md:789-873` (Phase 5) and `process.md:1017-1057` (delegation chain) are fully documented but no skill, command, or agent picks them up. The `reader-panel-review` skill's Phase 5 section (`SKILL.md:464-501`) covers synthesiser + cleanup — same name, different content.
   - Impact: A user reading `process.md` will expect Phase 5 derivative document development to be runnable; it is not — they would have to drive it manually.
   - Fix: Add a `derivative-document-development` skill (or a follow-up command) that takes the synthesis output (Section 8: Audience-Specific Gaps) and runs outline review + sign-off review against the same panel. This is non-trivial scope — likely a v0.4 feature. For v0.3.x, document the gap explicitly in `SKILL.md:464` or `process.md:789`.

6. **Marketplace description does not match `plugin.json` description.**
   - Evidence: `.claude-plugin/marketplace.json` says "…1 orchestration skill, 3 slash commands"; `plugins/muna-panel-review/.claude-plugin/plugin.json:4` says "…1 orchestration skill" (no command claim).
   - Impact: Internal inconsistency between the marketplace catalog and the plugin's own metadata.
   - Fix: Reconcile during the same fix as Major finding #1.

7. **No README.md at plugin root.**
   - Evidence: `plugins/muna-panel-review/` contains `ARCHITECTURE.md`, `process.md`, `config.md`, `config-template.md`, but no `README.md`. Most plugins in the marketplace ship a README.
   - Impact: A user browsing the plugin directory has no top-level orientation file. `ARCHITECTURE.md` serves the purpose but does not match marketplace convention.
   - Fix: Add a short `README.md` (or rename/symlink `ARCHITECTURE.md`).

8. **`config-template.md` and `config.md` live at plugin root, not inside `skills/`.**
   - Evidence: Most reference content in this marketplace lives inside `skills/using-X/*.md` (e.g., lyra-creative-writing's craft and genre sheets). This pack puts shared methodology at the plugin root and references it by absolute path from the skill (`SKILL.md:26-30`) and from agents (`persona-reader.md:48`, `persona-designer.md:14-15`, `panel-synthesiser.md:17-19`).
   - Impact: Layout is functionally correct (agents need stable paths and shared access; nesting under one specific skill would imply ownership), but inconsistent with repo convention. Could be intentional given the multi-agent shared-methodology design.
   - Fix: Document the rationale in `ARCHITECTURE.md` or `README.md` — explain why `process.md` is plugin-root rather than skill-nested. (Or move it under `skills/reader-panel-review/process.md` and update all references; this is the higher-effort option.)

9. **`panel-review.md` wrapper at `.claude/commands/` has no frontmatter.**
   - Evidence: `panel-review.md:1` starts with a blank line then `# Run a Reader Panel Review`. No `description:`, `allowed-tools:`, `argument-hint:` block. Same for `panel-config.md` and `panel-designer.md`.
   - Impact: Per the rubric (`SKILL.md:142-148`), command frontmatter is expected. Other repo-root wrappers (spot-check needed) may or may not declare frontmatter. If the convention is repo-root wrappers don't carry frontmatter, this is a Polish item, not Minor.
   - Fix: Add frontmatter if convention requires it; otherwise document the repo-root-wrapper convention in `CLAUDE.md`.

### Polish
9b. **Step A gate "concrete vs hedge" judgment is unanchored.** `SKILL.md:271` requires the coordinator to verify Step A "contains concrete expectations — not vague hedges". The judgment depends on coordinator interpretation; no examples of acceptable vs unacceptable Step A entries are given. A defensive coordinator could under-enforce, accepting "I expect content relevant to the document's topic" as a valid prediction. Consider adding 2-3 worked examples — one clear pass, one clear fail, one ambiguous — to anchor the judgment.

9c. **Team-mode runtime dependency is implicit.** The skill uses `TeamCreate`, `SendMessage`, `TeamDelete` throughout (`SKILL.md:184-198, 281, 297, 310, 322, 482, 501`). These are Claude Code team-mode primitives; the skill does not declare this dependency at the top. A user on a runtime without team-mode would not discover the gap until Phase 3. Consider a one-line prerequisite at `SKILL.md:25-30` ("Requires Claude Code team-mode primitives: TeamCreate, SendMessage, TeamDelete").

10. **`SKILL.md:17` says "This is an expensive skill"; `SKILL.md:19-23` does the cost-confirmation handshake.** This is good. Consider adding a one-line numerical estimate of token cost per persona-per-chapter (or per 100 chapters) so users can ballpark before committing.

11. **`persona-reader.md:14` operating rules are a numbered prose list rather than a structured table.** Consider a structured table (Rule / Why / Detection) for at-a-glance scanning during the agent's startup.

12. **`process.md:1133` lines is large.** The agents read `process.md` Phases 2-3 (reader), Phase 1 (designer), Phase 4 (synthesiser). Token cost per startup is high. Consider splitting into per-phase files referenced by the agents directly — `process-phase-1.md`, `process-phase-2-3.md`, `process-phase-4.md` — so each agent reads only what it needs. This would also align with the marketplace's `skills/using-X/sheet-name.md` convention. (Trade-off: more files to maintain.)

13. **`SKILL.md:104` directory-creation block is bash, not language-agnostic.** If the coordinator runs in an environment where `Bash` is unavailable (rare but possible per the rubric), the skill would fail. Consider adding a fallback note or making the directory-creation step tool-agnostic.

14. **Voice anchor payload (`SKILL.md:340-357`, `process.md:399-419`) and carry-forward payload (`SKILL.md:361-394`, `process.md:967-984`) are duplicated between the skill and `process.md`.** `SKILL.md:509-510` declares this is intentional ("the only methodology elements duplicated"). If `process.md` is later split, ensure both copies stay in sync — or extract to a shared file.

15. **`panel-synthesiser.md:81-103` Section 13 contamination constraint is well-stated.** Consider adding the same explicit framing to the synthesiser's Pass 3 (`panel-synthesiser.md:77`) where derivative-document specs are generated — these specs are arguably as contamination-vulnerable as persona suggestions.

## 6. Recommended Actions

**Priority 1 (Major — address before v0.4):**
1. Reconcile the marketplace claim of "3 slash commands" with filesystem reality. Either move wrappers into `plugins/muna-panel-review/commands/` (with proper frontmatter) or revise the description.
2. Make `panel-config.md` and `panel-designer.md` wrappers actually dispatch the relevant agent or load the skill at the right phase. Currently they are prose pages, not commands.
3. Remove the empty `skills/using-panel-review/` directory (or populate it with a real router if that pattern is intended).

**Priority 2 (Minor — address opportunistically):**
4. Add "Use when…" prefix to the skill description.
5. Decide whether Phase 5 (derivative-document development) and the delegation-chain variant need executable surfaces. If yes, scope a v0.4 feature. If no, document the gap explicitly in the skill.
6. Add a `README.md` at plugin root, or note `ARCHITECTURE.md` as the equivalent.
7. Reconcile `plugin.json` description with marketplace description.

**Priority 3 (Polish — defer to v0.5+):**
8. Consider splitting `process.md` (1133 lines) into per-phase files to reduce per-agent startup cost.
9. Add a numerical token-cost estimate to the cost warning.
10. Document the repo-root wrapper convention vs in-plugin command convention in `CLAUDE.md`.
11. Anchor the Step A "concrete vs hedge" judgment with worked examples (one pass, one fail, one ambiguous).
12. Declare the Claude Code team-mode runtime dependency (`TeamCreate`, `SendMessage`, `TeamDelete`) in `SKILL.md:25-30` prerequisites.
13. Document the cross-pack composition pattern with `lyra-creative-writing` (beta-reader simulation) — either in `process.md`'s Adaptation section or as a `compositions.md` reference at the plugin root.
14. Consider whether the synthesiser's Pass 3 (derivative document specs generation) needs the same contamination-constraint framing applied to Section 13 — both project knowledge from journals onto something that would be consumed by readers who have not seen the panel.

## 7. Reviewer Notes

**Strengths.** This pack is one of the most disciplined examples of anti-contamination design in the marketplace. The cardinal rule (no read-ahead) is enforced through four independent mechanisms: structural (hashed filenames + tool restrictions), procedural (Step A gate, coordinator-controlled file access), behavioural (persona-reader operating rules), and detectable (`process.md:949-957` describes how contamination is detected post-hoc). The contamination constraints on the designer (`persona-designer.md:77-91`) and synthesiser (`panel-synthesiser.md:96-103`) extend the same discipline to agents that legitimately read content — a sophisticated separation of "may read" from "may predict from reading".

The epistemic tier grading (`process.md:756-762`) and the convergent-reasons test (`process.md:764`) are mature methodology contributions — they prevent the synthesis from collapsing into "everyone agreed, so it must be true" failure mode. The control persona (`process.md:104-122`) and unreliable narrator (`process.md:125-143`) calibration mechanisms are documented at a level that survives the pressure scenarios in section 4.

**Weaknesses.** The packaging story is unfinished. The marketplace catalog promises three slash commands the plugin does not ship; the repo-root wrappers do not all do what their filenames imply; the empty `using-panel-review/` directory implies a planned-but-abandoned restructure. None of these affect the pack's methodology, which is the actual product — but they affect what a user receives when installing.

The methodology has surface area beyond what is currently runnable (Phase 5, delegation chain). This is a future-feature observation, not a defect — the existing scope is internally coherent.

**Confidence Assessment.** Medium-high. I read the skill, all three agents, the architecture document, and ~75% of `process.md` (Phases 1-5 in pieces, plus execution notes including the delegation chain and the process manifest). I did not read every line of `config-template.md` or `config.md` — they are reference material for the user, not part of the runtime path. Confidence on Findings 1-3 (Major) is high — these are filesystem-verifiable. Confidence on Findings 5 (Phase 5 gap) is high — the gap is plainly visible in `process.md:789-873` vs `SKILL.md:464-501`. Confidence on Polish items 11-15 is lower — these are subjective design suggestions.

**Information Gaps.** I did not run live behavioral tests via subagent dispatch (per task constraints — read-only review). The three behavioural traces in section 4 are reasoning-only walkthroughs of the skill's structural protections, not empirical confirmations.

**Caveats.** Per the rubric (`SKILL.md:25`), "Adding a `tools:` key restricts the agent to that exact set" — this pack's three agents all declare `tools:` and the restrictions are load-bearing safety mechanisms (not maintenance burden). My Pass rating on dimension 4 (Agent design) is contingent on this being a deliberate design choice; if the user intended these agents to inherit parent context, the restrictions should be re-audited. The agent bodies make clear (`persona-reader.md:36-40`, `ARCHITECTURE.md:19`) that the restrictions are intentional — so this caveat is documentary only.

**Risk Assessment.** The Major findings are filesystem-and-metadata defects that affect what users *receive* from the marketplace, not what the methodology *does*. None is severity-Critical because the underlying skill, agents, and methodology are intact and runnable in the skillpacks-repo developer context. A user installing via the marketplace catalog without cloning the repo gets a degraded experience — the skill works, but the three convenience wrappers don't ship. Fix difficulty is low: relocate three files into `plugins/muna-panel-review/commands/`, add minimal frontmatter, adjust the marketplace description. The empty `using-panel-review/` directory is a one-command cleanup. Phase-5 / delegation-chain executable surfaces are higher-effort and appropriately deferred to v0.4.

**Comparison to peer packs.** Against `lyra-creative-writing` (already reviewed in `/home/john/skillpacks/reviews/lyra-creative-writing.md`), which has 22 reference sheets and 11 agents, this pack is structurally simpler (1 workflow skill + 3 agents) but methodologically denser per-file. `lyra-creative-writing` ships commands at `plugins/lyra-creative-writing/commands/` (3 files); `muna-panel-review` does not — this is the asymmetry behind Major finding #1. The cross-pack composition claim ("composes with muna-panel-review for beta-reader simulation" in the lyra marketplace description) is supported by the persona-designer affordance, but no skill or sheet in either pack documents the integration pattern. Polish opportunity: a short "Composing with `lyra-creative-writing`" section in `process.md` or `ARCHITECTURE.md` would close that loop.

**Final verdict.** v0.3.2 is a methodologically mature pack with structural-packaging defects. The three Major findings are fixable in a v0.3.3 patch (no methodology change, only file relocation and metadata reconciliation). Phase-5 / delegation-chain executable surfaces are a v0.4 conversation. Recommend proceeding with the Priority-1 fixes before any methodology expansion — the gap between marketplace-claim and marketplace-reality is the first thing a new user encounters.
