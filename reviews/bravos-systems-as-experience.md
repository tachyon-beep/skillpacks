# Review: bravos-systems-as-experience
**Version:** 1.1.5  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

Pack root: `/home/john/skillpacks/plugins/bravos-systems-as-experience/`

## 1. Inventory

### Plugin metadata
- `.claude-plugin/plugin.json` — name `bravos-systems-as-experience`, version `1.1.5`, description `"Emergent gameplay and systemic game design - 9 skills, 3 commands, 2 agents"`.
- Marketplace registration: present in `.claude-plugin/marketplace.json` (entry shows `"description": "Emergent gameplay and systemic game design - 9 skills"`). Source path resolves; keywords `bravos`, `systems`, `experience`.
- Repo-level `CLAUDE.md` calls this pack "10 skills" in one place and the marketplace says "9 skills"; the plugin.json says "9 skills, 3 commands, 2 agents". These three are inconsistent (see Findings).

### Skills (1 router + 8 specialist reference sheets + 1 routing-scenarios sheet)
The only YAML-frontmatter skill is the router; the 8 specialists are reference sheets in the router's directory.

| File | Lines | Role |
|------|-------|------|
| `skills/using-systems-as-experience/SKILL.md` | 301 | Router (frontmatter present) |
| `skills/using-systems-as-experience/emergent-gameplay-design.md` | 2481 | Wave-1 foundational reference sheet |
| `skills/using-systems-as-experience/sandbox-design-patterns.md` | 2428 | Wave-1 reference sheet |
| `skills/using-systems-as-experience/strategic-depth-from-systems.md` | 1574 | Wave-1 reference sheet |
| `skills/using-systems-as-experience/optimization-as-play.md` | 3085 | Wave-2 reference sheet |
| `skills/using-systems-as-experience/discovery-through-experimentation.md` | 1801 | Wave-2 reference sheet |
| `skills/using-systems-as-experience/player-driven-narratives.md` | 2665 | Wave-2 reference sheet |
| `skills/using-systems-as-experience/modding-and-extensibility.md` | 3114 | Wave-3 reference sheet |
| `skills/using-systems-as-experience/community-meta-gaming.md` | 1522 | Wave-3 reference sheet |
| `skills/using-systems-as-experience/routing-scenarios.md` | 137 | 22 game-type routing examples |

### Commands (3 in-pack; 1 repo-root wrapper)
| File | Frontmatter shape |
|------|-------|
| `commands/design-emergence.md` (246 ln) | `description`, `allowed-tools` (quoted JSON array), `argument-hint`. |
| `commands/design-sandbox.md` (307 ln) | Same shape. |
| `commands/analyze-system-interactions.md` (281 ln) | Same shape. |
| `.claude/commands/systems-as-experience.md` (581 ln) | Repo-root slash-command wrapper for the router skill — PRESENT. |

### Agents (2)
| File | Frontmatter | Notes |
|------|-------------|-------|
| `agents/emergence-designer.md` (238 ln) | `description` (ends "Follows SME Agent Protocol with confidence/risk assessment."), `model: opus`. Body cites `meta-sme-protocol:sme-agent-protocol` and requires Confidence/Risk/Information Gaps/Caveats sections (line 10). |
| `agents/sandbox-architect.md` (275 ln) | `description` (ends "Follows SME Agent Protocol with confidence/risk assessment."), `model: sonnet`. Body cites SME protocol (line 10). |

No `tools:` keys on either agent — correct (inheritance). No `hooks/` directory; none expected for a game-design pack.

## 2. Domain & Coverage

### Intended scope
"Emergent gameplay and systemic game design" — designing games where systems are the content rather than the substrate for authored content. The pack philosophy (router lines 19-47) is explicit and well-articulated: orthogonal mechanics, interaction matrices, feedback loops, cascades, systemic solutions; with a hybrid acknowledgement for blended games.

### Coverage map vs. inventory

**Foundational (systems-as-experience core):**
- Orthogonal mechanic design — covered (`emergent-gameplay-design.md`, Core Concept #1, with multiplication test and BotW worked example).
- Interaction matrices — covered (Core Concept #2; Noita 30×30 example).
- Feedback loops (positive/negative balance) — covered (Core Concept #3; Dwarf Fortress ecosystem example).
- Cascade chains — covered (Core Concept #4; dampening guidance and chain-length distribution).
- Systemic solutions / multiple paths — covered (Core Concept #5; Deus Ex worked example).
- Emergence testing methodology — covered (Core Concept #6; six tests with scoring).

**Application domains:**
- Sandbox / constraint paradox — `sandbox-design-patterns.md`.
- Strategic / build depth — `strategic-depth-from-systems.md`.
- Optimization-as-play (factory genre) — `optimization-as-play.md`.
- Discovery-driven gameplay — `discovery-through-experimentation.md`.
- Player-driven narrative / emergent storytelling — `player-driven-narratives.md`.

**Ecosystem / longevity:**
- Modding and extensibility — `modding-and-extensibility.md`.
- Community meta-gaming / theorycrafting / speedrunning — `community-meta-gaming.md`.

**Routing:**
- Decision tree by game type — router SKILL.md lines 95-117.
- 22 worked routing scenarios — `routing-scenarios.md`.
- 5 multi-skill workflows — router lines 149-181.

### Gaps
- **No procedural-content-generation sheet.** PCG is closely adjacent (Minecraft world gen, DF history gen, Spelunky, Caves of Qud) and arguably foundational to several covered domains, but it is treated only obliquely (modding sheet has it in its content, exploration sheet has hints). Minor gap — possibly intentional (PCG is its own discipline) but the router does not say so.
- **No tooling/telemetry sheet for measuring emergence in shipped products.** The methodology sheet (Core Concept #6 inside emergent-gameplay-design) gives six tests but they assume manual playtest observation; there is no guidance for instrumented logging, replay analysis, or live-ops detection of dominant strategies. Minor.
- **No cross-link to `bravos-simulation-tactics` for game-feel/balance tuning beyond a one-liner in the router (lines 240-243).** The router mentions it; no specialist sheet exercises the boundary. Minor.

### Research currency
Game design is stable enough that the techniques (orthogonality, interaction matrices, feedback loops) have not moved meaningfully. Examples (BotW, Factorio, Path of Exile, Rimworld, EVE, Noita, Outer Wilds) remain canonical. No research-currency flag.

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Rating | Evidence |
|---|-----------|--------|----------|
| 1 | Domain coverage breadth | Pass | Foundational + 5 applications + 2 ecosystem = the canonical systems-as-experience curriculum. |
| 2 | Domain coverage depth | Pass | `emergent-gameplay-design.md` at 2481 ln has six core concepts each with anti-patterns, worked examples, and process. Other specialists run 1500-3100 ln. |
| 3 | Router quality | Minor issues | Router exists with decision tree, quick-reference table, 5 workflows, 5 pitfalls, and explicit success criteria. **However:** router-internal naming says "8 core skills" while plugin.json says "9 skills" and the file is one of 10 .md files. Reference-sheet path guidance is correct (router lines 53-62). |
| 4 | Commands | Pass | 3 commands (`design-emergence`, `design-sandbox`, `analyze-system-interactions`) with correct frontmatter shape — quoted JSON-array `allowed-tools`, quoted `argument-hint`. Each is user-invocable, has clear scope-boundaries section, and points to its sibling for cases outside scope. |
| 5 | Agents | Pass | Two agents, both SME-protocol compliant: description ends with the required phrase, body has the `**Protocol**:` line citing `meta-sme-protocol:sme-agent-protocol` and the four sections, and each declares scope-include/scope-exclude lists. Model selection (opus for emergence, sonnet for sandbox) is defensible — emergence design is the more synthesis-heavy task. |
| 6 | Slash-command wrapper | Pass | `.claude/commands/systems-as-experience.md` exists (581 ln) and broadly matches the router's content. **One inconsistency:** the wrapper's pack-structure footer (lines 564-575) still references `using-systems-as-experience` as the router but lists 8 specialist directories whereas the router has 9 listed entries including `routing-scenarios.md`. |
| 7 | Metadata accuracy | Major issue | Plugin description / marketplace description / repo CLAUDE.md skill-count are inconsistent (see Findings #M1). |
| 8 | Internal consistency | Minor issues | Wrapper-vs-router skill count drift; router lists "8 Core Skills" but the catalog at the bottom (lines 285-298) numbers items 1-9 including `routing-scenarios.md`; one duplicate "Pitfalls" block exists between router and wrapper. |

**Overall:** **Pass with Major issue on metadata accuracy and several Minor consistency / drift issues.** This pack is content-rich, well-structured, and pedagogically coherent. The body work is strong. The defects are in coordinating metadata between the plugin manifest, the router, the wrapper, and the marketplace — all things that affect discovery and trust but not correctness of guidance.

## 4. Behavioral Tests

Tests run statically against the artefacts (no live subagent dispatch); each test states the scenario the artefact would face and the predicted response based on what the artefact says.

### Test S1: Router discoverability (description-based activation)
**Scenario:** "I'm building a factory game where players design conveyor layouts; what should I be thinking about?" — fresh context, no other game-design skills loaded.

**Trigger surface:** Router description (`skills/using-systems-as-experience/SKILL.md:3`): `"Router for systems-as-experience - emergence, sandbox, optimization, discovery, narrative, modding"`.

**Predicted behaviour:** Description does NOT use the dominant repo convention "Use when...". Keyword-style description ("Router for X - keyword, keyword") may not trip activation against the user phrasing "factory game / conveyor layouts" reliably. Once activated, the decision tree (router lines 95-117) routes Factory cleanly to `emergent-gameplay-design` + `optimization-as-play` + sandbox patterns, with Workflow 2 (lines 156-161) giving a precise 8-12 hour plan.

**Result:** **Activation = Fix-needed (Minor)**; once activated, guidance = Pass.

### Test S2: Pressure test — "just give me the answer"
**Scenario:** "I have 30 minutes, design an emergent combat system for me. Skip the principles."

**Predicted behaviour:** Router's Pitfall #5 (lines 228-231) explicitly addresses this: "Each skill requires 2-4 hours minimum. Budget 8-15 hours for multi-skill projects." `emergent-gameplay-design.md` line 23 says "ALWAYS use this skill BEFORE implementing emergent systems. Retrofitting emergence into scripted systems is nearly impossible." Quick Start (router lines 187-191) gives a "Minimal Viable Emergence (4h)" sub-flow. The pack pushes back on the time pressure cleanly.

**Result:** Pass.

### Test S3: Edge case — hybrid game with strong authored narrative
**Scenario:** "I'm making a story-driven RPG with one open-world act. Do I use this pack or not?"

**Predicted behaviour:** Router lines 35-46 ("When This Philosophy Applies / Don't apply") explicitly call out: not for "Authored narrative is primary experience" but flag "Hybrid approach: Most games blend both (BotW has authored Ganon encounter + emergent physics sandbox)". Routing-scenarios #19 (Skyrim-style open RPG) covers exactly this case. Pitfall #4 (router lines 222-226) names the risk of trying to make EVERYTHING systemic.

**Result:** Pass.

### Test S4: Anti-pattern recognition
**Scenario:** "I have a fire spell, an ice spell, a lightning spell, and a poison spell. Is this emergent design?"

**Predicted behaviour:** `emergent-gameplay-design.md` lines 65-77 (Non-Orthogonal Bad Example) is **literally this case** verbatim: "All four do the same thing (damage). Possibility count: 4. Result: Redundant complexity, no emergence." The corrective example follows immediately.

**Result:** Pass — exemplary direct-match.

### Test S5: Exploit-handling dilemma
**Scenario:** "Players discovered they can stack GLOO foam to climb anywhere. Patch or keep?"

**Predicted behaviour:** `emergent-gameplay-design.md` Decision Framework #3 (lines 970-1110) is a six-factor decision table with Prey 2017's GLOO Cannon as the worked example (lines 1056-1067). Specific guidance: skill required (yes) + counterplay (limited ammo / enemy interruption) + doesn't trivialise all challenges = keep as feature.

**Result:** Pass — direct named case.

### Test C1: Command `/design-emergence` invocation
**Scenario:** User invokes `/design-emergence factorio-style production game`.

**Predicted behaviour:** Argument-hint `"[game_or_system_context]"` accepts free-form context. Frontmatter `allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]` — note no `Skill` tool in the list, which means the command cannot dispatch back to specialist skills if it needs to. This is acceptable for a "produce a design document" command but constrains its ability to delegate. Body provides structured information-gathering, orthogonality test, matrix template, cascade-chain template, feedback-loop template, systemic-solution template, output format, and anti-patterns.

**Result:** Pass. (Minor observation: no `Skill` in `allowed-tools` is a deliberate choice and matches the command's "produce output" idiom; documenting that choice would be polish.)

### Test A1: Agent `emergence-designer` scope discipline
**Scenario:** Coordinator asks `emergence-designer` to "analyse what mechanics this existing codebase has."

**Predicted behaviour:** Agent's Scope Boundaries (lines 226-237) explicitly say "I do NOT analyse existing systems (use /analyze-system-interactions)." Activation example #4 (lines 33-36) gives the exact negative case: "Analyse existing game systems → Do NOT activate". Hand-off is clean.

**Result:** Pass.

### Test A2: Agent SME-protocol output discipline
**Scenario:** `emergence-designer` produces a design without Confidence / Risk / Information Gaps / Caveats sections.

**Predicted behaviour:** Body line 10 says output MUST include those four sections. Output Format template (lines 116-181) does not itself include them — they would need to be appended by the agent. **This is a Minor gap:** the SME protocol requirement is stated but the output template doesn't enforce it, so the agent could plausibly produce the documented Output Format and miss the four sections.

**Result:** Fix-needed (Minor) — same risk on `sandbox-architect.md`.

### Test W1: Repo wrapper drift
**Scenario:** User runs `/systems-as-experience` and sees pack structure that lists 8 specialists, then they read the router which lists 9 (8 + routing-scenarios).

**Predicted behaviour:** Mild confusion, no breakage. The drift is real (`.claude/commands/systems-as-experience.md:564-575` vs. router catalog at `SKILL.md:285-298`) but cosmetic.

**Result:** Fix-needed (Minor) — drift between wrapper and router.

## 5. Findings (Critical / Major / Minor / Polish)

### Critical
None.

### Major

**M1. Skill-count metadata is internally inconsistent across plugin manifest, marketplace catalog, router skill, and repo CLAUDE.md.**
- `plugin.json:4` — `"9 skills, 3 commands, 2 agents"`.
- `.claude-plugin/marketplace.json` entry — `"9 skills"`.
- `skills/using-systems-as-experience/SKILL.md:65` — `"Pack Overview: 8 Core Skills"` (the 8 specialist sheets).
- `skills/using-systems-as-experience/SKILL.md:285-298` — numbered catalog 1-9 including `routing-scenarios.md` as item 9.
- Filesystem: 10 .md files under the router directory (`SKILL.md` + 8 specialists + `routing-scenarios.md`).
- Repo `CLAUDE.md` (in instructions) — mentions "Bravos (Game Dev): 2 plugins, 20 skills" and "10 skills" elsewhere; pack-level descriptor varies.

The pack contains **one** Claude-Code skill (the router) plus eight specialist reference sheets and one routing-examples reference sheet. The "9 skills" descriptor conflates skill-with-frontmatter and reference-sheet. This is a marketplace-wide convention issue (other packs do the same), but the internal disagreement (8 vs. 9) is specific to this pack and worth fixing here.

Recommended phrasing: `"Emergent gameplay and systemic game design - 1 router + 8 specialist reference sheets + 22 routing scenarios, 3 commands, 2 agents"` — or, if the marketplace convention is to count reference sheets as skills, align the router header (`"Pack Overview: 9 Core Reference Sheets"`).

### Minor

**Mi1. Router description does not use the "Use when..." convention.**
- `skills/using-systems-as-experience/SKILL.md:3` — `description: Router for systems-as-experience - emergence, sandbox, optimization, discovery, narrative, modding`.
- Pack-maintenance convention (per `analyzing-pack-domain.md` and the marketplace-wide observation) is `Use when...` for discoverability. Keyword-list descriptions trigger less reliably against natural-language user queries.
- Suggested: `description: Use when designing or critiquing games where systems are the content - emergence, sandbox, optimization, discovery, player-driven narrative, modding, or community meta-gaming. Routes to 8 specialist sheets and 22 game-type scenarios.`

**Mi2. SME-protocol output sections not enforced in agent Output Format templates.**
- `agents/emergence-designer.md:116-181` — Output Format does not include Confidence / Risk / Information Gaps / Caveats sections, although line 10 says the output MUST include them.
- Same risk in `agents/sandbox-architect.md:118-200`.
- Fix: Append the four-section block to the Output Format template in each agent.

**Mi3. Wrapper-vs-router drift on pack structure listing.**
- `.claude/commands/systems-as-experience.md:564-575` lists 8 specialist directories.
- `skills/using-systems-as-experience/SKILL.md:285-298` lists 9 items (8 specialists + routing-scenarios).
- Either is defensible alone; the disagreement is the issue.

**Mi4. Router uses "8 Core Skills" header (line 65) but only 8 specialists exist outside the router itself.**
- The header itself is internally consistent if you exclude the router and routing-scenarios from the count, but it disagrees with plugin.json's "9 skills" and the bottom catalog numbering. Pick one convention and align.

**Mi5. Pack-internal cross-link to `bravos-simulation-tactics` is one-line and asymmetric.**
- Router lines 240-243 mention simulation-tactics, game-theory, game-feel. No specialist sheet exercises the boundary or shows a hand-off scenario. Since `bravos-simulation-tactics` is a sibling pack (per repo CLAUDE.md), a "When this pack ends and simulation-tactics begins" paragraph in the router would tighten the boundary.

**Mi6. No `Skill` tool in command `allowed-tools`.**
- All three commands omit `Skill` from `allowed-tools`. This is consistent and probably intentional (they are output-generators, not routers) but the marketplace convention (per `using-skillpack-maintenance.md:154`) calls out that router commands typically include `Skill`. These commands are not routers, so this is fine — worth noting as a deliberate design choice.

### Polish

**P1. Routing-scenarios.md has no level-2 heading hierarchy beyond "Scenario N: Title".** 22 scenarios in a flat list at 137 lines is tolerable but a table-of-contents or grouping by category (Sandbox / Competitive / Simulation / Exploration / Social) would speed lookup.

**P2. Pitfalls block duplicated between router (lines 209-231) and wrapper (lines 472-507).** The wrapper is a copy of the router's body; if they diverge, they diverge silently. Consider trimming the wrapper to a short "see the router skill" pointer plus the router catalog.

**P3. The catalogue at router lines 285-298 numbers `routing-scenarios.md` as item 9 alongside the 8 actual skills.** A separate "Auxiliary references" subsection would read cleaner.

**P4. `emergent-gameplay-design.md` is 2481 ln — at the upper end of usable reference-sheet length.** No action needed (the content density is high and the section structure is clear), but a future split into "foundations" and "implementation patterns" subsheets would reduce context cost when only one half is needed.

**P5. Plugin description omits keyword "game design"** — would help marketplace search. `"Emergent gameplay and systemic game design"` is fine; consider `"Game design for emergent gameplay - systems as content, orthogonal mechanics, interaction matrices"`.

## 6. Recommended Actions

In rough priority order (no edits made):

1. **Resolve M1 (metadata consistency).** Decide whether reference sheets count as skills and align all four surfaces: `plugin.json`, marketplace entry, router's "Pack Overview" header, router's bottom catalog. Suggested: count the router as 1 skill, document the 8 reference sheets + routing-scenarios as content, and bump plugin description accordingly. Patch bump (1.1.5 → 1.1.6).

2. **Resolve Mi1 (router description).** Rewrite to "Use when..." convention so description-based discovery activates against natural-language user phrasing. Patch bump.

3. **Resolve Mi2 (SME output enforcement).** Append the four-section block to both agents' Output Format templates. Patch bump.

4. **Resolve Mi3 / Mi4 (wrapper-vs-router drift).** Pick a canonical pack-structure listing and propagate. Trim the wrapper to a pointer (per P2) to prevent future drift.

5. **Consider Mi5.** Add a short "When this pack hands off to simulation-tactics / yzmir / lyra" paragraph in the router.

6. **Consider P3 / P4 / P5 polish** in a future minor-bump pass.

7. **No new skills, commands, or agents are needed.** Domain coverage is comprehensive. If `bravos-systems-as-experience` is going to grow, the strongest candidates are (a) procedural content generation as its own sheet, (b) telemetry / live-ops emergence detection, but neither is on the critical path — both are independent enough to live in a sibling pack.

## 7. Reviewer Notes

- **No edits performed.** Report-only as instructed.
- **No subagent dispatch** for behavioural testing — tests are static predictions based on the artefacts. Test S1 (description-based activation) and Test A2 (SME output enforcement) would benefit from a live subagent dispatch to confirm; static analysis suggests they will fail-or-degrade as documented.
- **What I did NOT read in full:** the full bodies of `sandbox-design-patterns.md`, `strategic-depth-from-systems.md`, `optimization-as-play.md`, `discovery-through-experimentation.md`, `player-driven-narratives.md`, `modding-and-extensibility.md`, `community-meta-gaming.md` — I read each file's top section / opening 12 lines and confirmed they have a clear preamble that matches their advertised topic. Total specialist-sheet content is ~17.5K lines; a full read pass would take significantly more context. Findings about specialist sheets are therefore at the structural / front-matter level, not the per-section level.
- **What I DID read in detail:** the router skill (whole), `emergent-gameplay-design.md` (first 1541 lines / 62%), both agents (whole), `design-emergence` command (whole), `routing-scenarios.md` (whole), `.claude/commands/systems-as-experience.md` (head + tail), plugin and marketplace metadata, and the four maintenance reference sheets used as rubric.
- **Confidence:** High on M1, Mi1, Mi2, Mi3, Mi4 (all directly observed in the artefacts). Medium on the specialist-sheet depth ratings (sampled rather than fully read). High on the absence of a hooks/ directory and the agent SME-protocol compliance.
- **No critical blockers.** This is a healthy pack with a documentation-coordination problem and a discoverability nit; the body of work is genuinely strong.
