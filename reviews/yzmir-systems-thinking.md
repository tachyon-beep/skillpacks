# Review: yzmir-systems-thinking
**Version:** 1.1.4  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

## 1. Inventory

**Plugin metadata** (`plugins/yzmir-systems-thinking/.claude-plugin/plugin.json:1-18`):
- name: `yzmir-systems-thinking`
- version: `1.1.4`
- description: "Systems thinking methodology - patterns, leverage points, archetypes, modeling, visualization - 6 skills, 3 commands, 2 agents"
- license: CC-BY-SA-4.0
- author: tachyon-beep
- keywords: `yzmir`, `systems-thinking`, `systems`, `modeling`

**Marketplace registration** (`.claude-plugin/marketplace.json:620-621`):
- Registered. `"source": "./plugins/yzmir-systems-thinking"` — directory exists.

**Router skill** (1):
| File | Path | Lines |
|------|------|------|
| `using-systems-thinking/SKILL.md` | `plugins/yzmir-systems-thinking/skills/using-systems-thinking/SKILL.md` | 488 |

**Reference sheets** (6, all colocated beside router):
| Sheet | Path (same dir as SKILL.md) | Lines |
|-------|------|------|
| `recognizing-system-patterns.md` | same dir | 226 |
| `systems-archetypes-reference.md` | same dir | 918 |
| `leverage-points-mastery.md` | same dir | 502 |
| `stocks-and-flows-modeling.md` | same dir | 1251 |
| `causal-loop-diagramming.md` | same dir | 781 |
| `behavior-over-time-graphs.md` | same dir | 843 |

Total reference content: ~4521 lines. Router SKILL.md description claims "6 skills" — semantically these are reference sheets attached to one router skill, not six independent SKILL.md skills. The plugin.json description repeats the same "6 skills" framing. Marketplace convention treats `*.md` siblings of a router as reference sheets (no frontmatter, content-only) — which they are here (`recognizing-system-patterns.md:1-2` shows no YAML, just an H1 — matches `systems-archetypes-reference.md`, `stocks-and-flows-modeling.md`, etc.).

**Commands** (3):
| Command file | Frontmatter description | argument-hint |
|---|---|---|
| `commands/analyze-system.md` | "Initiate systematic systems analysis with pattern recognition, archetype matching, and intervention design" | `"[problem_description_or_domain]"` |
| `commands/find-leverage-points.md` | "Identify high-leverage intervention points using Meadows' 12-level hierarchy" | `"[proposed_solution_or_problem]"` |
| `commands/map-dynamics.md` | "Map causal loops, build stock-flow models, and create behavior-over-time graphs" | `"[system_description_or_problem]"` |

All three commands declare `allowed-tools` as a quoted JSON array. `analyze-system.md:3` includes `"Task"` (subagent dispatch); `find-leverage-points.md:3` and `map-dynamics.md:3` do not include `"Task"` — likely correct since they are direct-execution narrower commands. None include `"Skill"` — commands do not currently dispatch back to the router skill, which is acceptable for these self-contained command bodies.

**Agents** (2):
| Agent | Model | SME-protocol compliance |
|---|---|---|
| `agents/pattern-recognizer.md` | `sonnet` | YES — description ends with "Follows SME Agent Protocol with confidence/risk assessment." (line 2); body cites `meta-sme-protocol:sme-agent-protocol` at line 10 and requires Confidence/Risk/Information Gaps/Caveats sections |
| `agents/leverage-analyst.md` | `opus` | YES — description ends with SME phrase (line 2); body cites protocol at line 10 |

Neither agent declares `tools:` — correct, inherits parent context per marketplace convention.

**Slash-command wrapper**: `/home/john/skillpacks/.claude/commands/systems-thinking.md` — **EXISTS** (444 lines). Body mirrors router SKILL.md content with the activation-fence sections removed; no frontmatter, leading H1 `# Using Systems-Thinking (Meta-Skill Router)` — matches the marketplace wrapper pattern (compare `system-architect.md`, `python-engineering.md`).

**Hooks**: none.

## 2. Domain & Coverage

### Boundary Confirmation

The pack's stated boundary against sibling packs is articulated in `using-systems-thinking/SKILL.md:407-440`:

- **`yzmir-simulation-foundations`** — implementation of numerical simulation based on a systems model. systems-thinking does design; simulation-foundations does numerics. Clear handoff.
- **`axiom-system-architect`** — code/architecture structure. Edge case explicitly called out (`SKILL.md:420`): "Software architecture CAN have systems dynamics (technical debt accumulation, team coordination). Use **both**."
- **`yzmir-deep-rl`** — environment analysis before agent design. Handoff documented (`SKILL.md:436-439`).

The "When NOT to Use" table (`SKILL.md:411-419`) is well-formed: well-understood optimization → standard profiling; one-time decisions → decision analysis; data analysis → data science; legal → ordis security-architect; UX → lyra ux-designer; code architecture → axiom system-architect.

### Coverage Map

Systems-thinking is a **stable domain** (Meadows / Senge / Forrester canon, mid-1990s through 2008). No research-currency risk; the body of knowledge is mature.

**Foundational** (must be covered):
- Stocks, flows, accumulation — present in `stocks-and-flows-modeling.md` (1251 lines, comprehensive)
- Reinforcing vs balancing loops — present in `recognizing-system-patterns.md`, `causal-loop-diagramming.md`
- Delays and time constants — present in `stocks-and-flows-modeling.md`, marked on CLDs per `causal-loop-diagramming.md` step 5
- S-curves and limits to growth — present (`recognizing-system-patterns.md`, archetype)
- Polarity logic with double-test — present (`causal-loop-diagramming.md` step 3, `map-dynamics.md:43-47`)

**Core techniques** (should be covered):
- Causal loop diagrams with construction process — present (6-step in `causal-loop-diagramming.md`)
- Stock-flow models with equilibrium analysis — present
- Behavior-over-time graphs with 7-step construction, 70-80% scale rule — present (`behavior-over-time-graphs.md`)
- Systems archetypes (Meadows / Senge classics) — `systems-archetypes-reference.md` covers 10 archetypes (router SKILL.md:85 claims "10 classic archetypes"; `pattern-recognizer.md:67-80` table shows 9 entries; `analyze-system.md:58-71` table shows 10 — minor inconsistency, see findings)
- Meadows' 12 leverage points — present in `leverage-points-mastery.md`, mirrored across `find-leverage-points.md` command and `leverage-analyst.md` agent

**Advanced** (nice to have):
- Multi-stock dynamics — covered in `stocks-and-flows-modeling.md` and Scenario 6 routing
- Multi-scenario BOT comparison — covered (`behavior-over-time-graphs.md`)
- Dominant-loop analysis — covered (`pattern-recognizer.md:103-107`)
- Group Model Building (GMB), causal mapping with stakeholders — **not explicitly covered**, but probably out of scope for a methodology pack vs. a facilitation pack
- Stella/Vensim/iThink tool mentions — **absent**, intentional (pack is methodology, not tooling)

**Coverage assessment**: Excellent. The pack covers the core Donella Meadows curriculum end-to-end: pattern recognition → archetype matching → CLDs → stock-flow → leverage points → BOT visualization. The six reference sheets compose a coherent learning path, with explicit prerequisites documented (`SKILL.md:354-368`).

### Faction fit

Yzmir = "mathematical, systematic, optimization-focused" (per `/home/john/skillpacks/CLAUDE.md`). Systems thinking is closer to **qualitative-then-quantitative** than pure optimization — but the pack does carry mathematical rigor (equilibrium calculation, time constants, D/R ratio, 95%-of-equilibrium = 3τ rule in `stocks-and-flows-modeling.md`). Acceptable fit.

## 3. Fitness Scorecard (8 dimensions)

| Dimension | Rating | Notes |
|---|---|---|
| **Discoverability** | Minor issue | Router SKILL.md description (`SKILL.md:3`) does not begin with "Use when..." — reads as a noun-phrase summary ("Router for systems thinking methodology - patterns, leverage points, archetypes..."). Marketplace convention strongly prefers "Use when..." trigger phrasing for description-based activation. Compare to e.g. `lyra-ux-designer` / `yzmir-llm-specialist` siblings that follow "Routes ... when ..." patterns. |
| **Routing/Coverage** | Pass | Six scenarios + four workflows + decision tree cover the realistic problem space. Cross-pack handoffs documented (simulation-foundations, system-architect, deep-rl). When-NOT table present. |
| **Structural integrity** | Pass | Reference sheets colocated beside router; no broken sheet links; explicit "how to access reference sheets" path note (`SKILL.md:22-37`); slash-command wrapper present and matches marketplace pattern. |
| **Component typing** | Pass | Skills auto-invoke (router), commands explicit-trigger, agents are SME-protocol-compliant specialists. No commands that should be skills or vice versa. |
| **Command quality** | Minor issue | `analyze-system.md:209-222` and `map-dynamics.md:233-241` contain `glob.glob("plugins/...")` Python snippets framed as "Cross-Pack Discovery" — this is a code pattern that does not run anywhere (commands don't auto-execute Python), and tests against a hardcoded marketplace directory layout. It's vestigial guidance rather than an executable check, and is potentially misleading to users invoking the command. |
| **Agent quality** | Pass | Both agents are SME-protocol compliant with verbatim "Confidence Assessment / Risk Assessment / Information Gaps / Caveats" requirement; `pattern-recognizer` = sonnet (correct — analytical-but-bounded task); `leverage-analyst` = opus (correct — synthesis across hierarchy levels with prerequisite/risk reasoning is harder). Scope boundaries declared on both ("I analyze X / I do NOT do Y"). |
| **Rationalization resistance** | Pass | Router SKILL.md `:374-385` rationalization table is strong (9 entries, each with reality + counter-guidance + red-flag column). The "Red Flags Checklist" at `:392-403` is concrete. Commands (`analyze-system.md:225-232`, `find-leverage-points.md:180-189`) and agents (`leverage-analyst.md:179-185`) carry parallel rationalization tables. Discipline is enforced top-to-bottom. |
| **Cross-references** | Minor issue | Archetype-count inconsistency: router `SKILL.md:86` says "10 classic archetypes", `pattern-recognizer.md:67-80` lists 9 in the signature-match table (missing Accidental Adversaries in that table though it appears in the quick-reference at `:151-163`), and `analyze-system.md:58-71` lists all 10. The reference sheet `systems-archetypes-reference.md` (918 lines) is the source of truth but I did not verify the exact count there. Inconsistency between agent's diagnostic table and command's table is the kind of drift the rubric flags. |

**Overall: Minor** — pack is structurally sound and content-rich; issues are polish-level (description phrasing, vestigial code snippets, one cross-reference drift between an agent and a command).

## 4. Behavioral Tests

This is a Stage-1-through-4 review; no live subagent dispatch tests were run. The behavioral concerns surfaced from structural inspection alone:

### Discoverability scenario (untested, inferred risk)

A fresh-context Claude asked "Our incident rate keeps going back up after every fix; we keep adding more on-call rotations and runbooks but it doesn't help" — would the router skill auto-activate?

The current description (`SKILL.md:3`): `"Router for systems thinking methodology - patterns, leverage points, archetypes, stocks-flows, causal loops, BOT graphs"`. This is a **what-it-is** description, not a **when-to-use-it** trigger. The dominant repo convention ("Use when [trigger condition] — [what the skill does]" per `reviewing-pack-structure.md`) would bias auto-activation more reliably. Risk: the model may not auto-load this router on a clearly-systems-thinking-flavored complaint.

The slash-command wrapper at `.claude/commands/systems-thinking.md` is a safety net — users can explicitly invoke `/systems-thinking`. But auto-activation is weakened.

### Pressure-resistance scenario (untested, inferred strength)

"This is urgent, just tell me what to do" applied to the router: the rationalization table at `SKILL.md:374-385` directly addresses "We don't have time for analysis" → "Route to stocks-and-flows-modeling - 15 min calculation vs wrong 6-month commitment". Both `analyze-system.md:225-232` and `leverage-analyst.md:179-185` carry parallel counter-guidance. **Likely passes** pressure tests — discipline is restated at every layer.

### Edge-case scenario (untested, inferred concern)

A user with a problem that has **no feedback dynamics** but resembles a systems problem (e.g. "scaling out a service" stated in systemic language). The When-NOT-to-Use table at `SKILL.md:411-419` does call this out ("Well-understood algorithm optimization — Standard profiling/optimization — No feedback dynamics"), and the edge-case at `:420` is good ("Software architecture CAN have systems dynamics... use both"). **Likely passes** — explicit decline paths exist.

### Agent activation scenarios (inferred from frontmatter)

`pattern-recognizer.md:20-40` has four activation `<example>` blocks including a negative example ("Calculate when we'll hit the limit" → Do NOT activate, use stock-flow modeling). `leverage-analyst.md:20-38` mirrors this pattern with a negative example ("What archetype is this?" → Do NOT activate). Both agents have explicit decline paths. **Likely passes**.

## 5. Findings

### Critical
None.

### Major
None.

### Minor

**M1. Router description does not follow "Use when..." convention** (`plugins/yzmir-systems-thinking/skills/using-systems-thinking/SKILL.md:3`).
- Current: `"Router for systems thinking methodology - patterns, leverage points, archetypes, stocks-flows, causal loops, BOT graphs"`
- Rubric (`reviewing-pack-structure.md`, `using-skillpack-maintenance/SKILL.md:133`): descriptions should start with "Use when..." for discoverability.
- Suggested form (illustrative — final wording is editorial): `"Use when facing complex problems with feedback loops, persistent failures despite fixes, unintended consequences, or delays between action and result — routes to systems thinking specialists (patterns, archetypes, leverage points, stock-flow modeling, CLDs, BOT graphs)."`
- Impact: weakens auto-activation on description-based discovery.

**M2. Vestigial "Cross-Pack Discovery" Python snippets in commands** (`commands/analyze-system.md:209-222`, `commands/map-dynamics.md:232-241`).
- The blocks use `glob.glob("plugins/yzmir-simulation-foundations/plugin.json")` / `glob.glob("plugins/axiom-system-architect/plugin.json")` as if the command were executing Python introspection of the local marketplace.
- Commands are markdown instructions to the model; they do not auto-run code. The snippets read as ritual rather than functional guidance and bake in a fragile hardcoded marketplace layout (the path `plugins/yzmir-simulation-foundations/plugin.json` is missing the `.claude-plugin/` segment used elsewhere in the repo).
- Suggested fix: replace with prose handoff: "If the user needs to implement a numerical simulation from a systems model, hand off to `yzmir-simulation-foundations`. If they need code/architecture critique, hand off to `axiom-system-architect`."

**M3. Archetype-count drift between components** (`pattern-recognizer.md:67-80` vs `analyze-system.md:58-71` vs router `SKILL.md:86`).
- Router claims "10 classic archetypes"; `analyze-system.md` table lists 10 (including Accidental Adversaries); `pattern-recognizer.md`'s signature-match table at `:67-80` lists 9 (Accidental Adversaries is missing there but appears in the quick-reference table at `:151-163`).
- Functionally low impact (the agent's quick-reference does carry the 10th), but causes a maintenance ambiguity if a reader trusts the symptom-table as canonical.
- Suggested fix: add "Accidental Adversaries — Mutual harm, good intentions" row to the symptom table at `pattern-recognizer.md:67-80`.

**M4. Plugin description double-counts** (`plugins/yzmir-systems-thinking/.claude-plugin/plugin.json:4`).
- Reads "6 skills, 3 commands, 2 agents". Strictly, the pack has **1 skill (router) + 6 reference sheets + 3 commands + 2 agents**. The "6 skills" framing in the plugin.json is a folk-count that matches how the router refers to them ("Pack Overview: 6 Core Skills") but does not match the marketplace's structural definition (reference sheets are not SKILL.md skills).
- Low impact (the meaning is clear), but creates the same ambiguity flagged in M3 for archetype counts. Consistent terminology would be "6 reference sheets" or "6 specialist skill areas".

### Polish

**P1. Router SKILL.md does not declare `allowed-tools`** (`skills/using-systems-thinking/SKILL.md:1-4`).
- Per `using-skillpack-maintenance/SKILL.md:138`, "`allowed-tools` on skills is rare in this marketplace; most SKILL.md files omit it." Current state matches convention — **no action required**. Noted only for completeness.

**P2. Workflow time-estimates are aspirational** (`SKILL.md:81, 88, 95, 102, 109, 116, 290-330`).
- Each specialist sheet declares "Time: 45-60 min" / "60-90 min" / etc.; workflows total "~2.5-3 hours" / "~4 hours". These are presented as if accurate calibrations — they are credible orders-of-magnitude but not verified by behavioral testing.
- No action required; flagged because a user planning a meeting around "Workflow 4: ~4 hours" may over-commit.

**P3. `analyze-system.md` "Phase 1-6" total may exceed argued time budget** (`commands/analyze-system.md:34-125`).
- Phases sum: 30 + 30 + 60 + 30 + 30 + 30 = 210 min (3.5h). The command also offers a "60-minute deadline" mode (`:196-205`). Add a one-line note that full Phase 1–6 is ~3.5h; current document has no top-level time estimate.

**P4. Slash-command wrapper has a blank first line** (`.claude/commands/systems-thinking.md:1`).
- The file starts with an empty line before `# Using Systems-Thinking (Meta-Skill Router)`. Compare sibling wrappers (`system-architect.md`, `python-engineering.md`) — they have the same leading blank line. Matches sibling convention; no action.

## 6. Recommended Actions

In order of return-on-effort:

1. **M1 — Rewrite router description with "Use when..." phrasing.** One-line edit to `SKILL.md:3`. Highest discoverability impact for lowest cost. Patch bump.

2. **M2 — Replace vestigial Python snippets with prose handoffs** in `analyze-system.md:209-222` and `map-dynamics.md:232-241`. Two commands, one block each. Patch bump.

3. **M3 — Add Accidental Adversaries row to `pattern-recognizer.md:67-80` symptom table.** One-line addition. Patch bump.

4. **M4 — Reconcile "6 skills" framing in plugin.json description.** Editorial; could be deferred to next minor bump. Suggested phrasing: "Systems thinking methodology — router + 6 reference sheets, 3 commands, 2 agents — patterns, leverage points, archetypes, stocks-flows, causal loops, BOT graphs." Same change applies to marketplace.json description if it carries the same phrasing.

5. **Optional — Add Stage 3 behavioral testing.** This review covered Stages 1–4 of structural review. To convert minor inferences into evidence, dispatch a fresh-context subagent test:
   - Scenario A (discoverability): "Our on-call rotation keeps growing but burnout is getting worse." Does the model auto-load `using-systems-thinking`?
   - Scenario B (pressure): Same problem + "This is urgent, just tell me what to add — more headcount or better tooling?" Does the router resist and route to archetype-matching?
   - Scenario C (negative): "Optimize this hot path in our Python service." Does the When-NOT table decline correctly?

Aggregate version impact: **patch bump (1.1.4 → 1.1.5)** for M1+M2+M3 corrections. M4 is editorial and could ride along or wait. No major bump warranted — the pack is structurally sound.

## 7. Reviewer Notes

- Review covers Stage 1 (inventory + domain) → Stage 4 (findings); Stage 5 (execution) skipped per instructions.
- No edits made to plugin files; this is a report-only review.
- The pack is exceptionally well-developed for its domain — 4521 lines of reference content plus discipline-enforcing rationalization tables at every layer (router, commands, agents). The Donella Meadows canon is honored with the 12-leverage-points hierarchy treated as load-bearing throughout (router, two commands, one agent — all carry the same hierarchy with consistent numbering).
- Both agents carry the full SME Agent Protocol contract with verbatim section names ("Confidence Assessment", "Risk Assessment", "Information Gaps", "Caveats") — they will produce parseable output that downstream callers can depend on.
- The slash-command wrapper at `.claude/commands/systems-thinking.md` exists and follows the marketplace pattern. The check called out in the task brief ("Missing = Major") **passes**.
- The marketplace catalog entry at `.claude-plugin/marketplace.json:620-621` correctly points at the existing directory; no orphaned registration.
- Risk-of-review: I did not open `systems-archetypes-reference.md` (918 lines) to verify it really enumerates all 10 archetypes referenced in the router. This is the load-bearing source for M3; if the reference sheet is canonical and has 10 entries, then M3 collapses to a one-row fix in the agent table. If it has 9 or 11, the inconsistency is deeper.
- Confidence in this review: **High** for structural / SME-protocol / wrapper / marketplace findings; **Medium** for behavioral inferences (which were not validated by live subagent runs).
