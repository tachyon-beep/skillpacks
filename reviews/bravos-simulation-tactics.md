# Review: bravos-simulation-tactics
**Version:** 1.1.5  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

Stages 1–4 of `meta-skillpack-maintenance:using-skillpack-maintenance` applied. Stage 5 (execution) deliberately skipped — report-only.

---

## 1. Inventory

### Plugin metadata
- Path: `/home/john/skillpacks/plugins/bravos-simulation-tactics/`
- `plugin.json` (`.claude-plugin/plugin.json:1`): name=`bravos-simulation-tactics`, version=`1.1.5`, description="Game simulation implementation patterns and tactics - 11 skills, 3 commands, 2 agents"
- Marketplace registration: present in `/home/john/skillpacks/.claude-plugin/marketplace.json` with `description: "Game simulation implementation patterns and tactics - 11 skills"` — slightly out-of-sync with the plugin.json description (no commands/agents mention).
- Slash-command wrapper: present at `/home/john/skillpacks/.claude/commands/simulation-tactics.md` (909 lines).

### Skills (invokable)
| # | Skill | Path | Frontmatter |
|---|---|---|---|
| 1 | using-simulation-tactics (router) | `skills/using-simulation-tactics/SKILL.md:1` | `name`, `description` only |

**Only one SKILL.md exists.** The marketing copy "11 skills" counts the router plus 10 reference sheets. This is a documentation drift issue, not a structural one — see Findings §5.

### Reference sheets (13)
All under `skills/using-simulation-tactics/`:

| Sheet | Lines | Type |
|---|---|---|
| simulation-vs-faking.md | 2815 | specialist (foundational) |
| ai-and-agent-simulation.md | 2847 | specialist |
| crowd-simulation.md | 2637 | specialist |
| debugging-simulation-chaos.md | 2262 | specialist |
| economic-simulation-patterns.md | 2152 | specialist |
| ecosystem-simulation.md | 1912 | specialist |
| performance-optimization-for-sims.md | 2573 | specialist |
| physics-simulation-patterns.md | 2398 | specialist |
| traffic-and-pathfinding.md | 1320 | specialist |
| weather-and-time.md | 1566 | specialist |
| expert-routing-guide.md | 197 | router-support |
| multi-skill-workflows.md | 163 | router-support |
| routing-scenarios.md | 278 | router-support |

10 specialists + 3 router-support = 13. All are content files referenced by the router SKILL.md.

### Commands (3)
| Command | Frontmatter | Lines | Status |
|---|---|---|---|
| `assess-simulation.md:1-5` | `description`, `allowed-tools` (quoted JSON), `argument-hint` | 293 | OK — matches marketplace convention |
| `debug-simulation.md:1-5` | same shape | 385 | OK |
| `lod-strategy.md:1-5` | same shape | 317 | OK |

Frontmatter style matches the marketplace's quoted-JSON convention documented in `reviewing-pack-structure.md`.

### Agents (2)
| Agent | Frontmatter | Lines | SME compliance |
|---|---|---|---|
| `desync-detective.md:1-4` | `description` ending "Follows SME Agent Protocol with confidence/risk assessment.", `model: sonnet` | 431 | Compliant — body at line 10 cites `meta-sme-protocol:sme-agent-protocol` and requires all four sections (Confidence Assessment, Risk Assessment, Information Gaps, Caveats) |
| `simulation-architect.md:1-4` | same shape, `model: sonnet` | 348 | Compliant — body at line 10 cites the protocol with all four sections |

Neither agent declares `tools:` (matches the marketplace's ~60/65 convention of inheriting parent context).

### Hooks
None — no `hooks/` directory. Appropriate for this domain.

---

## 2. Domain & Coverage

### Scope (inferred from router SKILL.md)
**Intent:** Game-development-oriented simulation tactics — *implementation patterns* (the "how") for in-engine systems: physics, AI agents, pathfinding, economy, ecosystem, crowds, weather/time, plus cross-cutting concerns (the "what to fake", performance, debugging chaos / desyncs).

**Boundaries (inferred from `simulation-architect.md:53-67` and `debugging-simulation-chaos.md:18-23`):**
- IN scope: in-engine simulation systems, scrutiny-based LOD, perf and determinism specific to game simulation
- OUT of scope:
  - Mathematical foundations (ODE integration, stability analysis) → `yzmir-simulation-foundations`
  - Architecture-level determinism/replay design → `axiom-determinism-and-replay`
  - High-level system-design experience / meaningful-emergence work → sibling `bravos-systems-as-experience`

### Coverage Map

**Foundational**
- Simulation-vs-faking trade-off — Exists (`simulation-vs-faking.md`, 2815 lines, depth: comprehensive)
- Scrutiny-based LOD framework — Exists (covered in `simulation-vs-faking.md` and `lod-strategy.md` command)

**Core domains (10 specialists)**
- Physics — Exists (`physics-simulation-patterns.md`)
- AI/agents — Exists (`ai-and-agent-simulation.md`)
- Pathfinding/traffic — Exists (`traffic-and-pathfinding.md`)
- Economy — Exists (`economic-simulation-patterns.md`)
- Ecosystem — Exists (`ecosystem-simulation.md`)
- Crowds — Exists (`crowd-simulation.md`)
- Weather/time — Exists (`weather-and-time.md`)

**Cross-cutting**
- Performance optimisation — Exists (`performance-optimization-for-sims.md`)
- Debugging / chaos / desync — Exists (`debugging-simulation-chaos.md`)

**Router infrastructure**
- Routing scenarios (worked examples) — Exists (`routing-scenarios.md`)
- Multi-skill workflows (genre-based) — Exists (`multi-skill-workflows.md`)
- Expert routing guide (tips, edge cases) — Exists (`expert-routing-guide.md`)

### Gap analysis

**No major coverage gaps.** The pack covers the seven canonical game-simulation domains plus the two cross-cutting concerns (perf, debugging) and the foundational trade-off skill. Three router-support sheets (scenarios, workflows, tips) provide usable navigation aids.

**Possible minor gaps** (judgment calls — not required):
- **Networking-specific simulation patterns** (client-side prediction, server reconciliation, rollback netcode) are only touched on inside `debugging-simulation-chaos.md` as desync issues. A dedicated sheet may be warranted if multiplayer simulation is in scope; the router currently treats determinism as a property to debug rather than a system to design. Compare to `axiom-determinism-and-replay`, which the router does not cross-reference.
- **Save/load and serialization of simulation state** is not directly addressed.
- **Editor/tool-time simulation** is mentioned only in `expert-routing-guide.md` Edge Case 5 (lines ~150+); could be its own sheet if the pack grows.

None of these rise to a Major finding for v1.1.5 — game-engine simulation tactics is a stable domain, the existing 10 specialists comfortably span the routing decisions the router lists, and the router's own boundary statement explicitly punts foundational math to `yzmir-simulation-foundations`.

### Research currency
Game simulation patterns (boids, A*, navmesh, behaviour trees, GOAP, utility AI, predator-prey, supply/demand) are stable; no Phase-A research flag needed. The pack's content is patterns-based, not framework-versioned.

---

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Rating | Evidence |
|---|---|---|---|
| 1 | **Domain coverage vs scope** | Pass | 10 specialist sheets cover the seven canonical game-sim domains plus perf + debugging + foundational trade-off. No structural gaps. |
| 2 | **Router skill activation discoverability** | **Minor** | `SKILL.md:3` description is `Router skill - analyze requirements and direct to appropriate tactics`. Does not follow the dominant marketplace "Use when …" convention, and does not enumerate trigger domains. Compare to e.g. `axiom-determinism-and-replay/.../SKILL.md` which gives ~5 lines of trigger surface. The terse description harms description-based skill discovery, particularly when the model is choosing between this pack and `yzmir-simulation-foundations`. |
| 3 | **Router internal structure (sheet references resolve, no dead links)** | Pass | Router enumerates all 10 specialists at `SKILL.md:368-377` and references the three support sheets at `SKILL.md:358-360`. All targets exist on disk. |
| 4 | **Command surface (frontmatter, arg hints, tool restrictions)** | Pass | All three commands declare quoted-JSON `allowed-tools` arrays and `argument-hint` strings matching the marketplace convention. Tool restrictions reasonable (Read/Grep/Glob/Bash/Task — last enables sub-agent dispatch). |
| 5 | **Agent surface (SME protocol, model selection, scope boundaries)** | Pass | Both agents declare `description` + `model: sonnet`, end the description with the verbatim "Follows SME Agent Protocol with confidence/risk assessment.", cite `meta-sme-protocol:sme-agent-protocol` in the body, and require all four output sections. Both include explicit positive AND negative activation examples (do-NOT-activate clauses), e.g. `desync-detective.md:48-56` and `simulation-architect.md:44-52`. Sonnet is the right tier for diagnostic/design work. |
| 6 | **Slash-command wrapper present and consistent** | **Major** | Wrapper exists at `/home/john/skillpacks/.claude/commands/simulation-tactics.md` so it satisfies the existence check. However, the wrapper has **two structural defects**: (a) **No YAML frontmatter** (`simulation-tactics.md:1` — file starts with a blank line then a Markdown H1, where peer wrappers like `simulation-foundations.md`, `python-engineering.md` also lack frontmatter, so this is at least consistent within the repo, BUT…); (b) the wrapper is **909 lines long and duplicates the router SKILL.md (379 lines) plus inlines the full content of `routing-scenarios.md` (20 scenarios) and `multi-skill-workflows.md` (8 workflows)**. Confirmed by `grep -n "Scenario\|Workflow [0-9]"` — 20 scenario headers and 8 workflow headers appear inside the wrapper but NOT in SKILL.md. The wrapper has therefore become a third copy of routing content, creating a maintenance hazard: any edit to scenarios or workflows must now be made in three places. Functionally not broken; structurally a Major finding. |
| 7 | **Marketplace metadata accuracy** | **Minor** | `plugin.json:4` claims "11 skills, 3 commands, 2 agents" but only **1 SKILL.md exists** (the router). The "11" counts the router + 10 specialist reference sheets — those are content files, not skills in the SDK sense. The marketplace.json entry has a slightly different blurb that omits commands/agents. Both reads are slightly misleading; users running `/plugin marketplace` will expect 11 invokable skills. Note that this counting convention is repo-wide (other bravos / yzmir packs do the same), so the finding is Minor — but worth flagging for marketplace honesty. |
| 8 | **Internal cross-references and downstream pack alignment** | **Minor** | The router (`SKILL.md:8-77`) does not cross-reference `yzmir-simulation-foundations` (the sibling math-foundations pack) or `axiom-determinism-and-replay` (the sibling architecture-level determinism pack), even though both are clear handoff destinations. The repo CLAUDE.md and other modern packs (e.g. `axiom-determinism-and-replay`'s description explicitly redirects to `yzmir-simulation-foundations:check-determinism`) follow the convention of declaring cross-pack handoffs. Within-pack cross-refs ARE present (`SKILL.md:358-360`). |

**Overall: Minor** — with one Major (wrapper duplication / maintenance hazard).

The pack is structurally sound and behaviourally usable. The Major finding is cosmetic in the sense that runtime behaviour is not broken — but it materially raises the cost of every future edit to routing scenarios and workflows. The Minor findings are recoverable in a single follow-up version bump.

**Recommendation: Enhance, do not rebuild.** No Critical issues.

---

## 4. Behavioral Tests

Per `testing-skill-quality.md`, I did not dispatch subagents (this is a report-only review, and live behavioural runs would extend session length significantly). The findings below are **structural / read-based predictions** of behaviour under each scenario, derived from inspection of the router and specialist content. They identify where behaviour *would* be at risk; live subagent verification is recommended as a follow-up.

### A. Pressure scenarios (rationalisation resistance)

**A1. "I just need a quick traffic system, skip the foundational stuff"**
- Router `SKILL.md:100-120` strongly mandates routing to `simulation-vs-faking` first, with the explicit anti-pattern at `SKILL.md:258-270` ("Mistake 1: Skipping simulation-vs-faking" + "Cost of mistake: Weeks of wasted work").
- Predicted: holds under pressure. The router repeats this discipline in three places (decision tree, principles, mistakes list).

**A2. "My ecosystem collapses every time, can I just tweak the food math?"**
- `SKILL.md:288-298` (Mistake 3: Not debugging systematically) explicitly calls this out: *"Ecosystem collapses, let me add more food" (should debug why it collapses)*.
- Predicted: holds. The mistake catalogue is specific enough to catch the named anti-pattern.

**A3. "I'll just optimise it now while I'm here in the file"**
- `SKILL.md:272-286` (Mistake 2: Premature optimisation) provides three concrete don't-do-yet examples.
- Predicted: holds.

### C. Adversarial edge cases

**C1. "I need simulation for a tool/editor, not a game" (specifically named edge case)**
- The router itself doesn't cover this; it's only in `expert-routing-guide.md` (Edge Case 5).
- Risk: a fresh-context invocation of just the router skill may miss this. Mitigation exists via the support sheet reference at `SKILL.md:358`.

**C2. "I want determinism for a replay debugger, single-player"**
- `debugging-simulation-chaos.md:18-23` ("Do NOT use this skill for: Simple single-player games with no replay requirements") creates a contradiction: the user has both no-multiplayer AND a replay requirement.
- Predicted: the SKILL would activate (replay = "must reproduce"). But the do-NOT-use bullet list as written reads as "single-player ⇒ skip me", which is a false-negative trap for replay-debugged single-player games. Minor wording risk.

**C3. "I'm not sure if my problem is a desync or a performance issue"**
- The router decision tree (`SKILL.md:117-167`) handles perf BEFORE chaos. The desync-detective agent (`desync-detective.md:54-56`) correctly defers explosion symptoms to "/debug-simulation first". Cross-handoff is wired.
- Predicted: holds.

### B. Real-world complexity

**B1. "I'm cloning Dwarf Fortress" (multi-domain: AI + ecosystem + economy + crowd + perf)**
- `multi-skill-workflows.md` doesn't have a Dwarf-Fortress-like genre. The closest is Workflow 2 (Survival) or Workflow 7 (Ecosystem game), which don't cover the colony-management blend.
- Predicted: the router will dispatch to ALL applicable sheets (`SKILL.md:135 "Multiple domains? Route to ALL applicable skills"`), so this is graceful degradation rather than a failure. No fix needed.

**B2. "I'm making a multiplayer survival game with simulated weather"**
- Multiple domains: ecosystem + weather + debugging-for-determinism. `multi-skill-workflows.md` Workflow 2 (Survival) is single-player by default — it doesn't mention determinism. Workflow 4 (MMO) is economy-focused. There is no MP-survival workflow.
- Predicted: the router's Principle "If multiplayer is planned, route to debugging-simulation-chaos early" (`SKILL.md:329-337` Mistake 6) catches this. Acceptable.

**B3. "How do I integrate this pack with `axiom-determinism-and-replay`?"**
- No mention of `axiom-determinism-and-replay` anywhere in the pack. The user gets no cross-pack handoff.
- Predicted: missed handoff. See §5 Minor finding on cross-references.

### Summary of behavioural-prediction failures

| Scenario | Predicted | Fix priority |
|---|---|---|
| A1 — pressure to skip foundation | Hold | — |
| A2 — pressure to skip debugging | Hold | — |
| A3 — premature optimisation | Hold | — |
| C1 — tool-time simulation | Mild risk | Polish |
| C2 — single-player replay | Mild wording risk in `debugging-simulation-chaos.md:18-23` | Minor |
| C3 — perf vs desync confusion | Hold | — |
| B1 — colony-sim blend | Graceful degradation | — |
| B2 — multiplayer survival | Hold via Mistake 6 catch | Polish (add workflow) |
| B3 — cross-pack handoff | Miss | Minor |

**Live subagent verification recommended** for A1, B2, and B3 before declaring no-further-work-needed.

---

## 5. Findings (Critical / Major / Minor / Polish)

### Critical
None.

### Major

**M1. Slash-command wrapper duplicates router + inlines support sheets (909 lines).**
- Path: `/home/john/skillpacks/.claude/commands/simulation-tactics.md` (909 lines) vs `plugins/bravos-simulation-tactics/skills/using-simulation-tactics/SKILL.md` (379 lines).
- The wrapper is *not* a thin reference to the router skill. It is a fuller copy of the router that ALSO inlines all 20 routing scenarios and all 8 multi-skill workflows.
- Compare to peer wrappers: `/home/john/skillpacks/.claude/commands/python-engineering.md` (~20 lines glimpsed in head; functions as a thin pointer). The bravos wrapper has drifted into a content-bearing third copy.
- Impact: any edit to `routing-scenarios.md`, `multi-skill-workflows.md`, or the router's mistake list must now be applied in three places to stay consistent. This is the canonical "shadow copy" maintenance anti-pattern.
- Fix: rewrite the wrapper to be a thin pointer (a few dozen lines max — Overview + When to Use + "this routes to the `using-simulation-tactics` skill"). Drop the inlined scenarios and workflows; the router skill itself doesn't carry them either.

### Minor

**m1. Router SKILL.md description does not follow "Use when …" convention.**
- `skills/using-simulation-tactics/SKILL.md:3`: `description: Router skill - analyze requirements and direct to appropriate tactics`
- Marketplace dominant convention (per `reviewing-pack-structure.md` and per inspection of axiom-* pack descriptions) is `Use when [trigger] — [what it does]`. Most modern axiom packs use multi-sentence descriptions enumerating trigger domains plus do-NOT-use clauses.
- Impact: model description-based discovery is less precise. When choosing between `using-simulation-tactics` and `using-simulation-foundations` (mathematical) or `using-determinism-and-replay`, the current description gives the model no help with disambiguation.
- Note: `bravos-systems-as-experience` has the same issue; this may be a faction-style decision. Even so, a multi-domain "Use when" description would improve discovery.

**m2. plugin.json claims "11 skills, 3 commands, 2 agents" but only one SKILL.md exists.**
- `plugin.json:4`. The 10 "specialist skills" are reference sheets under the router. This is a marketplace-wide naming convention but creates a small honesty gap with users browsing `/plugin marketplace`.
- Suggested fix: either bring marketplace.json in line with plugin.json, or rephrase as "router + 10 reference sheets, 3 commands, 2 agents".

**m3. Marketplace.json description out of sync with plugin.json description.**
- `marketplace.json` entry: `"Game simulation implementation patterns and tactics - 11 skills"`.
- `plugin.json`: `"Game simulation implementation patterns and tactics - 11 skills, 3 commands, 2 agents"`.
- Trivial; harmonise on a single phrasing.

**m4. No cross-references to sibling packs (`yzmir-simulation-foundations`, `axiom-determinism-and-replay`).**
- Router has no "see also" / "for math, use X" / "for architecture-level determinism, use Y" section. Other modern routers (e.g. `axiom-static-analysis-engineering`, `axiom-determinism-and-replay`) explicitly redirect to siblings in the description and/or body.
- Impact: users land in this pack when they should land in foundations, or vice versa, and get no handoff signal.

**m5. `debugging-simulation-chaos.md:18-23` has a do-NOT-use bullet that misfires on single-player replay debugging.**
- The bullet `"Simple single-player games with no replay requirements (determinism not critical)"` is correct, but the *positive* trigger list above it should explicitly include single-player-replay-debug. As written, a reader could infer that all single-player work is out of scope. Wording tweak only.

### Polish

**p1. `multi-skill-workflows.md` could add a "Multiplayer Survival" workflow.** Covered partially by Workflow 2 (single-player survival) + Mistake 6 (multiplayer determinism), but a combined workflow would close gauntlet test B2 cleanly.

**p2. `expert-routing-guide.md` Edge Case 5 (tool/editor simulation) should be promoted into the router SKILL.md's Edge Cases section.** It's the only routing edge case that isn't game-specific and is most likely to be missed in fresh-context invocations.

**p3. Marketplace.json keywords could include `lod`, `desync`, `determinism` for findability.** Current keywords (`bravos`, `simulation`, `tactics`) are very generic.

---

## 6. Recommended Actions

Ordered by ROI. Each line is independently mergeable.

1. **(Major, M1) Slim the slash-command wrapper.** Reduce `/home/john/skillpacks/.claude/commands/simulation-tactics.md` from 909 lines to a thin pointer (model: ~20–40 lines like `python-engineering.md`). Drop the inlined scenarios and workflows; users get them by loading the skill. **Single biggest maintenance win.**
2. **(Minor, m1) Rewrite SKILL.md `description:`** as a "Use when …" multi-clause string enumerating the seven simulation domains plus the two cross-cutting concerns, with a do-NOT-use clause pointing at `yzmir-simulation-foundations` for math.
3. **(Minor, m4) Add a "Cross-pack handoffs" section to the router** naming `yzmir-simulation-foundations` (math), `axiom-determinism-and-replay` (architecture-level determinism), and `bravos-systems-as-experience` (high-level system design).
4. **(Minor, m5) Tweak `debugging-simulation-chaos.md` do-NOT-use bullet** to clarify single-player-replay-debug IS in scope.
5. **(Minor, m2 / m3) Harmonise plugin.json and marketplace.json descriptions.** Pick one phrasing; update both.
6. **(Polish, p1)** Add a "Multiplayer Survival" workflow to `multi-skill-workflows.md`.
7. **(Polish, p2)** Promote Edge Case 5 (tool-time sim) into router SKILL.md.
8. **(Polish, p3)** Expand `marketplace.json` keywords.

### Suggested version bump
Per `using-skillpack-maintenance:SKILL.md:243-251` (version bump rules):
- Action 1 alone: **Patch** (1.1.5 → 1.1.6) — content shuffled, no API/behaviour change.
- Actions 1–5 together: **Minor** (1.1.5 → 1.2.0) — enhanced guidance, harmonised metadata, cross-pack handoffs added. **This is the default-recommended bundle.**
- Adding Action 6+ does not escalate.

### Live verification recommended after fixes
Run subagent gauntlet on scenarios A1, B2, B3 from §4 to confirm wrapper-slimming did not break discovery.

---

## 7. Reviewer Notes

- This review is read-only and predictive. No subagent dispatches were performed (Stage 3 substituted with structural inspection). Sections marked "Predicted" should be live-tested before declaring fixes complete.
- The pack is the older bravos style (older convention, larger sheets — `simulation-vs-faking.md` is 2815 lines). Newer marketplace packs (`axiom-determinism-and-replay`, `axiom-static-analysis-engineering`) split content into ~13 narrower sheets. A *future* refactor could decompose the giant specialists; this is out of scope for v1.1.5 maintenance.
- The slash-command wrapper duplication appears to be a one-off drift, not a pattern: `simulation-foundations.md` (the yzmir wrapper) is also ~400 lines, but `python-engineering.md` is thin. Worth a follow-up audit across all wrappers — outside this review's scope.
- The pack does not register an entry in any "router skills available as slash commands" index that I could verify; the SLASH_COMMANDS.md mentioned in CLAUDE.md was not inspected during this pass.
- Agent SME-protocol compliance is *exemplary* in this pack; both agents follow the marketplace contract verbatim. This is one of the cleanest SME-compliant agent pairs in the marketplace.
- No security, licence, or attribution issues observed.
