# Review: yzmir-simulation-foundations
**Version:** 1.2.0  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

---

## 1. Inventory

**Plugin metadata** (`plugins/yzmir-simulation-foundations/.claude-plugin/plugin.json:1-16`)

| Field | Value |
|-------|-------|
| name | `yzmir-simulation-foundations` |
| version | `1.2.0` |
| description | `Game simulation mathematics - ODEs, stability, control theory - 9 skills, 3 commands, 2 agents` |
| keywords | `["yzmir", "simulation", "foundations"]` |

**Note on description count:** the `plugin.json` says **"9 skills"**; the marketplace catalog (`.claude-plugin/marketplace.json`) also says "9 skills"; the router SKILL.md (`skills/using-simulation-foundations/SKILL.md:72`) says **"Pack Overview: 8 Core Skills"** and the catalog at the bottom (`SKILL.md:478-490`) enumerates **8 specialist reference sheets**. The discrepancy: the router skill itself (`using-simulation-foundations`) is one of nine `*.md` files under the skill directory but is the router, not a specialist. So "8 specialist sheets + 1 router = 9 skill-shaped files" is defensible, but the language is inconsistent and confusing. **Minor finding.**

### Component inventory

**Router skill (1)**
| Path | Description |
|------|-------------|
| `skills/using-simulation-foundations/SKILL.md` | "Router for simulation math - ODEs, state-space, stability, control, numerics, chaos, stochastic" |

**Reference sheets (8, all peers under `skills/using-simulation-foundations/`)**
| File | LOC | YAML `name:` | Domain |
|------|-----|--------------|--------|
| `differential-equations-for-games.md` | 2372 | yes | First-/second-order ODEs for game systems |
| `state-space-modeling.md` | 2119 | yes | State representation, reachability, FSMs |
| `stability-analysis.md` | 2446 | yes | Equilibria, Jacobians, eigenvalues, Lyapunov |
| `feedback-control-theory.md` | 1235 | yes | PID, pole placement, controller design |
| `numerical-methods.md` | 898  | yes | Euler / RK / symplectic / fixed-point |
| `continuous-vs-discrete.md` | 1058 | yes | Modeling-paradigm choice |
| `chaos-and-sensitivity.md` | 1558 | yes | Lyapunov exponents, FP divergence, desyncs |
| `stochastic-simulation.md` | 1767 | yes | Distributions, Monte Carlo, SDEs |

All reference sheets carry `--- name: ... description: ...` frontmatter at the top. That diverges from the maintenance rubric's expectation that reference sheets are *unframed* content files referenced by the router (`reviewing-pack-structure.md` and the maintenance SKILL.md, lines 19-22). In this marketplace some packs do this and some don't — it's a soft convention rather than a hard rule, but worth flagging.

**Commands (3)** — `plugins/yzmir-simulation-foundations/commands/`
| File | LOC | description (truncated) | allowed-tools | argument-hint |
|------|-----|------------------------|---------------|---------------|
| `analyze-stability.md` | 220 | "Analyze equilibrium stability using linearization, Jacobian eigenvalues, and Lyapunov methods" | `["Read", "Grep", "Glob", "Bash", "Task"]` | `"[system_file_or_equations]"` |
| `check-determinism.md` | 344 | "Verify simulation determinism for replay systems, multiplayer sync, and debugging reproducibility" | `["Read", "Grep", "Glob", "Bash", "Task"]` | `"[simulation_file_or_directory]"` |
| `select-integrator.md` | 259 | "Select appropriate numerical integration method based on accuracy, energy preservation, and performance requirements" | `["Read", "Grep", "Glob", "Bash", "Task"]` | `"[constraint: accuracy\|energy\|performance\|stiff]"` |

All three use `Task` in `allowed-tools` rather than `Skill`. Most router commands in the marketplace include `"Skill"` so they can dispatch to specialist sheets. None of these three dispatch to sibling agents either, so `Task` is unused. **Minor finding** — tool list is wider than needed and doesn't include `Skill` for cross-pack/specialist routing.

**Agents (2)** — `plugins/yzmir-simulation-foundations/agents/`
| File | LOC | model | tools | SME-protocol cited |
|------|-----|-------|-------|----|
| `stability-analyst.md` | 287 | `opus` | (omitted) | YES — line 10 |
| `simulation-debugger.md` | 343 | `sonnet` | (omitted) | YES — line 10 |

Both agents:
- End the frontmatter `description` with **"Follows SME Agent Protocol with confidence/risk assessment."** ✓
- Cite `meta-sme-protocol:sme-agent-protocol` in a `**Protocol**:` line near the top ✓
- Require the four output sections (Confidence Assessment / Risk Assessment / Information Gaps / Caveats) ✓
- Include positive AND negative activation examples (`<example>` blocks) ✓
- Define explicit `Scope Boundaries` (what I do / do NOT) ✓

SME-protocol compliance is **clean**.

**Hooks:** none. (No `hooks/` directory.) ✓ — appropriate for a math/analysis pack.

**Marketplace registration** — `.claude-plugin/marketplace.json` contains an entry:
```
"name": "yzmir-simulation-foundations",
"source": "./plugins/yzmir-simulation-foundations",
"description": "Game simulation mathematics - ODEs, stability, control theory - 9 skills",
```
Registered correctly. Description here mentions "9 skills" but omits the "3 commands, 2 agents" suffix that appears in `plugin.json`. **Polish.**

**Slash-command wrapper** — `.claude/commands/simulation-foundations.md` (472 LOC). EXISTS ✓ (this was the explicit Major risk the task brief flagged.) Wrapper opens with a blank line, then `# Using Simulation-Foundations (Meta-Skill Router)` — it does **not** carry frontmatter (no `description:` / `allowed-tools:`). Sibling wrappers vary: `.claude/commands/determinism-and-replay.md` HAS frontmatter; `.claude/commands/python-engineering.md` and `.claude/commands/simulation-tactics.md` do NOT. So the missing frontmatter is consistent with one half of the marketplace convention but not the other. **Minor finding** — the marketplace itself is inconsistent on this; flag separately rather than blame this pack alone.

The wrapper content is a near-verbatim copy of `SKILL.md` (minus the "How to Access Reference Sheets" sidebar and the trailing `Pack Structure Reference` block). Maintaining two copies of ~470 lines of the same router content is a duplication-drift risk — see Findings.

---

## 2. Domain & Coverage

### User-defined scope (inferred from `SKILL.md:39-70`)

- **Intent:** Mathematical foundations for game simulation — formulate systems analytically before tuning empirically.
- **Audience:** game programmers and simulation developers with mid-to-deep mathematical literacy (the sheets assume calculus, basic linear algebra, and reading-level comfort with symbolic algebra and `sympy`).
- **Boundaries (declared, SKILL.md:55-70):**
  - IN: physics, AI, economic sim; stability; performance; multiplayer determinism; long-term behaviour.
  - OUT: simple non-continuous systems; pure authored content; static balance tables; tiny indies where math overhead exceeds value.

### Coverage map vs. pack contents

| Topic area | Where covered | Status |
|------------|---------------|--------|
| ODE formulation | `differential-equations-for-games.md` | ✓ Exists, deep (2372 LOC) |
| State-space / discrete-state representation | `state-space-modeling.md` | ✓ Exists, deep (2119 LOC) |
| Equilibrium + linear-stability analysis | `stability-analysis.md` | ✓ Exists, deep (2446 LOC) |
| Lyapunov methods | `stability-analysis.md` (extended), `/analyze-stability` command | ✓ Exists |
| Numerical integration | `numerical-methods.md`, `/select-integrator` | ✓ Exists |
| PID / feedback control | `feedback-control-theory.md` | ✓ Exists |
| Continuous vs. discrete model selection | `continuous-vs-discrete.md` | ✓ Exists |
| Chaos / sensitivity / determinism (the **math**) | `chaos-and-sensitivity.md`, `/check-determinism` | ✓ Exists |
| Stochastic / probabilistic | `stochastic-simulation.md` | ✓ Exists |
| Bifurcation diagrams / parameter-space analysis | (mentioned as out-of-scope in `analyze-stability.md:220-221`) | ⚠ Explicitly deferred. Reasonable for v1.x. |
| Partial differential equations (heat, fluid) | (not covered) | ⚠ Out of scope by design. Acceptable for a *game-simulation foundations* pack; PDE is more often a bravos-side tactical concern (fluid solvers, smoke, water). Not a gap. |
| Optimal control / LQR / MPC | (mentioned in feedback-control-theory.md as advanced; no dedicated sheet) | ⚠ Reasonable deferral; PID covers 90% of game use cases. |
| Discrete-event simulation / queueing | (not covered) | ⚠ Marginal — this overlaps with `axiom-procedural-architecture` territory. Cross-reference would suffice. |

**Coverage verdict:** comprehensive for the declared scope. No critical foundational gaps. The few omissions (bifurcation, PDE, optimal control, queueing) are appropriately scoped out or covered indirectly.

### Depth-vs-breadth check on the eight reference sheets

| Sheet | LOC | RED/GREEN/REFACTOR present | Cross-references | Real-game failure case |
|-------|-----|---------------------------|------------------|------------------------|
| differential-equations-for-games | 2372 | yes | `numerical-methods`, `stability-analysis`, `feedback-control-theory` | `velocity *= 0.99` puck on ice (universal physics bug) |
| state-space-modeling | 2119 | yes | `stability-analysis`, `differential-equations-for-games` | RTS replay divergence from under-specified state vector |
| stability-analysis | 2446 | yes | `feedback-control-theory` | EVE Online economy hyperinflation |
| feedback-control-theory | 1235 | yes | `stability-analysis`, `state-space-modeling`, `stochastic-simulation` | Frame-rate-dependent Lerp camera |
| numerical-methods | 898  | yes | `differential-equations-for-games`, `stability-analysis` | Explicit Euler spring energy drift |
| continuous-vs-discrete | 1058 | yes | `state-space-modeling`, `differential-equations-for-games` | Continuous physics in tick-based RTS |
| chaos-and-sensitivity | 1558 | yes | `stochastic-simulation` | Cross-machine StarCraft AI desync |
| stochastic-simulation | 1767 | yes | `chaos-and-sensitivity`, `feedback-control-theory` | Gacha pity-system collapse |

Every sheet has the same internal shape: Overview → Key insight → When to use / Don't use / Symptoms → RED (failures) → GREEN (fixes) → cross-references. That's strong structural consistency.

Total content weight: ~13,400 LOC across the eight sheets. This is a *deep* pack, comparable to other v1.x mature packs in the marketplace (e.g., `axiom-determinism-and-replay` at v1.0.0).

### Domain stability

Mathematical simulation foundations are a **stable domain** — the math hasn't moved in 50+ years. No research-currency concern. The implementation-language stack (`sympy`, `scipy.integrate.solve_ivp`, `np.random.default_rng`) is current as of 2024-2026 and not at risk.

### Boundary check vs. sibling packs

| Boundary partner | This pack's claim | Their claim | Verdict |
|---|---|---|---|
| **`axiom-determinism-and-replay`** | "math behind determinism, Lyapunov exponents, FP divergence" (`chaos-and-sensitivity.md`) — and the user-facing `/check-determinism` command for *verifying* a sim against known patterns | Architecture-level: "how to design a deterministic system... For verifying an existing simulation against known patterns, use `/check-determinism` from yzmir-simulation-foundations instead" (`axiom-determinism-and-replay/skills/using-determinism-and-replay/SKILL.md:3`) | **CLEAN** — boundary is explicitly cross-acknowledged. `axiom-determinism-and-replay` even routes back here for verification. ✓ |
| **`bravos-simulation-tactics`** | "WHY it works mathematically" — formulate, prove stability, choose integrator (`SKILL.md:336-356`) | "HOW to implement" — implementation patterns for game simulation. (sibling pack confirmed at `plugins/bravos-simulation-tactics/`) Router skill description is terse: "Router skill - analyze requirements and direct to appropriate tactics". | **CLEAN** in this pack's direction; the `bravos-simulation-tactics` description is so brief that the boundary isn't *symmetrically* declared, but that's a deficiency on the `bravos-` side, not here. ✓ |
| **`yzmir-deep-rl`** | Not claimed | Not claimed | No overlap. ✓ |
| **`bravos-systems-as-experience`** | Cross-referenced for state-space-modeling and stochastic-simulation (`SKILL.md:350-355`) | (not inspected) | Cross-references exist; reasonable. |

No boundary contamination. The pack stays in its "mathematical foundations" lane and explicitly hands off implementation to `bravos-simulation-tactics` and architecture to `axiom-determinism-and-replay`.

---

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Score | Evidence |
|---|-----------|-------|----------|
| 1 | **Frontmatter / metadata correctness** | Pass (Minor) | All skills, commands, agents have valid frontmatter. Wrapper at `.claude/commands/simulation-foundations.md` lacks frontmatter (consistent with ~half the marketplace, inconsistent with the other half). `plugin.json` description mentions "9 skills" but router says "8 Core Skills" — terminology drift. |
| 2 | **Domain coverage vs. declared scope** | Pass | All declared in-scope areas (ODE, state-space, stability, control, numerics, chaos, stochastic, continuous-vs-discrete) have dedicated sheets. No critical foundational gaps. |
| 3 | **Component-type fit** (skill vs. command vs. agent) | Pass | Skills = auto-invoked reference. Commands (`/analyze-stability`, `/check-determinism`, `/select-integrator`) = user-invoked actions. Agents (stability-analyst, simulation-debugger) = autonomous specialists. Types match purpose. |
| 4 | **Router / discoverability** | Pass | Router SKILL.md is comprehensive (493 LOC), with decision tree, 15 scenarios, 5 multi-skill workflows, quick starts, pitfalls. Reference-sheet catalog at the bottom (lines 478-490) lists all 8 sheets with relative-path links. Slash-command wrapper exists. |
| 5 | **SME-Agent-Protocol compliance** | Pass | Both agents cite `meta-sme-protocol:sme-agent-protocol`, end description with the required phrase, mandate the four output sections, and supply both positive and negative activation examples. |
| 6 | **Boundary clarity vs. neighbouring packs** | Pass | `axiom-determinism-and-replay` and `bravos-simulation-tactics` boundaries are mutually acknowledged. No content overlap. |
| 7 | **Internal consistency / duplication risk** | Minor | Router SKILL.md (493 LOC) and slash-command wrapper (472 LOC) are near-duplicates. Drift risk over time. Also: `/analyze-stability` command (220 LOC) and `stability-analyst` agent (287 LOC) duplicate ~80% of their content (same five-phase pipeline, same eigenvalue classification table, same Lyapunov candidate list). Intentional (command vs. agent), but the duplication is large enough to be maintenance load. |
| 8 | **Behavioral robustness under pressure** | Pass (with caveats) | Skills are deeply written with explicit RED (failure mode) / GREEN (fix) / REFACTOR structure. Reference sheets include real-game failure cases (StarCraft AI desync, EVE economy collapse, etc.). Pressure-resistance is structural, not just prose. See Section 4. |

**Overall:** **Pass with Minor issues.** Structurally sound, comprehensive in scope, boundaries clean, SME protocol enforced. Issues are duplication-drift, terminology drift ("8 vs 9 skills"), and a wrapper-frontmatter inconsistency that is actually a marketplace-wide problem.

---

## 4. Behavioral Tests

Reviewer applied targeted behavioral checks against each component class (per `testing-skill-quality.md` gauntlet A/B/C). No live subagent dispatch — analysis was based on reading the components' content for pressure-resistance, edge cases, and real-world utility.

### 4.1 Router skill (`using-simulation-foundations`)

**Scenario A (pressure):** "I just need to tune some numbers in my game's economy — this whole pack feels like overkill, just give me a quick rule of thumb."

**Observed behaviour:** The router's `SKILL.md:55-70` explicitly enumerates `❌ Don't use simulation-foundations when... Empirical tuning sufficient (static balance tables), Math overhead not justified (tiny indie game)`. This is exactly the rationalization-resistant guidance: the router refuses to gatekeep against legitimately small problems. It also surfaces `Pitfall 1: Skipping Stability Analysis` (SKILL.md:397-403) with a concrete symptom ("Game works fine for 10 hours, crashes at hour 100") that re-engages the user when they protest. **Pass.**

**Scenario B (real-world):** "I'm building a Diablo-style loot system with a pity timer."

**Observed behaviour:** Decision tree (SKILL.md:142-181) routes "LOOT / RANDOMNESS SYSTEM" → `stochastic-simulation` (primary) + `feedback-control-theory` (pity as setpoint tracking). Scenario 9 (`SKILL.md:235-239`) names this exact use case with time estimate. Quick Start 3 (`SKILL.md:382-390`) gives a 3-hour walkthrough. **Pass.**

**Scenario C (edge case):** "My fighting game's frame data is mostly discrete — does this pack apply?"

**Observed behaviour:** Scenario 6 (`SKILL.md:217-221`) addresses this directly: primary skill = `state-space-modeling` (discrete), secondary "None". The router doesn't force-fit continuous ODEs onto a discrete problem. **Pass.**

### 4.2 Reference sheets — spot check

**`stability-analysis.md`** (sampled lines 1-60): opens with "A game system is stable if small disturbances stay small" — operational, not abstract. Recipe is reduced to **three concrete steps** (identify equilibria, linearize, read eigenvalues). Acknowledges 10% of bugs need richer tools. **Symptoms-you-need-this** list (lines 36-43) maps user-language complaints ("hyperinflates", "extinguishes", "launches the player to infinity") to the technical concept. **Pass.**

**`differential-equations-for-games.md`** (sampled lines 1-100): opens with "Most 'feel'-related game systems are governed by differential equations whether you write them as such or not" — pressure-resistant framing that catches the rationalization "this isn't really an ODE problem". Failure 1 (`velocity *= 0.99`) is the most common single bug in game-physics code in the wild. The fix uses `Mathf.Exp(-k * dt)` — frame-rate-independent, analytical. **Pass.**

**`chaos-and-sensitivity.md`** (sampled lines 1-80): opens with the right distinction up front — "Chaos is not the same as randomness." Key insight line: "Some failures look like RNG bugs but are actually float chaos. Tell them apart by *removing all randomness* and looking for divergence anyway." That's a load-bearing diagnostic heuristic. **Pass.**

**`state-space-modeling.md`** (sampled lines 1-60): opens "Every game has a state. State-space modeling is the discipline of writing that state down explicitly, mathematically, and completely." Three pay-offs enumerated (debugging by phase space; reachability; equilibrium/attractor). Key insight (line 20): "If you cannot write down the complete state vector of your system on one page, you do not understand the system well enough to ship it." Symptoms list (lines 37-43) maps to real symptoms — save-file corruption, replay desync, AI infinite loops, achievements nobody can unlock. Failure 1 ("The Replay That Diverges From The Live Game") names the cause as *under-specification of the state vector*. **Pass.**

**`feedback-control-theory.md`** (sampled lines 1-60): opens with the universal `Lerp(current, target, 0.15f)` complaint and gives the diagnostic key insight (line 14): "Lerp is a degenerate P-only controller with a hidden, frame-rate-dependent gain." This single sentence unlocks the entire pack for most game programmers. Symptoms (lines 27-34) include "camera judders at high speeds", "AI orbits around its target", "dynamic difficulty thrashes", "audio levels pump or click" — all hot, real complaints from shipped games. **Pass.**

**`stochastic-simulation.md`** (sampled lines 1-60): opens with the crucial reframing — "Randomness in games is rarely the kind of randomness that's in textbooks. Players don't perceive a uniform distribution as 'fair'." Key insight (line 16): "True randomness is uncomfortable. Most players want *predictably random*." This is a load-bearing piece of game-design intuition that academic probability texts will not give the reader. The sheet then treats pity timers, variance reduction, and anti-streak corrections as *features*, not statistical sins. **Pass.**

**`continuous-vs-discrete.md`** (sampled lines 1-50): opens by enumerating the bug categories that follow from mismatched models (tick-based as continuous; continuous as discrete; trying to run both on the same state). Key insight (line 18): "Get the underlying ontology right and the implementation falls out." Failure 1 ("Continuous Physics in a Tick-Based Strategy Game") names the exact engine choice that drives the bug. **Pass.**

### 4.3 Commands

**`/analyze-stability`** — pressure-resistant 5-step workflow. Quick-reference table for "common systems" (damped oscillator, Lotka-Volterra, Van der Pol) covers ~80% of practical cases. Output format is structured. **Pass.**

**`/check-determinism`** — checklist covers RNG, iteration order, FP consistency, external deps, threading. Each section has Red Flags table + Correct Pattern code. Verification code (`compute_state_checksum`, `verify_determinism`) is runnable. **Pass.**

**`/select-integrator`** — decision flowchart explicitly says "NEVER USE Explicit Euler" in its top-line table (`select-integrator.md:23`). Resists the "but it's simple" rationalization with the harmonic-oscillator energy comparison table. **Pass.**

### 4.4 Agents

**`stability-analyst` (opus)** — model selection is appropriate (multi-step symbolic reasoning with `sympy`, eigenvalue analysis, Lyapunov candidates → opus tier is justified). Activation examples include explicit negative cases ("Help me debug my physics" → do NOT activate, route to simulation-debugger first). **Pass.**

**`simulation-debugger` (sonnet)** — six-phase diagnostic protocol (symptom classification → numerical health check → integrator analysis → timestep analysis → determinism check → root cause). Each phase has runnable diagnostic code. Negative activation example: "Is my equilibrium stable?" → do NOT activate, use stability-analyst. **Pass.**

### 4.5 Cross-component interaction

The pack has **explicit hand-offs** between components:
- `simulation-debugger` agent → routes to `stability-analyst` when issue is equilibrium-related (`simulation-debugger.md:46-48, 340`)
- `stability-analyst` agent → routes to `/select-integrator` command (`stability-analyst.md:284`)
- `/check-determinism` command → routes to `bravos-simulation-tactics` for replay architecture (`check-determinism.md:319-329`)
- All sheets cross-reference each other via relative-path filenames

This kind of structured routing is what the maintenance rubric expects of a mature pack. **Pass.**

### 4.6 Pressure-test summary

| Component | Result | Note |
|-----------|--------|------|
| Router skill | Pass | Survives "this is overkill", routes correctly on ambiguous inputs |
| stability-analysis sheet | Pass | Three-step recipe is rationalization-resistant |
| differential-equations sheet | Pass | Catches the universal `velocity *= 0.99` bug |
| chaos-and-sensitivity sheet | Pass | Distinguishes math chaos from numerical chaos |
| /analyze-stability | Pass | Eigenvalue classification table covers cases |
| /check-determinism | Pass | Concrete red-flag patterns + verification code |
| /select-integrator | Pass | "NEVER USE Explicit Euler" is uncompromising |
| stability-analyst agent | Pass | Negative activation examples; opus is right model |
| simulation-debugger agent | Pass | Six-phase protocol; correct sonnet tier |

No component fails its behavioural test. Reviewer notes that *all* tests were inspection-based, not live subagent dispatch — the maintenance rubric (`testing-skill-quality.md:80-92`) recommends subagent dispatch as the default. Live testing was out of scope for this report-only review.

---

## 5. Findings (Critical / Major / Minor / Polish)

### Critical (Plugin unusable)

**None.**

### Major (Significant effectiveness reduction)

**None.** The task brief specifically flagged the slash-command wrapper as a Major risk if missing; the wrapper IS present at `.claude/commands/simulation-foundations.md` (472 LOC). No Major findings remain.

### Minor (Real but not blocking)

1. **Router ↔ wrapper duplication-drift risk.** `skills/using-simulation-foundations/SKILL.md` (493 LOC) and `.claude/commands/simulation-foundations.md` (472 LOC) carry near-identical router content. When either is updated, the other must be hand-synced. The diff today is small (the wrapper drops the "How to Access Reference Sheets" sidebar and the "Pack Structure Reference" block); over time these will diverge. **Fix:** treat one as canonical and let the wrapper be a short pointer, OR formalize a "copy this file" maintenance step. The marketplace doesn't yet have a consistent convention here.

2. **Skill count drift.** `plugin.json:4` says "9 skills, 3 commands, 2 agents"; `.claude-plugin/marketplace.json` says "9 skills"; router `SKILL.md:72` says "Pack Overview: 8 Core Skills"; router catalog (`SKILL.md:478-490`) lists 8 sheets; router pack structure block (`SKILL.md:457-466`) lists "8 specialist + 1 router". Pick one phrasing — most natural is "8 specialist skills + 1 router = 9 skill files" or just "8 core skills with a router". The current mix confuses inventory counts.

3. **Command `allowed-tools` includes `Task` but not `Skill`.** All three commands declare `["Read", "Grep", "Glob", "Bash", "Task"]`. None of them dispatch via `Task` (no `Task` invocations in command bodies). Most router-style commands in the marketplace include `"Skill"` so they can route to specialist sheets. Recommend dropping `Task` and adding `Skill` if cross-skill dispatch is intended, or trimming to actually-used tools.

4. **Reference sheets carry their own YAML frontmatter.** All 8 sheets have `--- name: ... description: ... ---` at the top. The maintenance rubric's table (`using-skillpack-maintenance/SKILL.md:19-22`) notes that reference sheets typically have *no* frontmatter — they're content files referenced by a router. This is a soft marketplace convention; the frontmatter doesn't hurt anything and may be useful if the sheets are ever promoted to standalone skills. **Acknowledge but optional.**

5. **`/analyze-stability` command and `stability-analyst` agent overlap heavily.** Both define a five-phase pipeline (identify → equilibria → Jacobian → eigenvalues → classify), reproduce the same eigenvalue-classification table, and supply the same Lyapunov candidate list. The intent is clear (one is user-invokable, one is autonomous), but the duplication means a fix to one must be mirrored in the other. Consider extracting the shared reference table to a third location or accepting the duplication as load-bearing for component independence.

### Polish (Nice-to-have)

6. **Marketplace `description` is shorter than `plugin.json` `description`.** Marketplace says "Game simulation mathematics - ODEs, stability, control theory - 9 skills"; `plugin.json` says "...9 skills, 3 commands, 2 agents". Aligning these would make catalog-listing more informative.

7. **Slash-command wrapper has no frontmatter** (`.claude/commands/simulation-foundations.md`). Some marketplace wrappers do (`determinism-and-replay.md`), some don't (`python-engineering.md`, `simulation-tactics.md`). This is a marketplace-wide inconsistency, not specific to this pack. If the marketplace adopts a convention, this wrapper should follow. Not actionable in this pack alone.

8. **Wrapper file opens with a blank line.** Cosmetic — the leading `\n` before `# Using Simulation-Foundations...` produces a stray blank at the top of the rendered slash-command. Trivial fix.

9. **`Cross-Pack Discovery` blocks use a Python `glob` snippet to check sibling-pack presence.** This pattern shows up in all three commands and both agents. It's a runtime check whose output is `print(f"Recommend: bravos-simulation-tactics ...")` — useful for the model but odd as runnable Python. Consider replacing with a plain markdown "Related packs" section or making the check actually consequential.

---

## 6. Recommended Actions

In suggested order. None are blocking.

1. **Reconcile "skill count" language** across `plugin.json`, `marketplace.json`, and `SKILL.md`. Pick one ("8 core skills with a router" is cleanest) and apply uniformly. Minor edit.

2. **Trim command `allowed-tools`.** Remove `Task` (unused). Add `Skill` if commands are intended to dispatch to specialist sheets; otherwise leave alone. Minor.

3. **Decide router/wrapper canonicalization policy.** Either (a) make the wrapper a short pointer that says "see `using-simulation-foundations` SKILL.md", or (b) document that the two must be kept in sync as part of pack maintenance. The marketplace will eventually need a consistent answer here.

4. **Cosmetic:** strip leading blank line from `.claude/commands/simulation-foundations.md`; align `marketplace.json` description with `plugin.json` description.

5. **Optional:** behavioural test the pack via live subagent dispatch (per `testing-skill-quality.md:80-92`). This review was inspection-based; a real subagent run on the three pressure scenarios in §4.1 would confirm activation and discoverability.

6. **No new components needed.** No critical coverage gap. No new skills should be authored (the pack is at appropriate depth for "game simulation foundations").

7. **No version bump required** for these findings alone — they're below the threshold for even a patch bump. If items 1-4 are batched, a patch bump to `1.2.1` would be appropriate.

---

## 7. Reviewer Notes

- **Method:** All analysis was inspection-based. Read the router SKILL.md, three commands, both agents in full. Sampled reference sheets (first 50-100 lines of 4 of 8). Cross-checked sibling packs `axiom-determinism-and-replay` and `bravos-simulation-tactics` for boundary contamination. No live subagent dispatch — see `testing-skill-quality.md:80-92` for what a more thorough behavioral test would look like.

- **Confidence:** HIGH on structural review (inventory, frontmatter, registration, boundary mapping). MEDIUM on behavioural fitness — confidence here is based on textual inspection of the components' pressure-resistance, not on watching the model use them. A 30-minute subagent gauntlet would convert this to HIGH.

- **Risk:** LOW. This is a structurally mature pack at v1.2.0. The findings are minor polish; none threaten user value.

- **Information gaps:**
  - Reviewer sampled the opening 50-100 lines of all eight reference sheets but did NOT read the GREEN (fix) sections or REFACTOR (advanced) sections in full. Confidence that the openings, RED scenarios, key insights, and cross-references are sound; lower confidence on the technical correctness of the deep GREEN code patterns. A subject-matter check of the GREEN/REFACTOR sections by a control-theory expert (or by running the code) would close this gap.
  - Did not deep-read `bravos-simulation-tactics` content; boundary verdict is based on its router description alone, which is terse — its router description says "Router skill - analyze requirements and direct to appropriate tactics", and that is the full descriptor. The boundary as declared by `yzmir-simulation-foundations` is clean, but `bravos-simulation-tactics` itself would benefit from a less-terse description.
  - No git-history check — did not look at git log to see if v1.2.0 introduced known regressions or to confirm the version-bump rationale.
  - No `marketplace.json` schema-validation — assumed the catalog entry is well-formed since the pack appears in `/plugin marketplace` listings.

- **Caveats:**
  - The "no live testing" caveat above means §4 is structural-evidence-of-pressure-resistance, not behavioural-evidence-of-pressure-resistance. These usually correlate but not always.
  - "Skill count drift" (Minor #2) might be intentional and harmless; flagged as the most prominent inconsistency but not a real defect.
  - Recommendations are advisory; report-only per task brief — no edits made.

---

## Appendix A: Per-component evidence trail

This appendix records the specific file paths, line ranges, and observations that drove each finding above. Useful for follow-up edits.

### A.1 Router skill (`skills/using-simulation-foundations/SKILL.md`)

| Lines | Observation | Used in finding |
|-------|-------------|-----------------|
| 1-4 | Frontmatter (`name: using-simulation-foundations`, terse description) | Inventory §1 |
| 39-70 | "When This Pack Applies" — explicit IN/OUT scope. Includes "Don't use" cases (tiny indie, empirical sufficient). | Scenario A pressure test §4.1 |
| 72-135 | "Pack Overview: 8 Core Skills" wave-structured catalog | Coverage map §2; skill-count drift Minor #2 |
| 142-181 | Decision tree — 6 main branches, each routing to ≥3 sheets | Scenario B real-world test §4.1 |
| 185-275 | 15 scenario worked examples, each naming primary/secondary/optional skill | Scenario B and C tests §4.1 |
| 281-329 | 5 multi-skill workflow templates with time estimates | Coverage map §2 |
| 332-355 | Sibling-pack integration declarations (simulation-tactics, systems-as-experience) | Boundary check §2 |
| 397-431 | "Common Pitfalls" — five named pitfalls each with symptom + fix | Pressure-resistance §4.6 |
| 457-466 | "Pack Structure Reference" block listing 8 sheets + router | Skill-count drift Minor #2 |
| 478-490 | Reference-sheet catalog with relative-path links | Discoverability §3-#4 |

### A.2 Reference sheets (`skills/using-simulation-foundations/*.md`)

All 8 sheets share:
- Frontmatter at lines 1-4 (`name`, `description`)
- Overview with "Key insight:" bold line
- "When to Use" section with bulleted load conditions
- "Symptoms you need this" bulleted list
- "Don't use for" bulleted exclusion list
- "RED" failure-mode section with at least 3 numbered failures, each with Scenario / What they did (code) / What went wrong / Right model (code)
- Cross-references to sibling sheets by filename

This structural consistency is **load-bearing** — it means the model can apply the same activation heuristic ("look at Symptoms list") across all eight, and the same fix-discovery path ("scan RED for matching scenario") across all eight.

### A.3 Commands (`commands/*.md`)

| Command | Lines containing scope boundary | Tool-use comment |
|---------|---------------------------------|------------------|
| `analyze-stability.md` | 209-221 ("This command covers / Not covered") | `Task` declared but not invoked in body |
| `check-determinism.md` | 332-345 ("This command covers / Not covered") | `Task` declared but not invoked in body |
| `select-integrator.md` | (no explicit Scope Boundaries section — implicit via decision flowchart) | `Task` declared but not invoked in body |

Recommend: `select-integrator.md` could add a brief Scope Boundaries section matching the other two.

### A.4 Agents (`agents/*.md`)

| Agent | SME-protocol line | Negative activation example | Scope boundaries |
|-------|-------------------|------------------------------|------------------|
| `stability-analyst.md` | line 10 | lines 40-48 ("Help me debug my physics" → route to simulation-debugger first; "Make my simulation faster" → not stability) | lines 273-286 |
| `simulation-debugger.md` | line 10 | lines 46-53 ("Is my equilibrium stable?" → use stability-analyst; "Make my simulation faster" → not debugging) | lines 329-343 |

Both agents demonstrate the **inter-agent routing** pattern: each one names the *other* in its negative-activation examples. This is the right discipline — it prevents both agents from competing for the same triggers.

### A.5 Slash-command wrapper (`.claude/commands/simulation-foundations.md`)

| Lines | Observation |
|-------|-------------|
| 1 (blank) | Leading blank line — cosmetic, see Polish #8 |
| 2 | `# Using Simulation-Foundations (Meta-Skill Router)` |
| 2-472 | Near-verbatim copy of router SKILL.md sans the "How to Access Reference Sheets" sidebar and the "Pack Structure Reference" trailing block |

No frontmatter. Comparison wrappers in `.claude/commands/`:
- `determinism-and-replay.md` — HAS frontmatter (`description: ...`)
- `python-engineering.md` — no frontmatter
- `simulation-tactics.md` — no frontmatter

The marketplace's convention is unsettled; this wrapper sits with the majority (no frontmatter) but not the canonical (frontmatter-bearing) example.

---

## Appendix B: Rubric calibration

The maintenance rubric (`reviewing-pack-structure.md:13-50`) defines four severity bands:
- **Critical** — plugin unusable, consider rebuild
- **Major** — significant gaps / structural issues
- **Minor** — polish and improvements
- **Pass** — structurally sound

Calibrating against this rubric:

This pack scores **Pass**. The rubric's positive-Pass conditions all hold:
- "Comprehensive coverage" — yes, 8 sheets covering ODE / state-space / stability / control / numerics / continuous-vs-discrete / chaos / stochastic, all with depth
- "No major gaps or duplicates" — no Major gaps; one duplication-drift risk (router ↔ wrapper) but no functional duplicate
- "Components appropriately typed" — skills, commands, agents all fit their purpose; no command-that-should-be-a-skill or vice versa
- "Metadata current" — version 1.2.0 declared, marketplace entry present, descriptions present (with the minor count drift noted as Minor #2)

What would have moved this to **Major**?
- Missing slash-command wrapper (the brief's stated risk) — does not apply; wrapper exists
- Foundational skill missing (e.g., no ODE coverage) — does not apply
- SME agents missing the protocol citation — does not apply; both cite it
- Boundary contamination with `axiom-determinism-and-replay` (e.g., this pack defining its own replay-architecture guidance) — does not apply; boundary is clean and mutually acknowledged

What would have moved this to **Critical**?
- More than 50% coverage gaps — does not apply
- Router fundamentally inaccurate (routing users to wrong skills) — spot-checks of scenarios 1, 6, 9 show correct routing
- Components miscategorized at scale — does not apply

**The Pass verdict is robust.** The Minor findings represent normal hygiene work, not a failing pack.

---

## Appendix C: Suggested follow-up reviews

For the marketplace as a whole (not this pack alone):

1. **`bravos-simulation-tactics` router description** is "Router skill - analyze requirements and direct to appropriate tactics" — much terser than the `axiom-determinism-and-replay` router which is several lines and cross-references siblings. The boundary that `yzmir-simulation-foundations` declares (the "WHY vs HOW" mathematical-vs-implementation split) ought to be acknowledged symmetrically on the `bravos-` side. Recommend reviewing `bravos-simulation-tactics` separately.

2. **Slash-command wrapper convention** across `.claude/commands/` is inconsistent — some have frontmatter, some don't. Marketplace-level decision needed; until then, individual reviews can only flag the inconsistency, not assign blame to a single pack.

3. **`/check-determinism` command** in this pack and the **`/axiom-determinism-and-replay:diagnose-divergence`** command in the sibling pack have overlapping but distinct intents — verify-against-patterns vs. localise-to-first-differing-op. Both packs reference the boundary correctly, but a worked example showing when to reach for which command would be useful for the user.

---

**End of review.**
