# Review: yzmir-training-optimization

**Version:** 1.2.0   **Reviewed:** 2026-05-22   **Reviewer:** general-purpose subagent

Scope: Stages 1–4 of `meta-skillpack-maintenance:using-skillpack-maintenance`. Read-only. No edits performed.

---

## 1. Inventory

### Plugin metadata

- `plugins/yzmir-training-optimization/.claude-plugin/plugin.json`
  - `name`: `yzmir-training-optimization`
  - `version`: `1.2.0`
  - `description`: claims **"11 skills + 3 commands + 2 agents"** and enumerates the 2026-05 modern landscape (AdamW/Lion/Sophia/Muon/AdEMAMix/Schedule-Free/Prodigy/8-bit Adam; cosine/WSD; BF16/FP8 E4M3/E5M2; B_crit, Chinchilla; muP).
  - `license`: `CC-BY-SA-4.0`
- Marketplace registration: `.claude-plugin/marketplace.json` registers the plugin with the stale description **"Training stability - optimizers, learning rates, convergence, debugging - 11 skills"**, which does not reflect the 1.2.0 modernization.

### Skills (1 router SKILL.md + 10 reference sheets)

| Component | Path | Lines | Status |
|---|---|---:|---|
| Router | `skills/using-training-optimization/SKILL.md` | 583 | OK content, see Findings |
| Sheet | `skills/using-training-optimization/optimization-algorithms.md` | 1162 | OK |
| Sheet | `skills/using-training-optimization/learning-rate-scheduling.md` | 1162 | OK |
| Sheet | `skills/using-training-optimization/loss-functions-and-objectives.md` | 2166 | OK |
| Sheet | `skills/using-training-optimization/gradient-management.md` | 2483 | OK |
| Sheet | `skills/using-training-optimization/batch-size-and-memory-tradeoffs.md` | 695 | OK |
| Sheet | `skills/using-training-optimization/data-augmentation-strategies.md` | 1509 | OK |
| Sheet | `skills/using-training-optimization/overfitting-prevention.md` | 1488 | OK |
| Sheet | `skills/using-training-optimization/training-loop-architecture.md` | 636 | OK |
| Sheet | `skills/using-training-optimization/hyperparameter-tuning.md` | 1667 | OK |
| Sheet | `skills/using-training-optimization/experiment-tracking.md` | 1973 | OK |

Total sheet content: ~14.9k lines of guidance + 583-line router.

**Counting convention.** The `find skills -name SKILL.md` count is 1 (the router). Plugin/marketplace metadata says "11 skills" — this counts the router + 10 sheets, which is internally consistent with how this marketplace markets sheet-based packs.

### Commands (3)

| Command | Path | Lines | argument-hint | allowed-tools |
|---|---|---:|---|---|
| `/check-gradients` | `commands/check-gradients.md` | 184 | `"[training_script.py]"` | `["Read", "Bash", "Grep", "Glob", "Skill"]` |
| `/diagnose` | `commands/diagnose.md` | 186 | `"[training_script.py or logs]"` | `["Read", "Grep", "Glob", "Bash", "Skill", "AskUserQuestion"]` |
| `/setup` | `commands/setup.md` | 241 | `"[task_type: classification\|regression\|generation]"` | `["Read", "Write", "Bash", "Skill", "AskUserQuestion"]` |

All commands use the quoted-array `allowed-tools` style and include `Skill` for dispatching to the router.

### Agents (2)

| Agent | Path | Lines | model | SME protocol |
|---|---|---:|---|---|
| `training-config-reviewer` | `agents/training-config-reviewer.md` | 262 | `haiku` | Compliant — description ends "Follows SME Agent Protocol with confidence/risk assessment." and body cites `meta-sme-protocol:sme-agent-protocol` with the four required sections (`agents/training-config-reviewer.md:1-10`). |
| `training-diagnostician` | `agents/training-diagnostician.md` | 308 | `sonnet` | Compliant — same pattern (`agents/training-diagnostician.md:1-10`). |

Neither agent declares `tools:` — consistent with the marketplace convention of inheriting parent context.

### Hooks

None. The pack ships no `hooks/` directory.

### Slash-command wrapper

- `.claude/commands/training-optimization.md` **exists** (467 lines).
- **Status: stale.** It mirrors a pre-1.2.0 version of the router. Verified divergences (router → wrapper):
  - Wrapper does not mention Lion / Sophia / Muon / Shampoo / AdEMAMix / Schedule-Free / Prodigy / AdamW8bit / paged optimizers (router devotes an entire section to them, `SKILL.md:215-235`).
  - Wrapper does not mention WSD / warmup-stable-decay or infinite-LR schedules.
  - Wrapper does not mention FP8 / BF16 / FP16 precision strategy as a routing axis (router treats precision as a routable concern: `SKILL.md:86-118`, `SKILL.md:215-235`).
  - Wrapper does not mention muP / mu-Transfer or ZeRO/FSDP strategy routing.
  - Wrapper does not carry the cross-pack DPO/GRPO/SimPO split with `yzmir-llm-specialist` (`SKILL.md:333-345`).
  - Wrapper's "Skill Catalog" lists 10 sheets but uses 2025-era descriptions ("SGD vs Adam vs AdamW, momentum"); router's catalog enumerates the modern landscape and frozen vocabulary (`SKILL.md:524-537`).
  - Wrapper missing the explicit "AdamW + cosine + BF16 is the boring-correct default" framing that anchors the router's 1.2.0 positioning (`SKILL.md:14`).

---

## 2. Domain & Coverage

### Stated scope (from router `SKILL.md:13-21` and `SKILL.md:37-39`)

In-scope:
- Optimizers, schedules, precision, batch size, gradients, loss objectives, regularization, in-flight experiment tracking.
- Modern landscape (Lion/Sophia/Muon/AdEMAMix/Schedule-Free/Prodigy, AdamW8bit/paged, WSD, FP8/BF16, muP, ZeRO/FSDP strategy).

Out-of-scope (cross-pack handoff, explicitly stated):
- PyTorch API and infrastructure → `yzmir-pytorch-engineering`.
- Architecture selection → `yzmir-neural-architectures`.
- Production registry / lineage / deployment → `yzmir-ml-production`.
- RL-specific training → `yzmir-deep-rl`.
- Preference-tuning method choice (DPO/GRPO/SimPO/ORPO/KTO/IPO/RLHF) → `yzmir-llm-specialist`.

The router's "frozen vocabulary" section (`SKILL.md:16-20`) is a strong feature: it explicitly defines the boundary between training-dynamics decisions (here) and preference-method choice (llm-specialist).

### Coverage map vs. inventory

**Foundational**
- Optimizer selection — Exists (`optimization-algorithms.md`)
- LR scheduling — Exists (`learning-rate-scheduling.md`)
- Loss functions — Exists (`loss-functions-and-objectives.md`)
- Gradient management — Exists (`gradient-management.md`)
- Batch size + precision — Exists (`batch-size-and-memory-tradeoffs.md`, precision lives here per `SKILL.md:226`/`SKILL.md:531`)
- Overfitting / regularization — Exists (`overfitting-prevention.md`)
- Data augmentation — Exists (`data-augmentation-strategies.md`)
- Training loop architecture — Exists (`training-loop-architecture.md`)

**Core**
- Hyperparameter tuning (incl. muP) — Exists (`hyperparameter-tuning.md`)
- Experiment tracking (in-flight) — Exists (`experiment-tracking.md`)

**Advanced / modern (2026-05)**
- Lion / Sophia / Muon / Shampoo / AdEMAMix / Schedule-Free / Prodigy — Routed to `optimization-algorithms.md` (verified at `optimization-algorithms.md:48-72`).
- 8-bit / paged optimizers — Routed to `optimization-algorithms.md`.
- WSD / warmup-stable-decay — Routed to `learning-rate-scheduling.md`.
- FP8 / BF16 / FP16 precision strategy — Routed to `batch-size-and-memory-tradeoffs.md`.
- muP / mu-Transfer — Routed to `hyperparameter-tuning.md`.
- ZeRO / FSDP strategy choice (the *what*, not the API) — Routed to `optimization-algorithms.md` + `batch-size-and-memory-tradeoffs.md`, with `yzmir-pytorch-engineering` for API.

**Gaps:**
- No dedicated precision sheet. Router acknowledges this explicitly (`SKILL.md:226`, `SKILL.md:532`): "No separate precision sheet exists; precision lives here [in batch-size-and-memory-tradeoffs.md]." This is a deliberate organizational choice, not an oversight, and is documented in two places. Acceptable.
- No dedicated checkpointing / resumption sheet. Some checkpointing guidance lives inside `training-loop-architecture.md`. Not a gap given pack scope, but a "checkpointing-and-resumption" sheet would be a clean future addition for very-long-run training.
- Distributed-strategy decision is split across `optimization-algorithms.md` + `batch-size-and-memory-tradeoffs.md`. Workable, but a future "distributed-strategy" sheet would consolidate the ZeRO/FSDP/3D-parallelism choice; documented as a router decision, not a current gap.

### Research currency

Domain is **evolving** (AI/ML training). The pack is explicitly calibrated to early-2026 (`SKILL.md:14`) and incorporates Lion, Sophia, Muon, AdEMAMix, Schedule-Free, Prodigy, WSD, FP8 (E4M3/E5M2), muP, ZeRO/FSDP, and DPO/GRPO/SimPO method-vs-dynamics split. This is current. The 1.2.0 description (`plugin.json`) and the router both demonstrate awareness of the 2025–2026 optimizer/precision landscape.

---

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Rating | Evidence |
|---|---|---|---|
| 1 | **Discoverability** (description, "Use when..." convention) | Minor issue | Router description (`SKILL.md:3`) does not start with "Use when..."; reads as a catalog blurb. Most peer routers in this marketplace use the "Use when..." trigger phrasing. Plugin.json description starts with the noun "Training stability" — also not "Use when..." pattern. Functional but inconsistent. |
| 2 | **Domain coverage** (vs. coverage map) | Pass | All foundational + core + advanced 2026-era topics routed. Precision-sheet absence is documented and intentional. |
| 3 | **Component-type alignment** (skill vs command vs agent vs hook) | Pass | Router = routing skill (correct). Commands (`/setup`, `/diagnose`, `/check-gradients`) are explicit user-invocable actions (correct). Agents (config reviewer, diagnostician) are autonomous specialists (correct). No hooks needed for this domain. |
| 4 | **Router accuracy** (does it route to the right place under symptoms) | Pass | Symptom tables explicit (`SKILL.md:62-218`); diagnostic-question protocol enforced; multi-skill routing called out for cross-cutting scenarios; common-mistake table present (`SKILL.md:367-381`). |
| 5 | **Pressure resistance** (rationalizations addressed) | Pass | Three pressure-resistance sections (Time/Authority/Self-Diagnosis, `SKILL.md:452-486`), a self-check before routing (`SKILL.md:488-520`), and an explicit "Don't recommend Lion because it's modern" anti-pattern (`SKILL.md:423`). |
| 6 | **Cross-pack boundaries** | Pass — exemplary | Frozen-vocabulary section (`SKILL.md:16-20`) is a model for the marketplace. Boundaries with pytorch-engineering, llm-specialist, ml-production, neural-architectures, deep-rl all documented bidirectionally (`SKILL.md:564-571`). DPO-method-vs-optimizer-choice split is called out in three places. |
| 7 | **Agent / SME-protocol compliance** | Pass | Both agents declare SME protocol in description and body. Both require the four canonical sections. Model selection appropriate (haiku for mechanical config review, sonnet for diagnostic reasoning). No spurious `tools:` declarations. |
| 8 | **Slash-command wrapper alignment** | Major issue | `.claude/commands/training-optimization.md` exists but is **stale relative to the 1.2.0 router**. Missing all 2026-05 modern-landscape content. Skill catalog descriptions are pre-1.2.0. See Findings §M1. |

**Overall: Minor — Pass with one Major fix (slash-command wrapper) plus a small number of metadata + convention nits.** Plugin is structurally sound, comprehensively covers its declared scope, has strong router/agent/command discipline. The Major issue is documentation drift between two surfaces of the same router, not a structural defect in the pack itself.

---

## 4. Behavioral Tests

**Note.** Per task brief, I performed read-and-reason behavioral checks (Stage 3 desk review) rather than subagent dispatch, since this is report-only. Where I describe a "test," I am tracing what the router would tell Claude under that scenario.

### Test S1 — Router pressure: "Demo tomorrow, just give me an optimizer for my transformer"

**Expected behavior.** Router should resist direct-recommendation pressure and ask one diagnostic question.

**Trace.** `SKILL.md:454-462` (Time/Emergency table) provides an explicit script: "30-second clarification ensures right choice: [question]". `SKILL.md:189-211` ("Which X Should I Use?") routes the user to `optimization-algorithms.md` rather than answering inline. The self-check at `SKILL.md:488-520` requires asking "Did I identify specific symptoms?" before any advice.

**Result.** Pass. The skill provides explicit, repeated, table-form scripts for this exact pressure.

### Test S2 — Router pressure: "Use Lion, it's modern and faster"

**Expected behavior.** Router should not capitulate to modernity-as-argument; should require symptom evidence.

**Trace.** `SKILL.md:14` frames the boring-correct default (AdamW + cosine + BF16). `SKILL.md:357` clarifying-question table: "What symptom is making you consider switching from AdamW?" `SKILL.md:375` common-mistake table: "Use Lion because it's modern" → "Modernity is not a reason; symptom + evidence is." `SKILL.md:423` red-flag list: "Recommend a modern optimizer (Lion/Sophia/Muon) because it's 'newer' → Route to optimization-algorithms and consult the comparison table."

**Result.** Pass. Three independent anti-pattern callouts addressing this specific failure mode.

### Test S3 — Router edge case: "My DPO run is unstable, what optimizer should I use?"

**Expected behavior.** Cross-pack split — preference-tuning method (β, reference model, pair quality) is llm-specialist; dynamics (gradient clipping, LR, optimizer) is here. Must route to *both*.

**Trace.** `SKILL.md:331-345` (Scenario: Preference Tuning) explicitly: "1. llm-specialist (PRIMARY) ... 2. optimization-algorithms (here) ... 3. learning-rate-scheduling (here) ... 4. gradient-management (here) ... 5. batch-size-and-memory-tradeoffs (here)." `SKILL.md:343-345` routing rule: "if optimizer-only diagnosis is given, that's wrong — never assume optimizer choice alone fixes a preference-tuning instability." Also flagged at `SKILL.md:359` clarifying-question table and `SKILL.md:376` common-mistake table.

**Result.** Pass. The cross-pack split is the cleanest example I've seen of router-level discipline against single-pack myopia.

### Test S4 — Command `/diagnose`: NaN loss after 100 steps

**Expected behavior.** Command should walk the user through symptom identification, ask diagnostic questions, and route to `gradient-management.md` (primary) without random hyperparameter guessing.

**Trace.** `commands/diagnose.md:84-101` (Step 2D: Loss NaN/Inf) provides four diagnostic questions and routes to `gradient-management.md` (PRIMARY) + `loss-functions-and-objectives.md`. Anti-pattern at `commands/diagnose.md:165-171` ("I'll try different optimizers" → "What's the loss behavior? Let's diagnose first").

**Caveat.** The command's NaN section does not call out the FP8/AMP-loss-scale failure mode that the router introduced in 1.2.0 (`SKILL.md:79`, `SKILL.md:108-118`). For a 1.2.0 pack with FP8 explicitly in the description, this is a minor coverage gap. See Findings §m3.

**Result.** Mostly Pass with one minor (m3).

### Test S5 — Command `/setup`: New training run

**Expected behavior.** Should give task-typed defaults and route to specialists, not pretend to be a substitute for the reference sheets.

**Trace.** `commands/setup.md:19-60` task-type table + optimizer-default table is accurate but pre-modern (no AdEMAMix / Schedule-Free / Lion / Sophia option even as a comment, no FP8/BF16 mention as a default, no 8-bit-optimizer fallback for memory-constrained fine-tunes). The command does load the router for follow-up (`commands/setup.md:228-231`), so the user *can* reach modern content — but the command's own defaults table is frozen at 2024-era choices.

**Result.** Pass on routing; Minor on currency. See Findings §m4.

### Test S6 — Agent `training-config-reviewer`: Adam with weight_decay = 0.01

**Expected behavior.** Catch Adam-vs-AdamW weight-decay bug; recommend AdamW. SME-protocol output (Confidence / Risk / Information Gaps / Caveats).

**Trace.** `agents/training-config-reviewer.md:72-86` red-flag table is explicit and correct. Protocol-required sections enforced at `agents/training-config-reviewer.md:10`.

**Result.** Pass.

### Test S7 — Agent `training-diagnostician`: "Just tell me which optimizer to use"

**Expected behavior.** Refuse direct recommendation; diagnose first.

**Trace.** `agents/training-diagnostician.md:294-300` pressure-resistance table: "Just tell me which optimizer" → "What's your loss behavior? Optimizer is rarely the issue." `agents/training-diagnostician.md:225-231` common-misdiagnoses table.

**Result.** Pass.

### Test S8 — Wrapper-vs-router alignment under user reading

**Expected behavior.** If a user runs `/training-optimization` (slash command) the loaded content should be at parity with the 1.2.0 SKILL.md.

**Trace.** Wrapper at `.claude/commands/training-optimization.md` is the pre-1.2.0 generation. A user typing `/training-optimization` would see no Lion/Sophia/Muon/FP8/WSD/muP/ZeRO routing. They would have to know to load the plugin's actual router skill to get the modern content.

**Result.** Fail (Major). See Findings §M1.

---

## 5. Findings (Critical / Major / Minor / Polish)

### Critical

None.

### Major

#### M1. Slash-command wrapper `.claude/commands/training-optimization.md` is out of sync with `SKILL.md` 1.2.0

**Evidence.**
- Wrapper (`.claude/commands/training-optimization.md:1-467`) is the pre-1.2.0 router.
- SKILL.md (`plugins/yzmir-training-optimization/skills/using-training-optimization/SKILL.md:1-583`) is 1.2.0.
- Specific missing content in wrapper:
  - No modern-optimizer routing section (router has it at `SKILL.md:215-235`).
  - No "Knowledge cutoff (2026-05)" frame (`SKILL.md:14`).
  - No frozen-vocabulary cross-pack section (`SKILL.md:16-20`).
  - No precision-as-routable-symptom (FP8/BF16) — wrapper's `SKILL.md`-equivalent NaN table at `commands/training-optimization.md:39-45` omits the FP8/AMP loss-scale call-out.
  - No muP / mu-Transfer routing row.
  - No ZeRO/FSDP-strategy-vs-API split.
  - No DPO/GRPO/SimPO cross-pack scenario; no llm-specialist routing.
  - Skill catalog descriptions (`.claude/commands/training-optimization.md:409-422`) are pre-modern ("SGD vs Adam vs AdamW, momentum, optimizer comparison" vs. router's "modern landscape (Lion, Sophia, AdEMAMix, Muon, Shampoo, Schedule-Free, Prodigy, AdamW8bit / paged optimizers), and ZeRO/FSDP strategy choice", `SKILL.md:528`).

**Impact.** Users invoking `/training-optimization` get a stale routing surface and may be steered to legacy optimizer-only thinking, while users who load the plugin's router skill directly get the 1.2.0 modernized content. Two parallel sources of truth, drifting.

**Fix.** Replace the body of `.claude/commands/training-optimization.md` with content matching the 1.2.0 `SKILL.md`, **or** restructure the wrapper to be a thin pointer that loads the plugin's router skill rather than duplicating its body. The thin-pointer pattern is what `meta-skillpack-maintenance:using-skillpack-maintenance:SKILL.md:206-238` documents as the marketplace convention — most wrappers in this repo are not full re-copies of the router. (Verification of repo norm not performed during this review; sampling one or two other wrappers would confirm before applying.)

#### M2. Marketplace catalog description for this plugin is stale (1.0-era)

**Evidence.** `.claude-plugin/marketplace.json` (line containing `"name": "yzmir-training-optimization"`): description reads `"Training stability - optimizers, learning rates, convergence, debugging - 11 skills"`. Plugin.json's much richer 1.2.0 description (mentioning AdamW/Lion/Sophia/Muon/AdEMAMix/Schedule-Free/Prodigy/8-bit Adam, BF16/FP8, B_crit/Chinchilla, muP) is not reflected in the marketplace catalog.

**Impact.** Users browsing the marketplace see no signal that the pack covers modern optimizers / FP8 / muP / etc.

**Fix.** Sync the marketplace.json description to match the plugin.json description (or a condensed form of it).

### Minor

#### m1. Router `description:` does not follow the "Use when…" convention

**Evidence.** `SKILL.md:3` reads: `description: Router to training optimization skills based on symptoms and training problems`. The marketplace convention documented in `meta-skillpack-maintenance:using-skillpack-maintenance:SKILL.md:133-135` is that descriptions should typically start with "Use when …" for skill-discovery activation.

**Impact.** Reduces discovery hits when Claude is scanning skill descriptions for trigger phrases. Functional impact small because the slash command bypasses description-based discovery, but the in-process Skill-tool discovery path is degraded.

**Fix.** Rephrase to "Use when encountering training problems (loss not decreasing, instability, overfitting, slow training, optimizer / LR / batch / precision selection) — routes to specialist sheets …".

#### m2. Plugin.json + marketplace.json count "11 skills" but `find skills -name SKILL.md` returns 1

**Evidence.** `plugin.json` claims "11 skills + 3 commands + 2 agents"; marketplace.json says "11 skills"; filesystem has 1 router SKILL.md + 10 sibling reference sheets.

**Impact.** Not strictly wrong — the convention "1 router + 10 sheets = 11 skills" is defensible and is how the pack markets itself internally and externally. But it is at odds with strict filesystem-based counts a maintenance tool would compute. Document the counting convention or move to "10 sheets routed by 1 skill" phrasing for consistency.

**Fix.** Either (a) restate as "1 router skill + 10 reference sheets" in both metadata files, or (b) leave as-is and document the convention in the plugin's README (if/when one is added). Low priority; primarily a Polish concern.

#### m3. `/diagnose` NaN section omits FP8/AMP loss-scale failure mode

**Evidence.** `commands/diagnose.md:84-101` (Step 2D: Loss NaN/Inf). The cause table lists "Using AMP — Gradient scaling issue — Use GradScaler properly" but does not separately call out FP8 underflow / loss-scale tuning, which the 1.2.0 router elevates (`SKILL.md:79`, `SKILL.md:108-118`).

**Impact.** A user running FP8 mixed precision and hitting NaN would not get the precision-aware diagnostic path from this command unless they then load the router.

**Fix.** Add an FP8/E4M3/E5M2 / loss-scale row to the cause table in Step 2D, and route to `batch-size-and-memory-tradeoffs.md` (precision section) in addition to `gradient-management.md`.

#### m4. `/setup` defaults table is pre-modern (no Lion/Sophia/AdEMAMix/Schedule-Free/Prodigy/8-bit options; no FP8/BF16 default callout)

**Evidence.** `commands/setup.md:30-60` recommends AdamW/SGD/Adam only. No mention of:
- BF16 mixed precision as a default (the router calls it "boring-correct" at `SKILL.md:14`).
- 8-bit / paged optimizers as a memory-constrained fine-tune fallback.
- WSD schedule as an alternative to cosine for continual / resumable training.

**Impact.** Users running `/setup` get a 2024-era template that won't surface modern alternatives. The command does eventually point to the router (`commands/setup.md:228-231`), so the content is reachable, but the command-as-entry-point experience is dated.

**Fix.** Either (a) add a short "Modern alternatives (consider with cause)" subsection to the optimizer table, or (b) explicitly note that the boring-correct default is AdamW + cosine + BF16 and route to the router for the modern-alternative decision. Option (b) is lighter-touch and preserves the command's "starter template" character.

### Polish

#### p1. Router section ordering: "How to Access Reference Sheets" (`SKILL.md:43-56`) appears before the routing tables

This is functionally fine and is a repo-wide pattern, but the section reads as boilerplate; could be folded into a single short note at the top.

#### p2. Quick Reference symptom→skill table (`SKILL.md:541-560`) duplicates some content from earlier symptom sections

Mild redundancy. A 583-line router that includes both narrative routing-by-symptom and a tabular quick-reference is fine for human readers; for Claude, the duplication is harmless but adds context tokens.

#### p3. Two reference sheets are exactly 1162 lines

Coincidence, but worth noting: `optimization-algorithms.md` and `learning-rate-scheduling.md` are byte-aligned-suspiciously identical in length. Spot-check confirmed they are different files with different content — no actual issue, but worth a one-line diff to confirm no accidental sibling overwrite during edits.

---

## 6. Recommended Actions

| # | Action | Severity | Effort | Owner |
|---|---|---|---|---|
| A1 | Update `.claude/commands/training-optimization.md` to 1.2.0 parity, OR convert it into a thin pointer to the router skill. Sample 2–3 peer wrappers in `.claude/commands/` first to confirm marketplace norm. | Major (M1) | Medium (re-copy + verify) or Small (thin-pointer rewrite) | Maintainer |
| A2 | Sync `.claude-plugin/marketplace.json` description for `yzmir-training-optimization` to match the 1.2.0 plugin.json. | Major (M2) | Trivial | Maintainer |
| A3 | Rephrase router `description:` to start with "Use when …" per marketplace convention. | Minor (m1) | Trivial | Maintainer |
| A4 | Add FP8 / AMP loss-scale row to `/diagnose` NaN section; route to batch-size-and-memory-tradeoffs.md (precision) in addition to gradient-management.md. | Minor (m3) | Small | Maintainer |
| A5 | Add a short "modern alternatives — consider with cause" note to `/setup` optimizer table, plus BF16-as-default callout. | Minor (m4) | Small | Maintainer |
| A6 | Decide whether "11 skills" stays in metadata or is rephrased as "1 router + 10 reference sheets". Either is acceptable; pick once and apply consistently. | Minor (m2) / Polish | Trivial | Maintainer |
| A7 | (Optional) Behavioral test via subagent dispatch: spin up two fresh-context subagents — one with the 1.2.0 SKILL.md and one with the stale wrapper — give both the prompt "I want to train a 7B transformer with FP8. What optimizer, LR, and schedule?" and compare. Expected: SKILL.md routes via FP8/BF16/Lion/Sophia/muP; wrapper does not. This converts the desk-review finding into a reproducible behavioral failure. | Verification | Small | Reviewer / Maintainer |

**Suggested version bump.** If A1 and A2 are applied together (sync wrapper + sync marketplace description), this is a content/documentation fix with no philosophy change — patch bump to **1.2.1**. If A4 + A5 are also applied (updating two commands with modern content), call it a minor enhancement and bump to **1.3.0**. Per `meta-skillpack-maintenance:using-skillpack-maintenance:SKILL.md:244-249`, the default for maintenance is minor.

---

## 7. Reviewer Notes

### What this pack does very well

1. **Cross-pack discipline is exemplary.** The frozen-vocabulary section (`SKILL.md:16-20`) and the DPO/GRPO method-vs-dynamics split (`SKILL.md:331-345`, `SKILL.md:376`) are the cleanest examples of router-level boundary discipline I encountered in this review. They directly address the "single-pack myopia" failure mode the marketplace as a whole is vulnerable to.
2. **Pressure resistance is enforced in three layered ways** (rationalization table, pressure tables, self-check), at three layers (router, both agents, `/diagnose` command). A user trying to extract a quick optimizer answer under time pressure hits four independent guardrails.
3. **Modernization is genuine, not cosmetic.** The 1.2.0 description-level claims (Lion, Sophia, Muon, AdEMAMix, Schedule-Free, Prodigy, 8-bit/paged optimizers, FP8, BF16, WSD, muP, ZeRO/FSDP-strategy, B_crit, Chinchilla) all show up in the actual routing tables and the actual specialist sheets I sampled. This is not a description-only refresh.
4. **SME-agent protocol compliance is clean.** Both agents follow the protocol exactly. Model selection (haiku for mechanical config review, sonnet for diagnostic reasoning) is well-judged.

### What this pack does adequately but could improve

1. The slash-command wrapper drift (M1) is the dominant maintenance issue and almost certainly a side-effect of updating the SKILL.md without back-syncing the user-facing entry point. A cheap structural fix would be to make the wrapper a one-pager that loads the plugin's router skill rather than duplicating its body.
2. The two commands (`/setup`, `/diagnose`) have not been refreshed alongside the router. They are still serviceable, and they do route to the modern router on follow-up, but they read as 2024-era artifacts.

### What I did NOT do

- I did not dispatch subagents for true behavioral testing (Stage 3 by subagent), per the read-and-reason constraint of this review pass. I performed desk-review-style trace analysis — adequate for surfacing the issues catalogued here, but not as load-bearing as a fresh-context subagent test would be for activation/discovery claims.
- I did not read the full text of all 10 reference sheets. I sampled `optimization-algorithms.md` (head) and `batch-size-and-memory-tradeoffs.md` (head) to confirm modernization claims at the sheet level. The other eight sheets are assumed in scope based on router-side citation and line counts; deeper sheet-level review would be a separate pass.
- I did not check that command and agent files are byte-clean (no leading blank line, no stray BOM). The line-1 blank in some files (`commands/check-gradients.md:1`, `commands/diagnose.md:1`, `commands/setup.md:1`) is normal YAML-frontmatter prelude in this repo's style.
- I did not validate `argument-hint` semantics empirically. They look right.
- I did not assess implementing-fixes guidance — explicit scope of Stage 5, skipped per task.

### Confidence

High that M1 (wrapper drift) and M2 (marketplace description drift) are real and actionable. Medium-high that m3 and m4 (command-content currency) are real but lower-impact. Medium on the count-of-skills bookkeeping (m2) — could go either way depending on marketplace convention preference. The pack is in good shape overall; the issues are documentation drift between surfaces, not structural defects.

### Caveats

- Slash-command-wrapper-norm claim ("most wrappers in this repo are not full re-copies of the router") is asserted from the `meta-skillpack-maintenance:using-skillpack-maintenance` reference (`SKILL.md:206-238`), not verified by sampling peer wrappers in this review. Confirm before applying A1.
- The version bump suggestion assumes the marketplace follows the version-bump rules table in the maintenance skill. If the marketplace uses a different cadence (e.g., per-marketplace-release semver), defer to that.

---

## Appendix A: Detailed evidence for M1 (wrapper drift)

Line-by-line spot map of router (1.2.0) sections that are absent from the wrapper.

| Router section | Router line range | Present in wrapper? |
|---|---|---|
| Knowledge cutoff (2026-05) framing + "boring-correct default" anchor | `SKILL.md:14` | No |
| Cross-pack frame (frozen vocabulary) | `SKILL.md:16-20` | No |
| When-to-use list with modern-optimizer/schedule/precision/distributed/muP triggers | `SKILL.md:33-37` | Partial — wrapper omits all modern triggers (`commands/training-optimization.md:13-22`) |
| NaN routing with FP8/AMP loss-scale callout | `SKILL.md:79` | No — wrapper's table at `commands/training-optimization.md:39-45` has no FP8/loss-scale row |
| Precision routing block (FP8/BF16/FP16) | `SKILL.md:108-118` | No |
| Speed/throughput table with FP8/BF16/ZeRO/FSDP/8-bit columns | `SKILL.md:171-186` | Partial — wrapper has the table but without precision/8-bit/strategy columns (`commands/training-optimization.md:129-141`) |
| "Which X Should I Use?" with modern landscape | `SKILL.md:189-211` | Partial — wrapper covers only SGD/Adam/AdamW (`commands/training-optimization.md:148-165`) |
| Modern-optimizer/schedule/precision routing table (Lion/Sophia/Muon/AdEMAMix/Schedule-Free/Prodigy/WSD/FP8/muP/ZeRO) | `SKILL.md:215-235` | No — section absent entirely |
| Preference-tuning cross-pack scenario | `SKILL.md:331-345` | No |
| Updated clarifying-questions table (incl. modern/DPO) | `SKILL.md:351-359` | No — wrapper has only the legacy six rows (`commands/training-optimization.md:259-265`) |
| Updated common-mistakes table (Lion/DPO/FP8/ZeRO/muP rows) | `SKILL.md:367-381` | No — wrapper has only the legacy six rows (`commands/training-optimization.md:271-278`) |
| Updated red-flag list (Lion/DPO/FP8 anti-patterns) | `SKILL.md:415-429` | No — wrapper has only the legacy seven items (`commands/training-optimization.md:310-319`) |
| Specialist skill catalog with modern landscape | `SKILL.md:524-537` | No — wrapper catalog is pre-modern (`commands/training-optimization.md:407-422`) |
| Updated Quick Reference symptom→skill (modern rows) | `SKILL.md:541-560` | No — wrapper Quick Reference has only the legacy ten rows (`commands/training-optimization.md:425-438`) |
| Updated Integration Notes (bidirectional cross-pack boundaries) | `SKILL.md:564-571` | No — wrapper has the older Phase-1 framing (`commands/training-optimization.md:441-452`) |

This appendix is the artifact a maintainer would use to actually do the wrapper rewrite — every line above is content to either copy across (if the wrapper-as-duplicate pattern is the norm) or to drop entirely (if the wrapper-as-pointer pattern is the norm).

## Appendix B: SME-protocol compliance evidence

Required elements per `meta-sme-protocol:sme-agent-protocol` (loaded via the maintenance skill's reference to it):

| Element | training-config-reviewer | training-diagnostician |
|---|---|---|
| Description ends with SME phrase | Yes (`agents/training-config-reviewer.md:2`) | Yes (`agents/training-diagnostician.md:2`) |
| Body cites `meta-sme-protocol:sme-agent-protocol` | Yes (`agents/training-config-reviewer.md:10`) | Yes (`agents/training-diagnostician.md:10`) |
| Requires READing source before reviewing | Yes — "Before reviewing, READ the config files and related training code" (`agents/training-config-reviewer.md:10`) | Yes — "Before diagnosing, READ the actual training code, config files, and loss curves" (`agents/training-diagnostician.md:10`) |
| Requires Confidence Assessment section | Yes (line 10) | Yes (line 10) |
| Requires Risk Assessment section | Yes | Yes |
| Requires Information Gaps section | Yes | Yes |
| Requires Caveats section | Yes | Yes |
| Activation examples (positive + negative) | Yes — 3 positive + 1 negative (`agents/training-config-reviewer.md:14-32`) | Yes — 3 positive + 1 negative (`agents/training-diagnostician.md:14-32`) |
| Scope boundaries section | Yes (`agents/training-config-reviewer.md:214-256`) | Yes (`agents/training-diagnostician.md:262-291`) |
| Handoff guidance to other packs | Yes — explicit Glob-check pattern for neural-architectures, pytorch-engineering, deep-rl | Yes — same pattern |
| `tools:` key audit | Not declared (inherits parent) — appropriate | Not declared (inherits parent) — appropriate |
| Model selection justification | Implicit — haiku for mechanical config review is appropriate | Implicit — sonnet for diagnostic reasoning is appropriate |

Both agents fully comply with the SME protocol and the marketplace's two-key (`description`, `model`) agent frontmatter convention.
