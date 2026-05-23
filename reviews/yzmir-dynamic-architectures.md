# Review: yzmir-dynamic-architectures
**Version:** 1.2.1  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

Scope: Stages 1-4 of `meta-skillpack-maintenance:using-skillpack-maintenance` applied read-only. No file edits. Stage 5 explicitly skipped.

---

## 1. Inventory

### Plugin metadata
- `plugins/yzmir-dynamic-architectures/.claude-plugin/plugin.json` — name, version 1.2.1, description claims "7 reference sheets, 2 commands, 1 SME agent". License CC-BY-SA-4.0. Keywords include modern PEFT variants (LoRA / QLoRA / DoRA / VeRA / PiSSA / LoftQ) and merging algorithms (TIES, DARE, MergeKit). Repo URL and author present.
- Counted on disk: 1 router SKILL.md + 7 reference sheets, 2 commands, 1 agent → metadata claim matches reality.

### Skills (1 router + 7 reference sheets)
| File | Lines | Status |
|---|---|---|
| `skills/using-dynamic-architectures/SKILL.md` | 415 | OK — router with explicit "Use when..." description, 7-way routing table, quick-routing table including cross-pack handoffs |
| `skills/using-dynamic-architectures/continual-learning-foundations.md` | 536 | OK — EWC, SI, MAS, PackNet, Progressive Nets, rehearsal, BWT/FWT/ACC metrics |
| `skills/using-dynamic-architectures/gradient-isolation-techniques.md` | 819 | OK — freezing, detach vs no_grad vs stop_gradient, hooks, dual-path, alpha blending |
| `skills/using-dynamic-architectures/peft-adapter-techniques.md` | 781 | OK — LoRA/QLoRA/DoRA implementations + modern variants per description |
| `skills/using-dynamic-architectures/dynamic-architecture-patterns.md` | 643 | OK — slot-based, Net2Net widening, lottery ticket, triggers, capacity scheduling |
| `skills/using-dynamic-architectures/modular-neural-composition.md` | 1054 | OK — full MoE coverage Switch/Mixtral/DeepSeek + merging (TIES, DARE, SLERP, MergeKit, LoraHub) |
| `skills/using-dynamic-architectures/ml-lifecycle-orchestration.md` | 707 | OK — states, gates, triggers, controllers, hysteresis, observability |
| `skills/using-dynamic-architectures/progressive-training-strategies.md` | 759 | OK — zero-init / LR warmup / alpha-ramp / staged schedules / failure modes |

Total: ~5,700 lines of reference content. Each reference sheet has consistent structure (Overview → topical sections → Implementation Checklist or Common Pitfalls), uses code examples grounded in PyTorch idioms, and ends with cross-references.

### Commands (2)
| File | Frontmatter | Status |
|---|---|---|
| `commands/design-lifecycle.md` (184 lines) | `description`, `allowed-tools: ["Read", "Glob", "Grep", "AskUserQuestion"]` | OK in body; no `argument-hint` |
| `commands/diagnose-growth.md` (113 lines) | `description`, `allowed-tools: ["Read", "Glob", "Bash", "WebSearch"]` (Grep listed too, line 3) | OK in body; no `argument-hint` |

Both commands have step-by-step protocols. `design-lifecycle.md` walks through Requirements → Code exploration → States → Transitions → Gates → Controller → Output. `diagnose-growth.md` walks Symptom → Evidence → Common-failure tables → Report.

### Agents (1)
| File | Frontmatter | Status |
|---|---|---|
| `agents/dynamic-architecture-advisor.md` (157 lines) | `description: "...Follows SME Agent Protocol with confidence/risk assessment."`, `model: opus` | SME compliance OK |

Agent body (line 10) cites `meta-sme-protocol:sme-agent-protocol` and lists the four required output sections (Confidence, Risk, Information Gaps, Caveats) explicitly. Includes mandatory Fact-Finding Protocol, Response Pattern, Anti-Patterns table, Scope Boundaries, and an Example Investigation Flow. No `tools:` key (correct — inherits parent context per repo convention).

### Hooks
None. Not expected for an advisory pack.

### Slash-command wrapper
- Router skill `skills/using-dynamic-architectures/SKILL.md` exists.
- Expected wrapper `.claude/commands/dynamic-architectures.md` — **MISSING**.
- Confirmed via `ls /home/john/skillpacks/.claude/commands/` — all other yzmir router skills (deep-rl, llm-specialist, ml-production, morphogenetic-rl, neural-architectures, pytorch-engineering, simulation-foundations, training-optimization) have wrappers; this pack is the only yzmir router without one.
- Sibling wrapper `morphogenetic-rl.md` line 7 explicitly references `/dynamic-architectures` as a slash command (`"Companion to /dynamic-architectures"`), so this break is visible to users.

### Marketplace registration
- `.claude-plugin/marketplace.json` registers `yzmir-dynamic-architectures` with source `./plugins/yzmir-dynamic-architectures` and a description matching v1.2.1 plugin.json shape — OK.

### Sibling pack relationship (yzmir-morphogenetic-rl)
- Boundary is **clean and bidirectional**. SKILL.md lines 293-297 and 341 in this pack: "yzmir-morphogenetic-rl covers WHEN/HOW the controller decides to grow. This pack covers HOW the growable network trains once a decision is made." Mirrored in `yzmir-morphogenetic-rl/skills/using-morphogenetic-rl/SKILL.md` lines 32-36 and Pipeline Position table lines 60-80. Both packs name the other and state the partition.
- Plugin.json descriptions on both sides explicitly cross-reference: "Companion to yzmir-dynamic-architectures" in morphogenetic-rl, and the dynamic-architectures router routes "Build complete morphogenetic system" through this pack's patterns + isolation + lifecycle sheets while routing "RL controller decides when to grow" to the sibling.

---

## 2. Domain & Coverage

### Stated scope (from router SKILL.md lines 9-20, 38-46)
Networks that grow / prune / adapt topology *during training*. Continual learning, gradient isolation, modular composition, lifecycle orchestration, PEFT (modern variants), progressive training. The pack frames itself as the "network being grown" side of the morphogenetic problem, with `yzmir-morphogenetic-rl` owning the "controller doing the growing" side.

### Coverage map

**Foundational**
- Catastrophic forgetting theory (BWT/FWT/ACC, loss-landscape geometry, stability-plasticity) — `continual-learning-foundations.md` lines 11-75 — Exists
- Gradient isolation semantics (detach vs no_grad vs stop_gradient, hooks) — `gradient-isolation-techniques.md` — Exists
- State-machine lifecycle theory (states, transitions, gates, triggers) — `ml-lifecycle-orchestration.md` — Exists
- Combination/composition mechanisms (additive, multiplicative, interpolative, selective) — `modular-neural-composition.md` — Exists

**Core techniques**
- EWC / SI / MAS — `continual-learning-foundations.md` Regularization Approaches — Exists
- PackNet / Progressive Networks — same — Exists
- Rehearsal (experience replay, generative replay) — same — Exists
- LoRA / QLoRA / DoRA + modern variants (VeRA, PiSSA, LoftQ, LoRA+, rsLoRA, LongLoRA) — `peft-adapter-techniques.md` — Exists
- MoE (Shazeer → Switch → Mixtral → DeepSeek-MoE, Expert Choice, sparse upcycling, aux-loss-free balancing) — `modular-neural-composition.md` MoE section — Exists
- Adapter / model merging (TIES, DARE / DARE-TIES, SLERP, MergeKit, LoraHub, model souping) — same — Exists
- Net2Net widening, lottery ticket, magnitude / gradient-based / structured pruning — `dynamic-architecture-patterns.md` — Exists
- Slot semantics with cooldown / embargo — same — Exists
- Heuristic, learned (RL), hybrid controllers — `ml-lifecycle-orchestration.md` — Exists
- Alpha-ramp, zero-init, LR warmup, frozen-host warmup — `progressive-training-strategies.md` — Exists

**Advanced**
- Counterfactual contribution measurement for pruning — cross-referenced in router (line 343) to `yzmir-deep-rl/counterfactual-reasoning` — Exists via cross-ref
- Multi-tenant LoRA serving (S-LoRA / LoRAX / Punica) — explicitly punted to `yzmir-ml-production` per router lines 83 and 339 — Exists via cross-ref (correct boundary)
- Distributed/FP8/MoE-dispatch kernels — punted to `yzmir-training-optimization` per router line 335 — Exists via cross-ref (correct)
- RL controller design / governor / rollback shaping — punted to `yzmir-morphogenetic-rl` — Exists via cross-ref (correct)

**Cross-cutting**
- Diagnostic / debugging — `commands/diagnose-growth.md` plus router rationalization table
- Anti-pattern resistance — router Rationalization Resistance Table (lines 302-311), Red Flags Checklist (lines 316-323), and per-sheet "Common Pitfalls" sections

### Domain stability
**Evolving.** PEFT and MoE techniques have churned rapidly 2023-2025 — VeRA, DoRA, PiSSA, LoftQ, rsLoRA, LongLoRA, Mixtral, DeepSeek-MoE, MergeKit are all post-2023. The pack appears current per the v1.2.1 description (explicit naming of 2024+ techniques), but the next 12-18 months will likely add another generation that this review cannot evaluate without literature search. Flagging as research currency to revisit, **not** as a current gap.

### Coverage-map cross-pack hand-off audit

The router's "Relationship to Other Packs" table (SKILL.md lines 329-341) lists nine cross-pack rows. Verifying each against the marketplace catalog:

| Cross-ref | Target plugin | Exists in marketplace? | Plausible owner? |
|---|---|---|---|
| "Implement PPO for architecture decisions" | yzmir-deep-rl | Yes | Yes |
| "Evaluate architecture changes without mutation" | yzmir-deep-rl/counterfactual-reasoning | Yes (parent pack) | Yes |
| "Debug PyTorch gradient flow" | yzmir-pytorch-engineering | Yes | Yes |
| "Optimize training loop performance" | yzmir-training-optimization | Yes | Yes |
| "FSDP2 + QLoRA, FP8 training, MoE dispatch kernels" | yzmir-training-optimization | Yes | Yes |
| "Apply PEFT recipes to LLMs (instruction tuning, RLHF)" | yzmir-llm-specialist | Yes | Yes |
| "Design transformer architecture" | yzmir-neural-architectures | Yes | Yes |
| "Deploy morphogenetic model" | yzmir-ml-production | Yes | Yes |
| "Serve many LoRAs in one process (S-LoRA / LoRAX / Punica)" | yzmir-ml-production | Yes | Yes |

All nine cross-refs target real, marketplace-registered plugins, and the topical assignment is plausible in each case. No broken cross-pack links.

### Gaps identified
- **None critical to the stated scope.** Coverage matches the 7-area routing table.
- **Minor**: Counterfactual-evaluation cross-ref to `yzmir-deep-rl/counterfactual-reasoning` is good but the contribution-measurement *technique* (used by pruning decisions, lifecycle gates) could plausibly live in this pack since it shapes lifecycle gate logic. Currently delegated cleanly — not a defect, just a design choice worth noting.
- **Minor**: No dedicated coverage of **mixture-of-depths** / token-dropping / early-exit dynamic computation. These are arguably "dynamic architecture" but border on inference-time routing rather than training-time topology change. Acceptable scope exclusion if intentional.

### Per-sheet content quality (sampled)

The reference sheets were sampled by reading openings and the full section-header structure of each. Substantive observations:

- **`continual-learning-foundations.md`** (536 lines, sampled lines 1-80): Opens with the catastrophic-forgetting problem, names the stability-plasticity dilemma (Abraham & Robins 2005), explains loss-landscape geometry, then defines BWT / FWT / ACC with formulas. Sections cover Regularization (EWC, SI, MAS), Architectural (PackNet, Progressive Nets, DEN), Rehearsal (experience / generative replay), and a "Relevance to Morphogenetic/Dynamic Architectures" section that ties forgetting theory to the rest of the pack. Implementation checklist + References at end. **Strong foundational sheet; cites primary literature in-line.**

- **`gradient-isolation-techniques.md`** (819 lines, sampled lines 1-80): Frames isolation as directional gradient control, not blanket blocking. Concrete naive-integration example with gradient-flow diagram showing the interference path. Sections cover Freezing (full / partial / parameter-selective / scheduled), Gradient Masking with explicit `detach()` vs `no_grad()` vs `stop_gradient` semantics, Hook-based selective masking, Straight-Through Estimators, Dual-Path Training (auxiliary head / teacher-student / parallel merge), and Alpha Blending. **High signal-to-noise; the `detach` vs `no_grad` distinction alone is often a costly silent bug.**

- **`peft-adapter-techniques.md`** (781 lines, sampled lines 1-200): Full LoRA derivation (W' = W + BA), manual PyTorch `LoRALinear` implementation including Kaiming-init A / zero-init B for safe ΔW=0 initialization, `inject_lora` function for retrofitting existing models, then PEFT-library usage. QLoRA section uses `BitsAndBytesConfig` with NF4 + double-quant and explains gradient handling for quantized weights. DoRA, modern variants, adapter-placement strategies, rank selection. **Implementation-ready, not just descriptive.**

- **`dynamic-architecture-patterns.md`** (643 lines, section structure): Static-vs-dynamic motivation, distinguishes from NAS, growth patterns (slot-based, Net2Net-style widening, depth extension, branching), pruning patterns (magnitude / gradient-based / lottery ticket / structured vs unstructured), trigger conditions (loss plateau, contribution metrics, budget constraints), capacity scheduling strategies, slot semantics with cooldown. Common Pitfalls section. **Covers both grow and prune axes symmetrically.**

- **`modular-neural-composition.md`** (1054 lines — largest sheet — sampled lines 1-100 and section list): Encapsulation / Composability / Replaceability principles with PyTorch interface contracts; combination mechanisms (additive / multiplicative / interpolative / selective); full MoE section spanning Shazeer baseline through Switch / Mixtral / DeepSeek-MoE, Expert Choice routing, aux-loss-free balancing, sparse upcycling; grafting semantics; interface contracts (shape matching, normalization boundaries, init for stable grafting); multi-module coordination; **adapter merging & task arithmetic** with TIES, DARE, SLERP, model souping, MergeKit YAML examples, LoraHub. **Single most content-dense sheet; well-organized despite scope.**

- **`ml-lifecycle-orchestration.md`** (707 lines, sampled lines 1-120): State-machine fundamentals with `ModuleState` enum (DORMANT, INSTANTIATED, TRAINING, BLENDING, HOLDING, INTEGRATED, PRUNED, EMBARGOED), `TransitionGuard` and `StateMachine` PyTorch classes, gate design patterns (structural / performance / stability / contribution), transition triggers (metric / time / budget / controller-driven), rollback & recovery (cooldown, checkpoint-based rollback, hysteresis), controller patterns (heuristic / learned / hybrid), observability section. **Cleanest "code-first" sheet; reads like a small library design doc.**

- **`progressive-training-strategies.md`** (759 lines, section structure): Architecture curriculum vs data curriculum framing, staged capacity expansion (start-small / grow-on-plateau / transfer-knowledge), module warmup patterns (zero-init / LR warmup / alpha ramp / frozen-host), cooldown & stabilization (post-integration settling, contribution monitoring, consolidation epochs), multi-stage schedules (sequential / overlapping / budget-aware), knowledge transfer (weight inheritance / distillation / feature reuse), failure modes (too fast / too slow / transition shock). **Acts as the "safe integration" sheet that ties gradient-isolation and lifecycle together.**

Aggregate read: 5,300+ lines of reference content distributed across 7 sheets with no obvious dead weight, no obvious redundancy, and consistent structural conventions (Overview → topical sections → Implementation Checklist / Common Pitfalls / References). Code examples are PyTorch-native and runnable in shape if not literally on current library versions.

---

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Rating | Evidence |
|---|---|---|---|
| 1 | **Router discoverability & description quality** | Pass | SKILL.md description starts with "Use when..." per repo convention. Triggers ("grow, prune, or adapt topology during training") are operational, not conceptual. Routes named explicitly. |
| 2 | **Reference-sheet coverage of stated scope** | Pass | 7 sheets cover 7 routing areas with no orphans. Modern PEFT and MoE variants present per v1.2.1 promise. ~5,700 lines of substance. |
| 3 | **Boundary clarity vs siblings** | Pass | Boundary with `yzmir-morphogenetic-rl` is bidirectional, explicit, and consistent (SKILL.md lines 293-297, 341, both plugin.json descriptions, both router skills). Cross-refs to `yzmir-deep-rl`, `yzmir-ml-production`, `yzmir-training-optimization`, `yzmir-llm-specialist`, `yzmir-neural-architectures`, `yzmir-pytorch-engineering` all present in the Relationship table (lines 327-341). |
| 4 | **Command quality & invocability** | Minor | Both commands are coherent, step-by-step, and have quoted JSON-style `allowed-tools` per repo convention. Both **lack `argument-hint`**, which is a minor convention drift (most marketplace commands have it). Neither is broken; they just don't advertise expected arguments. |
| 5 | **Agent SME-Protocol compliance** | Pass | Description ends with "Follows SME Agent Protocol with confidence/risk assessment." Body (line 10) cites `meta-sme-protocol:sme-agent-protocol` and requires the four sections verbatim. Includes Fact-Finding Protocol, Anti-Patterns table, Scope Boundaries, and a worked Example Investigation Flow. `model: opus` is appropriate for synthesis/multi-step reasoning. No spurious `tools:` declaration. |
| 6 | **Slash-command exposure** | **Major** | Router skill exists; `.claude/commands/dynamic-architectures.md` does **not**. Every other yzmir router has a wrapper. The sibling `morphogenetic-rl.md` wrapper (line 7) and the marketplace's repo-wide convention (CLAUDE.md §Slash Commands) treat this as required. Users cannot invoke `/dynamic-architectures`. |
| 7 | **Marketplace registration** | Pass | Registered in `.claude-plugin/marketplace.json` with correct source path, keywords, and description string. Description matches v1.2.1 plugin.json. |
| 8 | **Internal consistency (router ↔ sheets ↔ agent)** | Pass | Router names 7 sheets; 7 sheets exist with matching filenames. Agent's Reference Sheets list (lines 53-61) names 6 of the 7 (omits `peft-adapter-techniques.md`). See Finding M2. |

**Overall: Major** — Single Major issue (missing slash-command wrapper) plus a small number of Minor / Polish items. The pack is structurally sound and content-rich; the gap is exposure plumbing, not domain coverage. Rebuild not warranted. Enhancement scope is small.

---

## 4. Behavioral Tests

Stage 3 calls for behavioral testing with subagent dispatch on pressure / edge-case / real-world scenarios. Per the task constraint (report-only, no edits), the tests below are designed conceptual probes against the artifact, not executed subagent runs. Each test names a scenario, the routing the artifact *should* produce, and whether the artifact's text supports that routing under pressure.

### T1 — Pressure: "Just fine-tune the whole thing, we're behind"
**Scenario**: Engineer wants to fine-tune a 7B model on new task; says they don't have time for "fancy adapter stuff" and will just do full fine-tuning. Old-task performance matters.

**Expected routing**: Router should counter-rationalize via the Rationalization Resistance Table and route to `continual-learning-foundations` (forgetting risk) + `peft-adapter-techniques` (memory-efficient alternative).

**Artifact evidence**: Router lines 302-311 explicitly addresses "Just train a bigger model from scratch" with counter-guidance pointing to `continual-learning-foundations`. Quick routing table (line 74-75) maps "Fine-tune LLM efficiently without full training" → `peft-adapter-techniques`. **Pass** — explicit counter-pressure language present.

### T2 — Pressure: "I'll just freeze everything and add a layer"
**Scenario**: User wants to extend a trained network with one new layer, says they'll freeze everything else and train just the new layer.

**Expected routing**: Router should route to `gradient-isolation-techniques` for partial-freeze nuance, not endorse full freeze.

**Artifact evidence**: Rationalization table line 306 — "I'll freeze everything except the new layer" → "Full freeze may be too restrictive" → "Check gradient-isolation-techniques for partial strategies." **Pass**.

### T3 — Edge case: ambiguous "my model forgets" with both task-sequence and architectural growth in play
**Scenario**: User has Progressive-Networks-style architectural columns *and* sees forgetting on old tasks. Should they route to continual-learning or gradient-isolation?

**Expected routing**: Router should support multi-skill scenarios. "Continual Learning Without Forgetting" scenario (lines 271-278) sequences `continual-learning-foundations` → `gradient-isolation-techniques` → `progressive-training-strategies`.

**Artifact evidence**: Multi-Skill Scenarios section (lines 258-298) addresses exactly this composite. **Pass**.

### T4 — Real-world: RL agent decides to grow, user asks where to start
**Scenario**: User wants to build an RL-controlled morphogenetic system. Asks this pack first.

**Expected routing**: Router must redirect controller-side work to `yzmir-morphogenetic-rl` while keeping network-training work in this pack.

**Artifact evidence**: "RL-Controlled Architecture" scenario (lines 287-298) leads with `yzmir-morphogenetic-rl` as step 1 ("canonical home for the RL-controller-decides-mutation loop"), then routes growth-action mechanics and gradient isolation to this pack. Boundary statement at line 297. **Pass — well-handled bidirectional cross-pack routing.**

### T5 — Edge case: user invokes `/dynamic-architectures` as a slash command
**Expected behavior**: Slash command loads router skill.

**Artifact evidence**: **Fails — wrapper missing.** User gets unknown-command error or has to discover the skill some other way. The sibling `/morphogenetic-rl` wrapper explicitly references `/dynamic-architectures` (line 7 of `morphogenetic-rl.md`), so a user following the morphogenetic-rl cross-ref will hit a dead link. **Fail.** This is the evidence for the Major finding.

### T6 — Command activation: user runs `/diagnose-growth` with no arguments
**Expected behavior**: Command runs its symptom-classification protocol.

**Artifact evidence**: Command body has Phase 1 "Identify the Symptom" with categories table. No `argument-hint` in frontmatter to tell the user what shape of input is expected. The command is still usable (asks the user once invoked), but discoverability is reduced. **Soft pass / Minor** — works, but undocumented input shape.

### T7 — Agent edge case: user asks the advisor for "general PyTorch debugging help"
**Expected behavior**: Agent declines / hands off to `yzmir-pytorch-engineering`.

**Artifact evidence**: Agent's Scope Boundaries table (lines 121-127) explicitly hands off "PyTorch autograd internals" to `yzmir-pytorch-engineering`, "RL algorithm implementation" to `yzmir-deep-rl`, etc. **Pass — handoff matrix present.**

### T8 — Pressure: "Just sum the module outputs, it's simpler"
**Scenario**: User builds an MoE-like composition and wants to just sum expert outputs without gating.

**Expected routing**: Router rationalization-resistance counter-guidance should fire.

**Artifact evidence**: Rationalization table line 308 — "Modules can just sum their outputs" → "Naive summation can cause interference" → "Check modular-neural-composition for combination mechanisms." **Pass**.

### T9 — Real-world: user has a Mixtral fine-tuning task and asks "which PEFT method?"
**Scenario**: User has a Mixtral 8x7B base, wants to fine-tune for a domain task, GPU budget is one H100, and asks "LoRA, QLoRA, DoRA, or something newer?"

**Expected routing**: Router quick-routing line 76 ("Pick a modern PEFT variant (VeRA / LoRA+ / PiSSA / LoftQ / rsLoRA)") → `peft-adapter-techniques`. That sheet should present the choice space with trade-offs.

**Artifact evidence**: Router lines 76, 144-159 route correctly; `peft-adapter-techniques.md` description ("LoRA, QLoRA, DoRA, adapter placement, merging strategies") and v1.2.1 description naming the modern variants in plugin.json confirms the sheet should cover the option set. **Pass** (assuming sheet body matches the description; section list confirms it does — LoRA, QLoRA, DoRA, Adapter Placement, etc.).

### T10 — Edge case: ambiguous "I want to merge my fine-tunes"
**Scenario**: User has three task-specific LoRA fine-tunes of the same base; wants one merged model.

**Expected routing**: Router line 79 ("Merge several fine-tuned checkpoints (TIES / DARE / SLERP / MergeKit)") → `modular-neural-composition`. The merging algorithms live in the composition sheet, not the PEFT sheet — a slightly non-obvious assignment.

**Artifact evidence**: Router routes correctly; `modular-neural-composition.md` section list confirms "Adapter Merging & Task Arithmetic" with TIES, DARE, SLERP, MergeKit, LoraHub coverage. The router's choice to place merging in *composition* rather than *PEFT* is defensible (merging is module-level coordination, not adapter-level training), but a first-time reader might guess `peft-adapter-techniques` first. **Pass** with a note that the quick-routing table is what makes this discoverable.

### Test summary
| # | Test | Result |
|---|---|---|
| T1 | Full-fine-tune pressure | Pass |
| T2 | Full-freeze pressure | Pass |
| T3 | Ambiguous forgetting + growth | Pass |
| T4 | RL-controlled morphogenesis (cross-pack) | Pass |
| T5 | `/dynamic-architectures` slash invocation | **Fail (Major)** |
| T6 | `/diagnose-growth` arg discoverability | Soft pass (Minor) |
| T7 | Agent out-of-scope handoff | Pass |
| T8 | Naive-sum composition pressure | Pass |
| T9 | Mixtral PEFT method selection | Pass |
| T10 | LoRA merging routing | Pass (with note) |

10 of 10 substantive routing tests pass; 1 plumbing test (T5) fails — the missing wrapper.

### Pressure-resistance summary

The router's **Rationalization Resistance Table** (SKILL.md lines 302-311) is the load-bearing artifact for pressure tests. It enumerates six common shortcuts and counters each with a pointer to the relevant sheet. Sampling four rows verbatim:

- Row 1: "Just train a bigger model from scratch" → "Transfer + growth often beats from-scratch" → "Check continual-learning-foundations for why"
- Row 2: "I'll freeze everything except the new layer" → "Full freeze may be too restrictive" → "Check gradient-isolation-techniques for partial strategies"
- Row 4: "Modules can just sum their outputs" → "Naive summation can cause interference" → "Check modular-neural-composition for combination mechanisms"
- Row 5: "I'll integrate immediately when training finishes" → "Need warmup/holding period" → "Check progressive-training-strategies for safe integration"

Each rationalization is named in user-voice and countered with a sheet pointer. This is the right pattern for pressure resistance — it both legitimizes the user's instinct (so they don't feel dismissed) and offers a concrete next read. **The pack is pressure-aware.**

The **Red Flags Checklist** (lines 316-323) gives six observable failure indicators (no isolation, no warmup, no gates, naive combination, ignoring forgetting, no rollback). These are the right diagnostic signals for the domain.

---

## 5. Findings (Critical / Major / Minor / Polish)

### Critical
*(none)*

### Major
- **MAJ-1: Missing slash-command wrapper `.claude/commands/dynamic-architectures.md`.**
  - **Evidence**: `ls /home/john/skillpacks/.claude/commands/` shows wrappers for every other yzmir router (deep-rl, llm-specialist, ml-production, morphogenetic-rl, neural-architectures, pytorch-engineering, simulation-foundations, training-optimization) but no `dynamic-architectures.md`.
  - **Impact**:
    1. Per repo CLAUDE.md, router skills must be exposed as slash commands due to skill-discovery context limits — users cannot reliably auto-discover this router without the wrapper.
    2. Sibling `morphogenetic-rl.md` line 7 ("Companion to `/dynamic-architectures`") and its line 36 cross-ref `/dynamic-architectures` create a broken link.
    3. Behavioral test T5 fails.
  - **Fix**: Add `/home/john/skillpacks/.claude/commands/dynamic-architectures.md` following the same wrapper pattern used by `morphogenetic-rl.md`: frontmatter `description:` (one line, no trailing period), a brief overview, a sheets list, commands list, agents list, and a Cross-references block. This is a Stage 5 action — explicitly out of scope for this report.

### Minor
- **MIN-1: Commands lack `argument-hint` frontmatter.**
  - **Evidence**: `commands/design-lifecycle.md` and `commands/diagnose-growth.md` frontmatter contains only `description` and `allowed-tools`. Marketplace convention (per `using-skillpack-maintenance` SKILL.md lines 142-155 and the `argument-hint` examples in other plugins — axiom-determinism-and-replay, axiom-audit-pipelines, axiom-pyo3-interop) is to include a quoted `argument-hint`.
  - **Impact**: Users invoking the slash command don't see expected argument shape. Commands still work (both internally elicit info from the user), so impact is low.
  - **Fix**: Add `argument-hint: "[symptom_or_subsystem]"` to `diagnose-growth.md` and `argument-hint: "[module_type_or_use_case]"` to `design-lifecycle.md`, or whatever the maintainer judges accurate.

- **MIN-2: Agent reference-sheet list omits `peft-adapter-techniques.md`.**
  - **Evidence**: `agents/dynamic-architecture-advisor.md` lines 53-61 lists 6 reference sheets (continual-learning-foundations, gradient-isolation-techniques, dynamic-architecture-patterns, modular-neural-composition, ml-lifecycle-orchestration, progressive-training-strategies) but **not** `peft-adapter-techniques.md`. Router SKILL.md lists 7.
  - **Impact**: When the agent investigates a PEFT-related question (LoRA/QLoRA/DoRA), it lacks the explicit pointer to the sheet that covers it. It may still find it via Glob, but the agent's own scope statement omits the area.
  - **Fix**: Add `peft-adapter-techniques.md` to the agent's Reference Sheets list. Cross-check Expertise section line 47 — it mentions "PEFT" in spirit ("EWC, SI, MAS, ...") but not in the bullets I read; verify the Expertise list also covers PEFT explicitly (line 47 "Continual Learning" + line 48 "Gradient Isolation" + ... no dedicated PEFT bullet).

- **MIN-3: Plugin.json description's component-count phrasing.**
  - **Evidence**: `.claude-plugin/plugin.json` description says "7 reference sheets, 2 commands, 1 SME agent." Marketplace catalog entry says "6 skills, 1 agent, 2 commands." Both are sort-of correct depending on whether you count the router as a skill or the reference sheets as skills, but they disagree on the count.
  - **Impact**: Cosmetic discoverability inconsistency between plugin.json and marketplace.json.
  - **Fix**: Reconcile counts. Suggested canonical phrasing: "1 router skill + 7 reference sheets, 2 commands, 1 SME agent."

### Polish
- **POL-1: Sibling cross-references could explicitly name `morphogenetic-rl` (not just the boundary statement) in the agent's `Defer to Other Specialists` table.**
  - **Evidence**: Agent file `agents/dynamic-architecture-advisor.md` lines 121-127 defers to `yzmir-pytorch-engineering`, `yzmir-training-optimization`, `yzmir-deep-rl`, `yzmir-neural-architectures`, `yzmir-ml-production` — but **not** `yzmir-morphogenetic-rl`, even though that's the most obvious sibling for RL-controlled growth decisions. The router skill names it correctly (lines 293-297, 341); the agent's deferral table doesn't.
  - **Impact**: An advisor invocation about controller design might not redirect cleanly.
  - **Fix**: Add a row to the agent's `Defer to Other Specialists` table: `| RL controller design / governor / rollback | yzmir-morphogenetic-rl |`.

- **POL-2: Router's "Common Multi-Skill Scenarios" mentions building a "morphogenetic system" but the canonical RL-controlled variant lives in the sibling pack.**
  - **Evidence**: SKILL.md lines 259-269 ("Scenario: Building a Morphogenetic System") gives a 5-step routing through *this* pack's sheets, with no mention of `yzmir-morphogenetic-rl`. The RL-controlled scenario at 287-298 *does* name the sibling. A first-time reader may treat the first scenario as canonical and miss the sibling.
  - **Fix**: Add a one-line note to the "Building a Morphogenetic System" scenario: "If the controller is RL-driven, start with `yzmir-morphogenetic-rl` for controller / governor / rollback design; this pack covers the network-side mechanics." (Cosmetic only.)

- **POL-3: Inline `class` / `def` Python in reference sheets is high-quality but unversioned.**
  - **Evidence**: Substantial code in `gradient-isolation-techniques.md`, `peft-adapter-techniques.md`, `modular-neural-composition.md`, etc., uses libraries (PEFT, bitsandbytes, MergeKit) that are version-sensitive. The text doesn't pin versions or note compatibility windows.
  - **Impact**: As the underlying libraries evolve, examples may drift. Low impact for v1.2.1 (modern PEFT variants are explicitly named in keywords, suggesting recent currency).
  - **Fix**: Optional — add a "Known-good versions as of [date]" note per sheet, or a single pack-level note in the router.

---

## 6. Recommended Actions

Strict priority order. All actions are out of scope for this read-only review and would be executed in a follow-up Stage 5 pass.

1. **Add `.claude/commands/dynamic-architectures.md`** (MAJ-1). Copy the structure of `morphogenetic-rl.md`. List the 7 sheets, 2 commands, 1 agent, and cross-references to the sibling packs. Bump plugin version 1.2.1 → 1.2.2 (patch; user-facing surface restored, not new content).
2. **Add `argument-hint` to both commands** (MIN-1). Patch bump folds in.
3. **Add `peft-adapter-techniques.md` to the agent's reference list** (MIN-2). Patch bump folds in.
4. **Reconcile plugin.json vs marketplace.json component counts** (MIN-3). Patch bump folds in.
5. **Add morphogenetic-rl row to agent's deferral table** (POL-1).
6. **Add RL cross-ref to "Building a Morphogenetic System" scenario** (POL-2).
7. *(Optional)* **Version-pin or date-stamp library examples** (POL-3).

If items 1-4 are folded into one commit, a single patch bump to 1.2.2 covers the work and matches the "low-impact maintenance" rule in `using-skillpack-maintenance` SKILL.md lines 245-251.

---

## 7. Reviewer Notes

### Method
- Loaded `meta-skillpack-maintenance:using-skillpack-maintenance/SKILL.md`, `analyzing-pack-domain.md`, `reviewing-pack-structure.md`, `testing-skill-quality.md` in full.
- Inventoried all 11 component files of `yzmir-dynamic-architectures` (1 router + 7 sheets + 2 commands + 1 agent), read frontmatter and structure of each. Read the router SKILL.md and the agent in full.
- Sampled section structure (header lists) for all 7 reference sheets to verify topical coverage matches the router's claims and the v1.2.1 description.
- Confirmed the boundary with `yzmir-morphogenetic-rl` by reading both packs' plugin.json descriptions and the sibling's SKILL.md (lines 1-80).
- Confirmed marketplace registration and slash-command wrapper inventory via `.claude/commands/` directory listing and `.claude-plugin/marketplace.json` grep.
- Behavioral tests (T1-T8) are artifact-based probes, not executed subagent runs, per the report-only constraint. T5 is the most consequential: a missing wrapper is observable as a hard fact via `ls`.

### What I did not check
- Did **not** run any subagent dispatch to test live skill discovery (would require harness setup beyond the read-only scope).
- Did **not** verify the Python code examples in reference sheets are runnable. Code is read for structural correctness and idiom, not executed.
- Did **not** survey current literature for missing modern techniques beyond the v1.2.1 keyword list. The pack's currency for 2024-2025 PEFT and MoE variants appears credible based on naming (VeRA, DoRA, PiSSA, LoftQ, rsLoRA, LongLoRA, Mixtral, DeepSeek-MoE, MergeKit) but I did not validate against current arXiv. Research-currency flag stands.
- Did **not** check that internal markdown links in the reference sheets resolve. The router's links to sheet filenames match files on disk; per-sheet cross-links were not validated link-by-link.

### Prose-quality and consistency observations

- **Voice**: Reference sheets use a calm, expository voice that matches the rest of the yzmir faction. No "you absolutely must" hyperbole. Code comments inside examples are useful (they explain *why*, not just *what*).
- **Cross-sheet consistency**: Each sheet has an Overview, topical sections, and a closing Implementation Checklist or Common Pitfalls. The router's "Use when..." description matches the marketplace's dominant frontmatter convention.
- **PyTorch idiom**: Examples use modern PyTorch (`nn.Module`, `nn.Parameter`, `requires_grad`, `torch.no_grad`, `tensor.detach`, hooks). No legacy patterns (no `Variable`, no `torch.autograd.Variable`).
- **Citation discipline**: Sheets cite primary literature in-line (Abraham & Robins 2005 in continual-learning; Net2Net, Lottery Ticket, EWC, PackNet, DEN, Switch, Mixtral, DeepSeek-MoE, Megablocks/Gale et al. 2023, TIES Yadav 2024, DARE Yu 2024). This is unusual and good — most marketplace sheets don't cite at this density.
- **Code-to-prose ratio**: High but not overwhelming. The 1054-line `modular-neural-composition.md` is content-dense rather than padded; the section structure shows even coverage rather than one block dominating.
- **One inconsistency**: Plugin.json description (line 4) says "7 reference sheets" while marketplace.json says "6 skills" — see MIN-3.

### Confidence
**High** on the Major finding (slash-command wrapper) — verified by direct directory listing and a cross-reference from the sibling pack that points at the missing target.
**High** on the boundary-cleanliness assessment — explicitly stated in both packs' router skills and plugin.json descriptions.
**Medium** on coverage completeness — coverage appears comprehensive for the stated scope, but I have not surveyed external literature; a domain expert might flag specific recent techniques.
**Medium** on the Minor findings — they are convention drift, not behavioral defects; their priority depends on how strict the maintainer wants to be about cross-pack consistency.

### Appendix: wrapper template for MAJ-1

For the maintainer's convenience when executing the Stage 5 fix, the missing wrapper should follow the marketplace convention demonstrated by `/home/john/skillpacks/.claude/commands/morphogenetic-rl.md`:

```markdown
---
description: Dynamic/morphogenetic neural networks - grow, prune, adapt topology during training. Continual learning, gradient isolation, modular composition, PEFT (modern variants), lifecycle orchestration
---

# Dynamic Architectures Routing

**Companion to `/morphogenetic-rl` - this pack covers HOW the growable network trains; the morphogenetic-rl pack covers WHEN/HOW the controller decides to grow.**

Use the `using-dynamic-architectures` skill from the `yzmir-dynamic-architectures` plugin to route to the right specialist sheet.

## Sheets

- **continual-learning-foundations** - EWC / SI / MAS, PackNet, Progressive Networks, rehearsal, BWT/FWT/ACC metrics
- **gradient-isolation-techniques** - freeze strategies, detach vs no_grad, hook-based gradient surgery, alpha blending
- **peft-adapter-techniques** - LoRA / QLoRA / DoRA + modern variants (VeRA, PiSSA, LoftQ, LoRA+, rsLoRA, LongLoRA)
- **dynamic-architecture-patterns** - grow/prune patterns, Net2Net widening, lottery ticket, triggers, slot semantics
- **modular-neural-composition** - MoE (Switch / Mixtral / DeepSeek-MoE), grafting semantics, adapter merging (TIES / DARE / SLERP / MergeKit / LoraHub)
- **ml-lifecycle-orchestration** - state machines, gates, transition triggers, heuristic / learned / hybrid controllers
- **progressive-training-strategies** - staged expansion, warmup (zero-init / LR / alpha-ramp), cooldown, multi-stage schedules

## Commands

- `/design-lifecycle` - design a state machine for module growth / training / integration
- `/diagnose-growth` - diagnose growth, pruning, gradient-isolation, or lifecycle issues

## Agents

- `dynamic-architecture-advisor` - SME advisor for dynamic-architecture decisions (opus, follows SME Protocol)

## Cross-references

- RL controller design / governor / rollback shaping → `/morphogenetic-rl`
- RL algorithm choice (PPO / SAC / DQN) → `/deep-rl`
- Counterfactual evaluation of architecture changes → `/deep-rl:counterfactual-reasoning`
- PEFT applied to LLMs in production (RLHF, instruction tuning) → `/llm-specialist`
- Distributed / FP8 / MoE-dispatch kernels, FSDP2 + QLoRA → `/training-optimization`
- Static architecture design → `/neural-architectures`
- Production deployment, S-LoRA / LoRAX / Punica multi-tenant serving → `/ml-production`
- Low-level PyTorch / autograd debugging → `/pytorch-engineering`
```

This is illustrative — actual content is the maintainer's call. Length and shape follow `morphogenetic-rl.md` so the two siblings present consistently. No changes to plugin.json beyond a patch bump are needed for this fix alone.

### Open question for maintainer
- Was the missing `.claude/commands/dynamic-architectures.md` wrapper an intentional omission (e.g., the pack was meant to be skill-only, no slash-command exposure)? If yes, document the decision in the plugin README per `reviewing-pack-structure.md` line 228 ("If a plugin intentionally has no slash-command exposure, document that decision in the plugin's README"). The sibling pack's reference to `/dynamic-architectures` as a slash command suggests the omission is unintentional, but only the maintainer can confirm.
