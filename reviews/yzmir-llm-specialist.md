# Review: yzmir-llm-specialist
**Version:** 1.2.0  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

Stages 1-4 of `meta-skillpack-maintenance:using-skillpack-maintenance` applied. Stage 5 (execution) deliberately skipped per task scope. Report-only — no files modified.

---

## 1. Inventory

### Plugin metadata
- **Path:** `/home/john/skillpacks/plugins/yzmir-llm-specialist/`
- **plugin.json** (`/home/john/skillpacks/plugins/yzmir-llm-specialist/.claude-plugin/plugin.json`): name `yzmir-llm-specialist`, version `1.2.0`, license `CC-BY-SA-4.0`, advertises "10 reference sheets, 3 commands, 2 agents".
- **Marketplace registration:** present in `/home/john/skillpacks/.claude-plugin/marketplace.json` with source `./plugins/yzmir-llm-specialist`. OK.
- **Faction:** Yzmir (AI/ML). Consistent with the rest of the AI/ML cluster.

### Router skill
- `skills/using-llm-specialist/SKILL.md` (322 lines). Frontmatter `name: using-llm-specialist`, `description:` does **not** start with the "Use when..." convention — see Finding 5.1.

### Reference sheets (10 — matches plugin.json claim)

| Sheet | Lines | Header style | Domain |
|---|---|---|---|
| `reasoning-models.md` | 322 | `## Context` | Reasoning-tier models, thinking-token economics |
| `prompt-engineering-patterns.md` | 1071 | `## Context` | Chat/instruct prompting, few-shot, CoT, format spec |
| `agentic-patterns-and-mcp.md` | 311 | `## Context` | Agent loops, tool use, MCP, structured output |
| `context-engineering-and-prompt-caching.md` | 288 | `## Context` | Prompt-cache design, long-context layout, cache vs RAG vs FT |
| `rag-architecture-patterns.md` | 573 | `## Context` | Chunking, hybrid retrieval, re-ranking, evaluation |
| `llm-finetuning-strategies.md` | 506 | `## Context` | SFT / DPO / GRPO / LoRA family, modern preference tuning |
| `context-window-management.md` | 512 | `## When to Use This Skill` | Long-context vs RAG, RULER, lost-in-the-middle |
| `llm-evaluation-metrics.md` | 1770 | `## When to Use This Skill` | Task metrics, LLM-as-judge, capability suites, red-teaming |
| `llm-inference-optimization.md` | 641 | `## When to Use This Skill` | Latency, throughput, KV cache, quantization, serving stacks |
| `llm-safety-alignment.md` | 517 | `## When to Use This Skill` | OWASP LLM Top 10 (2025), jailbreaks, PII, prompt injection |

Header style drift: 6 of 10 sheets use the newer `## Context` opener (clear contextual framing of "the mistake we're avoiding"); 4 older sheets use `## When to Use This Skill`. Content is consistent across both styles — capability-tier vocabulary appears in both. Drift is cosmetic only. Minor.

**Inter-sheet cross-references observed (sampled):**
- `agentic-patterns-and-mcp.md` line 17: "This sheet uses the **capability-tier vocabulary** from [reasoning-models.md](reasoning-models.md). Read that sheet first if you haven't."
- `context-engineering-and-prompt-caching.md` line 16: same pattern, anchors to `reasoning-models.md`.
- `prompt-engineering-patterns.md` lines 17-28: explicit fork at the top of the sheet — "If you're targeting a reasoning model, prompt rules differ — sometimes drastically", with a decision tree on line 33.
- `rag-architecture-patterns.md` lines 38-66: decision tree that cross-routes to `context-engineering-and-prompt-caching.md` (when corpus < 200k tokens and static) and `reasoning-models.md` (when task is pure reasoning).
- `llm-finetuning-strategies.md` lines 35-39: cross-routes to `rag-architecture-patterns.md`, `reasoning-models.md`, and `agentic-patterns-and-mcp.md` depending on failure mode.

The cross-referencing discipline is consistent and bidirectional — sheets recognize their place in the larger graph and defer to siblings rather than duplicate.

### Commands (3 — matches plugin.json claim)

| Command | Frontmatter | Allowed tools | Lines |
|---|---|---|---|
| `commands/debug-generation.md` | description, allowed-tools, argument-hint OK | `["Read", "Grep", "Glob", "Bash", "Task"]` | 249 |
| `commands/optimize-inference.md` | description, allowed-tools, argument-hint OK | `["Read", "Grep", "Glob", "Bash", "Task", "Write"]` | 239 |
| `commands/rag-audit.md` | description, allowed-tools, argument-hint OK | `["Read", "Grep", "Glob", "Bash", "Task", "Write"]` | 337 |

All three use quoted-JSON-array `allowed-tools` and quoted-string `argument-hint`, matching the marketplace convention from `reviewing-pack-structure.md`.

### Agents (2 — matches plugin.json claim)

| Agent | Model | SME-protocol cite | Four sections required | Lines |
|---|---|---|---|---|
| `agents/llm-diagnostician.md` | `sonnet` | Yes, line 10 | Yes | 346 |
| `agents/llm-safety-reviewer.md` | `opus` | Yes, line 10 | Yes | 240 |

Both agents declare only `description` and `model` (no `tools:` restriction — matches marketplace convention). Both descriptions end with "Follows SME Agent Protocol with confidence/risk assessment." Both bodies cite `meta-sme-protocol:sme-agent-protocol` and require the four output sections (Confidence / Risk / Information Gaps / Caveats).

### Slash-command wrapper

- File exists: `/home/john/skillpacks/.claude/commands/llm-specialist.md` (209 lines).
- **Significant content drift** — see Finding 4.1 (Major). The wrapper still describes a 7-skill router; the actual router covers 10 reference sheets with capability-tier vocabulary and the reasoning-vs-chat split that the wrapper never mentions.

### Hooks

None. Plugin has no `hooks/` directory; this is appropriate for a content-only pack.

---

## 2. Domain & Coverage

### User-defined scope (inferred from plugin.json + router)
LLM application engineering — prompting (incl. reasoning models), RAG, fine-tuning (SFT / DPO / GRPO), agentic patterns + MCP, context engineering + prompt caching, inference optimization (vLLM / SGLang / TensorRT-LLM), evaluation (incl. LLM-as-judge bias controls), and safety (OWASP LLM Top 10).

The router (`SKILL.md` lines 10-14) explicitly anchors the pack to a **knowledge cutoff of 2026-05** and to the **capability-tier vocabulary** (`frontier-reasoning`, `frontier-general`, `fast-cheap`, `on-device`) rather than hardcoded model IDs. This is a deliberate design decision — calibrated for ~quarterly model-roster churn — and shows in every sheet that was opened during this review.

### Domain map (coverage status)

Foundational:
- Capability-tier vocabulary — Exists (`reasoning-models.md` lines 18-31, repeated in every other sheet).
- Reasoning vs chat/instruct split — Exists (`reasoning-models.md`, called out from `prompt-engineering-patterns.md` lines 17-28).
- Prompt engineering basics (instruction clarity, few-shot, CoT, format spec) — Exists (`prompt-engineering-patterns.md`).
- System message design — Exists.

Core techniques:
- Thinking-token budgeting (Anthropic adaptive thinking, OpenAI o-series reasoning effort, etc.) — Exists (`reasoning-models.md` lines 68-79+).
- RAG architecture (chunking, hybrid retrieval, re-ranking) — Exists (`rag-architecture-patterns.md`).
- Fine-tuning ladder (PPO → DPO → IPO/KTO/SimPO/ORPO → GRPO) — Exists (`llm-finetuning-strategies.md` lines 11-16).
- LoRA family (LoRA, QLoRA, DoRA, rsLoRA, LoftQ, LongLoRA) — Exists.
- Prompt caching mechanics + cost math — Exists (`context-engineering-and-prompt-caching.md`).
- Agentic patterns: ReAct, planner/executor, multi-agent — Exists (`agentic-patterns-and-mcp.md`).
- MCP (Model Context Protocol) — Exists.
- Structured output (provider features, Outlines, Instructor) — Exists.
- Context-window management, lost-in-the-middle, RULER — Exists (`context-window-management.md` lines 22-26).
- Inference optimization (parallelization, caching, model routing, serving stacks) — Exists (`llm-inference-optimization.md`).
- Evaluation: task metrics, LLM-as-judge, golden sets, contamination checks — Exists (`llm-evaluation-metrics.md`).
- Capability suites (Inspect AI, OLMES, OpenAI Evals, lm-evaluation-harness) — Exists.
- Red-teaming primer (PAIR, GCG, AutoDAN) — Exists (`llm-evaluation-metrics.md`, `llm-safety-alignment.md` line 20).
- OWASP LLM Top 10 (2025 edition) — Exists (`llm-safety-alignment.md` lines 24-37).

Advanced / cross-cutting:
- Reasoning-model evaluation (thinking-token tracking, over-thinking loops) — Exists (`reasoning-models.md`, evaluation sheet Part 8).
- Cache vs RAG vs fine-tune decision — Exists (`context-engineering-and-prompt-caching.md`).
- Prompt-injection structural defenses (not just sanitization) — Exists (`llm-safety-alignment.md`).
- Agentic safety (when tools are in play) — Exists (`llm-safety-alignment.md` line 22).

### Domain coverage depth observations

The pack treats four areas with notable rigor that deserves to be called out — these are the dimensions where the sheets exceed a "comprehensive reference" bar and reach "discipline-enforcing" territory:

1. **Tool-use loop discipline (`agentic-patterns-and-mcp.md`).** Lines 20-80 lay out the cross-provider tool loop (OpenAI / Anthropic / Gemini differences), parallel-tool-call traps (order dependence, partial failure, cost compounding), and a max-iteration safeguard pattern. The discipline is consistent across providers rather than provider-specific tutorial.
2. **Cache-aware prompt structure (`context-engineering-and-prompt-caching.md`).** Lines 32-79 give a four-provider caching comparison table (Anthropic explicit, OpenAI automatic, Gemini implicit, Gemini explicit) with pricing ratios, TTLs, and the single anchoring rule: "stable prefix first, volatile suffix last." This is direct cost-engineering content, not buzzword caching.
3. **Modern fine-tuning lineage (`llm-finetuning-strategies.md`).** Lines 11-16 enumerate the algorithm family in time order (PPO → DPO → IPO / KTO / SimPO / ORPO → GRPO) and the LoRA family (LoRA, QLoRA, DoRA, rsLoRA, LoftQ, LongLoRA). This is current as of 2026-05 and avoids the common error of presenting DPO as the endpoint.
4. **Safety-as-threat-model first (`llm-safety-alignment.md`).** Lines 24-37 lead with the OWASP LLM Top 10 (2025 edition) rather than burying it in an appendix. LLM01-LLM10 are each named, with structural prompt-injection defenses (not just input sanitization) discussed as a category.

These four areas are the pack's signal of seriousness — they're where a content-only LLM pack would coast on summarization, and this pack instead invests in framework-grade depth.

### Gaps observed

1. **Multimodal prompting** — explicitly acknowledged as missing by the router (SKILL.md line 70, 207, 272: "dedicated multimodal sheet forthcoming"). General principles are folded into `prompt-engineering-patterns.md` but no dedicated sheet covers image-token costs, resolution settings, multimodal evaluation, or multimodal jailbreaks. **Honest-gap acknowledgement is good practice** — the router warns the model not to freelance — but the gap itself is medium-priority for 2026-vintage LLM work.
2. **Embedding-model selection** — `rag-architecture-patterns.md` discusses retrieval but the choice of embedding model (dense vs sparse, multilingual, code-aware, domain-tuned) is light. Not blocking; could expand.
3. **Cost modeling** — `context-engineering-and-prompt-caching.md` covers caching cost math; broader unit-economics (cost-per-task across tiers, break-even for FT vs RAG vs cached prompt) is implicit but not assembled in one place.
4. **No gap in router coverage of the rest of the domain** — all 10 sheets are real and substantive (range 288-1770 lines), and they cover the modern (2026) shape of LLM engineering.

### Research currency

Pack explicitly self-dates to 2026-05 and uses capability-tier abstraction to defer ID-pinning. Sheet sampled for currency:
- `reasoning-models.md` cites DeepSeek-R1 (Guo 2025), OpenAI o1, Claude adaptive thinking, Qwen3-thinking — current as of 2026-05.
- `llm-safety-alignment.md` cites OWASP LLM Top 10 (2025 edition) — current.
- `llm-finetuning-strategies.md` lineage (PPO → DPO → IPO/KTO/SimPO/ORPO → GRPO) — current.
- `context-window-management.md` discusses 1M-context tier honestly, with RULER caveat — current.

No deprecated guidance spotted in the headers/openings reviewed. Sheets are not stale.

### Cross-pack boundary discipline

The router file (`SKILL.md` lines 138-144) and the individual sheets explicitly defer to sibling packs at the boundary:
- **Adversarial-ML threat modeling, supply-chain risk** → `ordis-security-architect` (cited from router + safety sheet).
- **Production serving, deployment infrastructure, monitoring** → `yzmir-ml-production` (cited from router).
- **Training at scale, FSDP2 / FP8 / optimizer-state sharding, muP learning-rate transfer** → `yzmir-training-optimization` (cited from router + fine-tuning sheet line 16).
- **Top-level AI/ML routing** → `yzmir-ai-engineering-expert` (cited from router).

This is the right discipline — the pack covers LLM *application* engineering and explicitly does NOT try to absorb training infrastructure, serving infrastructure, or general-purpose security architecture. The fine-tuning sheet's explicit "cross-ref, don't duplicate" instruction at line 16 is the kind of authorial discipline that prevents pack sprawl.

---

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Rating | Notes |
|---|---|---|---|
| 1 | Domain coverage | Pass | 10 sheets covering reasoning, chat prompting, agentic+MCP, RAG, FT, context-eng+caching, context-window, eval, inference-opt, safety. Multimodal flagged as known gap. |
| 2 | Router accuracy | Pass | Router decision tree maps cleanly to the 10 sheets; reasoning-model gate is Step 0 (correct precedence). Capability-tier vocabulary used throughout. |
| 3 | Description triggering | Minor | Router `description:` does not lead with "Use when..." — see Finding 5.1. Still discoverable, but inconsistent with the marketplace convention. |
| 4 | Cross-references | Pass | Sheets cross-link (e.g., reasoning-models → agentic-patterns; finetuning-strategies → training-optimization pack; safety → ordis-security-architect). |
| 5 | Component-type alignment | Pass | Skills are auto-invokable reference content; commands (`/debug-generation`, `/optimize-inference`, `/rag-audit`) are explicit user actions; agents are SME reviewers. No misalignment. |
| 6 | SME-protocol compliance (agents) | Pass | Both agents cite `meta-sme-protocol:sme-agent-protocol`, declare model only, and require the four output sections verbatim. |
| 7 | Marketplace + wrapper hygiene | Major | Plugin registered. **Slash-command wrapper `.claude/commands/llm-specialist.md` is significantly out of sync** with the router skill — describes 7 specialized skills, missing 3 (reasoning, agentic-patterns-and-mcp, context-engineering-and-prompt-caching), missing the capability-tier vocabulary, missing the reasoning-vs-chat split, missing OWASP LLM Top 10 framing. See Finding 4.1. |
| 8 | Research currency | Pass | Self-dated to 2026-05; references are current (DeepSeek-R1, OWASP LLM Top 10 2025, GRPO, adaptive thinking). Capability-tier abstraction is the right hedge against model-ID churn. |

**Overall: Major** — driven entirely by the stale slash-command wrapper (`/llm-specialist`). The plugin's actual content (router + 10 sheets + 3 commands + 2 agents) is in good shape; one user-facing surface (the wrapper) was not updated when the pack expanded from 7 to 10 sheets and adopted the capability-tier vocabulary.

If the wrapper is treated as a discoverability surface only (and users always end up loading the actual router skill, which has the right content), the rating drops to **Minor**. Treating it as Major because the wrapper is the *first thing a user sees when typing `/llm-specialist`* and its content directly contradicts the router's framing.

### Per-dimension rationale

**Dimension 1 — Domain coverage.** Pass. The pack covers the eight first-class LLM-engineering domains for 2026: reasoning models, chat/instruct prompting, agentic loops + MCP, RAG, fine-tuning, context engineering + caching, inference optimization, evaluation. Safety is a ninth standalone surface. The only acknowledged gap is multimodal (explicit in `SKILL.md` line 207). No gap was found that the router does not already acknowledge.

**Dimension 2 — Router accuracy.** Pass. Decision tree is structured as Step 0 (reasoning-model gate) → Step 1 (task category). Step 0 placement is the critical correctness call — putting it after Step 1 would route reasoning-model users to chat prompting before catching their model class. The placement is correct. Eleven routing examples in `SKILL.md` lines 148-208 cover the realistic question shapes. Cross-pack references at lines 138-144 are well-placed.

**Dimension 3 — Description triggering.** Minor. The router description does not lead with "Use when..." (see Finding 5.1). Sibling routers in the marketplace consistently use this prefix; the convention is documented in the rubric's SKILL.md line 133. However, the description does contain searchable keywords (LLM, prompt, reasoning, RAG, fine-tuning, agentic, MCP, evaluation, safety) sufficient for discovery — the convention break is cosmetic, not functional.

**Dimension 4 — Cross-references.** Pass. The sheets practice deferral discipline: `prompt-engineering-patterns.md` forks immediately at the top if the model is a reasoning model; `rag-architecture-patterns.md` cross-routes to caching when corpus is small; `llm-finetuning-strategies.md` defers to RAG / reasoning / agentic patterns based on failure mode; cross-pack references to `yzmir-training-optimization` and `ordis-security-architect` are bidirectional from the sheet content. Internal links use the documented `[name.md](name.md)` relative-path pattern that the router's "How to Access Reference Sheets" section (SKILL.md lines 30-43) explicitly documents.

**Dimension 5 — Component-type alignment.** Pass. Three components map cleanly to three invocation modes: (a) skills (router + 10 sheets) are model-auto-invoked when LLM-engineering keywords appear in conversation; (b) commands (`/debug-generation`, `/optimize-inference`, `/rag-audit`) are user-explicit triage entry points with appropriate Write/Bash permissions for the diagnostic action they perform; (c) agents (llm-diagnostician, llm-safety-reviewer) are SME reviewers dispatched by orchestrators with the four-section output protocol. No type is being misused (no command pretending to be a skill, no agent that should be a hook).

**Dimension 6 — SME-protocol compliance.** Pass. Both agents pass the four-point checklist from `analyzing-pack-domain.md` line 78: (a) description ends with "Follows SME Agent Protocol with confidence/risk assessment.", (b) body cites `meta-sme-protocol:sme-agent-protocol`, (c) body requires Confidence Assessment, Risk Assessment, Information Gaps, Caveats sections verbatim, (d) no spurious `tools:` restriction. `llm-diagnostician.md` is on `sonnet` (appropriate for diagnostic depth), `llm-safety-reviewer.md` is on `opus` (appropriate for safety synthesis where false negatives are costly).

**Dimension 7 — Marketplace + wrapper hygiene.** Major. Plugin is registered in marketplace.json with correct source path. But the slash-command wrapper at `/home/john/skillpacks/.claude/commands/llm-specialist.md` is significantly out of sync with the router skill — see Finding 4.1 and Finding 5.5 for specifics. This is the single Major finding in the review.

**Dimension 8 — Research currency.** Pass. Self-dated to 2026-05 explicitly in the router (line 12). Capability-tier abstraction is the right hedge against the documented quarterly model-roster churn. Sampled sheet currency:
- DeepSeek-R1 reasoning trace work (Guo et al. 2025) — current.
- Claude adaptive thinking + `effort` parameter — current as of 2026-05.
- GRPO in the fine-tuning lineage — current.
- OWASP LLM Top 10 (2025 edition) — current.
- RULER long-context benchmark — current.
- 1M-context tier explicit acknowledgement (with recall caveat) — current.
- Outlines and Instructor as structured-output libraries — still active in 2026.
- vLLM, SGLang, TensorRT-LLM as serving stacks — still the relevant frame as of 2026-05.

No deprecated guidance surfaced during the sample.

---

## 4. Behavioral Tests

Tests were conducted as inline reasoning over the pack's content rather than fresh-subagent dispatch (lower fidelity per `testing-skill-quality.md` line 88, acceptable for scorecard-grade triage). For real Stage-3 validation, fresh-subagent dispatch should run the same scenarios.

### T1 — Pressure: "Just give me the model name to use"

Scenario: User says "I need to ship a customer-support bot tomorrow. Just tell me which model to call." This pressures the router to skip the capability-tier framing and recommend a specific model ID.

Router behavior expected: SKILL.md lines 285-293 (Pressure-Resistance: Don't Skip the Router) and lines 12-14 (capability-tier framing + "verify current model IDs in provider docs") explicitly resist this. The router has anti-shortcut text ("Spend the 30 seconds to route correctly") and never names a model ID itself.

Result: **Pass.** The router holds — it identifies the task as chat-tier work, routes to `prompt-engineering-patterns.md`, and tells the user to pick the current model from provider docs.

### T2 — Edge case: Reasoning-model question disguised as a prompting question

Scenario: "I'm using o3 — should I add 'think step by step' to my prompts?"

Router behavior expected: Step 0 of the decision tree (SKILL.md lines 49-54) catches reasoning models *before* the user is routed to `prompt-engineering-patterns.md`. Example 2 (lines 154-158) covers this exact case and answers "usually no."

Result: **Pass.** Step 0 is correctly placed before Step 1, and `prompt-engineering-patterns.md` itself (lines 17-28) defers to `reasoning-models.md` when the target is a reasoning model.

### T3 — Real-world complexity: Multi-skill RAG + eval + caching

Scenario: "I'm building a RAG system over 5k docs, need to evaluate retrieval quality, and want to cache the long system prompt for cost." Three skills apply.

Router behavior expected: SKILL.md lines 211-228 (Multiple Skills May Apply) provides explicit multi-skill ordering. Pattern 1 (lines 234-243) gives a build-from-scratch sequence.

Result: **Pass.** Router instructs "Start with the primary skill (rag-architecture-patterns), then reference secondary (llm-evaluation-metrics, context-engineering-and-prompt-caching)." Order is correct (architecture first, then evaluation, then optimization).

### T4 — Agent activation gates

Scenarios from `agents/llm-diagnostician.md` lines 17-46 and `llm-safety-reviewer.md` lines 17-41 — positive triggers ("LLM keeps hallucinating", "Check this for safety issues") and explicit negatives ("Model is too slow → use /optimize-inference", "Check for security issues → use llm-safety-reviewer agent").

Result: **Pass.** Both agents have positive AND negative activation examples, and they correctly hand off to each other (diagnostician declines safety work, safety-reviewer declines performance work).

### T5 — Wrapper-vs-router consistency

Scenario: User types `/llm-specialist` and expects the same coverage and vocabulary as the actual router skill.

Result: **Fail (Major).** The wrapper (`/home/john/skillpacks/.claude/commands/llm-specialist.md`) describes 7 specialized skills; the router skill (`skills/using-llm-specialist/SKILL.md`) describes 10. The wrapper has no reasoning-vs-chat split, no capability-tier vocabulary, no agentic/MCP coverage, no caching coverage. A user invoking the slash command receives a stale mental model of the pack and may not be routed to `reasoning-models.md` for reasoning-model tasks.

### T6 — Command tool-restriction sanity

Scenarios: Can `/debug-generation` (line 3: `["Read", "Grep", "Glob", "Bash", "Task"]`) actually do its job? It searches code with Grep, reads files, runs Bash for inspection, and dispatches a Task. No Write — which is correct for a diagnostic command.

`/optimize-inference` (line 3) adds Write because it suggests code patches. `/rag-audit` (line 3) likewise.

Result: **Pass.** Tool restrictions are minimal-and-sufficient. No spurious permissions.

### T7 — Pressure on the fine-tuning recommendation

Scenario: "I have 50 examples and my model gives wrong answers — should I fine-tune?" This is a classic premature-fine-tuning request.

Expected: `llm-finetuning-strategies.md` lines 20-39 lay out a decision tree that goes "Try a stronger prompt with a stronger model" → "Add RAG, reasoning, or tool use" → only then consider fine-tuning. `commands/debug-generation.md` lines 211-225 has explicit "Do NOT fine-tune for: tone/style, output formatting, few examples (< 100 insufficient), quick experiments, recent information (use RAG instead)."

Result: **Pass.** Both the sheet and the command have explicit rationalization-resistance for this case. The agent (`llm-diagnostician.md` lines 263-278) repeats the same gates with the "1000+ high-quality examples" minimum.

### T8 — Edge case: agentic safety under prompt injection via tool results

Scenario: "Build an agent that fetches web pages and summarizes them. How do I prevent the fetched page from instructing the agent?"

Expected: `agentic-patterns-and-mcp.md` line 9 names this as a known mistake ("Prompt-injection-via-tool-results — treating retrieved content as trusted instructions"); `llm-safety-alignment.md` LLM01 (line 28) and LLM06 (line 33) cover this from the threat-model side; the safety sheet line 22 mentions "Agentic safety (when tools are in play)" as a first-class formula component.

Result: **Pass.** The threat is named at multiple layers (sheet content, threat model, formula). The agent (`llm-safety-reviewer.md`) would activate on this scenario per its activation examples.

### T9 — Edge case: a user asking for help with a model that doesn't exist in the sheet

Scenario: "I'm using Mistral Large 3 / Llama 5 / [some 2026-vintage model the pack didn't name explicitly]."

Expected: The capability-tier abstraction (`reasoning-models.md` lines 18-31, repeated in other sheets) is exactly designed for this — the pack tells the model to identify the tier (frontier-reasoning / frontier-general / fast-cheap / on-device) and verify the model ID in provider docs, rather than pattern-matching against a hardcoded list.

Result: **Pass.** The tier abstraction is the right architectural call; it ages gracefully where ID-pinned content would not.

### Test summary

| Test | Component | Result |
|---|---|---|
| T1 Pressure (model-ID shortcut) | Router | Pass |
| T2 Edge case (reasoning-model gate) | Router | Pass |
| T3 Multi-skill complexity | Router | Pass |
| T4 Agent activation gates | Both agents | Pass |
| T5 Wrapper-router consistency | Slash command wrapper | **Fail (Major)** |
| T6 Command tool restrictions | All 3 commands | Pass |
| T7 Pressure (premature fine-tune) | Fine-tuning sheet + diag agent + debug-gen cmd | Pass |
| T8 Edge case (agentic prompt injection) | Agentic + safety sheets | Pass |
| T9 Edge case (unknown 2026 model) | Capability-tier abstraction across all sheets | Pass |

---

## 5. Findings

### Critical
None.

### Major

**Finding 4.1 — Slash-command wrapper out of sync with router skill**

- **Path:** `/home/john/skillpacks/.claude/commands/llm-specialist.md`
- **Evidence:**
  - Line 16 (and Summary section, lines 198-205): "The 7 specialized skills" — names 7 but the pack has 10.
  - Missing skills from wrapper: `reasoning-models`, `agentic-patterns-and-mcp`, `context-engineering-and-prompt-caching`.
  - Missing concepts: capability tiers (frontier-reasoning, frontier-general, fast-cheap, on-device), reasoning-vs-chat split, MCP, prompt caching, OWASP LLM Top 10.
  - Line 54: "(4k, 8k, 32k, 128k tokens)" — context-length guidance is pre-2026; the router and `context-window-management.md` discuss 128k-200k baseline and 1M+ tiers (`context-window-management.md` lines 31-35).
  - Plugin.json version is 1.2.0 (so the pack has been versioned forward), but the wrapper was not touched in the bump.
- **Impact:** Users invoking `/llm-specialist` get a stale routing table that omits reasoning-model gating (the highest-leverage decision in 2026 LLM work), agentic/MCP work, and prompt caching. They may not realize the pack covers these.
- **Recommendation (do not execute — report only):** Regenerate the wrapper from `skills/using-llm-specialist/SKILL.md` to mirror the current decision tree, including Step 0 (reasoning-model gate), capability-tier vocabulary, and the 10-sheet catalog. Or, alternatively, have the wrapper be a thin pointer that loads the router skill via the Skill tool (the lightest contract).

### Minor

**Finding 5.1 — Router description does not follow "Use when..." convention**

- **Path:** `/home/john/skillpacks/plugins/yzmir-llm-specialist/skills/using-llm-specialist/SKILL.md` line 3.
- **Current:** `description: LLM specialist router to prompt engineering, reasoning models, agentic / MCP, RAG, fine-tuning, context engineering, evaluation, inference optimization, and safety skills.`
- **Reference (from `using-skillpack-maintenance` SKILL.md line 133):** "Does the description start with 'Use when...' (the dominant repo convention for discoverability)?"
- **Impact:** Mild discovery degradation. Sibling routers like `using-pytorch-engineering` and `using-deep-rl` do start with "Routes ..." or "Use when ...". Cosmetic but worth aligning.
- **Recommendation:** Reword to e.g. "Use when working on LLM applications — prompting (incl. reasoning models), agentic/MCP, RAG, fine-tuning, context engineering and prompt caching, inference optimization, evaluation, or safety. Routes to the right specialist sheet."

**Finding 5.2 — Header-style drift across reference sheets**

- 6 sheets use `## Context` (the newer "what mistake we're avoiding" framing): `reasoning-models`, `prompt-engineering-patterns`, `agentic-patterns-and-mcp`, `context-engineering-and-prompt-caching`, `rag-architecture-patterns`, `llm-finetuning-strategies`.
- 4 sheets use `## When to Use This Skill`: `context-window-management`, `llm-evaluation-metrics`, `llm-inference-optimization`, `llm-safety-alignment`.
- **Impact:** Cosmetic. Content is current in both groups (the older-headered sheets have been updated to capability-tier vocabulary internally — verified for `context-window-management.md` and `llm-inference-optimization.md`).
- **Recommendation:** Optional cleanup pass to harmonize on `## Context` (the newer style is more useful to a model reading the sheet for the first time because it frames failure modes upfront). Low priority.

**Finding 5.3 — Multimodal coverage gap (acknowledged)**

- The router itself flags this honestly (`SKILL.md` lines 70, 207, 272: "Multimodal coverage forthcoming"). This is correct discipline — telling the model not to freelance is better than implying coverage that doesn't exist. But the gap is real for 2026 LLM work (image-token economics, video tokens, multimodal jailbreaks, multimodal evals).
- **Impact:** Medium-priority content gap. Not a process issue.
- **Recommendation:** Add a `multimodal-prompting.md` sheet in a future pack refresh. Out of scope for the current review (report-only).

**Finding 5.4 — `commands/debug-generation.md` and `agents/llm-diagnostician.md` overlap in scope**

- The command (`commands/debug-generation.md`) and the agent (`agents/llm-diagnostician.md`) both diagnose LLM output quality with a symptom-triage table, decision tree, and fix patterns. The command is for user-explicit invocation (`/debug-generation`), the agent for autonomous dispatch by an orchestrator. The split is defensible — one is a user-explicit entry, one is an SME reviewer — but the overlap is large enough that it could confuse maintainers.
- **Impact:** Low. They genuinely serve different invocation modes (per `analyzing-pack-domain.md` lines 203-210). Documenting the distinction in either file's header would help.
- **Recommendation:** Optional — add a "See also" note in each pointing at the other and explaining the user-vs-agent distinction.

**Finding 5.5 — Wrapper's context-length figures are pre-2026**

- **Path:** `/home/john/skillpacks/.claude/commands/llm-specialist.md` line 54.
- **Current:** "Context window limits (4k, 8k, 32k, 128k tokens)".
- **Reference (`context-window-management.md` lines 31-35):** Capability tiers have 128k-200k baseline, with 1M+ on select tiers; "4k/8k" is no longer the relevant frame for any modern frontier model.
- **Impact:** Folded into Finding 4.1; called out separately because it's a concrete content-currency issue, not just a missing-sheet issue.
- **Recommendation:** Replace the parenthetical with capability-tier language matching the actual sheet.

**Finding 5.6 — Plugin description in `plugin.json` claims "10 reference sheets, 3 commands, 2 agents" — accurate**

- **Path:** `/home/john/skillpacks/plugins/yzmir-llm-specialist/.claude-plugin/plugin.json`.
- This is recorded as a Pass, not a finding. The claim matches the inventory exactly (10 SKILL/sheet files, 3 commands, 2 agents). Worth noting because pack metadata accuracy is a frequent drift point in marketplace packs.

### Polish

**Finding P.1 — Wrapper file has a leading blank line**

`/home/john/skillpacks/.claude/commands/llm-specialist.md` line 1 is blank, then heading on line 2. Several other wrappers in `.claude/commands/` start directly with the heading. Cosmetic.

**Finding P.3 — Capability-tier vocabulary not stated in plugin.json description**

- `plugin.json` description mentions concrete topics (RAG, fine-tuning, MCP, vLLM/SGLang/TensorRT-LLM) but does not foreground that the pack is calibrated to a knowledge cutoff and uses capability-tier abstraction. A user browsing the marketplace would not know from the description alone that this pack has deliberate model-ID-rotation discipline.
- **Impact:** Discoverability for users specifically looking for non-stale LLM guidance.
- **Recommendation:** Mention "calibrated to 2026-05 with capability-tier vocabulary (no hardcoded model IDs)" in the description on next minor bump.

**Finding P.2 — `optimize-inference.md` and `rag-audit.md` "Cross-Pack Discovery" sections use shell-style `glob.glob("plugins/yzmir-...")` checks**

- Example: `commands/optimize-inference.md` lines 234-239, `commands/rag-audit.md` (not loaded but per pack conventions). This pattern hardcodes the `plugins/` path layout, which works in this repo but is fragile if a user installs only this plugin (no sibling `plugins/` directory). The agents (`llm-diagnostician.md` lines 314-329, `llm-safety-reviewer.md` lines 213-224) use the same pattern.
- **Impact:** Low. The intent is sound (suggest related packs), but the implementation assumes co-installation in the marketplace layout.
- **Recommendation:** Replace with a plain "Cross-pack references: See also `yzmir-pytorch-engineering`, `yzmir-training-optimization`, `ordis-security-architect`" — the router skill already does this correctly (SKILL.md lines 138-144).

---

## 6. Recommended Actions

In priority order. **Report-only — do not execute.**

1. **(Major, do first)** Regenerate `/home/john/skillpacks/.claude/commands/llm-specialist.md` from the current `skills/using-llm-specialist/SKILL.md`. Mirror the 10-sheet catalog, Step 0 reasoning-model gate, capability-tier vocabulary, and the OWASP LLM Top 10 framing. Patch bump suffices if no other changes (1.2.0 → 1.2.1) since this is wrapper-only and behavior-aligning.
2. **(Minor)** Reword router `description:` (Finding 5.1) to start with "Use when..." for discoverability consistency with sibling routers. Patch bump.
3. **(Minor, optional)** Harmonize sheet headers on `## Context` (Finding 5.2). Patch bump if pure cosmetic.
4. **(Minor)** Cross-link `commands/debug-generation.md` and `agents/llm-diagnostician.md` so maintainers see the user-vs-agent distinction (Finding 5.4). Patch bump.
5. **(Medium, future)** Add `multimodal-prompting.md` sheet covering image/audio/video token economics, multimodal jailbreaks, and multimodal evals (Finding 5.3). Minor bump (new component) — and per `using-skillpack-maintenance:writing-skills`, this requires its own RED-GREEN-REFACTOR pass, not inline drafting.
6. **(Polish)** Replace `glob.glob("plugins/...")` cross-pack-discovery snippets with plain "See also" prose (Finding P.2). No bump, polish-only.
7. **(Polish)** Strip the leading blank line in the wrapper file (Finding P.1). Folded into action 1.

Aggregate version impact: actions 1-4 together = patch bump (1.2.0 → 1.2.1). Action 5 alone, when done = minor bump (1.2.x → 1.3.0). No major-bump candidates — no philosophy shift, no component removal.

---

### What a fix-pass would look like (estimate, not a commitment)

If actions 1-4 were executed in a single sitting:
- Action 1 (regenerate wrapper): ~30-60 minutes. Read the router, mirror Steps 0-1, copy the catalog, update Quick Reference table to 10 rows, drop the "4k/8k" framing.
- Action 2 (router description): ~5 minutes. One-line frontmatter change.
- Action 3 (header harmonization): ~30 minutes. Touch 4 sheets; pure cosmetic.
- Action 4 (cross-link command + agent): ~10 minutes.
- Total: ~1.5-2 hours, plus version bump and commit.

Action 5 (multimodal sheet) is a separate project — per `using-skillpack-maintenance` Stage 5 checkpoint, it requires `superpowers:writing-skills` with full RED-GREEN-REFACTOR. Estimate: half-day to a day for the sheet, plus router updates and a minor version bump (1.2.x → 1.3.0).

---

## 7. Reviewer Notes

- **Method:** Stages 1-4 of `meta-skillpack-maintenance:using-skillpack-maintenance` applied. Behavioral testing (Stage 3) was done inline against the visible content of the router and sheets, rather than via fresh-subagent dispatch (per the rubric, this is "lowest fidelity but acceptable for a quick sanity check" — `testing-skill-quality.md` line 88). Findings rated Major or higher should be re-validated with fresh-subagent runs before execution.
- **Stage 5 (execution) deliberately skipped** per task instructions. No files were modified.
- **What I didn't read:** I did not read every line of every reference sheet — I sampled the first 40-80 lines of each plus the full router, full wrapper, both agents, and all three commands. Total: ~3500 lines of the ~7600-line content surface. Findings about currency are based on what I sampled; deeper sheets (especially `llm-evaluation-metrics.md` at 1770 lines and `prompt-engineering-patterns.md` at 1071 lines) were not exhaustively reviewed and could contain stale internal references that this report did not surface.
- **Confidence on Major finding (4.1):** High. The wrapper file and router file were read in full, and the discrepancy is direct and verifiable.
- **Confidence on Pass ratings:** Medium. Sample size for sheet currency was ~40-80 lines per sheet from the top; deeper content could reveal issues. The high-confidence signal is structural (file presence, frontmatter, marketplace registration, SME-protocol citations) and headline-level (capability-tier vocabulary present, OWASP 2025 cited, reasoning-model split present). For high-stakes execution, run fresh-subagent dispatch against each sheet with a hostile scenario.
- **Pack quality overall:** This is a well-maintained AI/ML pack with deliberate calibration to 2026-vintage LLM engineering. The capability-tier abstraction is the right architectural call for a domain with quarterly model-roster churn. The single user-facing surface that wasn't kept in sync (the slash-command wrapper) is the only material finding.
