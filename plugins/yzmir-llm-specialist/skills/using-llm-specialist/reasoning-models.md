
# Reasoning Models

## Context

You're choosing between a chat model and a reasoning model, or your reasoning-model results are worse than expected. Common mistakes:
- **Reaching for reasoning models reflexively** (paying the "thinking tax" on tasks a fast-cheap model would solve in one pass)
- **Prompting them like chat models** ("think step by step" instructions, heavy few-shot examples) which constrains or duplicates the trace they already produce
- **Ignoring thinking-token economics** (output-token bills dominated by hidden reasoning, latency budgets blown)
- **Evaluating only the final answer** (missing reasoning regressions, over-thinking loops, confidently-wrong answers)
- **Hardcoding model IDs** in prompts and configs that go stale within a quarter

**This skill provides a decision framework for when to use reasoning models, how to prompt and evaluate them, the cost/latency profile, and the failure modes that distinguish them from chat models.**

This is sheet 1 of the LLM-specialist refresh; the **capability-tier vocabulary defined here is referenced from the rest of the pack.**


## Capability Tiers (cross-pack vocabulary)

To avoid hardcoding model IDs that go stale every quarter, this pack uses four capability tiers. Treat tier as the unit of architectural decision; pick the specific model ID at deploy time from current provider docs.

| Tier | What it is | When to reach for it |
|---|---|---|
| **frontier-reasoning** | Models with explicit reasoning/thinking tokens, exposed budget or effort knobs, optimized for math/code/multi-step planning. | Hard reasoning, agentic-loop reliability, eval-grade verification, rare/expensive tasks. |
| **frontier-general** | Top-of-the-line chat/instruct models without dedicated reasoning passes, broad capability surface. | Default production tier when latency matters and reasoning is moderate. |
| **fast-cheap** | Smaller, faster, cheaper instruct models (often 1/10 the price of frontier-general). | High-volume classification, extraction, summarization, routing, simple tool calls. |
| **on-device** | Open-weight models running locally (laptop, phone, edge). | Privacy-bound data, offline operation, latency floors below ~100ms TTFT. |

**As of 2026-05**, frontier-reasoning examples include the OpenAI o-series, Claude extended/adaptive thinking, Gemini thinking-tier models, DeepSeek-R1 / `deepseek-reasoner`, and Qwen3-thinking variants. Verify provider docs for current model IDs and pricing — never hardcode an ID in prose, prompt template, or commit message.

> **Why tiers and not IDs?** Provider lineups churned three times between 2024 and 2026. A skill that names "GPT-4o" or "Claude 3.5 Sonnet" is wrong within months. A skill that says "frontier-reasoning tier" stays correct.


## When Reasoning Models Beat Chat Models

The decision is about **task shape**, not difficulty. A "hard" extraction task is still extraction; a "simple" math word problem can still trip a chat model.

### Reach for reasoning models (frontier-reasoning) when:

- **Math and quantitative problems** — multi-step arithmetic, algebra, optimization, proofs. Reasoning models trained with reinforcement learning on verifiable math/code rewards consistently outperform chat models on competition-grade benchmarks ([DeepSeek-R1 paper, Guo et al. 2025](https://arxiv.org/abs/2501.12948); [OpenAI o1 system card](https://openai.com/index/openai-o1-system-card/)).
- **Code reasoning** — debugging by reasoning about state, multi-file refactors, race conditions, complex type inference. Not "write me a CRUD endpoint" (chat-tier handles that).
- **Multi-step planning** — task decomposition, constraint satisfaction, scheduling, route planning.
- **Agentic-loop reliability** — when the model must decide which of N tools to call and recover from tool errors. Reasoning models loop with fewer dead-ends and self-correct on bad tool results. Cross-ref [agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md).
- **Eval-grade verification** — using the model as a judge or as a checker on critical paths.

### Don't reach for reasoning models when:

- **Simple lookups and classification** — "is this review positive?" Chat-tier or fast-cheap tier wins on cost and latency by 5-50x.
- **Conversational chat** — reasoning traces wreck pacing; users wait through invisible thinking. Chat-tier or frontier-general is correct.
- **Latency-critical paths** — anything where p95 must be under ~2 seconds. Thinking inflates time-to-first-token by seconds to minutes.
- **Output-style-sensitive tasks** — creative writing, marketing copy, brand voice. Reasoning models often produce drier, more clinical prose because the RL reward signal optimized for correctness, not voice.
- **High-volume routing/triage** — the cost compounds; route through fast-cheap and only escalate to reasoning when the routing model flags hard cases.

### Decision matrix

| Task type | Tier | Why |
|---|---|---|
| "Solve this AIME-style problem" | frontier-reasoning | RL-on-math wins decisively |
| "Classify this support ticket" | fast-cheap | Reasoning is wasted spend |
| "Plan a 5-step refactor across 12 files" | frontier-reasoning | Multi-step planning |
| "Write a press release" | frontier-general | Voice and style |
| "Pick the right tool from 8 options and call it" | frontier-reasoning (or frontier-general with care) | Agentic reliability |
| "Extract entities from this email" | fast-cheap | Pattern matching |
| "Summarize this 200-page document" | frontier-general (or 1M-context tier) | Volume, not reasoning depth |
| "Verify a numerical answer is correct" | frontier-reasoning | Verification benefits from thinking |


## Thinking-Token Budget Controls

Each provider exposes thinking control differently. Verify the current parameter names and ranges in provider docs — these change.

### Anthropic — Claude extended thinking / adaptive thinking

- `thinking.type: "enabled"` with `budget_tokens` (integer, ≥1024) for older Claude reasoning-capable models ([Claude API docs: Building with extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)).
- `thinking.type: "adaptive"` with `effort` parameter for newer Claude models, where the model dynamically allocates thinking based on problem difficulty ([Claude API docs: Adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking)).
- `budget_tokens` must be less than `max_tokens`, except when interleaved thinking with tools is enabled (then bounded by the context window).
- In Claude 4 and later models, the budget applies to *full* thinking tokens, not the summarized output the API returns.

### OpenAI — o-series / GPT-5+ reasoning

- `reasoning.effort` parameter with values like `minimal | low | medium | high | xhigh`; defaults and supported values are model-dependent ([OpenAI reasoning models guide](https://developers.openai.com/api/docs/guides/reasoning); [OpenAI community thread on supported values](https://community.openai.com/t/request-for-compatibility-matrix-reasoning-effort-sampling-parameters-across-gpt-5-series/1371738)).
- Effort is a coarse abstraction over compute; the exact internal token budget is opaque. You pay for output tokens that include hidden reasoning tokens.
- Some sampling parameters (temperature, top_p) are restricted or ignored on reasoning models — check the compatibility matrix per model.

### Google — Gemini thinking

- Gemini 2.5: `thinkingBudget` (integer 0–24576, where 0 disables thinking).
- Gemini 3+: `thinking_level` with discrete levels (e.g., `LOW | MEDIUM | HIGH`), defaulting to dynamic high. ([Gemini API: Thinking](https://ai.google.dev/gemini-api/docs/thinking)).

### DeepSeek — `deepseek-reasoner`

- Set `model=deepseek-reasoner`; thinking is on by default with `effort=high`. Some agent contexts (Claude Code, OpenCode) auto-bump to `max`.
- Thinking content is returned in a separate `reasoning_content` field at the same level as `content` ([DeepSeek API docs: Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)).

### Qwen — Qwen3 / Qwen3-thinking / QwQ

- Qwen3 supports a hybrid thinking mode that can be toggled on/off; some Qwen3 variants are dedicated thinking-only models (e.g., `qwen3-235b-a22b-thinking-2507`) ([Qwen3 announcement, Alibaba 2025](https://qwenlm.github.io/blog/qwen3/)). On Alibaba Model Studio, thinking is exposed via parameters described in the [deep-thinking-models guide](https://www.alibabacloud.com/help/en/model-studio/deep-thinking).
- QwQ-32B was the predecessor reasoning model; Qwen3-30B-A3B (MoE, ~3B activated) outperforms QwQ-32B on most reasoning benchmarks per the Qwen team's release notes.

### xAI — Grok thinking

- Grok exposes a thinking mode (chain-of-thought style). Specifics on a budget knob vary by model and platform; **verify in xAI's current API docs before relying on a budget parameter** — this sheet does not assert one exists.

> **Anti-pattern:** Setting `budget_tokens` or `effort` to maximum "to be safe." You pay for unused thinking, latency balloons, and you get diminishing-returns reasoning quality. Start at the minimum (1024 for Anthropic, `low` for OpenAI, dynamic-default for Gemini) and raise only when evals demand it.


## Prompting Reasoning Models: Less Is More

Reasoning models were post-trained to think before answering. Prompting them like chat models actively hurts performance.

### What to do

- **Keep system prompts short.** State the role and the hard constraints. Resist filling them with examples or scaffolding.
- **State the task and the success criterion plainly.** "Solve this. Show your final answer in `\boxed{}`." Done.
- **Provide *minimal* context.** If the model needs three files to reason, give three files; don't dump twelve "for context."
- **Set output structure with structured outputs**, not by example. See [agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md).
- **Trust the trace.** If the model is wrong, raise the budget/effort one notch before rewriting the prompt.

### What to avoid

- **"Think step by step."** The model is *already* thinking. Adding the phrase either no-ops or duplicates the trace. ([OpenAI o1 prompting guide](https://platform.openai.com/docs/guides/reasoning-best-practices) / equivalent provider guidance.)
- **Heavy few-shot examples (5+ examples).** With chat models, few-shot teaches format. With reasoning models, few-shot can constrain the reasoning trace harmfully — the model anchors on the example traces instead of reasoning fresh. Use 0–2 examples, and only for output format, not for reasoning style.
- **Chain-of-thought scaffolds in the prompt.** "First do X, then Y, then Z." This constrains the reasoning model's natural decomposition. Let it plan.
- **Temperature tuning in the chat-model way.** Many reasoning models ignore or restrict temperature; the effective sampling is controlled internally. Read provider compatibility notes per model.

### Provider-specific prompting notes

- **Anthropic adaptive thinking:** the model auto-scales thinking; for hard tasks the `effort` parameter is your dial.
- **OpenAI reasoning models:** the [OpenAI reasoning best-practices guide](https://platform.openai.com/docs/guides/reasoning-best-practices) explicitly recommends short prompts and warns against "think step by step" instructions.
- **DeepSeek-R1:** the team's [release post](https://api-docs.deepseek.com/news/news250120) recommends including a final-answer format directive (e.g., "Please reason step by step, and put your final answer within `\boxed{}`") for math evaluation, *but* this is a benchmarking convention, not a general recipe.


## Cost and Latency Profile: The "Thinking Tax"

Reasoning models change the economics in three ways.

### 1. Output-token bill is dominated by hidden reasoning

A reasoning model that produces a 100-token visible answer often emits 2,000–20,000 reasoning tokens to get there. You are billed for **all** of them at output-token rates. A query that costs $0.0002 on a fast-cheap model can cost $0.05 on a frontier-reasoning model — a 250x swing.

**Mitigations:**
- Cap `budget_tokens` / use `effort=low` for tasks where you've shown via eval that low effort is enough.
- Route through a fast-cheap classifier that decides whether a query needs reasoning.
- Cache aggressively: the *prompt* portion is cacheable across queries the same way as any other model. See [context-engineering-and-prompt-caching.md](context-engineering-and-prompt-caching.md).

### 2. Latency: time-to-first-visible-token is high

Thinking is wall-clock-blocking. A 5-second thinking pass means 5 seconds of nothing visible to the user. For UX, this is fine for batch, deadly for chat.

**Mitigations:**
- Stream a "thinking..." indicator if the provider exposes thinking events (Anthropic streams thinking deltas, for example).
- Use reasoning models for non-interactive paths: nightly evals, async agentic tasks, batch verification.
- For interactive paths, route to frontier-general unless the task obviously needs reasoning.

### 3. The "thinking tax" is real and measurable

The thinking tax is the extra cost and latency you pay for thinking on tasks a chat model would have solved correctly in one pass.

**Quantifying it:**
- Pick 100 production queries, run them through your current chat model and through a reasoning model.
- Measure: accuracy delta, cost delta, p50/p95 latency delta.
- If accuracy delta is < 5pp on tasks worth < $X each, the thinking tax is not worth it.

> **Cross-ref:** [yzmir-ml-production](../../yzmir-ml-production/) covers serving-stack ops including KV-cache strategy, batching, and the cost monitoring that makes the thinking tax visible.


## Evaluation: Reason About the Reason

Reasoning models break standard eval pipelines in three ways.

### 1. Track thinking tokens separately

Your dashboard needs three counters: input tokens, *visible* output tokens, *thinking* output tokens. Without the third you cannot diagnose a cost regression.

### 2. Eval reasoning quality, not just final answer

A reasoning model that gets the right answer via a wrong trace is a ticking bomb — it will fail on the next adversarial input. Eval the trace:
- **Faithfulness** — does the trace match the answer? (LLM-as-judge or human spot-check.)
- **Step validity** — are the intermediate steps individually correct?
- **No "lucky guess" patterns** — model arrives at the answer without justification.

For math/code, ground-truth correctness is the cleanest signal. For open-ended tasks, use rubric-based judging. Cross-ref [llm-evaluation-metrics.md](llm-evaluation-metrics.md).

### 3. Watch for reasoning-mode regressions

Reasoning models can "get dumber" when:
- Effort budget is too low for the task — model starts reasoning, gives up, returns a chat-style guess.
- Prompt format pushes the model out of distribution — heavy few-shot or scaffolding.
- Tool-calling agentic loops where tool results are noisy — model gets confused mid-trace.

Add eval cases that probe these specifically: low-budget hard problems, prompts with adversarial scaffolding, agent traces with bad tool returns.


## Common Failure Modes

### Failure mode 1: Over-thinking loops

The model keeps reasoning, restating, second-guessing. Visible symptom: max thinking budget hit, no answer or repetitive answer.

**Causes:** budget too high, ambiguous task, model uncertain about success criterion.

**Fixes:** lower the budget, sharpen the success criterion in the prompt, add an explicit "if uncertain, output `UNKNOWN`" escape hatch.

### Failure mode 2: Under-thinking on hard problems

The model gives a chat-style guess on a problem that needed thinking.

**Causes:** budget too low, effort too low, task triggered the model's "I know this" heuristic.

**Fixes:** raise budget, force higher effort, add a verification step ("after answering, verify by re-deriving").

### Failure mode 3: Confidently wrong

The model produces a long, internally-consistent trace that arrives at a wrong answer with high confidence.

**Causes:** RL reward signals optimize for plausible-looking traces; on out-of-distribution problems, the model hallucinates plausibility.

**Fixes:** require external verification (run the code, check the math symbolically); pair with a chat-tier judge; ensemble multiple reasoning runs and look for disagreement.

### Failure mode 4: Refusal-when-uncertain inflation

Some reasoning models, post-RL, became more conservative — they refuse when uncertain. Sometimes this is correct; often it's a regression vs. their chat-model siblings.

**Causes:** safety post-training emphasis, uncertainty calibration shifted.

**Fixes:** explicit prompt permission ("if uncertain, give your best guess and flag uncertainty"); fall back to a chat-tier model when a reasoning model refuses; track refusal rate as a first-class metric.

### Failure mode 5: Style regression on creative tasks

Output reads as drier, more bulleted, less voiced than the chat-tier sibling.

**Causes:** RL on verifiable rewards (math, code) doesn't reward style.

**Fix:** don't use reasoning models for style-sensitive tasks. Route style work to frontier-general.


## Reasoning Models in Agent Loops

Reasoning models change agent-loop dynamics. They:

- Recover better from tool errors (they reason about why a tool failed).
- Decide tool-call order more reliably on multi-step plans.
- But: they amplify the cost of every loop iteration (each iteration is a full thinking pass).

**Pattern:** use frontier-reasoning at the planner/orchestrator layer; use fast-cheap or frontier-general at the worker/sub-agent layer that executes individual tool calls. This is the orchestrator-worker pattern. See [agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md) for full coverage of multi-agent orchestration.


## Anti-Patterns

### Anti-pattern 1: "Use the smartest model by default"

**Wrong:** Default to frontier-reasoning for every query "to be safe."

**Right:** Default to fast-cheap; escalate to frontier-general or frontier-reasoning when evals show the cheaper tier underperforms by a margin worth paying for.

**Principle:** Compute is not free. The thinking tax is real and compounds with volume.

### Anti-pattern 2: "Add 'think step by step' to make it better"

**Wrong:** Append "Let's think step by step" to a reasoning-model prompt.

**Right:** Trust the trained reasoning. State the task plainly.

**Principle:** Reasoning models already think; redundant scaffolding constrains rather than helps.

### Anti-pattern 3: "Hardcode the model ID"

**Wrong:** `model="o1-preview"` (or any specific ID) baked into config and docs.

**Right:** Reference by tier in code via env var; pick the ID at deploy time.

**Principle:** Provider lineups churn; tier-based abstraction outlives any specific ID. **Model lineup current as of 2026-05; revisit quarterly.**

### Anti-pattern 4: "Eval on final answer only"

**Wrong:** Pass/fail metric on `output == ground_truth`.

**Right:** Score the trace, track thinking tokens, watch for confidently-wrong patterns.

**Principle:** Reasoning quality is the asset; final-answer-only metrics miss the regressions.

### Anti-pattern 5: "Reasoning model in the chat path"

**Wrong:** Putting a reasoning model behind an interactive chat UI.

**Right:** Frontier-general for chat; reasoning model in async/batch paths.

**Principle:** Latency dominates UX; thinking blocks first-token by seconds to minutes.

### Anti-pattern 6: "Max budget, just in case"

**Wrong:** Set `budget_tokens=64000` or `effort=xhigh` everywhere.

**Right:** Calibrate budget per task class via eval; default to low/min.

**Principle:** Diminishing returns past task-appropriate budget; cost compounds linearly.


## Summary

**Capability tiers** (cross-pack vocabulary): frontier-reasoning, frontier-general, fast-cheap, on-device. Reference by tier, not by model ID.

**When reasoning beats chat:** math, code reasoning, multi-step planning, agent reliability. Otherwise prefer frontier-general or fast-cheap.

**Prompting:** less is more. Short system prompt, plain task, minimal context, 0–2 examples max. **No** "think step by step." **No** heavy CoT scaffolds.

**Cost/latency:** thinking tokens dominate the bill and the wall-clock. Cap effort/budget, route through cheap tiers first, batch where possible.

**Eval:** track thinking tokens separately, score the trace, watch for confidently-wrong and refusal-inflation regressions.

**Failure modes:** over-thinking loops, under-thinking guesses, confidently wrong, refusal-when-uncertain, style regression.

**Cross-refs:**
- [agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md) — reasoning models in tool-use loops
- [context-engineering-and-prompt-caching.md](context-engineering-and-prompt-caching.md) — caching reasoning prompts
- [llm-evaluation-metrics.md](llm-evaluation-metrics.md) — eval discipline for reasoning quality
- [yzmir-ml-production](../../yzmir-ml-production/) — serving cost monitoring, the thinking-tax dashboard
- [yzmir-training-optimization](../../yzmir-training-optimization/) — preference-tuning training dynamics behind RL-on-verifiable-rewards models

---

*Model lineup and provider feature set current as of 2026-05; revisit quarterly. Verify provider docs for current model IDs and pricing.*
