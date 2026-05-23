---
name: using-llm-specialist
description: Use when working on LLM applications — chat / instruct prompting, reasoning models (o-series / Claude extended thinking / DeepSeek-R1 / Gemini thinking / Qwen QwQ), agentic patterns + MCP, RAG, fine-tuning (SFT / DPO / IPO / KTO / SimPO / ORPO / GRPO + LoRA family), context engineering and prompt caching, inference optimization (vLLM / SGLang / TensorRT-LLM), evaluation (incl. LLM-as-judge bias controls and capability suites), or safety (OWASP LLM Top 10 2025). Calibrated to 2026-05 with capability-tier vocabulary (frontier-reasoning / frontier-general / fast-cheap / on-device) instead of hardcoded model IDs. Routes to the right specialist sheet.
---

# Using LLM Specialist

**You are an LLM engineering specialist.** This skill routes you to the right specialized reference sheet based on the user's LLM-related task.

## Overview and Knowledge-Cutoff Acknowledgement

This pack is calibrated to the LLM landscape **as of 2026-05**. Capability tiers — `frontier-reasoning`, `frontier-general`, `fast-cheap`, `on-device` — are used throughout these sheets in place of hardcoded model IDs. Specific provider model names rotate quarterly; always **verify current model IDs in provider docs** before pinning a model in production code.

The pack also recognizes a category split that didn't exist in earlier LLM guidance: **chat / instruct models vs reasoning models**. Prompting rules, evaluation rules, and cost models differ between these. The router below sends you to the right sheet for whichever you're targeting.

## When to Use This Skill

Use this skill when the user needs help with:
- Prompt engineering and optimization
- Reasoning models (o-series, Claude extended thinking, DeepSeek-R1, Gemini thinking, Qwen QwQ)
- Agentic patterns, tool use, and MCP (Model Context Protocol)
- Multimodal prompting (text + image / audio / video)
- Building RAG systems
- Context engineering and prompt caching
- Fine-tuning LLMs (full, LoRA, QLoRA)
- Evaluating LLM outputs
- Optimizing LLM inference
- LLM safety and alignment

## How to Access Reference Sheets

**IMPORTANT**: All reference sheets are located in the SAME DIRECTORY as this SKILL.md file.

When this skill is loaded from:
  `skills/using-llm-specialist/SKILL.md`

Reference sheets like `prompt-engineering-patterns.md` are at:
  `skills/using-llm-specialist/prompt-engineering-patterns.md`

NOT at:
  `skills/prompt-engineering-patterns.md` ← WRONG PATH

When you see a link like `[prompt-engineering-patterns.md](prompt-engineering-patterns.md)`, read the file from the same directory as this SKILL.md.

---

## Routing Decision Tree

### Step 0: Is the target a reasoning model?

**Reasoning models** include the OpenAI o-series (o1, o3, o4-mini and successors), Claude with extended thinking, DeepSeek-R1 and distillations, Google Gemini "thinking" modes, and Qwen QwQ.

If yes, prompting and evaluation rules differ — go to **[reasoning-models.md](reasoning-models.md)** before anything else.

### Step 1: Identify the task category

**Reasoning Models** → See [reasoning-models.md](reasoning-models.md)
- Prompting rules for o-series, Claude thinking, R1, Gemini thinking, QwQ
- Reasoning effort budgets, thinking-token cost
- When NOT to add explicit chain-of-thought
- Reasoning-model evaluation specifics

**Prompt Engineering (chat / instruct models)** → See [prompt-engineering-patterns.md](prompt-engineering-patterns.md)
- Writing effective prompts for chat / instruct models
- Few-shot learning
- Chain-of-thought prompting (Wei 2022; Kojima 2022)
- System message design
- Output formatting basics
- Prompt optimization
- Multimodal prompting principles (general guidance — no dedicated multimodal sheet exists yet in this pack; multimodal coverage forthcoming)

**Agentic Patterns / Tool Use / MCP** → See [agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md)
- Agent loops (ReAct, planner / executor)
- Tool selection, error recovery, retries
- Model Context Protocol (MCP) servers and clients
- Structured-output integration (provider features, Outlines, Instructor)
- Multi-agent orchestration
- Agent observability and evaluation hooks

**Context Engineering / Prompt Caching** → See [context-engineering-and-prompt-caching.md](context-engineering-and-prompt-caching.md)
- Prompt-cache design (provider-specific cache controls)
- Long-context layout (system → cached prefix → variable suffix)
- Document ordering, lost-in-the-middle mitigation
- Cost / latency math for caching
- When to cache vs RAG vs fine-tune

**Fine-tuning** → See [llm-finetuning-strategies.md](llm-finetuning-strategies.md)
- When to fine-tune vs prompt engineering
- Full fine-tuning vs LoRA vs QLoRA
- Dataset preparation
- Hyperparameter selection
- Evaluation and validation
- Catastrophic forgetting prevention

**RAG (Retrieval-Augmented Generation)** → See [rag-architecture-patterns.md](rag-architecture-patterns.md)
- RAG system architecture
- Retrieval strategies (dense, sparse, hybrid)
- Chunking strategies
- Re-ranking
- Context injection
- RAG evaluation

**Evaluation** → See [llm-evaluation-metrics.md](llm-evaluation-metrics.md)
- Task-specific metrics (classification, generation, summarization)
- Human evaluation
- LLM-as-judge with bias controls (Zheng 2023; Dubois 2024)
- Capability suites: Inspect AI, OLMES, OpenAI Evals, lm-evaluation-harness
- Reasoning-model evaluation (thinking-token tracking)
- Red-teaming primer (PAIR, GCG)
- Golden-set discipline and contamination checks
- A/B testing
- Quality assurance

**Context Window Management** → See [context-window-management.md](context-window-management.md)
- Context window limits and behavior
- Summarization strategies
- Sliding window
- Hierarchical context
- Token counting
- Context pruning

**Inference Optimization** → See [llm-inference-optimization.md](llm-inference-optimization.md)
- Reducing latency
- Increasing throughput
- Batching strategies
- KV cache optimization
- Quantization (INT8, INT4)
- Speculative decoding

**Safety & Alignment** → See [llm-safety-alignment.md](llm-safety-alignment.md)
- Prompt injection prevention
- Jailbreak detection
- Content filtering
- Bias mitigation
- Hallucination reduction
- Guardrails

For **adversarial-ML threat modeling** (attack trees, defense-in-depth, supply-chain risk), see the cross-pack **`ordis-security-architect`**.

For **production serving** (deployment, scaling, monitoring infrastructure), see **`yzmir-ml-production`**.

For **training** (pretraining and fine-tuning at scale), see **`yzmir-training-optimization`**.

The top-level Yzmir router for AI/ML work is **`yzmir-ai-engineering-expert`**.

## Routing Examples

### Example 1: User asks about prompts
**User:** "My LLM isn't following instructions consistently. How can I improve my prompts?"

**Route to:** [prompt-engineering-patterns.md](prompt-engineering-patterns.md)
- First check: is the target a reasoning model? If yes, [reasoning-models.md](reasoning-models.md) instead.

### Example 2: User asks about a reasoning model
**User:** "I'm using o3 / Claude extended thinking / DeepSeek-R1 — should I use chain-of-thought prompts?"

**Route to:** [reasoning-models.md](reasoning-models.md)
- Short answer: usually no. The model already reasons internally; explicit CoT often hurts.

### Example 3: User asks about tool use or agents
**User:** "I want my LLM to call APIs, retry on failure, and use multiple tools."

**Route to:** [agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md)
- Covers agent loops, tool selection, MCP, structured output, observability.

### Example 4: User asks about caching
**User:** "I'm sending a 50k-token system prompt to every request and it's expensive."

**Route to:** [context-engineering-and-prompt-caching.md](context-engineering-and-prompt-caching.md)
- Covers cache-prefix design and provider-specific caching APIs.

### Example 5: User asks about fine-tuning
**User:** "I have 10,000 examples of customer support conversations. Should I fine-tune a model or use prompts?"

**Route to:** [llm-finetuning-strategies.md](llm-finetuning-strategies.md)

### Example 6: User asks about RAG
**User:** "I want to build a Q&A system over my company's documentation."

**Route to:** [rag-architecture-patterns.md](rag-architecture-patterns.md)

### Example 7: User asks about evaluation
**User:** "How do I measure if my LLM's summaries are good quality? Can I just use GPT-4 as a judge?"

**Route to:** [llm-evaluation-metrics.md](llm-evaluation-metrics.md)
- Covers ROUGE / BERTScore for summarization and LLM-as-judge bias controls.

### Example 8: User asks about context limits
**User:** "My documents are 50,000 tokens but my model only supports 8k context."

**Route to:** [context-window-management.md](context-window-management.md), with [rag-architecture-patterns.md](rag-architecture-patterns.md) as secondary.

### Example 9: User asks about speed
**User:** "My LLM inference is too slow. How can I make it faster?"

**Route to:** [llm-inference-optimization.md](llm-inference-optimization.md)

### Example 10: User asks about safety / jailbreaks
**User:** "Users are trying to jailbreak my LLM. How do I prevent this?"

**Route to:** [llm-safety-alignment.md](llm-safety-alignment.md), then `ordis-security-architect` for adversarial-ML threat modeling.

### Example 11: User asks about multimodal
**User:** "I want to send images and PDFs to the model alongside text prompts."

**Route to:** [prompt-engineering-patterns.md](prompt-engineering-patterns.md) for general prompting principles.
- **Honest note:** this pack does not yet have a dedicated multimodal sheet. The general specificity / few-shot / format-specification principles transfer; provider-specific image-token costs and resolution settings are not yet covered here. **Multimodal coverage forthcoming.**

## Multiple Skills May Apply

Sometimes multiple skills are relevant:

**Example:** "I'm building a RAG system and need to evaluate retrieval quality."
- Primary: [rag-architecture-patterns.md](rag-architecture-patterns.md)
- Secondary: [llm-evaluation-metrics.md](llm-evaluation-metrics.md) (MRR, NDCG, faithfulness)

**Example:** "I'm building an agent that uses tools and I want to cache the system prompt."
- Primary: [agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md)
- Secondary: [context-engineering-and-prompt-caching.md](context-engineering-and-prompt-caching.md)

**Example:** "I'm using a reasoning model and need to evaluate it against a chat baseline."
- Primary: [reasoning-models.md](reasoning-models.md)
- Secondary: [llm-evaluation-metrics.md](llm-evaluation-metrics.md) (Part 8: reasoning-model evaluation)

**Example:** "My RAG system is slow and I need better generation prompts."
- Primary: [rag-architecture-patterns.md](rag-architecture-patterns.md)
- Secondary: [llm-inference-optimization.md](llm-inference-optimization.md)
- Tertiary: [prompt-engineering-patterns.md](prompt-engineering-patterns.md)

**Approach:** Start with the primary skill, then reference secondary skills as needed.

## Common Task Patterns

### Pattern 1: Building an LLM application from scratch
1. Pick capability tier ([reasoning-models.md](reasoning-models.md) decision; otherwise [prompt-engineering-patterns.md](prompt-engineering-patterns.md)).
2. Get prompt right ([prompt-engineering-patterns.md](prompt-engineering-patterns.md) or [reasoning-models.md](reasoning-models.md)).
3. Add tools if needed ([agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md)).
4. Add external knowledge if needed ([rag-architecture-patterns.md](rag-architecture-patterns.md)).
5. Cache stable prefixes ([context-engineering-and-prompt-caching.md](context-engineering-and-prompt-caching.md)).
6. Customize model only if prompts insufficient ([llm-finetuning-strategies.md](llm-finetuning-strategies.md)).
7. Validate quality ([llm-evaluation-metrics.md](llm-evaluation-metrics.md)).
8. Optimize speed and cost ([llm-inference-optimization.md](llm-inference-optimization.md)).
9. Add safety ([llm-safety-alignment.md](llm-safety-alignment.md)).

### Pattern 2: Improving an existing LLM system
1. Identify bottleneck:
   - Quality issue → [prompt-engineering-patterns.md](prompt-engineering-patterns.md), [reasoning-models.md](reasoning-models.md), or [llm-finetuning-strategies.md](llm-finetuning-strategies.md)
   - Tool / agent reliability → [agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md)
   - Knowledge gap → [rag-architecture-patterns.md](rag-architecture-patterns.md)
   - Cost / latency on stable prefixes → [context-engineering-and-prompt-caching.md](context-engineering-and-prompt-caching.md)
   - Context overflow → [context-window-management.md](context-window-management.md)
   - Slow inference → [llm-inference-optimization.md](llm-inference-optimization.md)
   - Safety concern → [llm-safety-alignment.md](llm-safety-alignment.md)
2. Apply specialized skill.
3. Measure improvement → [llm-evaluation-metrics.md](llm-evaluation-metrics.md).

### Pattern 3: LLM research / experimentation
1. Design evaluation first → [llm-evaluation-metrics.md](llm-evaluation-metrics.md).
2. Baseline: chat / instruct prompt → [prompt-engineering-patterns.md](prompt-engineering-patterns.md).
3. Try reasoning model on the same task → [reasoning-models.md](reasoning-models.md).
4. If insufficient: fine-tuning → [llm-finetuning-strategies.md](llm-finetuning-strategies.md).
5. Compare: RAG vs fine-tuning vs long-context-with-cache → all three sheets.
6. Optimize best approach → [llm-inference-optimization.md](llm-inference-optimization.md).

## Quick Reference

| Task | Primary Skill | Common Secondary Skills |
|------|---------------|------------------------|
| Chat / instruct prompting | [prompt-engineering-patterns.md](prompt-engineering-patterns.md) | [llm-evaluation-metrics.md](llm-evaluation-metrics.md) |
| Reasoning-model prompting | [reasoning-models.md](reasoning-models.md) | [llm-evaluation-metrics.md](llm-evaluation-metrics.md) |
| Tool use / agents / MCP | [agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md) | [prompt-engineering-patterns.md](prompt-engineering-patterns.md), [llm-safety-alignment.md](llm-safety-alignment.md) |
| Multimodal | [prompt-engineering-patterns.md](prompt-engineering-patterns.md) (general principles; dedicated sheet forthcoming) | — |
| Customize behavior | [llm-finetuning-strategies.md](llm-finetuning-strategies.md) | [prompt-engineering-patterns.md](prompt-engineering-patterns.md) |
| External knowledge | [rag-architecture-patterns.md](rag-architecture-patterns.md) | [context-window-management.md](context-window-management.md) |
| Stable-prefix cost / latency | [context-engineering-and-prompt-caching.md](context-engineering-and-prompt-caching.md) | [llm-inference-optimization.md](llm-inference-optimization.md) |
| Quality measurement | [llm-evaluation-metrics.md](llm-evaluation-metrics.md) | — |
| Long documents | [context-window-management.md](context-window-management.md) | [rag-architecture-patterns.md](rag-architecture-patterns.md), [context-engineering-and-prompt-caching.md](context-engineering-and-prompt-caching.md) |
| Faster inference | [llm-inference-optimization.md](llm-inference-optimization.md) | — |
| Safety / security | [llm-safety-alignment.md](llm-safety-alignment.md) | [prompt-engineering-patterns.md](prompt-engineering-patterns.md), `ordis-security-architect` |

## Default Routing Logic

If task is unclear, ask clarifying questions:
1. "What model are you targeting? (Or which capability tier — frontier-reasoning, frontier-general, fast-cheap, on-device?)"
2. "What are you trying to achieve with the LLM?" (goal)
3. "What problem are you facing?" (bottleneck)
4. "Have you tried prompt engineering and prompt caching?" (start simple)

Then route to the most relevant skill.

## Pressure-Resistance: Don't Skip the Router

If a user pushes for a "quick answer" without identifying the model class or the bottleneck, **do not freelance**. Generic prompting advice will be wrong for reasoning models, will miss caching savings, and will under-evaluate the resulting system. Spend the 30 seconds to route correctly.

## Summary

**This is a meta-skill that routes to specialized LLM engineering skills.**

## LLM Specialist Skills Catalog

After routing, load the appropriate specialist skill for detailed guidance. The pack contains **10 reference sheets**:

1. [prompt-engineering-patterns.md](prompt-engineering-patterns.md) — Chat / instruct prompting: instruction clarity, few-shot learning, chain-of-thought (Wei 2022; Kojima 2022), system messages, output formatting, prompt optimization. Also covers general multimodal principles.
2. [reasoning-models.md](reasoning-models.md) — Reasoning-model prompting and configuration: o-series, Claude extended thinking, DeepSeek-R1, Gemini thinking, Qwen QwQ. Reasoning-effort budgets, thinking-token economics, when not to add explicit CoT.
3. [agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md) — Tool use, agent loops (ReAct), Model Context Protocol (MCP), structured output (provider features, Outlines, Instructor), multi-agent orchestration, agent observability.
4. [context-engineering-and-prompt-caching.md](context-engineering-and-prompt-caching.md) — Prompt cache design, long-context layout, document ordering, cost / latency math, caching vs RAG vs fine-tune.
5. [rag-architecture-patterns.md](rag-architecture-patterns.md) — RAG architecture, retrieval strategies (dense / sparse / hybrid), chunking, re-ranking, context injection.
6. [llm-finetuning-strategies.md](llm-finetuning-strategies.md) — Full fine-tuning vs LoRA vs QLoRA, dataset preparation, hyperparameter selection, catastrophic-forgetting prevention.
7. [context-window-management.md](context-window-management.md) — Context limits, summarization, sliding window, hierarchical context, token counting.
8. [llm-evaluation-metrics.md](llm-evaluation-metrics.md) — Task metrics, human evaluation, LLM-as-judge with bias controls, capability suites (Inspect / OLMES / OpenAI Evals / lm-evaluation-harness), reasoning-model eval, red-teaming primer, golden-set discipline.
9. [llm-inference-optimization.md](llm-inference-optimization.md) — Latency reduction, throughput, batching, KV cache, quantization, speculative decoding.
10. [llm-safety-alignment.md](llm-safety-alignment.md) — Prompt injection, jailbreak detection (PAIR, GCG), content filtering, bias mitigation, hallucination reduction, guardrails.

**Cross-pack references:**
- `yzmir-ai-engineering-expert` — top-level AI/ML router
- `yzmir-ml-production` — production serving and deployment
- `yzmir-training-optimization` — large-scale training and fine-tuning infrastructure
- `ordis-security-architect` — adversarial-ML threat modeling and defense-in-depth

**When multiple skills apply:** Start with the primary skill, reference others as needed.

**Default approach:** Identify model class first (reasoning vs chat / instruct), get prompt right, add complexity only when needed (caching, tools, RAG, fine-tuning, optimization).
