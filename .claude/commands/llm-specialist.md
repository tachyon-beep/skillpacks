---
description: Use when doing LLM application engineering - chat / instruct prompting, reasoning models (o-series / Claude extended thinking / DeepSeek-R1 / Gemini thinking / Qwen QwQ), agentic patterns + MCP, RAG, fine-tuning (SFT / DPO / IPO / KTO / SimPO / ORPO / GRPO + LoRA family), context engineering and prompt caching, inference optimization (vLLM / SGLang / TensorRT-LLM), evaluation (incl. LLM-as-judge bias controls and capability suites), and safety (OWASP LLM Top 10 2025). Calibrated to 2026-05 with capability-tier vocabulary (frontier-reasoning / frontier-general / fast-cheap / on-device) instead of hardcoded model IDs.
---

# LLM Specialist Routing

**Chat / instruct models and reasoning models are different categories. Prompting rules, evaluation rules, and cost models diverge between them — the router's Step 0 is the reasoning-vs-chat gate, not an afterthought. Model IDs rotate quarterly; this pack uses capability tiers (`frontier-reasoning`, `frontier-general`, `fast-cheap`, `on-device`) so guidance ages gracefully. For training infrastructure use `/training-optimization`; for production serving use `/ml-production`; for adversarial-ML threat modeling use `/security-architect`.**

Use the `using-llm-specialist` skill from the `yzmir-llm-specialist` plugin to route to the right specialist sheet. Content authority lives in `plugins/yzmir-llm-specialist/skills/using-llm-specialist/SKILL.md` - this wrapper is a thin pointer.

## Routing entry point

**Step 0 (do this first):** Is the target a reasoning model (OpenAI o-series, Claude extended thinking, DeepSeek-R1 and distillations, Gemini "thinking", Qwen QwQ)? If yes, go to `reasoning-models` *before* any other sheet — generic prompting advice misroutes here.

**Step 1:** Otherwise, identify the task category and route below.

## Sheets

- **reasoning-models** - o-series / Claude extended thinking / DeepSeek-R1 / Gemini thinking / Qwen QwQ; reasoning-effort budgets, thinking-token economics, when NOT to add explicit chain-of-thought, reasoning-model evaluation specifics
- **prompt-engineering-patterns** - chat / instruct prompting: instruction clarity, few-shot, chain-of-thought (Wei 2022 / Kojima 2022), system message design, output formatting, prompt optimization; also covers general multimodal principles
- **agentic-patterns-and-mcp** - agent loops (ReAct, planner / executor), tool selection and error recovery, Model Context Protocol (MCP) servers and clients, structured output (provider features + Outlines + Instructor), multi-agent orchestration, agent observability; prompt-injection-via-tool-results named as a first-class hazard
- **context-engineering-and-prompt-caching** - four-provider caching comparison (Anthropic explicit / OpenAI automatic / Gemini implicit / Gemini explicit), cache-prefix anchoring rule (stable prefix first, volatile suffix last), long-context layout, cost / latency math, cache-vs-RAG-vs-fine-tune decision
- **rag-architecture-patterns** - RAG architecture, retrieval strategies (dense / sparse / hybrid), chunking, re-ranking, context injection, RAG evaluation; cross-routes to caching for small / static corpora and to reasoning-models for pure reasoning
- **llm-finetuning-strategies** - modern preference-tuning lineage (PPO → DPO → IPO / KTO / SimPO / ORPO → GRPO), LoRA family (LoRA / QLoRA / DoRA / rsLoRA / LoftQ / LongLoRA), dataset preparation, hyperparameters, catastrophic-forgetting prevention, premature-fine-tune gates
- **context-window-management** - 128k-200k baseline vs 1M+ tier honesty with RULER recall caveat, summarization strategies, sliding window, hierarchical context, token counting, lost-in-the-middle mitigation
- **llm-evaluation-metrics** - task metrics, human evaluation, LLM-as-judge with bias controls (Zheng 2023; Dubois 2024), capability suites (Inspect AI / OLMES / OpenAI Evals / lm-evaluation-harness), reasoning-model eval (thinking-token tracking), red-teaming primer (PAIR / GCG / AutoDAN), golden-set discipline and contamination checks
- **llm-inference-optimization** - latency reduction, throughput, batching, KV cache, quantization (INT8 / INT4), speculative decoding, serving stacks (vLLM / SGLang / TensorRT-LLM)
- **llm-safety-alignment** - OWASP LLM Top 10 (2025 edition) leads, structural prompt-injection defenses (not just sanitization), jailbreak detection, content filtering, bias mitigation, hallucination reduction, agentic safety when tools are in play, guardrails

## Commands

- `/yzmir-llm-specialist:debug-generation` - diagnose poor LLM output quality with symptom-triage table and decision tree; rejects premature-fine-tune; user-explicit entry point
- `/yzmir-llm-specialist:optimize-inference` - systematic latency / throughput / cost optimization across caching, parallelization, model routing, quantization
- `/yzmir-llm-specialist:rag-audit` - audit an existing RAG system against retrieval / chunking / re-ranking / evaluation discipline

## Agents

- `llm-diagnostician` (sonnet) - SME reviewer for LLM output-quality issues; dispatched autonomously where `/debug-generation` is for explicit invocation; declines safety work and hands off
- `llm-safety-reviewer` (opus) - SME reviewer for safety / alignment / prompt-injection / jailbreak surfaces; declines performance work and hands off

Both agents follow the SME Agent Protocol with Confidence / Risk / Information Gaps / Caveats sections.

## Cross-references

- Top-level AI/ML routing → `/ai-engineering`
- Production serving, deployment, monitoring infrastructure → `/ml-production`
- Pretraining and fine-tuning at scale (FSDP2 / FP8 / optimizer sharding / muP) → `/training-optimization`
- Adversarial-ML threat modeling, supply-chain risk, defense-in-depth → `/security-architect`
- PyTorch-level debugging (NaN / OOM / autograd) → `/pytorch-engineering`

## Known gap

Multimodal prompting has no dedicated sheet yet. General principles transfer via `prompt-engineering-patterns`; provider-specific image-token economics, resolution settings, multimodal jailbreaks, and multimodal evals are flagged as forthcoming. Do not freelance in this gap.
