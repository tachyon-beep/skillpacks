# Refresh: yzmir-llm-specialist

**Verdict:** HIGH / L effort. **Most time-sensitive pack in the marketplace** — near-rewrite required.

## Context

- Pack path: `/home/john/skillpacks/plugins/yzmir-llm-specialist/`
- Full review: `/tmp/skillpack-refresh-review/yzmir-llm-specialist.md`
- Purpose: LLM application skills — prompting, RAG, fine-tuning, evaluation, inference, safety.

## Why refresh

Pervasive 2024-vintage content. The LLM landscape has moved hard:

**Stale model references:**
- GPT-3.5, GPT-4-turbo, Llama-2, Claude 3 only — none of the 2025+ flagship models.

**Missing entire reasoning-model class:**
- o-series (o1, o3-mini, o4)
- Claude extended thinking
- DeepSeek-R1
- Gemini thinking

**Missing agentic patterns:**
- Tool-use loops (modern Claude/OpenAI tool-calling)
- Multi-agent orchestration
- MCP (Model Context Protocol) — clients, servers, transport
- Tool-use anti-patterns

**Missing multimodal as default** (vision, audio, code-in-image).

**Stale fine-tuning:**
- No DPO / IPO / SimPO
- No GRPO
- No constitutional AI updates
- No RLAIF current state

**Stale RAG:**
- No contextual retrieval (Anthropic 2024 paper)
- No late-interaction / ColBERT-v2 current
- No modern embeddings (e.g. Voyage, Cohere v3+, OpenAI v3, Stella, BGE-M3)
- No modern rerankers (Cohere v3 rerank, Jina, Voyage)
- No hybrid retrieval current state

**Stale serving stack:**
- No vLLM, SGLang, TensorRT-LLM
- No continuous batching, paged attention, speculative decoding

**Missing context engineering:**
- Prompt caching (5-min, 1-hour TTL on Anthropic; OpenAI prompt caching)
- Extended caching strategies
- Cache-aware prompt structure

## Scope — DO (campaign-level)

Treat this as a **campaign**, not a single PR. Coordinate with the AI cluster:
- `yzmir-ai-engineering-expert` (router updated first)
- `yzmir-ml-production` (serving stack)
- `yzmir-training-optimization` (DPO/GRPO crossover)
- `yzmir-pytorch-engineering` (FSDP2/torch.compile crossover)
- `ordis-security-architect` (LLM threats — prompt injection, exfil)

**For this pack specifically:**

1. **Strip dated model lists.** Replace with capability-tier framing (frontier reasoning, frontier general, fast/cheap, on-device) and direct readers to provider docs for current model IDs.
2. **Add 3 new sheets** (or merge equivalents into existing if structure better):
   - Reasoning models — when, how to prompt, how to evaluate, cost profile, common failure modes.
   - Agentic patterns — tool-use loops, MCP integration, multi-agent orchestration, agentic anti-patterns.
   - Context engineering & prompt caching — cache-aware prompt structure, TTL strategy, token economics.
3. **Refresh RAG sheet.** Add contextual retrieval, modern embeddings/rerankers, hybrid retrieval, late interaction.
4. **Refresh fine-tuning.** Add DPO / IPO / SimPO / GRPO; reframe SFT vs preference tuning vs RLHF clearly.
5. **Refresh inference/serving.** Cover vLLM / SGLang / TensorRT-LLM / continuous batching / paged attention / speculative decoding.
6. **Refresh evaluation.** LLM-as-judge current state, golden set discipline, behavioral suites (e.g. Inspect AI), red-teaming primer.
7. **Refresh safety pointer.** Cross-ref `ordis-security-architect` for LLM threat modeling.

## Scope — DO NOT

- Do not include hardcoded model IDs that drift (use capability tiers instead).
- Do not duplicate content that belongs in `yzmir-training-optimization` or `yzmir-ml-production`.
- Do not fabricate technique names — every method named must have a paper or production system.

## Acceptance criteria

1. Zero references to GPT-3.5 / GPT-4-turbo / Llama-2 / Claude 3 as if current.
2. Reasoning models, agentic patterns, MCP, multimodal, prompt caching, contextual retrieval, DPO/GRPO, vLLM all covered.
3. Router skill updated to point at all sheets.
4. Cross-pack references to `yzmir-ml-production`, `yzmir-training-optimization`, `ordis-security-architect`.
5. `plugin.json` version bumped (major — this is a substantive rewrite).

## Process

1. Read `/tmp/skillpack-refresh-review/yzmir-llm-specialist.md` for full evidence.
2. Read every SKILL.md in this pack.
3. Confirm the AI router has already been refreshed (`yzmir-ai-engineering-expert`).
4. Plan: list each sheet to add/edit/remove. Run plan past user before writing.
5. Write sheets one at a time, verify each compiles to working examples (cite, don't fabricate).
6. Update router skill last.
7. Bump version.

## Constraints

- This is the most-cited LLM source for users of the marketplace — accuracy matters more than completeness.
- Every named technique must be cited (paper, blog, official docs).
- No model-ID hardcoding — capability tiers + "check provider docs for current IDs".
- Acknowledge knowledge cutoff explicitly: write to be useful for the next 6-12 months, not pretend to be timeless.
