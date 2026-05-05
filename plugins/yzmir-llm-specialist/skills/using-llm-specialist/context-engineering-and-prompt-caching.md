
# Context Engineering and Prompt Caching

## Context

You're building an LLM application where the prompt has grown past a few hundred tokens — system instructions, examples, retrieved documents, conversation history, tool definitions, schemas. Common mistakes:
- **Treating prompt caching as a "nice optimization later"** instead of designing for it from day one
- **Cache-busting your own prefix** — putting timestamps, request IDs, or UUIDs at the top of the prompt
- **Reordering volatile and stable content** so the cache breaks on every request
- **Assuming the cache survives** across different system prompts, models, or auth scopes
- **Stuffing context "to be safe"** — paying linear input cost on tokens the model never uses
- **Treating context as a passive container** instead of a designed, versioned, monitored asset

**This skill provides patterns for prompt caching across providers, cache-aware prompt structure, the discipline of context engineering, and the failure modes that turn cached prompts into expensive footguns.**

This sheet uses the **capability-tier vocabulary** from [reasoning-models.md](reasoning-models.md). Read that sheet first if you haven't.


## Why Caching Matters

LLM input tokens cost money and time. A prompt with 50 KB of system instructions, examples, and retrieved docs costs roughly the same per call as a 100-token prompt of unique user input — *unless* you cache the stable prefix.

Three reasons caching is first-class strategy, not an afterthought:

1. **Token economics.** Cached read tokens are 5–10x cheaper than uncached tokens (provider-dependent). On a high-traffic agent that re-uses a 30 KB system prompt, this is the single largest cost lever.
2. **Latency.** Cached prefixes skip prefill compute. Time-to-first-token can drop by 50–80% on cache hit ([OpenAI prompt caching announcement, 2024-10](https://openai.com/index/api-prompt-caching/)).
3. **Reliability.** Cached content has been processed before; some attention/state is preallocated. The model behaves more predictably on repeated context.

**The design implication:** the *order* and *stability* of your prompt determines cache hit rate, which determines your cost and latency. Prompt structure is no longer a stylistic choice — it's a performance concern. Cache hit rate becomes a metric you monitor.


## Provider Caching Landscape

Three approaches, three contracts. Get the details right.

### Anthropic — explicit cache_control breakpoints

You mark cache breakpoints explicitly with `cache_control: {"type": "ephemeral"}` on a content block. Up to four breakpoints per request. Tokens before each breakpoint are cached together; the model checks for a cache hit on the longest matching prefix.

- **Default TTL:** 5 minutes after the most recent use (sliding).
- **Extended TTL:** 1 hour, opted in via `cache_control: {"type": "ephemeral", "ttl": "1h"}`. As of 2026-05, no beta header is required for either TTL value — set it directly.
- **Pricing:** 5-minute cache write tokens cost 1.25× base input; 1-hour cache write tokens cost 2× base input; cache *read* tokens cost 0.1× base input.
- **Workspace isolation:** as of 2026-02-05, caches are isolated per workspace within an organization (previously org-level).

Source: [Anthropic prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

### OpenAI — automatic prefix caching

No explicit markers. The platform automatically caches prefixes ≥1024 tokens; on a subsequent request that shares that prefix, the cached portion is discounted (typically 50% off input tokens, model-dependent). The minimum 1024-token threshold is documented in the [OpenAI prompt-caching guide](https://platform.openai.com/docs/guides/prompt-caching) and [OpenAI prompt-caching announcement](https://openai.com/index/api-prompt-caching/).

- **Trigger:** ≥1024 tokens of shared prefix.
- **Discount:** typically 50% on input tokens for cached portion (model-dependent — verify in current pricing page).
- **Caller-side observability:** response includes `usage.prompt_tokens_details.cached_tokens` so you can monitor hit rate.
- **No TTL guarantees** beyond "recent prompts get cached"; treat it as best-effort.

### Google Gemini — implicit + explicit caching

Two modes ([Gemini context caching docs](https://ai.google.dev/gemini-api/docs/caching)):

- **Implicit caching** (default for Gemini 2.5 and later): automatic, opt-in only via consenting to the discount. Minimum sizes are model-dependent — for Gemini 2.5 Flash, ~1024 tokens; for 2.5 Pro, ~2048 tokens. Cost: discount applied on hits with no storage charge.
- **Explicit caching:** you create a cache via API, get a cache name, and reference it in subsequent requests. Default TTL is 60 minutes; configurable via `ttl` or `expire_time`. Cost: input-token discount (90% on Gemini 2.5+, 75% on 2.0) plus a storage charge proportional to cache size and TTL.

Use implicit when the workload pattern is unpredictable; use explicit when you have a known stable context (e.g., a corpus, a long doc) referenced across many queries.

### Quick comparison

| Provider | Trigger | TTL | Pricing on hit | Pricing on write |
|---|---|---|---|---|
| Anthropic | Explicit `cache_control` breakpoint | 5 min default; 1 hr extended | 0.1× input | 1.25× (5min) / 2× (1hr) |
| OpenAI | Automatic, ≥1024 token prefix | Best-effort, undocumented | ~0.5× input (model-dependent) | None (automatic) |
| Gemini implicit | Automatic, model-dependent min | Best-effort | Discounted | None |
| Gemini explicit | Explicit `cache.create()` | Default 60 min, configurable | 0.1× input (2.5+) | Storage charge per token-hour |

> **Verify before deploying.** All three providers have changed pricing and minimums multiple times since 2024. Treat this table as a starting reference; check current docs.


## Cache-Aware Prompt Structure

The single most important rule: **stable prefix first, volatile suffix last.**

```
┌──────────────────────────────────┐
│ 1. System instructions (stable)  │ ◀── cache here
├──────────────────────────────────┤
│ 2. Tool definitions (stable)     │ ◀── cache here (Anthropic: same breakpoint)
├──────────────────────────────────┤
│ 3. Long-lived examples (stable)  │ ◀── cache here
├──────────────────────────────────┤
│ 4. Retrieved documents (mostly   │ ◀── cache here if doc set is stable per session
│    stable per session)           │
├──────────────────────────────────┤
│ 5. Conversation history (grows   │     less cacheable — see below
│    monotonically)                │
├──────────────────────────────────┤
│ 6. Current user query (volatile) │     never cacheable
└──────────────────────────────────┘
```

### Conversation history as an append-only log

Multi-turn conversations have a natural caching shape: the prefix grows monotonically, never reorders. Each turn extends the prefix by one user/assistant pair; the previous prefix stays cacheable. Anthropic's recommended pattern is to put a `cache_control` breakpoint on the *last* assistant message of the previous turn so the entire history-up-to-now is cached on the next user turn.

### Common breakage patterns

- **Timestamp at the top.** "Today is 2026-05-14T10:32:01Z." breaks the cache on every request. Move timestamps to the *user* message or strip them entirely.
- **Request ID at the top.** Same problem. If you need a request ID for logging, never put it in the prompt.
- **Dynamic ordering of "context" sections.** Sorting examples by relevance per query, reordering tool definitions per call. Pick a stable order and live with it.
- **Per-user customizations injected early.** "User name: Jane" at the top of a system prompt that's otherwise identical across users. Move user-specific content out of the cached prefix or accept the loss of cache hits across users.
- **A/B test variant in the system prompt.** If you're A/B-testing two system prompts, both fully break each other's cache. Acceptable cost during the experiment; not acceptable as a permanent state.
- **Tool definitions reordered.** Some frameworks sort tools alphabetically each call; some by registration order. Pick one and pin it.

### Cache-hit-rate as a metric

Add `cached_tokens` (or its equivalent) to your dashboards. A healthy chat agent with a stable system prompt should have cache hit rates well above 80% on input tokens after the first turn. If you see hit rate collapse, audit the prompt for new volatile content at the top.


## When Caching Beats RAG

The frontier-general and frontier-reasoning tiers now offer 1M+ token contexts. This re-opens a design question that RAG had closed: should you put the whole document set in context and cache it, or retrieve dynamically?

### Caching wins when:

- **Document set fits in context** (with safety margin — say, the corpus is 200K tokens and the model has 1M context).
- **Queries are diverse over the same corpus** — a high cache hit rate amortizes the one-time cache write across all queries.
- **Latency matters more than cost-per-query** — cache reads are faster than retrieval + generation roundtrips.
- **The corpus is stable** — you're not constantly invalidating the cache with new documents.

### RAG wins when:

- **Corpus exceeds context** — terabytes of docs, millions of records.
- **Queries are narrowly targeted** — each query needs a small, specific subset; loading everything wastes attention.
- **Documents update frequently** — cache invalidation churn destroys hit rate.
- **You need source citations** with strong provenance — RAG returns explicit document references.

### Hybrid: cache the rerankers and prompts, not the corpus

A common-sense middle ground: keep RAG for retrieval over the large corpus, but cache the system prompt, tool/schema definitions, and the rerank/judge prompts. Retrieval results vary per query (volatile suffix); everything else is stable prefix. Cross-ref [rag-architecture-patterns.md](rag-architecture-patterns.md) for retrieval discipline.

### The 1M-token tier

For workloads where a single "session" has a stable, large context (a 500-page legal brief; a whole codebase repo), the 1M-context tier with explicit caching (Anthropic 1-hour cache, Gemini explicit cache) is often a clean replacement for what would have been a RAG pipeline. The decision rule: **how often does the context change relative to the queries?** Many queries per stable context = cache wins. Few queries per dynamic context = RAG wins.

Cross-ref [context-window-management.md](context-window-management.md) for the long-context tier and the "lost in the middle" trade-offs of dumping everything into context.


## Context Engineering as a Discipline

Context engineering is the practice of treating context as a designed, versioned, monitored asset — not a passive container.

### Sub-agents for context isolation

When a task requires expanding the working context (e.g., "search the codebase for all callers of function X" returns 50 files of evidence), don't pollute the parent agent's context with all 50 files. Spawn a sub-agent with a clean context, let it do the search, and have it return *only* the summarized result. The parent context stays small. This is the same pattern documented under sub-agents in [agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md), used here as a context-engineering tactic.

### File-based memory: read/write tools

Long-running agents shouldn't try to keep everything in the context window. Give the agent file-read and file-write tools (or equivalent: vector store, key-value store) and instruct it to write notes/decisions to files and read them back when needed. The context window holds the *current* working set; the file system holds the *cumulative* memory.

This is the "blackboard" pattern from multi-agent literature, applied to a single long-running agent. It maps cleanly onto how human engineers work: working memory plus notes plus reference material.

### Todo lists as compression

A todo list — written by the agent to a file at the start of a task and updated as steps complete — is a high-density compression of "what I'm doing and why." Reading the todo list back into the prompt at the start of each iteration costs ~50–200 tokens but conveys what would otherwise take 5,000 tokens of context replay.

This is the design that powers Claude Code's plan/todo feature and similar agent harnesses ([Anthropic engineering: How we built our multi-agent research system, 2025](https://www.anthropic.com/engineering/built-multi-agent-research-system)).

### Context compaction strategies for long-running agents

When a conversation grows beyond a comfortable size (say, 60–80% of the model's context window), you need a compaction strategy. Three patterns:

1. **Summarize-and-truncate.** At a threshold, ask the model (or a cheaper summarizer) to summarize the older portion and replace it. Loses fidelity; preserves the running shape.
2. **Hierarchical summary.** Keep the most recent N turns verbatim; summarize the next N–2N turns; summarize-of-summaries the rest. Pyramidal compaction.
3. **External memory + selective recall.** Write turns to file storage; in the active prompt, keep only what's needed for the current step. The agent reads back specific files when relevant.

Cross-ref [context-window-management.md](context-window-management.md) for token-counting, budgeting, and chunk-selection mechanics.

### Context as code

Treat your context like source code:

- **Version it.** When the system prompt changes, bump a version. Pin tool definitions. Track which prompt version produced which output for incident triage.
- **Test it.** Eval-pin specific prompts; when you change a prompt, run the eval suite. Cross-ref [llm-evaluation-metrics.md](llm-evaluation-metrics.md).
- **Monitor it.** Cache hit rate, prompt-token count per request, hit-rate-by-prompt-version. A regression on cache hit rate is as worth a page as a regression on accuracy.
- **Review it.** Code-review prompt changes the way you review code changes. Lint for "stable-prefix-first" violations.


## Anti-Patterns

### Anti-pattern 1: Hash-busting via UUIDs early in the prompt

**Wrong:**
```
[Request ID: 7f3a-9d12-...]  ← cache breaks on every request
[System instructions: ...]
[Conversation: ...]
```

**Right:** Move request IDs out of the prompt entirely (log them alongside the request). If you must include something per-request at the top, accept the cache loss and document the cost.

**Principle:** Anything unique per request that lives in the cacheable prefix kills cache hit rate.

### Anti-pattern 2: Assuming cache survives across system-prompt variations

**Wrong:** Two services share a similar system prompt with one differing line; assume both benefit from a single cache.

**Right:** Caches are keyed on exact prefix bytes (and on model, account, sometimes workspace). Different system prompts = different caches. Either align them exactly or treat them as separate cache targets.

**Principle:** Caches don't fuzzy-match; near-identical isn't identical.

### Anti-pattern 3: Over-caching ephemeral data

**Wrong:** Wrapping per-user, per-request RAG results with `cache_control` breakpoints "to be efficient."

**Right:** Cache only what is actually stable across calls. Marking volatile content for caching wastes cache-write tokens with no read-side payoff (and can churn the cache in pathological cases).

**Principle:** Caching has a write cost. Caching things that won't be read again is pure cost.

### Anti-pattern 4: Stuffing context "to be safe"

**Wrong:** "Just include the entire 200K-token codebase in case the model needs it."

**Right:** Include what the model needs for *this* task; use sub-agents or RAG for selective recall. Cache the parts that *are* needed across calls.

**Principle:** Linear input cost; quadratic-ish attention spread over irrelevant content; "lost-in-the-middle" degradation. More context isn't free.

### Anti-pattern 5: Reordering tools or examples per request

**Wrong:** Sorting tool definitions by recent-use frequency or examples by similarity to the current query, every request.

**Right:** Pin a stable order. The "smarter" ordering's cache penalty is almost always larger than its retrieval benefit.

**Principle:** Cache stability beats marginal relevance ordering.

### Anti-pattern 6: Treating prompt caching as a late optimization

**Wrong:** "We'll add caching when costs become a problem."

**Right:** Design for caching from the first prompt: stable prefix, volatile suffix, no early timestamps. Caching is essentially free to design *in*, expensive to retrofit.

**Principle:** Cache-aware structure costs nothing up front and saves substantial cost forever.

### Anti-pattern 7: No cache hit rate monitoring

**Wrong:** Cache is on; "we'll trust it."

**Right:** Monitor `cached_tokens` (OpenAI), Anthropic `usage.cache_read_input_tokens` / `cache_creation_input_tokens`, Gemini cache metrics. Set alerts on regressions. Investigate any prompt change that drops hit rate.

**Principle:** What isn't measured regresses. Cache hit rate is a first-class production metric.

### Anti-pattern 8: Hardcoded model IDs and assumptions about cache portability

**Wrong:** Code assumes a cached prefix on `model-A` will hit on `model-B`.

**Right:** Caches are keyed on model. Switching models invalidates caches. Plan for warm-up cost on model migrations. Reference models by capability tier (cross-ref [reasoning-models.md](reasoning-models.md)) and document the migration cost in your runbook.

**Principle:** Model migrations and cache portability are not free; budget for warm-up.


## Summary

**Caching is first-class strategy.** Token economics, latency, and reliability all improve substantially. Designing for cache hit rate is a structural concern, not a late optimization.

**Provider models differ:**
- Anthropic: explicit `cache_control` breakpoints, 5-min default / 1-hr extended TTL, 0.1× read.
- OpenAI: automatic prefix caching ≥1024 tokens, ~50% discount on hit.
- Gemini: implicit (default in 2.5+) plus explicit `cache.create()` with configurable TTL.

**Cache-aware structure:** stable prefix first (system, tools, examples, stable docs), volatile suffix last (history, current query). Never put timestamps or request IDs in the cached prefix.

**Caching vs. RAG:** caching wins when corpus fits in context and queries are diverse over a stable corpus; RAG wins when corpus exceeds context, queries are narrow, or docs change frequently. Hybrid (RAG retrieval + cached system/rerank prompts) is often the right answer.

**Context engineering as discipline:**
- Sub-agents for context isolation.
- File-based memory for cumulative state.
- Todo lists as compression.
- Compaction strategies (summarize-truncate, hierarchical, external recall) for long-running agents.
- Context as code: version, test, monitor, review.

**Cross-refs:**
- [reasoning-models.md](reasoning-models.md) — capability tiers; thinking-token cost interaction with caching
- [agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md) — sub-agent context isolation; tool-result compaction
- [rag-architecture-patterns.md](rag-architecture-patterns.md) — RAG-vs-cache decision in detail
- [context-window-management.md](context-window-management.md) — token counting, the 1M-token tier, "lost in the middle"
- [llm-evaluation-metrics.md](llm-evaluation-metrics.md) — eval-pinning prompts; monitoring prompt regressions
- [yzmir-ml-production](../../yzmir-ml-production/) — serving stack ops, cache hit rate as production metric

---

*Model lineup and provider feature set current as of 2026-05; revisit quarterly. Verify provider docs for current model IDs and pricing.*
