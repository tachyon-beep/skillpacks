
# Context Window Management Skill

## When to Use This Skill

Use this skill when:
- Processing documents longer than the chosen model's effective context
- Building multi-turn conversational agents
- Implementing RAG systems with retrieved context
- Handling user inputs of unknown length
- Managing long-running conversations (customer support, assistants)
- Optimizing cost and latency for context-heavy applications

**When NOT to use:** Short, fixed-length inputs that obviously fit (tweet classification, single-line form filling).

## Core Principle

**Context is finite, and "long context" is not a free pass.**

Modern frontier-general models default to 128k-200k context. Some now claim 1M+. But two things remain true:

1. **Hard limits still bite.** Going over the model's max-tokens limit is a hard error, not a soft degradation.
2. **Long-context recall is imperfect.** Even at 1M tokens, retrieval accuracy degrades with depth. RULER and similar benchmarks consistently show that effective context is well below claimed context. See "lost in the middle" and the RULER discussion below.

Token counting, budgeting, and active context management are still mandatory. **The right defaults have shifted from "chunk aggressively into 4k" to "consider whether long context, prompt caching, or chunking is the best move for this workload."**

## Capability Tiers and Context (Vocabulary)

This pack uses capability tiers; never hardcode model IDs.

| Tier | Typical context | Notes |
|------|-----------------|-------|
| **frontier-reasoning** | 128k-200k+ | Some variants offer 1M+ via beta or special endpoints. Reasoning models *spend output tokens* on hidden thinking — your output budget needs to be large. |
| **frontier-general** | 128k-200k (most), 1M+ on select models | Today's default for production. Both Anthropic and Google offer 1M-context tiers; check provider docs for the current ID and any beta-flag requirements. |
| **fast-cheap** | 32k-200k | Smaller frontier-class models often inherit the same long context. |
| **on-device** | 8k-128k | Open-weights models vary widely; small/older models still cap at 8-32k. |

**Always check the provider's model card for the current limit.** Treat the number you read in code as a runtime config value, not a constant.

## Context-Length Tiers (Workload Framework)

| Workload | Typical context need | Recommendation |
|----------|---------------------|----------------|
| **Short** | 8k-32k | Most chat, classification, extraction. Any tier works. Smallest, cheapest model that meets quality wins. |
| **Standard** | 32k-200k | The default for "feed me a moderate document and ask questions." Frontier-general handles natively; no chunking needed below ~150k. |
| **Long** | 200k-1M+ | Whole-codebase comprehension, large-document Q&A, long agent traces. Use a 1M-context model **only if** you've measured that recall is acceptable for your workload. Otherwise, chunk + retrieve. |
| **Extreme** | >1M | Almost always wrong. Chunking + RAG + summary memory is more reliable than relying on raw context recall at extreme lengths. |

### When Long Context Is the Right Move

- The task genuinely needs *cross-document* reasoning (compare across many sources, find relationships).
- Document is dense and chunking breaks meaning (legal filings, contracts, code with deep dependencies).
- You can prompt-cache the long prefix so repeated queries are cheap.

### When Chunking + Retrieval Is Still Right

- You have many documents, only a few relevant per query.
- You can build good embeddings + reranking.
- You need provenance/citations tied to chunks.
- Costs at long-context scale are prohibitive (input tokens add up fast even with caching).

## Context Management Framework

```
┌──────────────────────────────────────────────────┐
│          1. Count Tokens                         │
│  tiktoken / Anthropic count_tokens / SentencePiece│
└────────────┬─────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────┐
│          2. Check Against Provider Limits        │
│  Look up current limit; treat as config          │
└────────────┬─────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────┐
│          3. Token Budget Allocation              │
│  System + Tools + Context + Query + Output       │
└────────────┬─────────────────────────────────────┘
             │
        ┌────┴────┐
        │ Fits?   │
        └────┬────┘
      ┌──────┴──────┐
      │ Yes         │ No
      ▼             ▼
 ┌─────────┐   ┌─────────────────────┐
 │ Proceed │   │ Strategy menu:      │
 │ + cache │   │ • Chunk + retrieve  │
 │ static  │   │ • Compress / sumzn │
 │ prefix  │   │ • Sub-agent isolat'n│
 └─────────┘   │ • Long-context tier │
               │ • Hybrid            │
               └─────────┬───────────┘
                         │
                         ▼
               ┌──────────────────┐
               │ Apply & Validate │
               └──────────────────┘
```

## Part 1: Token Counting

LLMs tokenize text (not characters or words). Token counts vary by:

- Language (English ~4 chars/token, Chinese ~2 chars/token)
- Content (code ~3 chars/token, prose ~4.5 chars/token)
- Tokenizer (cl100k vs o200k vs SentencePiece vs BPE variants)

**Character / word counts are unreliable estimates.**

### tiktoken (OpenAI Tokenizers)

```bash
pip install tiktoken
```

```python
import tiktoken

def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    """Count tokens for OpenAI/Azure-OpenAI models.

    o200k_base — current GPT-4o family and successors.
    cl100k_base — GPT-4 / GPT-3.5 era.
    Fall back to o200k for unknown modern models.
    """
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception:
        enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))
```

For chat-message overhead (role tokens, separators), the exact accounting varies by model — use the SDK's `usage` field after a call to ground-truth your local counter, then cache the offset.

### Anthropic and Other Providers

Anthropic exposes `client.messages.count_tokens(...)` for accurate Claude tokenization (different from cl100k/o200k). Google AI SDK exposes `count_tokens` similarly. For open-weights models, use the model's own tokenizer (`AutoTokenizer.from_pretrained(...)` from `transformers`).

**Don't use the wrong tokenizer.** Counting Claude prompts with tiktoken under-counts by 5-15% on average — enough to cause hard 400 errors near the boundary.

### Quick Approximation (Dashboards Only)

```python
def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars/token for English prose. NOT for API calls."""
    return len(text) // 4
```

## Part 2: Provider Limits and Budgeting

### Don't Hardcode Limits

Provider context limits change. Even between dot-versions of the same model family, max-tokens often shifts. Treat the limit as runtime config:

```python
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    """Resolved at startup from config/env, not baked into code."""
    model_id: str
    tier: str                    # frontier-reasoning | frontier-general | fast-cheap | on-device
    max_context_tokens: int      # full window (input + output)
    max_output_tokens: int       # provider-imposed cap on a single response

def load_model_config(tier: str) -> ModelConfig:
    """Look up model id and limits for a tier from config/env.

    Always cross-check the limit against the provider's current docs.
    """
    return ModelConfig(
        model_id=os.environ[f"MODEL_{tier.upper().replace('-', '_')}_ID"],
        tier=tier,
        max_context_tokens=int(os.environ[f"MODEL_{tier.upper().replace('-', '_')}_CTX"]),
        max_output_tokens=int(os.environ[f"MODEL_{tier.upper().replace('-', '_')}_OUT"]),
    )
```

### Token Budget Allocation

```python
@dataclass
class TokenBudget:
    total: int
    system: int
    tools: int
    history: int
    retrieved_context: int
    query: int
    reserved_output: int
    safety_margin: int

    def overflow(self) -> int:
        consumed = (
            self.system + self.tools + self.history + self.retrieved_context
            + self.query + self.reserved_output + self.safety_margin
        )
        return max(0, consumed - self.total)

    def context_room(self) -> int:
        return (
            self.total - self.system - self.tools - self.history - self.query
            - self.reserved_output - self.safety_margin
        )

def build_budget(cfg: ModelConfig, system_tokens: int, tool_tokens: int,
                 history_tokens: int, query_tokens: int,
                 reserved_output: int, safety_margin: int = 200) -> TokenBudget:
    return TokenBudget(
        total=cfg.max_context_tokens,
        system=system_tokens,
        tools=tool_tokens,
        history=history_tokens,
        retrieved_context=0,  # caller fills this from .context_room()
        query=query_tokens,
        reserved_output=min(reserved_output, cfg.max_output_tokens),
        safety_margin=safety_margin,
    )
```

For reasoning models, set `reserved_output` generously — extended thinking can consume thousands of tokens before the visible answer starts.

## Part 3: Chunking Strategies

When the workload genuinely needs more than the model can hold (or recall reliably), chunk and retrieve.

### Fixed-Size Chunking

```python
import tiktoken

def chunk_by_tokens(text: str, chunk_size: int = 1000, overlap: int = 200,
                    encoding_name: str = "o200k_base") -> list[str]:
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks
```

Simple, predictable, but oblivious to structure. Acceptable for homogeneous prose; bad for code, tables, conversations.

### Semantic Chunking

Split at structure boundaries first (headings, paragraphs, sentences), falling back to character split only when a unit is itself too large. LangChain's `RecursiveCharacterTextSplitter` is the canonical implementation; equivalents exist in LlamaIndex and Haystack.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_semantically(text: str, chunk_size: int = 1000, overlap: int = 100,
                       length_fn=None) -> list[str]:
    """Split at semantic boundaries with token-aware sizing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=length_fn or (lambda s: len(s) // 4),
    )
    return splitter.split_text(text)
```

For **code**, use a syntax-aware splitter (tree-sitter-based) so chunks respect function and class boundaries.

### Hierarchical Summarization (Map-Reduce)

For whole-document summarization:

1. Chunk → summarize each chunk (map).
2. Concatenate summaries → summarize the concatenation (reduce).
3. Iterate if the reduce step still overflows.

This works, but **prefer cached long-context inference** when the workload is "ask many questions about one big document." Map-reduce loses information at every reduce step; long context with prompt caching preserves the document and is cheaper after cache warm-up.

## Part 4: Lost-in-the-Middle and the RULER Reality Check

Liu et al. (2023), "Lost in the Middle: How Language Models Use Long Contexts" ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172)), showed that even strong models retrieve most reliably from context start and end, with a noticeable accuracy dip in the middle.

**This has not been fully solved by 2026.** Subsequent work confirms the pattern persists at long lengths:

- **RULER** (Hsieh et al., 2024) — synthetic benchmark with multi-needle retrieval, multi-hop tracing, and aggregation tasks across configurable context lengths. Across 17 long-context models evaluated, almost all show large performance drops as context grows; only ~half maintain acceptable performance at 32k despite advertising much longer windows. [arXiv:2404.06654](https://arxiv.org/abs/2404.06654) / [github.com/NVIDIA/RULER](https://github.com/NVIDIA/RULER).
- **Needle-in-a-Haystack variants** — gkamradt's original NIAH ([github.com/gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)) is now considered a *floor* for long-context capability, not a ceiling. Models that ace single-needle NIAH frequently fail on multi-needle, multi-hop, or aggregation variants at the same length.

**Implication:** A model claiming 1M-token context does not guarantee that information at depth 800k is reliably accessible. Always:

1. Run a recall sanity check on your actual workload before committing to a long-context-only design.
2. Place critical information at the top or bottom of the context, not buried.
3. Use RAG with explicit retrieval whenever the answer depends on finding specific facts in a long corpus.

## Part 5: Intelligent Truncation

When chunking is wrong (single-pass QA, conversation tail), truncate strategically.

### Truncate Middle (Preserve Intro + Conclusion)

```python
def truncate_middle(text: str, max_tokens: int, encoding_name: str = "o200k_base") -> str:
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    keep_start = int(max_tokens * 0.4)
    keep_end = int(max_tokens * 0.4)
    marker = enc.encode("\n\n[... middle truncated ...]\n\n")
    return enc.decode(tokens[:keep_start] + marker + tokens[-keep_end:])
```

Works well for documents with strong intro/conclusion structure (papers, RFCs, executive summaries).

### Truncate Beginning (Recent-Wins)

```python
def truncate_from_start(text: str, max_tokens: int, encoding_name: str = "o200k_base") -> str:
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    return text if len(tokens) <= max_tokens else enc.decode(tokens[-max_tokens:])
```

Right answer for chat logs and event streams where recency dominates relevance.

### Extractive Truncation (Relevance-Wins)

Score sentences by similarity to the query (TF-IDF, BM25, or embedding cosine), keep top-k within budget, restore original order before sending. Works well as a cheap pre-filter ahead of an LLM call.

## Part 6: Conversation Context Management

### Sliding Window

Keep the last N user/assistant pairs plus the system message.

```python
class SlidingWindowChat:
    def __init__(self, model_id: str, system: str, max_turns: int = 10):
        self.model_id = model_id
        self.system = {"role": "system", "content": system}
        self.max_turns = max_turns
        self.messages: list[dict] = [self.system]

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        # Keep system + last (max_turns * 2) turns
        keep_n = self.max_turns * 2
        if len(self.messages) > keep_n + 1:
            self.messages = [self.system] + self.messages[-keep_n:]

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
```

Predictable cost, drops old context.

### Token-Based Truncation

Same idea but with a token budget instead of a turn count — prevents long messages from exploding the window.

### Summarization + Sliding Window (Hybrid)

When the conversation crosses a threshold, summarize old turns into a compact "memory" message and keep the most recent turns verbatim. Cross-ref `agentic-patterns-and-mcp.md` for sub-agent context isolation, which extends this idea: hand off old context to a sub-agent that returns a structured summary, then continue in the parent with a fresh window.

## Part 7: RAG Context Budgeting

```python
def select_chunks_within_budget(
    candidates: list[tuple[dict, float]],   # (chunk, relevance_score), pre-sorted desc
    budget_tokens: int,
    count_fn,
) -> tuple[list[dict], int]:
    selected: list[dict] = []
    used = 0
    for chunk, _score in candidates:
        n = count_fn(chunk["content"])
        if used + n > budget_tokens:
            continue   # Try smaller chunks rather than stop at first overflow
        selected.append(chunk)
        used += n
    return selected, used
```

Prefer **over-retrieve + rerank + budget-fit** over "top-k-by-similarity." The reranker (cross-encoder or LLM judge) reorders candidates by genuine relevance; the budget fitter then picks the densest set of useful chunks.

For repeated retrieval against the same corpus, embed cache hits and reranker scores, not raw queries.

## Part 8: Prompt Caching as a First-Class Strategy

Prompt caching reframes the cost calculus for long-context workloads. With static prefixes — system prompts, tool definitions, retrieved corpora, full documents — marked cacheable, **subsequent calls pay roughly an order of magnitude less for the cached tokens** (Anthropic publishes ~10% of input price for cache reads at the time of writing; OpenAI and Google have analogous mechanisms with their own multipliers).

Implications for context strategy:

- **Long context becomes cheaper** when many queries share the same long prefix. A 200k-token document you query 50 times with caching is dramatically less expensive than naively re-sending it each call.
- **Chunking is no longer the obvious cost win** for "many questions, one document" workloads — long-context + caching often beats chunk + retrieve on cost *and* quality.
- **Order matters.** Cacheable static content goes at the start; volatile content (the user's current query) at the end. Any change to the cached prefix invalidates the cache.

This sheet sketches the implication; the full design pattern — breakpoint placement, multi-segment caches, TTL selection, cross-provider differences, cache-key hygiene — is in `context-engineering-and-prompt-caching.md`. Treat that sheet as required reading before designing a long-context system.

## Part 9: Compression Strategies

When you cannot afford the long-context bill *and* chunking is awkward, compress.

- **Summarization** — replace old turns or full documents with model-generated summaries. Quality varies; use for memory of "what happened" rather than "exact facts."
- **Key-value extraction** — extract structured fields from a long source (entities, dates, decisions, action items) and pass the structured form forward. More faithful than free-form summary for facts.
- **Hierarchical summary** — chunk → per-chunk summary → meta-summary, retaining the per-chunk layer for retrieval. Lets you navigate from a top-level view to source detail.
- **Sub-agent context isolation** — spawn a sub-agent with its own fresh context window to handle a bounded subtask, return only its result to the parent. The parent's context never sees the sub-agent's working memory. Cross-ref `agentic-patterns-and-mcp.md`.

Compression is lossy by definition. Validate that downstream tasks still work on the compressed representation before committing.

## Part 10: Cost and Latency Effects (Relative Framing)

Provider prices change. Frame trade-offs in *ratios*, not dollars:

- **Long context is roughly proportional in cost to its token count.** Doubling the context roughly doubles the input bill (modulo caching).
- **Cached input ≈ 10% of normal input price** on Anthropic; comparable order on other providers. Verify current numbers in provider docs.
- **Latency scales with prompt length.** First-token-latency grows approximately with input length on most engines (prefill cost). Cached prefixes dramatically cut prefill time as well as bill.
- **Output tokens are typically 3-5× the price of input tokens.** Be especially careful with reasoning models that consume output budget on hidden thinking.

Cost-optimization heuristics:

1. **Cache** every static prefix. This is the single biggest lever.
2. **Right-size the tier.** Use fast-cheap for routing/extraction even inside a frontier-general-led pipeline.
3. **Cap output**. Reserve only what you need; large `max_tokens` is rarely used but always priced.
4. **Reuse retrieval**. Cache embeddings; cache reranker scores; cache the final selected chunks if the query pattern repeats.

## Part 11: Complete Example — Managed RAG

```python
import os
from dataclasses import dataclass

@dataclass
class RAGResult:
    answer: str
    chunks_used: int
    context_tokens: int
    output_tokens: int
    cache_hit: bool

class ManagedRAGSystem:
    def __init__(self, tier: str = "frontier-general", reserved_output: int = 800):
        self.cfg = load_model_config(tier)
        self.reserved_output = reserved_output

    def query(self, question: str, retrieved_chunks: list[dict],
              system_prompt: str, count_fn) -> RAGResult:
        # 1. Compute budget
        budget = build_budget(
            cfg=self.cfg,
            system_tokens=count_fn(system_prompt),
            tool_tokens=0,
            history_tokens=0,
            query_tokens=count_fn(question),
            reserved_output=self.reserved_output,
        )
        room = budget.context_room()

        # 2. Fit retrieved chunks into the room
        ranked = sorted(retrieved_chunks, key=lambda c: c["score"], reverse=True)
        selected, used = select_chunks_within_budget(
            [(c, c["score"]) for c in ranked], room, count_fn
        )
        context = "\n\n---\n\n".join(c["content"] for c in selected)

        # 3. Call the model with the static prefix marked cacheable
        #    (system + tools + retrieved context = cacheable; query = volatile)
        #    See context-engineering-and-prompt-caching.md for full pattern.
        answer, output_tokens, cache_hit = self._call(system_prompt, context, question)

        return RAGResult(
            answer=answer,
            chunks_used=len(selected),
            context_tokens=used,
            output_tokens=output_tokens,
            cache_hit=cache_hit,
        )

    def _call(self, system: str, context: str, question: str) -> tuple[str, int, bool]:
        # Provider-specific call wired in your client layer.
        raise NotImplementedError
```

## Summary

**Context window management is mandatory; long context is not a bypass.**

1. **Count tokens** with the right tokenizer for the provider; treat counts as ground truth.
2. **Look up limits dynamically**; never hardcode. Capability tiers, not model IDs.
3. **Frame the workload** by context tier: short / standard / long / extreme — pick the strategy that fits.
4. **Budget** every component: system, tools, history, context, query, output, safety margin.
5. **Chunk + retrieve when many questions ↔ many documents**; **long context + cache when many questions ↔ one big document**.
6. **Trust RULER, not NIAH headlines.** Validate recall on your workload before relying on long context.
7. **Truncate strategically** (middle / start / extractive) when overflow is unavoidable.
8. **Manage conversations** with sliding window, token budget, or summarization + recent.
9. **Compress** with summarization, KV extraction, hierarchical summary, or sub-agent isolation.
10. **Cache** every stable prefix. The full pattern is in `context-engineering-and-prompt-caching.md`.

**Cross-references:**
- `context-engineering-and-prompt-caching.md` — prompt caching design depth.
- `agentic-patterns-and-mcp.md` — sub-agent context isolation.
- `reasoning-models.md` — output-token planning for thinking models.
- `llm-inference-optimization.md` — serving stack and cost-latency-quality trade-offs.
- `yzmir-ml-production` — RAG ops, embedding pipelines, retrieval evaluation.

---

*Model lineup current as of 2026-05; revisit quarterly.*
