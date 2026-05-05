
# RAG Architecture Patterns

## Context

You're building a RAG (Retrieval-Augmented Generation) system to give LLMs access to external knowledge. Common mistakes:
- **No chunking strategy** (full docs → overflow, poor precision)
- **Naive dense-only retrieval** (cosine similarity alone → misses exact matches, IDs, rare terms)
- **No re-ranking** (irrelevant results prioritized)
- **No evaluation** (can't measure or optimize quality)
- **Context overflow** (too many chunks → cost, latency, "lost in the middle")
- **Treating every problem as a RAG problem** (when long-context + caching, or fine-tuning, or just a better prompt would dominate)

**This skill provides effective RAG architecture: chunking, contextual retrieval, hybrid search with RRF, cross-encoder and late-interaction re-ranking, agentic patterns, evaluation, and the long-context-vs-RAG trade-off.**


## What is RAG?

**RAG = Retrieval-Augmented Generation**

**Problem:** LLMs have knowledge cutoffs and cannot access private, post-training, or rapidly-changing data.

**Solution:** Retrieve relevant evidence, inject into the prompt, generate the answer with citations.

```python
# Without RAG:
answer = llm("What is our return policy?")
# LLM: "I don't have access to your specific return policy."

# With RAG:
relevant_docs = retrieval_system.search("return policy")
context = "\n".join(relevant_docs)
prompt = f"Context:\n{context}\n\nQuestion: What is our return policy?\nAnswer:"
answer = llm(prompt)
```


## Should You Use RAG? (Decision Tree)

Before designing a pipeline, decide whether RAG is the right tool. The answer changed materially once frontier models gained 200k–1M-token context windows and prompt caching.

```
1. Is the corpus small (< ~200k tokens) AND mostly static?
   YES → Skip RAG. Stuff the corpus into context with prompt caching.
         (See context-engineering-and-prompt-caching.md.)
   NO  → Continue.

2. Is the corpus large (millions+ tokens), updated frequently,
   or do you need per-query citations / audit trails?
   YES → RAG (likely with hybrid + reranker).
   NO  → Continue.

3. Is the task pure reasoning, style, or general knowledge?
   YES → RAG won't help. Use a stronger model, fine-tune,
         or improve prompts.
   NO  → Continue.

4. Do queries hit a small recurring slice of a large corpus?
   YES → Long-context with prompt caching for the hot slice +
         RAG for the rest is often best.
   NO  → Pure RAG.
```

**When long context beats RAG:**
- Corpus fits in the model's window with caching.
- Queries are diverse and unpredictable (any chunk could matter).
- You need cross-document synthesis the retriever would shred.
- You don't need explicit citations.

**When RAG still wins:**
- Corpus is huge (10M–10B tokens) or constantly changing.
- You must cite specific sources for every claim.
- You need access control / per-tenant filtering at retrieval time.
- Cost matters: RAG sends ~5k tokens of context, long-context sends ~500k.
- Auditability is a hard requirement (regulated industries).

Cross-references: `context-engineering-and-prompt-caching.md` for prompt-cache mechanics, `context-window-management.md` for stuffing strategies, and `agentic-patterns-and-mcp.md` for tool-call retrieval.


## RAG Pipeline Anatomy

```
User Query
    │
    ▼
1. Query Processing      (rewrite, expand, decompose, HyDE)
    │
    ▼
2. Hybrid Retrieval      (dense + sparse, fused with RRF)
    │
    ▼
3. Re-ranking            (cross-encoder or late-interaction)
    │
    ▼
4. Context Selection     (token-budget-aware, lost-in-middle aware)
    │
    ▼
5. Prompt Construction   (citations, metadata, system rules)
    │
    ▼
6. LLM Generation        (with optional grounded-answer verification)
    │
    ▼
Answer (+ citations + provenance)
```

The five evergreen invariants — chunk, embed, retrieve, rerank, generate — have not changed. Almost everything else has, especially in the last 18 months.


## Component 1: Document Processing & Chunking

### Why chunk?

Embedding models and LLMs have token limits, retrieval precision is higher per-chunk than per-document, and citations need granularity. Chunks of 300–800 tokens with 10–20% overlap remain the practical default.

### Strategies

**1. Recursive character splitting** (LangChain default — still the right starting point):

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],  # try semantic first
)
chunks = splitter.split_text(document)
```

**2. Semantic / structure-aware chunking** — split on Markdown headers, code-block boundaries, sentence embeddings (LlamaIndex `SemanticSplitterNodeParser`, LangChain `MarkdownHeaderTextSplitter`).

**3. Layout-aware chunking for PDFs** — use a parser that preserves tables, figures, and page boundaries (Unstructured, LlamaParse, Docling, Marker). Garbage-in still dominates retrieval failures; bad PDF extraction is the #1 cause.

### Always preserve metadata

```python
chunk_record = {
    "text": chunk,
    "metadata": {
        "source": "policy.pdf",
        "page": 42,
        "section": "Returns",
        "doc_id": "policy-v3.2",
        "chunk_id": "policy-v3.2#42-3",
    },
}
```

You will need this for citations, filtering, and incremental re-indexing.


## Component 2: Contextual Retrieval (Anthropic, 2024)

The single biggest free win in retrieval quality since this sheet was first written: **prepend a short LLM-generated context blurb to each chunk before you embed it and before you index it for BM25.** Anthropic reported a 49% reduction in top-20 retrieval failures (5.7% → 2.9%) from contextual embeddings + contextual BM25, and 67% when combined with reranking.

The technique is simple. For each chunk, ask a cheap, prompt-cached LLM call:

> "Here is the whole document. Here is one chunk from it. In one or two sentences, situate this chunk within the document so it is interpretable on its own."

Prepend that 50–100-token blurb to the chunk text, then embed and BM25-index the concatenation. The original chunk is what you return for generation; the contextualized version is what you index.

```python
# Pseudocode — exact API surface depends on your provider SDK.
def contextualize_chunk(full_doc: str, chunk: str, llm) -> str:
    prompt = (
        "<document>\n" + full_doc + "\n</document>\n"
        "Here is a chunk we want to situate within the whole document:\n"
        "<chunk>\n" + chunk + "\n</chunk>\n"
        "Give a short context (1-2 sentences) that situates this chunk "
        "within the document for retrieval. Answer only with the context."
    )
    return llm.complete(prompt, max_tokens=120)

# Index the contextualized version; serve the original chunk to the LLM.
indexed_text = contextualize_chunk(doc, chunk, llm) + "\n\n" + chunk
```

Use **prompt caching** on the document portion — every chunk in a document reuses the same `<document>` prefix. Without caching this is too expensive; with caching it's roughly free.

Reference: Anthropic, "Introducing Contextual Retrieval," September 2024 — <https://www.anthropic.com/news/contextual-retrieval>.


## Component 3: Embeddings — Modern Lineup

Hardcoded model IDs date instantly; treat the table as **capability tiers** and check provider docs for the current best name. As of 2026-05, the practical landscape is:

| Tier | Representative models | Dimensions | Why it lives here |
|------|----------------------|-----------|--------------------|
| Frontier proprietary | Voyage 3 / 3-large; Cohere Embed v3 / v4 (multilingual); OpenAI text-embedding-3-large | 1024–3072, MRL-truncatable on Voyage | Top of MTEB, strong on out-of-domain text, multilingual |
| Strong open-source | BGE-M3; Jina embeddings v3; Nomic Embed v1.5; mxbai-embed-large | 768–1024 (Jina + Nomic MRL-truncatable) | Self-hosted, multilingual, multi-functional (dense+sparse+colbert in BGE-M3) |
| Lightweight | all-MiniLM-L6-v2; bge-small | 384 | Edge / mobile / cheap baseline |
| Domain-specific | SPECTER2 (academic); CodeBERT/UniXcoder (code); medical/legal variants | varies | Out-perform general models on their domain |

Verify before adopting:
- Voyage: <https://docs.voyageai.com/docs/embeddings>, <https://blog.voyageai.com/2025/01/07/voyage-3-large/>
- Cohere Embed v3: <https://docs.cohere.com/docs/cohere-embed>
- Jina v3: <https://jina.ai/models/jina-embeddings-v3/>, paper <https://arxiv.org/abs/2409.10173>
- Nomic v1.5: <https://huggingface.co/nomic-ai/nomic-embed-text-v1.5>
- BGE-M3: paper <https://arxiv.org/abs/2402.03216>, <https://huggingface.co/BAAI/bge-m3>
- OpenAI text-embedding-3-large: <https://platform.openai.com/docs/guides/embeddings>

**Always benchmark on your own data**. MTEB rankings are a starting point, not an answer; relative ordering on a domain-specific eval set frequently disagrees.

```python
# Direct provider SDK is now preferred over LangChain wrappers for embeddings,
# both for clarity and to avoid integration drift.
import voyageai

vo = voyageai.Client()
embeddings = vo.embed(
    texts=chunks,
    model="voyage-3-large",
    input_type="document",        # use "query" at retrieval time
    output_dimension=1024,        # MRL truncation
).embeddings
```

### Matryoshka embeddings (MRL)

Matryoshka Representation Learning (Kusupati et al., NeurIPS 2022 — <https://arxiv.org/abs/2205.13147>) trains a single embedding so that **truncating it to a smaller dimension still gives a useful representation**. Voyage 3-large, Jina v3, Nomic v1.5, OpenAI text-embedding-3, and Gemini Embedding all expose this.

Practical effect: store full-dimension vectors once, run cheap-and-fast similarity at 256-d for first-stage retrieval, and re-score with full-dimension on the top-100. You get most of the recall at a fraction of the storage and latency.

```python
# Voyage exposes output_dimension directly. Other providers expose a `dimensions`
# parameter (OpenAI text-embedding-3) or accept post-hoc np.array slicing
# (Nomic, Jina) — confirm in current docs.
small = vo.embed(query, model="voyage-3-large", output_dimension=256).embeddings[0]
```

Binary or int8 quantization (Voyage, Cohere, Jina all support this) compounds the savings — Voyage reports binary 512-d outperforming float-3072-d on some benchmarks at 200× less storage. Verify on your data.


## Component 4: Vector Stores

The vector-DB market keeps churning. The decision axes have not:

| Need | Reasonable choices (verify current state) |
|------|------------------------------------------|
| Local dev / < 1M chunks | Chroma, LanceDB, FAISS, DuckDB-VSS |
| Self-hosted production | Qdrant, Weaviate, Milvus, pgvector / pgvector-rs |
| Managed / serverless | Pinecone, Turbopuffer, Vespa Cloud, MongoDB Atlas Vector Search, Cohere/Voyage hosted, Azure AI Search, Vertex AI Vector Search |
| Hybrid (BM25 + vector) first-class | Vespa, Weaviate, Elasticsearch, OpenSearch, Qdrant (sparse vectors), Turbopuffer |
| Multi-vector / late interaction | Vespa, Qdrant (multi-vector), Weaviate (ColBERT-style), LlamaIndex + ColBERT |

```python
# Modern langchain imports (langchain >= 0.1; community split):
from langchain_community.vectorstores import Chroma, FAISS
from langchain_qdrant import QdrantVectorStore
from langchain_postgres import PGVector       # pgvector now lives here
# OLD (broken since langchain 0.1):
# from langchain.vectorstores import Chroma   # do not use
```

For anything serious, prefer the provider SDK directly (Qdrant client, Pinecone client, etc.). LangChain wrappers are useful for prototypes; they often lag the provider's newest features (sparse vectors, multi-vector, payload filtering DSL).


## Component 5: Retrieval Strategies

### Dense retrieval

Cosine similarity on embeddings. Strong on semantic matches, weak on exact strings, IDs, rare terminology, and out-of-vocabulary product codes.

### Sparse retrieval

**BM25** (Robertson 1995) is still the right baseline. It catches the exact-match cases dense retrieval misses, has zero training cost, and is interpretable. Production engines (Elasticsearch, OpenSearch, Lucene-backed Vespa) ship it tuned.

**SPLADE** (Formal et al., SIGIR 2021/2022 — <https://arxiv.org/abs/2107.05720>, v2 <https://arxiv.org/abs/2109.10086>) is the modern sparse-neural option: a BERT MLM head produces sparse term-weighted expansions, indexed in an inverted index like BM25 but with learned semantics. Use SPLADE when you need lexical-style retrieval with neural recall and your engine supports sparse vectors (Qdrant, Vespa, Pinecone sparse-dense).

**BGE-M3** is unique in producing dense, sparse, and ColBERT-style multi-vector outputs from a single model — convenient when you don't want to run three encoders.

### Hybrid retrieval with Reciprocal Rank Fusion

Combining dense and sparse is now table stakes. The fusion method that has held up since 2009 is **RRF (Reciprocal Rank Fusion)** — Cormack, Clarke & Büttcher, SIGIR 2009 — <https://dl.acm.org/doi/10.1145/1571941.1572114>:

```
RRF_score(d) = Σ_i  1 / (k + rank_i(d))      typically k = 60
```

It's ranking-only (no score calibration needed across heterogeneous retrievers), and it consistently beats CombSUM/CombMNZ/Condorcet in the original paper and in practice. Most modern engines (Elastic, Vespa, Weaviate, Qdrant, OpenSearch) ship RRF natively.

```python
# Manual RRF if your store doesn't expose it:
def rrf(rank_lists: list[list[str]], k: int = 60, top_k: int = 20) -> list[str]:
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for r, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + r + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]
```

When you do want weighted score fusion (e.g. you have well-calibrated scores from a learned ranker), normalize per-retriever first; RRF is the safer default.

### Late interaction: ColBERT and ColPali

Single-vector embeddings collapse a chunk to one point. **Late interaction** keeps a vector per token and scores via MaxSim across all query-token / doc-token pairs at retrieval time. ColBERTv2 (Santhanam et al., NAACL 2022 — <https://arxiv.org/abs/2112.01488>) made this practical with residual compression.

**ColPali / ColQwen** (Faysse et al., 2024 — <https://arxiv.org/abs/2407.01449>) extends late interaction to **visual document retrieval**: feed page images directly to a vision-language model (PaliGemma-3B / Qwen2-VL), produce multi-vector page embeddings, and skip OCR entirely. On the ViDoRe benchmark this beats text-extraction pipelines on slide decks, financial PDFs, and any doc where layout carries meaning. Use ColPali / ColQwen when:
- Documents are visually complex (tables, figures, multi-column layouts).
- OCR pipelines are dropping signal.
- You can afford the larger storage footprint of multi-vector indexes.

ColBERT-style indexes are supported in PLAID, Vespa, Qdrant (multi-vector), and the `colbert-ai` / RAGatouille libraries.

### HyDE — Hypothetical Document Embeddings

Gao et al., 2022 — <https://arxiv.org/abs/2212.10496>. Use a cheap LLM to generate a hypothetical answer to the query, then retrieve docs similar to the *answer* rather than the *question*. Helps when queries are short and answers are long. Cheap, no training needed, occasionally a big win.


## Component 6: Re-ranking — Modern Models

Bi-encoders (what your vector store uses) are fast but coarse. **Cross-encoders** read query and document jointly and score them, at the cost of one model call per (query, doc) pair. Standard pipeline: retrieve k=50–200 with hybrid, rerank to k=5–10 with a cross-encoder.

| Reranker | Provider / paper | Notes |
|----------|------------------|-------|
| Cohere Rerank 3.5 | Hosted API — <https://docs.cohere.com/docs/rerank>, <https://cohere.com/blog/rerank-3pt5> | 4096-token context, multilingual, SOTA on BEIR class benchmarks at release |
| Voyage rerank-2 / 2.5 | <https://docs.voyageai.com/docs/reranker>, <https://blog.voyageai.com/2024/09/30/rerank-2/>, <https://blog.voyageai.com/2025/08/11/rerank-2-5/> | 16k context (rerank-2), instruction-following (2.5) |
| Jina reranker v2 | <https://jina.ai/reranker/> | Multilingual, fast |
| BGE-reranker-v2-m3 | <https://huggingface.co/BAAI/bge-reranker-v2-m3> | Open-source, multilingual, strong |
| ms-marco-MiniLM (legacy) | <https://huggingface.co/cross-encoder> | Tiny, fast, weaker than the above; OK as a baseline |

**Default recommendation in 2026:** start with a hosted reranker (Cohere Rerank 3.5 or Voyage rerank-2.5) for quality, drop to BGE-reranker-v2-m3 self-hosted if you need on-prem, fall back to ms-marco-MiniLM only when latency budget is sub-50ms and quality is secondary.

```python
import cohere
co = cohere.Client()

def rerank(query: str, docs: list[str], top_n: int = 5) -> list[tuple[str, float]]:
    res = co.rerank(model="rerank-v3.5", query=query, documents=docs, top_n=top_n)
    return [(docs[r.index], r.relevance_score) for r in res.results]
```


## Component 7: Query Processing

### Query rewriting and decomposition

Use a cheap LLM to rewrite ambiguous queries, decompose multi-hop questions into sub-queries, or strip conversational fluff before retrieval. Decomposition is essentially mandatory for multi-hop QA — the retriever cannot find a single chunk that answers a query whose answer requires joining three.

### Query expansion

Generate paraphrases or related terms with an LLM, retrieve for each, fuse with RRF. Useful when query vocabulary diverges from corpus vocabulary.

### HyDE

See Component 5. Particularly useful for short keyword-style queries against long-form docs.


## Component 8: Agentic and Multi-Hop RAG

Single-shot retrieve-then-generate fails when:
- Initial retrieval is wrong and the model hallucinates from bad context.
- The question requires multi-hop reasoning across documents.
- The corpus genuinely doesn't have the answer and the model should say so.

### Corrective RAG (CRAG)

Yan et al., 2024 — <https://arxiv.org/abs/2401.15884>. A lightweight retrieval evaluator (a 0.77B classifier in the paper) scores retrieved chunks. If confidence is high, use them. If low, fall back to web search. If ambiguous, decompose-recompose. Buys you robustness against bad retrieval at modest cost.

### Self-RAG

Asai et al., ICLR 2024 — <https://arxiv.org/abs/2310.11511>. Train (or prompt) the LLM to emit reflection tokens that decide when to retrieve, what to retrieve, and how to critique retrieved evidence. The critique is per-passage and per-claim, which improves citation accuracy on long-form generation.

### GraphRAG

Edge et al., Microsoft Research, 2024 — <https://arxiv.org/abs/2404.16130>, <https://microsoft.github.io/graphrag/>. Build a knowledge graph from the corpus with an LLM, run hierarchical Leiden community detection, summarize each community bottom-up. At query time, route global ("what are the major themes?") questions through community summaries and local ("what does paragraph 3 say?") questions through standard chunk retrieval. Strong on **sensemaking** queries that vanilla RAG handles poorly.

### Agentic / tool-call RAG

The current production frontier: the LLM is given retrieval as a tool, and decides when, what, and how often to call it within a single turn. This composes naturally with MCP-served retrieval tools, multi-step plans, and self-correction. Cross-ref `agentic-patterns-and-mcp.md` for the protocol-level details and tool-design patterns.

```python
# Sketch — actual tool call surface depends on your harness.
tools = [
    {"name": "search_docs", "description": "Search internal corpus.",
     "input_schema": {"query": "string", "k": "integer"}},
    {"name": "fetch_doc",   "description": "Fetch full doc by id.",
     "input_schema": {"doc_id": "string"}},
]
# The agent loop runs: model → tool_use → tool_result → model → ...
# until the model emits a final answer with citations.
```


## Component 9: Context Selection and the Lost-in-the-Middle Problem

Liu et al., 2023 — <https://arxiv.org/abs/2307.03172> — showed LLMs systematically attend to the start and end of long contexts and miss the middle. Even on modern long-context models the effect persists, just with a higher ceiling. Two consequences:

1. **Don't dump all 50 retrieved chunks** into the prompt. Cap at 5–10 reranked chunks (~3–8k tokens) unless you have evidence more helps.
2. **Order matters.** Place the most relevant chunk first or last; "second-most-relevant first, most-relevant last" is a reasonable heuristic.

```python
def order_for_lost_in_middle(reranked: list[Doc]) -> list[Doc]:
    if len(reranked) <= 2:
        return reranked
    # Put the most relevant at the end (LLMs often weight the recency edge harder).
    head, *middle, tail = reranked
    return [head, *reversed(middle), tail]
```

**Contextual compression** — feeding each chunk through an extractive LLM step that keeps only sentences relevant to the query — pays off when chunks are long and the retriever returns dense matches. LangChain ships `ContextualCompressionRetriever` for this; LlamaIndex has equivalent post-processors.


## Component 10: Prompt Construction

Always:
1. Tell the model to answer only from the context, and to say "I don't know" otherwise.
2. Include source IDs and instruct citation.
3. Include relevant metadata (doc title, section, date) — it helps the model assess source quality and trust.

```python
context_block = "\n\n".join(
    f"[{i+1}] ({c.metadata['source']}, p.{c.metadata['page']})\n{c.text}"
    for i, c in enumerate(retrieved)
)
prompt = (
    "Answer the question using only the context below. "
    "Cite sources as [1], [2], etc. If the answer is not present, say "
    "'I don't have enough information.'\n\n"
    f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"
)
```

For high-stakes deployments, follow generation with a **groundedness check**: a second LLM call (or a fine-tuned NLI model) that verifies each generated sentence is entailed by the cited chunks. RAGAS, TruLens, and DeepEval all ship this.


## Evaluation

### Retrieval metrics

The math hasn't changed: **MRR**, **Recall@k**, **Precision@k**, **NDCG@k**, plus the binary "did the gold chunk appear in the top-20" used by the contextual-retrieval paper.

```python
import numpy as np

def mrr(results: list[list[str]], gold: list[set[str]]) -> float:
    scores = []
    for ranks, relevant in zip(results, gold):
        for i, doc in enumerate(ranks):
            if doc in relevant:
                scores.append(1 / (i + 1))
                break
        else:
            scores.append(0.0)
    return float(np.mean(scores))

def recall_at_k(results, gold, k=5):
    return float(np.mean([
        len(set(r[:k]) & g) / max(len(g), 1)
        for r, g in zip(results, gold)
    ]))
```

**NDCG** uses graded relevance and position discount; use scikit-learn's `ndcg_score` rather than rolling your own.

**Targets** depend on the corpus, but as rough sanity bands: MRR > 0.7, Recall@10 > 0.85, NDCG@10 > 0.7.

### End-to-end metrics

For generation quality, prefer **LLM-as-judge** with a strong frontier model and a rubric, plus token-level F1 / EM where you have gold answers. Modern frameworks bundle the full pipeline:
- **RAGAS** — <https://docs.ragas.io/> — faithfulness, answer relevancy, context precision/recall.
- **TruLens** — <https://www.trulens.org/> — RAG triad, traceable evals.
- **DeepEval** — <https://docs.confident-ai.com/> — pytest-style assertions.

Run these in CI against a frozen eval set; that's how you catch regressions when you swap embeddings or prompts.


## Complete Modern Pipeline (Reference Implementation)

```python
from __future__ import annotations
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import voyageai, cohere

class ModernRAG:
    def __init__(self, docs: list[dict]):
        self.vo = voyageai.Client()
        self.co = cohere.Client()
        self.qdrant = QdrantClient(":memory:")

        # 1. Chunk + contextualize (prompt-cache the document prefix in real code)
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        self.chunks = []
        for doc in docs:
            for piece in splitter.split_text(doc["text"]):
                ctx = self._contextualize(doc["text"], piece)
                self.chunks.append({
                    "text": piece,                   # what we serve to the LLM
                    "indexed_text": f"{ctx}\n\n{piece}",  # what we embed + BM25
                    "metadata": doc["metadata"],
                })

        # 2. Embed (Matryoshka-truncated to 1024 for speed; full-dim could be 2048)
        embeddings = self.vo.embed(
            texts=[c["indexed_text"] for c in self.chunks],
            model="voyage-3-large",
            input_type="document",
            output_dimension=1024,
        ).embeddings

        # 3. Index dense + sparse (Qdrant supports both natively)
        # ... omitted: collection creation with named dense + sparse vectors,
        #              upsert chunks, build BM25 sparse vectors via fastembed.

    def _contextualize(self, full: str, chunk: str) -> str:
        # Cheap LLM call; in production, prompt-cache `full`.
        return self.co.chat(message=(
            f"<document>\n{full}\n</document>\n"
            f"<chunk>\n{chunk}\n</chunk>\n"
            "Give a 1-2 sentence context situating this chunk."
        ), model="command-r-plus").text

    def retrieve(self, query: str, k: int = 8) -> list[dict]:
        q_dense = self.vo.embed(
            texts=[query], model="voyage-3-large", input_type="query",
            output_dimension=1024,
        ).embeddings[0]
        # Dense + sparse hybrid retrieval, fused server-side with RRF in Qdrant.
        # ... omitted: hybrid query, returns ~50 candidates.
        candidates = []  # placeholder

        # Cross-encoder rerank to top-k
        reranked = self.co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=[c["text"] for c in candidates],
            top_n=k,
        )
        return [candidates[r.index] for r in reranked.results]

    def answer(self, query: str) -> dict:
        retrieved = self.retrieve(query)
        context = "\n\n".join(
            f"[{i+1}] ({c['metadata']['source']})\n{c['text']}"
            for i, c in enumerate(retrieved)
        )
        # ... call generation LLM with grounded-answer prompt ...
        return {"answer": "...", "sources": retrieved}
```


## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| No contextualization of chunks | Retrieval recall plateaus around 90–95% | Add Anthropic-style contextual blurbs (see Component 2). |
| Dense-only retrieval | Misses IDs, SKUs, rare terms | Hybrid + RRF. |
| No reranker | Top-5 contains the gold chunk only ~60% of the time | Over-retrieve k=50, rerank to k=5 with Cohere/Voyage/BGE. |
| Bad PDF extraction | Tables / figures missing from chunks | Switch to layout-aware parser; consider ColPali for visual docs. |
| Stuffing 50 chunks into context | Costs spike, quality plateaus or regresses | Cap reranked context, order for lost-in-middle. |
| RAG when long-context wins | Small corpus, complex synthesis, no citation requirement | Use prompt caching with full corpus; cross-ref `context-engineering-and-prompt-caching.md`. |
| No eval harness | "Did this change make things better?" unanswerable | Frozen eval set + RAGAS/TruLens in CI. |
| LangChain wrapper mismatch | `ImportError: cannot import name X from langchain.vectorstores` | Use `langchain_community.vectorstores` or the provider SDK. |


## Summary

1. **Decide if RAG is the right tool.** Long-context + caching wins on small static corpora; RAG wins on huge / fresh / auditable corpora.
2. **Chunk well, then contextualize.** Anthropic's contextual retrieval is the highest-leverage no-cost improvement of the last two years.
3. **Hybrid retrieval with RRF is the default.** Dense alone misses lexical, sparse alone misses semantic.
4. **Rerank.** A modern cross-encoder (Cohere Rerank 3.5, Voyage rerank-2.5, BGE-reranker-v2-m3) is the difference between "demo" and "production."
5. **Late interaction (ColBERT, ColPali) for visual / layout-rich docs.**
6. **Agentic RAG (CRAG, Self-RAG, GraphRAG) for failure modes vanilla RAG can't handle** — bad retrieval, multi-hop, sensemaking.
7. **Evaluate.** Frozen eval set, retrieval metrics, end-to-end LLM-judge, in CI.

Cross-references: `yzmir-ml-production` for serving / inference cost, `ordis-security-architect` for prompt-injection hardening of retrieved content, `context-engineering-and-prompt-caching.md` and `reasoning-models.md` and `agentic-patterns-and-mcp.md` (this campaign) for adjacent decisions.

---

*Model lineup current as of 2026-05; revisit quarterly.*
