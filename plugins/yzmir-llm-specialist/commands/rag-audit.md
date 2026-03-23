---
description: Audit RAG system quality - chunking, retrieval, re-ranking, evaluation
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write"]
argument-hint: "[rag_system_path]"
---

# RAG System Audit Command

You are auditing a RAG (Retrieval-Augmented Generation) system. Follow the systematic audit framework.

## Core Principle

**RAG quality depends on retrieval precision.** Fix retrieval before touching the LLM.

## Audit Framework

```
1. Document Processing → 2. Retrieval Quality → 3. Context Management → 4. Generation → 5. Evaluation
```

## Phase 1: Document Processing Audit

### Check Chunking Strategy

```bash
# Search for chunking configuration
grep -rn "chunk_size\|chunk_overlap\|text_splitter" --include="*.py"

# Look for RecursiveCharacterTextSplitter or similar
grep -rn "TextSplitter\|split_text\|split_documents" --include="*.py"
```

**Chunking Checklist:**

| Check | Good | Bad |
|-------|------|-----|
| Chunk size | 500-1000 tokens | < 200 or > 2000 tokens |
| Overlap | 10-20% of chunk size | 0% or > 50% |
| Boundaries | Semantic (paragraphs, sections) | Fixed character count |
| Metadata | Source, page, section preserved | Lost during processing |

**Common Issues:**

```python
# BAD: No overlap (context lost at boundaries)
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# GOOD: Semantic splitting with overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,  # 20% overlap
    separators=["\n\n", "\n", ". ", " ", ""]  # Semantic boundaries
)
```

### Check Embedding Model

```bash
# Search for embedding configuration
grep -rn "embedding\|OpenAIEmbeddings\|SentenceTransformer" --include="*.py"
```

**Embedding Checklist:**

| Check | Good | Bad |
|-------|------|-----|
| Model choice | Domain-appropriate | Generic small model |
| Dimension | Matches vector DB config | Mismatch |
| Batch processing | Batched embedding | One-at-a-time |

## Phase 2: Retrieval Quality Audit

### Check Retrieval Method

```bash
# Search for retrieval configuration
grep -rn "similarity_search\|as_retriever\|search_kwargs" --include="*.py"

# Check for hybrid search
grep -rn "BM25\|EnsembleRetriever\|hybrid" --include="*.py"

# Check for re-ranking
grep -rn "rerank\|cross.encoder\|CrossEncoder" --include="*.py"
```

**Retrieval Checklist:**

| Check | Good | Bad |
|-------|------|-----|
| Search type | Hybrid (dense + sparse) | Dense only |
| k value | Over-retrieve (k=20), then re-rank | Fixed small k (k=3) |
| Re-ranking | Cross-encoder re-ranking | No re-ranking |
| Filtering | Metadata filters when applicable | No filtering |

**Common Issues:**

```python
# BAD: Dense-only, fixed k, no re-ranking
results = vectorstore.similarity_search(query, k=5)

# GOOD: Hybrid search with re-ranking
from langchain.retrievers import EnsembleRetriever, BM25Retriever

# Hybrid retrieval
dense_retriever = vectorstore.as_retriever(search_kwargs={'k': 20})
sparse_retriever = BM25Retriever.from_texts(chunks)
hybrid = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)

# Over-retrieve then re-rank
initial_results = hybrid.get_relevant_documents(query)[:20]
final_results = rerank(query, initial_results, top_k=5)
```

### Measure Retrieval Quality

```python
def audit_retrieval(retriever, test_queries, ground_truth):
    """Measure retrieval quality metrics."""
    mrr_scores = []
    precision_scores = []
    recall_scores = []

    for query, relevant_docs in zip(test_queries, ground_truth):
        results = retriever.get_relevant_documents(query)
        result_ids = [r.metadata.get('id') for r in results[:5]]

        # MRR: Position of first relevant result
        for i, doc_id in enumerate(result_ids):
            if doc_id in relevant_docs:
                mrr_scores.append(1 / (i + 1))
                break
        else:
            mrr_scores.append(0)

        # Precision@5
        relevant_in_top5 = len(set(result_ids) & set(relevant_docs))
        precision_scores.append(relevant_in_top5 / 5)

        # Recall@5
        recall_scores.append(relevant_in_top5 / len(relevant_docs))

    return {
        'MRR': sum(mrr_scores) / len(mrr_scores),
        'Precision@5': sum(precision_scores) / len(precision_scores),
        'Recall@5': sum(recall_scores) / len(recall_scores)
    }

# Targets:
# MRR > 0.7 (first relevant at rank ~1.4)
# Precision@5 > 0.6 (60% of top-5 relevant)
# Recall@5 > 0.5 (find 50% of relevant docs)
```

## Phase 3: Context Management Audit

### Check Context Budget

```bash
# Search for context/token limits
grep -rn "max_tokens\|context_length\|token_limit" --include="*.py"

# Check for context compression
grep -rn "compress\|summarize\|ContextualCompression" --include="*.py"
```

**Context Checklist:**

| Check | Good | Bad |
|-------|------|-----|
| Token budget | Explicit limit (e.g., 4000) | No limit |
| Chunk count | 3-7 chunks | 10+ chunks |
| Ordering | Most relevant at start/end | Random order |
| Compression | Applied when needed | Always full chunks |

**Lost in the Middle Problem:**

```python
# LLMs pay less attention to middle of context
# Place most important info at START and END

def order_for_attention(chunks):
    """Order chunks to avoid 'lost in the middle' problem."""
    if len(chunks) <= 2:
        return chunks

    # Best at start, second-best at end
    return [chunks[0]] + chunks[2:-1] + [chunks[1]]
```

## Phase 4: Generation Audit

### Check Prompt Template

```bash
# Search for RAG prompt templates
grep -rn "Context:\|Based on.*context\|Answer.*following" --include="*.py" -A5
```

**Generation Checklist:**

| Check | Good | Bad |
|-------|------|-----|
| Context injection | Clearly marked section | Mixed with instructions |
| Citation support | Source references requested | No citation |
| Fallback handling | "I don't know" instruction | Silent hallucination |
| Format guidance | Explicit output format | Implicit |

**Good RAG Prompt:**

```python
prompt = f"""Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't have enough information."
Cite sources using [1], [2], etc.

Context:
{context_with_numbered_sources}

Question: {query}

Answer (with citations):"""
```

## Phase 5: End-to-End Evaluation

### Create Test Set

```python
# Minimum 50 test cases with:
# - Query
# - Expected answer (ground truth)
# - Relevant source documents

test_cases = [
    {
        "query": "What is the return policy?",
        "expected_answer": "Returns accepted within 30 days with receipt.",
        "relevant_docs": ["policy_doc_1", "faq_doc_3"]
    },
    # ... more cases
]
```

### Measure End-to-End Quality

```python
def evaluate_rag_e2e(rag_system, test_cases):
    """End-to-end RAG evaluation."""
    results = {
        'retrieval_mrr': [],
        'answer_f1': [],
        'answer_exact_match': [],
        'hallucination_rate': 0,
    }

    hallucinations = 0

    for case in test_cases:
        # Get RAG response
        response = rag_system.query(case['query'])

        # Retrieval quality
        retrieved_ids = [d.metadata['id'] for d in response['sources']]
        relevant_found = any(rid in case['relevant_docs'] for rid in retrieved_ids)
        results['retrieval_mrr'].append(1.0 if relevant_found else 0.0)

        # Answer quality (token F1)
        pred_tokens = set(response['answer'].lower().split())
        true_tokens = set(case['expected_answer'].lower().split())
        overlap = pred_tokens & true_tokens

        if overlap:
            precision = len(overlap) / len(pred_tokens)
            recall = len(overlap) / len(true_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        results['answer_f1'].append(f1)

        # Exact match
        results['answer_exact_match'].append(
            response['answer'].strip().lower() == case['expected_answer'].strip().lower()
        )

        # Hallucination check (answer not grounded in sources)
        if not answer_grounded_in_sources(response['answer'], response['sources']):
            hallucinations += 1

    return {
        'MRR': sum(results['retrieval_mrr']) / len(results['retrieval_mrr']),
        'Answer_F1': sum(results['answer_f1']) / len(results['answer_f1']),
        'Exact_Match': sum(results['answer_exact_match']) / len(results['answer_exact_match']),
        'Hallucination_Rate': hallucinations / len(test_cases)
    }

# Targets:
# MRR > 0.7
# Answer F1 > 0.7
# Hallucination Rate < 5%
```

## Common RAG Issues and Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| Poor chunking | Answers incomplete, missing context | Increase overlap, use semantic boundaries |
| Dense-only search | Missing exact keyword matches | Add BM25 hybrid search |
| No re-ranking | Wrong docs prioritized | Add cross-encoder re-ranking |
| Too much context | Slow, expensive, confused answers | Limit to top-5, compress |
| No evaluation | Can't measure quality | Create test set, measure MRR/F1 |

## Audit Report Template

After audit, provide:

1. **Document Processing**
   - Chunking: size, overlap, boundaries
   - Embeddings: model, dimensions

2. **Retrieval Quality**
   - Method: dense/hybrid/re-ranked
   - Metrics: MRR, Precision@5, Recall@5

3. **Context Management**
   - Budget: tokens, chunk count
   - Ordering: attention-aware?

4. **Generation**
   - Prompt quality: citations, fallback
   - Hallucination rate

5. **Recommendations**
   - Priority fixes
   - Expected improvement
