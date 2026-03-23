
# RAG Architecture Patterns

## Context

You're building a RAG (Retrieval-Augmented Generation) system to give LLMs access to external knowledge. Common mistakes:
- **No chunking strategy** (full docs → overflow, poor precision)
- **Poor retrieval** (cosine similarity alone → misses exact matches)
- **No re-ranking** (irrelevant results prioritized)
- **No evaluation** (can't measure or optimize quality)
- **Context overflow** (too many chunks → cost, latency, 'lost in middle')

**This skill provides effective RAG architecture: chunking, hybrid search, re-ranking, evaluation, and complete pipeline design.**


## What is RAG?

**RAG = Retrieval-Augmented Generation**

**Problem:** LLMs have knowledge cutoffs and can't access private/recent data.

**Solution:** Retrieve relevant information, inject into prompt, generate answer.

```python
# Without RAG:
answer = llm("What is our return policy?")
# LLM: "I don't have access to your specific return policy."

# With RAG:
relevant_docs = retrieval_system.search("return policy")
context = '\n'.join(relevant_docs)
prompt = f"Context: {context}\n\nQuestion: What is our return policy?\nAnswer:"
answer = llm(prompt)
# LLM: "Our return policy allows returns within 30 days..." (from retrieved docs)
```

**When to use RAG:**
- ✅ Private data (company docs, internal knowledge base)
- ✅ Recent data (news, updates since LLM training cutoff)
- ✅ Large knowledge base (can't fit in prompt/fine-tuning)
- ✅ Need citations (retrieval provides source documents)
- ✅ Changing information (update docs, not model)

**When NOT to use RAG:**
- ❌ General knowledge (already in LLM)
- ❌ Small knowledge base (< 100 docs → few-shot examples in prompt)
- ❌ Reasoning tasks (RAG provides facts, not reasoning)


## RAG Architecture Overview

```
User Query
    ↓
1. Query Processing (optional: expansion, rewriting)
    ↓
2. Retrieval (dense + sparse hybrid search)
    ↓
3. Re-ranking (refine top results)
    ↓
4. Context Selection (top-k chunks)
    ↓
5. Prompt Construction (inject context)
    ↓
6. LLM Generation
    ↓
Answer (with citations)
```


## Component 1: Document Processing & Chunking

### Why Chunking?

**Problem:** Documents are long (10k-100k tokens), embeddings and LLMs have limits.

**Solution:** Split documents into chunks (500-1000 tokens each).

### Chunking Strategies

**1. Fixed-size chunking (simple, works for most cases):**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Characters (roughly 750 tokens)
    chunk_overlap=200,  # Overlap for continuity
    separators=["\n\n", "\n", ". ", " ", ""]  # Try these in order
)

chunks = splitter.split_text(document)
```

**Parameters:**
- `chunk_size`: 500-1000 tokens typical (600-1500 characters)
- `chunk_overlap`: 10-20% of chunk_size (continuity between chunks)
- `separators`: Try semantic boundaries first (paragraphs > sentences > words)

**2. Semantic chunking (preserves meaning):**

```python
def semantic_chunking(text, max_chunk_size=1000):
    # Split on semantic boundaries
    sections = text.split('\n\n## ')  # Markdown headers

    chunks = []
    current_chunk = []
    current_size = 0

    for section in sections:
        section_size = len(section)

        if current_size + section_size <= max_chunk_size:
            current_chunk.append(section)
            current_size += section_size
        else:
            # Flush current chunk
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [section]
            current_size = section_size

    # Flush remaining
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks
```

**Benefits:** Preserves topic boundaries, more coherent chunks.

**3. Recursive chunking (LangChain default):**

```python
# Try splitting on larger boundaries first, fallback to smaller
separators = [
    "\n\n",  # Paragraphs (try first)
    "\n",    # Lines
    ". ",    # Sentences
    " ",     # Words
    ""       # Characters (last resort)
]

# For each separator:
# - If chunk fits: Done
# - If chunk too large: Try next separator
# Result: Largest semantic unit that fits in chunk_size
```

**Best for:** Mixed documents (code + prose, structured + unstructured).

### Chunking Best Practices

**Metadata preservation:**
```python
chunks = []
for page_num, page_text in enumerate(pdf_pages):
    page_chunks = splitter.split_text(page_text)

    for chunk_idx, chunk in enumerate(page_chunks):
        chunks.append({
            'text': chunk,
            'metadata': {
                'source': 'document.pdf',
                'page': page_num,
                'chunk_id': f"{page_num}_{chunk_idx}"
            }
        })

# Later: Cite sources in answer
# "According to page 42 of document.pdf..."
```

**Overlap for continuity:**
```python
# Without overlap: Sentence split across chunks (loss of context)
chunk1 = "...the process is simple. First,"
chunk2 = "you need to configure the settings..."

# With overlap (200 chars):
chunk1 = "...the process is simple. First, you need to configure"
chunk2 = "First, you need to configure the settings..."
# Overlap preserves context!
```

**Chunk size guidelines:**
```
Embedding model limit | Chunk size
----------------------|------------
512 tokens           | 400 tokens (leave room for overlap)
1024 tokens          | 800 tokens
2048 tokens          | 1500 tokens

Typical: 500-1000 tokens per chunk (balance precision vs context)
```


## Component 2: Embeddings

### What are Embeddings?

**Vector representation of text capturing semantic meaning.**

```python
text = "What is the return policy?"
embedding = embedding_model.encode(text)
# embedding: [0.234, -0.123, 0.891, ...] (384-1536 dimensions)

# Similar texts have similar embeddings (high cosine similarity)
query_emb = embed("return policy")
doc1_emb = embed("Returns accepted within 30 days")  # High similarity
doc2_emb = embed("Product specifications")  # Low similarity
```

### Embedding Models

**Popular models:**

```python
# 1. OpenAI embeddings (API-based)
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# Dimensions: 1536, Cost: $0.02 per 1M tokens

# 2. Sentence Transformers (open-source, local)
from sentence_transformers import SentenceTransformer
embeddings = SentenceTransformer('all-MiniLM-L6-v2')
# Dimensions: 384, Cost: $0 (local), Fast

# 3. Domain-specific
embeddings = SentenceTransformer('allenai-specter')  # Scientific papers
embeddings = SentenceTransformer('msmarco-distilbert-base-v4')  # Search/QA
```

**Selection criteria:**

| Model | Dimensions | Speed | Quality | Cost | Use Case |
|-------|------------|-------|---------|------|----------|
| OpenAI text-3-small | 1536 | Medium | Very Good | $0.02/1M | General (API) |
| OpenAI text-3-large | 3072 | Slow | Excellent | $0.13/1M | High quality |
| all-MiniLM-L6-v2 | 384 | Fast | Good | $0 | General (local) |
| all-mpnet-base-v2 | 768 | Medium | Very Good | $0 | General (local) |
| msmarco-* | 768 | Medium | Excellent | $0 | Search/QA |

**Evaluation:**
```python
# Test on your domain!
from sentence_transformers import util

query = "What is the return policy?"
docs = ["Returns within 30 days", "Shipping takes 5-7 days", "Product warranty"]

for model_name in ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'msmarco-distilbert-base-v4']:
    model = SentenceTransformer(model_name)

    query_emb = model.encode(query)
    doc_embs = model.encode(docs)

    similarities = util.cos_sim(query_emb, doc_embs)[0]
    print(f"{model_name}: {similarities}")

# Pick model with highest similarity for relevant doc
```


## Component 3: Vector Databases

**Store and retrieve embeddings efficiently.**

### Popular Vector DBs:

```python
# 1. Chroma (simple, local)
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_texts(chunks, embeddings)

# 2. Pinecone (managed, scalable)
import pinecone
pinecone.init(api_key="...", environment="...")
vectorstore = Pinecone.from_texts(chunks, embeddings, index_name="my-index")

# 3. Weaviate (open-source, scalable)
from langchain.vectorstores import Weaviate
vectorstore = Weaviate.from_texts(chunks, embeddings)

# 4. FAISS (Facebook, local, fast)
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_texts(chunks, embeddings)
```

### Vector DB Selection:

| Database | Type | Scale | Cost | Hosting | Best For |
|----------|------|-------|------|---------|----------|
| Chroma | Local | Small (< 1M) | $0 | Self | Development |
| FAISS | Local | Medium (< 10M) | $0 | Self | Production (self-hosted) |
| Pinecone | Cloud | Large (billions) | $70+/mo | Managed | Production (managed) |
| Weaviate | Both | Large | $0-$200/mo | Both | Production (flexible) |

### Similarity Search:

```python
# Basic similarity search
query = "What is the return policy?"
results = vectorstore.similarity_search(query, k=5)
# Returns: Top 5 most similar chunks

# With scores
results = vectorstore.similarity_search_with_score(query, k=5)
# Returns: [(chunk, similarity_score), ...]
# similarity_score: 0.0-1.0 (higher = more similar)

# With threshold
results = vectorstore.similarity_search_with_score(query, k=10)
filtered = [(chunk, score) for chunk, score in results if score > 0.7]
# Only keep highly similar results
```


## Component 4: Retrieval Strategies

### 1. Dense Retrieval (Semantic)

**Uses embeddings (what we've discussed).**

```python
query_embedding = embedding_model.encode(query)
# Find docs with embeddings most similar to query_embedding
results = vectorstore.similarity_search(query, k=10)
```

**Pros:**
- ✅ Semantic similarity (understands meaning, not just keywords)
- ✅ Handles synonyms, paraphrasing

**Cons:**
- ❌ Misses exact keyword matches
- ❌ Can confuse similar-sounding but different concepts

### 2. Sparse Retrieval (Keyword)

**Classic information retrieval (BM25, TF-IDF).**

```python
from langchain.retrievers import BM25Retriever

# BM25: Keyword-based ranking
bm25_retriever = BM25Retriever.from_texts(chunks)
results = bm25_retriever.get_relevant_documents(query)
```

**How BM25 works:**
```
Score(query, doc) = sum over query terms of:
  IDF(term) * (TF(term) * (k1 + 1)) / (TF(term) + k1 * (1 - b + b * doc_length / avg_doc_length))

Where:
- TF = term frequency (how often term appears in doc)
- IDF = inverse document frequency (rarity of term)
- k1, b = tuning parameters
```

**Pros:**
- ✅ Exact keyword matches (important for IDs, SKUs, technical terms)
- ✅ Fast (no neural network)
- ✅ Explainable (can see which keywords matched)

**Cons:**
- ❌ No semantic understanding (misses synonyms, paraphrasing)
- ❌ Sensitive to exact wording

### 3. Hybrid Retrieval (Dense + Sparse)

**Combine both for best results!**

```python
from langchain.retrievers import EnsembleRetriever

# Dense retriever (semantic)
dense_retriever = vectorstore.as_retriever(search_kwargs={'k': 20})

# Sparse retriever (keyword)
sparse_retriever = BM25Retriever.from_texts(chunks)

# Ensemble (hybrid)
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]  # Equal weight (tune based on evaluation)
)

results = hybrid_retriever.get_relevant_documents(query)
```

**When hybrid helps:**

```python
# Query: "What is the SKU for product ABC-123?"

# Dense only:
# - Might retrieve: "product catalog", "product specifications"
# - Misses: Exact SKU "ABC-123" (keyword)

# Sparse only:
# - Retrieves: "ABC-123" (keyword match)
# - Misses: Semantically similar products

# Hybrid:
# - Retrieves: Exact SKU + related products
# - Best of both worlds!
```

**Weight tuning:**
```python
# Evaluate different weights
for dense_weight in [0.3, 0.5, 0.7]:
    sparse_weight = 1 - dense_weight

    retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[dense_weight, sparse_weight]
    )

    mrr = evaluate_retrieval(retriever, test_set)
    print(f"Dense:{dense_weight}, Sparse:{sparse_weight} → MRR:{mrr:.3f}")

# Example output:
# Dense:0.3, Sparse:0.7 → MRR:0.65
# Dense:0.5, Sparse:0.5 → MRR:0.72  # Best!
# Dense:0.7, Sparse:0.3 → MRR:0.68
```


## Component 5: Re-Ranking

**Refine coarse retrieval ranking with cross-encoder.**

### Why Re-Ranking?

```
Retrieval (bi-encoder):
- Encodes query and docs separately
- Fast: O(1) for pre-computed doc embeddings
- Coarse: Single similarity score

Re-ranking (cross-encoder):
- Jointly encodes query + doc
- Slow: O(n) for n docs (must process each pair)
- Precise: Sees query-doc interactions
```

**Pipeline:**
```
1. Retrieval: Get top 20-50 (fast, broad)
2. Re-ranking: Refine to top 5-10 (slow, precise)
```

### Implementation:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load cross-encoder for re-ranking
model = AutoModelForSequenceClassification.from_pretrained(
    'cross-encoder/ms-marco-MiniLM-L-6-v2'
)
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, retrieved_docs, top_k=5):
    # Score each doc with cross-encoder
    scores = []
    for doc in retrieved_docs:
        inputs = tokenizer(query, doc, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            score = model(**inputs).logits[0][0].item()
        scores.append((doc, score))

    # Sort by score (descending)
    reranked = sorted(scores, key=lambda x: x[1], reverse=True)

    # Return top-k
    return [doc for doc, score in reranked[:top_k]]

# Usage
initial_results = vectorstore.similarity_search(query, k=20)  # Over-retrieve
final_results = rerank(query, initial_results, top_k=5)  # Re-rank
```

### Re-Ranking Models:

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| ms-marco-MiniLM-L-6-v2 | 80MB | Fast | Good | General |
| ms-marco-MiniLM-L-12-v2 | 120MB | Medium | Very Good | Better quality |
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | 120MB | Medium | Very Good | Multilingual |

### Impact of Re-Ranking:

```python
# Without re-ranking:
results = vectorstore.similarity_search(query, k=5)
mrr = 0.55  # First relevant at rank ~2

# With re-ranking:
initial = vectorstore.similarity_search(query, k=20)
results = rerank(query, initial, top_k=5)
mrr = 0.82  # First relevant at rank ~1.2

# Improvement: 27% better ranking!
```


## Component 6: Query Processing

### Query Expansion

**Expand query with synonyms, related terms.**

```python
def expand_query(query, llm):
    prompt = f"""
    Generate 3 alternative phrasings of this query:

    Original: {query}

    Alternatives (semantically similar):
    1.
    2.
    3.
    """

    alternatives = llm(prompt)
    # Retrieve using all variants, merge results
    all_results = []
    for alt_query in [query] + alternatives:
        results = vectorstore.similarity_search(alt_query, k=10)
        all_results.extend(results)

    # Deduplicate and re-rank
    unique_results = list(set(all_results))
    return rerank(query, unique_results, top_k=5)
```

### Query Rewriting

**Simplify or decompose complex queries.**

```python
def rewrite_query(query, llm):
    # Complex query
    if is_complex(query):
        prompt = f"""
        Break this complex query into simpler sub-queries:

        Query: {query}

        Sub-queries:
        1.
        2.
        """
        sub_queries = llm(prompt)

        # Retrieve for each sub-query
        all_results = []
        for sub_q in sub_queries:
            results = vectorstore.similarity_search(sub_q, k=5)
            all_results.extend(results)

        return all_results

    return vectorstore.similarity_search(query, k=5)
```

### HyDE (Hypothetical Document Embeddings)

**Generate hypothetical answer, retrieve similar docs.**

```python
def hyde_retrieval(query, llm, vectorstore):
    # Generate hypothetical answer
    prompt = f"Answer this question in detail: {query}"
    hypothetical_answer = llm(prompt)

    # Retrieve docs similar to hypothetical answer (not query)
    results = vectorstore.similarity_search(hypothetical_answer, k=5)

    return results

# Why this works:
# - Queries are short, sparse
# - Answers are longer, richer
# - Doc-to-doc similarity (answer vs docs) better than query-to-doc
```


## Component 7: Context Management

### Context Budget

```python
max_context_tokens = 4000  # Budget for retrieved context

selected_chunks = []
total_tokens = 0

for chunk in reranked_results:
    chunk_tokens = count_tokens(chunk)

    if total_tokens + chunk_tokens <= max_context_tokens:
        selected_chunks.append(chunk)
        total_tokens += chunk_tokens
    else:
        break  # Stop when budget exceeded

# Result: Best chunks that fit in budget
```

### Lost in the Middle Problem

**LLMs prioritize start and end of context, miss middle.**

```python
# Research finding: Place most important info at start or end

def order_for_llm(chunks):
    # Best chunks at start and end
    if len(chunks) <= 2:
        return chunks

    # Put most relevant at positions 0 and -1
    ordered = [chunks[0]]  # Most relevant (start)
    ordered.extend(chunks[1:-1])  # Less relevant (middle)
    ordered.append(chunks[-1])  # Second most relevant (end)

    return ordered
```

### Contextual Compression

**Filter retrieved chunks to most relevant sentences.**

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Compressor: Extract relevant sentences
compressor = LLMChainExtractor.from_llm(llm)

# Wrap retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

# Retrieved chunks are automatically filtered to relevant parts
compressed_docs = compression_retriever.get_relevant_documents(query)
```


## Component 8: Prompt Construction

### Basic RAG Prompt:

```python
context = '\n\n'.join(retrieved_chunks)

prompt = f"""
Answer the question based on the context below. If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:
"""

answer = llm(prompt)
```

### With Citations:

```python
context_with_ids = []
for i, chunk in enumerate(retrieved_chunks):
    context_with_ids.append(f"[{i+1}] {chunk['text']}")

context = '\n\n'.join(context_with_ids)

prompt = f"""
Answer the question based on the context below. Cite sources using [number] format.

Context:
{context}

Question: {query}

Answer (with citations):
"""

answer = llm(prompt)
# Output: "The return policy allows returns within 30 days [1]. Shipping takes 5-7 business days [3]."
```

### With Metadata:

```python
context_with_metadata = []
for chunk in retrieved_chunks:
    source = chunk['metadata']['source']
    page = chunk['metadata']['page']
    context_with_metadata.append(f"From {source} (page {page}):\n{chunk['text']}")

context = '\n\n'.join(context_with_metadata)

prompt = f"""
Answer the question and cite your sources.

Context:
{context}

Question: {query}

Answer:
"""
```


## Evaluation Metrics

### Retrieval Metrics

**1. Mean Reciprocal Rank (MRR):**

```python
def calculate_mrr(retrieval_results, relevant_docs):
    """
    MRR = average of (1 / rank of first relevant doc)

    Example:
    Query 1: First relevant at rank 2 → 1/2 = 0.5
    Query 2: First relevant at rank 1 → 1/1 = 1.0
    Query 3: No relevant docs → 0
    MRR = (0.5 + 1.0 + 0) / 3 = 0.5
    """
    mrr_scores = []

    for results, relevant in zip(retrieval_results, relevant_docs):
        for i, result in enumerate(results):
            if result in relevant:
                mrr_scores.append(1 / (i + 1))
                break
        else:
            mrr_scores.append(0)  # No relevant found

    return np.mean(mrr_scores)

# Interpretation:
# MRR = 1.0: First result always relevant (perfect!)
# MRR = 0.5: First relevant at rank ~2 (good)
# MRR = 0.3: First relevant at rank ~3-4 (okay)
# MRR < 0.3: Poor retrieval (needs improvement)
```

**2. Precision@k:**

```python
def calculate_precision_at_k(retrieval_results, relevant_docs, k=5):
    """
    Precision@k = (# relevant docs in top-k) / k

    Example:
    Top 5 results: [relevant, irrelevant, relevant, irrelevant, irrelevant]
    Precision@5 = 2/5 = 0.4
    """
    precision_scores = []

    for results, relevant in zip(retrieval_results, relevant_docs):
        top_k = results[:k]
        relevant_in_topk = len([r for r in top_k if r in relevant])
        precision_scores.append(relevant_in_topk / k)

    return np.mean(precision_scores)

# Target: Precision@5 > 0.7 (70% of top-5 are relevant)
```

**3. Recall@k:**

```python
def calculate_recall_at_k(retrieval_results, relevant_docs, k=5):
    """
    Recall@k = (# relevant docs in top-k) / (total relevant docs)

    Example:
    Total relevant: 5
    Found in top-5: 2
    Recall@5 = 2/5 = 0.4
    """
    recall_scores = []

    for results, relevant in zip(retrieval_results, relevant_docs):
        top_k = results[:k]
        relevant_in_topk = len([r for r in top_k if r in relevant])
        recall_scores.append(relevant_in_topk / len(relevant))

    return np.mean(recall_scores)

# Interpretation:
# Recall@5 = 1.0: All relevant docs in top-5 (perfect!)
# Recall@5 = 0.5: Half of relevant docs in top-5
```

**4. NDCG (Normalized Discounted Cumulative Gain):**

```python
def calculate_ndcg(retrieval_results, relevance_scores, k=5):
    """
    NDCG considers position and graded relevance (0, 1, 2, 3...)

    DCG = sum of (relevance / log2(rank + 1))
    NDCG = DCG / ideal_DCG (normalized to 0-1)
    """
    from sklearn.metrics import ndcg_score

    # relevance_scores: 2D array of relevance (0-3) for each result
    # Higher relevance = more relevant

    ndcg = ndcg_score(relevance_scores, retrieval_results, k=k)
    return ndcg

# NDCG = 1.0: Perfect ranking
# NDCG > 0.7: Good ranking
# NDCG < 0.5: Poor ranking
```

### Generation Metrics

**1. Exact Match:**

```python
def calculate_exact_match(predictions, ground_truth):
    """Percentage of predictions that exactly match ground truth."""
    matches = [pred == truth for pred, truth in zip(predictions, ground_truth)]
    return np.mean(matches)
```

**2. F1 Score (token-level):**

```python
def calculate_f1(prediction, ground_truth):
    """F1 score based on token overlap."""
    pred_tokens = prediction.split()
    truth_tokens = ground_truth.split()

    common = set(pred_tokens) & set(truth_tokens)

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1
```

**3. LLM-as-Judge:**

```python
def evaluate_with_llm(answer, ground_truth, llm):
    """Use LLM to judge answer quality."""
    prompt = f"""
    Rate the quality of this answer on a scale of 1-5:
    1 = Completely wrong
    2 = Mostly wrong
    3 = Partially correct
    4 = Mostly correct
    5 = Completely correct

    Ground truth: {ground_truth}
    Answer to evaluate: {answer}

    Rating (1-5):
    """

    rating = llm(prompt)
    return int(rating)
```

### End-to-End Evaluation

```python
def evaluate_rag_system(rag_system, test_set):
    """
    Complete evaluation: retrieval + generation
    """
    # Retrieval metrics
    retrieval_results = []
    relevant_docs = []

    # Generation metrics
    predictions = []
    ground_truth = []

    for test_case in test_set:
        query = test_case['query']

        # Retrieve
        retrieved = rag_system.retrieve(query)
        retrieval_results.append(retrieved)
        relevant_docs.append(test_case['relevant_docs'])

        # Generate
        answer = rag_system.generate(query, retrieved)
        predictions.append(answer)
        ground_truth.append(test_case['expected_answer'])

    # Calculate metrics
    metrics = {
        'retrieval_mrr': calculate_mrr(retrieval_results, relevant_docs),
        'retrieval_precision@5': calculate_precision_at_k(retrieval_results, relevant_docs, k=5),
        'generation_f1': np.mean([calculate_f1(p, t) for p, t in zip(predictions, ground_truth)]),
        'generation_exact_match': calculate_exact_match(predictions, ground_truth),
    }

    return metrics
```


## Complete RAG Pipeline

### Basic Implementation:

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load documents
documents = load_documents('docs/')

# 2. Chunk documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 3. Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Create retrieval chain
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
    return_source_documents=True
)

# 5. Query
result = qa_chain({"query": "What is the return policy?"})
answer = result['result']
sources = result['source_documents']
```

### Advanced Implementation (Hybrid + Re-ranking):

```python
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class AdvancedRAG:
    def __init__(self, documents):
        # Chunk
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunks = splitter.split_documents(documents)

        # Embeddings
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(self.chunks, self.embeddings)

        # Hybrid retrieval
        dense_retriever = self.vectorstore.as_retriever(search_kwargs={'k': 20})
        sparse_retriever = BM25Retriever.from_documents(self.chunks)

        self.retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.5, 0.5]
        )

        # Re-ranker
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )

        # LLM
        self.llm = OpenAI(temperature=0)

    def retrieve(self, query, k=5):
        # Hybrid retrieval (over-retrieve)
        initial_results = self.retriever.get_relevant_documents(query)[:20]

        # Re-rank
        scores = []
        for doc in initial_results:
            inputs = self.rerank_tokenizer(
                query, doc.page_content,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )
            score = self.rerank_model(**inputs).logits[0][0].item()
            scores.append((doc, score))

        # Sort by score
        reranked = sorted(scores, key=lambda x: x[1], reverse=True)

        # Return top-k
        return [doc for doc, score in reranked[:k]]

    def generate(self, query, retrieved_docs):
        # Build context
        context = '\n\n'.join([f"[{i+1}] {doc.page_content}"
                               for i, doc in enumerate(retrieved_docs)])

        # Construct prompt
        prompt = f"""
        Answer the question based on the context below. Cite sources using [number].
        If the answer is not in the context, say "I don't have enough information."

        Context:
        {context}

        Question: {query}

        Answer:
        """

        # Generate
        answer = self.llm(prompt)

        return answer, retrieved_docs

    def query(self, query):
        retrieved_docs = self.retrieve(query, k=5)
        answer, sources = self.generate(query, retrieved_docs)

        return {
            'answer': answer,
            'sources': sources
        }

# Usage
rag = AdvancedRAG(documents)
result = rag.query("What is the return policy?")
print(result['answer'])
print(f"Sources: {[doc.metadata for doc in result['sources']]}")
```


## Optimization Strategies

### 1. Caching

```python
import functools

@functools.lru_cache(maxsize=1000)
def cached_retrieval(query):
    """Cache retrieval results for common queries."""
    return vectorstore.similarity_search(query, k=5)

# Saves embedding + retrieval cost for repeated queries
```

### 2. Async Retrieval

```python
import asyncio

async def async_retrieve(queries, vectorstore):
    """Retrieve for multiple queries in parallel."""
    tasks = [vectorstore.asimilarity_search(q, k=5) for q in queries]
    results = await asyncio.gather(*tasks)
    return results
```

### 3. Metadata Filtering

```python
# Filter by metadata before similarity search
results = vectorstore.similarity_search(
    query,
    k=5,
    filter={"source": "product_docs"}  # Only search product docs
)

# Faster (smaller search space) + more relevant (right domain)
```

### 4. Index Optimization

```python
# FAISS index optimization
import faiss

# 1. Train index on sample (faster search)
quantizer = faiss.IndexFlatL2(embedding_dim)
index = faiss.IndexIVFFlat(quantizer, embedding_dim, n_clusters)
index.train(sample_embeddings)

# 2. Set search parameters
index.nprobe = 10  # Trade-off: accuracy vs speed

# Result: 5-10× faster search with minimal quality loss
```


## Common Pitfalls

### Pitfall 1: No chunking
**Problem:** Full docs → overflow, poor precision
**Fix:** Chunk to 500-1000 tokens

### Pitfall 2: Dense-only retrieval
**Problem:** Misses exact keyword matches
**Fix:** Hybrid search (dense + sparse)

### Pitfall 3: No re-ranking
**Problem:** Coarse ranking, wrong results prioritized
**Fix:** Over-retrieve (k=20), re-rank to top-5

### Pitfall 4: Too much context
**Problem:** > 10k tokens → cost, latency, 'lost in middle'
**Fix:** Top 5 chunks (5k tokens), optimize retrieval precision

### Pitfall 5: No evaluation
**Problem:** Can't measure or optimize
**Fix:** Build test set, measure MRR, Precision@k


## Summary

**Core principles:**

1. **Chunk documents**: 500-1000 tokens, semantic boundaries, overlap for continuity
2. **Hybrid retrieval**: Dense (semantic) + Sparse (keyword) = best results
3. **Re-rank**: Over-retrieve (k=20-50), refine to top-5 with cross-encoder
4. **Evaluate systematically**: MRR, Precision@k, Recall@k, NDCG for retrieval; F1, Exact Match for generation
5. **Keep context focused**: Top 5 chunks (~5k tokens), optimize retrieval not context size

**Pipeline:**
```
Documents → Chunk → Embed → Vector DB
Query → Hybrid Retrieval (k=20) → Re-rank (k=5) → Context → LLM → Answer
```

**Metrics targets:**
- MRR > 0.7 (first relevant in top ~1.4)
- Precision@5 > 0.7 (70% of top-5 relevant)
- Generation F1 > 0.8 (80% token overlap)

**Key insight:** RAG quality depends on retrieval precision. Optimize retrieval (chunking, hybrid search, re-ranking, evaluation) before adding context or changing LLMs.
