---
name: context-window-management
description: Manage context windows via token counting and truncation for documents and conversations.
---

# Context Window Management Skill

## When to Use This Skill

Use this skill when:
- Processing documents longer than model context limit
- Building multi-turn conversational agents
- Implementing RAG systems with retrieved context
- Handling user inputs of unknown length
- Managing long-running conversations (customer support, assistants)
- Optimizing cost and latency for context-heavy applications

**When NOT to use:** Short, fixed-length inputs guaranteed to fit in context (e.g., tweet classification, short form filling).

## Core Principle

**Context is finite. Managing it is mandatory.**

LLM context windows have hard limits:
- GPT-3.5-turbo: 4k tokens (~3k words)
- GPT-3.5-turbo-16k: 16k tokens (~12k words)
- GPT-4: 8k tokens (~6k words)
- GPT-4-turbo: 128k tokens (~96k words)
- Claude 3 Sonnet: 200k tokens (~150k words)

Exceeding these limits = API crash. No graceful degradation. Token counting and management are not optional.

**Formula:** Token counting (prevent overflow) + Budgeting (allocate efficiently) + Management strategy (truncation/chunking/summarization) = Robust context handling.

## Context Management Framework

```
┌──────────────────────────────────────────────────┐
│          1. Count Tokens                         │
│  tiktoken, model-specific encoding               │
└────────────┬─────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────┐
│          2. Check Against Limits                 │
│  Model-specific context windows                  │
└────────────┬─────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────┐
│          3. Token Budget Allocation              │
│  System + Context + Query + Output               │
└────────────┬─────────────────────────────────────┘
             │
             ▼
        ┌────┴────┐
        │ Fits?   │
        └────┬────┘
      ┌──────┴──────┐
      │ Yes         │ No
      ▼             ▼
 ┌─────────┐   ┌─────────────────────┐
 │ Proceed │   │ Choose Strategy:     │
 └─────────┘   │ • Chunking           │
               │ • Truncation         │
               │ • Summarization      │
               │ • Larger model       │
               │ • Compression        │
               └─────────┬───────────┘
                         │
                         ▼
               ┌──────────────────┐
               │ Apply & Validate │
               └──────────────────┘
```

## Part 1: Token Counting

### Why Token Counting Matters

LLMs tokenize text (not characters or words). Token counts vary by:
- Language (English ~4 chars/token, Chinese ~2 chars/token)
- Content (code ~3 chars/token, prose ~4.5 chars/token)
- Model (different tokenizers)

**Character/word counts are unreliable estimates.**

### Tiktoken: OpenAI's Tokenizer

**Installation:**
```bash
pip install tiktoken
```

**Basic Usage:**

```python
import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    """
    Count tokens for given text and model.

    Args:
        text: String to tokenize
        model: Model name (determines tokenizer)

    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4/3.5-turbo

    return len(encoding.encode(text))

# Examples
text = "Hello, how are you today?"
print(f"Tokens: {count_tokens(text)}")  # Output: 7 tokens

document = "Large document with 10,000 words..."
tokens = count_tokens(document, model="gpt-4")
print(f"Document tokens: {tokens:,}")  # Output: Document tokens: 13,421
```

**Encoding Types by Model:**

| Model | Encoding | Notes |
|-------|----------|-------|
| gpt-3.5-turbo | cl100k_base | Default for GPT-3.5/4 |
| gpt-4 | cl100k_base | Same as GPT-3.5 |
| gpt-4-turbo | cl100k_base | Same as GPT-3.5 |
| text-davinci-003 | p50k_base | Legacy GPT-3 |
| code-davinci-002 | p50k_base | Codex |

**Counting Chat Messages:**

```python
def count_message_tokens(messages, model="gpt-3.5-turbo"):
    """
    Count tokens in chat completion messages.

    Chat format adds overhead: role names, formatting tokens.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = 0

    # Message formatting overhead (varies by model)
    tokens_per_message = 3  # Every message: <|im_start|>role\n, <|im_end|>\n
    tokens_per_name = 1  # If name field present

    for message in messages:
        tokens += tokens_per_message
        for key, value in message.items():
            tokens += len(encoding.encode(value))
            if key == "name":
                tokens += tokens_per_name

    tokens += 3  # Every reply starts with assistant message

    return tokens

# Example
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about Python."},
    {"role": "assistant", "content": "Python is a high-level programming language..."}
]

total_tokens = count_message_tokens(messages)
print(f"Total tokens: {total_tokens}")
```

**Token Estimation (Quick Approximation):**

```python
def estimate_tokens(text):
    """
    Quick estimation: ~4 characters per token for English prose.

    Not accurate for API calls! Use tiktoken for production.
    Useful for rough checks and dashboards.
    """
    return len(text) // 4

# Example
text = "This is a sample text for estimation."
estimated = estimate_tokens(text)
actual = count_tokens(text)
print(f"Estimated: {estimated}, Actual: {actual}")
# Output: Estimated: 9, Actual: 10 (close but not exact)
```

---

## Part 2: Model Context Limits and Budgeting

### Context Window Sizes

```python
MODEL_LIMITS = {
    # OpenAI GPT-3.5
    "gpt-3.5-turbo": 4_096,
    "gpt-3.5-turbo-16k": 16_384,

    # OpenAI GPT-4
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-4-turbo": 128_000,
    "gpt-4-turbo-2024-04-09": 128_000,

    # Anthropic Claude
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,

    # Open source
    "llama-2-7b": 4_096,
    "llama-2-13b": 4_096,
    "llama-2-70b": 4_096,
    "mistral-7b": 8_192,
    "mixtral-8x7b": 32_768,
}

def get_context_limit(model):
    """Get context window size for model."""
    return MODEL_LIMITS.get(model, 4_096)  # Default: 4k
```

### Token Budget Allocation

For systems with multiple components (RAG, chat with history), allocate tokens:

```python
def calculate_token_budget(
    model="gpt-3.5-turbo",
    system_message_tokens=None,
    query_tokens=None,
    output_tokens=500,
    safety_margin=50
):
    """
    Calculate remaining budget for context (e.g., retrieved documents).

    Args:
        model: LLM model name
        system_message_tokens: Tokens in system message (if known)
        query_tokens: Tokens in user query (if known)
        output_tokens: Reserved tokens for model output
        safety_margin: Extra buffer to prevent edge cases

    Returns:
        Available tokens for context
    """
    total_limit = MODEL_LIMITS[model]

    # Reserve tokens
    reserved = (
        (system_message_tokens or 100) +  # System message (estimate if unknown)
        (query_tokens or 100) +           # User query (estimate if unknown)
        output_tokens +                   # Model response
        safety_margin                     # Safety buffer
    )

    context_budget = total_limit - reserved

    return {
        'total_limit': total_limit,
        'context_budget': context_budget,
        'reserved_system': system_message_tokens or 100,
        'reserved_query': query_tokens or 100,
        'reserved_output': output_tokens,
        'safety_margin': safety_margin
    }

# Example
budget = calculate_token_budget(
    model="gpt-3.5-turbo",
    system_message_tokens=50,
    query_tokens=20,
    output_tokens=500
)

print(f"Total limit: {budget['total_limit']:,}")
print(f"Context budget: {budget['context_budget']:,}")
# Output:
# Total limit: 4,096
# Context budget: 3,376 (can use for retrieved docs, chat history, etc.)
```

**RAG Token Budgeting:**

```python
def budget_for_rag(
    query,
    system_message="You are a helpful assistant. Answer using the provided context.",
    model="gpt-3.5-turbo",
    output_tokens=500
):
    """Calculate available tokens for retrieved documents in RAG."""
    system_tokens = count_tokens(system_message, model)
    query_tokens = count_tokens(query, model)

    budget = calculate_token_budget(
        model=model,
        system_message_tokens=system_tokens,
        query_tokens=query_tokens,
        output_tokens=output_tokens
    )

    return budget['context_budget']

# Example
query = "What is the company's return policy for defective products?"
available_tokens = budget_for_rag(query, model="gpt-3.5-turbo")
print(f"Available tokens for retrieved documents: {available_tokens}")
# Output: Available tokens for retrieved documents: 3,376

# This means we can retrieve ~3,376 tokens worth of documents
# At ~500 tokens/chunk, that's 6-7 document chunks
```

---

## Part 3: Chunking Strategies

When document exceeds context limit, split into chunks and process separately.

### Fixed-Size Chunking

**Simple approach:** Split into equal-sized chunks.

```python
def chunk_by_tokens(text, chunk_size=1000, overlap=200, model="gpt-3.5-turbo"):
    """
    Split text into fixed-size token chunks with overlap.

    Args:
        text: Text to chunk
        chunk_size: Target tokens per chunk
        overlap: Overlapping tokens between chunks (for continuity)
        model: Model for tokenization

    Returns:
        List of text chunks
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        start += chunk_size - overlap  # Overlap for continuity

    return chunks

# Example
document = "Very long document with 10,000 tokens..." * 1000
chunks = chunk_by_tokens(document, chunk_size=1000, overlap=200)
print(f"Split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}: {count_tokens(chunk)} tokens")
```

**Pros:**
- Simple, predictable chunk sizes
- Works for any text

**Cons:**
- May split mid-sentence, mid-paragraph (poor semantic boundaries)
- Overlap creates redundancy
- No awareness of document structure

### Semantic Chunking

**Better approach:** Split at semantic boundaries (paragraphs, sections).

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_semantically(text, chunk_size=1000, overlap=200):
    """
    Split text at semantic boundaries (paragraphs, sentences).

    Uses LangChain's RecursiveCharacterTextSplitter which tries:
    1. Split by paragraphs (\n\n)
    2. If chunk still too large, split by sentences (. )
    3. If sentence still too large, split by words
    4. Last resort: split by characters
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,  # Approximate: 4 chars/token
        chunk_overlap=overlap * 4,
        separators=["\n\n", "\n", ". ", " ", ""],  # Priority order
        length_function=lambda text: count_tokens(text)  # Use actual token count
    )

    chunks = splitter.split_text(text)
    return chunks

# Example
document = """
# Introduction

This is the introduction to the document.
It contains several paragraphs of introductory material.

## Methods

The methods section describes the experimental procedure.
We used a randomized controlled trial with 100 participants.

## Results

The results show significant improvements in...
"""

chunks = chunk_semantically(document, chunk_size=500, overlap=50)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} ({count_tokens(chunk)} tokens):\n{chunk[:100]}...\n")
```

**Pros:**
- Respects semantic boundaries (complete paragraphs, sentences)
- Better context preservation
- More readable chunks

**Cons:**
- Chunk sizes vary (some may be too large)
- More complex implementation

### Hierarchical Chunking (Map-Reduce)

**Best for summarization:** Summarize chunks, then summarize summaries.

```python
def hierarchical_summarization(document, chunk_size=3000, model="gpt-3.5-turbo"):
    """
    Summarize long document using map-reduce approach.

    1. Split document into chunks (MAP)
    2. Summarize each chunk individually
    3. Combine chunk summaries (REDUCE)
    4. Generate final summary from combined summaries
    """
    import openai

    # Step 1: Chunk document
    chunks = chunk_semantically(document, chunk_size=chunk_size)
    print(f"Split into {len(chunks)} chunks")

    # Step 2: Summarize each chunk (MAP)
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Summarize the following text concisely."},
                {"role": "user", "content": chunk}
            ],
            temperature=0
        )
        summary = response.choices[0].message.content
        chunk_summaries.append(summary)
        print(f"Chunk {i+1} summary: {summary[:100]}...")

    # Step 3: Combine summaries (REDUCE)
    combined_summaries = "\n\n".join(chunk_summaries)

    # Step 4: Generate final summary
    final_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Synthesize the following summaries into a comprehensive final summary."},
            {"role": "user", "content": combined_summaries}
        ],
        temperature=0
    )

    final_summary = final_response.choices[0].message.content
    return final_summary

# Example
long_document = "Research paper with 50,000 tokens..." * 100
summary = hierarchical_summarization(long_document, chunk_size=3000)
print(f"Final summary:\n{summary}")
```

**Pros:**
- Handles arbitrarily long documents
- Preserves information across entire document
- Parallelizable (summarize chunks concurrently)

**Cons:**
- More API calls (higher cost)
- Information loss in successive summarizations
- Slower than single-pass

---

## Part 4: Intelligent Truncation Strategies

When chunking isn't appropriate (e.g., single-pass QA), truncate intelligently.

### Strategy 1: Truncate from Middle (Preserve Intro + Conclusion)

```python
def truncate_middle(text, max_tokens=3500, model="gpt-3.5-turbo"):
    """
    Keep beginning and end, truncate middle.

    Useful for documents with important intro (context) and conclusion (findings).
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text  # Fits, no truncation needed

    # Allocate: 40% beginning, 40% end, 20% lost in middle
    keep_start = int(max_tokens * 0.4)
    keep_end = int(max_tokens * 0.4)

    start_tokens = tokens[:keep_start]
    end_tokens = tokens[-keep_end:]

    # Add marker showing truncation
    truncation_marker = encoding.encode("\n\n[... middle section truncated ...]\n\n")

    truncated_tokens = start_tokens + truncation_marker + end_tokens
    return encoding.decode(truncated_tokens)

# Example
document = """
Introduction: This paper presents a new approach to X.
Our hypothesis is that Y improves performance by 30%.

[... 10,000 tokens of methods, experiments, detailed results ...]

Conclusion: We demonstrated that Y improves performance by 31%,
confirming our hypothesis. Future work will explore Z.
"""

truncated = truncate_middle(document, max_tokens=500)
print(truncated)
# Output:
# Introduction: This paper presents...
# [... middle section truncated ...]
# Conclusion: We demonstrated that Y improves...
```

### Strategy 2: Truncate from Beginning (Keep Recent Context)

```python
def truncate_from_start(text, max_tokens=3500, model="gpt-3.5-turbo"):
    """
    Keep end, discard beginning.

    Useful for logs, conversations where recent context is most important.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    # Keep last N tokens
    truncated_tokens = tokens[-max_tokens:]
    return encoding.decode(truncated_tokens)

# Example: Chat logs
conversation = """
[Turn 1 - 2 hours ago] User: How do I reset my password?
[Turn 2] Bot: Go to Settings > Security > Reset Password.
[... 50 turns ...]
[Turn 51 - just now] User: What was that password reset link again?
"""

truncated = truncate_from_start(conversation, max_tokens=200)
print(truncated)
# Output: [Turn 48] ... [Turn 51 - just now] User: What was that password reset link again?
```

### Strategy 3: Extractive Truncation (Keep Most Relevant)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extractive_truncation(document, query, max_tokens=3000, model="gpt-3.5-turbo"):
    """
    Keep sentences most relevant to query.

    Uses TF-IDF similarity to rank sentences by relevance to query.
    """
    # Split into sentences
    sentences = document.split('. ')

    # Calculate TF-IDF similarity to query
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query] + sentences)
    query_vec = vectors[0]
    sentence_vecs = vectors[1:]

    # Similarity scores
    similarities = cosine_similarity(query_vec, sentence_vecs)[0]

    # Rank sentences by similarity
    ranked_indices = np.argsort(similarities)[::-1]

    # Select sentences until token budget exhausted
    selected_sentences = []
    token_count = 0
    encoding = tiktoken.encoding_for_model(model)

    for idx in ranked_indices:
        sentence = sentences[idx] + '. '
        sentence_tokens = len(encoding.encode(sentence))

        if token_count + sentence_tokens <= max_tokens:
            selected_sentences.append((idx, sentence))
            token_count += sentence_tokens
        else:
            break

    # Sort selected sentences by original order (maintain flow)
    selected_sentences.sort(key=lambda x: x[0])

    return ''.join([sent for _, sent in selected_sentences])

# Example
document = """
The company was founded in 1995 in Seattle.
Our return policy allows returns within 30 days of purchase.
Products must be in original condition with tags attached.
Refunds are processed within 5-7 business days.
We offer free shipping on orders over $50.
The company has 500 employees worldwide.
"""

query = "What is the return policy?"

truncated = extractive_truncation(document, query, max_tokens=150)
print(truncated)
# Output: Our return policy allows returns within 30 days. Products must be in original condition. Refunds processed within 5-7 days.
```

---

## Part 5: Conversation Context Management

Multi-turn conversations require active context management to prevent unbounded growth.

### Strategy 1: Sliding Window

**Keep last N turns.**

```python
class SlidingWindowChatbot:
    def __init__(self, model="gpt-3.5-turbo", max_history=10):
        """
        Chatbot with sliding window context.

        Args:
            model: LLM model
            max_history: Maximum conversation turns to keep (user+assistant pairs)
        """
        self.model = model
        self.max_history = max_history
        self.system_message = {"role": "system", "content": "You are a helpful assistant."}
        self.messages = [self.system_message]

    def chat(self, user_message):
        """Add message, generate response, manage context."""
        import openai

        # Add user message
        self.messages.append({"role": "user", "content": user_message})

        # Apply sliding window (keep system + last N*2 messages)
        if len(self.messages) > (self.max_history * 2 + 1):  # +1 for system message
            self.messages = [self.system_message] + self.messages[-(self.max_history * 2):]

        # Generate response
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

# Example
bot = SlidingWindowChatbot(max_history=5)  # Keep last 5 turns

for turn in range(20):
    user_msg = input("You: ")
    response = bot.chat(user_msg)
    print(f"Bot: {response}")

    # Context automatically managed: always ≤ 11 messages (1 system + 5*2 user/assistant)
```

**Pros:**
- Simple, predictable
- Constant memory/cost
- Recent context preserved

**Cons:**
- Loses old context (user may reference earlier conversation)
- Fixed window may be too small or too large

### Strategy 2: Token-Based Truncation

**Keep messages until token budget exhausted.**

```python
class TokenBudgetChatbot:
    def __init__(self, model="gpt-3.5-turbo", max_tokens=3000):
        """
        Chatbot with token-based context management.

        Keeps messages until token budget exhausted (newest to oldest).
        """
        self.model = model
        self.max_tokens = max_tokens
        self.system_message = {"role": "system", "content": "You are a helpful assistant."}
        self.messages = [self.system_message]

    def chat(self, user_message):
        import openai

        # Add user message
        self.messages.append({"role": "user", "content": user_message})

        # Token management: keep system + recent messages within budget
        total_tokens = count_message_tokens(self.messages, self.model)

        while total_tokens > self.max_tokens and len(self.messages) > 2:
            # Remove oldest message (after system message)
            removed = self.messages.pop(1)
            total_tokens = count_message_tokens(self.messages, self.model)

        # Generate response
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

# Example
bot = TokenBudgetChatbot(max_tokens=2000)

for turn in range(20):
    user_msg = input("You: ")
    response = bot.chat(user_msg)
    print(f"Bot: {response}")
    print(f"Context tokens: {count_message_tokens(bot.messages)}")
```

**Pros:**
- Adaptive to message length (long messages = fewer kept, short messages = more kept)
- Precise budget control

**Cons:**
- Removes from beginning (loses early context)

### Strategy 3: Summarization + Sliding Window

**Best of both: Summarize old context, keep recent verbatim.**

```python
class SummarizingChatbot:
    def __init__(self, model="gpt-3.5-turbo", max_recent=5, summarize_threshold=10):
        """
        Chatbot with summarization + sliding window.

        When conversation exceeds threshold, summarize old turns and keep recent verbatim.

        Args:
            model: LLM model
            max_recent: Recent turns to keep verbatim
            summarize_threshold: Turns before summarizing old context
        """
        self.model = model
        self.max_recent = max_recent
        self.summarize_threshold = summarize_threshold
        self.system_message = {"role": "system", "content": "You are a helpful assistant."}
        self.messages = [self.system_message]
        self.summary = None  # Stores summary of old context

    def summarize_old_context(self):
        """Summarize older messages (beyond recent window)."""
        import openai

        # Messages to summarize: after system, before recent window
        num_messages = len(self.messages) - 1  # Exclude system message
        if num_messages <= self.summarize_threshold:
            return  # Not enough history yet

        # Split: old (to summarize) vs recent (keep verbatim)
        old_messages = self.messages[1:-(self.max_recent*2)]  # Exclude system + recent

        if not old_messages:
            return

        # Format for summarization
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in old_messages
        ])

        # Generate summary
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Summarize the following conversation concisely, capturing key information, user goals, and important context."},
                {"role": "user", "content": conversation_text}
            ],
            temperature=0
        )

        self.summary = response.choices[0].message.content

        # Update messages: system + summary + recent
        recent_messages = self.messages[-(self.max_recent*2):]
        summary_message = {
            "role": "system",
            "content": f"Previous conversation summary: {self.summary}"
        }

        self.messages = [self.system_message, summary_message] + recent_messages

    def chat(self, user_message):
        import openai

        # Add user message
        self.messages.append({"role": "user", "content": user_message})

        # Check if summarization needed
        num_turns = (len(self.messages) - 1) // 2  # Exclude system message
        if num_turns >= self.summarize_threshold:
            self.summarize_old_context()

        # Generate response
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

# Example
bot = SummarizingChatbot(max_recent=5, summarize_threshold=10)

# Long conversation
for turn in range(25):
    user_msg = input("You: ")
    response = bot.chat(user_msg)
    print(f"Bot: {response}")

    # After turn 10, old context (turns 1-5) summarized, turns 6-10+ kept verbatim
```

**Pros:**
- Preserves full conversation history (in summary form)
- Recent context verbatim (maintains fluency)
- Bounded token usage

**Cons:**
- Extra API call for summarization (cost)
- Information loss in summary
- More complex

---

## Part 6: RAG Context Management

RAG systems retrieve documents and include in context. Token budgeting is critical.

### Dynamic Document Retrieval (Budget-Aware)

```python
def retrieve_with_token_budget(
    query,
    documents,
    embeddings,
    model="gpt-3.5-turbo",
    output_tokens=500,
    max_docs=20
):
    """
    Retrieve documents dynamically based on token budget.

    Args:
        query: User query
        documents: List of document dicts [{"id": ..., "content": ...}, ...]
        embeddings: Pre-computed document embeddings
        model: LLM model
        output_tokens: Reserved for output
        max_docs: Maximum documents to consider

    Returns:
        Selected documents within token budget
    """
    from sentence_transformers import SentenceTransformer, util

    # Calculate available token budget
    available_tokens = budget_for_rag(query, model=model, output_tokens=output_tokens)

    # Retrieve top-k relevant documents (semantic search)
    query_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode(query)
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    top_indices = similarities.argsort(descending=True)[:max_docs]

    # Select documents until budget exhausted
    selected_docs = []
    token_count = 0

    for idx in top_indices:
        doc = documents[idx]
        doc_tokens = count_tokens(doc['content'], model)

        if token_count + doc_tokens <= available_tokens:
            selected_docs.append(doc)
            token_count += doc_tokens
        else:
            # Budget exhausted
            break

    return selected_docs, token_count

# Example
query = "What is our return policy?"
documents = [
    {"id": 1, "content": "Our return policy allows returns within 30 days..."},
    {"id": 2, "content": "Shipping is free on orders over $50..."},
    # ... 100 more documents
]

selected, tokens_used = retrieve_with_token_budget(
    query, documents, embeddings, model="gpt-3.5-turbo"
)

print(f"Selected {len(selected)} documents using {tokens_used} tokens")
# Output: Selected 7 documents using 3,280 tokens (within budget)
```

### Chunk Re-Ranking with Token Budget

```python
def rerank_and_budget(query, chunks, model="gpt-3.5-turbo", max_tokens=3000):
    """
    Over-retrieve, re-rank, then select top chunks within token budget.

    1. Retrieve k=20 candidates (coarse retrieval)
    2. Re-rank with cross-encoder (fine-grained scoring)
    3. Select top chunks until budget exhausted
    """
    from sentence_transformers import CrossEncoder

    # Re-rank with cross-encoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[query, chunk['content']] for chunk in chunks]
    scores = cross_encoder.predict(pairs)

    # Sort by relevance
    ranked_chunks = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Select until budget exhausted
    selected_chunks = []
    token_count = 0

    for chunk, score in ranked_chunks:
        chunk_tokens = count_tokens(chunk['content'], model)

        if token_count + chunk_tokens <= max_tokens:
            selected_chunks.append((chunk, score))
            token_count += chunk_tokens
        else:
            break

    return selected_chunks, token_count

# Example
chunks = [
    {"id": 1, "content": "Return policy: 30 days with receipt..."},
    {"id": 2, "content": "Shipping: Free over $50..."},
    # ... 18 more chunks
]

selected, tokens = rerank_and_budget(query, chunks, max_tokens=3000)
print(f"Selected {len(selected)} chunks, {tokens} tokens")
```

---

## Part 7: Cost and Performance Optimization

Context management affects cost and latency.

### Cost Optimization

```python
def calculate_cost(tokens, model="gpt-3.5-turbo"):
    """
    Calculate API cost based on token count.

    Pricing (as of 2024):
    - GPT-3.5-turbo: $0.002 per 1k tokens (input + output)
    - GPT-4: $0.03 per 1k input, $0.06 per 1k output
    - GPT-4-turbo: $0.01 per 1k input, $0.03 per 1k output
    """
    pricing = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    }

    rates = pricing.get(model, {"input": 0.002, "output": 0.002})
    input_cost = (tokens / 1000) * rates["input"]

    return input_cost

# Example: Cost comparison
conversation_tokens = 3500
print(f"GPT-3.5: ${calculate_cost(conversation_tokens, 'gpt-3.5-turbo'):.4f}")
print(f"GPT-4: ${calculate_cost(conversation_tokens, 'gpt-4'):.4f}")
# Output:
# GPT-3.5: $0.0053
# GPT-4: $0.1050 (20× more expensive!)
```

**Cost optimization strategies:**
1. **Compression:** Summarize old context (reduce tokens)
2. **Smaller model:** Use GPT-3.5 instead of GPT-4 when possible
3. **Efficient retrieval:** Retrieve fewer, more relevant docs
4. **Caching:** Cache embeddings, avoid re-encoding

### Latency Optimization

```python
# Latency increases with context length
import time

def measure_latency(context_tokens, model="gpt-3.5-turbo"):
    """
    Rough latency estimates (actual varies by API load).

    Latency = Fixed overhead + (tokens × per-token time)
    """
    fixed_overhead_ms = 500  # API call, network
    time_per_token_ms = {
        "gpt-3.5-turbo": 0.3,  # ~300ms per 1k tokens
        "gpt-4": 1.0,          # ~1s per 1k tokens (slower)
    }

    per_token = time_per_token_ms.get(model, 0.5)
    latency_ms = fixed_overhead_ms + (context_tokens * per_token)

    return latency_ms

# Example
for tokens in [500, 2000, 5000, 10000]:
    latency = measure_latency(tokens, "gpt-3.5-turbo")
    print(f"{tokens:,} tokens: {latency:.0f}ms ({latency/1000:.1f}s)")
# Output:
# 500 tokens: 650ms (0.7s)
# 2,000 tokens: 1,100ms (1.1s)
# 5,000 tokens: 2,000ms (2.0s)
# 10,000 tokens: 3,500ms (3.5s)
```

**Latency optimization strategies:**
1. **Reduce context:** Keep only essential information
2. **Parallel processing:** Process chunks concurrently (map-reduce)
3. **Streaming:** Stream responses for perceived latency reduction
4. **Caching:** Cache frequent queries

---

## Part 8: Complete Implementation Example

**RAG System with Full Context Management:**

```python
import openai
import tiktoken
from sentence_transformers import SentenceTransformer, util

class ManagedRAGSystem:
    def __init__(
        self,
        model="gpt-3.5-turbo",
        embedding_model="all-MiniLM-L6-v2",
        max_docs=20,
        output_tokens=500
    ):
        self.model = model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.max_docs = max_docs
        self.output_tokens = output_tokens

    def query(self, question, documents):
        """
        Query RAG system with full context management.

        Steps:
        1. Calculate token budget
        2. Retrieve relevant documents within budget
        3. Build context
        4. Generate response
        5. Return response with metadata
        """
        # Step 1: Calculate token budget
        system_message = "Answer the question using only the provided context."
        budget = calculate_token_budget(
            model=self.model,
            system_message_tokens=count_tokens(system_message),
            query_tokens=count_tokens(question),
            output_tokens=self.output_tokens
        )
        context_budget = budget['context_budget']

        # Step 2: Retrieve documents within budget
        query_embedding = self.embedding_model.encode(question)
        doc_embeddings = self.embedding_model.encode([doc['content'] for doc in documents])
        similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
        top_indices = similarities.argsort(descending=True)[:self.max_docs]

        selected_docs = []
        token_count = 0

        for idx in top_indices:
            doc = documents[idx]
            doc_tokens = count_tokens(doc['content'], self.model)

            if token_count + doc_tokens <= context_budget:
                selected_docs.append(doc)
                token_count += doc_tokens
            else:
                break

        # Step 3: Build context
        context = "\n\n".join([doc['content'] for doc in selected_docs])

        # Step 4: Generate response
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0
        )

        answer = response.choices[0].message.content

        # Step 5: Return with metadata
        return {
            'answer': answer,
            'num_docs_retrieved': len(selected_docs),
            'context_tokens': token_count,
            'total_tokens': response.usage.total_tokens,
            'cost': calculate_cost(response.usage.total_tokens, self.model)
        }

# Example usage
rag = ManagedRAGSystem(model="gpt-3.5-turbo")

documents = [
    {"id": 1, "content": "Our return policy allows returns within 30 days of purchase with receipt."},
    {"id": 2, "content": "Refunds are processed within 5-7 business days."},
    # ... more documents
]

result = rag.query("What is the return policy?", documents)

print(f"Answer: {result['answer']}")
print(f"Retrieved: {result['num_docs_retrieved']} documents")
print(f"Context tokens: {result['context_tokens']}")
print(f"Total tokens: {result['total_tokens']}")
print(f"Cost: ${result['cost']:.4f}")
```

---

## Summary

**Context window management is mandatory for production LLM systems.**

**Core strategies:**
1. **Token counting:** Always count tokens before API calls (tiktoken)
2. **Budgeting:** Allocate tokens to system, context, query, output
3. **Chunking:** Fixed-size, semantic, or hierarchical for long documents
4. **Truncation:** Middle-out, extractive, or structure-aware
5. **Conversation management:** Sliding window, token-based, or summarization
6. **RAG budgeting:** Dynamic retrieval, re-ranking with budget constraints

**Optimization:**
- Cost: Compression, smaller models, efficient retrieval
- Latency: Reduce context, parallel processing, streaming

**Implementation checklist:**
1. ✓ Count tokens with tiktoken (not character/word counts)
2. ✓ Check against model-specific limits
3. ✓ Allocate token budget for multi-component systems
4. ✓ Choose appropriate strategy (chunking, truncation, summarization)
5. ✓ Manage conversation context proactively
6. ✓ Monitor token usage, cost, and latency
7. ✓ Test with realistic data (long documents, long conversations)

Context is finite. Manage it deliberately, or face crashes, quality degradation, and cost overruns.
