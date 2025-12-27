---
description: Diagnose poor LLM output quality - prompts, fine-tuning decisions, RAG issues
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[symptom_or_file]"
---

# LLM Generation Debugging Command

You are diagnosing poor LLM output quality. Follow the systematic decision tree.

## Core Principle

**Start with prompts. Fine-tuning is last resort.** 90% of quality issues are fixable with better prompts.

## Symptom Triage

### Step 1: Identify the Symptom

| Symptom | Likely Cause | First Action |
|---------|--------------|--------------|
| Inconsistent outputs | Temperature too high, vague instructions | Check temperature, add examples |
| Wrong format | Missing format specification | Add explicit format in prompt |
| Hallucinations | No grounding, missing context | Add RAG or fact-checking |
| Ignoring instructions | Prompt too long, buried instructions | Move key instructions to start |
| Wrong tone/style | Missing style guidance | Add style examples |
| Factually incorrect | Knowledge cutoff, domain mismatch | Add RAG or fine-tune |
| Too verbose/brief | No length guidance | Specify length constraints |
| Repeating itself | Temperature too low, context issues | Adjust temperature, check context |

### Step 2: Check Prompt Engineering Basics

Search for common prompt issues:

```bash
# Check for system messages
grep -rn "role.*system" --include="*.py"
# Missing system message = inconsistent behavior

# Check for few-shot examples
grep -rn "examples\|few.shot" --include="*.py"
# Missing examples = model guesses format

# Check temperature settings
grep -rn "temperature" --include="*.py"
# temperature=1 = high variance, temperature=0 = deterministic

# Check for explicit format instructions
grep -rn "JSON\|format\|structure" --include="*.py" -A2 -B2
```

## Diagnosis Decision Tree

```
Quality Issue Detected
        │
        ▼
┌─────────────────────────────────────┐
│ 1. Is there a clear system message? │
│    (role, guidelines, constraints)  │
└───────────────┬─────────────────────┘
                │
        No ─────┼───── Yes
        │       │       │
        ▼       │       ▼
   Add system   │  ┌─────────────────────────────────────┐
   message      │  │ 2. Are there few-shot examples?     │
                │  │    (3-5 input→output pairs)         │
                │  └───────────────┬─────────────────────┘
                │                  │
                │          No ─────┼───── Yes
                │          │       │       │
                │          ▼       │       ▼
                │     Add 3-5      │  ┌─────────────────────────────────────┐
                │     examples     │  │ 3. Is output format specified?      │
                │                  │  │    (JSON schema, markdown, etc.)    │
                │                  │  └───────────────┬─────────────────────┘
                │                  │                  │
                │                  │          No ─────┼───── Yes
                │                  │          │       │       │
                │                  │          ▼       │       ▼
                │                  │     Add format   │  ┌─────────────────────────────────────┐
                │                  │     spec         │  │ 4. Is temperature appropriate?      │
                │                  │                  │  │    (0 for factual, 0.7 for creative)│
                │                  │                  │  └───────────────┬─────────────────────┘
                │                  │                  │                  │
                │                  │                  │          No ─────┼───── Yes
                │                  │                  │          │       │       │
                │                  │                  │          ▼       │       ▼
                │                  │                  │     Adjust       │  Still failing?
                │                  │                  │     temperature  │  Consider RAG or
                │                  │                  │                  │  fine-tuning
```

## Fix Patterns

### Fix 1: Add Clear System Message

```python
# BEFORE (vague)
messages = [{"role": "user", "content": query}]

# AFTER (clear)
messages = [
    {"role": "system", "content": """You are a customer support agent for TechCorp.

Guidelines:
- Be helpful, professional, and concise
- If you don't know something, say so
- Never make up product information
- Always end with "Is there anything else I can help with?"

Response format: 1-2 paragraphs, friendly but professional tone."""},
    {"role": "user", "content": query}
]
```

### Fix 2: Add Few-Shot Examples

```python
# BEFORE (no examples)
prompt = f"Classify this review: {review}"

# AFTER (with examples)
prompt = f"""Classify the sentiment of product reviews.

Examples:
Review: "This product exceeded my expectations!"
Sentiment: positive

Review: "Broke after one week. Waste of money."
Sentiment: negative

Review: "It's okay, nothing special."
Sentiment: neutral

Now classify:
Review: "{review}"
Sentiment:"""
```

### Fix 3: Specify Output Format

```python
# BEFORE (ambiguous)
prompt = "Extract the key information from this document."

# AFTER (explicit format)
prompt = """Extract key information from this document.

Output as JSON with exactly these fields:
{
    "title": "document title",
    "date": "YYYY-MM-DD format",
    "summary": "2-3 sentence summary",
    "key_points": ["point 1", "point 2", "point 3"],
    "action_items": ["action 1", "action 2"]
}

Document:
{document}

JSON output:"""
```

### Fix 4: Adjust Temperature

```python
# For factual/consistent outputs
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    temperature=0,  # Deterministic
)

# For creative/varied outputs
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    temperature=0.7,  # Some creativity
)

# Temperature guide:
# 0.0 = Always same output (facts, extraction, classification)
# 0.3 = Slight variation (summarization, Q&A)
# 0.7 = Moderate variation (creative writing, brainstorming)
# 1.0 = High variation (poetry, stories)
```

### Fix 5: Add Chain-of-Thought

```python
# BEFORE (direct answer)
prompt = f"What is the answer to: {complex_question}"

# AFTER (reasoning first)
prompt = f"""Solve this step by step:

Question: {complex_question}

Think through this carefully:
1. First, identify the key information
2. Then, consider what approach to use
3. Work through the solution step by step
4. Finally, state the answer

Solution:"""
```

## When to Consider Fine-Tuning

Only consider fine-tuning if ALL of these are true:

1. ✅ Prompts fail (quality < 90% after optimization)
2. ✅ Have 1000+ high-quality examples
3. ✅ Need consistency that prompts can't provide
4. ✅ Task requires domain knowledge not in base model
5. ✅ Have budget for training ($100-1000+)

**Do NOT fine-tune for:**
- ❌ Tone/style matching (use system message)
- ❌ Output formatting (use format specification)
- ❌ Few examples (< 100 insufficient)
- ❌ Quick experiments (prompts iterate faster)
- ❌ Recent information (use RAG instead)

## When to Add RAG

Consider RAG if:
- Model lacks specific knowledge (company docs, recent data)
- Hallucinations on factual questions
- Need citations/sources
- Information changes frequently

```python
# Simple RAG check
if needs_external_knowledge(query):
    context = retrieve_relevant_docs(query)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context:"
```

## Output Format

After diagnosis, provide:

1. **Root Cause**: Which prompt element was missing/wrong
2. **Specific Fix**: Exact code change needed
3. **Before/After Examples**: Show improvement
4. **Verification**: How to test the fix worked
5. **Monitoring**: How to track quality going forward
