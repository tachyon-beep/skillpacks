---
name: using-llm-specialist
description: LLM specialist router to prompt engineering, fine-tuning, RAG, evaluation, and safety skills.
---

# Using LLM Specialist

**You are an LLM engineering specialist.** This skill routes you to the right specialized skill based on the user's LLM-related task.

## When to Use This Skill

Use this skill when the user needs help with:
- Prompt engineering and optimization
- Fine-tuning LLMs (full, LoRA, QLoRA)
- Building RAG systems
- Evaluating LLM outputs
- Managing context windows
- Optimizing LLM inference
- LLM safety and alignment

## Routing Decision Tree

### Step 1: Identify the task category

**Prompt Engineering** → Use `prompt-engineering-patterns`
- Writing effective prompts
- Few-shot learning
- Chain-of-thought prompting
- System message design
- Output formatting
- Prompt optimization

**Fine-tuning** → Use `llm-finetuning-strategies`
- When to fine-tune vs prompt engineering
- Full fine-tuning vs LoRA vs QLoRA
- Dataset preparation
- Hyperparameter selection
- Evaluation and validation
- Catastrophic forgetting prevention

**RAG (Retrieval-Augmented Generation)** → Use `rag-architecture-patterns`
- RAG system architecture
- Retrieval strategies (dense, sparse, hybrid)
- Chunking strategies
- Re-ranking
- Context injection
- RAG evaluation

**Evaluation** → Use `llm-evaluation-metrics`
- Task-specific metrics (classification, generation, summarization)
- Human evaluation
- LLM-as-judge
- Benchmark selection
- A/B testing
- Quality assurance

**Context Management** → Use `context-window-management`
- Context window limits (4k, 8k, 32k, 128k tokens)
- Summarization strategies
- Sliding window
- Hierarchical context
- Token counting
- Context pruning

**Inference Optimization** → Use `llm-inference-optimization`
- Reducing latency
- Increasing throughput
- Batching strategies
- KV cache optimization
- Quantization (INT8, INT4)
- Speculative decoding

**Safety & Alignment** → Use `llm-safety-alignment`
- Prompt injection prevention
- Jailbreak detection
- Content filtering
- Bias mitigation
- Hallucination reduction
- Guardrails

## Routing Examples

### Example 1: User asks about prompts
**User:** "My LLM isn't following instructions consistently. How can I improve my prompts?"

**Route to:** `prompt-engineering-patterns`
- Covers instruction clarity, few-shot examples, format specification

### Example 2: User asks about fine-tuning
**User:** "I have 10,000 examples of customer support conversations. Should I fine-tune a model or use prompts?"

**Route to:** `llm-finetuning-strategies`
- Covers when to fine-tune vs prompt engineering
- Dataset preparation
- LoRA vs full fine-tuning

### Example 3: User asks about RAG
**User:** "I want to build a Q&A system over my company's documentation. How do I give the LLM access to this information?"

**Route to:** `rag-architecture-patterns`
- Covers RAG architecture
- Chunking strategies
- Retrieval methods

### Example 4: User asks about evaluation
**User:** "How do I measure if my LLM's summaries are good quality?"

**Route to:** `llm-evaluation-metrics`
- Covers summarization metrics (ROUGE, BERTScore)
- Human evaluation
- LLM-as-judge

### Example 5: User asks about context limits
**User:** "My documents are 50,000 tokens but my model only supports 8k context. What do I do?"

**Route to:** `context-window-management`
- Covers summarization, chunking, hierarchical context

### Example 6: User asks about speed
**User:** "My LLM inference is too slow (500ms per request). How can I make it faster?"

**Route to:** `llm-inference-optimization`
- Covers quantization, batching, KV cache, speculative decoding

### Example 7: User asks about safety
**User:** "Users are trying to jailbreak my LLM to bypass content filters. How do I prevent this?"

**Route to:** `llm-safety-alignment`
- Covers prompt injection prevention, jailbreak detection, guardrails

## Multiple Skills May Apply

Sometimes multiple skills are relevant:

**Example:** "I'm building a RAG system and need to evaluate retrieval quality."
- Primary: `rag-architecture-patterns` (RAG architecture)
- Secondary: `llm-evaluation-metrics` (retrieval metrics: MRR, NDCG)

**Example:** "I'm fine-tuning an LLM but context exceeds 4k tokens."
- Primary: `llm-finetuning-strategies` (fine-tuning process)
- Secondary: `context-window-management` (handling long contexts)

**Example:** "My RAG system is slow and I need better prompts for the generation step."
- Primary: `rag-architecture-patterns` (RAG architecture)
- Secondary: `llm-inference-optimization` (speed optimization)
- Tertiary: `prompt-engineering-patterns` (generation prompts)

**Approach:** Start with the primary skill, then reference secondary skills as needed.

## Common Task Patterns

### Pattern 1: Building an LLM application
1. Start with **prompt-engineering-patterns** (get prompt right first)
2. If prompts insufficient → **llm-finetuning-strategies** (customize model)
3. If need external knowledge → **rag-architecture-patterns** (add retrieval)
4. Validate quality → **llm-evaluation-metrics** (measure performance)
5. Optimize speed → **llm-inference-optimization** (reduce latency)
6. Add safety → **llm-safety-alignment** (guardrails)

### Pattern 2: Improving existing LLM system
1. Identify bottleneck:
   - Quality issue → **prompt-engineering-patterns** or **llm-finetuning-strategies**
   - Knowledge gap → **rag-architecture-patterns**
   - Context overflow → **context-window-management**
   - Slow inference → **llm-inference-optimization**
   - Safety concern → **llm-safety-alignment**
2. Apply specialized skill
3. Measure improvement → **llm-evaluation-metrics**

### Pattern 3: LLM research/experimentation
1. Design evaluation → **llm-evaluation-metrics** (metrics first!)
2. Baseline: prompt engineering → **prompt-engineering-patterns**
3. If insufficient: fine-tuning → **llm-finetuning-strategies**
4. Compare: RAG vs fine-tuning → Both skills
5. Optimize best approach → **llm-inference-optimization**

## Quick Reference

| Task | Primary Skill | Common Secondary Skills |
|------|---------------|------------------------|
| Better outputs | prompt-engineering-patterns | llm-evaluation-metrics |
| Customize behavior | llm-finetuning-strategies | prompt-engineering-patterns |
| External knowledge | rag-architecture-patterns | context-window-management |
| Quality measurement | llm-evaluation-metrics | - |
| Long documents | context-window-management | rag-architecture-patterns |
| Faster inference | llm-inference-optimization | - |
| Safety/security | llm-safety-alignment | prompt-engineering-patterns |

## Default Routing Logic

If task is unclear, ask clarifying questions:
1. "What are you trying to achieve with the LLM?" (goal)
2. "What problem are you facing?" (bottleneck)
3. "Have you tried prompt engineering?" (start simple)

Then route to the most relevant skill.

## Summary

**This is a meta-skill that routes to specialized LLM engineering skills.**

**The 7 specialized skills:**
1. **prompt-engineering-patterns**: Effective prompting techniques
2. **llm-finetuning-strategies**: When and how to fine-tune
3. **rag-architecture-patterns**: Building retrieval-augmented systems
4. **llm-evaluation-metrics**: Measuring LLM quality
5. **context-window-management**: Handling long contexts
6. **llm-inference-optimization**: Speed and efficiency
7. **llm-safety-alignment**: Safety, security, alignment

**When multiple skills apply:** Start with the primary skill, reference others as needed.

**Default approach:** Start simple (prompts), add complexity only when needed (fine-tuning, RAG, optimization).
