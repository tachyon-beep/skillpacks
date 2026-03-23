---
description: Systematic LLM inference optimization - caching, parallelization, model routing
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write"]
argument-hint: "[target_file_or_description]"
---

# LLM Inference Optimization Command

You are optimizing LLM inference for production. Follow the systematic optimization framework.

## Core Principle

**Measure before optimizing. Profile, don't guess.** The bottleneck is rarely where you think.

## Optimization Framework

```
1. Measure Baseline → 2. Set Requirements → 3. Apply Optimizations → 4. Evaluate Trade-offs → 5. Monitor
```

## Phase 1: Measure Baseline

Before any optimization, establish reproducible metrics:

```python
import time
import statistics

def measure_baseline(llm_func, test_queries, num_runs=10):
    """Establish baseline latency, cost, and throughput."""
    latencies = []

    for query in test_queries[:num_runs]:
        start = time.perf_counter()
        response = llm_func(query)
        latencies.append(time.perf_counter() - start)

    return {
        'latency_p50': statistics.median(latencies),
        'latency_p95': sorted(latencies)[int(0.95 * len(latencies))],
        'latency_mean': statistics.mean(latencies),
        'throughput': 1 / statistics.mean(latencies),  # queries/sec
    }

# Run baseline
baseline = measure_baseline(your_llm_call, test_queries)
print(f"Baseline P50: {baseline['latency_p50']*1000:.0f}ms")
print(f"Baseline P95: {baseline['latency_p95']*1000:.0f}ms")
print(f"Throughput: {baseline['throughput']:.2f} queries/sec")
```

## Phase 2: Identify Optimization Opportunities

Search the codebase for optimization opportunities:

```bash
# Sequential API calls (parallelize!)
grep -rn "openai\." --include="*.py" | grep -v "async"

# Missing caching
grep -rn "ChatCompletion.create\|messages.create" --include="*.py"
# Check if any caching layer exists

# Model usage (could use cheaper models?)
grep -rn "gpt-4\|claude-3-opus" --include="*.py"
# Evaluate if GPT-3.5/Haiku would suffice

# Streaming disabled
grep -rn "stream=False\|stream.*=.*False" --include="*.py"
```

## Phase 3: Apply Optimizations

### Optimization 1: Parallelization (10× throughput, FREE)

```python
import asyncio

async def parallel_llm_calls(queries, concurrency=10):
    """Process queries in parallel with rate limiting."""
    semaphore = asyncio.Semaphore(concurrency)

    async def call_with_limit(query):
        async with semaphore:
            return await async_llm_call(query)

    tasks = [call_with_limit(q) for q in queries]
    return await asyncio.gather(*tasks)

# Before: 100 queries × 1s = 100s
# After:  100 queries / 10 concurrent = 10s (10× faster, same cost!)
```

### Optimization 2: Caching (60%+ cost reduction)

```python
import hashlib
from functools import lru_cache

class LLMCache:
    def __init__(self):
        self.cache = {}  # Use Redis in production

    def _key(self, prompt, model):
        return hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()

    def get_or_call(self, prompt, model, llm_func):
        key = self._key(prompt, model)

        if key in self.cache:
            return self.cache[key], True  # cache hit

        result = llm_func(prompt, model)
        self.cache[key] = result
        return result, False  # cache miss

# 60-70% of queries are repeated (FAQs, common questions)
# Cache hit = $0 cost, <10ms latency
```

### Optimization 3: Model Routing (10× cost reduction)

```python
def route_to_model(query, task_type):
    """Route to cheapest model that can handle the task."""

    # Simple tasks → cheap models
    simple_tasks = ['classification', 'extraction', 'summarization', 'translation']
    if task_type in simple_tasks:
        return 'gpt-3.5-turbo'  # or claude-3-haiku

    # Complex tasks → capable models
    complex_tasks = ['reasoning', 'code_generation', 'creative']
    if task_type in complex_tasks:
        return 'gpt-4-turbo'  # or claude-3-sonnet

    return 'gpt-3.5-turbo'  # default to cheaper

# GPT-4: $30/1M tokens vs GPT-3.5: $1.50/1M = 20× cheaper
# 80% of tasks can use cheaper model → 80% cost savings
```

### Optimization 4: Streaming (Better UX)

```python
def stream_response(prompt, model="gpt-4"):
    """Stream tokens as they're generated."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.get("content"):
            yield chunk.choices[0].delta.content

# Without streaming: User waits 20s, sees nothing
# With streaming: First token in 0.5s, continuous output
# Bounce rate: 40% → 5%
```

### Optimization 5: Batch API (50% cost reduction for offline)

```python
# For non-real-time workloads (batch processing, analysis)
# OpenAI Batch API: 50% discount, 24h completion window

batch_job = openai.Batch.create(
    input_file_id=uploaded_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# Real-time: $10 for 1M tokens
# Batch API: $5 for 1M tokens (50% savings!)
```

## Phase 4: Evaluate Trade-offs

Use Pareto analysis to find optimal configuration:

| Configuration | Latency P95 | Cost/1k | Quality |
|---------------|-------------|---------|---------|
| GPT-4, no cache | 2.5s | $30 | 0.95 |
| GPT-3.5, no cache | 0.8s | $1.50 | 0.85 |
| GPT-3.5 + cache | 0.1s | $0.60 | 0.85 |
| GPT-3.5 + cache + routing | 0.2s | $0.40 | 0.88 |

**Selection criteria:**
- Latency-critical: GPT-3.5 + cache
- Quality-critical: GPT-4 + cache
- Cost-critical: GPT-3.5 + cache + routing + batch

## Phase 5: Monitor Production

Track key metrics continuously:

```python
def log_llm_call(model, latency_ms, input_tokens, output_tokens, cache_hit):
    """Log metrics for monitoring."""
    metrics = {
        'model': model,
        'latency_ms': latency_ms,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'cost': calculate_cost(model, input_tokens, output_tokens),
        'cache_hit': cache_hit,
        'timestamp': datetime.now()
    }
    # Send to monitoring system (Datadog, Prometheus, etc.)
```

**Alert thresholds:**
- P95 latency > 2× baseline
- Cache hit rate < 50%
- Error rate > 1%
- Cost per query > 2× budget

## Optimization Checklist

After analysis, provide:

1. **Baseline Metrics**: Current latency, throughput, cost
2. **Identified Opportunities**: Which optimizations apply
3. **Recommended Changes**: Specific code modifications
4. **Expected Impact**: Projected latency/cost/quality changes
5. **Monitoring Setup**: How to track improvements

## Cross-Pack Discovery

For PyTorch/model-level optimization:

```python
import glob
pytorch_pack = glob.glob("plugins/yzmir-pytorch-engineering/plugin.json")
if not pytorch_pack:
    print("Recommend: yzmir-pytorch-engineering for model-level profiling")
```
