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
grep -rn "completions\.create\|messages\.create\|models\.generate_content" --include="*.py" | grep -v "await"

# Missing caching
grep -rn "completions\.create\|messages\.create" --include="*.py"
# Check if any caching layer exists (provider prompt cache AND response cache)

# Hardcoded model IDs (should resolve through a tier/config layer)
grep -rnE "model\s*=\s*[\"'][a-zA-Z0-9._-]+[\"']" --include="*.py"
# For each hit: is a frontier tier doing work a fast-cheap tier could serve?

# Legacy SDK surface (openai<1.0 was removed in Nov 2023 — these raise APIRemovedInV1)
grep -rn "openai\.ChatCompletion\|openai\.Completion\|openai\.Batch\b" --include="*.py"

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

### Optimization 3: Tier Routing (order-of-magnitude cost reduction)

Route by *capability tier*, never by hardcoded model ID — provider lineups rotate
quarterly and pinned IDs get retired. See `llm-inference-optimization.md` (Part 3)
for the full router.

```python
import os

def route_to_tier(task_type: str) -> str:
    """Route to the cheapest capability tier that can handle the task."""

    # Simple tasks → fast-cheap tier
    if task_type in ('classification', 'extraction', 'summarization', 'translation'):
        return "fast-cheap"

    # Multi-step logic / math / planning → reasoning tier
    if task_type == 'reasoning':
        return "frontier-reasoning"

    # Other complex work → frontier-general
    if task_type in ('code_generation', 'creative'):
        return "frontier-general"

    return "fast-cheap"  # default to cheaper


# Resolve tier → current model ID through config, never inline.
MODEL_FOR_TIER = {
    "frontier-reasoning": os.getenv("MODEL_FRONTIER_REASONING"),
    "frontier-general":   os.getenv("MODEL_FRONTIER_GENERAL"),
    "fast-cheap":         os.getenv("MODEL_FAST_CHEAP"),
}

# Frontier-general input typically costs ~10-30× fast-cheap on the same provider;
# frontier-reasoning adds hidden thinking tokens on top. Verify current ratios on
# the provider's pricing page — they move quarterly.
# If 80% of tasks route to fast-cheap → ~80% cost reduction.
```

### Optimization 4: Streaming (Better UX)

```python
from openai import OpenAI

client = OpenAI()

def stream_response(prompt: str, model: str):
    """Stream tokens as they're generated (openai>=1.0 client)."""
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:          # delta is a pydantic model, not a dict
            yield delta.content

# Note: openai.ChatCompletion.create is the pre-1.0 form, removed Nov 2023 —
# calling it raises APIRemovedInV1.
#
# Without streaming: User waits 20s, sees nothing
# With streaming: First token in 0.5s, continuous output
```

### Optimization 5: Batch API (~50% cost reduction for offline)

```python
# For non-real-time workloads (bulk processing, offline analysis).
# OpenAI Batch API: ~50% discount vs synchronous, 24h completion window.

batch_input = client.files.create(file=open("requests.jsonl", "rb"), purpose="batch")

batch_job = client.batches.create(
    input_file_id=batch_input.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

# Real-time: 1× input price
# Batch API: ~0.5× input price
# (openai.Batch.create is the removed pre-1.0 form.)
```

## Phase 4: Evaluate Trade-offs

Use Pareto analysis to find optimal configuration:

| Configuration | Latency P95 | Relative cost/1k | Quality |
|---------------|-------------|------------------|---------|
| frontier-general, no cache | 2.5s | 20× | 0.95 |
| fast-cheap, no cache | 0.8s | 1× (baseline) | 0.85 |
| fast-cheap + response cache | 0.1s | 0.4× | 0.85 |
| fast-cheap + cache + tier routing | 0.2s | 0.27× | 0.88 |

Costs are expressed *relative to the fast-cheap baseline*, not in dollars —
absolute prices go stale within a quarter. The ~20× frontier-vs-fast-cheap spread
is the order of magnitude to expect; measure your own workload.

**Selection criteria:**
- Latency-critical: fast-cheap + cache
- Quality-critical: frontier tier + cache
- Cost-critical: fast-cheap + cache + tier routing + batch

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
pytorch_pack = glob.glob("plugins/yzmir-pytorch-engineering/.claude-plugin/plugin.json")
if not pytorch_pack:
    print("Recommend: yzmir-pytorch-engineering for model-level profiling")
```
