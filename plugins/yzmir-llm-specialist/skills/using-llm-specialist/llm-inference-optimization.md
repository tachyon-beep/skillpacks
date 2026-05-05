
# LLM Inference Optimization Skill

## When to Use This Skill

Use this skill when:
- Building production LLM applications with latency requirements
- Processing large batches of requests (classification, summarization, extraction)
- Optimizing cost for high-volume applications
- Improving throughput for batch processing
- Enhancing user experience with streaming
- Balancing cost, latency, and quality trade-offs
- Choosing or tuning a self-hosted serving stack

**When NOT to use:** Prototyping or single-query experiments where optimization is premature.

## Core Principle

**Performance is not automatic. Optimization is systematic.**

Without optimization:
- Sequential processing leaves throughput on the table
- No caching wastes spend on repeated queries
- Wrong tier of model is 5-20× too expensive for the same quality
- No streaming hurts perceived latency on long generations
- Single-objective tuning produces poor cost-latency-quality trade-offs

**Formula:** Parallelization + Caching (answer + prompt) + Capability-tier routing + Streaming + Right serving stack + Multi-objective trade-off analysis = Production-grade performance.

## Capability Tiers (Vocabulary)

This sheet, and the sister sheets in this pack, refer to four capability tiers rather than specific model IDs:

| Tier | Profile | Typical use |
|------|---------|-------------|
| **frontier-reasoning** | Deep multi-step reasoning, tool use, agentic planning. Highest cost per token, highest output-token consumption (extended thinking). | Math/science, complex coding, multi-step agents, hard refactors. See `reasoning-models.md`. |
| **frontier-general** | Strong general intelligence with 128k-200k context as a baseline. Mid-high cost. | Most production chat/RAG/extraction; default for "quality matters". |
| **fast-cheap** | Smaller frontier-class models tuned for low latency / low cost. | Classification, extraction, routing, summarization, high-QPS chat. |
| **on-device** | Open-weights models (~1-30B) running locally or behind your own GPU/CPU stack. | Privacy, air-gapped, edge, cost floor for very high volume. |

**Never hardcode model IDs in long-lived code.** Provider lineups change quarterly. Look up the current ID for each tier from the provider's docs (Anthropic model card, OpenAI models page, Google AI Studio, Meta/Mistral/Qwen releases) and inject via config.

## Optimization Framework

```
┌─────────────────────────────────────────┐
│      1. Measure Baseline                │
│  Latency, Cost, Quality, Throughput     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      2. Set Requirements                │
│  Acceptable latency, Budget, Quality    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      3. Apply Optimizations             │
│  Parallel → Cache → Route → Serve       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      4. Evaluate Trade-offs             │
│  Cost vs Latency vs Quality (Pareto)    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      5. Monitor Production              │
│  Track metrics, Detect regressions      │
└─────────────────────────────────────────┘
```

## Part 1: Parallelization

### Async/Await for Concurrent Requests

**Problem:** Sequential API calls underutilize the network and the provider's batched scheduler.

**Solution:** Concurrent requests with async/await + a semaphore for rate limiting.

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def classify_async(text: str, semaphore: asyncio.Semaphore, model: str) -> str:
    """Classify text asynchronously with rate limiting (openai>=1.0 SDK)."""
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,  # Inject from config: a fast-cheap tier model
            messages=[
                {"role": "system", "content": "Classify sentiment: positive/negative/neutral"},
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content

async def classify_batch_parallel(texts: list[str], model: str, concurrency: int = 10) -> list[str]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [classify_async(t, semaphore, model) for t in texts]
    return await asyncio.gather(*tasks)
```

Concurrency turns wall-clock latency into throughput at the same per-request cost. Tune the semaphore against the provider's published RPM/TPM limits and back off on 429s.

### Provider Batch APIs (Offline)

For offline workloads (nightly classification, eval grading, embeddings backfill), use the provider's Batch API. Both OpenAI and Anthropic offer ~50% discount on batch with 24-hour SLAs (verify current discount in provider docs; the headline number is the right order of magnitude but providers do change it).

```python
# OpenAI Batch (openai>=1.0)
from openai import OpenAI
import json

client = OpenAI()

# Build a JSONL file of requests
with open("batch_input.jsonl", "w") as f:
    for i, text in enumerate(texts):
        f.write(json.dumps({
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_id_from_config,  # capability-tier lookup
                "messages": [
                    {"role": "system", "content": "Classify sentiment."},
                    {"role": "user", "content": text},
                ],
            },
        }) + "\n")

batch_file = client.files.create(file=open("batch_input.jsonl", "rb"), purpose="batch")
batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)
```

| Use Case | Real-time | Batch |
|----------|-----------|-------|
| User-facing chat | Yes | No |
| Document classification (10k+ docs) | No (expensive) | Yes (50% off) |
| Nightly eval / grading | No | Yes |
| Embeddings backfill | No | Yes |

## Part 2: Caching

### Answer Caching (Repeated Queries)

For FAQs and templated queries, a simple key-value cache eliminates repeat work.

```python
import hashlib
from typing import Optional

class AnswerCache:
    def __init__(self):
        self.cache: dict[str, str] = {}

    def _key(self, query: str, model: str) -> str:
        return hashlib.sha256(f"{model}:{query.lower().strip()}".encode()).hexdigest()

    def get(self, query: str, model: str) -> Optional[str]:
        return self.cache.get(self._key(query, model))

    def set(self, query: str, answer: str, model: str) -> None:
        self.cache[self._key(query, model)] = answer
```

In production, back this with Redis (or a CDN edge cache) and add a TTL. Hit rates of 30-70% are typical for support bots and templated assistants.

### Prompt Caching (Static Context)

Most providers now offer prompt caching: a marked prefix is stored server-side and reused at a fraction of the input price on subsequent calls. **Cache reads are typically ~10% of the normal input price** (Anthropic's published rate at time of writing); cache writes are typically modestly more expensive than uncached input. Check the provider's pricing page for the current multiplier.

```python
import anthropic

client = anthropic.Anthropic()

def rag_with_prompt_caching(query: str, context: str, model: str) -> str:
    """Mark static context as cacheable. First call writes the cache; subsequent
    calls within the cache TTL read at the discounted rate."""
    response = client.messages.create(
        model=model,  # frontier-general or frontier-reasoning per task
        max_tokens=500,
        system=[
            {"type": "text", "text": "Answer questions using only the provided context."},
            {
                "type": "text",
                "text": f"Context:\n{context}",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[{"role": "user", "content": query}],
    )
    return response.content[0].text
```

When prompt caching pays:

| Scenario | What's cacheable | Effective discount on prefix |
|----------|------------------|------------------------------|
| RAG with stable knowledge base | 50k+ tokens of policies/products | Large — most input is cached |
| Multi-turn agent with long system prompt + tools | 5-20k tokens of instructions/tool defs | Large after first turn |
| Document Q&A | 10k+ token document | Large across multiple questions |
| One-shot, no repetition | None | Caching is net-negative (write cost without read benefit) |

**Cross-ref:** `context-engineering-and-prompt-caching.md` covers cache-key design, TTL selection, cache-segment ordering, breakpoint placement, and cross-provider differences in depth.

## Part 3: Capability-Tier Routing

### Task-Based Tier Selection

**Problem:** Using a frontier-reasoning model for simple classification is 10-30× the cost for indistinguishable quality.

**Solution:** Route by task type, complexity, and reasoning need to a capability tier; resolve the tier to a current model ID via config.

```python
from enum import Enum
from dataclasses import dataclass

class TaskType(Enum):
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    REASONING = "reasoning"           # multi-step logic, math, planning
    CREATIVE = "creative"
    CODE_GENERATION = "code_generation"
    AGENTIC = "agentic"               # tool use, multi-turn planning

class Tier(Enum):
    FRONTIER_REASONING = "frontier-reasoning"
    FRONTIER_GENERAL = "frontier-general"
    FAST_CHEAP = "fast-cheap"
    ON_DEVICE = "on-device"

@dataclass(frozen=True)
class RouteRequest:
    task: TaskType
    complexity: str = "medium"   # "low" | "medium" | "high"
    latency_budget_ms: int = 5000
    output_token_budget: int = 500
    needs_reasoning: bool = False
    privacy_sensitive: bool = False

class TierRouter:
    """Resolve a task to a capability tier. Model IDs come from config, not code."""

    DEFAULT_TIER: dict[TaskType, Tier] = {
        TaskType.CLASSIFICATION: Tier.FAST_CHEAP,
        TaskType.EXTRACTION: Tier.FAST_CHEAP,
        TaskType.SUMMARIZATION: Tier.FAST_CHEAP,
        TaskType.TRANSLATION: Tier.FAST_CHEAP,
        TaskType.CREATIVE: Tier.FRONTIER_GENERAL,
        TaskType.CODE_GENERATION: Tier.FRONTIER_GENERAL,
        TaskType.REASONING: Tier.FRONTIER_REASONING,
        TaskType.AGENTIC: Tier.FRONTIER_GENERAL,
    }

    @classmethod
    def route(cls, req: RouteRequest) -> Tier:
        # Privacy override: pin to local
        if req.privacy_sensitive:
            return Tier.ON_DEVICE

        tier = cls.DEFAULT_TIER[req.task]

        # Escalate on high complexity or explicit reasoning need
        if req.needs_reasoning or req.complexity == "high":
            if tier == Tier.FAST_CHEAP:
                tier = Tier.FRONTIER_GENERAL
            elif tier == Tier.FRONTIER_GENERAL and req.needs_reasoning:
                tier = Tier.FRONTIER_REASONING

        # Latency override: reasoning tier blows latency budgets due to thinking tokens
        if req.latency_budget_ms < 2000 and tier == Tier.FRONTIER_REASONING:
            tier = Tier.FRONTIER_GENERAL  # Accept some quality loss for SLA

        # Output volume override: huge outputs on the reasoning tier are expensive;
        # prefer general unless reasoning is genuinely required
        if req.output_token_budget > 4000 and tier == Tier.FRONTIER_REASONING and not req.needs_reasoning:
            tier = Tier.FRONTIER_GENERAL

        return tier

# Usage: resolve to a current model ID via your config layer
MODEL_FOR_TIER = {
    Tier.FRONTIER_REASONING: os.getenv("MODEL_FRONTIER_REASONING"),
    Tier.FRONTIER_GENERAL: os.getenv("MODEL_FRONTIER_GENERAL"),
    Tier.FAST_CHEAP: os.getenv("MODEL_FAST_CHEAP"),
    Tier.ON_DEVICE: os.getenv("MODEL_ON_DEVICE"),
}
```

Decision criteria, in priority order:

1. **Privacy / data residency** — forces on-device.
2. **Reasoning need** — multi-step logic, math, code planning → frontier-reasoning. See `reasoning-models.md` for when reasoning models actually help vs. hurt.
3. **Output-token volume** — extended-thinking models burn output tokens on hidden reasoning; for high-volume bulk output, prefer frontier-general.
4. **Latency budget** — reasoning models have minimum latency floors (often several seconds) regardless of prompt size.
5. **Cost budget** — at high QPS, fast-cheap is the only tier that pencils out for non-reasoning tasks.
6. **Task complexity** — escalate one tier on high complexity.

### Tier Cascade (Try Cheap First)

For workloads where most queries are easy but a tail is hard, try fast-cheap first and escalate on a quality signal (low judge score, parsing failure, model-emitted "I'm not sure").

```python
def cascade_generation(prompt: str, judge_fn, threshold: float = 0.8) -> tuple[str, Tier]:
    """Try fast-cheap → frontier-general → frontier-reasoning, escalating only on need."""
    for tier in [Tier.FAST_CHEAP, Tier.FRONTIER_GENERAL, Tier.FRONTIER_REASONING]:
        result = call_model(MODEL_FOR_TIER[tier], prompt)
        if judge_fn(result, prompt) >= threshold:
            return result, tier
    return result, Tier.FRONTIER_REASONING
```

If only ~10% of queries escalate, average cost stays close to fast-cheap while tail-quality reaches frontier-reasoning.

## Part 4: Streaming

Token streaming converts a long generation wait into a smooth perceived experience. First-token latency replaces full-response latency as the user-visible metric.

```python
from openai import OpenAI

client = OpenAI()

def stream_response(prompt: str, model: str):
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

For web apps, expose this via Server-Sent Events (SSE) or HTTP/2 streaming; mobile clients can use the OpenAI/Anthropic SDK's native iterator. Stream both for chat UX and for long-form generation (articles, code, plans).

## Part 5: Serving Stack (Self-Hosted Inference)

When you run open-weights models yourself, the serving stack determines throughput, latency-under-load, and which optimizations are even available. The major engines:

### vLLM

**When to use:** Default first choice for a self-hosted production server on NVIDIA GPUs (and increasingly AMD MI300/Intel/TPU). Broadest model coverage of any high-perf engine.

**Headline feature:** **PagedAttention** — manages the KV cache as fixed-size pages with an OS-style page table, eliminating fragmentation and enabling KV-cache sharing across requests. Introduced in Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023 ([arXiv:2309.06180](https://arxiv.org/abs/2309.06180)).

**Key knobs:** `--gpu-memory-utilization`, `--max-num-batched-tokens`, `--max-model-len`, `--enable-prefix-caching`, tensor / pipeline parallelism via `--tensor-parallel-size`.

**Hardware:** NVIDIA (primary), AMD ROCm, Intel Gaudi/XPU, AWS Neuron, TPU (varying maturity).

**Repo:** [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm).

### SGLang

**When to use:** Workloads with heavy shared-prefix reuse — agent frameworks where many requests share the same long system prompt and tool definitions, structured-output workloads, multi-turn conversations.

**Headline feature:** **RadixAttention** — KV cache is stored in a radix tree keyed on token sequence; any prefix shared across requests is cached and reused automatically (LRU-evicted). Strong gains on prefix-heavy traffic; minimal benefit on all-unique prompts. Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" ([arXiv:2312.07104](https://arxiv.org/abs/2312.07104)).

**Key knobs:** `--mem-fraction-static`, `--max-running-requests`, schedule policy, structured-output backends (XGrammar/outlines).

**Hardware:** NVIDIA, AMD.

**Repo:** [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang).

### TensorRT-LLM

**When to use:** Maximum throughput on NVIDIA H100/H200/B200 in a stable production deployment where you can pay the build-time cost and accept narrower model support.

**Headline feature:** Ahead-of-time engine compilation with kernel fusion, FP8/INT4/INT8 quantization, in-flight batching, speculative decoding (Medusa, EAGLE, draft-target), and custom CUDA kernels per architecture. Per published H100 benchmarks, TensorRT-LLM typically leads vLLM at high concurrency once compiled.

**Key knobs:** Engine build flags (precision, max batch, max input/output, KV-cache reuse), runtime `kv_cache_free_gpu_memory_fraction`, parallelism config.

**Hardware:** NVIDIA only (H100/H200/B200 sweet spot; A100 supported).

**Docs:** [nvidia.github.io/TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/).

### TGI (Text Generation Inference)

**When to use:** Hugging Face ecosystem integration, Inference Endpoints, multi-backend deployments where you want one frontend that can swap between TRT-LLM, vLLM, or llama.cpp behind a stable API.

**Headline feature:** Production-grade Rust/Python server with continuous batching, tensor parallelism, and a multi-backend architecture letting you route workloads to vLLM/TRT-LLM/llama.cpp without changing client code. See HF's "Introducing multi-backends for Text Generation Inference" announcement.

**Key knobs:** `--backend`, `--max-batch-prefill-tokens`, `--max-input-tokens`, `--max-total-tokens`, quantization flags.

**Hardware:** NVIDIA, AMD ROCm, Intel, AWS Inferentia/Trainium.

**Repo:** [github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference).

### llama.cpp

**When to use:** CPU-only or hybrid CPU+GPU inference; Apple Silicon (Metal); edge devices; absolute portability; the GGUF model ecosystem.

**Headline feature:** **GGUF format + k-quants** — hierarchical affine quantization (Q2_K through Q8_0) with block-level scale/min metadata, enabling 2-8 bit quantization with graceful quality degradation. CPU+GPU split (`-ngl` layers offloaded to GPU) is unique in the ecosystem.

**Key knobs:** `-ngl` (GPU layers), `-c` (context), `-t` (threads), quantization variant (Q4_K_M is the common sweet spot), `--mmap`.

**Hardware:** CPU (x86, ARM), NVIDIA, AMD, Apple Metal, Vulkan.

**Repo:** [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

### ExLlamaV2

**When to use:** Single-GPU or small-multi-GPU consumer/prosumer setups (4090, 5090, A6000) running quantized open-weights models with maximum throughput-per-VRAM.

**Headline feature:** **EXL2 quantization format** — mixed-precision per-layer (different bits per tensor based on sensitivity), tightly integrated CUDA kernels, very low memory overhead. Strongest single-user throughput on consumer GPUs in its class.

**Key knobs:** Quantization bpw (bits-per-weight), `max_seq_len`, batch size (small batches are its strength).

**Hardware:** NVIDIA (consumer + datacenter), AMD ROCm.

**Repo:** [github.com/turboderp-org/exllamav2](https://github.com/turboderp-org/exllamav2).

### MLC-LLM

**When to use:** Cross-platform client-side deployment — browser (WebGPU), iOS, Android, embedded — where you cannot assume server inference. Universal compilation via Apache TVM.

**Headline feature:** TVM-based compilation pipeline targets every major runtime (CUDA, Metal, Vulkan, WebGPU, OpenCL) from a single source. The only mainstream stack with first-class WebGPU and mobile support.

**Key knobs:** Target runtime, quantization mode (q4f16_1 is common), KV-cache config.

**Hardware:** Everything. Literally.

**Repo:** [github.com/mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm).

### Picking a Stack

| You need… | Pick |
|-----------|------|
| Fastest path to production on NVIDIA, broad model support | **vLLM** |
| Heavy shared-prefix workloads (agents, structured output) | **SGLang** |
| Absolute peak NVIDIA throughput, willing to compile | **TensorRT-LLM** |
| HF ecosystem, multi-backend abstraction | **TGI** |
| CPU / Apple Silicon / edge / GGUF | **llama.cpp** |
| Maxing a single consumer GPU on quantized models | **ExLlamaV2** |
| Browser / mobile / cross-platform client | **MLC-LLM** |

**Cross-ref:** `yzmir-ml-production` (`optimize-inference`, `deploy-model`) covers ops-level concerns — autoscaling, rolling deploys, GPU scheduling, multi-tenant SLOs — in depth. This sheet covers stack selection and inference-time techniques.

## Part 6: Continuous (In-Flight) Batching

**Static batching** locks the GPU to a fixed batch until the slowest sequence finishes — short requests sit idle waiting for long ones.

**Dynamic batching** groups requests that arrive close in time but still runs them as a single locked unit until completion.

**Continuous batching** (also called "in-flight batching", "iteration-level scheduling") evaluates the queue *after every forward pass*. As soon as a sequence finishes, its slot is filled by a waiting request on the next iteration. Combined with PagedAttention's paged KV cache, this drives 2-10× throughput improvements over static batching at the same latency. Popularized by vLLM and now standard in TensorRT-LLM ("in-flight batching"), TGI, and SGLang. See the PagedAttention paper ([arXiv:2309.06180](https://arxiv.org/abs/2309.06180)) for the foundational treatment.

You don't typically configure continuous batching directly — picking a modern serving engine gives it to you. What you tune are the knobs that govern how aggressively the scheduler packs the batch (`--max-num-batched-tokens`, `--max-num-seqs` in vLLM; equivalents in other engines).

## Part 7: Speculative Decoding

Speculative decoding accelerates autoregressive generation by guessing several tokens ahead with a cheap process and verifying them with the target model in parallel. Verified prefixes are accepted; the first rejection rolls back to standard generation. Throughput gains of 2-3× are common; quality is mathematically identical to the target model's distribution under standard speculative-sampling acceptance rules.

### Draft-Target (Classic)

A small draft model proposes K tokens; the large target model scores them in one forward pass and accepts the longest valid prefix. Choose a draft model that's much smaller than the target but trained on similar data (e.g., a 1B draft for a 70B target from the same family).

**When to use:** When a small same-family model is available; simplest to deploy.
**Trade-off:** Draft quality matters — too dissimilar a draft = low acceptance rate = no speedup.

### Medusa

**Pattern:** Add multiple small "heads" on top of the target model, each predicting a future position (head k predicts token at position +k). At inference, all heads predict in parallel; the target verifies them in tree-attention mode. No separate draft model needed. Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads," [arXiv:2401.10774](https://arxiv.org/abs/2401.10774).

**When to use:** You can fine-tune extra heads on the target; want to avoid running a second model.
**Trade-off:** Heads must be trained; modest VRAM overhead.

### EAGLE / EAGLE-2 / EAGLE-3

**Pattern:** Draft is performed at the *feature* level (penultimate hidden state) rather than the token level, then projected through the target's LM head — better acceptance than naive draft-target at similar cost. EAGLE-2 adds a context-dependent dynamic draft tree; EAGLE-3 removes a feature-prediction objective that was hurting larger models. Reported ~3-6× speedups on standard benchmarks. Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty," [arXiv:2401.15077](https://arxiv.org/abs/2401.15077); EAGLE-2 [arXiv:2406.16858](https://arxiv.org/abs/2406.16858); EAGLE-3 [arXiv:2503.01840](https://arxiv.org/abs/2503.01840).

**When to use:** You want the highest published acceptance rates and can train an EAGLE head against your target. Now natively supported by vLLM, SGLang, and TensorRT-LLM.
**Trade-off:** Training the EAGLE module requires target-model logits/features.

### Decision Criteria

| Situation | Method |
|-----------|--------|
| Small same-family draft model already exists | Draft-target |
| Cannot run a second model; can fine-tune | Medusa |
| Want best published acceptance, can train | EAGLE-2/3 |
| Latency-sensitive serving on TRT-LLM/vLLM | Whatever the engine has built in |

If your serving engine ships with one of these enabled, that's almost always the right starting point.

## Part 8: Quantization for Inference (Naming Only — See ML-Production)

For LLM inference specifically, the relevant formats:

- **AWQ (Activation-aware Weight Quantization)** — protects the ~1% most salient weights based on activation statistics; very fast quantization, no backprop. Strong default for 4-bit weight-only on GPU. Lin et al., [arXiv:2306.00978](https://arxiv.org/abs/2306.00978).
- **GPTQ** — second-order, layer-by-layer post-training quantization; widely supported. Frantar et al., [arXiv:2210.17323](https://arxiv.org/abs/2210.17323).
- **FP8** — NVIDIA H100/H200 native 8-bit float. Near-BF16 quality at half the memory and bandwidth; often the best speed/quality combo on H100+ in TensorRT-LLM and vLLM.
- **GGUF k-quants** — llama.cpp's hierarchical affine quantizer (Q2_K through Q8_0). Q4_K_M is the everyday sweet spot for CPU/edge.

When to use what, in inference terms:

| Setup | Quantization choice |
|-------|---------------------|
| H100/H200/B200 production serving | FP8 (TensorRT-LLM or vLLM) |
| A100/L40S serving, open-weights | AWQ-int4 or GPTQ-int4 in vLLM/SGLang |
| Consumer GPU single-user | EXL2 (ExLlamaV2) or AWQ |
| CPU / Mac / edge | GGUF Q4_K_M (llama.cpp) |
| Quality-first, willing to spend VRAM | BF16 or FP8 (no INT4) |

**Cross-ref:** `yzmir-ml-production` covers quantization ops in depth — calibration data selection, accuracy-vs-throughput sweeps, mixed-precision strategies, INT4 vs INT8 vs FP8 trade-offs, and validation methodology.

## Part 9: Cost-Latency-Quality Trade-offs

### Multi-Objective (Pareto) Analysis

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Option:
    name: str
    latency_p95_s: float
    relative_cost: float    # arbitrary unit; compare options to each other
    quality: float          # 0-1

    def dominates(self, other: "Option") -> bool:
        better_or_equal = (
            self.latency_p95_s <= other.latency_p95_s
            and self.relative_cost <= other.relative_cost
            and self.quality >= other.quality
        )
        strictly_better = (
            self.latency_p95_s < other.latency_p95_s
            or self.relative_cost < other.relative_cost
            or self.quality > other.quality
        )
        return better_or_equal and strictly_better

def pareto_frontier(options: list[Option]) -> list[Option]:
    return [o for o in options if not any(other.dominates(o) for other in options)]
```

Use **relative-cost** units (multiples of your fast-cheap baseline) rather than dollar values — provider pricing changes, but cost ratios between tiers move slowly.

### Requirements-Based Selection

```python
def select(options: list[Option], max_latency: float, max_cost: float, min_quality: float) -> Option:
    feasible = [
        o for o in options
        if o.latency_p95_s <= max_latency and o.relative_cost <= max_cost and o.quality >= min_quality
    ]
    if not feasible:
        raise ValueError("No option meets all constraints")
    return min(feasible, key=lambda o: o.relative_cost / o.quality)
```

## Part 10: Production Monitoring

```python
from dataclasses import dataclass, field
from collections import deque
import numpy as np

@dataclass
class QueryMetrics:
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    relative_cost: float
    cache_hit: bool
    tier: str
    model_id: str

class PerformanceMonitor:
    def __init__(self, window: int = 10000):
        self.metrics: deque[QueryMetrics] = deque(maxlen=window)

    def log(self, m: QueryMetrics) -> None:
        self.metrics.append(m)

    def summary(self) -> dict:
        if not self.metrics:
            return {}
        lat = [m.latency_ms for m in self.metrics]
        return {
            "n": len(self.metrics),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)),
            "cache_hit_rate": float(np.mean([m.cache_hit for m in self.metrics])),
            "cache_token_share": float(
                sum(m.cached_input_tokens for m in self.metrics)
                / max(1, sum(m.input_tokens for m in self.metrics))
            ),
            "by_tier": self._by("tier"),
            "by_model": self._by("model_id"),
        }

    def _by(self, attr: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.metrics:
            k = getattr(m, attr)
            out[k] = out.get(k, 0) + 1
        return out
```

Track per-tier latency/cost so a router change is auditable. Watch `cache_token_share` — if it drops, your prompt structure has drifted (a cache key changed) and your costs will jump.

## Summary

**Inference optimization is systematic.**

1. **Parallelize** with async/await and a semaphore; use Batch APIs for offline.
2. **Cache** answers (hot keys) and prompts (static prefixes — see `context-engineering-and-prompt-caching.md`).
3. **Route by capability tier** (frontier-reasoning / frontier-general / fast-cheap / on-device); never hardcode model IDs; resolve via config.
4. **Stream** long generations.
5. **Pick the right serving stack** for self-hosted: vLLM (default), SGLang (prefix reuse), TensorRT-LLM (peak NVIDIA), TGI (HF ecosystem), llama.cpp (CPU/edge), ExLlamaV2 (single GPU), MLC-LLM (cross-platform).
6. **Use continuous batching, speculative decoding, and quantization** as supported by your stack — these are 2-10× multipliers, not optional polish.
7. **Pareto** the cost-latency-quality space; select against explicit constraints.
8. **Monitor** per-tier metrics and cache-hit rates in production.

**Cross-references:**
- `yzmir-ml-production` — serving stack ops, autoscaling, quantization ops detail.
- `context-engineering-and-prompt-caching.md` — prompt caching design depth.
- `reasoning-models.md` — when frontier-reasoning actually helps.
- `agentic-patterns-and-mcp.md` — agentic workloads (heavy prefix reuse → SGLang).
- `ordis-security-architect` — LLM threat modeling for serving infrastructure.

---

*Model lineup current as of 2026-05; revisit quarterly.*
