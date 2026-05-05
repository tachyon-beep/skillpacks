# Step 5 — AI Cluster Refresh Campaign — Sheet-Level Plan

Generated 2026-05-05 after Phase 0.1 (router refresh) for user signoff.
Source: per-pack reviews in `/tmp/skillpack-refresh-review/`.

## Conventions

- **KEEP** — leave as-is (already correct or evergreen).
- **EDIT** — surgical updates inside the existing structure.
- **REPLACE** — substantial rewrite of significant portions.
- **NEW** — sheet does not yet exist; create it.
- **No fabrication.** Every named technique cited (paper, official docs, production system).
- **Capability-tier framing** for model lineup; no hardcoded model IDs in prose. Provider docs are linked for current IDs.
- **Knowledge-cutoff acknowledgement** in LLM sheets ("model lineup current as of <date>; revisit quarterly").

---

## Phase 1 — yzmir-llm-specialist (anchor pack)

Drives the cross-pack vocabulary for reasoning, agentic, multimodal, serving, prompt caching.

| Sheet | Lines | Action | Notes |
|-------|-------|--------|-------|
| `SKILL.md` (router) | 233 | EDIT | Add 3 new sheets to routing tree; add reasoning/agentic/multimodal triggers; add capability-tier framing; "freshness band" footer. |
| `prompt-engineering-patterns.md` | 973 | EDIT | Strip hardcoded `model="gpt-4"`. Add a "reasoning-model prompts ≠ chat prompts" callout that points to new `reasoning-models.md`. CoT framed as "model property OR prompt technique," with decision tree. Cite Anthropic / OpenAI / DeepSeek prompting guides. |
| `rag-architecture-patterns.md` | 1168 | REPLACE (substantial) | Add **contextual retrieval** (Anthropic 2024); update embedding leaderboard to Voyage-3 / Cohere Embed v3 / Jina v3 / Nomic / BGE-M3 / OpenAI text-embedding-3-large; add **Matryoshka** embeddings (MRL); update reranker section to Cohere Rerank 3 / Voyage rerank-2 / Jina v2 / BGE-reranker-v2-m3; add late-interaction (ColBERTv2 / ColPali) for visual-doc RAG; add long-context-vs-RAG decision section; add agentic / multi-hop RAG (corrective-RAG, self-RAG, GraphRAG) primer. Anatomy stays. |
| `llm-finetuning-strategies.md` | 969 | REPLACE (substantial) | Reframe SFT → preference tuning → online RLHF lineage. Add **DPO**, **IPO**, **KTO**, **SimPO**, **ORPO**, **GRPO** with one-paragraph each + decision matrix. Add LoRA-family advances (DoRA, rsLoRA, LongLoRA, LoftQ). Add modern stacks: TRL, Axolotl, Unsloth, LLaMA-Factory. LR/memory tables refreshed off Llama-2 examples. Cite each method's paper. |
| `llm-inference-optimization.md` | 1032 | REPLACE (model router); EDIT (rest) | `ModelRouter` rewritten in capability-tier framing (frontier-reasoning / frontier-general / fast-cheap / on-device) — no hardcoded IDs; Anthropic-prompt-caching pricing replaced with relative cost framing. Add **serving-stack** section (vLLM, SGLang, TensorRT-LLM, TGI, llama.cpp, ExLlamaV2, MLC). Add continuous batching / paged attention / prefix caching / speculative decoding (Medusa / EAGLE / draft-models) at conceptual level. Cross-ref `yzmir-ml-production` for ops detail. |
| `context-window-management.md` | 1225 | REPLACE (model lists, pricing); EDIT (rest) | Replace 4k-baseline thinking with 128k–200k–1M tier framework; pricing tables replaced with relative-cost framing + "check provider docs" pointers. Add lost-in-the-middle 2024 follow-ups (RULER, NIAH variants). Add **prompt caching** as first-class strategy with cross-ref to new caching sheet. |
| `llm-evaluation-metrics.md` | 1558 | EDIT | Mostly evergreen (sklearn / BERTScore / ROUGE / MRR / NDCG). Add LLM-as-judge current state (judge-model bias, position bias, length bias, calibration), behavioral suites (Inspect AI), reasoning-model eval (separate thinking-token costs from output tokens), red-teaming primer. Cross-ref `ordis-security-architect` for adversarial. |
| `llm-safety-alignment.md` | 944 | REPLACE (judge model + SDK); EDIT (taxonomy) | Migrate from legacy `openai.Moderation.create` to `openai>=1.0` SDK; replace gpt-3.5/gpt-4 judge defaults with capability-tier framing. Add **Llama Guard 2/3, ShieldGemma, NeMo Guardrails, WildGuard, PromptGuard**. Update jailbreak taxonomy (DAN → GCG, PAIR, AutoDAN, many-shot jailbreaking). Cross-ref OWASP LLM Top 10 (defer details to `ordis-security-architect`). Add agentic-safety section (tool authorization, sandboxing, confused-deputy). |
| `reasoning-models.md` | — | **NEW** | When to use reasoning models (o-series, Claude extended thinking, DeepSeek-R1 / R1-distill, Gemini thinking, Qwen-QwQ); how to prompt them (less-is-more; CoT prompting often hurts); cost & latency profile (output-token-dominated); eval considerations; common failure modes (over-thinking, infinite loops, refusal). Decision matrix "when reasoning model beats chat model". |
| `agentic-patterns-and-mcp.md` | — | **NEW** | Tool-use loops; provider differences (OpenAI tools, Anthropic tools, Gemini function calling); structured outputs (`response_format`, JSON-schema mode, Outlines, Instructor); parallel tool calls; agent error recovery; sub-agent / orchestrator-worker patterns; computer use; **MCP** (clients, servers, transport); agentic anti-patterns (context bloat, infinite loops, tool sprawl, confused-deputy). |
| `context-engineering-and-prompt-caching.md` | — | **NEW** | Anthropic ephemeral + 1-hour cache TTL; OpenAI auto-prefix caching; Gemini context caching; cache-aware prompt structure (stable-prefix-first); when caching beats RAG; context engineering as discipline (sub-agents, file-based memory, todo lists, compression strategies for agents). |

**Plugin metadata:** `version` 1.1.4 → **1.2.0** (3 new sheets + substantial rewrites = minor bump under semver-ish convention used by other recent refreshes; consider 2.0.0 if scope balloons).

**Risk:** code in legacy sheets uses `openai==0.x` API — wholesale migration to `openai>=1.0` is mandatory or examples won't run.

---

## Phase 2 — yzmir-ml-production

Aligns serving stack and observability with Phase-1 vocabulary.

| Sheet | Lines | Action | Notes |
|-------|-------|--------|-------|
| `SKILL.md` (router) | 390 | EDIT | Tighten LLM-specialist boundary; add "if shipping LLMs, see X" callouts; add LLM-observability triggers; add cross-ref to phase-1 serving-stack section. |
| `quantization-for-inference.md` | 991 | REPLACE (Part 7); EDIT (rest) | Migrate every `torch.quantization` example to `torch.ao.quantization`; add **PT2E export quantization** flow. Replace 50-line LLM hand-wave with reference matrix: GPTQ, AWQ, AQLM, HQQ, bitsandbytes NF4/FP4, **FP8 (E4M3/E5M2)** for Hopper, GGUF k-quants. Hardware compatibility table (Ampere / Ada / Hopper / CPU). Tool mapping (AutoAWQ, AutoGPTQ, llama.cpp, ExLlamaV2, TensorRT-LLM). |
| `model-serving-patterns.md` | 1667 | REPLACE (TorchServe section); NEW PART (LLM serving) | Add a Part covering **vLLM, SGLang, TensorRT-LLM, TGI, Ray Serve, BentoML, Triton**. Selection matrix mirroring existing FastAPI/TorchServe/gRPC/ONNX one. Demote TorchServe to "maintenance mode — prefer X for new work" honestly; keep TorchServe content for legacy. Migrate FastAPI examples from `@app.on_event("startup")` to `lifespan` context manager. ONNX `opset_version` bumped to 17 consistently. |
| `model-compression-techniques.md` | 1194 | EDIT | Migrate `torch.quantization` references to `torch.ao.quantization`. Distillation/pruning conceptual content stays. |
| `hardware-optimization-strategies.md` | 1323 | EDIT | Migrate `torch.quantization` calls; add FP8 (Hopper / Ada / Blackwell) and TensorRT-LLM mention; bump opsets; flag `expandable_segments`. |
| `production-monitoring-and-alerting.md` | 1412 | REPLACE (LLM section); EDIT (rest) | RED metrics / Prometheus / Grafana stays. Add **LLM observability** section: Arize Phoenix, Langfuse, Helicone, OpenTelemetry GenAI semconv (and "what's stable / what's still beta"). Cost-per-request / token streams / hallucination eval distinguished from drift / latency. Cross-ref ml-production drift-detection. |
| `mlops-pipeline-automation.md` | 2615 | EDIT | Reposition Airflow as one-of-many; add Prefect 2/3, Dagster, Flyte, Metaflow, ZenML, Argo Workflows, Kubeflow Pipelines v2 with one-paragraph each and selection guidance. Update MLflow content to current. |
| `experiment-tracking-and-versioning.md` | 2565 | EDIT | Add W&B, Comet, Neptune as alternatives; add Hugging Face Hub as registry option; mention BentoML model registry. |
| `deployment-strategies.md` | 3482 | EDIT | Mostly evergreen (shadow → canary → A/B → 100% progression). Add managed-services section (SageMaker, Vertex Prediction, Azure ML online endpoints, Modal, Replicate, Together, Anyscale, Inferentia/Trainium, TPU). Refresh dated examples. |
| `production-debugging-techniques.md` | 3466 | EDIT | Add LLM-specific debugging (token-stream inspection, hallucination triage). Mostly evergreen. |
| `scaling-and-load-balancing.md` | 2823 | EDIT | Add continuous-batching at conceptual level; LLM-traffic-shape considerations. |

**Plugin metadata:** `version` 1.1.4 → **1.2.0**.

**Boundary discipline:** ml-production owns serving-stack selection + ops; llm-specialist owns generation-quality tuning. Every cross-ref bidirectional.

---

## Phase 3 — yzmir-training-optimization

Aligns optimizer / schedule / precision / parallelism vocabulary with the others.

| Sheet | Lines | Action | Notes |
|-------|-------|--------|-------|
| `SKILL.md` (router) | 491 | EDIT | Add modern-optimizer / WSD / FP8 / 8-bit-optimizer triggers; preserve excellent rationalization-resistance content; add explicit cross-ref to llm-specialist for preference tuning (DPO/GRPO). |
| `optimization-algorithms.md` | 1832 | REPLACE (modern-optimizer section); EDIT (rest) | Add 2024–2025 optimizer landscape: **Lion** (Google 2023), **Sophia** (Stanford 2023), **Shampoo / distributed Shampoo**, **AdEMAMix** (2024), **Muon** (2024), **Schedule-Free** (Defazio 2024), **Prodigy / D-Adaptation**, **CAME**. Each with paper cite, "when it beats AdamW / when it doesn't" subsection. Add **8-bit optimizers** (`bitsandbytes.optim.AdamW8bit`, `PagedAdamW8bit`). Add `fused=True`/`foreach=True` defaults. Update comparison table. Demote RNN/LSTM branch to legacy footnote. Preserve Adam-vs-AdamW correctness warning. |
| `learning-rate-scheduling.md` | 2723 | EDIT | Add **WSD (warmup-stable-decay)** as first-class option with "use this for LLM pretraining" guidance — frame as alternative with continuation properties, not strictly superior. Add LR-free schedule entry (Schedule-Free / Prodigy). Cosine etc. preserved. |
| `batch-size-and-memory-tradeoffs.md` | 1651 | REPLACE (mixed-precision section); EDIT (rest) | Migrate `torch.cuda.amp.autocast` / `GradScaler` to namespace-neutral `torch.amp.autocast('cuda', ...)` / `torch.amp.GradScaler('cuda', ...)`. Promote BF16 to default on Ampere+ (no GradScaler needed); FP16 only when targeting older GPUs. Add **FP8 / Transformer Engine** subsection for Hopper (E4M3/E5M2). Add **critical batch size / McCandlish gradient-noise scale**. Modern LLM batch sizes (millions of tokens) referenced. |
| `training-loop-architecture.md` | 882 | EDIT | Migrate deprecated `torch.cuda.amp.GradScaler` to `torch.amp.GradScaler('cuda')`. |
| `gradient-management.md` | 2442 | EDIT | Update gradient-clip / accumulation guidance for 8-bit optimizers and FP8. |
| `hyperparameter-tuning.md` | 1635 | EDIT | Add **muP / mu-Transfer** for LR transfer across model scales. Optuna/Ray Tune/W&B Sweeps preserved. |
| `experiment-tracking.md` | 1942 | EDIT | Crosslink to ml-production experiment-tracking; remove duplicate content; add reproducibility checklist for FP8 training. |
| `data-augmentation-strategies.md` | 1483 | KEEP / EDIT | Largely evergreen. Add note on synthetic-data + LLM-generated augmentation patterns. |
| `loss-functions-and-objectives.md` | 2138 | EDIT | Add preference-loss family pointer (DPO, IPO, KTO, SimPO, ORPO, GRPO) — owned by llm-specialist; this sheet just names them and cross-refs. |
| `overfitting-prevention.md` | 1464 | KEEP | Evergreen. Optional: add LLM-specific overfit-on-instruction-data note. |

**Plugin metadata:** `version` 1.1.5 → **1.2.0**.

**ZeRO/FSDP nomenclature:** add a short pointer page or paragraph naming ZeRO-1/2/3 + FSDP1/FSDP2 equivalents and "see pytorch-engineering for API; see this pack for *strategy*". Keeps boundary clean.

---

## Phase 4 — yzmir-pytorch-engineering

PyTorch API truth layer last; reconciles agents/commands (already current) with sheet content (lagging).

| Sheet | Lines | Action | Notes |
|-------|-------|--------|-------|
| `SKILL.md` (router) | 399 | EDIT | Add `torch.compile` / FSDP / FSDP2 / FlexAttention / CUDA Graphs / NVTX symptom rows. Preserve excellent routing/pressure-resistance content. |
| `mixed-precision-and-optimization.md` | 1349 | REPLACE (AMP API); ADD (compile, FlexAttention) | Migrate every `torch.cuda.amp.autocast/GradScaler` to `torch.amp.autocast("cuda", ...)` / `torch.amp.GradScaler("cuda")`; add API migration note. Add `torch.compile` section (modes: `default`, `reduce-overhead`, `max-autotune`; `dynamic=True`; recompilation triggers; `torch._dynamo.config.cache_size_limit`; regional compile; `fullgraph=True` discipline; FSDP+compile interaction). Add **FlexAttention** (`torch.nn.attention.flex_attention` PT 2.5+) with `score_mod` / `block_mask` / causal+document-mask examples. Modernize `autocast()` syntax to `torch.autocast("cuda", dtype=torch.bfloat16)`. Honest treatment of `torch.compile` failure modes (graph breaks, recompiles, dynamic-shape limitations, debugging). |
| `distributed-training-strategies.md` | 1848 | REPLACE (FairScale section) | Replace FairScale ZeRO content with **FSDP1** (`FullyShardedDataParallel`, `MixedPrecision`, `BackwardPrefetch`, `auto_wrap_policy`) and **FSDP2** (`fully_shard`, `use_orig_params=True` semantics, per-parameter sharding, `MixedPrecisionPolicy`, `OffloadPolicy`). Keep DDP basics. DeepSpeed mentioned as alternative path. Add **DTensor / `init_device_mesh`** sketch. Multi-node section pointed at device mesh. |
| `performance-profiling.md` | 1893 | EDIT | Promote `torch.compile` from passing mention to first-class section. Add **CUDA Graphs** (`torch.cuda.graph()` context, `make_graphed_callables`, when to use, kernel-launch-bound regimes). Add **NVTX** + Nsight Systems integration (`torch.cuda.nvtx.range`, `nsys profile`). Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for fragmentation. Preserve four-phase methodology and CUDA-Events-vs-time.time warning. |
| `tensor-operations-and-memory.md` | 1029 | EDIT | Add `channels_last` memory format for CNN throughput. Add `expandable_segments` to OOM remediation. Reference holistic-trace-analysis tools. |
| `checkpointing-and-reproducibility.md` | 1925 | EDIT | Migrate `torch.cuda.amp.GradScaler` type hints to `torch.amp.GradScaler("cuda")`. Add FSDP2 `state_dict_type` semantics, optimizer-state sharding, rank-0 saving discipline. |
| `module-design-patterns.md` | 1785 | EDIT | Add `torch.compile`-friendly module patterns; mention `flex_attention` integration; activation-checkpointing variants (selective / full / custom). |
| `debugging-techniques.md` | 1803 | EDIT | Add NaN/Inf debugging under FP8; FSDP2-specific gotchas; `torch.compile` debugging (`TORCH_LOGS=...`, `torch._dynamo.explain`). |
| `custom-autograd-functions.md` | 2828 | KEEP | Largely evergreen. Optional: note `torch.compile` interaction with custom autograd (graph breaks). |

**Plugin metadata:** `version` 1.1.4 → **1.2.0**.

**Reconciliation gate:** after sheet edits, verify every API name in `agents/*.md` and `commands/*.md` is taught somewhere in `skills/using-pytorch-engineering/*.md`. Currently agents reference `torch.compile` / FSDP2 / PT 2.9 which sheets don't teach — refresh closes this gap.

---

## Phase 5 — Marketplace + closure

| Action | Detail |
|--------|--------|
| `.claude-plugin/marketplace.json` version bump | Reflect 4-pack minor bump campaign. |
| Cross-pack reference audit | Every cross-ref in the 4 packs verified bidirectional and resolves to a real sheet. |
| Conventional-commit per pack | `feat(llm-specialist): refresh for reasoning, agentic, MCP, modern serving stack`, etc. |
| Memory snapshot | Note any non-obvious decisions (e.g., capability-tier framing convention) into auto-memory. |

---

## Verification budget

Each pack will need WebSearch / WebFetch for at least:
- **llm-specialist:** Anthropic prompt-caching docs, MCP spec, Voyage / Cohere / Jina embedding docs, OpenAI structured outputs, Llama Guard 3 release notes, contextual-retrieval paper, DPO/GRPO papers.
- **ml-production:** vLLM / SGLang / TensorRT-LLM docs (current flags), `torch.ao.quantization` PT2E flow, Phoenix / Langfuse docs, OTel GenAI semconv state.
- **training-optimization:** Lion / Sophia / Muon / AdEMAMix / Schedule-Free papers; bitsandbytes 8-bit optimizer docs; transformer-engine FP8 docs; WSD references.
- **pytorch-engineering:** PyTorch 2.9 release notes; FSDP2 `fully_shard` API; FlexAttention examples; CUDA Graphs guide; expandable_segments doc.

**Time budget:** WebSearch is mandatory per prompt constraints — every named technique cited.

---

## Execution model

Within a phase, dispatch parallel `general-purpose` subagents (one per sheet group) to draft updates against the per-sheet plan above. Synthesizer (this session) reviews each subagent output before commit. Cross-pack vocabulary frozen at end of phase 1 and reused in phases 2–4.

**Hard gate per pack:** plan-table compliance verified (every sheet has the action listed above applied) before bumping version and committing.
