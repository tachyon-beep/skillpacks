# Refresh: yzmir-ml-production

**Verdict:** HIGH / L effort. Strong router and durable concepts but specifics rotted.

## Context

- Pack path: `/home/john/skillpacks/plugins/yzmir-ml-production/`
- Full review: `/tmp/skillpack-refresh-review/yzmir-ml-production.md`
- Purpose: deploying / serving / monitoring / optimizing ML models in production.

## Why refresh

- **Deprecated `torch.quantization` API** — replaced by `torch.ao.quantization` with new flow.
- **Missing modern LLM serving:** no vLLM, no TGI, no Triton (current), no TensorRT-LLM, no SGLang.
- **Over-promotes TorchServe** — production reality has shifted to vLLM / specialized servers.
- **No LLM observability:** missing Arize Phoenix, Langfuse, OpenTelemetry GenAI semantic conventions.
- **Dated MLOps orchestrators:** check Kubeflow, MLflow, Metaflow, Prefect, Dagster current state.
- 10 long sheets — proportional rewrite needed.

## Scope — DO

1. **Quantization sheet.** Update to `torch.ao.quantization` API. Add AWQ, GPTQ, AQLM, bitsandbytes 8/4-bit, FP8 (H100/H200), int4 weight-only schemes. Distinguish weight-only vs activation-aware vs static vs dynamic.
2. **Serving sheet(s).** Add vLLM (paged attention, continuous batching, speculative decoding, prefix caching), SGLang, TensorRT-LLM. Reframe TorchServe as one of several options for non-LLM models. Add inference-server selection matrix.
3. **Monitoring/observability.** Add LLM-specific observability: Arize Phoenix, Langfuse, OpenTelemetry GenAI semconv. Distinguish traditional ML monitoring (drift, latency) from LLM monitoring (cost-per-request, token streams, hallucination eval).
4. **MLOps orchestrator coverage.** Verify versions, deprecate / update where appropriate.
5. **Model registry.** Refresh — MLflow Model Registry current state, alternatives (W&B Models, BentoML).
6. **Continuous batching, paged attention, speculative decoding.** Dedicated coverage somewhere.

## Scope — DO NOT

- Do not duplicate fine-tuning content (lives in `yzmir-training-optimization`).
- Do not duplicate LLM-application patterns (lives in `yzmir-llm-specialist`).
- Do not name specific model IDs that drift; use capability framings.

## Acceptance criteria

1. Zero references to deprecated `torch.quantization` (use `torch.ao.quantization`).
2. vLLM / SGLang / TensorRT-LLM each covered with at minimum: when to use, key flags, anti-patterns.
3. LLM observability section present with at least Phoenix or Langfuse.
4. TorchServe presented in honest context (legacy, still useful for non-LLM, but not the default for LLM serving).
5. Cross-pack references to `yzmir-llm-specialist` for LLM application patterns and `yzmir-training-optimization` for FP8/precision crossover.
6. `plugin.json` version bumped (major — substantive rewrite).

## Process

1. Read `/tmp/skillpack-refresh-review/yzmir-ml-production.md`.
2. Read every SKILL.md in this pack.
3. Confirm `yzmir-llm-specialist` and `yzmir-training-optimization` refresh status — coordinate cross-references.
4. List each sheet → keep / edit / replace. Plan past user before writing.
5. Write per sheet, verify code examples actually run (or are cited from official docs).
6. Update router last.
7. Bump version.

## Constraints

- Every API mentioned must be verified current (e.g. `torch.ao.quantization` paths exist).
- No fabrication of flags / config keys — cite from current docs.
- Capability-tier framing for LLM model size, not specific model IDs.
