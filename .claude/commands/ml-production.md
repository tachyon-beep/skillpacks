---
description: Use when shipping ML models (classical or LLM) to production - quantization (torch.ao.quantization, AWQ/GPTQ/AQLM/FP8), LLM serving stacks (vLLM/SGLang/TensorRT-LLM/TGI/Triton), MLOps automation (Prefect/Dagster/Flyte/Airflow), observability (Phoenix/Langfuse/OTel GenAI), deployment rollout strategies, scaling, monitoring, and production debugging
---

# ML Production Routing

**Production ML is operational discipline, not training discipline. Models in production are serving stacks, rollout strategies, telemetry pipelines, and incident response. For training-phase convergence/speed use `/training-optimization`; for prompt/RAG/agent/eval-methodology quality use `/llm-specialist`; for AI threat modeling use `/security-architect`.**

Use the `using-ml-production` skill from the `yzmir-ml-production` plugin to route to the right specialist sheet. Content authority lives in `plugins/yzmir-ml-production/skills/using-ml-production/SKILL.md` — this wrapper is a thin pointer. Tool inventory is calibrated to 2026-05; re-check vendor docs for fast-moving areas (LLM serving, observability, quantization toolchains).

## Sheets

- **quantization-for-inference** — PTQ/QAT/dynamic/static and the PT2E (`prepare_pt2e`/`convert_pt2e`) graph path under `torch.ao.quantization`; INT8/INT4 ops; for LLMs the AWQ/GPTQ/AQLM/FP8 *ops* wiring (format choice belongs to `/llm-specialist`)
- **model-compression-techniques** — structured/unstructured pruning, knowledge distillation, architecture-level compression
- **hardware-optimization-strategies** — GPU/CPU/edge tuning, TensorRT, ONNX Runtime, kernel selection, batch sizing for hardware fit
- **model-serving-patterns** — FastAPI / gRPC / ONNX RT / request batching / containerization for general models; **TorchServe flagged as Legacy / Maintenance Mode**; **LLM-serving stacks (vLLM, SGLang, TensorRT-LLM, TGI, Triton)** with continuous batching, PagedAttention, speculative decoding, KV-cache management
- **deployment-strategies** — A/B testing, canary, shadow mode, blue-green, rollback procedures (RED/GREEN/REFACTOR)
- **scaling-and-load-balancing** — HPA, replicas, traffic shaping, cost-aware autoscaling; **LLM-specific scaling: KV-cache pressure, batch-size autoscaling**
- **experiment-tracking-and-versioning** — MLflow, W&B, model registries, reproducibility, lineage; **LLM extension: prompt versions, RAG eval-set versioning, fine-tune lineage**
- **mlops-pipeline-automation** — Prefect / Dagster / Flyte / Airflow, feature stores, data validation, automated retraining; **LLM extension: RAG-index rebuilds, fine-tune pipelines, eval gates**
- **production-monitoring-and-alerting** — RED metrics, drift detection, dashboards, SLAs/SLOs, Prometheus + Grafana + OTel; **LLM observability: Phoenix (Arize), Langfuse, OpenTelemetry GenAI semantic conventions, hallucination/tool-call/cost telemetry**
- **production-debugging-techniques** — RED-phase anti-patterns, GREEN-phase methodology, root-cause + post-mortem templates; **Part 10 LLM-Specific Debugging: cost spikes, retry storms, hallucination triage**

## Commands

- `/optimize-inference` — model-side inference optimization sweep across quantization, compression, and hardware tuning
- `/diagnose-inference` — production performance/accuracy/output-quality triage with evidence-gathering protocol
- `/deploy-model` — end-to-end deployment scaffolding: serving pattern selection, rollout strategy, monitoring wiring, rollback plan

## Agents

- `mlops-architect` — forward design: maturity assessment, experiment tracking, model registry, CI/CD pipeline, automated retraining
- `inference-debugger` — production-failure triage: categorize → gather evidence → reproduce in isolation → isolate component; slow / accuracy / OOM / intermittent playbooks

## Cross-references

- LLM generation quality, prompt/RAG/agent design, eval methodology, fine-tune strategy choice → `/llm-specialist` (dual-route for LLMs: ops here, methodology there)
- Training-phase convergence/speed → `/training-optimization`
- Distributed training, low-level profiling, CUDA/memory → `/pytorch-engineering`
- AI/LLM threat modeling — prompt injection, data exfiltration via tool-calls, model-artifact supply chain, PII in logs, model-registry access control → `/security-architect`
- "Performance" is three-way ambiguous: speed/latency, accuracy degradation, or LLM output quality — always clarify before routing
- Tool choices (Kubernetes, TorchServe, specific vendor) do not change routing; route by concern. TorchServe is in maintenance mode — for new LLM work, route to the LLM-serving stacks in `model-serving-patterns`
