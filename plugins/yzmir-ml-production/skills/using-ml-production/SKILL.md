---
name: using-ml-production
description: "Use when a model or ML dataset must be released, served, monitored or recovered with operational evidence."
---

# ML Production

Use this contract for the concrete task. Apply a short relevant check for a small change; expand investigation when the failure, risk or requested artifact warrants it. Resolve facts from the repository and runtime before imposing a process. Delegation is optional and should answer a bounded unresolved question.

## Release and operation contract

Identify the artifact, deployment target, traffic/resource budget, quality acceptance criteria and rollback owner. Dataset release and evaluation-set construction can precede training; serving needs a runnable model artifact.

- Bind model, preprocessing/tokenizer, data/prompt versions, dependencies and configuration to a release identity. Detect train/serve skew with representative inputs.
- Validate model output and failure behavior on required slices, not only endpoint health. Canary/shadow decisions need quality and resource evidence plus a reversible rollout.
- Measure queueing, batch behavior, memory/KV-cache pressure, warmup and tail latency on the actual hardware/traffic shape before choosing scaling or quantization.
- Test quantized/compressed artifacts against the task metric and supported kernels; smaller weights do not guarantee faster serving or acceptable quality.
- Define observability for input/output quality, drift, errors, tool calls and cost as applicable. Protect sensitive logs and distinguish model failure from serving/infrastructure failure.
- Preserve dataset composition, provenance, labels, deduplication/contamination checks, slice coverage and release documentation. Schema validity alone does not establish content quality.
- Exercise rollback/restore and record artifact compatibility. Do not equate deployment, health checks and live quality acceptance.

## Evidence and output

Produce a release/serving decision record, dataset release manifest, benchmark, incident diagnosis or rollout evidence. Include the artifact/config identity, measured workload, quality/resource results, unresolved limits and recovery path. Use existing backend/devops integration rather than copying an unrelated service scaffold.

## Fault-specific references

- Memory/latency attributed to the model: quantization/compression/hardware sheets.
- Queueing/batching/engine choice: `model-serving-patterns.md`; traffic capacity: `scaling-and-load-balancing.md`.
- Unsafe rollout or recovery: `deployment-strategies.md`.
- Missing lineage or automation: tracking/versioning and pipeline sheets.
- Content quality or contaminated eval set: `dataset-curation-and-quality.md`; statistical inference belongs to counterfactual statistics.
- Quality/drift/cost signals or an incident: monitoring/debugging sheets.

LLM prompt, retrieval and judge methodology belongs to LLM specialist. Framework implementation belongs to PyTorch engineering. Verify current serving APIs/tool maintenance status in upstream docs before adopting examples.

## Optional references

All sheets below are in this directory. Choose a sheet because its checks or examples help the task; there is no requirement to read the catalog in sequence. Verify time-sensitive APIs and numerical/performance claims before relying on examples.

- [dataset curation and quality](dataset-curation-and-quality.md)
- [deployment strategies](deployment-strategies.md)
- [experiment tracking and versioning](experiment-tracking-and-versioning.md)
- [hardware optimization strategies](hardware-optimization-strategies.md)
- [mlops pipeline automation](mlops-pipeline-automation.md)
- [model compression techniques](model-compression-techniques.md)
- [model serving patterns](model-serving-patterns.md)
- [production debugging techniques](production-debugging-techniques.md)
- [production monitoring and alerting](production-monitoring-and-alerting.md)
- [quantization for inference](quantization-for-inference.md)
- [scaling and load balancing](scaling-and-load-balancing.md)
