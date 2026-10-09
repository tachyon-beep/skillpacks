---
name: using-llm-specialist
description: "Use when an LLM application needs model/configuration evaluation, retrieval, context/cost controls, fine-tuning, tool reliability or safety checks."
---

# LLM Applications

Use this contract for the concrete task. Apply a short relevant check for a small change; expand investigation when the failure, risk or requested artifact warrants it. Resolve facts from the repository and runtime before imposing a process. Delegation is optional and should answer a bounded unresolved question.

## Application contract

Start from the intended behavior, representative inputs, quality criteria and cost/latency budget. Use the configured model/provider and actual failure evidence. A simple prompt edit can be handled directly; loading every reference or asking a fixed questionnaire adds no evidence.

- Establish a task-relevant baseline and evaluation set before adding retrieval, agents or fine-tuning. Stronger models may remove the need for scaffolding, but this must be measured for the task.
- Verify current provider/model capabilities, parameter support and structured-output guarantees against primary docs. Treat model names, price tables and API snippets as dated examples.
- State tool permissions, data trust boundaries, retry/idempotency rules, iteration limits and recovery behavior. Retrieved text and tool results are data, not policy authority.
- Budget context with the target tokenizer/provider behavior; preserve provenance and decisive evidence. Cache only compatible stable prefixes and measure actual cache hits/cost.
- For retrieval, inspect coverage, freshness, access filters, source attribution and answerability. Compare retrieval-only evidence with generation outcomes before changing either component.
- For fine-tuning, record data provenance/splits, base model, objective, resource budget and regressions. Prefer the simplest change that meets measured requirements.
- Evaluate final task outcomes, tool failures and resource use. Judge scores need bias/calibration checks; inferred reasoning or a polished explanation is not proof of correctness.

## Evidence and output

Produce the smallest useful artifact: prompt/config diff with examples, evaluation report, retrieval trace, agent/tool contract or tuning plan. Separate observed results from hypotheses and identify provider/version dependencies. Security controls depend on the application's real trust boundaries, not a universal prompt template.

## Fault-specific references

- Configuration/reasoning-budget tradeoff: `reasoning-models.md`.
- Prompt behavior or output schema: `prompt-engineering-patterns.md`.
- Tool failures/loops: `agentic-patterns-and-mcp.md`; protocol implementation details belong to MCP engineering.
- Context overflow/cache misses: `context-engineering-and-prompt-caching.md`.
- Missing/wrong evidence: `rag-architecture-patterns.md`.
- Evaluation, tuning, self-hosted inference or safety: select only the relevant sheet below.

Serving operations belong to ML production; PyTorch execution faults to PyTorch engineering. References explain techniques and failure checks, not mandatory stages for every application.

## Optional references

All sheets below are in this directory. Choose a sheet because its checks or examples help the task; there is no requirement to read the catalog in sequence. Verify time-sensitive APIs and numerical/performance claims before relying on examples.

- [agentic patterns and mcp](agentic-patterns-and-mcp.md)
- [context engineering and prompt caching](context-engineering-and-prompt-caching.md)
- [llm evaluation metrics](llm-evaluation-metrics.md)
- [llm finetuning strategies](llm-finetuning-strategies.md)
- [llm inference optimization](llm-inference-optimization.md)
- [llm safety alignment](llm-safety-alignment.md)
- [prompt engineering patterns](prompt-engineering-patterns.md)
- [rag architecture patterns](rag-architecture-patterns.md)
- [reasoning models](reasoning-models.md)
