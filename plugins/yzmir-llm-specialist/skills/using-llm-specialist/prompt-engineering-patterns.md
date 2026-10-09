# Prompt contracts and evaluation

Use for a concrete instruction-following, output-contract or prompt-regression problem. A capable model does not need a tutorial before a routine prompt edit.

## Checks

- State the actual task, available evidence, audience and observable success criteria. Put application policy in its trusted instruction channel; keep quoted/retrieved content distinguishable as data.
- Remove contradictory instructions and unnecessary scaffolding. Compare a short baseline before adding examples, decomposition or retries.
- Use examples for ambiguous edge cases or a specific output distribution; keep held-out evaluation cases separate from prompt examples.
- For machine outputs, prefer a supported schema/tool contract and validate downstream. Handle refusal, truncation and provider errors explicitly; syntactically valid JSON is not semantic correctness.
- Reasoning-effort and sampling controls are provider/model dependent. Do not prescribe explicit chain-of-thought, a temperature value or few-shot length for every model.
- Preserve source provenance, uncertainty and abstention where evidence is missing. A request for fluent confidence does not create factual support.
- Version prompts with model/config and evaluate representative slices, repeated runs where stochastic variation matters, cost and latency. Compare the changed behavior rather than subjective prompt polish.

## Deliverable

A prompt/config diff, examples of the desired boundary behavior and the observed regression/quality result or an explicit unrun evaluation plan. Keep a rollback version for deployed prompts.

## Related references

- [reasoning budgets](reasoning-models.md)
- [tool and schema boundaries](agentic-patterns-and-mcp.md)
- [evaluation and judges](llm-evaluation-metrics.md)
- [retrieval evidence](rag-architecture-patterns.md)
