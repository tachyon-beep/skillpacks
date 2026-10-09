# Reasoning configuration as a measured choice

Use when supported inference-effort controls or model capability tradeoffs affect a task. Model/provider names and parameter support change; inspect current primary documentation and the deployed configuration.

## Workload vocabulary

`frontier-reasoning`, `frontier-general`, `fast-cheap` and `on-device` are rough workload roles, not fixed quality rankings or provider products. Bind a concrete candidate/version and evaluate it. A model may support multiple effort modes and tool/multimodal capabilities.

## Checks

- Compare task correctness and failure cases against a simple baseline under the same input/evidence contract.
- Measure output/reasoning-token accounting, p50/p95 latency, rate limits and total task cost. Hidden inference work is not necessarily visible or retrievable as a trustworthy trace.
- Use supported effort/budget controls; verify accepted parameters instead of carrying a chat-model sampling recipe into another API.
- Do not demand private chain-of-thought as a correctness check. Evaluate final answers, cited evidence, tool behavior and reproducible outcomes; request concise verifiable explanations when useful.
- Additional effort may help some tasks and waste budget on others. Measure marginal quality/cost and the failure rate, including overlong or truncated outputs.
- In agent loops, budget the whole task, not one call. Bound iterations/retries and verify tool authorization/recovery independently of model capability.
- Rerun representative cases when model versions or effort defaults change. A previous cost/quality result is version-specific.

## Deliverable

A configuration decision with candidate/version, task set, quality/resource evidence, failure modes and fallback. Label unmeasured tier comparisons as hypotheses.

See [evaluation](llm-evaluation-metrics.md), [context/caching](context-engineering-and-prompt-caching.md) and [inference operations](llm-inference-optimization.md) for the particular boundary involved.

## Existing source pointers

Check current applicability/version before using a recipe. These pointers are optional supporting sources.

- <https://arxiv.org/abs/2501.12948>
- <https://platform.claude.com/docs/en/build-with-claude/extended-thinking>
- <https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking>
- <https://developers.openai.com/api/docs/guides/reasoning>
- <https://platform.openai.com/docs/guides/reasoning-best-practices>
