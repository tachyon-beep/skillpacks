---
description: Optimize LLM cost and latency with task-specific source and runtime evidence.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write"]
argument-hint: "[target_file_or_description]"
---

# Optimize LLM cost and latency

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Measure representative quality, token/reasoning use, cache hits, concurrency/queueing and p50/p95 latency. Identify the actual bottleneck; compare supported caching, routing, batching, bounded parallelism or model/config alternatives. Retest quality and resource use under comparable traffic; streaming changes perceived responsiveness, not necessarily completion time.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-llm-specialist/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [llm-inference-optimization](../skills/using-llm-specialist/llm-inference-optimization.md)
- [context-engineering-and-prompt-caching](../skills/using-llm-specialist/context-engineering-and-prompt-caching.md)
