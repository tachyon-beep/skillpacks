---
description: Review LLM trust and harm boundaries with task-specific source and runtime evidence.
model: opus
---

# Review LLM trust and harm boundaries

Apply this review/design to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Trace user/retrieved/tool data to privileged actions, confidential data and outputs. Inspect policy authority, tool permissions, sandbox/approval boundaries, sensitive logging and injection tests. Assess task-relevant moderation/bias/privacy controls with actual exposure evidence; prompts and filter lists alone do not prove isolation.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-llm-specialist/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [llm-safety-alignment](../skills/using-llm-specialist/llm-safety-alignment.md)
- [agentic-patterns-and-mcp](../skills/using-llm-specialist/agentic-patterns-and-mcp.md)
