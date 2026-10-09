---
description: Diagnose PyTorch memory behavior with task-specific source and runtime evidence.
model: sonnet
---

# Diagnose PyTorch memory behavior

Apply this review/design to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Capture allocated/reserved/peak memory and a representative memory trace with target versions. Separate live tensors/graphs/hooks, temporary activations, optimizer state and fragmentation; inspect distributed per-rank effects. Verify a minimal lifetime/layout/checkpoint/sharding repair under the same workload rather than blindly clearing cache.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-pytorch-engineering/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [tensor-operations-and-memory](../skills/using-pytorch-engineering/tensor-operations-and-memory.md)
- [performance-profiling](../skills/using-pytorch-engineering/performance-profiling.md)
