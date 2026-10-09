---
description: Profile a representative PyTorch workload with task-specific source and runtime evidence.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write"]
argument-hint: "[script_path] [--cpu|--gpu|--memory|--io]"
---

# Profile a representative PyTorch workload

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Record versions/device, input shapes/batch, warmup, synchronization and eager/compiled mode. Capture CPU/GPU and memory traces; distinguish host, compute, memory, communication and recompilation costs. Compare any optimization with identical measurement conditions and correctness checks; state profiling overhead and untested workloads.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-pytorch-engineering/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [performance-profiling](../skills/using-pytorch-engineering/performance-profiling.md)
- [mixed-precision-and-optimization](../skills/using-pytorch-engineering/mixed-precision-and-optimization.md)
