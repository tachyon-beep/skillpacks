---
description: Locate simulation replay divergence with task-specific source and runtime evidence.
model: sonnet
---

# Locate simulation replay divergence

Apply this review/design to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Define equivalence/platform scope and reproduce fixed-input replay. Compare canonical state checkpoints, bisect the first divergent tick/subsystem and inspect RNG ownership, event/iteration order, arithmetic and snapshot completeness. Verify the minimal repair on recorded failing inputs; checksums must cover relevant state rather than unstable object serialization.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-simulation-tactics/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [debugging-simulation-chaos](../skills/using-simulation-tactics/debugging-simulation-chaos.md)
