---
description: Audit canonical form and semantic identity with task-specific source and runtime evidence.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Skill"]
argument-hint: "<path-to-canonicaliser-and-hasher-code>"
---

# Audit canonical form and semantic identity

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Inspect declared equivalences and normalization rules; test idempotence, reorder/relabel invariance and non-equivalent separation. Include downstream-distinguished nodes, symmetric branches, identity/nonlinearity deletion, parallel-edge/splice effects and cycle preconditions. Verify owned serialization/hash version and collision policy; record limits rather than claiming general semantic equivalence.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-structure-synthesis/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [canonicalisation-and-normal-forms](../skills/using-structure-synthesis/canonicalisation-and-normal-forms.md)
- [equivalence-detection-and-semantic-hashing](../skills/using-structure-synthesis/equivalence-detection-and-semantic-hashing.md)
- [structural-verification](../skills/using-structure-synthesis/structural-verification.md)
