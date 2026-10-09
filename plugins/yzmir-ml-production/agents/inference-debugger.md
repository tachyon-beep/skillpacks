---
description: Diagnose serving or model regressions with task-specific source and runtime evidence.
model: sonnet
---

# Diagnose serving or model regressions

Apply this review/design to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Bind failures to artifact/config and request/input slices. Separate queueing/preprocessing/model/serialization errors from quality/drift; reproduce representative failures and compare a known-good release. Profile synchronized warm workloads for resource issues and inspect lineage/train-serve skew for quality issues. Verify repair and rollback evidence without assuming the incident cause.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-ml-production/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [production-debugging-techniques](../skills/using-ml-production/production-debugging-techniques.md)
- [production-monitoring-and-alerting](../skills/using-ml-production/production-monitoring-and-alerting.md)
