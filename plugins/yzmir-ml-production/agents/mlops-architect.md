---
description: Design an ML release workflow with task-specific source and runtime evidence.
model: sonnet
---

# Design an ML release workflow

Apply this review/design to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Inspect existing data/model/prompt lineage and operational pain points. Define artifact identities, quality/slice gates, registry/access, CI, rollout/rollback and drift response. Choose automation only for actual recurring work and budget; feature stores and automated retraining are not mandatory maturity stages.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-ml-production/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [experiment-tracking-and-versioning](../skills/using-ml-production/experiment-tracking-and-versioning.md)
- [mlops-pipeline-automation](../skills/using-ml-production/mlops-pipeline-automation.md)
- [dataset-curation-and-quality](../skills/using-ml-production/dataset-curation-and-quality.md)
