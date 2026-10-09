---
description: Assess model and update stability with task-specific source and runtime evidence.
model: opus
---

# Assess model and update stability

Apply this review/design to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

State equilibrium/domain and perturbation assumptions. Analyze the continuous model and actual discrete update separately; Jacobian/eigenvalues are local and nonhyperbolic cases need additional analysis. Compare step-size/sensitivity/boundary traces and identify unsupported global claims. Recommend a checkable method or bound rather than promising universal stability.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-simulation-foundations/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [stability-analysis](../skills/using-simulation-foundations/stability-analysis.md)
- [numerical-methods](../skills/using-simulation-foundations/numerical-methods.md)
- [feedback-control-theory](../skills/using-simulation-foundations/feedback-control-theory.md)
