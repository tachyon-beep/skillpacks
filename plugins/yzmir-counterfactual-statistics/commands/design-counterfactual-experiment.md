---
description: Write an executable analysis plan with task-specific source and runtime evidence.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit", "Skill", "Task"]
argument-hint: "[research question, or path to a design doc / eval harness]"
---

# Write an executable analysis plan

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Define unit, pairing/RNG contract, data roles, endpoint/utility, horizon, practical effect, pilot variance and planned unit count. Specify tests, corrections, interim looks, exclusions and abort/success criteria. Save the plan in the requested format/location and record its version/timestamp before confirmatory data is inspected. Do not launch expensive experiments or create commits solely because a template says to.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-counterfactual-statistics/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [preregistration-and-exploratory-vs-confirmatory](../skills/using-counterfactual-statistics/preregistration-and-exploratory-vs-confirmatory.md)
- [power-and-sample-size-for-paired-designs](../skills/using-counterfactual-statistics/power-and-sample-size-for-paired-designs.md)
- [common-random-numbers-and-matching](../skills/using-counterfactual-statistics/common-random-numbers-and-matching.md)
