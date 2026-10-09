---
description: Review a mutation governor with task-specific source and runtime evidence.
model: opus
---

# Review a mutation governor

Apply this review/design to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Trace policy proposals through independent vetoes, joint resource checks, hysteresis, post-action watch windows and rollback. Verify the policy cannot modify its own gates, the snapshot restores all affected state and governor decisions enter learning/telemetry correctly. Inspect boundary and gate-gaming tests; clean findings are permitted with evidenced coverage.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-morphogenetic-rl/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [governor-and-safety-gates](../skills/using-morphogenetic-rl/governor-and-safety-gates.md)
- [rollback-as-rl-signal](../skills/using-morphogenetic-rl/rollback-as-rl-signal.md)
- [safety-gated-seed-fsm](../skills/using-morphogenetic-rl/safety-gated-seed-fsm.md)
