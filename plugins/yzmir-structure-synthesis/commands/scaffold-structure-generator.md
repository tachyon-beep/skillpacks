---
description: Scaffold generation and verification contracts with task-specific source and runtime evidence.
allowed-tools: ["Read", "Write", "Bash", "Skill"]
argument-hint: "<project-name> [--representation=pointer-based|edge-list|latent-decoder]"
---

# Scaffold generation and verification contracts

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Fit existing project abstractions. Implement a bounded representation/generator, cheap legality checks, canonicalizer/full verifier and versioned identity at explicit boundaries. Add meaningful round-trip, cycle/type/budget, idempotence, equivalence/separation and mutation checks. Treat predicted-utility-guided search as an explicit selection policy when required, not an unobservable verifier preference.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-structure-synthesis/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [graph-representations-for-generation](../skills/using-structure-synthesis/graph-representations-for-generation.md)
- [structural-verification](../skills/using-structure-synthesis/structural-verification.md)
- [canonicalisation-and-normal-forms](../skills/using-structure-synthesis/canonicalisation-and-normal-forms.md)
