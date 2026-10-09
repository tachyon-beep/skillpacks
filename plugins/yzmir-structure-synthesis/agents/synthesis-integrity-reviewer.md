---
description: Review structure identity and verification with task-specific source and runtime evidence.
model: opus
---

# Review structure identity and verification

Apply this review/design to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Trace raw candidate through cheap legality checks, canonicalization, full gate, hashing and downstream selection. Test semantic-preservation/idempotence and label/order invariance, tied branches, edge/operator attributes and hash-version behavior. Check post-canonical diversity, provenance and selection boundaries. A clean review is valid when relevant patterns are ruled out with evidence; no finding quota applies.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-structure-synthesis/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [synthesis-anti-patterns](../skills/using-structure-synthesis/synthesis-anti-patterns.md)
- [canonicalisation-and-normal-forms](../skills/using-structure-synthesis/canonicalisation-and-normal-forms.md)
- [equivalence-detection-and-semantic-hashing](../skills/using-structure-synthesis/equivalence-detection-and-semantic-hashing.md)
