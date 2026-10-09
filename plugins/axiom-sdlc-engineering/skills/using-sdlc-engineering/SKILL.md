---
name: using-sdlc-engineering
description: "Use when an organization explicitly selects a software governance profile requiring lifecycle evidence, traceability, risk decisions or measurement."
---

# Software Governance

This is an opt-in organizational policy profile, not a default maturity mandate. Read the applicable policy and identify its authority, scope, version and actual required evidence. Do not infer a CMMI level from task complexity or compliance keywords, or put repository prose above current user instructions.

## Workflow

1. Establish the selected policy/standard, accountable owners, current lifecycle and evidence consumers. If no governance need is selected, use ordinary engineering checks without adopting this profile.
2. Tailor requirements, design/review, change/configuration, verification and acceptance evidence to actual risk and policy. One linked record can satisfy several needs.
3. Connect requirements to design/change/test/acceptance and version the baseline. Record changes, decisions and waivers with owner and reactivation/expiry trigger where needed.
4. Verify selected controls against implementation and actual records. Separate compliance mapping from a claim of compliance/certification.
5. Measure only decisions that need data. Use declared definitions, denominator, source/window and assumptions; do not turn examples into quotas.
6. Review gaps and corrective actions. Quality is established by evidence, not review duration, comment counts, coverage floors or a required number of findings.

## Retained specialist entrypoints

- [Lifecycle adoption](../lifecycle-adoption/SKILL.md): assess/tailor/adopt policy without disrupting delivery.
- [Requirements](../requirements-lifecycle/SKILL.md): acceptance and change traceability.
- [Governance and risk](../governance-and-risk/SKILL.md): decision/risk/waiver and verification governance.
- [Quantitative management](../quantitative-management/SKILL.md): reproducible measurements and honest uncertainty.

Design/debt evidence lives in solution architecture, test technique in quality engineering, and GitHub/Azure operational recipes in DevOps. Use those only for the unresolved engineering concern; no mandatory multi-pack chain applies. Standards mapping requires a named edition and authoritative source; local prescriptions must be labeled local.

## Working contract

Use the smallest deliverable that makes the decision or change reviewable. Existing project records can satisfy these fields; do not create duplicate documents. Read only references needed for the unresolved question. Ordinary work does not require loading a specialist, delegating, or asking a routing question.

Use tools and current project evidence where available. Distinguish observed facts, inferences and unknowns. Report material findings with a source, consequence and proposed action; report a supported clean result when appropriate. State checks run, checks omitted and residual uncertainty without mandatory report sections. Honor current user authorization and applicable project policy; ask only when a missing decision materially blocks progress.
