# Selecting a software governance profile

This repository offers an opt-in governance workflow. It does not prescribe a maturity level for every project or establish certification/compliance. Use the current standard edition and the organization's authorized policy as the source of requirements.

## Tailor before adoption

Identify the policy owner, scope, version, evidence consumer, decision and review cadence. Explain why each control is required. A regulation or maturity label alone does not establish which concrete controls apply.

| Concern | Minimum useful evidence when applicable |
|---|---|
| Requirements | Source/owner, acceptance criterion, baseline and change decision |
| Design | Alternatives, affected boundaries, assumptions and decision rationale |
| Verification | Requirement/failure addressed, executed method, environment and result |
| Change/configuration | Artifact identity, approved change, dependencies and recovery path |
| Risk/waiver | Trigger, impact, owner, action and expiry/reactivation condition |
| Acceptance | Accountable consumer, criteria, observed result and unresolved gaps |
| Measurement | Definition, denominator, source/window, assumptions and decision supported |

Existing linked records can supply these fields. Create additional artifacts only for an actual consumer or required control. Separate local policy from a sourced external requirement.

## Apply proportionately

Start with the current delivery problem. Pilot the smallest useful control and assess its cost and outcome before wider adoption. Preserve delivery continuity and existing authorization. A local repair does not automatically need a formal architecture pack, governance rollout or external approval.

Review duration, number of comments, number of findings, generic coverage percentages and fixed reviewer counts are not evidence of quality. If an authorized policy specifies a gate, record that source and evaluate the gate honestly; do not invent a requirement from this reference.

Use the [SDLC pack](../plugins/axiom-sdlc-engineering/skills/using-sdlc-engineering/SKILL.md) for governance selection, requirements, risk and measurement. Use [solution architecture](../plugins/axiom-solution-architect/skills/using-solution-architect/SKILL.md), [quality engineering](../plugins/ordis-quality-engineering/skills/using-quality-engineering/SKILL.md) or [DevOps](../plugins/axiom-devops-engineering/skills/using-devops-engineering/SKILL.md) for unresolved technical work. These are optional owners, not a mandatory invocation sequence.

The previous comprehensive prescription is retained in Git history. This replacement removes unsupported universal process mandates; it is not a mapping to any specific edition of CMMI or a regulatory standard.
