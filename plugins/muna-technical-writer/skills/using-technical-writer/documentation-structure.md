# Documentation Artifact Contracts

Choose only sections needed for the artifact and its reader. The facts come from
actual code, decisions and operations; templates do not supply missing evidence.

| Artifact | Essential contract |
|---|---|
| ADR | Context/constraints, decision, alternatives, consequences, status/date and source. Do not invent historical rationale. |
| API reference | Inputs/auth, outputs/errors, version/conditions, complete runnable examples and relevant contracts. |
| Runbook | Trigger, prerequisites/access, ordered actions, expected checks, stop/rollback/recovery and escalation authority. |
| README/quick start | Purpose/audience, supported environment, minimal working path, expected result and next step. |
| Architecture | System boundary, components/data/control flow, deployment/trust boundaries, decisions and unresolved assumptions. |
| Executive brief | Decision needed, evidence, options/cost/impact, uncertainty and next action. |

A small artifact can satisfy its contract in a paragraph. Add navigation, glossary
or separate reference pages only when readers need them. Cross-links should offer
depth without withholding essential task information.

Verify examples and links using [documentation-testing.md](documentation-testing.md).
Use [incident-response-documentation.md](incident-response-documentation.md) for
incident-specific evidence and [operational-acceptance-documentation.md](operational-acceptance-documentation.md)
for authorization artifacts. Use `muna-wiki-management` for relationships among
documents; [diagram-conventions.md](diagram-conventions.md) for diagram contracts.
