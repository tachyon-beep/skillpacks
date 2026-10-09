---
name: using-security-architect
description: "Use when system trust boundaries, threat models, control verification, AI/tool privileges, supply-chain assurance, classified information flows, or authorization evidence need design or review."
---

# Security Architecture

Ground security decisions in the actual system, attacker capabilities, assets,
trust boundaries and deployment context. Model knowledge or a checklist cannot
supply current compliance obligations, working controls, or authority to accept risk.

## Workflow

1. Inspect the system/design, data flows, identities, privileges, dependencies,
   existing findings and relevant operational evidence. State scope and assumptions.
2. Trace plausible attacker-controlled inputs through boundaries to protected
   assets or actions. Separate demonstrated paths, design weaknesses and untested
   hypotheses. Use threat frameworks to improve coverage, not inflate finding count.
3. Select controls at actual enforcement points. State expected protection,
   bypass/failure conditions, verification method and residual risk.
4. Test relevant controls with existing security tools/harnesses when authorized.
   Prefer native scan/findings workflows for executable code assessment; this pack
   also covers architecture, information flow and authorization artifacts.
5. Verify current official standards, framework versions and jurisdiction details
   when they govern the decision. Dated tables are discovery aids, not legal or
   accreditation sign-off. Cite the version and source used.
6. Respect the user's scope and existing authorization. Approval requirements
   follow actual impact, permissions and organizational authority. Do not require
   another skill or a human pause for every ordinary reversible implementation.

## Retrieve by threat or artifact

References are beside this file; load only relevant sections.

| Need | Reference |
|---|---|
| Decomposition, attack trees and enforcement gaps | [threat-modeling.md](threat-modeling.md) |
| Boundary controls and verification | [security-controls-design.md](security-controls-design.md), [secure-by-design-patterns.md](secure-by-design-patterns.md) |
| Existing design review | [security-architecture-review.md](security-architecture-review.md) |
| LLM, RAG, agents, MCP and tool actions | [llm-and-ai-security.md](llm-and-ai-security.md) |
| Dependencies, provenance, signatures and deployment policy | [supply-chain-security.md](supply-chain-security.md) |
| Classified information and trusted downgrade | [classified-systems-security.md](classified-systems-security.md) |
| Framework applicability and evidence mapping | [compliance-awareness-and-mapping.md](compliance-awareness-and-mapping.md) |
| RMF/ATO and actual risk-acceptance authority | [security-authorization-and-accreditation.md](security-authorization-and-accreditation.md) |
| Security decisions, residual risk and artifacts | [documenting-threats-and-controls.md](documenting-threats-and-controls.md) |

## Agentic and supply-chain boundaries

Treat retrieved text, tool results, model output and third-party artifacts as
untrusted inputs. Enforce user/tenant authorization and action scope outside the
model. Prompts, output schemas and human review can help but do not replace
privilege enforcement. Verify signatures against expected identity/issuer and
actual deployment policy; signing alone does not establish benign source.

Procedural decision-log integrity belongs to `axiom-audit-pipelines`; document
clarity to `muna-technical-writer`; LLM application quality to `yzmir-llm-specialist`.
Compose only when that boundary is in the requested task.

## Deliver

Return prioritized findings or a control/threat artifact with affected asset,
preconditions, evidence, impact, mitigation, verification and residual risk.
Report tools/checks actually run and coverage limits. Distinguish proposed
controls, tested enforcement, compliance mapping and authorized risk acceptance.
`threat-analyst` and `controls-designer` are optional bounded roles; no compulsory
multi-agent panel or universal four-section report is required.
