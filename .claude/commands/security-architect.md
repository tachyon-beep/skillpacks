---
description: Use when designing or reviewing security architecture - threat modeling (STRIDE), defense-in-depth controls, secure-by-design patterns, LLM/AI security (OWASP LLM Top 10:2025, MITRE ATLAS), supply-chain security (SLSA v1.1, SBOM, Sigstore), compliance mapping (NIST CSF 2.0, ISO 27001, PCI-DSS, GDPR, NIS2, EU AI Act), classified-systems MLS, and government authorization (ATO/RMF/FedRAMP). Routes to 10 specialist reference sheets, 3 commands, 2 SME agents.
---

# Security Architect Routing

**Security is architecture, not a final-stage filter. Threats drive controls; controls live at trust boundaries; compliance is consequence, not driver. For audit-log integrity and decision provenance use `/audit-pipelines`; for LLM application correctness (vs LLM security) use `/llm-specialist`.**

Use the `using-security-architect` skill from the `ordis-security-architect` plugin to route to the right specialist sheet. Content authority lives in `plugins/ordis-security-architect/skills/using-security-architect/SKILL.md` - this wrapper is a thin pointer.

## When to Use

- Designing a new system that has security-relevant surface (auth, data, networked components, agentic action)
- Reviewing an existing design for threats, gaps, or missing controls
- Mapping a system to a regulatory framework (HIPAA, PCI-DSS, GDPR, NIS2, EU AI Act, FedRAMP, IRAP/ISM)
- Authorizing a system for production / government use (ATO, RMF, SSP/SAR/POA&M)
- Threat modeling an LLM/agentic application or a build/release pipeline

**Don't use** for: UI styling, basic CRUD with existing tested auth, non-security refactoring, or audit-log specification (use `/audit-pipelines`).

## Sheets

### Core (universal - load for any security task)
- **threat-modeling** - STRIDE enumeration, attack trees, risk scoring (L x I), data-flow diagrams
- **secure-by-design-patterns** - zero-trust, least privilege, fail-secure, separation of concerns
- **security-controls-design** - trust-boundary-first defense-in-depth, control selection, encryption, secrets
- **security-architecture-review** - design-review checklist, gap analysis, evaluation rubric
- **documenting-threats-and-controls** - threat-model docs, security ADRs, control register templates

### Extensions (specialized - load only when context requires)
- **llm-and-ai-security** - OWASP LLM Top 10:2025, MITRE ATLAS, prompt injection, agentic threats, RAG security, model supply chain
- **supply-chain-security** - SLSA v1.1, SBOM (CycloneDX 1.6, SPDX 3.0), Sigstore/cosign, in-toto, dependency confusion, build provenance, xz-utils-class threats
- **classified-systems-security** - multi-level security (MLS), Bell-LaPadula, classification hierarchy, no-write-down enforcement
- **compliance-awareness-and-mapping** - NIST CSF 2.0, ISO 27001:2022, PCI-DSS v4.0.1, GDPR, NIS2, EU AI Act 2024/1689, ISM/IRAP, UK Cyber Essentials
- **security-authorization-and-accreditation** - RMF 7 steps, ATO/AIS, SSP/SAR/POA&M templates, FedRAMP, FISMA

## Commands

- `/ordis-security-architect:threat-model` - systematic STRIDE threat modeling with attack trees and risk scoring
- `/ordis-security-architect:design-controls` - layered controls at trust boundaries with defense-in-depth
- `/ordis-security-architect:security-review` - architecture review for security gaps and missing controls

## Agents

- `threat-analyst` - STRIDE enumeration, attack-tree construction, risk scoring; declines control-design tasks (hands off to `controls-designer`)
- `controls-designer` - trust-boundary control selection, defense-in-depth layering; declines threat-analysis tasks (hands off to `threat-analyst`)

Both agents follow the SME Agent Protocol with Confidence/Risk/Information Gaps/Caveats sections.

## Cross-references

- Decision-log integrity, canonical encoding, signed audit exports → `/audit-pipelines` (this pack designs *system controls*; audit-pipelines designs *the log itself*)
- LLM application correctness (prompts, RAG quality, fine-tuning) → `/llm-specialist` (this pack answers *how does an attacker abuse this*; Yzmir answers *how do I make it work well*)
- Writing SSPs, SARs, security ADRs → `/technical-writer`
- API-layer security implementation and review → `/web-backend`
- CI/CD pipeline mechanics and deployment gates → `/axiom-devops-engineering`
