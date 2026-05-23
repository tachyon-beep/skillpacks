---
description: TDD-validated architectural assessment of existing codebases - critical quality assessment, technical debt cataloging, security-first prioritization; enforces professional discipline against diplomatic softening, analysis paralysis, and stakeholder-pressure compromise
---

# System Architect Routing

**Architect assesses existing codebases under stakeholder pressure. Archaeologist documents what exists (neutral); architect critiques quality and prioritizes fixes (evaluative). For forward design of new systems, use `/solution-architect`; to first generate the architecture documentation an architect consumes, use `/system-archaeologist`.**

Use the `using-system-architect` skill from the `axiom-system-architect` plugin to route to the right specialist sheet. Content authority lives in `plugins/axiom-system-architect/skills/using-system-architect/SKILL.md` — this wrapper is a thin pointer.

## When to Use

- Writing an architecture quality assessment that stakeholders will read
- Cataloging technical debt under deadline pressure
- Building a security-first improvement roadmap when stakeholders disagree
- "Is this architecture good?" / "What should I fix first?"

**Don't use** for: documenting an existing codebase neutrally (`/system-archaeologist`), forward design of new systems or change sets (`/solution-architect`), security threat modeling (`/security-architect`), or refactoring strategy beyond prioritization (out of scope — pack stops at the three failure modes below).

## Sheets

- **assessing-architecture-quality** — direct evidence-based assessment, resist diplomatic softening, refuse sandwich structure, handle authority and economic pressure
- **identifying-technical-debt** — structured debt catalog, complete or partial-with-limitations, deliver before explaining, defeat analysis paralysis under deadline
- **prioritizing-improvements** — risk-based roadmap with security as Phase 1, refuse "we've never been breached" reasoning, reject bundling as deprioritization

## Commands

- `/axiom-system-architect:assess-architecture` — write the assessment with professional objectivity; requires archaeologist outputs as input
- `/axiom-system-architect:catalog-debt` — produce the debt catalog with deliver-first discipline
- `/axiom-system-architect:prioritize-improvements` — produce the security-first roadmap from the catalog

## Agents

- `architecture-critic` — writes assessments under pressure; refuses diplomatic softening; declines documentation-only requests (hands off to `/system-archaeologist`)
- `debt-cataloger` — catalogs debt under deadline; delivers before explaining methodology

Both agents follow the SME Agent Protocol with Confidence/Risk/Information Gaps/Caveats sections.

## Cross-references

- Documenting an existing codebase (input to this pack) → `/system-archaeologist`
- Forward design of new systems / change sets → `/solution-architect`
- Security threat modeling and control design (referred from architect findings) → `/security-architect`
- Professional formatting of ADRs and assessment documents → `/technical-writer`
- Python-specific modernization called out by an assessment → `/python-engineering`
