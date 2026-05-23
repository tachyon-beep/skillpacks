---
description: CMMI-based SDLC framework (Levels 2-4) - requirements lifecycle, design and build, quality assurance, governance and risk, quantitative management, platform integration (GitHub/Azure DevOps), adoption on greenfield or brownfield. Routes to 7 specialist skills and 4 SME agents.
---

# SDLC Engineering Routing

**SDLC discipline defines WHAT to do (process), not HOW to implement it. CMMI maturity scales with team size and regulatory burden - Level 2 fits 1-5 person teams at ~5% overhead, Level 3 fits 5+ team audit-ready work at ~10-15%, Level 4 fits regulated industries (FDA, medical) at ~20-25%. For implementation specifics use the relevant domain pack: Python tooling -> `/python-engineering`, security controls -> `/security-architect`, CI/CD pipelines -> `/axiom-devops-engineering`, test frameworks -> `/quality-engineering`, documentation drafting -> `/technical-writer`.**

Use the `using-sdlc-engineering` skill from the `axiom-sdlc-engineering` plugin to route to the right specialist sheet. Content authority lives in `plugins/axiom-sdlc-engineering/skills/using-sdlc-engineering/SKILL.md` - this wrapper is a thin pointer.

## When to Use

- Adopting CMMI on a new or existing project (greenfield/brownfield strategy)
- Setting up requirements tracking, traceability, or change control
- Establishing design discipline (ADRs, branching, coding standards, peer review)
- Defining test strategy, verification, or stakeholder validation
- Decision documentation, alternatives analysis, or risk management
- Measurement programs, DORA metrics, dashboards, statistical process control
- GitHub or Azure DevOps platform setup against a CMMI policy
- General "what process should I follow" questions

**Don't use** for: implementation details for specific technologies (route to `/python-engineering`, `/security-architect`, `/axiom-devops-engineering`, `/quality-engineering`), code architecture critique (`/system-architect`), or documentation drafting (`/technical-writer`).

## Skills

- **using-sdlc-engineering** - router; detects CMMI level (CLAUDE.md > user message > default Level 3), routes to specialist
- **lifecycle-adoption** - bootstrapping CMMI on greenfield or brownfield projects, parallel-tracks strategy, change management
- **requirements-lifecycle** - RD + REQM at the detected level: elicitation, analysis, specification, traceability, change management
- **design-and-build** - TS + CM + PI: architecture and design, ADR process, configuration management, branching, integration
- **quality-assurance** - VER + VAL: testing strategy, peer reviews, defect management, validation with stakeholders, QA metrics
- **governance-and-risk** - DAR + RSKM: decision analysis with alternatives, risk register, mitigation, escalation
- **quantitative-management** - MA + QPM + OPP: measurement planning, DORA metrics, process baselines, statistical analysis (SPC, Cp/Cpk)
- **platform-integration** - GitHub and Azure DevOps implementation patterns for CMMI traceability, audit trails, automation

## Agents

- `sdlc-advisor` (sonnet) - routes/advises on CMMI adoption, level selection, and process design across the seven process areas
- `architecture-decision-reviewer` (opus) - critiques ADRs and design decisions; rejects resume-driven design and authority-without-analysis patterns
- `quality-assurance-analyst` (opus) - reviews test strategy and quality posture; resists "tests later" and "skip tests to ship" pressure
- `bug-triage-specialist` (opus) - triages defects, enforces RCA for recurring bugs (>3 similar = systemic), blocks premature closure

All four agents follow the SME Agent Protocol with Confidence/Risk/Information Gaps/Caveats sections.

## Cross-references

- Python testing implementation and tooling -> `/python-engineering`
- E2E, performance, chaos, observability testing -> `/quality-engineering`
- Security threat modeling and control design -> `/security-architect`
- CI/CD pipeline architecture and deployment mechanics -> `/axiom-devops-engineering`
- Technical documentation drafting (ADR prose, runbooks) -> `/technical-writer`
- UX review process integration -> `/ux-designer`
