# Using Solution Architect

<!-- Content authority lives in plugins/axiom-solution-architect/skills/using-solution-architect/SKILL.md. Do not add inline router content to this command. -->

## Overview

**Solution Architect produces forward design artifacts from a brief / HLD / epic / brownfield change.**

This pack is the forward-design counterpart to the backward-looking Axiom pair:

- `axiom-system-archaeologist` → documents existing code (neutral)
- `axiom-system-architect` → assesses existing architecture (critical)
- **`axiom-solution-architect` (this pack)** → designs new/changed solutions (forward)

## When to Use

Use solution-architect skills when:

- You have a business brief, HLD, epic, or brownfield change request
- You need a traceable artifact set (not a one-off diagram or ad-hoc ADR)
- The design will be reviewed, handed off, or implemented by another team
- The context is enterprise (TOGAF phases, ArchiMate tooling, ARB submission)
- User asks: "Design me a solution for…" / "Take this brief and architect it" / "What artifacts do we need for this?"

Do **not** use this pack when:

- You are assessing an existing system → use `/system-architect`
- You are documenting an existing system → use `/system-archaeologist`
- You need process governance (branching, CI/CD, ADR lifecycle) → use `/sdlc-engineering`

## How to Invoke

The full router lives at `plugins/axiom-solution-architect/skills/using-solution-architect/SKILL.md`. Read SKILL.md for complete guidance on: routing scenarios, scope tier triggers and the Scope Tier table, enterprise activation criteria (the four hard gates — not keyword presence), Stop Conditions, Update Workflows, and the router-owned artifacts quality floor (07–13). Do not rely on this command file for any of those details.

## Specialist Skills

Eight specialist sheets live alongside SKILL.md:

`triaging-input-maturity` · `resisting-tech-and-scope-creep` · `quantifying-nfrs` · `writing-rigorous-adrs` · `maintaining-requirements-traceability` · `designing-for-integration-and-migration` · `mapping-to-togaf-archimate` · `assembling-solution-architecture-document`

Full one-line descriptions and reading order are in SKILL.md's catalog (bottom of file). Do not reproduce that catalog here.
