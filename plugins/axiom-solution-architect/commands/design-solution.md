---
description: Produce a complete solution-architecture artifact set from an input brief, HLD, epic, or brownfield change - routed end-to-end through triage, NFR quantification, tech/scope discipline, ADRs, RTM, integration/migration, optional TOGAF/ArchiMate, and assembly with consistency gate
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[input_file_or_description]"
---

# Design Solution Command

You are running the full solution-architect workflow. Your job is to take an input (brief, HLD, epic, or brownfield change) and produce the complete numbered artifact set in `solution-architecture/`, culminating in `99-solution-architecture-document.md` that has passed the consistency gate.

## Preconditions

Verify the following before starting:

```bash
# Check for an existing solution-architecture workspace (resume vs fresh)
ls solution-architecture/ 2>/dev/null

# For brownfield, check for archaeologist output
ls docs/arch-analysis-*/ 2>/dev/null
```

If `solution-architecture/` already exists with artifacts, ask the user whether to resume or start fresh (move existing to `solution-architecture.YYYY-MM-DD.bak/`).

## Workflow

1. **Triage** — use `triaging-input-maturity`
   → Produces `00-scope-and-context.md`, `01-requirements.md`, and a workflow plan
   → If input is brownfield and no archaeologist output exists, pause and recommend `/system-archaeologist` first

2. **NFRs** — use `quantifying-nfrs`
   → Produces `02-nfr-specification.md`, `03-nfr-mapping.md`

3. **Shape & tech** — use `resisting-tech-and-scope-creep`
   → Produces `04-solution-overview.md`, `05-tech-selection-rationale.md`, `06-descoped-and-deferred.md`
   → Significant decisions also flow through step 4

4. **ADRs** — use `writing-rigorous-adrs` for each significant decision
   → Produces `adrs/NNNN-*.md`

5. **Router-owned artifacts** — produce per catalog guidance in `using-solution-architect/SKILL.md`
   → `07-c4-context.md`, `08-c4-containers.md`, `09-component-specifications.md`, `10-data-model.md`, `11-interface-contracts.md`, `12-sequence-diagrams.md`, `13-deployment-view.md`

6. **Traceability** — use `maintaining-requirements-traceability`
   → Produces `14-requirements-traceability-matrix.md`

7. **Integration / migration / risks** — use `designing-for-integration-and-migration`
   → Produces `15-integration-plan.md`, `16-migration-plan.md` (brownfield only), `17-risk-register.md`

8. **TOGAF/ArchiMate (if enterprise)** — use `mapping-to-togaf-archimate`
   → Produces `archimate-model/`, `togaf-deliverable-map.md`

9. **Assembly** — use `assembling-solution-architecture-document`
   → Runs the 8-check consistency gate
   → If gate fails: report failures, fix artifacts, rerun. Do not emit SAD with silent waivers.
   → Produces `99-solution-architecture-document.md` + consistency gate report

## Output Location

All artifacts land in `solution-architecture/` (repo-root relative by default; override via argument if the project uses a different convention).

## Downstream Handoffs (suggest after completion)

- Security threat model → `/security-architect` reads `02-`, `04-`, `09-`
- Stakeholder polish → `/technical-writer` reads `99-`
- ADR lifecycle governance → `/sdlc-engineering` reads `adrs/`

## Scope Boundaries

Covered: the full forward-design workflow.

Not covered: operational runbooks, infrastructure-as-code implementation, execution scheduling (deferred).
