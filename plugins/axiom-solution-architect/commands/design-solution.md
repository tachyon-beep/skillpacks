---
description: Produce a complete solution-architecture artifact set from an input brief, HLD, epic, or brownfield change - routed end-to-end through triage, NFR quantification, tech/scope discipline, ADRs, RTM, integration/migration, optional TOGAF/ArchiMate, and assembly with consistency gate
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[input_file_or_description]"
---

# Design Solution Command

You are running the full solution-architect workflow. Your job is to take an input (brief, HLD, epic, or brownfield change) and produce the complete numbered artifact set in `solution-architecture/`, culminating in `99-solution-architecture-document.md` that has passed the consistency gate.

## Preconditions

The command takes a single argument: a file path to an input brief / HLD / epic / brownfield change-request, a directory of source material, or a short inline description of the change.

### Resolve the argument

```bash
INPUT="${ARGUMENTS}"

# Empty argument → ask the user
if [ -z "${INPUT}" ]; then
  # Use AskUserQuestion to collect:
  # "What brief / HLD / epic / change are we designing for?
  #  You can paste a description inline, or give me a path to a file or directory."
  :
fi

# File path → verify readable
if [ -f "${INPUT}" ]; then
  echo "Reading input file: ${INPUT}"
elif [ -d "${INPUT}" ]; then
  echo "Treating input as directory of source material: ${INPUT}"
  ls "${INPUT}" 2>/dev/null
else
  echo "Treating input as inline description: ${INPUT}"
fi
```

If the argument looks like a path (contains `/` or ends in `.md` / `.txt`) but the file or directory does not exist, stop and ask the user whether they meant a different path or want to treat the string as an inline description.

### Check for existing workspace (resume vs fresh)

```bash
# Is there already a solution-architecture/ workspace?
ls solution-architecture/ 2>/dev/null

# For brownfield, is there an archaeologist output to consume?
ls docs/arch-analysis-*/ 2>/dev/null
```

**Resume vs fresh protocol:**

If `solution-architecture/` already exists and contains any numbered artifacts, ask the user (via AskUserQuestion) to choose:

1. **Resume** — fill in missing artifacts, leave existing ones alone. Procedure:
   - `ls solution-architecture/` to identify which numbered artifacts exist.
   - Walk the workflow (step 1 through step 9) and *skip* any step whose artifacts are already present unless the user has explicitly asked for that step to be re-run.
   - For the first missing numbered artifact, read the artifacts immediately before it to understand the existing design, then resume from there.
   - Do **not** regenerate existing artifacts. If an existing artifact looks wrong, flag it to the user; do not silently overwrite.
2. **Fresh** — archive the existing workspace and restart. Procedure:
   - `mv solution-architecture/ solution-architecture.$(date +%Y-%m-%d).bak/` — move, do not delete. The backup is the user's undo.
   - Create a new empty `solution-architecture/` and start at step 1.
3. **Targeted rerun** — re-run a single step (e.g., "redo tech selection"). Procedure:
   - Archive *only* the artifacts owned by that step to `solution-architecture/.step-N.$(date +%Y-%m-%d).bak/`.
   - Re-run that step. Downstream steps may now be inconsistent — flag this and offer to re-run them.

If no `solution-architecture/` exists, proceed fresh without asking.

## Workflow

1. **Triage** — use `triaging-input-maturity`
   → Produces `00-scope-and-context.md`, `01-requirements.md`, and a workflow plan
   → If input is brownfield and no archaeologist output exists, pause and recommend `/system-archaeologist` first

2. **NFRs** — use `quantifying-nfrs`
   → Produces `02-nfr-specification.md`, `03-nfr-mapping.md`

3. **Shape & tech** — use `resisting-tech-and-scope-creep`
   → Produces `04-solution-overview.md`, `05-tech-selection-rationale.md`, `06-descoped-and-deferred.md`
   → Each significant decision identified here also requires an ADR (step 4)

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

- Security threat model → `ordis-security-architect` reads `02-`, `04-`, `09-`, `11-`, `15-`
- Stakeholder polish → `muna-technical-writer` reads `99-`
- ADR lifecycle governance → `axiom-sdlc-engineering` reads `adrs/`

## Scope Boundaries

Covered: the full forward-design workflow.

Not covered: operational runbooks, infrastructure-as-code implementation, execution scheduling (deferred).
