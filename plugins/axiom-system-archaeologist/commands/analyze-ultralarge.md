---
description: Initiate ultralarge-tier codebase archaeology with manual subsystem partitioning and per-module review-with-scribe orchestration
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "AskUserQuestion"]
argument-hint: "[directory_or_scope]"
---

# Analyze Ultralarge Command

You are initiating archaeology on an **ultralarge** codebase. The single-pass workflow used by `/analyze-codebase` will produce a thin, sample-driven catalog that misses entire subsystems at this scale. This command switches to the per-module track with manual partitioning and a scribe-mediated merge protocol.

## When To Use This Command (Not `/analyze-codebase`)

Run this command when ANY of:
- Source LOC > 100,000
- Subsystem candidates > 12
- Test corpus LOC ≥ source LOC
- Doc corpus > 500 markdown files
- Plugin-registry architecture present (categories × concrete plugins)

If unsure, run the detection checklist (see [partitioning-ultralarge-repos.md](../skills/using-system-archaeologist/partitioning-ultralarge-repos.md)).

For repos that don't trip any of these → use `/analyze-codebase` instead.

## Core Principle

**Subsystems are checkpoints. Modules are the unit of analysis. Schema is load-bearing.**

The orchestrator (you) is a state machine that dispatches, collects, and advances cursors. You do not read source. You do not synthesize. You coordinate.

## Mandatory Workflow

### Step 1: Confirm Tier

```bash
# Source LOC
find <src> -name "*.<ext>" | xargs wc -l 2>/dev/null | tail -1

# Subsystem candidates
find <src> -maxdepth 2 -type d | wc -l

# Test-to-source ratio
find <tests> -name "*.<ext>" | xargs wc -l | tail -1

# Doc corpus
find . -name "*.md" -not -path "./node_modules/*" | wc -l

# Plugin surface
find <plugin-roots> -maxdepth 4 -type d 2>/dev/null
```

If the repo does NOT cross any threshold, ask the user whether they still want the ultralarge track. Default answer is no — switch to `/analyze-codebase`.

If it DOES cross a threshold, document which one(s) in the coordination plan and proceed.

### Step 2: Create Workspace

```bash
mkdir -p docs/arch-analysis-$(date +%Y-%m-%d-%H%M)/ultralarge
mkdir -p docs/arch-analysis-$(date +%Y-%m-%d-%H%M)/temp
```

The `ultralarge/` subdirectory is the marker that this is the per-module track.

### Step 3: Operator Interview (MANDATORY, ~10 min)

Use AskUserQuestion to gather (one question at a time, or combined):

1. **Subsystem partition** — "What are the major subsystems as you understand them?"
2. **Stability map** — "Which are stable vs. actively changing?"
3. **Extension surfaces** — "Are there plugin registries, hook systems, or codegen surfaces?"
4. **Doc triage** — "Which docs are authoritative vs. aspirational vs. stale?"
5. **Goal** — "What's the goal of this archaeology — onboarding / pre-refactor / compliance / other?"

Write to `00-coordination.md` under `## Operator Interview`.

**Skipping this step is the most expensive mistake.** Spending 10 minutes here saves dispatching dozens of reviewers at the wrong granularity.

### Step 4: Cheap Structural Scan (no deep reads)

Confirm or refute the operator's mental model. Find discrepancies — they are signal.

Write findings to `01-discovery-findings.md` (existing contract).

### Step 5: Partition Plan

Produce `ultralarge/00-partition-plan.md` per the format in [partitioning-ultralarge-repos.md](../skills/using-system-archaeologist/partitioning-ultralarge-repos.md):

- Subsystem table (id, name, path, est. modules, est. LOC, tier-within-ultralarge, order)
- Extension surfaces table
- Test corpus decision (separate pass? interleaved? skipped?)
- Doc corpus triage table
- Order justification

Order rules (default):
1. Leaf subsystems first (low outbound deps)
2. Type / contract subsystems early (consumed by others)
3. Plugin registry before plugins
4. Test corpus last
5. High-churn subsystems last within their layer

### Step 6: Per-Subsystem Loop

For each subsystem in order:

#### 6a. Subsystem partition
```bash
mkdir -p docs/arch-analysis-.../ultralarge/subsystem-S<NN>/modules
```

Write `subsystem-S<NN>/00-partition.md` with the module list and order (leaves first within the subsystem).

#### 6b. Per-module loop (DISPATCH → COLLECT → MERGE → VALIDATE → CHECKPOINT → ADVANCE)

For each module in order:

**DISPATCH**: spawn 4 `module-reviewer` agents in parallel (one Task message with 4 tool calls):
- focus=interface
- focus=internals
- focus=deps
- focus=quality

Each gets a task spec citing [findings-schema.md](../skills/using-system-archaeologist/findings-schema.md) and the partial output path.

**COLLECT**: confirm all 4 partial files exist. Re-dispatch any that failed.

**MERGE**: spawn 1 `subsystem-scribe` agent. It merges 4 partials → 1 canonical.

**VALIDATE**: read scribe's return summary. Re-dispatch on:
- Scribe reports module needs re-review
- Scribe reports >2 unresolved conflicts
- Scribe could not produce canonical

**CHECKPOINT**: update `00-partition.md` with module status. This is the resume point.

**ADVANCE**: cursor → next module.

#### 6c. Subsystem synthesis

After all modules in subsystem are canonical, spawn `codebase-explorer` agent to synthesize the existing-format catalog entry from canonical data. Append to `02-subsystem-catalog.md`.

#### 6d. Subsystem validation

Spawn `analysis-validator` on the synthesized entry + a 3-canonical sample. Address NEEDS_REVISION before advancing.

#### 6e. Commit checkpoint

Each subsystem is a resumable boundary. Stop here if needed; resume reads `00-partition-plan.md` to find the next un-completed subsystem.

### Step 7: Inbound-Dependency Reconciliation (after all subsystems complete)

Spawn `codebase-explorer` to scan ALL canonical files across subsystems and populate `Dependencies (Inbound)` in each subsystem catalog entry. Mechanical, no source re-reading.

### Step 8: Final Synthesis

Standard pack outputs:
- `03-diagrams.md` (use `/generate-diagrams`)
- `04-final-report.md` (per existing contract)
- Optional: quality, security, test-infrastructure, dependency-analysis sheets

## Resource Reality

For a representative ultralarge subsystem of 30 modules:
- 30 × 4 reviewer dispatches = 120 reviewer invocations
- 30 scribe merges
- 1 synthesis + 1 validation
- **~152 agent invocations per subsystem**

For 12 subsystems: **~1,800 invocations.** This is the cost. Plan accordingly.

Optimization shortcuts (use sparingly, document each):
- Modules <50 LOC: one reviewer with focus=all (degraded confidence, mark accordingly)
- `__init__.py` re-export only: orchestrator may produce canonical directly with confidence=low

## Output Workspace Structure

```
docs/arch-analysis-YYYY-MM-DD-HHMM/
├── 00-coordination.md
├── 01-discovery-findings.md
├── 02-subsystem-catalog.md          # Synthesized from canonicals
├── 03-diagrams.md                   # Optional
├── 04-final-report.md               # Optional
├── ultralarge/
│   ├── 00-partition-plan.md
│   ├── subsystem-S01/
│   │   ├── 00-partition.md
│   │   ├── modules/
│   │   │   ├── <id>.interface.partial.yaml
│   │   │   ├── <id>.internals.partial.yaml
│   │   │   ├── <id>.deps.partial.yaml
│   │   │   ├── <id>.quality.partial.yaml
│   │   │   └── <id>.canonical.yaml
│   │   └── 99-subsystem-catalog-entry.md
│   └── subsystem-S02/
│       └── ...
└── temp/
    ├── task-*.md
    └── validation-*.md
```

## Handling Time Pressure

This track is **not compatible with hour-scale deadlines.** A 12-subsystem ultralarge repo is days of work even with parallel dispatch. If the user wants results in a few hours:

Provide scoped alternatives:
- **A) Tier 1 only** — Run partition step + holistic discovery. Defer per-module work.
- **B) One subsystem deep, rest shallow** — Pick the most critical subsystem; full ultralarge track on that one. Existing `analyze-codebase` for the rest with documented limitations.
- **C) Switch to `/analyze-codebase`** — Accept the sampling tradeoff. Document what was sampled and what was skipped.

Document the choice and limitations explicitly in `00-coordination.md`.

## Anti-Patterns

❌ Run this command on a repo that doesn't cross any ultralarge threshold (use `/analyze-codebase`)
❌ Skip the operator interview ("the directory structure is clear")
❌ Try to canonicalize all subsystems in one session
❌ Pick subsystem order arbitrarily (forward refs cause rework)
❌ Read source files yourself ("just spot-checking the reviewer")
❌ Skip per-module checkpoint update ("it's just one more module")
❌ Run subsystem synthesis before all modules in scope are canonical
❌ Patch a canonical file by hand (re-run the loop)
❌ Dispatch a subagent to "decide the partitioning" (orchestrator + operator role)
❌ Treat reviewer disagreement as noise (it's signal, log in provenance)

## Cross-References

- Tier criteria & partitioning protocol → [partitioning-ultralarge-repos.md](../skills/using-system-archaeologist/partitioning-ultralarge-repos.md)
- Per-module loop detail → [module-by-module-with-scribe.md](../skills/using-system-archaeologist/module-by-module-with-scribe.md)
- Findings schema (load-bearing) → [findings-schema.md](../skills/using-system-archaeologist/findings-schema.md)
- Subsystem catalog contract → [analyzing-unknown-codebases.md](../skills/using-system-archaeologist/analyzing-unknown-codebases.md)
- Validation protocol → [validating-architecture-analysis.md](../skills/using-system-archaeologist/validating-architecture-analysis.md)

## Scope Boundaries

**This command covers:**
- Tier confirmation
- Workspace + partitioning
- Per-module orchestration loop
- Subsystem checkpointing
- Inbound-dependency reconciliation

**Not covered (use existing commands):**
- Diagrams → `/generate-diagrams`
- Validation execution → `/validate-analysis`
- Dependency analysis → `/analyze-dependencies`
