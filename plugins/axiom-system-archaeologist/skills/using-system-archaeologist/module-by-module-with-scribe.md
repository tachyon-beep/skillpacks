# Module-by-Module With Scribe (Ultralarge Track)

## Purpose

Once a subsystem is partitioned (see [partitioning-ultralarge-repos.md](partitioning-ultralarge-repos.md)), each module in scope is archaeologized as a unit. This sheet defines the per-module orchestration loop: parallel reviewers fill schema partials, the scribe merges, the orchestrator advances.

## Core Principle

**The orchestrator is a state machine, not an analyst.** Per module, the loop is exactly:

```
DISPATCH → COLLECT → MERGE → VALIDATE → CHECKPOINT → ADVANCE
```

If any step contains "I'll just look at the code myself," the workflow has failed. The orchestrator does not read source files in this track.

---

## The Per-Module Loop

### Inputs
- `subsystem-S<NN>/00-partition.md` (module list with order)
- Cursor pointing to the next un-canonicalized module
- Workspace at `docs/arch-analysis-.../ultralarge/subsystem-S<NN>/`

### Step 1: DISPATCH (parallel)

Spawn **4 `module-reviewer` agents in a single message**, one per focus:

```
Task 1: module-reviewer | focus=interface  | module=<path> | output=<id>.interface.partial.yaml
Task 2: module-reviewer | focus=internals  | module=<path> | output=<id>.internals.partial.yaml
Task 3: module-reviewer | focus=deps       | module=<path> | output=<id>.deps.partial.yaml
Task 4: module-reviewer | focus=quality    | module=<path> | output=<id>.quality.partial.yaml
```

All four reviewers read the same module independently. They do NOT see each other's work. This is by design — convergent independent review surfaces disagreements (which the scribe records as conflicts).

**Task spec template** (one per reviewer dispatch):

```markdown
# Task: Module Review — <focus>

You are a `module-reviewer` agent operating on focus=<focus>.

## Module under review
- Path: <repo-relative path>
- Module ID: <slug>

## Schema
Read `<workspace>/ultralarge/findings-schema.md` (or the in-repo path
to that sheet) before producing output. Your output MUST conform to the
`<focus>.partial.yaml` schema exactly.

## Constraints
- Stay in your focus. Do not report findings outside this focus.
- One-line summaries only.
- Empty list `[]` for sections with no entries — never omit keys.
- Cite what you read in `confidence_evidence`.

## Output
Write to: <workspace>/ultralarge/subsystem-S<NN>/modules/<id>.<focus>.partial.yaml
Then return a one-paragraph summary of what you wrote (the orchestrator
does NOT read your full output).

## Self-validation before returning
Run the Validation Checklist (Reviewer Self-Check) from findings-schema.md.
If your partial fails any check, fix it before returning.
```

### Step 2: COLLECT

After all four return, the orchestrator does **not** open the partials. It only:

- Confirms all four files exist
- Notes any reviewer that returned with confidence=low (these flag the module for closer scribe attention)
- Updates the cursor: module is now in MERGE state

If any reviewer failed to produce its partial (errored, returned an empty file, returned prose), **re-dispatch only that reviewer**. Do not proceed to merge with missing focuses.

### Step 3: MERGE

Spawn **1 `subsystem-scribe` agent** with this task:

```markdown
# Task: Scribe Merge — Module <id>

You are the `subsystem-scribe` agent. Merge 4 partials into 1 canonical entry.

## Partials to merge
- <workspace>/.../modules/<id>.interface.partial.yaml
- <workspace>/.../modules/<id>.internals.partial.yaml
- <workspace>/.../modules/<id>.deps.partial.yaml
- <workspace>/.../modules/<id>.quality.partial.yaml

## Schema
Read `findings-schema.md` for canonical schema. Your output MUST conform.

## Mandate
- Copy entries from partials to canonical.
- Resolve duplicates (same class in two partials → keep interface version).
- Resolve conflicts → log in provenance.conflicts_resolved with both sides cited.
- Compute confidence.overall = MIN of partial confidences (deviation requires reasoning).
- DO NOT add new findings. Every line in canonical traces to a partial.

## Output
Write to: <workspace>/.../modules/<id>.canonical.yaml
Return a one-paragraph summary including: any conflicts resolved,
overall confidence, and (yes/no) whether this module appears to need
re-review (e.g., reviewer disagreement is severe).

## Self-validation
Run the Validation Checklist (Scribe Self-Check) from findings-schema.md.
```

### Step 4: VALIDATE (lightweight, per-module)

The orchestrator checks the scribe's return summary. Re-dispatch is required if:

- Scribe reports "module needs re-review"
- Scribe reports >2 unresolved conflicts
- Scribe could not produce canonical (missing schema fields)

Otherwise advance. **Heavyweight validation happens at subsystem boundary**, not per-module — see Step 6.

### Step 5: CHECKPOINT

Update `subsystem-S<NN>/00-partition.md`:

```markdown
| Module ID | Path | LOC | Order | Status | Confidence |
|-----------|------|-----|-------|--------|------------|
| S01-M001 | ... | 12 | 1 | canonical | high |
| S01-M002 | ... | 84 | 2 | canonical | high |
| S01-M003 | ... | 312 | 3 | in-progress | - |
```

This file is the resume point. After every module, it is current.

### Step 6: ADVANCE

Increment cursor to next module in order. Loop to DISPATCH.

When all modules in the subsystem reach `status=canonical`, exit the per-module loop and proceed to **Subsystem Synthesis** (below).

---

## Subsystem Synthesis (Once Per Subsystem)

This is a separate role — NOT the scribe. The scribe operates per-module. Synthesis operates over an entire subsystem.

Spawn a `codebase-explorer` agent (existing) with this task:

```markdown
# Task: Subsystem Catalog Synthesis — S<NN>

Read all canonical files in <workspace>/ultralarge/subsystem-S<NN>/modules/*.canonical.yaml

Produce subsystem catalog entry per the contract in
analyzing-unknown-codebases.md. Aggregate from canonical data:

- **Key Components** ← top-5 modules by inbound imports_internal references
- **Dependencies (Outbound)** ← union of imports_internal pointing outside this subsystem
- **Dependencies (Inbound)** ← will be filled when other subsystems' canonicals exist
- **Patterns Observed** ← recurring key_algorithms / framework_hooks across modules
- **Concerns** ← high-severity smells + coverage_gaps + cross-module duplications
- **Confidence** ← MIN of canonical confidence.overall across modules

Write to: <workspace>/ultralarge/subsystem-S<NN>/99-subsystem-catalog-entry.md
Then append the entry to <workspace>/02-subsystem-catalog.md.
```

After synthesis, **commit the subsystem checkpoint**. The session can stop here. Resume reads `00-partition-plan.md` and finds the next subsystem.

---

## Validation at Subsystem Boundary (Mandatory)

Before declaring a subsystem complete:

Spawn `analysis-validator` agent (existing) with input:
- The subsystem catalog entry (`99-subsystem-catalog-entry.md`)
- Sample of canonical files (e.g., 3 modules)
- Schema reference

Validator confirms:
- Catalog entry conforms to existing contract
- Catalog claims are traceable to canonical data
- No catalog claim contradicts a canonical entry

Validation output → `temp/validation-S<NN>.md`. NEEDS_REVISION → re-spawn synthesis or specific module reviewer.

---

## Inbound-Dependency Reconciliation Pass

After ALL subsystems are canonicalized, one final pass reconciles inbound dependencies:

For each subsystem S, scan all OTHER subsystems' canonical files for `imports_internal` entries that resolve into S's modules. Populate `Dependencies (Inbound)` in S's catalog entry.

This is mechanical. Spawn one `codebase-explorer` agent with the canonical-file paths and the rule. No re-reading of source.

---

## Resource Budget (Reference)

For one subsystem of 30 modules:

| Step | Agents | Per-module | Total |
|------|--------|------------|-------|
| Reviewers | 4 in parallel | per module | 30 × 4 = 120 |
| Scribe | 1 | per module | 30 |
| Subsystem synthesis | 1 | per subsystem | 1 |
| Validation | 1 | per subsystem | 1 |
| **Total per subsystem** | | | **~152 agent invocations** |

For 12 subsystems: **~1,800 invocations.** This is the cost. Plan for it.

**Optimizations available:**
- For modules <50 LOC: skip parallel reviewers, dispatch one `module-reviewer` with focus=all (degraded schema, marked in confidence). Use sparingly.
- For trivial modules (`__init__.py` with only re-exports): orchestrator may produce canonical directly with confidence=low and a note. Document the rule in `00-coordination.md` if you do this.
- Scribe merging is cheap; do not skip it. It's the audit trail.

---

## When To Re-Review A Module

A module enters re-review queue if any of:

1. Scribe reports unresolved conflicts (≥3)
2. Subsystem synthesis surfaces a contradiction (catalog says X, canonical says ¬X)
3. Validation report flags the module specifically
4. A later subsystem's canonical reveals this module was misunderstood (e.g., its public API is used differently than the interface partial said)

Re-review is the SAME loop on the SAME module: 4 reviewers + scribe. The new canonical OVERWRITES the old; provenance retains both `reviewer_run_id` sets.

**Do not patch a canonical file by hand.** Either it stands or it goes through the loop again.

---

## Anti-Patterns

**❌ Orchestrator reads source files to "spot-check" reviewer output.**
The orchestrator's role is dispatch + state-tracking, not analyst. Spot-checking burns context for marginal benefit.

**❌ Scribe adds findings the partials don't contain.**
Scribe is mechanical merge. Adding findings means the reviewers underspecified — re-dispatch the relevant reviewer instead.

**❌ Skip per-module checkpoint update because "it's just one more module".**
The checkpoint is the resume point. A multi-module gap means a failed session has to start over.

**❌ Run subsystem synthesis before all modules are canonical.**
Synthesis aggregates canonical entries. Partial coverage produces a misleading catalog with no clear confidence floor.

**❌ Treat reviewer disagreement as noise.**
Disagreement is signal. The conflicts log in provenance is the most valuable cross-reference for downstream readers.

**❌ Skip the inbound-dependency reconciliation pass.**
Without it, every subsystem's catalog says `Dependencies (Inbound): unknown` — the catalog is half-built.

---

## Cross-References

- Schema (load-bearing) → [findings-schema.md](findings-schema.md)
- Tier criteria & subsystem partitioning → [partitioning-ultralarge-repos.md](partitioning-ultralarge-repos.md)
- Subsystem catalog contract (synthesis target) → [analyzing-unknown-codebases.md](analyzing-unknown-codebases.md)
- Validation protocol → [validating-architecture-analysis.md](validating-architecture-analysis.md)
