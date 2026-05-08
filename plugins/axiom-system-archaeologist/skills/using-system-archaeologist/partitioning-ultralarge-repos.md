# Partitioning Ultralarge Repos

## Purpose

When a codebase exceeds the scale at which automated discovery can produce a usable subsystem catalog in one pass, the orchestrator must **manually partition** the repo into subsystems first, then archaeologize one subsystem at a time. This sheet defines the tier criteria, the partitioning protocol, and the ordering rules.

## Tier Definitions

| Tier | Criteria (any one trips it) | Workflow |
|---|---|---|
| **Small** | <5 subsystems, <20K LOC | Single-pass, sequential, often solo |
| **Large** | 5+ subsystems, 20K–100K LOC, 10+ plugins/services | Parallel subagent dispatch in one orchestrated pass — existing `analyze-codebase` workflow |
| **Ultralarge** | **>100K LOC source, OR >12 subsystems, OR plugin-registry architecture with extensible categories, OR test corpus ≥1× source LOC, OR doc corpus >500 markdown files** | Manual partition → per-subsystem module-by-module pass with scribe → checkpointed across sessions |

**The qualifying conditions are OR, not AND.** A 60K-LOC repo with 1,500 markdown docs is ultralarge because the doc corpus alone is its own archaeology problem. A 40K-LOC repo with 4 plugin categories × N concrete plugins is ultralarge because the plugin registry is a first-class extension surface.

**If in doubt, ask one question:** *"Could I read every source file in this repo within my context budget?"* If no, it's ultralarge.

---

## Why The Existing Workflow Doesn't Fit

`analyze-codebase` assumes:

1. The orchestrator can hold a mental model of all subsystems simultaneously
2. Subagent dispatch is the parallelization axis (one agent per subsystem)
3. A single deliverable pass produces the full catalog

At ultralarge scale, all three break:

1. **Subsystems-as-units exceed orchestrator context.** Even subagent *summaries* (one paragraph each) sum to 12+ paragraphs of competing detail. The orchestrator cannot prioritize across them.
2. **Per-subsystem subagents themselves run out of context.** A "subsystem" here might be 50K LOC of plugins. One subagent cannot read it all.
3. **Single-pass deliverables produce shallow catalogs.** The catalog reflects what was sampled, not what's there. For ultralarge repos, that gap matters.

The fix is not "more subagents" — that pushes the same problem one level down. The fix is **changing the unit of analysis from subsystem to module**, with subsystems becoming the *checkpoint boundary* rather than the parallelization unit.

---

## The Partitioning Protocol

### Step 1: Operator Interview (5–10 minutes)

Before reading any code, ask the user:

1. **What are the major subsystems as YOU understand them?** Not "what does the directory structure imply" — what does the operator believe the system contains? List them.
2. **Which subsystems are stable vs. actively changing?** Stable subsystems can be deferred or sampled lightly; changing ones need depth.
3. **Are there extension surfaces (plugin registries, hook systems, codegen targets)?** These deserve their own tier of treatment.
4. **What documents are authoritative vs. aspirational vs. stale?** ARCHITECTURE.md may or may not match reality. Operator knows.
5. **What's the goal of this archaeology?** Onboarding a new team? Pre-refactor inventory? Compliance? Different goals → different partitioning.

Write answers to `00-coordination.md` under `## Operator Interview`.

**Skipping this step is the most expensive mistake.** Spending 10 minutes here saves dispatching 50 reviewer agents at the wrong granularity.

### Step 2: Cheap Structural Scan (10–15 minutes, no deep reads)

Without reading implementation files, gather:

```bash
# Top-level shape
find <root> -maxdepth 2 -type d
cloc <root>                           # or fallback: find ... | wc -l + LOC count
find <root> -name "*.md" | wc -l     # doc corpus size
find tests -name "test_*.*" | wc -l  # test corpus shape

# Plugin/extension surface detection
ls <plugins-dir>/ 2>/dev/null
grep -r "@register" --include="*.py" -l | head    # registry-style hooks
```

**Goal:** confirm or refute the operator's mental model. Discrepancies are signal — investigate them, don't paper over.

Write findings to `01-discovery-findings.md` (existing contract).

### Step 3: Manual Subsystem Boundary Decisions

Produce a partition manifest at `ultralarge/00-partition-plan.md`:

```markdown
# Partition Plan

## Subsystems
| ID | Name | Path | Est. modules | Est. LOC | Tier within ultralarge | Order |
|----|------|------|--------------|----------|------------------------|-------|
| S01 | Telemetry | src/<pkg>/telemetry/ | ~22 | ~6,000 | small-within-ultralarge | 1 |
| S02 | Engine | src/<pkg>/engine/ | ~85 | ~24,000 | large-within-ultralarge | 4 |
| ... | ... | ... | ... | ... | ... | ... |

## Extension Surfaces (treated as own subsystems)
| ID | Surface | Categories | Concrete plugins |
|----|---------|------------|------------------|
| X01 | Plugin registry | sources, sinks, transforms, infrastructure | ~40 across categories |

## Test Corpus (treated as own subsystem if ≥1× source LOC)
| ID | Path | Files | LOC | Decision |
|----|------|-------|-----|----------|
| T01 | tests/ | 907 | 381K | Separate test-archaeology pass after source pass |

## Doc Corpus (triage, not archaeology)
| Status | Count | Examples | Action |
|--------|-------|----------|--------|
| Authoritative | ~6 | ARCHITECTURE.md, CONTRIBUTING.md | Read in full |
| Aspirational | ~30 | docs/proposals/* | Skim, mark as design intent |
| Stale | ~200 | docs/notes/2023-* | Note existence, do not read |
| Auto-generated | ~1400 | docs/api/* | Reference only |

## Order Justification
1. S01 (Telemetry) — small, leaf-position, calibrates protocol
2. S03 (Contracts) — defines types used elsewhere
3. ...
```

**The order field is critical.** Default ordering rule:

1. **Leaf subsystems first** (low outbound deps): they don't reference other subsystems' details, so module reviews can proceed without forward references.
2. **Type / contract subsystems early**: subsystems that define data types used elsewhere should be canonicalized before consumers, so consumer reviews can cite types accurately.
3. **Plugin registry before plugins**: catalog the registry / base classes before the concrete plugins.
4. **Test corpus last** (if treated separately): tests reference everything; review them once source canonicals exist.
5. **High-churn subsystems last within their layer**: their canonicals will go stale fastest.

### Step 4: Per-Subsystem Module Inventory

For each subsystem, BEFORE dispatching any reviewers, generate a module list:

```bash
find <subsystem-path> -name "*.<ext>" -not -path "*__pycache__*" -not -path "*.egg-info*"
```

Write to `ultralarge/subsystem-S<NN>/00-partition.md`:

```markdown
# Subsystem S01: Telemetry — Partition

## Modules in scope
| Module ID | Path | LOC | Order |
|-----------|------|-----|-------|
| S01-M001 | src/<pkg>/telemetry/__init__.py | 12 | 1 (leaf) |
| S01-M002 | src/<pkg>/telemetry/types.py | 84 | 2 (types) |
| S01-M003 | src/<pkg>/telemetry/collector.py | 312 | 3 |
| ... | ... | ... | ... |

## Cheap dependency scan (intra-subsystem)
S01-M003 imports from S01-M002 (types), S01-M001 (__init__)
S01-M005 imports from S01-M003, S01-M002
...

## Dependency leaves (review first)
S01-M001, S01-M002

## Order rationale
Leaves → types → collectors → exporters → public API
```

The intra-subsystem dependency scan is `grep -E "^(from|import)"` per file — cheap, shallow, sufficient for ordering. **Do not spawn reviewers to do this.** It's a script.

### Step 5: Checkpoint Discipline

After each subsystem's modules are all canonicalized and the subsystem catalog entry is synthesized, **stop and commit**:

```
docs/arch-analysis-YYYY-MM-DD-HHMM/
├── 00-coordination.md          # Updated with checkpoint
├── ultralarge/
│   ├── 00-partition-plan.md
│   ├── subsystem-S01/
│   │   ├── 00-partition.md
│   │   ├── modules/*.canonical.yaml      # Complete
│   │   └── 99-subsystem-catalog-entry.md # Synthesized
│   └── subsystem-S02/  ← next session resumes here
└── 02-subsystem-catalog.md     # Updated with S01 entry
```

Each subsystem is a **resumable checkpoint**. A new session reads `00-partition-plan.md`, identifies the next un-completed subsystem from the order column, and proceeds. No prior-session context required.

---

## Anti-Patterns

**❌ Skip the operator interview because "the directory structure is clear".**
Directory structure is a code-author's mental model frozen at one point in time. The operator's *current* mental model is more useful for partitioning.

**❌ Treat plugins as a single subsystem.**
A plugin registry with N concrete plugins is N+1 things: the registry (one subsystem) and the plugins (one subsystem per category, OR sampled if categories have many concrete plugins).

**❌ Try to canonicalize all subsystems in one session.**
Ultralarge means it doesn't fit. Plan for multi-session work from the start.

**❌ Pick subsystem order arbitrarily.**
Wrong order → forward references → consumer reviews cite types that aren't yet canonical → rework.

**❌ Dispatch a subagent to "decide the partitioning".**
Partitioning is an orchestrator + operator decision. A subagent does not have the operator interview context.

**❌ Use the existing `analyze-codebase` workflow on an ultralarge repo.**
It will appear to succeed and produce a thin, sample-driven catalog that misses entire subsystems. Detect ultralarge tier first; switch tracks deliberately.

---

## Detection Checklist

Before starting `analyze-codebase`, run:

```bash
# 1. Source LOC
find <src> -name "*.py" -o -name "*.ts" -o -name "*.go" -o -name "*.rs" \
  | xargs wc -l 2>/dev/null | tail -1

# 2. Subsystem candidate count
find <src> -maxdepth 2 -type d | wc -l

# 3. Test-to-source ratio
find tests -name "*.<ext>" | xargs wc -l | tail -1

# 4. Doc corpus
find . -name "*.md" -not -path "./node_modules/*" | wc -l

# 5. Plugin/extension surface
find . -path "*/plugins/*" -maxdepth 4 -type d | head
```

If any of:
- Source LOC > 100,000
- Subsystem candidates > 12
- Test LOC ≥ source LOC
- Doc corpus > 500
- Plugin-registry architecture present

→ **Switch to `/analyze-ultralarge` track.** Do not proceed with `/analyze-codebase`.

---

## Cross-References

- Schema for module findings → [findings-schema.md](findings-schema.md)
- Per-module workflow → [module-by-module-with-scribe.md](module-by-module-with-scribe.md)
- Existing single-pass workflow → [analyzing-unknown-codebases.md](analyzing-unknown-codebases.md)
- Subsystem catalog contract (synthesis target) → [analyzing-unknown-codebases.md](analyzing-unknown-codebases.md)
